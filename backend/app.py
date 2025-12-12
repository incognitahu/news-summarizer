# backend/app.py
"""
FastAPI backend for News Summarizer.

Endpoints:
  POST /summarize
    - body: { text, input_is_html, min_length, max_length, num_beams, use_reranker, top_k }
    - returns: { summary, qa_report, debug: {token_count, num_chunks, reranker_scores} }

  GET /health
    - returns: { status: "ok" }

Run:
  uvicorn backend.app:app --reload --port 8000
"""

from typing import Optional, List, Dict, Any
import logging
import time


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from .preprocess import preprocess_for_model
from .summarizer import NewsSummarizer
from .reranker import SBERTReranker
from .qa_checker import check_summary_against_source

logger = logging.getLogger("news_summarizer_app")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="News Summarizer API", version="0.1")

# Allow local browser UIs to call (adjust origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# Request / Response models
# ---------------------
class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Raw article text or HTML")
    input_is_html: bool = Field(True, description="If true, run HTML extraction")
    model_name: str = Field("sshleifer/distilbart-cnn-12-6", description="Model to use for summarization")
    min_length: int = Field(None, description="Minimum tokens in summary (None = use model defaults)")
    max_length: int = Field(None, description="Maximum tokens in summary (None = use model defaults)")
    num_beams: int = Field(4, description="Beam search width")
    length_penalty: float = Field(None, description="Length penalty during generation (None = use model defaults)")
    no_repeat_ngram_size: int = Field(3, description="No repeat ngram size")
    use_reranker: bool = Field(True, description="Use SBERT reranker when input is long")
    top_k: int = Field(5, description="Top-k chunk summaries to keep for fusion when reranking")
    extractive_prefilter: str = Field("none", description="none | lead (simple first-N sentences)")
    run_qa: bool = Field(True, description="Whether to run QA factuality checks (set false to skip QA)")
    extractive_method: str = Field(None, description="Use extractive summarization: tfidf | textrank | lead")


class EntityCheck(BaseModel):
    text: str
    label: Optional[str]
    in_source: bool
    qa_supported: bool
    qa_answer: Optional[str]
    qa_score: Optional[float]
    flagged: bool

class SentenceCheck(BaseModel):
    sentence: str
    entities: List[EntityCheck]
    flagged: bool

class QAReport(BaseModel):
    sentence_checks: List[SentenceCheck]
    overall_flagged_entities: int

class SummarizeResponse(BaseModel):
    summary: str
    qa_report: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None

# ---------------------
# Global singletons (loaded at startup)
# ---------------------
# You can change model names/devices here if needed.
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # dev-friendly; swap later to bart-large-cnn
SBERT_MODEL = "all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-uncased-distilled-squad"  # used inside qa_checker

# Instances (populated in startup event)
summarizer: Optional[NewsSummarizer] = None
reranker: Optional[SBERTReranker] = None

@app.on_event("startup")
def startup_event():
    global summarizer, reranker
    logger.info("Starting News Summarizer API - loading models (this may take a while)...")
    t0 = time.time()
    try:
        # Default device selection handled inside NewsSummarizer
        summarizer = NewsSummarizer(model_name=SUMMARIZER_MODEL, max_input_tokens=1024, chunk_overlap_sentences=2)
        reranker = SBERTReranker(model_name=SBERT_MODEL, device="cpu")
    except Exception as e:
        logger.exception(f"Model loading failed at startup: {e}")
        raise
    logger.info(f"Models loaded in {time.time() - t0:.1f}s")


# ---------------------
# Helper: orchestration pipeline
# ---------------------
def summarize_pipeline(
    raw_input: str,
    input_is_html: bool = True,
    min_length: int = 40,
    max_length: int = 160,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    extractive_prefilter: str = "none",
    use_reranker: bool = True,
    top_k: int = 5,
    run_qa: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate preprocessing -> summarization (single or chunked) -> optional reranker fusion -> optional QA checks.
    Returns dict with keys: summary (str), qa_report (dict or None), debug (dict).
    """
    global summarizer, reranker
    if summarizer is None:
        raise RuntimeError("Summarizer model is not initialized.")

    # 1) Preprocess input (HTML extraction, cleaning)
    cleaned_text, sentences = preprocess_for_model(raw_input, input_is_html=input_is_html, return_sentences=True)
    if not cleaned_text or len(cleaned_text.strip()) < 5:
        raise ValueError("Input too short after preprocessing.")

    # 2) Token estimate & choose path
    token_count = summarizer._count_tokens(cleaned_text)
    debug = {"token_count": token_count, "num_chunks": 0, "reranker_scores": None}
    logger.info(f"Preprocessed text tokens ~{token_count}")

    # quick single-pass path
    if token_count <= summarizer.max_input_tokens:
        logger.info("Single-pass summarization (input fits).")
        summary = summarizer._generate_summary_from_text(
            cleaned_text,
            min_length=min_length,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )
    else:
        # long-input path: chunk -> per-chunk summarize -> rerank -> fuse -> summarize
        logger.info("Long input detected: chunking & chunk-level summarization.")
        sentences_list = sentences if sentences else summarizer._split_into_sentences(cleaned_text)
        chunk_texts = summarizer._chunk_sentences_into_texts(sentences_list, summarizer.max_input_tokens)
        debug["num_chunks"] = len(chunk_texts)

        # per-chunk summarization
        chunk_summaries = []
        for i, ctext in enumerate(chunk_texts):
            try:
                cs = summarizer._generate_summary_from_text(
                    ctext,
                    min_length=max(20, min_length // 2),
                    max_length=max(50, max_length // 2),
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=True,
                )
            except Exception as e:
                logger.warning(f"Chunk summarization failed for chunk {i}: {e}")
                cs = ""  # skip
            if cs:
                chunk_summaries.append(cs)

        if not chunk_summaries:
            # fallback: summarize the first max_input_tokens worth of text
            logger.warning("All chunk summarizations failed; falling back to truncation summarization.")
            truncated_input = cleaned_text[: int(summarizer.max_input_tokens * 0.8)]
            summary = summarizer._generate_summary_from_text(
                truncated_input,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
        else:
            # Use reranker (SBERT) to select top-k chunk summaries (or fall back to naive fusion)
            if use_reranker and reranker is not None:
                try:
                    topk = reranker.select_top_k(cleaned_text, chunk_summaries, k=min(top_k, len(chunk_summaries)))
                    selected_texts = [t[2] for t in topk]
                    debug["reranker_scores"] = [{"idx": t[0], "score": t[1]} for t in topk]
                except Exception as e:
                    logger.warning(f"Reranker failed: {e}. Falling back to naive fusion.")
                    selected_texts = chunk_summaries
            else:
                selected_texts = chunk_summaries

            fused_input = " ".join(selected_texts)
            # final fusion summarization
            summary = summarizer._generate_summary_from_text(
                fused_input,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )

    # 3) QA checks (factuality) — only run if requested
    qa_report = None
    if run_qa:
        try:
            qa_report = check_summary_against_source(summary, cleaned_text)
        except Exception as e:
            logger.warning(f"QA checker failed: {e}")
            qa_report = {"error": str(e)}
    else:
        qa_report = {"skipped": True}

    return {"summary": summary, "qa_report": qa_report, "debug": debug}



# ---------------------
# API endpoints
# ---------------------
@app.post("/summarize", response_model=SummarizeResponse)
def api_summarize(req: SummarizeRequest):
    """
    Main summarization endpoint.
    """
    try:
        start = time.time()
        
        # Handle extractive methods
        if req.extractive_method:
            from .extractive import extractive_tfidf, extractive_textrank, extractive_lead
            cleaned_text, _ = preprocess_for_model(req.text, input_is_html=req.input_is_html)
            if req.extractive_method == "tfidf":
                summary = extractive_tfidf(cleaned_text, num_sentences=3)
            elif req.extractive_method == "textrank":
                summary = extractive_textrank(cleaned_text, num_sentences=3)
            elif req.extractive_method == "lead":
                summary = extractive_lead(cleaned_text, num_sentences=3)
            else:
                raise ValueError(f"Unknown extractive method: {req.extractive_method}")
            
            result = {"summary": summary, "qa_report": None, "debug": {"method": "extractive"}}
        else:
            # Load or reload summarizer if model changed
            global summarizer
            if summarizer is None or summarizer.model_name != req.model_name:
                logger.info(f"Loading/switching to model: {req.model_name}")
                summarizer = NewsSummarizer(model_name=req.model_name)
            
            result = summarize_pipeline(
                raw_input=req.text,
                input_is_html=req.input_is_html,
                min_length=req.min_length,
                max_length=req.max_length,
                num_beams=req.num_beams,
                length_penalty=req.length_penalty,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                extractive_prefilter=req.extractive_prefilter,
                use_reranker=req.use_reranker,
                top_k=req.top_k,
                run_qa=req.run_qa,
            )

        elapsed = time.time() - start
        logger.info(f"Summarization completed in {elapsed:.2f}s (tokens~{result['debug'].get('token_count')}).")
        return SummarizeResponse(summary=result["summary"], qa_report=result["qa_report"], debug=result["debug"])
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail="Internal summarization error")


@app.get("/health")
def health():
    return {"status": "ok"}


# If run as main, start uvicorn (useful when running the backend folder directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
