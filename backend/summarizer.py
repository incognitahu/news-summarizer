# backend/summarizer.py
"""
News summarizer module.

Features:
- Loads a seq2seq summarization model from Hugging Face (AutoModelForSeq2SeqLM + AutoTokenizer).
- Handles GPU if available; falls back to CPU.
- Long-input strategy: sentence-aware chunking -> per-chunk summarization -> fusion summary.
- Simple extractive prefilter: 'lead' (first-K sentences) or 'none'. (Hook for more advanced extractors.)
- Tunable generation parameters (beam size, length, no_repeat_ngram_size, etc).
- Defensive: uses spaCy (preferred) or NLTK for sentence segmentation; falls back to a simple splitter.
"""

from typing import List, Optional, Tuple
import logging
import math

# Transformers / torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
)
import torch

# at top of summarizer.py
from .reranker import SBERTReranker
from .qa_checker import check_summary_against_source

# Optional NLP libs for sentence splitting
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    import nltk
    _HAS_NLTK = True
    # Ensure punkt is available — silent if already present
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
except Exception:
    _HAS_NLTK = False

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NewsSummarizer:
    # Supported models with optimized configurations
    MODEL_CONFIGS = {
        "facebook/bart-large-cnn": {
            "max_length": 142,
            "min_length": 56,
            "length_penalty": 2.0,
        },
        "sshleifer/distilbart-cnn-12-6": {
            "max_length": 142,
            "min_length": 56,
            "length_penalty": 2.0,
        },
        "google/flan-t5-base": {
            "max_length": 150,
            "min_length": 50,
            "length_penalty": 1.0,
        },
        "google/pegasus-cnn_dailymail": {
            "max_length": 128,
            "min_length": 64,
            "length_penalty": 0.8,
        }
    }
    
    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        device: Optional[int] = None,
        max_input_tokens: int = 1024,
        chunk_overlap_sentences: int = 2,
    ):
        """
        Args:
            model_name: HF model id for seq2seq summarization (BART/T5/PEGASUS/LongT5, etc).
            device: None -> auto (GPU if available), int -> cuda device id, -1 -> CPU.
            max_input_tokens: maximum input tokens model can accept (approx). If input longer, chunking is used.
            chunk_overlap_sentences: number of sentences overlap between adjacent chunks to preserve context.
        """
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.chunk_overlap_sentences = chunk_overlap_sentences
        # inside __init__
        self.reranker = SBERTReranker(model_name="all-MiniLM-L6-v2", device="cpu")


        # device selection
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        # load tokenizer and model
        logger.info(f"Loading tokenizer & model: {model_name} (device={self.device})")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if self.device != -1 and torch.cuda.is_available():
            try:
                self.model = self.model.to(torch.device(f"cuda:{self.device}"))
            except Exception as e:
                logger.warning(f"Could not move model to cuda:{self.device} — running on CPU. Error: {e}")
        self.model.eval()

        # spaCy pipeline for sentence segmentation (preferred)
        if _HAS_SPACY:
            try:
                # use small model if available; otherwise only the sentencizer
                try:
                    self._nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
                except OSError:
                    # fallback: create blank English pipeline with sentencizer
                    self._nlp = spacy.blank("en")
                    self._nlp.add_pipe("sentencizer")
            except Exception as e:
                logger.warning(f"spaCy load failed, falling back to NLTK. Error: {e}")
                self._nlp = None
        else:
            self._nlp = None

    # -------------------------
    # Public methods
    # -------------------------
    def summarize(
        self,
        text: str,
        min_length: int = None,
        max_length: int = None,
        num_beams: int = 4,
        length_penalty: float = None,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        extractive_prefilter: str = "none",
    ) -> str:
        """
        Main entrypoint. Summarize `text` and return a string summary.

        Args:
            text: raw article text or cleaned HTML.
            min_length/max_length: generation length bounds (tokens). Uses model defaults if None.
            num_beams: beam size for beam search generation.
            length_penalty: length penalty for generation. Uses model defaults if None.
            extractive_prefilter: 'none' | 'lead' (first-K sentences). Hook to add more extractors later.

        Returns:
            summary string.
        """
        if not text or not text.strip():
            return ""
        
        # Get model-specific config defaults if not provided
        model_config = self.MODEL_CONFIGS.get(self.model_name, {})
        if min_length is None:
            min_length = model_config.get("min_length", 40)
        if max_length is None:
            max_length = model_config.get("max_length", 160)
        if length_penalty is None:
            length_penalty = model_config.get("length_penalty", 1.0)

        sentences = self._split_into_sentences(text)
        if extractive_prefilter == "lead":
            # simple extractive prefilter — keep first N sentences
            N = 8  # reasonable default; can be parameterized
            sentences = sentences[:N]

        # Reconstruct possibly shorter text after prefilter
        joined_text = " ".join(sentences).strip()
        if not joined_text:
            return ""

        # Measure token length
        token_count = self._count_tokens(joined_text)
        logger.info(f"Input tokens (approx): {token_count}; max_input_tokens={self.max_input_tokens}")

        if token_count <= self.max_input_tokens:
            # single-pass summarization
            return self._generate_summary_from_text(
                joined_text,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )
        else:
            # long-input strategy: chunk -> per-chunk summarization -> fuse
            chunk_texts = self._chunk_sentences_into_texts(sentences, self.max_input_tokens)
            logger.info(f"Text split into {len(chunk_texts)} chunks for summarization.")
            chunk_summaries = []
            for i, ctext in enumerate(chunk_texts):
                logger.info(f"Summarizing chunk {i+1}/{len(chunk_texts)} (approx {self._count_tokens(ctext)} tokens).")
                s = self._generate_summary_from_text(
                    ctext,
                    min_length=max(20, min_length // 2),  # shorter per-chunk min
                    max_length=max(50, max_length // 2),  # shorter per-chunk max
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                )
                chunk_summaries.append(s)

            # # Fuse chunk summaries into final summary (one more summarization pass)
            # fused_input = " ".join(chunk_summaries)
            # logger.info(f"Fusing {len(chunk_summaries)} chunk summaries (tokens: {self._count_tokens(fused_input)}).")
            # final_summary = self._generate_summary_from_text(
            #     fused_input,
            #     min_length=min_length,
            #     max_length=max_length,
            #     num_beams=num_beams,
            #     length_penalty=length_penalty,
            #     no_repeat_ngram_size=no_repeat_ngram_size,
            #     early_stopping=early_stopping,
            # )
            # return final_summary
            # ---- SBERT RERANKER UPGRADE ----
            K = 5  # number of chunk summaries to keep for fusion, tuneable

            try:
                reference_for_ranking = joined_text  # full original article
                candidates = chunk_summaries
                topk = self.reranker.select_top_k(reference_for_ranking, candidates, k=min(K, len(candidates)))
                selected_texts = [t[2] for t in topk]  # pick the selected chunk summaries
                fused_input = " ".join(selected_texts)
                logger.info(f"Reranker selected {len(selected_texts)} chunks for fusion.")
            except Exception as e:
                logger.warning(f"Reranker failed, falling back to naive fusion: {e}")
                fused_input = " ".join(chunk_summaries)

            # ---- FINAL PASS SUMMARY ----
            final_summary = self._generate_summary_from_text(
                fused_input,
                min_length=min_length,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )

            # ---- QA CHECK UPON FINAL SUMMARY ----
            try:
                qa_report = check_summary_against_source(final_summary, joined_text)
            except Exception as e:
                logger.warning(f"QA checker failed: {e}")
                qa_report = {"error": str(e)}

            # Return both summary + QA report so API layer can handle output
            return {"summary": final_summary, "qa_report": qa_report}


    # -------------------------
    # Internal helpers
    # -------------------------
    def _generate_summary_from_text(
        self,
        text: str,
        min_length: int,
        max_length: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
        early_stopping: bool,
    ) -> str:
        """
        Generates summary from a single (reasonably sized) text chunk using the underlying model.
        Uses model.generate for more control than pipeline.
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # Move tensors to device
        if self.device != -1 and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.device}")
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            self.model.to(device)
        else:
            device = torch.device("cpu")
            self.model.to(device)

        # generation kwargs
        gen_kwargs = dict(
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            use_cache=True,
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded[0].strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Return a list of sentence strings using spaCy (if available) or NLTK or a fallback splitter."""
        text = text.strip()
        if not text:
            return []

        # spaCy pipeline with sentencizer
        if self._nlp is not None:
            try:
                doc = self._nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"spaCy sentence split failed: {e}")

        # NLTK
        if _HAS_NLTK:
            try:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
                return [s.strip() for s in sentences if s.strip()]
            except Exception as e:
                logger.warning(f"NLTK sentence split failed: {e}")

        # Fallback: naive split by punctuation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_tokens(self, text: str) -> int:
        """Approximate token count using tokenizer (fast)."""
        try:
            tok = self.tokenizer.encode(text, truncation=False)
            return len(tok)
        except Exception:
            # fallback: rough estimate by words
            return max(0, len(text.split()))

    def _chunk_sentences_into_texts(self, sentences: List[str], max_tokens: int) -> List[str]:
        """
        Group sentences into chunks where each chunk has <= max_tokens (approx, by tokenizer).
        Uses sentence overlap to avoid cutting context.
        """
        if not sentences:
            return []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for i, sent in enumerate(sentences):
            s_tokens = self._count_tokens(sent)
            # if adding this sentence would exceed the budget, finalize current chunk
            if current_tokens + s_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                # start new chunk with overlap
                overlap = self.chunk_overlap_sentences
                if overlap <= 0:
                    current_chunk = [sent]
                    current_tokens = s_tokens
                else:
                    # keep last `overlap` sentences as start of new chunk
                    overlap_sents = current_chunk[-overlap:] if len(current_chunk) >= overlap else current_chunk[:]
                    current_chunk = overlap_sents + [sent]
                    current_tokens = sum(self._count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sent)
                current_tokens += s_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # safety: if any chunk is still too large (single sentence > max_tokens) then force split by words
        safe_chunks: List[str] = []
        for c in chunks:
            if self._count_tokens(c) <= max_tokens:
                safe_chunks.append(c)
            else:
                # split by approximate word counts
                words = c.split()
                approx_tokens_per_word = 1  # conservative
                max_words = max(100, math.floor(max_tokens / (approx_tokens_per_word or 1)))
                for i in range(0, len(words), max_words):
                    safe_chunks.append(" ".join(words[i:i + max_words]))
        return safe_chunks


# -------------------------
# Example usage (for dev/testing)
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_text = (
        "New research indicates that large language models are improving at summarization. "
        "The research explores a number of model families, including BART, T5, and LongT5, "
        "assessing their performance on standard news datasets and long-form inputs. "
        "Early results show that chunking inputs and fusing chunk summaries helps preserve "
        "important details while controlling hallucination."
    ) * 10  # make it longer to test chunking
    summarizer = NewsSummarizer(model_name="sshleifer/distilbart-cnn-12-6", max_input_tokens=512)
    summary = summarizer.summarize(demo_text, min_length=30, max_length=120)
    print("\n--- SUMMARY ---\n", summary)
