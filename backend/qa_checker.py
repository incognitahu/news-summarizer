# backend/qa_checker.py
"""
Robust QA-based factuality checker with spaCy NER (safe usage).

Behavior:
- Try to use spaCy en_core_web_sm (with NER).
- If spaCy full model fails to load or NER cannot run, fall back to a blank pipeline
  (sentencizer only) + a conservative capitalization heuristic for entities.
- Use Hugging Face QA pipeline to check whether entities found in summary are supported by source.
- Be defensive: catch exceptions per-entity and per-sentence; always return a structured report.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------- spaCy setup (defensive) --------
_Has_SPACY = False
_nlp = None
try:
    import spacy  # type: ignore
    try:
        # Try to load the full model (with NER)
        _nlp = spacy.load("en_core_web_sm")
        _Has_SPACY = True
        logger.info("spaCy full model loaded (en_core_web_sm). NER enabled.")
    except Exception as e_full:
        # Fall back to a blank pipeline with only sentencizer to avoid crashes
        logger.warning(f"spaCy full model not available or failed: {e_full}. Falling back to blank pipeline.")
        try:
            _nlp = spacy.blank("en")
            # add sentencizer so we can split into sentences safely
            if "sentencizer" not in _nlp.pipe_names:
                _nlp.add_pipe("sentencizer")
            _Has_SPACY = True
        except Exception as e_blank:
            logger.warning(f"Failed to create spaCy blank pipeline: {e_blank}. spaCy disabled.")
            _nlp = None
            _Has_SPACY = False
except Exception as e_import:
    logger.info(f"spaCy import failed or not installed: {e_import}. spaCy disabled.")
    _nlp = None
    _Has_SPACY = False

# -------- QA pipeline (Hugging Face) --------
_QA_MODEL = "distilbert-base-uncased-distilled-squad"
_qa_pipe = None
try:
    from transformers import pipeline  # type: ignore
    try:
        _qa_pipe = pipeline("question-answering", model=_QA_MODEL, tokenizer=_QA_MODEL)
        logger.info(f"QA pipeline loaded: {_QA_MODEL}")
    except Exception as e_qa:
        logger.warning(f"Failed to initialize QA pipeline ({_QA_MODEL}): {e_qa}")
        _qa_pipe = None
except Exception as e_trf:
    logger.warning(f"transformers import failed, QA disabled: {e_trf}")
    _qa_pipe = None

# -------- Helpers --------
def _sentence_split(text: str) -> List[str]:
    """Return list of sentences using spaCy if available, else naive split."""
    if not text:
        return []
    if _Has_SPACY and _nlp is not None:
        try:
            doc = _nlp(text)
            # doc.sents is available even for blank pipeline with sentencizer
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception as e:
            logger.warning(f"spaCy sentence split failed: {e}. Falling back to regex split.")
    # fallback
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract entities from `text`.
    Prefer spaCy NER when available; if spaCy NER isn't usable, fall back to a capitalization heuristic.
    Returns list of dicts: {"text": str, "label": Optional[str]}
    """
    if not text:
        return []
    # Try spaCy NER if the model actually has NER component
    if _Has_SPACY and _nlp is not None:
        try:
            # This may raise E109 if a component isn't initialized in some broken setups; catch it
            doc = _nlp(text)
            ents = []
            # If NER exists in the pipeline, doc.ents will contain entities; otherwise it will be empty
            for ent in getattr(doc, "ents", []):
                t = ent.text.strip()
                if t:
                    ents.append({"text": t, "label": ent.label_ if hasattr(ent, "label_") else None})
            # dedupe preserving order
            seen = set()
            uniq = []
            for e in ents:
                if e["text"] not in seen:
                    seen.add(e["text"])
                    uniq.append(e)
            if uniq:
                return uniq
            # if spaCy returned no ents, continue to heuristic below
        except Exception as e:
            logger.warning(f"spaCy NER failed (falling back): {e}")

    # Fallback heuristic: gather contiguous capitalized tokens as entity spans
    tokens = [t.strip() for t in text.replace("\n", " ").split() if t.strip()]
    candidates = []
    cur = []
    for tok in tokens:
        # Skip numeric tokens; check first char uppercase as naive proper noun heuristic
        if tok and tok[0].isupper() and not tok.isdigit():
            cur.append(tok)
        else:
            if cur:
                cand = " ".join(cur)
                if len(cand) > 1:
                    candidates.append({"text": cand, "label": "UNK"})
                cur = []
    if cur:
        cand = " ".join(cur)
        if len(cand) > 1:
            candidates.append({"text": cand, "label": "UNK"})
    # dedupe
    seen = set()
    uniq = []
    for e in candidates:
        if e["text"] not in seen:
            seen.add(e["text"])
            uniq.append(e)
    return uniq

def _in_context(entity: str, context: str) -> bool:
    """Case-insensitive substring check for entity existence in context."""
    if not entity or not context:
        return False
    return entity.lower() in context.lower()

def _qa_check(entity_text: str, context: str, min_score: float = 0.25) -> Dict[str, Any]:
    """
    Run QA pipeline to see if `entity_text` is supported by `context`.
    Returns dict: {"supported": bool, "answer": str, "score": float, "error": Optional[str]}
    """
    if not entity_text or not context:
        return {"supported": False, "answer": "", "score": 0.0, "error": "no_input"}

    if _qa_pipe is None:
        return {"supported": False, "answer": "", "score": 0.0, "error": "qa_pipeline_unavailable"}

    # trim context to manageable size (characters), prefer keeping beginning which often contains key facts
    max_ctx_chars = 3000
    context_snippet = context if len(context) <= max_ctx_chars else context[:max_ctx_chars]

    try:
        resp = _qa_pipe(question=entity_text, context=context_snippet, top_k=1)
        # HF returns dict or list depending on top_k
        best = resp[0] if isinstance(resp, list) and resp else resp
        if not isinstance(best, dict):
            return {"supported": False, "answer": "", "score": 0.0, "error": "unexpected_qa_response"}
        answer = best.get("answer", "").strip()
        score = float(best.get("score", 0.0))
        supported = bool(answer) and score >= min_score
        return {"supported": supported, "answer": answer, "score": score}
    except Exception as e:
        logger.warning(f"QA pipeline failed for entity '{entity_text}': {e}")
        return {"supported": False, "answer": "", "score": 0.0, "error": str(e)}

# -------- Main check function --------
def check_summary_against_source(summary: str, source: str, max_entities_per_sentence: int = 5) -> Dict[str, Any]:
    """
    Check each sentence in `summary` against `source`.
    Returns a dict:
    {
      "sentence_checks": [ { "sentence": str, "entities": [ {text,label,in_source,qa_supported,qa_answer,qa_score,qa_error,flagged} ], "flagged": bool }, ... ],
      "overall_flagged_entities": int,
      "errors": [ ... ]
    }
    """
    results: Dict[str, Any] = {"sentence_checks": [], "overall_flagged_entities": 0, "errors": []}
    try:
        if not summary or not source:
            return results

        sentences = _sentence_split(summary)

        for sent in sentences:
            sent_record: Dict[str, Any] = {"sentence": sent, "entities": [], "flagged": False}
            try:
                entities = extract_entities(sent)
            except Exception as e_ent:
                msg = f"entity_extraction_failed: {e_ent}"
                logger.warning(msg)
                results["errors"].append(msg)
                entities = []

            # check up to max_entities_per_sentence entities
            for ent in entities[:max_entities_per_sentence]:
                ent_text = ent.get("text", "").strip()
                ent_label = ent.get("label")
                try:
                    in_src = _in_context(ent_text, source)
                    qa_res = _qa_check(ent_text, source)
                    flagged = (not in_src) and (not qa_res.get("supported", False))
                    ent_entry = {
                        "text": ent_text,
                        "label": ent_label,
                        "in_source": in_src,
                        "qa_supported": qa_res.get("supported", False),
                        "qa_answer": qa_res.get("answer", ""),
                        "qa_score": qa_res.get("score", 0.0),
                        "qa_error": qa_res.get("error") if "error" in qa_res else None,
                        "flagged": flagged,
                    }
                    if flagged:
                        sent_record["flagged"] = True
                        results["overall_flagged_entities"] += 1
                    sent_record["entities"].append(ent_entry)
                except Exception as e_check:
                    err_msg = f"entity_check_failed for '{ent_text}': {e_check}"
                    logger.warning(err_msg)
                    results["errors"].append(err_msg)
                    # append minimal failed record to keep structure consistent
                    sent_record["entities"].append({
                        "text": ent_text,
                        "label": ent_label,
                        "in_source": False,
                        "qa_supported": False,
                        "qa_answer": "",
                        "qa_score": 0.0,
                        "qa_error": str(e_check),
                        "flagged": True,
                    })

            results["sentence_checks"].append(sent_record)
    except Exception as e_top:
        msg = f"qa_checker_top_level_exception: {e_top}"
        logger.exception(msg)
        results["errors"].append(msg)

    return results
