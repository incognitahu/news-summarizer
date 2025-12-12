# backend/metrics.py
"""
Evaluation metrics for summarization quality.

Implements:
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU
- BERTScore
"""

from typing import Dict, List, Union
import warnings

# ROUGE
try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

# BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    _HAS_BLEU = True
except ImportError:
    _HAS_BLEU = False

# BERTScore
try:
    import bert_score
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False


def calculate_rouge(generated: str, reference: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        generated: Generated summary
        reference: Reference summary
    
    Returns:
        Dictionary with rouge1, rouge2, rougeL scores (F1)
    """
    if not _HAS_ROUGE:
        warnings.warn("rouge-score not installed")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    except Exception as e:
        warnings.warn(f"ROUGE calculation failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculate_bleu(generated: str, reference: str) -> float:
    """
    Calculate BLEU score.
    
    Args:
        generated: Generated summary
        reference: Reference summary
    
    Returns:
        BLEU score (0-1)
    """
    if not _HAS_BLEU:
        warnings.warn("nltk not installed or punkt missing")
        return 0.0
    
    try:
        from nltk.tokenize import word_tokenize
        
        reference_tokens = [word_tokenize(reference.lower())]
        generated_tokens = word_tokenize(generated.lower())
        
        # Use smoothing to avoid zero scores
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing)
        
        return score
    except Exception as e:
        warnings.warn(f"BLEU calculation failed: {e}")
        return 0.0


def calculate_bertscore(generated: Union[str, List[str]], 
                       reference: Union[str, List[str]], 
                       lang: str = "en",
                       model_type: str = "microsoft/deberta-xlarge-mnli") -> Dict[str, float]:
    """
    Calculate BERTScore (semantic similarity).
    
    Args:
        generated: Generated summary or list of summaries
        reference: Reference summary or list of summaries
        lang: Language code (default: "en")
        model_type: BERT model to use
    
    Returns:
        Dictionary with precision, recall, f1 scores
    """
    if not _HAS_BERTSCORE:
        warnings.warn("bert-score not installed")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    try:
        # Convert to lists if strings
        if isinstance(generated, str):
            generated = [generated]
        if isinstance(reference, str):
            reference = [reference]
        
        # Calculate BERTScore
        P, R, F1 = bert_score.score(
            generated, 
            reference, 
            lang=lang,
            model_type=model_type,
            verbose=False
        )
        
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    except Exception as e:
        warnings.warn(f"BERTScore calculation failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def evaluate_summary(generated: str, 
                    reference: str,
                    include_bertscore: bool = False) -> Dict[str, float]:
    """
    Comprehensive evaluation of a generated summary.
    
    Args:
        generated: Generated summary
        reference: Reference (gold) summary
        include_bertscore: Whether to include BERTScore (slower)
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # ROUGE scores
    rouge_scores = calculate_rouge(generated, reference)
    results.update(rouge_scores)
    
    # BLEU score
    results["bleu"] = calculate_bleu(generated, reference)
    
    # BERTScore (optional, slower)
    if include_bertscore:
        bert_scores = calculate_bertscore(generated, reference)
        results["bertscore_precision"] = bert_scores["precision"]
        results["bertscore_recall"] = bert_scores["recall"]
        results["bertscore_f1"] = bert_scores["f1"]
    
    return results


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as readable string."""
    lines = []
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.4f}")
    return "\n".join(lines)
