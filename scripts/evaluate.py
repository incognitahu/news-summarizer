# scripts/evaluate.py
"""
Batch evaluation script for news summarization models.

Evaluates multiple models and methods on the BBC News Summary dataset:
- Abstractive: BART, T5, PEGASUS
- Extractive: TF-IDF, TextRank, Lead

Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU

Modes:
- "local" : import and call summarizer functions/classes directly (fast, requires Python imports)
- "api"   : call the running FastAPI endpoint at http://127.0.0.1:8000/summarize

Outputs:
- evaluation_results.json (per-model metrics + comparison table)
- prints a comparison summary to stdout
"""

import os
import json
import glob
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.summarizer import NewsSummarizer
from backend.extractive import extractive_tfidf, extractive_textrank, extractive_lead
from backend.metrics import evaluate_summary
from backend.preprocess import preprocess_for_model

# choose mode: "api" or "local"
MODE = os.environ.get("EVAL_MODE", "api")  # export EVAL_MODE=local to use local mode
API_URL = os.environ.get("EVAL_API_URL", "http://127.0.0.1:8000/summarize")

# directories
DEV_DIR = "data/dev"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# model / generation params (can be tuned)
GEN_PARAMS = {
    "input_is_html": False,
    "min_length": 40,
    "max_length": 200,
    "use_reranker": True,
    "top_k": 8,
    "run_qa": True,  # used only for QA factuality; when running remote API set to True for scoring
}

# metrics libs
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score

# optional: local summarizer import (only if MODE == "local" and project is importable)
LOCAL_SUMMARIZER = None
if MODE == "local":
    try:
        # adjust import based on your code layout. This assumes backend.summarizer exposes summarize_pipeline.
        from backend.app import summarize_pipeline  # type: ignore
        LOCAL_SUMMARIZER = summarize_pipeline
        print("Using local summarizer (imported summarize_pipeline).")
    except Exception as e:
        print("Failed to import local summarizer:", e)
        print("Falling back to API mode.")
        MODE = "api"

# helper: call API
def call_api_summarize(article_text: str, params: Dict[str, Any]):
    import requests
    payload = {"text": article_text}
    payload.update(params)
    r = requests.post(API_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

# helper: compute factuality using qa_checker (if available locally)
QA_AVAILABLE = False
try:
    from backend.qa_checker import check_summary_against_source  # type: ignore
    QA_AVAILABLE = True
except Exception:
    QA_AVAILABLE = False

# rouge scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def load_bbc_dataset(dataset_path: str) -> List[Tuple[str, str, str]]:
    """
    Load BBC News Summary dataset.
    
    Returns:
        List of (category, article_text, reference_summary) tuples
    """
    dataset = []
    categories = ["business", "entertainment", "politics", "sport", "tech"]
    
    for category in categories:
        articles_dir = Path(dataset_path) / "Summaries" / category
        if not articles_dir.exists():
            print(f"Warning: {articles_dir} not found, skipping {category}")
            continue
        
        # Get all article files
        article_files = sorted(articles_dir.glob("*.txt"))
        
        for article_file in article_files:
            try:
                with open(article_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                # BBC dataset format: first paragraph is summary, rest is article
                parts = content.split('\n\n', 1)
                if len(parts) == 2:
                    reference_summary = parts[0].strip()
                    article_text = parts[1].strip()
                    
                    if article_text and reference_summary:
                        dataset.append((category, article_text, reference_summary))
            except Exception as e:
                print(f"Error loading {article_file}: {e}")
    
    return dataset


def evaluate_model(model_name: str, dataset: List[Tuple[str, str, str]], 
                   max_samples: int = None) -> Dict[str, float]:
    """
    Evaluate an abstractive model on the dataset.
    
    Args:
        model_name: HuggingFace model name
        dataset: List of (category, article, reference) tuples
        max_samples: Maximum number of samples to evaluate (None = all)
    
    Returns:
        Dictionary with average metrics
    """
    print(f"\nEvaluating {model_name}...")
    summarizer = NewsSummarizer(model_name=model_name, device=-1)  # CPU for stability
    
    all_metrics = []
    samples = dataset[:max_samples] if max_samples else dataset
    
    for i, (category, article, reference) in enumerate(tqdm(samples, desc=model_name)):
        try:
            # Preprocess and summarize
            cleaned_text, _ = preprocess_for_model(article, input_is_html=False)
            summary = summarizer.summarize(cleaned_text)
            
            # Evaluate
            metrics = evaluate_summary(summary, reference, include_bertscore=False)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            continue
    
    # Average metrics
    if not all_metrics:
        return {}
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics


def evaluate_extractive(method: str, dataset: List[Tuple[str, str, str]], 
                        max_samples: int = None) -> Dict[str, float]:
    """
    Evaluate an extractive method on the dataset.
    
    Args:
        method: "tfidf", "textrank", or "lead"
        dataset: List of (category, article, reference) tuples
        max_samples: Maximum number of samples to evaluate (None = all)
    
    Returns:
        Dictionary with average metrics
    """
    print(f"\nEvaluating extractive-{method}...")
    
    method_funcs = {
        "tfidf": extractive_tfidf,
        "textrank": extractive_textrank,
        "lead": extractive_lead
    }
    
    func = method_funcs[method]
    all_metrics = []
    samples = dataset[:max_samples] if max_samples else dataset
    
    for i, (category, article, reference) in enumerate(tqdm(samples, desc=f"extractive-{method}")):
        try:
            # Preprocess and summarize
            cleaned_text, _ = preprocess_for_model(article, input_is_html=False)
            summary = func(cleaned_text, num_sentences=3)
            
            # Evaluate
            metrics = evaluate_summary(summary, reference, include_bertscore=False)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            continue
    
    # Average metrics
    if not all_metrics:
        return {}
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics


def main():
    # Configuration
    dataset_path = r"D:\Code\GitHub\SmartNewsSumm\BBC News Summary"
    max_samples = 100  # Set to None to evaluate entire dataset (slower)
    
    # Models to evaluate
    abstractive_models = [
        "sshleifer/distilbart-cnn-12-6",
        "facebook/bart-large-cnn",
        # "google/flan-t5-base",  # Uncomment to include T5
        # "google/pegasus-cnn_dailymail",  # Uncomment to include PEGASUS
    ]
    
    extractive_methods = ["tfidf", "textrank", "lead"]
    
    # Load dataset
    print("Loading BBC News Summary dataset...")
    dataset = load_bbc_dataset(dataset_path)
    print(f"Loaded {len(dataset)} articles")
    
    if not dataset:
        print("Error: No articles loaded. Check dataset path.")
        return
    
    # Results storage
    results = {}
    
    # Evaluate abstractive models
    for model_name in abstractive_models:
        try:
            start_time = time.time()
            metrics = evaluate_model(model_name, dataset, max_samples)
            elapsed = time.time() - start_time
            
            results[model_name] = metrics
            results[model_name]["time"] = elapsed
            
            print(f"{model_name}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            print(f"  Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")
    
    # Evaluate extractive methods
    for method in extractive_methods:
        try:
            start_time = time.time()
            metrics = evaluate_extractive(method, dataset, max_samples)
            elapsed = time.time() - start_time
            
            method_key = f"extractive-{method}"
            results[method_key] = metrics
            results[method_key]["time"] = elapsed
            
            print(f"{method_key}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            print(f"  Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"Failed to evaluate {method}: {e}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<35} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'BLEU':<10} {'Time(s)':<10}")
    print("-"*80)
    
    for model, metrics in results.items():
        rouge1 = metrics.get("rouge1", 0.0)
        rouge2 = metrics.get("rouge2", 0.0)
        rougeL = metrics.get("rougeL", 0.0)
        bleu = metrics.get("bleu", 0.0)
        elapsed = metrics.get("time", 0.0)
        
        print(f"{model:<35} {rouge1:<10.4f} {rouge2:<10.4f} {rougeL:<10.4f} {bleu:<10.4f} {elapsed:<10.2f}")
    
    # Save results to JSON
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

            total += 1
            if e.get("flagged"):
                flagged += 1
    frac = (flagged / total) if total > 0 else 0.0
    # article flagged: True if any flagged entities exist
    return {
        "overall_flagged_entities": int(flagged),
        "num_entities": int(total),
        "frac_flagged_entities": float(frac),
        "article_flagged": bool(flagged > 0)
    }

def load_dev_files(dev_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(dev_dir, "*.json")))
    items = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
            # require id, article, reference
            if "article" not in obj or "reference" not in obj:
                print(f"Skipping {f} — missing article/reference keys.")
                continue
            items.append({"id": obj.get("id", os.path.basename(f)), "article": obj["article"], "reference": obj["reference"], "path": f})
    return items

def main():
    items = load_dev_files(DEV_DIR)
    if not items:
        print("No dev files found in", DEV_DIR)
        return

    results = []
    cands = []
    refs = []
    ids = []

    for it in tqdm(items, desc="Evaluating"):
        aid = it["id"]
        art = it["article"]
        ref = it["reference"]

        start = time.time()
        if MODE == "api":
            resp = call_api_summarize(art, GEN_PARAMS)
            summary = resp.get("summary", "")
            qa_report = resp.get("qa_report", {})
            debug = resp.get("debug", {})
        else:
            # local: call imported pipeline
            try:
                out = LOCAL_SUMMARIZER(raw_input=art, **GEN_PARAMS)
                summary = out.get("summary", "")
                qa_report = out.get("qa_report", {})
                debug = out.get("debug", {})
            except Exception as e:
                print(f"Local summarizer failed for {aid}: {e}")
                summary = ""
                qa_report = {}
                debug = {}
        end = time.time()

        # metrics
        rouge_res = compute_rouge(ref, summary)
        cands.append(summary)
        refs.append(ref)
        ids.append(aid)

        # bertscore will be computed in batch below (more efficient)
        factual = compute_factuality_flags(summary, art)

        rec = {
            "id": aid,
            "summary": summary,
            "reference": ref,
            "rouge": rouge_res,
            "factual": factual,
            "qa_report": qa_report,
            "debug": debug,
            "time_s": end - start,
            "path": it.get("path")
        }
        results.append(rec)

        # small checkpoint save
        with open(os.path.join(OUT_DIR, "eval_partial.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    # compute BERTScore in batch
    print("Computing BERTScore (this may take a while)...")
    P, R, F = compute_bertscore(cands, refs)
    # attach bertscores
    for i, r in enumerate(results):
        r["bertscore"] = {"precision": P[i], "recall": R[i], "f1": F[i]}

    # aggregate
    agg = {
        "n": len(results),
        "rouge1_mean": sum(r["rouge"]["rouge1"] for r in results) / len(results),
        "rouge2_mean": sum(r["rouge"]["rouge2"] for r in results) / len(results),
        "rougeL_mean": sum(r["rouge"]["rougeL"] for r in results) / len(results),
        "bertscore_f1_mean": sum(r["bertscore"]["f1"] for r in results) / len(results),
        "frac_articles_with_flagged_entities": None
    }
    if QA_AVAILABLE:
        flagged_articles = sum(1 for r in results if r["factual"].get("article_flagged"))
        agg["frac_articles_with_flagged_entities"] = flagged_articles / len(results)

    out_all = {"meta": {"mode": MODE, "gen_params": GEN_PARAMS, "n": len(results)}, "results": results, "aggregate": agg}
    with open(os.path.join(OUT_DIR, "eval_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out_all, fh, indent=2)

    # pretty print
    print("\n=== Evaluation summary ===")
    print(f"Mode: {MODE}, Items: {len(results)}")
    print(f"ROUGE-1 mean: {agg['rouge1_mean']:.4f}, ROUGE-2 mean: {agg['rouge2_mean']:.4f}, ROUGE-L mean: {agg['rougeL_mean']:.4f}")
    print(f"BERTScore-F1 mean: {agg['bertscore_f1_mean']:.4f}")
    if QA_AVAILABLE:
        print(f"Articles with flagged entities: {flagged_articles}/{len(results)} ({agg['frac_articles_with_flagged_entities']:.2%})")
    print("Per-article results saved to:", os.path.join(OUT_DIR, "eval_results.json"))

if __name__ == "__main__":
    main()
