# scripts/evaluate.py
"""
Evaluation harness for the News Summarizer project.

Modes:
- "local" : import and call summarizer functions/classes directly (fast, requires Python imports)
- "api"   : call the running FastAPI endpoint at http://127.0.0.1:8000/summarize

Outputs:
- results/eval_results.json (per-article metrics + aggregates)
- prints a short summary to stdout
"""

import os
import json
import glob
import time
from typing import Dict, Any, List
from tqdm import tqdm

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

def compute_rouge(reference: str, predicted: str) -> Dict[str, float]:
    scores = scorer.score(reference, predicted)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

def compute_bertscore(cands: List[str], refs: List[str], model_type="microsoft/deberta-xlarge-mnli"):
    # return precision, recall, f1 arrays
    P, R, F = bertscore_score(cands, refs, lang="en", rescale_with_baseline=True)
    # convert tensors to floats
    return [float(x) for x in P], [float(x) for x in R], [float(x) for x in F]

def compute_factuality_flags(summary: str, source: str) -> Dict[str, Any]:
    """
    Use qa_checker.check_summary_against_source if available.
    Returns:
      {
        "overall_flagged_entities": int,
        "num_entities": int,
        "frac_flagged_entities": float,
        "article_flagged": bool
      }
    """
    if not QA_AVAILABLE:
        return {"overall_flagged_entities": None, "num_entities": None, "frac_flagged_entities": None, "article_flagged": None}

    res = check_summary_against_source(summary, source)
    total = 0
    flagged = 0
    for s in res.get("sentence_checks", []):
        for e in s.get("entities", []):
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
