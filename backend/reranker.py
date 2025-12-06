# backend/reranker.py
"""
SBERT-based reranker for chunk (or chunk-summary) selection.

Usage:
  embeddings = SBERTReranker(model_name="all-MiniLM-L6-v2")
  topk_texts = reranker.select_top_k(reference_text, list_of_candidate_texts, k=3)

Design:
- Compute embedding for the reference (full article or question) and candidate chunks.
- Rank candidates by cosine similarity to reference and return top-k.
- Small, fast SBERT models recommended for speed (all-MiniLM-L6-v2).
"""

from typing import List, Tuple
import numpy as np
import logging

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SBERTReranker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Args:
            model_name: SentenceTransformer model id (MiniLM variants are fast & good).
            device: 'cpu' or 'cuda'. If None, model decides automatically.
        """
        logger.info(f"Loading SBERT model: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32):
        """
        Encode list of texts -> np.ndarray embeddings.
        """
        if not texts:
            return np.array([])
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
        return embs

    def select_top_k(self, reference: str, candidates: List[str], k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Rank candidate texts by cosine similarity to reference and return top-k list of tuples:
        (original_index, score, candidate_text)
        """
        if not candidates:
            return []

        # encode
        ref_emb = self.encode([reference])[0]
        c_embs = self.encode(candidates)

        # cosine similarities
        sims = util.cos_sim(ref_emb, c_embs).cpu().numpy()[0]  # shape (n_candidates,)
        # sort by descending sim
        idx = np.argsort(-sims)
        topk = []
        for i in idx[:k]:
            topk.append((int(i), float(sims[i]), candidates[i]))
        logger.info(f"Selected top-{k} candidates (indices): {[t[0] for t in topk]}")
        return topk
