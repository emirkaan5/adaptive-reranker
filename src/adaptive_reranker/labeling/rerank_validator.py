from pydantic import BaseModel, model_validator
from typing import Any
import math

class RankedResult(BaseModel):
    doc_id: str
    score: float

    @model_validator(mode='before')
    @classmethod
    def normalize(cls, v: Any) -> dict:
        # already a dict with right keys
        if isinstance(v, dict):
            doc_id = next((v[k] for k in ["doc_id", "id", "document_id"] if k in v), None)
            score = next((v[k] for k in ["score", "relevance", "logit"] if k in v), None)
            score = float(score)
            if not math.isfinite(score):
                score = -1e9
            return {"doc_id": str(doc_id), "score": score}
        
        # tuple (doc_id, score)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            score = float(v[1])
            if not math.isfinite(score):
                score = -1e9
            return {"doc_id": str(v[0]), "score": score}

        # plain scalar score (supports numpy scalars too) — caller injects doc_id separately
        try:
            score = float(v)
            if not math.isfinite(score):
                score = -1e9
            return {"doc_id": "", "score": score}
        except (TypeError, ValueError):
            pass

        raise ValueError(f"Cannot parse reranker result: {v!r}")


def normalize_reranker_output(self, raw_output, candidates) -> list[tuple[str, float]]:
    """
    Normalize the reranker output. This is used to standardize the reranker output to a list of (doc_id, score) tuples.
    Args:
        raw_output: The raw output from the reranker.
        candidates: The candidates for the query.

    Returns:
        The normalized reranker output.
    """
    if raw_output is None:
        return []

    if hasattr(raw_output, "tolist"):
        raw_output = raw_output.tolist()

    doc_ids = [doc_id for doc_id, _ in candidates]
    if len(raw_output) == 0:
        return []

    # Plain scores list/array — zip with doc_ids first
    if not isinstance(raw_output[0], (dict, tuple, list)):
        raw_output = list(zip(doc_ids, raw_output))

    results = [RankedResult.model_validate(item) for item in raw_output]
    return sorted([(r.doc_id, r.score) for r in results], key=lambda x: x[1], reverse=True)

def call_reranker(self, reranker, pairs):
    """
    Call a reranker regardless of implementation style.
    Supports:
        - callable rerankers: reranker(pairs)
        - SentenceTransformers CrossEncoder: reranker.predict(pairs)
        - FlagEmbedding rerankers: reranker.compute_score(pairs)
    """
    if hasattr(reranker, "predict"):
        return reranker.predict(pairs)
    if hasattr(reranker, "compute_score"):
        return reranker.compute_score(pairs)
    if callable(reranker):
        return reranker(pairs)
    raise TypeError("Unsupported reranker object; provide a callable, .predict(), or .compute_score().")
