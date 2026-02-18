from adaptive_reranker.labeling.pipeline import Labeler
import math
import time
import ir_datasets
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
import torch

########################################
# CONFIG
########################################

DATASETS = {
    "NFCorpus": "beir/nfcorpus/test",
    # if you have your own wiki-1k as ir_datasets or custom, you can add it here too
}

TOP_K_RETRIEVE = 50    # BM25 candidates per query for reranking
K_EVAL = 10            # nDCG@10 / MRR@10
LAMBDA_COST = 0.05     # latency penalty weight (accuracy vs speed tradeoff)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


if __name__ == "__main__":
    print("Loading rerankers...")
    ce_l6 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    bge_reranker = FlagReranker(
        "BAAI/bge-reranker-v2-m3",
        use_fp16=(device == "cuda")
    )

    labeler = Labeler(DATASETS, {
        "ce_l6": ce_l6,
        "bge_reranker": bge_reranker
    }, device=device, top_k_retrieve=TOP_K_RETRIEVE, lambda_cost=LAMBDA_COST, eval_at_k=K_EVAL)
    
    labels = labeler.run()

    print("\nAll done. You should now have one CSV per dataset with labeled queries.")
