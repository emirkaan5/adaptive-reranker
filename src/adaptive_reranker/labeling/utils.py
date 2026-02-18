import math

def ndcg_at_k(run_for_q, qrels_for_q, k=10):
    """
    Calculate the nDCG@k for a given run and qrels.

    Args:
        run_for_q: The run for the query.
        qrels_for_q: The qrels for the query.
        k: The number of results to consider.

    Returns:
        The nDCG@k for the given run and qrels.
    """
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(run_for_q[:k], start=1):
        rel = qrels_for_q.get(doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal = sorted(qrels_for_q.values(), reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal[:k], start=1):
        idcg += (2**rel - 1) / math.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(run_for_q, qrels_for_q, k=10):
    """
    Calculate the MRR@k for a given run and qrels.

    Args:
        run_for_q: The run for the query.
        qrels_for_q: The qrels for the query.
        k: The number of results to consider.

    Returns:
        The MRR@k for the given run and qrels.
    """
    for rank, (doc_id, _) in enumerate(run_for_q[:k], start=1):
        if qrels_for_q.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def per_query_metrics(runs, qrels, k=10, eval_funcs = None,):
    """
    Calculate the per-query metrics for a given run and qrels.
    Args:
        runs: The runs for the queries.
        qrels: The qrels for the queries.
        k: The number of results to consider.

    Returns:
        The per-query metrics for the given run and qrels.
    """
    if eval_funcs is None:
        eval_funcs = {
            "ndcg": ndcg_at_k,
            "mrr": mrr_at_k
        }
    out = {}
    for qid, qrel in qrels.items():
        if qid not in runs:
            continue
        run_q = runs[qid]
        out[qid] = {eval_func_name: eval_func(run_q, qrel, k=k) for eval_func_name, eval_func in eval_funcs.items()}

    return out

def combine_score(self, m):
    """
    Combine the nDCG and MRR into a single [0,1] effectiveness scalar.
    Args:
        m: query metrics for a given query.

    Returns:
        The combined effectiveness scalar.
    """

    return 0.5 * m["ndcg"] + 0.5 * m["mrr"]
