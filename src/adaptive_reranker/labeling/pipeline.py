import math
import time
import ir_datasets
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import torch
from typing import Callable
from typing import Any
from rerank_validator import normalize_reranker_output, call_reranker
from utils import per_query_metrics, combine_score


class Labeler:
    def __init__(self, datasets : dict[str, str], rerankers : dict[str, Callable], eval_funcs:dict[str, Callable] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu", top_k_retrieve: int = 100, lambda_cost: float = 0.05, eval_at_k: int = 10):
        """
        Initialize the Labeler.
        Args:
            datasets: The datasets to label.
            rerankers: The rerankers to use.
            eval_funcs: The evaluation functions to use. (this is optional, if not provided, the default evaluation functions will be used)
            device: The device to use. (this is optional, if not provided, the default device will be used)
        """
        self.datasets = datasets
        self.rerankers = rerankers
        self.eval_funcs = eval_funcs
        self.device = device
        self.corpus, self.queries, self.qrels, self.tokenized_docs, self.doc_ids = self.build_corpus_and_qrels()
        self.top_k_retrieve = top_k_retrieve
        self.lambda_cost = lambda_cost
        self.eval_at_k = eval_at_k



    def bm25_search(self,query_text, dataset_id, top_k=100):
        """
        Perform a BM25 search for a given query and dataset id.
        Args:
            query_text: The query text.
            dataset_id: The dataset id.
            top_k: The number of results to return.

        Returns:
            The BM25 search results.
        """
        bm25 = BM25Okapi(self.tokenized_docs[dataset_id])
        tokens = query_text.lower().split()
        scores = bm25.get_scores(tokens)
        idx_scores = list(enumerate(scores))
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        idx_scores = idx_scores[:top_k]
        return [(self.doc_ids[dataset_id][i], float(score)) for i, score in idx_scores]



    def build_corpus_and_qrels(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, dict[str, int]]], dict[str, list[list[str]]], dict[str, list[str]]]:
        """
        Build the corpus.
        Args:
            dataset_id: The dataset to build the corpus from.
        """
        dataset_ids = list(self.datasets.values())
        corpus = {dataset_id: {} for dataset_id in dataset_ids}
        queries = {}
        qrels = {}
        tokenized_docs = {}
        doc_ids = {}
        for dataset_id in dataset_ids:
            ds = ir_datasets.load(dataset_id)
            for d in ds.docs_iter():
                text  = getattr(d, "text", "")
                title = getattr(d, "title", "")
                full_text = (title + " " + text).strip()
                corpus[dataset_id][d.doc_id] = full_text if full_text else text
            queries[dataset_id] = {q.query_id: q.text for q in ds.queries_iter()}
            doc_ids[dataset_id] = list(corpus[dataset_id].keys())
            doc_texts = [corpus[dataset_id][did] for did in doc_ids[dataset_id]]
            tokenized_docs[dataset_id] = [t.lower().split() for t in doc_texts]
            qrels[dataset_id] = {}
            for qr in ds.qrels_iter():
                qrels[dataset_id].setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance

        return corpus, queries, qrels, tokenized_docs, doc_ids


    def generate_query_latency(self, dataset_id : str, top_k : int = 100):
        """
        Generate the query latency for a given dataset and query id.
        Args:
            dataset_id: The dataset id.
            query_id: The query id.
            top_k: The number of results to return.

        Returns:
            The labels for the given dataset and query id.
        """
        bm25_runs = {}
        bm25_time = 0.0
        reranker_runs = {reranker_name: {} for reranker_name in self.rerankers.keys()}
        reranker_time_totals = {reranker_name: 0.0 for reranker_name in self.rerankers.keys()}
        num_queries = len(self.queries[dataset_id])
        if num_queries == 0:
            return reranker_runs, {name: 0.0 for name in self.rerankers}, bm25_runs, 0.0

        for i, (qid, qtext) in enumerate(self.queries[dataset_id].items(), start=1):
            t0 = time.perf_counter()
            cand = self.bm25_search(qtext, dataset_id, top_k=self.top_k_retrieve)
            bm25_time += time.perf_counter() - t0
            bm25_runs[qid] = cand

            for reranker_name, reranker in self.rerankers.items():
                t0 = time.perf_counter()
                pairs = [[qtext, self.corpus[dataset_id][doc_id]] for doc_id, _ in cand]
                raw_scores = call_reranker(reranker, pairs)
                reranker_runs[reranker_name][qid] = normalize_reranker_output(raw_scores, cand)
                reranker_time_totals[reranker_name] += time.perf_counter() - t0

        bm25_time /= num_queries
        reranker_times = {
            reranker_name: reranker_time_totals[reranker_name] / num_queries
            for reranker_name in self.rerankers.keys()
        }

        # Print summary of the results
        print(f"Dataset: {dataset_id}")
        print(f"Number of queries: {len(self.queries[dataset_id])}")
        print(f"BM25 runs: {len(bm25_runs)}")
        for reranker_name, reranker_run in reranker_runs.items():
            print(f"{reranker_name} runs: {len(reranker_run)}")
            print(f"Average latency: {reranker_times[reranker_name]} seconds")
        return reranker_runs, reranker_times,bm25_runs, bm25_time

    def build_labels(self, dataset_id : str):
        """
        Build the labels for a given dataset.
        Args:
            dataset_id: The dataset id.

        Returns:
            The labels for the given dataset.
        """
        reranker_runs, reranker_times, bm25_runs,bm25_time = self.generate_query_latency(dataset_id)

        bm25_metrics = per_query_metrics(bm25_runs, self.qrels[dataset_id])
        reranker_metrics = {}
        for reranker_name, reranker_run in reranker_runs.items():
            reranker_metrics[reranker_name] = per_query_metrics(reranker_run, self.qrels[dataset_id])

        metrics_all = {
            "bm25": bm25_metrics,
            **reranker_metrics
        }

        latencies = {
            "bm25": bm25_time,
            **reranker_times
        }

        max_lat = max(max(latencies.values()), 1e-12)
        model_names = list(metrics_all.keys())
        model_to_class = {"bm25": 0}
        model_to_class.update({name: idx for idx, name in enumerate(self.rerankers.keys(), start=1)})

        labels = {}
        for qid in self.qrels[dataset_id].keys():
        # ensure we have metrics for this qid in all three models
            if any(qid not in metrics_all[m] for m in model_names):
                continue

            utilities = {}
            eff_scores = {}

            for m in model_names:
                eff = combine_score(metrics_all[m][qid])     # [0,1]
                cost_norm = latencies[m] / max_lat          # [0,1]
                u = eff - self.lambda_cost * cost_norm           # utility
                utilities[m] = u
                eff_scores[m] = eff

            # pick best model by utility
            best_model, best_u = max(utilities.items(), key=lambda x: x[1])

            # map model -> class (0=bm25, 1..n=rerankers in declared order)
            cls = model_to_class[best_model]
            labels[qid] = {
                "class": cls,
                "best_model": best_model,
                "best_utility": best_u,
                "per_model": {
                    m: {
                        **metrics_all[m][qid],
                        "effectiveness": eff_scores[m],
                        "utility": utilities[m],
                        "latency": latencies[m],
                    }
                    for m in model_names
                },
            }

        return labels

    def save_labels(self, dataset_id: str, labels: dict[str, dict[str, Any]], path: str | None = None):
        """
        Flatten labels + query text into a CSV with one row per query.
        Per-model metrics are stored as prefixed columns (e.g. bm25_ndcg, bm25_mrr, ...).
        """
        rows = []
        for qid, lbl in labels.items():
            row = {
                "qid": qid,
                "query": self.queries[dataset_id].get(qid, ""),
                "class": lbl["class"],
                "best_model": lbl["best_model"],
                "best_utility": lbl["best_utility"],
            }
            for model_name, model_metrics in lbl["per_model"].items():
                for metric_name, value in model_metrics.items():
                    row[f"{model_name}_{metric_name}"] = value
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = path or f"labels_{dataset_id.replace('/', '_')}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} labelled queries to {out_path}")
        return df

    def run(self, output_dir: str = "labels") -> dict[str, pd.DataFrame]:
        """
        End-to-end pipeline: build labels for every dataset, save each as a
        CSV under *output_dir*, and return {dataset_id: DataFrame}.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        results: dict[str, pd.DataFrame] = {}
        for dataset_id in self.datasets.values():
            print(f"\n{'='*60}\nProcessing {dataset_id}\n{'='*60}")
            labels = self.build_labels(dataset_id)
            safe_name = dataset_id.replace("/", "_")
            path = os.path.join(output_dir, f"labels_{safe_name}.csv")
            df = self.save_labels(dataset_id, labels, path=path)
            results[dataset_id] = df

        print(f"\nDone — {len(results)} dataset(s) labelled.")
        return results