from typing import List, Dict
import math


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / len(retrieved_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / len(relevant_set)


def _dcg(scores: List[int]) -> float:
    dcg = 0.0
    for i, rel in enumerate(scores):
        denom = math.log2(i + 2)
        dcg += rel / denom
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    scores = [1 if doc in relevant_set else 0 for doc in retrieved_k]
    ideal = sorted(scores, reverse=True)
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(scores) / idcg


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / i
    return 0.0


def evaluate_query(retrieved: List[str], relevant: List[str], ks: List[int] = [1, 5, 10]) -> Dict:
    res = {}
    for k in ks:
        res[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        res[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        res[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    res["mrr"] = mrr(retrieved, relevant)
    return res


def evaluate_batch(results: List[Dict], ks: List[int] = [1, 5, 10]) -> Dict:
    """
    Evaluate a batch of retrieval results.

    Args:
        results: list of dicts with keys: 'retrieved' (List[str]) and 'relevant' (List[str])
    Returns:
        aggregated average metrics
    """
    agg = {}
    n = len(results) if results else 0
    if n == 0:
        return {}

    sums = {f"precision@{k}": 0.0 for k in ks}
    sums.update({f"recall@{k}": 0.0 for k in ks})
    sums.update({f"ndcg@{k}": 0.0 for k in ks})
    sums["mrr"] = 0.0

    for r in results:
        ev = evaluate_query(r.get("retrieved", []), r.get("relevant", []), ks)
        for k in ks:
            sums[f"precision@{k}"] += ev[f"precision@{k}"]
            sums[f"recall@{k}"] += ev[f"recall@{k}"]
            sums[f"ndcg@{k}"] += ev[f"ndcg@{k}"]
        sums["mrr"] += ev["mrr"]

    return {k: v / n for k, v in sums.items()}
