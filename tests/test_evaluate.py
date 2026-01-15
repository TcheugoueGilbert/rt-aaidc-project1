from src import evaluate


def test_precision_recall_mrr_ndcg():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d"]

    p1 = evaluate.precision_at_k(retrieved, relevant, 1)
    assert p1 == 0.0

    p2 = evaluate.precision_at_k(retrieved, relevant, 2)
    assert p2 == 0.5

    r2 = evaluate.recall_at_k(retrieved, relevant, 2)
    assert r2 == 0.5

    m = evaluate.mrr(retrieved, relevant)
    assert abs(m - 1.0/2.0) < 1e-6

    ndcg2 = evaluate.ndcg_at_k(retrieved, relevant, 2)
    assert ndcg2 > 0.0


def test_evaluate_query_and_batch():
    retrieved = ["x", "y", "z"]
    relevant = ["y"]
    res = evaluate.evaluate_query(retrieved, relevant, ks=[1, 3])
    assert "precision@1" in res and "precision@3" in res

    batch = [
        {"retrieved": ["a", "b"], "relevant": ["b"]},
        {"retrieved": ["c"], "relevant": ["d"]},
    ]
    agg = evaluate.evaluate_batch(batch, ks=[1, 2])
    assert "precision@1" in agg and "mrr" in agg
