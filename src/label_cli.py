"""Simple CLI to collect relevance judgments and run retrieval evaluation.

Features:
- Interactive labeling of top-k retrievals per query
- `--auto` mode for heuristic automatic labeling (useful for CI/demo)
- Saves annotations to a JSON file and runs `evaluate.evaluate_batch`
"""
import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"


def load_texts(data_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in sorted(data_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"content": text, "metadata": {"source": p.name}})
    return docs


def tfidf_retrieve(docs, query, topk=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [d["content"] for d in docs]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = sims.argsort()[::-1][:topk]
    return [{
        "ids": [f"doc_{i}" for i in idxs.tolist()],
        "scores": sims[idxs].tolist(),
        "texts": [texts[i] for i in idxs.tolist()],
    }]


def interactive_label(retrieved: List[str]) -> List[str]:
    relevant = []
    for i, rid in enumerate(retrieved, start=1):
        while True:
            prompt = (
                f"Mark result {i} (id={rid}) as relevant? "
                "[y/n/s(skip)]: "
            )
            ans = input(prompt).strip().lower()
            if ans in ("y", "yes"):
                relevant.append(rid)
                break
            if ans in ("n", "no"):
                break
            if ans in ("s", "skip"):
                return []
            print("Please answer 'y' or 'n' or 's'")
    return relevant


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="annotations.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-label using score threshold",
    )
    parser.add_argument("--auto-threshold", type=float, default=0.1)
    parser.add_argument("--queries-file", type=str, default=None)
    args = parser.parse_args(argv)

    docs = load_texts(DATA_DIR)
    if not docs:
        print("No documents found in data/. Add some .txt files first.")
        return 2

    if args.queries_file:
        qpath = Path(args.queries_file)
        raw = qpath.read_text(encoding="utf-8")
        queries = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        queries = [
            "What is MLOps?",
            "How to evaluate models?",
            "Feature engineering techniques",
        ]

    # Try VectorDB
    VectorDB = None
    try:
        from vectordb import VectorDB as _VDB

        VectorDB = _VDB
    except Exception:
        VectorDB = None

    annotations = []

    for q in queries:
        print("\n---\nQuery:", q)
        retrieved_ids = []
        retrieved_texts = []
        retrieved_scores = []

        if VectorDB is not None:
            try:
                vdb = VectorDB(
                    chunk_size=400,
                    chunk_overlap=80,
                    use_tfidf_rerank=True,
                )
                res = vdb.search(q, n_results=args.topk)
                retrieved_ids = res.get("ids", [])
                retrieved_texts = res.get("documents", [])
                retrieved_scores = res.get("distances", [])
            except Exception:
                VectorDB = None

        if VectorDB is None:
            tf = tfidf_retrieve(docs, q, topk=args.topk)[0]
            retrieved_ids = tf["ids"]
            retrieved_texts = tf["texts"]
            retrieved_scores = tf["scores"]

        for i, (rid, txt, sc) in enumerate(
            zip(retrieved_ids, retrieved_texts, retrieved_scores), start=1
        ):
            print(f"{i:>2}. id={rid} score={sc}")
            print("   ", txt[:240].replace("\n", " "))

        if args.auto:
            relevant = [
                rid
                for rid, sc in zip(retrieved_ids, retrieved_scores)
                if sc >= args.auto_threshold
            ]
            print("Auto-labeled relevant ids:", relevant)
        else:
            relevant = interactive_label(retrieved_ids)

        annotations.append(
            {"query": q, "retrieved": retrieved_ids, "relevant": relevant}
        )

    # Save annotations
    out_path = Path(args.output)
    out_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    print(f"Annotations saved to {out_path}")

    # Run evaluation
    try:
        from evaluate import evaluate_batch

        results = []
        for a in annotations:
            results.append({
                "retrieved": a["retrieved"],
                "relevant": a["relevant"],
            })
        metrics = evaluate_batch(results, ks=[1, 3, 5])
        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=2))
    except Exception:
        print("Failed to run evaluation:")
        traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
