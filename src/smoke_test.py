import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"


def load_text_files(data_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in sorted(data_dir.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                docs.append({"content": text, "metadata": {"source": p.name}})
        except Exception:
            print(f"Failed to read {p}")
    return docs


def tfidf_fallback_query(docs, query, topk=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [d["content"] if isinstance(d, dict) else str(d) for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf)[0]
    idxs = sims.argsort()[::-1][:topk]

    return [{
        "scores": sims[idxs].tolist(),
        "idxs": idxs.tolist(),
        "texts": [texts[i] for i in idxs],
    }]


def _print_header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_smoke_test(assertions: bool = True) -> int:
    _print_header("SMOKE TEST: load docs")
    docs = load_text_files(DATA_DIR)
    print(f"Loaded {len(docs)} docs from {DATA_DIR}")

    failures = []

    queries = [
        "What is MLOps?",
        "How to evaluate models?",
        "Feature engineering techniques",
    ]

    VectorDB = None
    try:
        from vectordb import VectorDB as _VDB

        VectorDB = _VDB
    except Exception:
        VectorDB = None

    if VectorDB is None:
        _print_header("TF-IDF fallback")
        for q in queries:
            print("\nQuery:", q)
            results = tfidf_fallback_query(docs, q, topk=3)
            if not results or not results[0]["texts"]:
                failures.append(f"No TF-IDF results for query: {q}")
                continue
            for rank, (score, text) in enumerate(
                zip(results[0]["scores"], results[0]["texts"]), start=1
            ):
                print(f"{rank:>2}. score={score:.4f}")
                print("   ", text[:240].replace("\n", " "))

    else:
        _print_header("VectorDB path")
        try:
            vdb = VectorDB(chunk_size=300, chunk_overlap=60, use_tfidf_rerank=True)
            print("Adding documents to VectorDB...")
            vdb.add_documents(docs)

            for q in queries:
                print("\nQuery:", q)
                res = vdb.search(q, n_results=3)
                docs_out = res.get("documents", [])
                metas = res.get("metadatas", [])
                dists = res.get("distances", [])

                if not docs_out:
                    failures.append(f"No VectorDB results for query: {q}")
                    print("No results — check embeddings or increase candidates")
                    continue

                for i, doc in enumerate(docs_out, start=1):
                    meta = metas[i - 1] if i - 1 < len(metas) else {}
                    dist = dists[i - 1] if i - 1 < len(dists) else None
                    src = meta.get("source") or f"doc_index={meta.get('doc_index','?')}"
                    print(f"{i:>2}. score={dist if dist is not None else 'N/A'}")
                    print("   ", src, "|", doc[:240].replace("\n", " "))

        except Exception:
            print("VectorDB failed — falling back to TF-IDF")
            traceback.print_exc()
            for q in queries:
                print("\nQuery:", q)
                results = tfidf_fallback_query(docs, q, topk=3)
                if not results or not results[0]["texts"]:
                    failures.append(f"No TF-IDF fallback results for query: {q}")
                    continue
                for rank, (score, text) in enumerate(
                    zip(results[0]["scores"], results[0]["texts"]), start=1
                ):
                    print(f"{rank:>2}. score={score:.4f}")
                    print("   ", text[:240].replace("\n", " "))

    # Basic assertions
    if assertions:
        if len(docs) == 0:
            failures.append("No documents were loaded from data directory")

    if failures:
        _print_header("SMOKE TEST: FAILURES")
        for f in failures:
            print("-", f)
        print("\nSmoke test completed with failures.")
        return 2

    _print_header("SMOKE TEST: SUCCESS")
    print("Basic checks passed — retrieval returned results for queries.")
    return 0


if __name__ == "__main__":
    code = run_smoke_test(assertions=True)
    sys.exit(code)
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"


def load_text_files(data_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in sorted(data_dir.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                docs.append({"content": text, "metadata": {"source": p.name}})
        except Exception:
            print(f"Failed to read {p}")
    return docs


def tfidf_fallback_query(docs, query, topk=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [d["content"] if isinstance(d, dict) else str(d) for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf)[0]
    idxs = sims.argsort()[::-1][:topk]

    scores = sims[idxs].tolist()
    sel_texts = [texts[i] for i in idxs]
    return [{
        "scores": scores,
        "idxs": idxs.tolist(),
        "texts": sel_texts,
    }]


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_smoke_test(assertions: bool = True) -> int:
    _print_header("SMOKE TEST: Loading documents")
    docs = load_text_files(DATA_DIR)
    print(f"Loaded {len(docs)} documents from {DATA_DIR}")

    failures = []

    queries = [
        "What is MLOps?",
        "How to evaluate models?",
        "Feature engineering techniques",
    ]

    # Lazy import VectorDB so the test can run in minimal env
    VectorDB = None
    try:
        from vectordb import VectorDB as _VDB

        VectorDB = _VDB
    except Exception:
        VectorDB = None

    if VectorDB is None:
        _print_header("VectorDB not available — using TF-IDF fallback")
        for q in queries:
            print("\nQuery:", q)
            results = tfidf_fallback_query(docs, q, topk=3)
            if not results or not results[0]["texts"]:
                failures.append(f"No TF-IDF results for query: {q}")
                continue
            for rank, (score, text) in enumerate(
                zip(results[0]["scores"], results[0]["texts"]), start=1
            ):
                print(f"{rank:>2}. score={score:.4f}")
                print("   ", text[:240].replace("\n", " "))

    else:
        _print_header("VectorDB available — running embedding ingestion + search")
        try:
            vdb = VectorDB(chunk_size=300, chunk_overlap=60, use_tfidf_rerank=True)
            print("Adding documents to VectorDB (may download models)...")
            vdb.add_documents(docs)

            for q in queries:
                print(f"\nQuery: {q}")
                res = vdb.search(q, n_results=3)
                docs_out = res.get("documents", [])
                metas = res.get("metadatas", [])
                dists = res.get("distances", [])

                if not docs_out:
                    failures.append(f"No VectorDB results for query: {q}")
                    print("No results — consider increasing candidate set or checking embeddings")
                    continue

                for i, doc in enumerate(docs_out, start=1):
                    meta = metas[i - 1] if i - 1 < len(metas) else {}
                    dist = dists[i - 1] if i - 1 < len(dists) else None
                    src = meta.get("source") or \
                        f"doc_index={meta.get('doc_index','?')}"
                    print(f"{i:>2}. score={dist if dist is not None else 'N/A'}")
                    print("   ", src, "|", doc[:240].replace("\n", " "))

        except Exception:
            print("VectorDB path failed — falling back to TF-IDF")
            traceback.print_exc()
            for q in queries:
                print(f"\nQuery: {q}")
                results = tfidf_fallback_query(docs, q, topk=3)
                if not results or not results[0]["texts"]:
                    failures.append(f"No TF-IDF fallback results for query: {q}")
                    continue
                for rank, (score, text) in enumerate(
                    zip(results[0]["scores"], results[0]["texts"]), start=1
                ):
                    print(f"{rank:>2}. score={score:.4f}")
                    print("   ", text[:240].replace("\n", " "))

    # Basic assertions
    if assertions:
        if len(docs) == 0:
            failures.append("No documents were loaded from data directory")

    if failures:
        _print_header("SMOKE TEST: FAILURES")
        for f in failures:
            print("-", f)
        print("\nSmoke test completed with failures.")
        return 2

    _print_header("SMOKE TEST: SUCCESS")
    print("All basic checks passed — retrieval returned results for queries.")
    return 0


if __name__ == "__main__":
    code = run_smoke_test(assertions=True)
    sys.exit(code)
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

VectorDB = None

DATA_DIR = ROOT / "data"


def load_text_files(data_dir: Path):
    docs = []
    for p in sorted(data_dir.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                docs.append({"content": text, "metadata": {"source": p.name}})
        except Exception:
            print(f"Failed to read {p}")
    return docs


def tfidf_fallback_query(docs, query, topk=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [d["content"] if isinstance(d, dict) else str(d) for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf)[0]
    idxs = sims.argsort()[::-1][:topk]
    return [(idxs.tolist(), [texts[i] for i in idxs])]


def run_smoke_test():
    print("Smoke test started: loading documents...")
    docs = load_text_files(DATA_DIR)
    print(f"Loaded {len(docs)} documents from {DATA_DIR}")

    # sample queries based on sample_documents.txt
    queries = [
        "What is MLOps?",
        "How to evaluate models?",
        "Feature engineering techniques",
    ]

    # Try to import VectorDB here to allow running smoke test even if optional deps are missing
    try:
        from vectordb import VectorDB as _VDB
        VectorDB = _VDB
    except Exception:
        VectorDB = None

    if VectorDB is None:
        print("VectorDB not available (missing optional dependencies). Using TF-IDF fallback for retrieval.")
        for q in queries:
            print("\n----\nQuery:", q)
            fb = tfidf_fallback_query(docs, q, topk=3)
            for idxs, texts in fb:
                for t in texts:
                    print("-", t[:400].replace('\n', ' '))
        return

    try:
        # configure a smaller chunk size for quick tests
        vdb = VectorDB(chunk_size=300, chunk_overlap=60, use_tfidf_rerank=True)
        print("Adding documents to VectorDB (this may download embeddings)...")
        vdb.add_documents(docs)

        for q in queries:
            print("\n----\nQuery:", q)
            res = vdb.search(q, n_results=3)
            docs_out = res.get("documents", [])
            metas = res.get("metadatas", [])
            ids = res.get("ids", [])
            dists = res.get("distances", [])
            if not docs_out:
                print("No results from VectorDB; falling back to TF-IDF")
                fb = tfidf_fallback_query(docs, q, topk=3)
                for idxs, texts in fb:
                    for t in texts:
                        print("-", t[:300].replace('\n',' '))
            else:
                for i, doc in enumerate(docs_out):
                    print(f"Result {i+1} (id={ids[i] if i < len(ids) else 'N/A'}):")
                    print(doc[:400].replace("\n", " "))

    except Exception:
        print("VectorDB smoke test failed with exception:")
        traceback.print_exc()
        print("Falling back to TF-IDF only retrieval")
        for q in queries:
            print("\n----\nQuery:", q)
            fb = tfidf_fallback_query(docs, q, topk=3)
            for idxs, texts in fb:
                for t in texts:
                    print("-", t[:400].replace('\n', ' '))


if __name__ == "__main__":
    run_smoke_test()
import os
import sys
import math
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")


def simple_chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start
        current_len = 0
        while end < len(words) and current_len + len(words[end]) + 1 <= chunk_size:
            current_len += len(words[end]) + 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap // 5, end - 1)
    return chunks


def load_documents(data_dir: str) -> List[str]:
    docs = []
    for fn in os.listdir(data_dir):
        if fn.endswith(".txt"):
            with open(os.path.join(data_dir, fn), "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    docs.append(txt)
    return docs


def run_smoke_test(query: str = "model evaluation methods"):
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents from {DATA_DIR}")
    all_chunks = []
    provenance = []
    for i, doc in enumerate(docs):
        chunks = simple_chunk_text(doc, chunk_size=300, overlap=50)
        for j, c in enumerate(chunks):
            all_chunks.append(c)
            provenance.append({"doc_index": i, "chunk_index": j})

    print(f"Created {len(all_chunks)} chunks")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(all_chunks)
        qv = vec.transform([query])
        sims = cosine_similarity(qv, X)[0]
        top_idx = sims.argsort()[::-1][:5]
        print("Top matches:")
        for idx in top_idx:
            print(f"- score={sims[idx]:.4f} doc={provenance[idx]['doc_index']} chunk={provenance[idx]['chunk_index']}")
            snippet = all_chunks[idx]
            print(snippet[:400].replace("\n", " "))
            print("---")
    except Exception as e:
        print("skipping TF-IDF step, sklearn not available:", e)
        # fallback: simple substring ranking
        scores = [1 if query.lower() in c.lower() else 0 for c in all_chunks]
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        for idx in top_idx:
            print(f"- match doc={provenance[idx]['doc_index']} chunk={provenance[idx]['chunk_index']}")
            print(all_chunks[idx][:400].replace("\n", " "))
            print("---")


if __name__ == '__main__':
    run_smoke_test()
