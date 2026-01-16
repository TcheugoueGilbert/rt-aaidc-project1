"""Streamlit-based annotator for retrieval labeling.

Run:
  pip install -r requirements.txt
  streamlit run src/label_app.py

This app lets you enter queries (or use predefined ones), retrieve top-k
results via VectorDB (if available) or TF-IDF fallback, mark relevance, save
annotations, and view evaluation metrics.
"""
from pathlib import Path
from typing import List, Dict, Any
import json
import traceback

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
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


def get_retriever():
    try:
        from vectordb import VectorDB

        return VectorDB(
            chunk_size=400,
            chunk_overlap=80,
            use_tfidf_rerank=True,
        )
    except Exception:
        return None


def run_app():
    st.title("RAG Retrieval Labeler")
    st.write("Annotate retrieval results and compute metrics")

    # use loader to support multiple file formats
    from loader import load_documents_from_dir

    # Ensure data dir exists and allow users to upload files
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    uploaded = st.sidebar.file_uploader(
        "Upload files", type=["txt", "pdf", "docx", "csv", "xls", "xlsx"], accept_multiple_files=True
    )
    if uploaded:
        for f in uploaded:
            try:
                target = DATA_DIR / f.name
                # avoid overwriting existing files
                if target.exists():
                    base = target.stem
                    suf = target.suffix
                    i = 1
                    while True:
                        candidate = DATA_DIR / f"{base}_{i}{suf}"
                        if not candidate.exists():
                            target = candidate
                            break
                        i += 1

                data = f.read()
                with open(target, "wb") as out:
                    out.write(data)
                st.sidebar.success(f"Saved {f.name} -> data/{target.name}")
            except Exception as e:
                st.sidebar.error(f"Failed to save {f.name}: {e}")

    docs = load_documents_from_dir(DATA_DIR)
    st.sidebar.write(f"Loaded {len(docs)} documents from data/")

    default_queries = [
        "What is MLOps?",
        "How to evaluate models?",
        "Feature engineering techniques",
    ]

    queries_text = st.sidebar.text_area(
        "Queries (one per line)", "\n".join(default_queries), height=150
    )
    queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
    topk = st.sidebar.slider("Top-k", 1, 10, 3)
    auto_threshold = st.sidebar.slider(
        "Auto threshold (TF-IDF score)", 0.0, 1.0, 0.05
    )
    use_auto = st.sidebar.checkbox("Auto-label by threshold", value=False)

    retriever = get_retriever()
    if retriever is None:
        st.sidebar.info("VectorDB not available — using TF-IDF fallback")

    out_file = st.sidebar.text_input(
        "Annotations output file", "annotations_streamlit.json"
    )

    if st.button("Run retrieval and annotate"):
        annotations = []
        for q in queries:
            st.header(f"Query: {q}")
            if retriever is not None:
                try:
                    res = retriever.search(q, n_results=topk)
                    ids = res.get("ids", [])
                    texts = res.get("documents", [])
                    scores = res.get("distances", [])
                except Exception:
                    st.error("VectorDB error, falling back to TF-IDF")
                    st.text(traceback.format_exc())
                    res = tfidf_retrieve(docs, q, topk=topk)[0]
                    ids = res["ids"]
                    texts = res["texts"]
                    scores = res["scores"]
            else:
                res = tfidf_retrieve(docs, q, topk=topk)[0]
                ids = res["ids"]
                texts = res["texts"]
                scores = res["scores"]

            selected = []
            for i, (rid, txt, sc) in enumerate(
                zip(ids, texts, scores), start=1
            ):
                st.subheader(f"{i}. id={rid} score={sc}")
                st.write(txt)
                if use_auto:
                    checked = sc >= auto_threshold
                    st.checkbox("relevant", value=checked, key=f"{q}-{rid}")
                    if checked:
                        selected.append(rid)
                else:
                    chk = st.checkbox("relevant", key=f"{q}-{rid}")
                    if chk:
                        selected.append(rid)

            annotations.append(
                {"query": q, "retrieved": ids, "relevant": selected}
            )

        # save
        Path(out_file).write_text(
            json.dumps(annotations, indent=2), encoding="utf-8"
        )
        st.success(f"Saved annotations to {out_file}")

        # evaluate
        try:
            from evaluate import evaluate_batch

            results = [
                {"retrieved": a["retrieved"], "relevant": a["relevant"]}
                for a in annotations
            ]
            metrics = evaluate_batch(results, ks=[1, 3, 5])
            st.subheader("Evaluation metrics")
            st.json(metrics)
        except Exception:
            st.error("Failed to compute evaluation metrics")
            st.text(traceback.format_exc())


if __name__ == "__main__":
    run_app()
