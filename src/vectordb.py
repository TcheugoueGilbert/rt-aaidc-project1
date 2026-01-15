import os
import re
import logging
from typing import List, Dict, Any, Optional
import math
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class VectorDB:
    """A focused, clean VectorDB implementation for this project."""

    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = None,
        use_tfidf_rerank: bool = True,
        rerank_k: int = 50,
    ):
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50"))
        self.chunk_strategy = chunk_strategy or os.getenv("CHUNK_STRATEGY", "simple")
        self.use_tfidf_rerank = use_tfidf_rerank
        self.rerank_k = rerank_k
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))

        if chromadb is not None:
            try:
                self.client = chromadb.PersistentClient(path="./chroma_db")
                self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description": "RAG document collection"})
            except Exception:
                self.client = None
                self.collection = None
        else:
            self.client = None
            self.collection = None

        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("Vector database initialized with collection: %s", self.collection_name)

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _split_into_sentences(self, text: str) -> List[str]:
        text = self._clean_text(text)
        sentences = re.split(r'(?<=[\.\!?])\s+|\n+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        text = self._clean_text(text)
        cs = chunk_size or self.chunk_size
        ov = overlap or self.chunk_overlap

        if self.chunk_strategy == "recursive" and RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(chunk_size=cs, chunk_overlap=ov, separators=["\n\n", "\n", " "])
            return splitter.split_text(text)

        if self.chunk_strategy == "sentence":
            sentences = self._split_into_sentences(text)
            if not sentences:
                return []
            chunks = []
            i = 0
            while i < len(sentences):
                j = i + 1
                while j < len(sentences) and len(" ".join(sentences[i:j+1])) <= cs:
                    j += 1
                chunk = " ".join(sentences[i:j])
                chunks.append(chunk)
                if j >= len(sentences):
                    break
                step = max(1, (j - i) - max(1, ov))
                i = i + step
            return chunks

        words = text.split()
        if not words:
            return []
        chunks = []
        cur_words = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > cs and cur_words:
                chunks.append(" ".join(cur_words))
                overlap_chars = ov
                keep = 0
                accum = 0
                for t in reversed(cur_words):
                    accum += len(t) + 1
                    keep += 1
                    if accum >= overlap_chars:
                        break
                cur_words = cur_words[-keep:] if keep > 0 else []
                cur_len = sum(len(t) + 1 for t in cur_words)
            cur_words.append(w)
            cur_len += len(w) + 1
        if cur_words:
            chunks.append(" ".join(cur_words))
        return chunks

    def add_documents(self, documents: List) -> None:
        logger.info("Processing %d documents...", len(documents))
        all_chunks = []
        all_metadatas = []
        all_ids = []
        doc_counter = 0
        chunk_counter = 0

        for doc_idx, doc in enumerate(documents):
            if isinstance(doc, dict):
                content = doc.get("content") or doc.get("text") or ""
                meta = doc.get("metadata") or {}
            else:
                content = str(doc)
                meta = {}

            chunks = self.chunk_text(content)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                m = {"doc_index": doc_idx, "chunk_index": chunk_idx}
                m.update(meta)
                all_metadatas.append(m)
                all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
                chunk_counter += 1
            doc_counter += 1

        if not all_chunks:
            logger.warning("No chunks to add.")
            return

        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

        if self.collection is not None:
            self.collection.add(
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids,
            )
            logger.info("Added %d chunks from %d documents to vector database", chunk_counter, doc_counter)
        else:
            logger.warning("ChromaDB collection not initialized; skipping add.")

    def _normalize_query(self, query: str) -> str:
        q = query.lower()
        q = re.sub(r"[^\\w\\s]", " ", q)
        q = re.sub(r"\\s+", " ", q).strip()
        return q

    def _expand_query(self, query: str, top_k: int = 5) -> str:
        try:
            if self.collection is None:
                return query
            data = self.collection.get(include=["documents"]) or {}
            docs = data.get("documents", [])
            flat = []
            for d in docs:
                if isinstance(d, list):
                    flat.extend(d)
                else:
                    flat.append(d)
            if not flat:
                return query
            vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
            X = vectorizer.fit_transform(flat)
            tfidf_vocab = vectorizer.get_feature_names_out()
            idf_scores = vectorizer.idf_
            top_idx = np.argsort(idf_scores)[-top_k:]
            top_terms = [tfidf_vocab[i] for i in top_idx]
            return query + " " + " ".join(top_terms)
        except Exception:
            return query

    def search(self, query: str, n_results: int = 5, metadata_filter: Optional[Dict[str,Any]] = None, expand_query: bool = False) -> Dict[str, Any]:
        qnorm = self._normalize_query(query)
        if expand_query:
            qnorm = self._expand_query(qnorm)

        query_embedding = self.embedding_model.encode([qnorm])[0]
        candidate_k = max(n_results, self.rerank_k if self.use_tfidf_rerank else n_results)

        if self.collection is None:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results or not results.get("documents"):
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0] if "ids" in results else []

        if metadata_filter:
            filtered_docs, filtered_metas, filtered_dists, filtered_ids = [], [], [], []
            for doc, meta, dist, id_ in zip(docs, metas, dists, ids):
                ok = True
                if isinstance(meta, dict):
                    for k, v in metadata_filter.items():
                        if meta.get(k) != v:
                            ok = False
                            break
                if ok:
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
                    filtered_dists.append(dist)
                    filtered_ids.append(id_)
            docs, metas, dists, ids = filtered_docs, filtered_metas, filtered_dists, filtered_ids

        if self.use_tfidf_rerank and docs:
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_docs = vectorizer.fit_transform(docs)
                tfidf_query = vectorizer.transform([query])
                sims = cosine_similarity(tfidf_query, tfidf_docs)[0]
                order = sims.argsort()[::-1]
                top_idx = order[:n_results]
                docs = [docs[i] for i in top_idx]
                metas = [metas[i] for i in top_idx]
                dists = [dists[i] for i in top_idx]
                ids = [ids[i] for i in top_idx]
            except Exception:
                docs = docs[:n_results]
                metas = metas[:n_results]
                dists = dists[:n_results]
                ids = ids[:n_results]
        else:
            docs = docs[:n_results]
            metas = metas[:n_results]
            dists = dists[:n_results]
            ids = ids[:n_results]

        return {"documents": docs, "metadatas": metas, "distances": dists, "ids": ids}

    def evaluate_retrieval(self, queries: List[Dict[str, Any]], k: int = 10, expand_query: bool = False) -> Dict[str, float]:
        precisions, recalls, mrrs, ndcgs = [], [], [], []
        for q in queries:
            query_text = q.get("query")
            relevant = set(q.get("relevant_ids", []))
            if not query_text or not relevant:
                continue
            res = self.search(query_text, n_results=k, expand_query=expand_query)
            retrieved_ids = res.get("ids", [])
            hit_set = [1 if rid in relevant else 0 for rid in retrieved_ids]
            precision = sum(hit_set) / float(k)
            recall = sum(hit_set) / float(len(relevant)) if relevant else 0.0
            rr = 0.0
            for idx, val in enumerate(hit_set, start=1):
                if val:
                    rr = 1.0 / idx
                    break
            dcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(hit_set, start=1))
            ideal_rels = [1] * min(len(relevant), k)
            idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(rr)
            ndcgs.append(ndcg)

        def _mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        return {"precision@k": _mean(precisions), "recall@k": _mean(recalls), "mrr": _mean(mrrs), "ndcg@k": _mean(ndcgs)}
