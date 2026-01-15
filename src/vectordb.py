import os
<<<<<<< HEAD
import re
import chromadb
import logging
from typing import List, Dict, Any, Optional
import math
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None
=======
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

# Local diagram path (per your request)
DIAGRAM_LOCAL_PATH = "/mnt/data/A_flat-design_digital_illustration_diagram_illustr.png"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class VectorDBConfig:
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    chroma_db_path: str = "./chroma_db"
    chunk_size_tokens: int = 150  # approximate words per chunk (not strict tokens)
    overlap_tokens: int = 30
    batch_size: int = 64
    embedding_batch_size: int = 32
    use_sentence_split: bool = True
    persist: bool = True  # whether to persist local DB
    metadata_fields: List[str] = field(default_factory=lambda: ["source", "title", "section"])
    device: Optional[str] = None  # e.g., "cuda" or "cpu"
>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02


class VectorDB:
    """
    Enhanced VectorDB wrapper around ChromaDB + SentenceTransformer embeddings.

    Key features:
    - Sliding-window chunking with overlap to preserve context continuity.
    - Batching for embeddings and robust error handling.
    - Metadata-aware storage and filtering.
    - Query preprocessing hook, hybrid-ish retrieval (embedding + lexical fallback),
      and reranking by cosine similarity.
    - Simple retrieval evaluation helpers (Recall@k, MRR).
    """

<<<<<<< HEAD
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
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        # Chunking config
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50"))
        self.chunk_strategy = chunk_strategy or os.getenv("CHUNK_STRATEGY", "simple")
        # Reranking config
        self.use_tfidf_rerank = use_tfidf_rerank
        self.rerank_k = rerank_k
        # performance / batching
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))

        # logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
=======
    def __init__(self, config: VectorDBConfig = None):
        self.config = config or VectorDBConfig()
        logger.info(f"Initializing VectorDB with collection: {self.config.collection_name}")

        # Initialize ChromaDB client (persistent)
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.config.chroma_db_path)
        self.client = chromadb.Client(settings)

        # Load embedding model
        logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
        model_kwargs = {}
        if self.config.device:
            model_kwargs["device"] = self.config.device
        self.embedder = SentenceTransformer(self.config.embedding_model_name, **model_kwargs)
>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"description": "RAG document collection"},
        )
        logger.info("VectorDB initialized.")

<<<<<<< HEAD
        self.logger.info(f"Vector database initialized with collection: {self.collection_name}")

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _split_into_sentences(self, text: str) -> List[str]:
        # basic sentence splitter that keeps punctuation
        # splits on (?<=[.!?]) + space(s) or newline boundaries
        text = self._clean_text(text)
        sentences = re.split(r'(?<=[\.!?])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """Create chunks using one of several strategies.

        Strategies:
          - "simple": approximate char-based chunks (word aware)
          - "sentence": group full sentences up to chunk_size characters
          - "recursive": use LangChain's RecursiveCharacterTextSplitter when available

        Overlap is interpreted as number of characters (for simple) or number of sentences (for sentence strategy).
        """
        text = self._clean_text(text)
        cs = chunk_size or self.chunk_size
        ov = overlap or self.chunk_overlap

        if self.chunk_strategy == "recursive" and RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cs, chunk_overlap=ov, separators=["\n\n", "\n", " "]
            )
            return splitter.split_text(text)

        if self.chunk_strategy == "sentence":
            sentences = self._split_into_sentences(text)
            if not sentences:
                return []
            chunks = []
            i = 0
            # overlap measured in sentences
            while i < len(sentences):
                cur = sentences[i]
                j = i + 1
                while j < len(sentences) and len(" ".join(sentences[i:j+1])) <= cs:
                    j += 1
                chunk = " ".join(sentences[i:j])
                chunks.append(chunk)
                if j >= len(sentences):
                    break
                # move start by window size minus overlap (sentences)
                step = max(1, (j - i) - max(1, ov))
                i = i + step
            return chunks

        # default: simple char-aware word grouping with approximate overlap
        words = text.split()
        if not words:
            return []
        chunks = []
        cur_words = []
        cur_len = 0
        for w in words:
            # +1 for space
            if cur_len + len(w) + 1 > cs and cur_words:
                chunks.append(" ".join(cur_words))
                # compute overlap in words approximated from overlap chars
                overlap_chars = ov
                # approximate words to keep for overlap
                keep = 0
                accum = 0
                for t in reversed(cur_words):
                    accum += len(t) + 1
                    keep += 1
                    if accum >= overlap_chars:
                        break
                # start new window with last `keep` words
                cur_words = cur_words[-keep:] if keep > 0 else []
                cur_len = sum(len(t) + 1 for t in cur_words)
            cur_words.append(w)
            cur_len += len(w) + 1
        if cur_words:
            chunks.append(" ".join(cur_words))
=======
    # -----------------------------
    # Chunking utilities
    # -----------------------------
    def _sentence_split(self, text: str) -> List[str]:
        # Basic sentence splitter using punctuation. Replace with spaCy or nltk for better results.
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return sentences

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Chunk text using a sliding window over sentences or words.

        Args:
            text: full document string.
            chunk_size: approx number of words per chunk (defaults to config).
            overlap: approx number of words overlap (defaults to config).

        Returns:
            List[str]: text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size_tokens
        overlap = overlap if overlap is not None else self.config.overlap_tokens

        if self.config.use_sentence_split:
            units = self._sentence_split(text)
        else:
            units = text.split()

        # Convert units list into rolling windows measured in units length
        chunks = []
        if not units:
            return chunks

        # If units are sentences, approximate by word count per sentence
        if self.config.use_sentence_split:
            # Convert sentences to words counts to manage chunk sizes approximately
            sentence_word_counts = [len(s.split()) for s in units]
            start_idx = 0
            while start_idx < len(units):
                # accumulate sentences until reaching chunk_size approx
                total_words = 0
                end_idx = start_idx
                while end_idx < len(units) and total_words + sentence_word_counts[end_idx] <= chunk_size:
                    total_words += sentence_word_counts[end_idx]
                    end_idx += 1
                if end_idx == start_idx:  # if single sentence larger than chunk_size
                    end_idx = start_idx + 1
                chunk_text = " ".join(units[start_idx:end_idx])
                chunks.append(chunk_text)
                # move window
                # compute approximate overlap in sentences
                overlap_words = overlap
                # move start to keep `overlap_words` worth of words
                # naive approach: move back by sentences until overlap
                if end_idx >= len(units):
                    break
                # determine next start by backing off sentences
                sent_back = 0
                back_words = 0
                i = end_idx - 1
                while i >= 0 and back_words < overlap_words:
                    back_words += sentence_word_counts[i]
                    i -= 1
                    sent_back += 1
                start_idx = max(0, end_idx - sent_back)
        else:
            # word-level sliding window
            words = units
            start = 0
            while start < len(words):
                end = min(len(words), start + chunk_size)
                chunk = " ".join(words[start:end])
                chunks.append(chunk)
                if end == len(words):
                    break
                start = max(0, end - overlap)

>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02
        return chunks

    # -----------------------------
    # Document ingestion
    # -----------------------------
    def _make_doc_id(self, doc_idx: int, chunk_idx: int, prefix: Optional[str] = None) -> str:
        prefix_str = f"{prefix}_" if prefix else ""
        return f"{prefix_str}doc_{doc_idx}_chunk_{chunk_idx}"

    def add_documents(self, documents: List[Dict[str, Any]], prefix: Optional[str] = None, overwrite: bool = False) -> Dict[str, int]:
        """
        Ingest documents into ChromaDB.

<<<<<<< HEAD
        self.logger.info("Processing %d documents...", len(documents))
        all_chunks = []
=======
        documents: list of dicts:
            {
              "id": optional unique id,
              "text": "full document text",
              "metadata": {"source": ..., "title": ..., ...}
            }

        Returns: summary dict with counts added
        """
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty list of dicts with 'text' and optional 'metadata' keys")

        all_texts = []
>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02
        all_metadatas = []
        all_ids = []

        doc_counter = 0
        chunk_counter = 0

        for doc_idx, doc in enumerate(documents):
<<<<<<< HEAD
            # support dicts with content + metadata or raw strings
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
            self.logger.warning("No chunks to add.")
            return

        # Compute embeddings in batches and add in batches to ChromaDB to reduce memory usage
        total = len(all_chunks)
        self.logger.info("Encoding and adding %d chunks in batches (batch_size=%d)", total, self.batch_size)
        start = 0
        while start < total:
            end = min(start + self.batch_size, total)
            batch_chunks = all_chunks[start:end]
            batch_emb = self.embedding_model.encode(batch_chunks, show_progress_bar=False)
            batch_meta = all_metadatas[start:end]
            batch_ids = all_ids[start:end]
            try:
                self.collection.add(
                    embeddings=batch_emb,
                    documents=batch_chunks,
                    metadatas=batch_meta,
                    ids=batch_ids,
                )
            except Exception as e:
                self.logger.exception("Failed to add batch to collection: %s", e)
                raise
            start = end

        self.logger.info("Added %d chunks from %d documents to vector database", chunk_counter, doc_counter)

    def _normalize_query(self, query: str) -> str:
        q = query.lower()
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _expand_query(self, query: str, top_k: int = 5) -> str:
        # lightweight expansion: append top TF-IDF keywords from the full collection
        try:
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
            # score query terms against corpus to find salient terms
            tfidf_vocab = vectorizer.get_feature_names_out()
            # pick top terms globally by idf
            idf_scores = vectorizer.idf_
            top_idx = np.argsort(idf_scores)[-top_k:]
            top_terms = [tfidf_vocab[i] for i in top_idx]
            return query + " " + " ".join(top_terms)
        except Exception:
            return query

    def search(self, query: str, n_results: int = 5, metadata_filter: Optional[Dict[str,Any]] = None, expand_query: bool = False) -> Dict[str, Any]:
=======
            text = doc.get("text") or doc.get("content")
            if not text or not text.strip():
                logger.warning(f"Skipping empty document at index {doc_idx}")
                continue
            metadata = doc.get("metadata", {}) or {}
            # Add also provided id if present to metadata
            doc_source_id = doc.get("id") or metadata.get("source") or f"doc_{doc_idx}"
            chunks = self.chunk_text(text, chunk_size=self.config.chunk_size_tokens, overlap=self.config.overlap_tokens)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = dict(metadata)  # copy
                chunk_meta.update({
                    "source": doc_source_id,
                    "orig_doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "text_excerpt": chunk[:200]
                })
                all_texts.append(chunk)
                all_metadatas.append(chunk_meta)
                all_ids.append(self._make_doc_id(doc_idx, chunk_idx, prefix))
                chunk_counter += 1
            doc_counter += 1

        if not all_texts:
            logger.info("No chunks to add.")
            return {"documents": doc_counter, "chunks": 0}

        # compute embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(all_texts), self.config.embedding_batch_size), desc="Embedding batches"):
            batch_texts = all_texts[i:i + self.config.embedding_batch_size]
            try:
                batch_embs = self.embedder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            except Exception as e:
                logger.exception(f"Embedding error on batch starting at {i}: {e}")
                # fallback: try per-item encoding to isolate problematic text
                batch_embs = []
                for t in batch_texts:
                    try:
                        emb = self.embedder.encode(t, convert_to_numpy=True)
                        batch_embs.append(emb)
                    except Exception as e2:
                        logger.warning(f"Skipping chunk due to embedding failure: {e2}")
                        batch_embs.append(np.zeros((self.embedder.get_sentence_embedding_dimension(),), dtype=float))
                batch_embs = np.vstack(batch_embs)
            embeddings.append(batch_embs)

        embeddings = np.vstack(embeddings)

        # Optionally overwrite collection entries by ids
        if overwrite:
            # remove existing ids first to avoid duplicates
            try:
                existing_ids = [iid for iid in all_ids if self.collection.get(ids=[iid]).get("ids")]
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
            except Exception:
                # Some versions of chroma may throw; ignore silently
                pass

        # Add to collection in batches (Chroma likes moderate sizes)
        add_batch_size = max(1, min(self.config.batch_size, len(all_texts)))
        for i in range(0, len(all_texts), add_batch_size):
            i_end = i + add_batch_size
            docs_batch = all_texts[i:i_end]
            ids_batch = all_ids[i:i_end]
            mets_batch = all_metadatas[i:i_end]
            embs_batch = embeddings[i:i_end].tolist()
            self.collection.add(documents=docs_batch, embeddings=embs_batch, metadatas=mets_batch, ids=ids_batch)

        if self.config.persist:
            try:
                self.client.persist()
            except Exception:
                # Some clients persist automatically; ignore if not supported
                pass

        logger.info(f"Added {chunk_counter} chunks from {doc_counter} documents.")
        return {"documents": doc_counter, "chunks": chunk_counter}

    # -----------------------------
    # Query preprocessing (hook)
    # -----------------------------
    def preprocess_query(self, query: str) -> str:
>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02
        """
        Hook to rewrite or expand queries. For high-quality systems you can call
        an LLM to rewrite vague queries into precise, retrieval-optimized queries.
        For now this is a no-op (returns query as-is) but kept as a single place to
        extend with LLM-based rewrites or synonym expansion.
        """
        # Example: apply simple normalization
        q = query.strip()
        # TODO: plug in LLM-based rewriter if desired
        return q

    # -----------------------------
    # Reranking
    # -----------------------------
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _rerank(self, query_embedding: np.ndarray, candidate_texts: List[str]) -> List[Tuple[int, float]]:
        """
        Recompute similarity between query embedding and candidate texts by
        encoding candidate texts and computing cosine similarity.
        Returns list of tuples (candidate_index, score) sorted descending by score.
        """
        if not candidate_texts:
            return []
        try:
            cand_embs = self.embedder.encode(candidate_texts, convert_to_numpy=True)
        except Exception as e:
            logger.warning(f"Failed to embed candidates for reranking: {e}")
            return []

        scores = [self._cosine_sim(query_embedding, emb) for emb in cand_embs]
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores

    # -----------------------------
    # Search
    # -----------------------------
    def search(self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None, rerank: bool = True) -> Dict[str, Any]:
        """
        Search for similar document chunks.

        Args:
            query: user query string.
            n_results: number of final results to return.
            where: optional metadata filter dict for ChromaDB.
            rerank: whether to rerank results using fresh cosine similarities.

        Returns:
            dict with keys: documents, metadatas, distances, ids, scores
        """
<<<<<<< HEAD
        qnorm = self._normalize_query(query)
        if expand_query:
            qnorm = self._expand_query(qnorm)

        query_embedding = self.embedding_model.encode([qnorm])[0]
        candidate_k = max(n_results, self.rerank_k if self.use_tfidf_rerank else n_results)

        # Query ChromaDB for a larger candidate set if reranking
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"]
        )
        # Handle empty results
        if not results or not results.get("documents"):
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0] if "ids" in results else []

        # apply metadata filtering if requested (exact match of provided keys)
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

        # Optional TF-IDF reranking for better lexical matching
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
                # if TF-IDF rerank fails, fall back to original ordering
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
        """Evaluate retrieval performance on a set of queries.

        Each query dict must contain:
          - 'query': the query string
          - 'relevant_ids': list of ground-truth chunk ids

        Returns mean metrics: Precision@k, Recall@k, MRR, NDCG@k
        """
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []

        for q in queries:
            query_text = q.get("query")
            relevant = set(q.get("relevant_ids", []))
            if not query_text or not relevant:
                continue
            res = self.search(query_text, n_results=k, expand_query=expand_query)
            retrieved_ids = res.get("ids", [])

            # Precision@k
            hit_set = [1 if rid in relevant else 0 for rid in retrieved_ids]
            precision = sum(hit_set) / float(k)
            recall = sum(hit_set) / float(len(relevant)) if relevant else 0.0

            # MRR
            rr = 0.0
            for idx, val in enumerate(hit_set, start=1):
                if val:
                    rr = 1.0 / idx
                    break

            # DCG / IDCG for NDCG@k
            dcg = 0.0
            for i, rel in enumerate(hit_set, start=1):
                dcg += (2 ** rel - 1) / math.log2(i + 1)
            ideal_rels = [1] * min(len(relevant), k)
            idcg = 0.0
            for i, rel in enumerate(ideal_rels, start=1):
                idcg += (2 ** rel - 1) / math.log2(i + 1)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            mrrs.append(rr)
            ndcgs.append(ndcg)

        # aggregate
        def _mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        return {
            "precision@k": _mean(precisions),
            "recall@k": _mean(recalls),
            "mrr": _mean(mrrs),
            "ndcg@k": _mean(ndcgs),
        }
=======
        q = self.preprocess_query(query)
        query_embedding = self.embedder.encode([q], convert_to_numpy=True)[0]

        # initial vector search (request slightly larger set for reranking)
        initial_k = min(max(n_results * 3, n_results + 5), 100)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=initial_k,
                where=where,
                include=["documents", "metadatas", "distances", "ids"]
            )
        except Exception as e:
            logger.exception(f"ChromaDB query failed: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": [], "scores": []}

        # unpack Chroma results (Chroma returns nested lists)
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0] if "ids" in results else []

        # If rerank is enabled, recompute similarity and sort
        if rerank and docs:
            rerank_scores = self._rerank(query_embedding, docs)
            # pick top n_results after rerank
            top = rerank_scores[:n_results]
            indices = [idx for idx, _ in top]
            scores = [score for _, score in top]
            docs = [docs[i] for i in indices]
            metadatas = [metadatas[i] for i in indices]
            distances = [distances[i] for i in indices] if distances else [None] * len(indices)
            ids = [ids[i] for i in indices] if ids else [None] * len(indices)
            return {"documents": docs, "metadatas": metadatas, "distances": distances, "ids": ids, "scores": scores}
        else:
            # truncate to requested n_results if needed
            docs = docs[:n_results]
            metadatas = metadatas[:n_results]
            distances = distances[:n_results] if distances else []
            ids = ids[:n_results] if ids else []
            scores = [None] * len(docs)
            return {"documents": docs, "metadatas": metadatas, "distances": distances, "ids": ids, "scores": scores}

    # -----------------------------
    # Retrieval Evaluation
    # -----------------------------
    def evaluate_retrieval(self, test_queries: List[str], expected_doc_ids: List[str], k: int = 5) -> Dict[str, float]:
        """
        Very simple evaluator: computes Recall@k and MRR over provided test set.
        expected_doc_ids should contain a single expected doc id per query (or list of acceptable ids).
        """
        if len(test_queries) != len(expected_doc_ids):
            raise ValueError("test_queries and expected_doc_ids must be same length")

        hits_at_k = 0
        reciprocal_ranks = []
        for q, expected in zip(test_queries, expected_doc_ids):
            res = self.search(q, n_results=k, rerank=True)
            ids = res.get("ids", [])
            # Normalize expected to list
            expected_list = expected if isinstance(expected, list) else [expected]
            # compute recall@k
            if any(e in ids for e in expected_list):
                hits_at_k += 1
            # compute reciprocal rank
            rank = None
            for i, rid in enumerate(ids):
                if rid in expected_list:
                    rank = i + 1
                    break
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        recall_at_k = hits_at_k / len(test_queries)
        mrr = float(sum(reciprocal_ranks) / len(reciprocal_ranks))
        return {"recall@k": recall_at_k, "mrr": mrr}

    # -----------------------------
    # Utility helpers
    # -----------------------------
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return simple stats about collection.
        """
        try:
            count = self.collection.count()
        except Exception:
            # fallback rough estimate
            count = len(self.collection.get(include=["ids"]).get("ids", []))
        return {"collection_name": self.config.collection_name, "count": count}

    def delete_collection(self):
        """
        Deletes all items in the collection (use with caution).
        """
        try:
            ids = self.collection.get(include=["ids"]).get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.exception(f"Failed to delete collection: {e}")
            return False


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    cfg = VectorDBConfig()
    vdb = VectorDB(cfg)

    # simple docs
    docs = [
        {"id": "intro", "text": "This is the introduction to the RAG project. It explains goals and scope.", "metadata": {"title": "Intro", "source": "manuscript"}},
        {"id": "deep", "text": "Advanced retrieval details: chunking, overlap, reranking and evaluation are essential.", "metadata": {"title": "Advanced", "source": "manual"}},
    ]
    vdb.add_documents(docs)

    q = "What should I consider for chunking and evaluation?"
    res = vdb.search(q, n_results=3)
    print("Search results:", res)

    # Evaluate with a toy test
    test_qs = ["chunking strategies", "evaluation metrics"]
    expected = ["intro", "deep"]
    print("Eval:", vdb.evaluate_retrieval(test_qs, expected, k=3))
    print("Diagram path (local):", DIAGRAM_LOCAL_PATH)
>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02
