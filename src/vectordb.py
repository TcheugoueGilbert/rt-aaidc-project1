import os
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


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

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

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

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
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        self.logger.info("Processing %d documents...", len(documents))
        all_chunks = []
        all_metadatas = []
        all_ids = []
        doc_counter = 0
        chunk_counter = 0

        for doc_idx, doc in enumerate(documents):
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
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
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