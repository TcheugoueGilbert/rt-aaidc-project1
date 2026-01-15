import builtins
import types
from src.vectordb import VectorDB


class DummyCollection:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.distances = []

    def add(self, embeddings, documents, metadatas, ids):
        # store documents as they are added
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        # distances not known at add time

    def get(self, include=None):
        # return stored documents in the shape VectorDB expects
        return {"documents": [self.documents]}

    def query(self, query_embeddings, n_results, include=None):
        # return top-n stored documents as candidates
        docs = self.documents[:n_results]
        metas = self.metadatas[:n_results]
        dists = [0.0 for _ in docs]
        ids = self.ids[:n_results]
        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [ids]}


class DummyEmbedder:
    def encode(self, inputs, show_progress_bar=False):
        # deterministic embedding (length of text) to avoid heavy models
        if isinstance(inputs, list):
            return [[len(str(x)) * 0.01] for x in inputs]
        return [[len(str(inputs)) * 0.01]]


def test_chunk_sentence_strategy():
    vdb = VectorDB(chunk_size=50, chunk_overlap=1, chunk_strategy="sentence")
    paragraph = (
        "This is a sentence. This is another sentence. Short one. "
        "Final sentence to wrap up the paragraph."
    )
    chunks = vdb.chunk_text(paragraph, chunk_size=50)
    assert isinstance(chunks, list)
    assert all(len(c) <= 50 or '\n' in c for c in chunks)


def test_add_and_search_with_stubs():
    vdb = VectorDB(chunk_size=1000, chunk_overlap=0, chunk_strategy="simple")
    # replace heavy external dependencies with stubs
    vdb.collection = DummyCollection()
    vdb.embedding_model = DummyEmbedder()

    docs = [
        {"content": "Alpha document about AI.", "metadata": {"topic": "ai"}},
        {"content": "Beta document about biotech.", "metadata": {"topic": "biotech"}},
    ]

    vdb.add_documents(docs)

    # search should use stubbed collection and embedder
    res = vdb.search("artificial intelligence", n_results=2)
    assert "documents" in res
    assert len(res["documents"]) <= 2

    # evaluate_retrieval should compute metrics without external calls
    queries = [{"query": "What is AI?", "relevant_ids": ["doc_0_chunk_0"]}]
    metrics = vdb.evaluate_retrieval(queries, k=2)
    assert "precision@k" in metrics or "precision@1" in metrics
