import os
import sys

# ensure project root is on sys.path so `src` can be imported
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.vectordb import VectorDB


def main():
    # instantiate with large chunk size so each document stays as one chunk
    vdb = VectorDB(collection_name="test_eval", chunk_size=10000, chunk_overlap=0, chunk_strategy="simple")

    docs = [
        {"content": "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.", "metadata": {"topic": "ai"}},
        {"content": "Biotechnology uses living organisms, cells, and biological systems to develop products and technologies.", "metadata": {"topic": "biotech"}},
        {"content": "Climate science studies the Earth's climate and how human activities influence global temperatures.", "metadata": {"topic": "climate"}},
        {"content": "Quantum computing harnesses quantum mechanics to perform certain computations faster than classical computers.", "metadata": {"topic": "quantum"}},
    ]

    print("Adding documents...")
    vdb.add_documents(docs)

    queries = [
        {"query": "What is artificial intelligence?", "relevant_ids": ["doc_0_chunk_0"]},
        {"query": "Explain biotechnology applications.", "relevant_ids": ["doc_1_chunk_0"]},
        {"query": "Global warming and climate change causes", "relevant_ids": ["doc_2_chunk_0"]},
        {"query": "Quantum speedups and quantum algorithms", "relevant_ids": ["doc_3_chunk_0"]},
    ]

    print("Running evaluation...")
    metrics = vdb.evaluate_retrieval(queries, k=3)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
