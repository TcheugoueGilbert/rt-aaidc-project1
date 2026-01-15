import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
  
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    results.append(text)
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()


        # Focus the assistant's role/scope — change this to suit your target use-case
        scope_description = os.getenv(
            "RAG_SCOPE",
            "You are a research assistant specialized in scientific and technical literature. Answer concisely, cite sources, and prefer exact excerpts from the provided context when possible.",
        )

        # Create RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            f"System: {scope_description}\n\n"
            "Use only the provided context to answer the question. If the answer is not contained in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

<<<<<<< HEAD
=======
    

>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02
        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        logger.info("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info("Using OpenAI model: %s", model_name)
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info("Using Groq model: %s", model_name)
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            logger.info("Using Google Gemini model: %s", model_name)
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
<<<<<<< HEAD
        # Retrieve candidates from the vector DB
        results = self.vector_db.search(input, n_results)
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        dists = results.get("distances", [])
        ids = results.get("ids", [])

        # Build a concise context that includes provenance for each chunk
        context_parts = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else None
            _id = ids[i] if i < len(ids) else None
            provenance = f"[source: doc={meta.get('doc_index', '?')}, chunk={meta.get('chunk_index', '?')}, id={_id}, dist={dist}]"
            context_parts.append(f"{provenance}\n{doc}")

        context = "\n\n".join(context_parts)

=======
        llm_answer = ""
        
        context_chunks = self.vector_db.search(input, n_results)
        context = "\n\n".join(context_chunks)
>>>>>>> ecd9bfc13ced73e07025e5b59a16bbbb23293c02
        # Generate answer using the chain
        llm_answer = self.chain.invoke({"context": context, "question": input})
        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        logger.info("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        logger.info("Loading documents...")
        sample_docs = load_documents()
        logger.info("Loaded %d sample documents", len(sample_docs))

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                logger.info(result)

    except Exception as e:
        logger.exception("Error running RAG assistant: %s", e)
        logger.error("Make sure you have set up your .env file with at least one API key:")
        logger.error("- OPENAI_API_KEY (OpenAI GPT models)")
        logger.error("- GROQ_API_KEY (Groq Llama models)")
        logger.error("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
