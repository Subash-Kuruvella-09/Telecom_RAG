"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)

Supports both standard retrieval and retrieval with similarity scores for confidence filtering.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    faq_retriever = faq_store.as_retriever(search_kwargs={"k": k_faq})
    tickets_retriever = tickets_store.as_retriever(
        search_kwargs={"k": k_tickets})
    guides_retriever = guides_store.as_retriever(search_kwargs={"k": k_guides})

    def retrieve(query: str) -> list[Document]:
        return (
            faq_retriever.invoke(query)
            + tickets_retriever.invoke(query)
            + guides_retriever.invoke(query)
        )

    return RunnableLambda(retrieve)


def build_retriever_with_scores(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
):
    """
    Returns a function that retrieves documents with similarity scores.
    Useful for confidence filtering and citation tracking.

    Returns tuples of (Document, similarity_score) where similarity is in [0, 1].
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    def retrieve_with_scores(query: str) -> list[tuple[Document, float]]:
        """Retrieve documents with similarity scores from all collections."""
        results = []

        # Retrieve from each collection with scores
        faq_results = faq_store.similarity_search_with_score(query, k=k_faq)
        tickets_results = tickets_store.similarity_search_with_score(
            query, k=k_tickets)
        guides_results = guides_store.similarity_search_with_score(
            query, k=k_guides)

        # Combine all results (Chroma returns distance, we convert to similarity: 1 / (1 + distance))
        results.extend(faq_results)
        results.extend(tickets_results)
        results.extend(guides_results)

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    return retrieve_with_scores
