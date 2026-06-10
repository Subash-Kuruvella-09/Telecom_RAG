"""
Builds the RAG chain with:
  - merged retriever with similarity scores
  - confidence filtering (fallback if no documents score above threshold)
  - source citations in the final answer
  - prompt → Qwen3-32B on Groq → string output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever_with_scores

# Minimum similarity score threshold for retrieval confidence (0-1 scale)
SIMILARITY_THRESHOLD = 0.5

FALLBACK_RESPONSE = """I don't have reliable information to answer that question accurately.

For immediate assistance, please:
• Call **611** from your mobile (free)
• Use the **MyTelecom app** for live chat (available 8am-10pm)
• Email **support@telecom.example.com**

Average wait time for calls is under 3 minutes."""

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from three sources:
- **FAQ** entries: General policy and how-to information
- **Ticket** records: Real resolved cases with step-by-step solutions  
- **Guide** pages: Detailed reference materials

IMPORTANT: Every statement in your answer MUST be backed by the context provided.
Cite your sources explicitly using this format:
- For FAQ: [FAQ #N] or [FAQ - Category Name]
- For Tickets: [Ticket #TICKETID] 
- For Guides: [Guide - Page N] or [Guide - Section Name]

Example: "You can check your data balance by dialing *123# [FAQ #1] or through the MyTelecom app [FAQ #1]."

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _extract_source_info(doc: Document) -> str:
    """Extract human-readable source information from document metadata."""
    metadata = doc.metadata
    source = metadata.get("source", "unknown").lower()

    if source == "faq":
        faq_id = metadata.get("faq_id", "?")
        category = metadata.get("category", "")
        if category:
            return f"[FAQ #{faq_id} - {category.title()}]"
        return f"[FAQ #{faq_id}]"
    elif source == "ticket":
        ticket_id = metadata.get("ticket_id", "?")
        return f"[Ticket #{ticket_id}]"
    elif source == "guide":
        page = metadata.get("page_number", "?")
        section = metadata.get("section_title", "Guide")
        return f"[Guide - {section} (p.{page})]"
    else:
        return "[Reference]"


def _format_docs(docs_with_scores: list[tuple[Document, float]]) -> str:
    """Format documents with source citations and similarity scores."""
    if not docs_with_scores:
        return ""

    sections = []
    for doc, score in docs_with_scores:
        source_info = _extract_source_info(doc)
        # Include similarity score as confidence indicator (debug info)
        sections.append(
            f"{source_info} (confidence: {score:.2f})\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


def _retrieve_with_confidence(retriever_fn):
    """
    Wraps the retriever to apply confidence filtering.
    Returns either formatted docs (if above threshold) or triggers fallback.
    """
    def retrieve_and_filter(query: str):
        # Get documents with scores
        docs_with_scores = retriever_fn(query)

        if not docs_with_scores:
            # No documents retrieved at all
            return None

        # Filter by threshold - keep only documents above threshold
        filtered_docs = [
            (doc, score) for doc, score in docs_with_scores
            if score >= SIMILARITY_THRESHOLD
        ]

        if not filtered_docs:
            # No documents meet confidence threshold
            return None

        # Return formatted docs with citations
        return _format_docs(filtered_docs)

    return retrieve_and_filter


def build_chain():
    """Build the RAG chain with confidence filtering and source citations."""
    retriever_fn = build_retriever_with_scores(
        k_faq=3, k_tickets=3, k_guides=3)

    # Wrap with confidence filtering
    retrieve_with_confidence = _retrieve_with_confidence(retriever_fn)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    def chain_with_fallback(input_data):
        """Execute chain with fallback for low-confidence queries."""
        question = input_data["question"] if isinstance(
            input_data, dict) else input_data

        # Attempt retrieval with confidence filtering
        context = retrieve_with_confidence(question)

        # If no confident retrieval, return fallback
        if context is None:
            return FALLBACK_RESPONSE

        # Otherwise, invoke the LLM chain
        chain = (
            {"context": RunnablePassthrough.map(
                lambda x: x), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Create input dict for chain
        full_input = {"context": context, "question": question}
        return chain.invoke(full_input)

    return RunnableLambda(chain_with_fallback)
