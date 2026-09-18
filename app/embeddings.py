import sys
from typing import Literal

from langchain_classic.prompts import ChatPromptTemplate
from langchain_protocol import TypedDict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import  GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph 
from chunking import chunk_documents
from extracting import extract_pdfs_from_folder
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class RAGState(TypedDict):

    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int




def create_sample_vectorstore() -> Chroma:
    """Create a sample vectorstore for testing."""
    # Create a sample vectorstore with some documents
    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
    )

    extracted_docs = extract_pdfs_from_folder("docs")
    chunked_docs = chunk_documents(extracted_docs)
    documents = [Document(page_content=chunk["content"], metadata=chunk["metadata"]) for chunk in chunked_docs]

    vectorstore = Chroma.from_documents(documents, embedding=embedding,collection_name="rag_vectorstore")
    return vectorstore


def retrieve_documents(state: RAGState) -> dict:
    """
    Retrieve documents based on the query.
    Uses rewritten_query if available, otherwise original query.
    """
    query = state.get("rewritten_query") or state["query"]

    print(f"\n[RETRIEVE] Searching for: '{query}'")

    vectorstore = state.get("_vectorstore")  # Injected at runtime
    if not vectorstore:
        # Fallback - create new (in production, pass via config)
        
        vectorstore = create_sample_vectorstore()

    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 3})
    documents = retriever.invoke(query)

    print(f"[RETRIEVE] Found {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(
            f"  {i}. {doc.metadata.get('source', 'unknown')}: {doc.page_content[:50]}"
        )

    return {"documents": documents}


def grade_documents(state: RAGState) -> dict:
    """
    Grade retrieved documents for relevance to the query.
    This is the KEY difference from traditional RAG - we evaluate before generating.
    """
    query = state["query"]
    documents = state["documents"]

    print(f"\n[GRADE] Evaluating {len(documents)} documents for relevance...")

    llm = GoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a relevance grader. Given a user query and a document,
determine if the document contains information relevant to answering the query.

Output ONLY a number between 0 and 1:
- 1.0 = Highly relevant, directly answers the query
- 0.7 = Somewhat relevant, contains related information
- 0.3 = Marginally relevant, tangentially related
- 0.0 = Not relevant at all

Output ONLY the number, nothing else.""",
            ),
            (
                "human",
                """Query: {query}

Document: {document}

Relevance score (0-1):""",
            ),
        ]
    )

    # Grade each document and calculate average
    scores = []
    relevant_docs = []

    for doc in documents:
        chain = grading_prompt | llm
        result = chain.invoke({"query": query, "document": doc.page_content})

        try:
            score = float(result)
        except ValueError:
            score = 0.5  # Default if parsing fails

        scores.append(score)
        print(f"  - {doc.metadata.get('source', 'unknown')}: {score:.2f}")

        if score >= 0.5:  # Keep documents with score >= 0.5
            relevant_docs.append(doc)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"[GRADE] Average relevance: {avg_score:.2f}")
    print(f"[GRADE] Keeping {len(relevant_docs)}/{len(documents)} documents")

    return {"documents": relevant_docs, "relevance_score": avg_score}

def rewrite_query(state: RAGState) -> dict:
    """
    Rewrite the query to improve retrieval.
    Called when initial retrieval doesn't find relevant documents.
    """
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[REWRITE] Attempt {retry_count + 1}: Improving query...")

    llm = GoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a query rewriter for a RAG system.
The original query didn't retrieve relevant documents.

Rewrite the query to be more specific and likely to match relevant documents.
Consider:
- Adding synonyms or related terms
- Being more specific about what information is needed
- Rephrasing to match how documentation is typically written

Output ONLY the rewritten query, nothing else.""",
            ),
            (
                "human",
                """Original query: {query}

Rewritten query:""",
            ),
        ]
    )

    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    rewritten = result

    print(f"[REWRITE] Original: '{query}'")
    print(f"[REWRITE] Rewritten: '{rewritten}'")

    return {"rewritten_query": rewritten, "retry_count": retry_count + 1}


def generate_answer(state: RAGState) -> dict:
    """
    Generate the final answer using retrieved documents.
    """
    query = state["query"]
    documents = state["documents"]

    print(f"\n[GENERATE] Creating answer from {len(documents)} documents...")

    llm = GoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    # Format documents
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in documents
        ]
    )

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant answering questions based on provided context.

Use ONLY the information in the context to answer. If the context doesn't contain
enough information, say so clearly.

Always cite your sources by mentioning which document the information came from.""",
            ),
            (
                "human",
                """Context:
{context}

Question: {query}

Answer:""",
            ),
        ]
    )

    chain = generate_prompt | llm
    result = chain.invoke({"context": context, "query": query})

    print(f"[GENERATE] Answer generated")

    return {"generation": result}




def generate_fallback(state: RAGState) -> dict:
    """
    Generate a fallback response when retrieval fails after all retries.
    """
    query = state["query"]

    print(f"\n[FALLBACK] Retrieval failed after {state.get('retry_count', 0)} attempts")

    fallback_message = f"""I couldn't find relevant information to answer your question: "{query}"

This could mean:
1. The information isn't in my knowledge base
2. Try rephrasing your question with different terms
3. The topic might not be covered in the available documents

Would you like to try a different question?"""

    return {"generation": fallback_message}


def should_retry_or_generate(
    state: RAGState,
) -> Literal["rewrite", "generate", "fallback"]:
    """
    Decide whether to retry retrieval or proceed to generation.

    This is the BRAIN of agentic RAG - making decisions based on retrieval quality.
    """
    relevance_score = state.get("relevance_score", 0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    documents = state.get("documents", [])

    print(
        f"\n[ROUTER] Evaluating: score={relevance_score:.2f}, retries={retry_count}/{max_retries}, docs={len(documents)}"
    )

    # If we have relevant documents, generate
    if relevance_score >= 0.5 and len(documents) > 0:
        print("[ROUTER] -> GENERATE (good relevance)")
        return "generate"

    # If we can retry, rewrite query
    if retry_count < max_retries:
        print("[ROUTER] -> REWRITE (low relevance, retrying)")
        return "rewrite"

    # Out of retries
    if len(documents) > 0:
        print("[ROUTER] -> GENERATE (out of retries, using available docs)")
        return "generate"
    else:
        print("[ROUTER] -> FALLBACK (no relevant documents)")
        return "fallback"





def build_agentic_rag_graph():
    """
    Build the LangGraph workflow for agentic RAG.

    Flow:
    1. retrieve -> grade -> [decision]
    2. If low relevance and retries left: rewrite -> retrieve (loop)
    3. If good relevance or out of retries: generate
    4. If no documents at all: fallback
    """

    # Create the graph with our state schema
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("fallback", generate_fallback)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges
    workflow.add_edge("retrieve", "grade")

    # Conditional edge from grade
    workflow.add_conditional_edges(
        "grade",
        should_retry_or_generate,
        {"rewrite": "rewrite", "generate": "generate", "fallback": "fallback"},
    )

    # After rewrite, go back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Terminal nodes
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # Compile the graph
    app = workflow.compile()

    return app


def run_demo():
    """Run the agentic RAG demo."""

    print("=" * 60)
    print("AGENTIC RAG DEMO")
    print("=" * 60)

    # Create vector store
    print("\nSetting up vector store...")
    vectorstore = create_sample_vectorstore()

    # Build the graph
    print("Building agentic RAG graph...")
    app = build_agentic_rag_graph()

    # Test queries
    test_queries = [
        "elt",  # Should find relevant docs
        
        
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        # Run the graph
        initial_state = {
            "query": query,
            "rewritten_query": "",
            "documents": [],
            "generation": "",
            "relevance_score": 0.0,
            "retry_count": 0,
            "max_retries": 2,
            "_vectorstore": vectorstore,  # Pass vectorstore via state
        }

        result = app.invoke(initial_state)

        print("\n" + "-" * 60)
        print("FINAL ANSWER:")
        print("-" * 60)
        print(result["generation"])

    # Cleanup
    vectorstore.delete_collection()

run_demo()