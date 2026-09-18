"""
LangGraph Agent with Production Error Handling
Retry logic, model fallback, and structured state management.
"""

from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable
from app.embeddings import build_agentic_rag_graph, create_sample_vectorstore
from app.config import get_settings


# === Agent State ===

class AgentState(TypedDict):
    """
    State for the production agent.
    Uses Annotated with add_messages reducer for message accumulation.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str
    
# === Agent Builder ===

class ProductionAgent:
    """
    Production LangGraph agent with:
    - Retry on failure (model fallback)
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = GoogleGenerativeAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,  # We handle retries ourselves
            api_key=settings.google_api_key,
        )
        self.fallback_llm = GoogleGenerativeAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.google_api_key,
        )
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph state machine."""

        def process_message(state: AgentState) -> dict:
            """Try to process the message with the primary model."""
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary model."""
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """Return a graceful error message."""
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, I'm having trouble processing your request "
                        "right now. Please try again in a moment."
                    ))
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt."""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback attempt."""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        def retrival(state: AgentState) -> dict:
            message = state["messages"][-1].content
            vectorstore = create_sample_vectorstore()
            if vectorstore is None:
                return {
                    "messages": [AIMessage(content=(
                        "I couldn't answer because the knowledge base is empty. "
                        "Please add source documents and try again."
                    ))],
                    "error": "knowledge_base_empty",
                    "model_used": "retrieval",
                }

            initial_state = {
                        "query": message,
                        "rewritten_query": "",
                        "documents": [],
                        "prompt": "",
                        "failed": "",
                        "generation": "",
                        "relevance_score": 0.0,
                        "retry_count": 0,
                        "max_retries": 2,
                        "_vectorstore": vectorstore,  # Pass vectorstore via state
                    }

            rag= build_agentic_rag_graph()
            result = rag.invoke(initial_state)

            if result.get("prompt"):
                return {
                    "messages": [HumanMessage(content=result["prompt"])],
                    "error": None,
                }

            return {
                "messages": [AIMessage(content=result.get("failed", "Retrieval failed."))],
                "error": "retrieval_failed",
            }

        def route_after_retrieval(state: AgentState) -> str:
            """Send only the retrieval-built prompt to the answer model."""
            return "process" if state.get("error") is None else "done"

        # Build the graph
        graph = StateGraph(AgentState)
        graph.add_node("retrival", retrival)
        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "retrival")

        graph.add_conditional_edges(
            "retrival",
            route_after_retrieval,
            {"process": "process", "done": END},
        )
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }