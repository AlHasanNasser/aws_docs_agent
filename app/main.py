import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app.agent import GeminiAnswerProvider, RAGAgent
from app.cache import ResponseCache
from app.config import get_settings
from app.embeddings import ChromaVectorStore, GeminiEmbeddingProvider
from app.models import ChatRequest, ChatResponse, HealthResponse, MetricsResponse
from app.monitoring import MetricsCollector, RequestTimer


def _configure_langsmith() -> None:
	settings = get_settings()
	if settings.langchain_api_key:
		os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
		os.environ.setdefault(
			"LANGCHAIN_TRACING_V2",
			str(settings.langchain_tracing_v2).lower(),
		)
		os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)


app = FastAPI(title="AWS RAG API", version="0.1.0")
cache = ResponseCache()
metrics = MetricsCollector()


@lru_cache(maxsize=1)
def get_agent() -> RAGAgent:
	_configure_langsmith()
	settings = get_settings()
	embeddings = GeminiEmbeddingProvider(api_key=settings.google_api_key)
	return RAGAgent(
		store=ChromaVectorStore(
			path="data/chroma",
			collection_name="aws-rag-gemini",
		),
		embeddings=embeddings,
		answer_provider=GeminiAnswerProvider(api_key=settings.google_api_key),
		n_results=5,
	)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
	settings = get_settings()
	return HealthResponse(
		status="healthy",
		environment=settings.app_env,
		version="1.0.0",
		checks={"vector_store": "configured", "agent": "configured"},
	)


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
	return MetricsResponse(**metrics.summary)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
	settings = get_settings()
	timer = RequestTimer()
	try:
		with timer:
			cached_response = cache.get(request.message)
			if cached_response is None:
				result = get_agent().ask(request.message)
				cache.set(request.message, result.text)
	except ValueError as error:
		metrics.record_request(timer.elapsed_ms, error=True)
		raise HTTPException(status_code=400, detail=str(error)) from error
	except Exception as error:
		metrics.record_request(timer.elapsed_ms, error=True)
		raise HTTPException(status_code=500, detail="Unable to answer request") from error

	if cached_response is not None:
		metrics.record_request(timer.elapsed_ms, cache_hit=True)
		return ChatResponse(
			response=cached_response,
			thread_id=request.thread_id,
			model_used=settings.primary_model,
			cached=True,
			processing_time_ms=timer.elapsed_ms,
		)

	metrics.record_request(timer.elapsed_ms, cache_hit=False)
	return ChatResponse(
		response=result.text,
		thread_id=request.thread_id,
		model_used=settings.primary_model,
		cached=False,
		processing_time_ms=timer.elapsed_ms,
	)
