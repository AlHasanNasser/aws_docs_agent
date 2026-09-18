import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app.agent import GeminiAnswerProvider, RAGAgent
from app.cache import ResponseCache
from app.config import get_settings
from app.embeddings import ChromaVectorStore, GeminiEmbeddingProvider
from app.models import (
	CacheStatsResponse,
	ChatRequest,
	ChatResponse,
	HealthResponse,
	MetricsResponse,
)
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
metrics = MetricsCollector()
cache = ResponseCache(ttl_seconds=get_settings().cache_ttl_seconds, metrics=metrics)


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



@app.get("/cache/stats", response_model=CacheStatsResponse)
def get_cache_stats() -> CacheStatsResponse:
	return CacheStatsResponse(**cache.stats)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
	settings = get_settings()
	timer = RequestTimer()
	agent = get_agent()
	allowed, cleaned_message, notes = agent.security.check_input(request.message)
	if not allowed:
		metrics.record_request(timer.elapsed_ms, error=True)
		raise HTTPException(status_code=400, detail=notes[0] if notes else "Question was rejected")

	try:
		with timer:
			cached_response = cache.get(request.message)
			if cached_response is None:
				result = agent.ask(cleaned_message)
				cache.set(request.message, result.text)
			final_llm_message = agent.last_prompt_for_llm
	except ValueError as error:
		metrics.record_request(timer.elapsed_ms, error=True)
		raise HTTPException(status_code=400, detail=str(error)) from error
	except Exception as error:
		metrics.record_request(timer.elapsed_ms, error=True)
		raise HTTPException(status_code=500, detail="Unable to answer request") from error

	if cached_response is not None:
		metrics.record_request(timer.elapsed_ms)
		return ChatResponse(
			response=cached_response,
			thread_id=request.thread_id,
			model_used=settings.primary_model,
			cached=True,
			processing_time_ms=timer.elapsed_ms,
			final_message_for_llm=final_llm_message or cleaned_message,
		)

	metrics.record_request(timer.elapsed_ms)
	return ChatResponse(
		response=result.text,
		thread_id=request.thread_id,
		model_used=settings.primary_model,
		cached=False,
		processing_time_ms=timer.elapsed_ms,
		final_message_for_llm=final_llm_message or cleaned_message,
	)
