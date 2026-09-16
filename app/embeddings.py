import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import chromadb
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()



class EmbeddingProvider(Protocol):
	"""Provider interface used by the Chroma indexer."""

	def embed(self, text: str) -> list[float]: ...


class GeminiEmbeddingProvider:
	"""Create retrieval embeddings with Google's Gemini API."""

	def __init__(
		self,
		*,
		model_id: str = "gemini-embedding-001",
		api_key: str | None = None,
		dimensions: int = 768,
	) -> None:
		self.model_id = model_id
		self.dimensions = dimensions
		self.client = genai.Client(api_key=api_key)

	def embed(
		self,
		text: str,
		*,
		task_type: str = "RETRIEVAL_DOCUMENT",
	) -> list[float]:
		response = self.client.models.embed_content(
			model=self.model_id,
			contents=text,
			config=types.EmbedContentConfig(
				task_type=task_type,
				output_dimensionality=self.dimensions,
			),
		)
		if not response.embeddings or not response.embeddings[0].values:
			raise RuntimeError("Gemini returned an empty embedding")
		return list(response.embeddings[0].values)

	def embed_query(self, text: str) -> list[float]:
		"""Embed a search query using Gemini's query task type."""
		return self.embed(text, task_type="RETRIEVAL_QUERY")


class ChromaVectorStore:
	"""Persist embeddings in a local or shared Chroma collection."""

	def __init__(
		self,
		*,
		path: str = "data/chroma",
		collection_name: str = "aws-rag-chunks",
		client: Any | None = None,
	) -> None:
		self.client = client or chromadb.PersistentClient(path=path)
		self.collection = self.client.get_or_create_collection(
			name=collection_name,
			metadata={"hnsw:space": "cosine"},
		)

	def index_batch(
		self,
		chunks: Iterable[dict[str, Any]],
		provider: EmbeddingProvider,
	) -> int:
		ids: list[str] = []
		documents: list[str] = []
		embeddings: list[list[float]] = []
		metadatas: list[dict[str, Any]] = []
		count = 0
		for chunk in chunks:
			content = str(chunk["content"])
			metadata = dict(chunk.get("metadata", {}))
			ids.append(_chunk_id(content, metadata))
			documents.append(content)
			embeddings.append(provider.embed(content))
			metadatas.append(_chroma_metadata(metadata))
			count += 1

		if ids:
			self.collection.upsert(
				ids=ids,
				documents=documents,
				embeddings=embeddings,
				metadatas=metadatas,
			)
		return count

	def delete_sources(self, sources: set[str]) -> int:
		"""Delete every stored chunk belonging to the supplied source paths."""
		if not sources:
			return 0

		stored = self.collection.get(include=["metadatas"])
		ids_to_delete = [
		document_id
		for document_id, metadata in zip(
			stored.get("ids", []), stored.get("metadatas", []), strict=True
		)
		if metadata and metadata.get("source") in sources
		]
		if ids_to_delete:
			self.collection.delete(ids=ids_to_delete)
		return len(ids_to_delete)

	def remove_deleted_documents(
		self,
		folder_path: str | Path = "docs",
		*,
		recursive: bool = True,
	) -> int:
		"""Remove chunks whose source PDF no longer exists in a folder."""
		folder_path = Path(folder_path)
		pattern = "**/*.pdf" if recursive else "*.pdf"
		current_sources = {
			str(path)
			for path in folder_path.glob(pattern)
			if path.is_file()
		}

		stored = self.collection.get(include=["metadatas"])
		stale_ids = [
			document_id
			for document_id, metadata in zip(
				stored.get("ids", []), stored.get("metadatas", []), strict=True
			)
			if metadata
			and metadata.get("source")
			and metadata["source"] not in current_sources
		]
		if stale_ids:
			self.collection.delete(ids=stale_ids)
		return len(stale_ids)

	def search(
		self,
		query_embedding: list[float],
		*,
		n_results: int = 5,
	) -> dict[str, Any]:
		"""Find the nearest stored chunks for a query embedding."""
		return self.collection.query(
			query_embeddings=[query_embedding],
			n_results=n_results,
		)


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
	"""Keep metadata scalar because Chroma rejects nested values."""
	return {
		key: value
		for key, value in metadata.items()
		if isinstance(value, (str, int, float, bool))
	}


def _chunk_id(content: str, metadata: dict[str, Any]) -> str:
	identity = json.dumps(
		{"content": content, "metadata": metadata},
		sort_keys=True,
		default=str,
	).encode("utf-8")
	return hashlib.sha256(identity).hexdigest()


def index_chunks(
	chunks: Iterable[dict[str, Any]],
	*,
	store: ChromaVectorStore,
	provider: EmbeddingProvider,
	batch_size: int = 32,
) -> int:
	"""Embed and persist chunks incrementally in Chroma."""
	if batch_size < 1:
		raise ValueError("batch_size must be at least 1")

	batch: list[dict[str, Any]] = []
	indexed = 0
	for chunk in chunks:
		batch.append(chunk)
		if len(batch) >= batch_size:
			indexed += store.index_batch(batch, provider)
			batch.clear()
	if batch:
		indexed += store.index_batch(batch, provider)
	return indexed
