from collections.abc import Iterator, Sequence
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
	if chunk_overlap >= chunk_size:
		raise ValueError("chunk_overlap must be smaller than chunk_size")
	return RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		length_function=len,
		separators=["\n\n", "\n", ". ", " ", ""],
		keep_separator=True,
	)


def _split_table(
	content: str,
	*,
	chunk_size: int,
	chunk_overlap: int,
) -> list[str]:
	"""Keep a Markdown table header with every row group."""
	lines = [line for line in content.splitlines() if line.strip()]
	if len(lines) < 3 or len(content) <= chunk_size:
		return [content]

	header = lines[:2]
	chunks: list[str] = []
	rows: list[str] = []
	current_size = sum(len(line) + 1 for line in header)

	for row in lines[2:]:
		row_size = len(row) + 1
		if rows and current_size + row_size > chunk_size:
			chunks.append("\n".join(header + rows))
			rows = []
			current_size = sum(len(line) + 1 for line in header)
		rows.append(row)
		current_size += row_size

	if rows:
		chunks.append("\n".join(header + rows))

	# A very wide individual row still needs a bounded fallback.
	result: list[str] = []
	for chunk in chunks:
		if len(chunk) <= chunk_size:
			result.append(chunk)
		else:
			result.extend(_splitter(chunk_size, chunk_overlap).split_text(chunk))
	return result


def _content_chunks(
	content: str,
	record_type: str,
	*,
	chunk_size: int,
	chunk_overlap: int,
) -> list[str]:
	if record_type == "table":
		return _split_table(
			content,
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)
	return _splitter(chunk_size, chunk_overlap).split_text(content)


def iter_chunks(
	records: Sequence[dict[str, Any]] | Iterator[dict[str, Any]],
	*,
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[dict[str, Any]]:
	"""Yield bounded, metadata-preserving chunks from extraction records.

	The input may be the lazy iterator from ``iter_pdfs_from_folder``. Chunks
	are yielded immediately, allowing callers to embed and persist them
	without collecting the entire PDF in memory.
	"""
	for record in records:
		content = str(record.get("content", "")).strip()
		if not content:
			continue

		record_type = str(record.get("type", "text"))
		metadata = dict(record.get("metadata", {}))
		metadata.update(
			{
				"source": record.get("source", metadata.get("source")),
				"page": record.get("page", metadata.get("page")),
				"type": record_type,
			}
		)

		pieces = _content_chunks(
			content,
			record_type,
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)
		for chunk_index, piece in enumerate(pieces):
			yield {
				"content": piece,
				"metadata": {
					**metadata,
					"chunk_index": chunk_index,
					"chunk_count": len(pieces),
				},
			}


def chunk_documents(
	records: Sequence[dict[str, Any]],
	*,
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
	"""Return all chunks for callers that explicitly need a list."""
	return list(
		iter_chunks(
			records,
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)
	)
