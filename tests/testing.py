from app.extracting import iter_pdfs_from_folder
from app.chunking import iter_chunks
from app.embeddings import (
    ChromaVectorStore,
    GeminiEmbeddingProvider,
    index_chunks,
)
from app.agent import GeminiAnswerProvider, RAGAgent

records = iter_pdfs_from_folder("docs")
chunks = iter_chunks(records)

provider = GeminiEmbeddingProvider()

store = ChromaVectorStore(
    path="data/chroma",
    collection_name="aws-rag-gemini",
)

deleted_chunks = store.remove_deleted_documents("docs")
print(f"Removed {deleted_chunks} chunks from deleted PDFs")

index_chunks(
    chunks,
    store=store,
    provider=provider,
    batch_size=32,
)

results = store.search(
    provider.embed_query("What is AWS Lambda?"),
    n_results=2
)

print(results)

agent = RAGAgent(
    store=store,
    embeddings=provider,
    answer_provider=GeminiAnswerProvider(),
    n_results=2,
)
print(agent.ask("What is AWS Lambda?"))