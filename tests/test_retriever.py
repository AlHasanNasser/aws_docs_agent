from langchain_core.documents import Document

from app.embeddings import retrieve_documents


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.query = None

    def invoke(self, query):
        self.query = query
        return self.documents


class FakeVectorStore:
    def __init__(self, documents):
        self.retriever = FakeRetriever(documents)
        self.search_kwargs = None

    def as_retriever(self, *, search_kwargs):
        self.search_kwargs = search_kwargs
        return self.retriever


def test_retrieve_documents_uses_rewritten_query_and_limits_results():
    documents = [
        Document(
            page_content="AWS Lambda runs code without provisioning servers.",
            metadata={"source": "lambda.md"},
        )
    ]
    vectorstore = FakeVectorStore(documents)

    result = retrieve_documents(
        {
            "query": "What is serverless compute?",
            "rewritten_query": "How does AWS Lambda run code?",
            "document_chunks": [],
            "_vectorstore": vectorstore,
        }
    )

    assert vectorstore.search_kwargs == {"k": 3}
    assert vectorstore.retriever.query == "How does AWS Lambda run code?"
    assert result == {"documents": documents}