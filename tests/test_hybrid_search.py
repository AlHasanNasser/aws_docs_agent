import unittest

from app.embeddings import ChromaVectorStore


class FakeCollection:
    def __init__(self):
        self.docs = [
            "Lambda functions are serverless compute units.",
            "This article is about cats and dogs.",
            "Lambda events trigger code execution.",
        ]
        self.metadatas = [
            {"source": "doc1"},
            {"source": "doc2"},
            {"source": "doc3"},
        ]

    def get(self, include=None):
        return {
            "ids": ["1", "2", "3"],
            "documents": self.docs,
            "metadatas": self.metadatas,
        }

    def query(self, query_embeddings, n_results=5):
        # semantic search should rank document 1 highest for lambda question
        return {
            "documents": [[self.docs[0], self.docs[2], self.docs[1]]],
            "metadatas": [[self.metadatas[0], self.metadatas[2], self.metadatas[1]]],
            "distances": [[0.1, 0.5, 0.9]],
        }


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()

    def get_or_create_collection(self, name, metadata=None):
        return self.collection


class HybridSearchTests(unittest.TestCase):
    def test_hybrid_search_ranks_keyword_matches_above_irrelevant_docs(self):
        store = ChromaVectorStore(client=FakeClient(), collection_name="demo")

        results = store.hybrid_search("what is lambda", [0.1, 0.2, 0.3], n_results=2)

        docs = results["documents"][0]
        self.assertEqual(docs[0], "Lambda functions are serverless compute units.")
        self.assertIn("Lambda events trigger code execution.", docs)


if __name__ == "__main__":
    unittest.main()
