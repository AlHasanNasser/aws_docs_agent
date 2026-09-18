from fastapi.testclient import TestClient

from app.main import app
from app.security import SecurityPipeline


class FakeAnswer:
    text = "ok"


class FakeAgent:
    def __init__(self):
        self.security = SecurityPipeline()

    def ask(self, question: str):
        return FakeAnswer()


def test_chat_returns_final_message_sent_to_llm(monkeypatch):
    monkeypatch.setattr("app.main.get_agent", lambda: FakeAgent())

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"message": "Contact me at test@example.com", "thread_id": "abc"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_message_for_llm"] == "Contact me at [EMAIL REDACTED]"
