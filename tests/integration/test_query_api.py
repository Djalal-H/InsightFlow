"""Happy-path test for POST /query."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from insightflow.main import app


def test_query_returns_answer_and_sources() -> None:
    """A successful query returns the mocked answer with an empty sources list."""
    with patch(
        "insightflow.api.routes.query.LiteLLMChatProvider.complete",
        new_callable=AsyncMock,
        return_value="This is a mocked answer.",
    ):
        client = TestClient(app)
        response = client.post("/query", json={"query": "What is InsightFlow?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This is a mocked answer."
    assert body["sources"] == []
