"""Logs never contain questions, passages, answers, tokens, or configured secrets."""

import pytest

from insightflow.core.config import Settings, get_settings
from insightflow.core.logging import configure_logging
from insightflow.rag.models import RetrievedChunk
from tests.fakes import ScriptedChatProvider
from tests.integration.test_query_api import (
    DETERMINISTIC_SETTINGS,
    FakeRetriever,
    client_for,
    make_app,
    make_candidate,
)

QUESTION = "CANARY-QUESTION-0071 what is the policy?"
PASSAGE = "CANARY-PASSAGE-0072 the policy text."
ANSWER = "CANARY-ANSWER-0073 the answer [1]."
API_KEY = "nvapi-hygiene-secret-0123456789"
BEARER = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJoeWdpZW5lIn0.c2lnbmF0dXJl"


def passage() -> RetrievedChunk:
    candidate = make_candidate()
    chunk = candidate.chunk.model_copy(update={"content": PASSAGE, "embedding_text": PASSAGE})
    return candidate.model_copy(update={"chunk": chunk})


@pytest.mark.parametrize("log_format", ["json", "text"])
async def test_a_full_request_at_debug_level_logs_no_content_or_secret(
    log_format: str, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("DEBUG", log_format=log_format, secrets=[API_KEY])  # type: ignore[arg-type]
    try:
        application = make_app(
            ScriptedChatProvider(answer=ANSWER), FakeRetriever([passage()])
        )
        application.dependency_overrides[get_settings] = lambda: Settings(
            _env_file=None, litellm_api_key=API_KEY, **DETERMINISTIC_SETTINGS
        )
        async with client_for(application) as client:
            response = await client.post(
                "/query", json={"query": QUESTION}, headers={"Authorization": BEARER}
            )
            stream = await client.post("/query/stream", json={"query": QUESTION})
    finally:
        configure_logging()

    assert response.status_code == 200 and stream.status_code == 200
    assert "CANARY-ANSWER-0073" in response.json()["answer"]
    logs = capsys.readouterr().err
    assert "query completed" in logs  # the logs were captured
    for forbidden in ("CANARY-QUESTION", "CANARY-PASSAGE", "CANARY-ANSWER", API_KEY, "eyJhbGci"):
        assert forbidden not in logs, forbidden
