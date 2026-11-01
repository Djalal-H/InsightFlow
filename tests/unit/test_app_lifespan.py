"""Tests for the application lifespan that owns the workflow checkpointer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from insightflow import main
from insightflow.api.dependencies import get_checkpointer
from insightflow.core.config import Settings


async def test_lifespan_opens_and_closes_the_sqlite_checkpointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data" / "checkpoints.sqlite"
    settings = Settings(_env_file=None, checkpoint_backend="sqlite", checkpoint_sqlite_path=path)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    application = main.create_app()

    async with main.lifespan(application):
        checkpointer = application.state.checkpointer
        assert isinstance(checkpointer, AsyncSqliteSaver)
        assert path.exists()
        request = SimpleNamespace(app=application)
        assert get_checkpointer(request) is checkpointer  # type: ignore[arg-type]

    with pytest.raises(ValueError):  # aiosqlite: no active connection
        await checkpointer.conn.execute("SELECT 1")


async def test_lifespan_without_checkpointing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, checkpoint_backend="none")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    application = main.create_app()

    async with main.lifespan(application):
        assert application.state.checkpointer is None


def test_agent_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, agent_max_attempts=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, checkpoint_backend="redis")


def test_get_checkpointer_is_none_without_a_lifespan() -> None:
    request = SimpleNamespace(app=main.create_app())

    assert get_checkpointer(request) is None  # type: ignore[arg-type]
