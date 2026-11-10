"""Domain models used by the agent workflow and stored in its checkpoints."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from insightflow.rag.models import RetrievedChunk
from insightflow.tools.models import ToolResult

VerificationStatus = Literal["passed", "failed", "unchecked", "skipped"]
VerificationMethod = Literal["none", "deterministic", "llm"]

ANSWER_ACTION = "answer"
SEARCH_TOOL = "search_documents"
"""The local document-search tool. The name is reserved: an MCP tool with it is ignored."""


class WorkflowModel(BaseModel):
    """Immutable, strict base for workflow state values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConversationTurn(WorkflowModel):
    """One completed question and answer in a conversation thread."""

    query: str
    answer: str


class VerificationResult(WorkflowModel):
    """The outcome of checking an answer against its observations."""

    status: VerificationStatus
    method: VerificationMethod
    issues: list[str] = Field(default_factory=list)


class Observation(WorkflowModel):
    """One action of the agent and what it returned.

    `result` is the audit record of the call. `passages` holds the document passages that a
    `search_documents` call returned; web pages are in `result.web_sources`.
    """

    attempt: int = Field(ge=1)
    thought: str = ""
    result: ToolResult
    passages: list[RetrievedChunk] = Field(default_factory=list)


class ModelOutput(BaseModel):
    """Base for JSON produced by a chat model: extra keys are ignored, not rejected."""

    model_config = ConfigDict(extra="ignore")


class AgentAction(ModelOutput):
    """Agent output for one step: a tool call, or `answer` to stop and answer."""

    thought: str = ""
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("thought")
    @classmethod
    def strip_thought(cls, value: str) -> str:
        return value.strip()

    @field_validator("action")
    @classmethod
    def normalize_action(cls, value: str) -> str:
        action = value.strip()
        if not action:
            raise ValueError("action must not be empty")
        return ANSWER_ACTION if action.lower() == ANSWER_ACTION else action

    @property
    def is_answer(self) -> bool:
        return self.action == ANSWER_ACTION


class JudgeVerdict(ModelOutput):
    """LLM-judge output for answer groundedness."""

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
