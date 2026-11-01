"""Domain models used by the agent workflow and stored in its checkpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Route = Literal["retrieve", "direct"]
VerificationStatus = Literal["passed", "failed", "unchecked", "skipped"]
VerificationMethod = Literal["none", "deterministic", "llm"]


class WorkflowModel(BaseModel):
    """Immutable, strict base for workflow state values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConversationTurn(WorkflowModel):
    """One completed question and answer in a conversation thread."""

    query: str
    answer: str


class VerificationResult(WorkflowModel):
    """The outcome of checking an answer against its context."""

    status: VerificationStatus
    method: VerificationMethod
    issues: list[str] = Field(default_factory=list)


class ModelOutput(BaseModel):
    """Base for JSON produced by a chat model: extra keys are ignored, not rejected."""

    model_config = ConfigDict(extra="ignore")


class RouteDecision(ModelOutput):
    """Router output: the path to take and a self-contained version of the question."""

    route: Route
    standalone_query: str

    @field_validator("standalone_query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return value.strip()


class JudgeVerdict(ModelOutput):
    """LLM-judge output for answer groundedness."""

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class RewrittenQuery(ModelOutput):
    """Rewrite output: a new retrieval query."""

    query: str

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return value.strip()
