from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InspectRequest(BaseModel):
    text: str = Field(min_length=1)


class InspectResponse(BaseModel):
    predicted_class: str
    decision: str
    malicious_probability: float
    safe_probability: float
    blocked: bool
    detector: str
    latency_ms: float
    risk_level: str
    block_threshold: float
    review_threshold: float
    explanation: str


class ChatRequest(BaseModel):
    # Preserve provider-specific and newer OpenAI-compatible fields such as
    # stream, tools, top_p, response_format, and tool_choice.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]

