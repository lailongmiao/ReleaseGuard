"""Schemas for evals."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ScoreSchema(BaseModel):
    """Score schema for evals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(
        strict=True, ge=0.0, le=1.0, allow_inf_nan=False, description="provide a score between 0 and 1"
    )
    reasoning: str = Field(min_length=1, max_length=240, description="provide a one sentence reasoning")
