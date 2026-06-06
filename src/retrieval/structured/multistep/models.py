"""Multistep plan data models."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class MultiStepStep(BaseModel):
    id: str
    purpose: str
    cypher: str
    expects: str = Field(default="rows")  # rows | scalar

    @field_validator("expects")
    @classmethod
    def _expects_allowed(cls, v: str) -> str:
        if v not in ("rows", "scalar"):
            return "rows"
        return v


class MultiStepPlan(BaseModel):
    needs_multistep: bool = False
    reason: str = ""
    steps: list[MultiStepStep] = Field(default_factory=list)
    final_answer_hint: str = ""
