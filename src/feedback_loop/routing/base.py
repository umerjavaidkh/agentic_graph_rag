"""Routing policy contracts for feedback-driven retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from ..hints import ModeHint


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of applying a routing policy to a mode hint."""

    action: str
    value: str


class RoutingPolicy(Protocol):
    """Maps a historical mode hint to an actionable routing decision."""

    name: str
    hint_agent: str

    def decide(self, hint: ModeHint) -> Optional[PolicyDecision]:
        """Return a decision when the hint mode is actionable for this policy."""
