"""RouteDecision: the router's output contract, in an mlx-free module.

Lives apart from moneypenny.router so consumers that only need the type
(ToolHost) don't transitively import the MLX stack — the app's import
affinity requires that no mlx-touching module loads on the main thread
(see moneypenny.app module docstring)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    tier: int
    tool: str | None
    args: dict
    confidence: float
