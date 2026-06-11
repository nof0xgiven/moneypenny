"""Thread-safe FIFO of pending briefing strings, drained by the engine loop.

Phase 1: plain FIFO is sufficient (single Tier-1 briefing in flight).
Phase 2 MUST upgrade this to a priority queue so CORRECTION briefings preempt
queued BRIEFINGs (spec component table)."""
from __future__ import annotations

import queue


class InjectionQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()

    def put(self, briefing: str) -> None:
        self._q.put_nowait(briefing)

    def get(self) -> str | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None
