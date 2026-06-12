"""EventBus: structured events from the session to the web dashboard.

Pure stdlib (no mlx, no numpy): moneypenny.app imports this at module level,
so it must stay clear of the MLX import-affinity rule (tests/test_app_imports.py).

Every event is a dict: {"type": <str>, "ts": <time.time()>, **data}.

Event types emitted by the Session (data keys beyond type/ts):
  session  {"state": "loading" | "ready" | "live" | "stopped",
            "home_control": bool (ready only)}
  audio    {"mic_rms": float, "out_rms": float}            every frame
  vad      {"event": str, "partial": str}                  each VAD event
  partial  {"text": str}                                   on partial change
  gate     {"transcript": str,
            "blocked": "empty" | "backchannel" | "duplicate" | "self_echo"}
           classification suppressed before the route worker
           (filter semantics: moneypenny/classify_gate.py)
  route    {"transcript": str, "tier": int, "tool": str|None,
            "confidence": float}                           route worker
  tool     after a Tier-1 execution attempt:
             {"ok": bool, "briefing": str|None, "transcript": str}
           or, when ToolHost dropped the route without acting (arg-validation
           failed with no tool evidence in the transcript, or unknown tool):
             {"ok": True, "briefing": None, "dropped": True,
              "tool": str|None, "transcript": str}
           or, for a tier-2/3 escalation (no execution):
             {"ok": False, "escalated": int, "transcript": str}
  briefing {"stage": "synthesized", "ms": float, "audio_s": float}
           then {"stage": "injected"}
  status   {"micq", "spkq", "underruns", "mic_rms_max", "vad",
            "asr_on", "fps", "asr_ms", "step_ms"}          every ~2s

Threading: emit() is safe from ANY thread — events are always marshalled to
the owning asyncio loop via call_soon_threadsafe, so subscriber queues and
the last-event cache are only ever touched on the loop thread. Everything
ELSE (subscribe/unsubscribe/last) is loop-thread-only: the subscriber list
and last-event cache are unlocked, and the web server — the only caller —
lives on the loop. Subscriber queues are bounded; on overflow the OLDEST
event is dropped (and counted on the queue's .dropped attribute) so a slow
consumer can neither grow memory nor block the loop.
"""
from __future__ import annotations

import asyncio
import time


class SubscriberQueue(asyncio.Queue):
    """Bounded per-subscriber queue; .dropped counts overflow drops."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize=maxsize)
        self.dropped = 0


class EventBus:
    DEFAULT_MAXSIZE = 256

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._subscribers: list[SubscriberQueue] = []
        self._last: dict[str, dict] = {}

    def emit(self, type_: str, **data) -> None:
        """Safe from any thread (always marshalled via call_soon_threadsafe)."""
        event = {"type": type_, "ts": time.time(), **data}
        self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: dict) -> None:
        """Loop thread only: fan out to subscribers, drop-oldest on overflow."""
        self._last[event["type"]] = event
        for q in self._subscribers:
            if q.full():
                q.get_nowait()
                q.dropped += 1
            q.put_nowait(event)

    def subscribe(self, maxsize: int = DEFAULT_MAXSIZE) -> SubscriberQueue:
        """Loop-thread only. maxsize must be positive: 0 would mean an
        UNBOUNDED asyncio.Queue, defeating the drop-oldest overflow guard.
        A real raise (not assert): asserts are stripped under python -O."""
        if maxsize <= 0:
            raise ValueError("subscriber queues must be bounded (maxsize > 0)")
        q = SubscriberQueue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: SubscriberQueue) -> None:
        """Loop-thread only."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def last(self, type_: str) -> dict | None:
        """Loop-thread only: most recent event of a type (the web server
        replays status/session to new connections)."""
        return self._last.get(type_)
