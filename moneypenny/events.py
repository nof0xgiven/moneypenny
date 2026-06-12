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
  route    {"transcript": str, "tier": int, "tool": str|None,
            "confidence": float}                           route worker
  tool     after a Tier-1 execution attempt:
             {"ok": bool, "briefing": str|None, "transcript": str}
           or, for a tier-2/3 escalation (no execution):
             {"ok": False, "escalated": int, "transcript": str}
  briefing {"stage": "synthesized", "ms": float, "audio_s": float}
           then {"stage": "injected"}
  status   {"micq", "spkq", "underruns", "mic_rms_max", "vad",
            "asr_on", "fps", "asr_ms", "step_ms"}          every ~2s

Threading: emit() is safe from ANY thread — events are always marshalled to
the owning asyncio loop via call_soon_threadsafe, so subscriber queues and
the last-event cache are only ever touched on the loop thread. Subscriber
queues are bounded; on overflow the OLDEST event is dropped (and counted on
the queue's .dropped attribute) so a slow consumer can neither grow memory
nor block the loop.
"""
from __future__ import annotations

import asyncio
import time


class EventBus:
    DEFAULT_MAXSIZE = 256

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._subscribers: list[asyncio.Queue] = []
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

    def subscribe(self, maxsize: int = DEFAULT_MAXSIZE) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        q.dropped = 0  # overflow counter, read by the web server for diagnostics
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def last(self, type_: str) -> dict | None:
        """Most recent event of a type (web server replays status/session)."""
        return self._last.get(type_)
