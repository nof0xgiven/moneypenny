"""In-process timers. Fire -> callback (app turns it into a briefing injection).

Thread-safety: set_timer/cancel_all may be called from any thread; all asyncio
scheduling is marshalled onto the owning loop via call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from typing import Callable

_UNIT_S = {"second": 1, "seconds": 1, "minute": 60, "minutes": 60, "hour": 3600, "hours": 3600}
_DURATION_RE = re.compile(r"^\s*(\d+)\s*(seconds?|minutes?|hours?)\s*$", re.IGNORECASE)


def parse_duration(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    m = _DURATION_RE.match(text)
    if not m:
        return None
    return int(m.group(1)) * _UNIT_S[m.group(2).lower()]


@dataclass
class ActiveTimer:
    label: str
    handle: asyncio.TimerHandle


class TimerService:
    def __init__(self, on_fire: Callable[[str], None], loop: asyncio.AbstractEventLoop) -> None:
        self._on_fire = on_fire
        self._loop = loop
        self._active: dict[int, ActiveTimer] = {}
        self._lock = threading.Lock()
        self._next_id = 0

    def set_timer(self, seconds: float, label: str) -> int:
        with self._lock:
            tid = self._next_id
            self._next_id += 1
        # schedule on the loop's own thread; safe from any caller thread
        self._loop.call_soon_threadsafe(self._schedule, tid, seconds, label)
        return tid

    def _schedule(self, tid: int, seconds: float, label: str) -> None:
        def fire() -> None:
            with self._lock:
                self._active.pop(tid, None)
            self._on_fire(label)

        handle = self._loop.call_later(seconds, fire)
        with self._lock:
            self._active[tid] = ActiveTimer(label=label, handle=handle)

    def active(self) -> list[ActiveTimer]:
        with self._lock:
            return list(self._active.values())

    def cancel_all(self) -> None:
        def _cancel() -> None:
            with self._lock:
                for t in self._active.values():
                    t.handle.cancel()
                self._active.clear()

        self._loop.call_soon_threadsafe(_cancel)
