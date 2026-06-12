"""Web server: serves the dashboard and bridges the EventBus over WebSocket.

GET /    -> moneypenny/static/index.html (the dashboard, Task 3)
GET /ws  -> on connect, replay the last `session` and `status` events, then
            stream every bus event as JSON until disconnect. Incoming
            {"cmd": "start"|"stop"} drives the Session; every command is
            answered with {"type": "ack", "cmd", "ok"[, "error"]} and a bad
            command (unknown cmd, malformed JSON) can never kill the socket.

Each connection holds its own bus subscription (subscribed before the
handshake completes so no event emitted after connect can be missed) and a
writer task draining that queue concurrently with the reader loop; the
finally block tears both down on disconnect.

No mlx anywhere on this import path: this module imports moneypenny.app,
which is itself guarded by tests/test_app_imports.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import WSMsgType, web
from dotenv import load_dotenv

from moneypenny.app import Session
from moneypenny.config import Config
from moneypenny.events import EventBus

log = logging.getLogger("moneypenny.web")

DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).parent / "static"

BUS_KEY = web.AppKey("bus", EventBus)
SESSION_KEY = web.AppKey("session", object)
REPLAY_TYPES = ("session", "status")


async def _index(request: web.Request) -> web.Response:
    return web.Response(
        text=(STATIC_DIR / "index.html").read_text(), content_type="text/html"
    )


async def _pump_events(ws: web.WebSocketResponse, q: asyncio.Queue) -> None:
    """Writer task: drain the subscriber queue into the socket."""
    while True:
        await ws.send_json(await q.get())


async def _send_ack(ws: web.WebSocketResponse, ack: dict) -> None:
    """A command can outlive an impatient client (session.stop() takes seconds
    while the engine resets); acking the now-dead socket is normal, not an
    error. Caught here so it can never crash the handler with an
    aiohttp.server traceback (regression: tests/test_web.py vanishing client)."""
    try:
        await ws.send_json(ack)
    except (RuntimeError, ConnectionResetError) as exc:
        # ConnectionResetError while the transport is closing; RuntimeError
        # once aiohttp considers the socket closed.
        log.debug("client gone before ack %s could be delivered: %r", ack, exc)


async def _handle_command(ws: web.WebSocketResponse, session, raw: str) -> None:
    """One inbound message -> exactly one ack; never raises."""
    cmd = None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
        cmd = parsed.get("cmd")
        if cmd == "start":
            await session.start()
        elif cmd == "stop":
            await session.stop()
        else:
            raise ValueError(f"unknown cmd {cmd!r}")
        await _send_ack(ws, {"type": "ack", "cmd": cmd, "ok": True})
    except Exception as exc:
        await _send_ack(
            ws, {"type": "ack", "cmd": cmd, "ok": False, "error": str(exc) or repr(exc)}
        )


async def _ws(request: web.Request) -> web.WebSocketResponse:
    bus = request.app[BUS_KEY]
    session = request.app[SESSION_KEY]
    ws = web.WebSocketResponse()
    # Subscribe BEFORE the handshake reply: once the client sees the 101, any
    # event emitted is guaranteed to be queued for it. Replay ordering: the
    # last() reads below happen AFTER this subscribe, so a session/status
    # event landing in between is both replayed AND queued — the client can
    # see it twice (never a gap). Clients must treat session/status events as
    # idempotent state snapshots, which they are.
    q = bus.subscribe()
    writer: asyncio.Task | None = None
    try:
        await ws.prepare(request)
        for type_ in REPLAY_TYPES:
            last = bus.last(type_)
            if last is not None:
                await ws.send_json(last)
        writer = asyncio.get_running_loop().create_task(_pump_events(ws, q))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await _handle_command(ws, session, msg.data)
    finally:
        if writer is not None:
            writer.cancel()
            try:
                await writer
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                # a send into a just-closed socket is normal on disconnect
                log.debug("ws writer closed: %r", exc)
        bus.unsubscribe(q)
    return ws


def build_app(bus: EventBus, session) -> web.Application:
    app = web.Application()
    app[BUS_KEY] = bus
    app[SESSION_KEY] = session
    app.router.add_get("/", _index)
    app.router.add_get("/ws", _ws)
    return app


async def serve(
    bus: EventBus, session, host: str = "127.0.0.1", port: int | None = None
) -> web.AppRunner:
    if port is None:
        port = int(os.environ.get("MONEYPENNY_WEB_PORT", DEFAULT_PORT))
    runner = web.AppRunner(build_app(bus, session))
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    log.info("web ui: http://%s:%d", host, port)
    return runner


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    load_dotenv()
    cfg = Config.from_env()
    bus = EventBus(asyncio.get_running_loop())
    session = Session(cfg, bus)
    await session.load()
    await serve(bus, session)
    # The UI starts/stops the session; the process itself just stays up.
    await asyncio.Event().wait()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
