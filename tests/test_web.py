"""Web server: HTTP index + WebSocket event stream / session control.

Real aiohttp server via TestServer/TestClient — nothing of OURS is mocked.
FakeSession is the sanctioned device/model boundary stand-in (Session.start()
needs real audio devices): a 10-line object with start/stop coroutines and a
running flag, same policy as the AudioIO boundary elsewhere in the suite.
"""
import asyncio
import contextlib
import logging

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from moneypenny.events import EventBus
from moneypenny.web import _send_ack, build_app, serve


class FakeSession:
    """Device/model boundary stand-in (start needs real audio hardware)."""

    def __init__(self, start_error: Exception | None = None) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.running = False
        self._start_error = start_error

    async def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.start_calls += 1
        self.running = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


@contextlib.asynccontextmanager
async def _client(session=None, bus=None):
    bus = bus if bus is not None else EventBus(asyncio.get_running_loop())
    session = session if session is not None else FakeSession()
    client = TestClient(TestServer(build_app(bus, session)))
    await client.start_server()
    try:
        yield client, bus, session
    finally:
        await client.close()


async def _drain_pending() -> None:
    """Let call_soon_threadsafe dispatches scheduled so far run."""
    await asyncio.sleep(0.05)


async def test_index_serves_html():
    async with _client() as (client, _bus, _session):
        resp = await client.get("/")
        assert resp.status == 200
        assert resp.content_type == "text/html"
        assert "Moneypenny" in await resp.text()


async def test_ws_replays_last_session_and_status_on_connect():
    bus = EventBus(asyncio.get_running_loop())
    bus.emit("session", state="ready", home_control=False)
    bus.emit("status", fps=12.5)
    await _drain_pending()
    async with _client(bus=bus) as (client, _bus, _session):
        ws = await client.ws_connect("/ws")
        first = await ws.receive_json(timeout=2)
        second = await ws.receive_json(timeout=2)
        assert first["type"] == "session"
        assert first["state"] == "ready"
        assert second["type"] == "status"
        assert second["fps"] == 12.5
        await ws.close()


async def test_ws_streams_events_emitted_after_connect():
    async with _client() as (client, bus, _session):
        ws = await client.ws_connect("/ws")
        bus.emit("vad", event="speech_start", partial="")
        ev = await ws.receive_json(timeout=2)
        assert ev["type"] == "vad"
        assert ev["event"] == "speech_start"
        await ws.close()


async def test_ws_start_cmd_acks_ok_and_calls_session():
    async with _client() as (client, _bus, session):
        ws = await client.ws_connect("/ws")
        await ws.send_json({"cmd": "start"})
        ack = await ws.receive_json(timeout=2)
        assert ack == {"type": "ack", "cmd": "start", "ok": True}
        assert session.start_calls == 1
        await ws.close()


async def test_ws_start_failure_acks_error():
    failing = FakeSession(start_error=RuntimeError("session already started"))
    async with _client(session=failing) as (client, _bus, _session):
        ws = await client.ws_connect("/ws")
        await ws.send_json({"cmd": "start"})
        ack = await ws.receive_json(timeout=2)
        assert ack["type"] == "ack"
        assert ack["cmd"] == "start"
        assert ack["ok"] is False
        assert "already started" in ack["error"]
        await ws.close()


async def test_ws_stop_cmd_acks_ok():
    async with _client() as (client, _bus, session):
        ws = await client.ws_connect("/ws")
        await ws.send_json({"cmd": "stop"})
        ack = await ws.receive_json(timeout=2)
        assert ack == {"type": "ack", "cmd": "stop", "ok": True}
        assert session.stop_calls == 1
        await ws.close()


async def test_ws_unknown_cmd_and_malformed_json_keep_socket_usable():
    async with _client() as (client, _bus, session):
        ws = await client.ws_connect("/ws")

        await ws.send_json({"cmd": "self_destruct"})
        ack = await ws.receive_json(timeout=2)
        assert ack["ok"] is False
        assert "self_destruct" in ack["error"]

        await ws.send_str("{not json")
        ack = await ws.receive_json(timeout=2)
        assert ack["ok"] is False

        await ws.send_str('"a bare string"')  # valid JSON, not an object
        ack = await ws.receive_json(timeout=2)
        assert ack["ok"] is False

        # socket survived all three: a valid cmd still works
        await ws.send_json({"cmd": "stop"})
        ack = await ws.receive_json(timeout=2)
        assert ack["ok"] is True
        assert session.stop_calls == 1
        await ws.close()


async def test_two_ws_clients_both_receive_events():
    async with _client() as (client, bus, _session):
        ws1 = await client.ws_connect("/ws")
        ws2 = await client.ws_connect("/ws")
        bus.emit("audio", mic_rms=0.5, out_rms=0.0)
        ev1 = await ws1.receive_json(timeout=2)
        ev2 = await ws2.receive_json(timeout=2)
        assert ev1["type"] == ev2["type"] == "audio"
        assert ev1["mic_rms"] == ev2["mic_rms"] == 0.5
        await ws1.close()
        await ws2.close()


async def test_client_vanishing_mid_command_is_not_a_server_error(caplog):
    """Contract: a client that disconnects between sending a command and
    receiving the ack (session.stop() takes seconds while the engine resets)
    must end the handler quietly, never as an aiohttp.server ERROR traceback.

    Uses the real serve() path (AppRunner/TCPSite) because TestServer's
    in-suite transport teardown skips the socket close that triggers it."""
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowSession(FakeSession):
        async def stop(self) -> None:
            entered.set()
            await release.wait()

    bus = EventBus(asyncio.get_running_loop())
    runner = await serve(bus, SlowSession(), port=0)  # ephemeral port
    port = runner.addresses[0][1]
    try:
        async with aiohttp.ClientSession() as http:
            ws = await http.ws_connect(f"http://127.0.0.1:{port}/ws")
            await ws.send_json({"cmd": "stop"})
            await asyncio.wait_for(entered.wait(), timeout=2)
        # ClientSession closed: client is gone while the server is in stop()
        await _drain_pending()
        release.set()
        await _drain_pending()  # handler acks the dead socket and finishes
        assert bus._subscribers == []
    finally:
        release.set()
        await runner.cleanup()
    server_errors = [r for r in caplog.records
                     if r.name == "aiohttp.server" and r.levelno >= logging.ERROR]
    assert server_errors == []


async def test_send_ack_swallows_closed_socket_runtime_error():
    """aiohttp raises RuntimeError (not ConnectionResetError) once the socket
    is closed/closing. Racing a real close into exactly that state is not
    reproducible deterministically in-suite, but an unprepared
    WebSocketResponse raises the same RuntimeError from the same send path —
    a real aiohttp object, no mocks."""
    ws = web.WebSocketResponse()  # no transport: send_json raises RuntimeError
    await _send_ack(ws, {"type": "ack", "cmd": "stop", "ok": True})  # must not raise


async def test_ws_disconnect_unsubscribes_from_bus():
    async with _client() as (client, bus, _session):
        ws = await client.ws_connect("/ws")
        await _drain_pending()  # let the handler subscribe
        assert len(bus._subscribers) == 1
        await ws.close()
        await _drain_pending()  # let the handler's finally run
        assert bus._subscribers == []
