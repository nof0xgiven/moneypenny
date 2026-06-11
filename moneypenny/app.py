"""Moneypenny main loop: the only file that touches real devices.

Flow per 80ms frame:
  mic -> VAD (cheap RMS, every frame) -> AsrGate -> [ASR if gated on] ->
  engine.step ; engine PCM -> speakers
  ASR runs ONLY around speech: pre-roll (320ms) flushed at speech_start,
  pair-batched frames while speaking, 640ms hangover after speech ends
  (see asr_gate.py). Silent frames cost zero ASR work. asr.add_frame runs on
  its own worker CONCURRENTLY with engine.step (asyncio.gather): per-frame
  cost is max(asr, step), not the sum — running them sequentially overran the
  80ms budget during speech (asr ~40ms amortized + step ~65ms), starving the
  speaker queue right when the model starts replying (measured: fps 7.6-10,
  audible underruns).
  VAD maybe_end (160ms silence)  -> classify on current ASR partial (P0.1:
                                    classification starts BEFORE the confirmed end;
                                    the partial lags <=160ms from pair-batching)
  VAD utterance_end (640ms)      -> if the final transcript materially grew and
                                    nothing executed yet, re-classify on it
  RouteDecision tier 1 -> ToolHost -> briefing text -> BriefingSynth (tts worker)
                                    -> engine.inject_audio (engine worker; the
                                    engine gates injection on output silence)

Threading model: engine.step and engine.inject_audio run on a single engine
worker thread (engine is single-threaded by design); router+tools run on a
second worker so classification never stalls the frame loop; ASR runs on a
third worker, overlapped with the engine step (above); briefing TTS runs on a
fourth (Kokoro synthesis used to run on the engine worker and stalled the
frame loop ~0.4-3s per briefing — now only the finished PCM crosses to the
engine worker). asyncio coordinates via run_in_executor. Worker exceptions
are surfaced via done-callbacks - never silently swallowed (invariant #2).

Thread affinity (hard requirement): MLX streams are thread-bound — each
thread has its own stream registry, and evaluating a graph whose ops were
scheduled on another thread's stream raises "There is no Stream(gpu, N) in
current thread." (then SIGBUS). So every MLX component is CONSTRUCTED on the
thread that will run it: VoiceEngine on the engine worker, Router on the
route worker (which also rebinds mlx_lm's import-time generation_stream — see
router.py), the ASR transcriber on the asr worker where add_frame/finish run,
and BriefingSynth on the tts worker. The pools therefore exist before the
models load.

Import affinity (measured hard requirement, the one sanctioned exception to
the imports-at-top rule): the MLX stack (engine/asr/router modules) is
imported INSIDE the loader functions, not at module top, so that the engine
worker is the first thread in the process to import mlx. Measured on this
hardware (M3 Ultra): with mlx first imported on the main thread and the
engine constructed on its worker, engine.step pays ~+80ms/frame (~183ms vs
~104ms co-resident; profiling shows the extra time as malloc/free churn
inside mlx eval on the non-importing thread). Thread QoS, MLX stream ids,
and executor overhead were each ruled out by measurement; import thread is
the reproducible lever. Guarded by tests/test_app_imports.py; numbers in
docs/decisions/0002 known limitation 6.

Known accepted limitation: if the user resumes speaking after maybe_end and
changes the meaning, an already-executed read-only tool (weather) is harmless;
Homey/timer commands need the full command present in the partial to classify
Tier 1 at all, and escalation bias pushes truncated partials to Tier 2.

Known Phase 1 limitation - speaker-queue latency ratchet: any engine-worker
stall (today: the model-load warm-up; formerly each inject() TTS) blocks
stepping, then the frame loop catches up by stepping through the backlog
faster than realtime, bursting the produced PCM into the unbounded
speaker_frames queue. After catch-up, production and consumption both run at
12.5 fps again, so the queue depth gained during the stall never drains:
every stall permanently adds its duration to mouth-to-ear latency for the
rest of the session. TODO(phase2): cap/drain speaker_frames on catch-up.

Drift protection (bounded catch-up): if the frame loop falls
>CATCHUP_TRIGGER_FRAMES behind real time DURING SILENCE, the oldest
room-tone frames are dropped down to CATCHUP_TARGET_FRAMES (single warning
logged). Frames are never dropped while in speech or ASR hangover - dropping
mid-utterance would corrupt the transcript; dropping silence just loses room
tone. Residual risk: fps can still dip below 12.5 during speech if the ASR
call exceeds the step time (per-frame cost is max(asr, step) since the two
overlap), but the dip is bounded and the queue drains in the following
silence.
Step budget (see decision 0002 known-limitation 6): the original ~183-213ms
idle step was the import-affinity penalty (fixed here via deferred imports)
plus synchronous mimi decode (fixed in engine.py via the one-frame decode
pipeline); idle step is now ~70ms, inside the 80ms budget, so catch-up only
fires after genuine stalls (model-load warm-up) rather than perpetually.

Known Phase 1 limitation - briefing drain vs. live mic: while a briefing
drains, the model hears the briefing audio but ASR/VAD still hear the real
mic. User speech during a drain can therefore classify and queue a second
briefing for which the model has no conversational antecedent.
TODO(phase2): consider suppressing classification while a drain is active.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import time

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

# NOTE: moneypenny.engine / .asr / .router are deliberately NOT imported here.
# They pull in the MLX stack, and the engine worker must be the first thread
# to import it (see "Import affinity" in the module docstring).
from moneypenny.asr_gate import AsrGate
from moneypenny.audio import AudioIO
from moneypenny.briefing import compose
from moneypenny.config import Config
from moneypenny.injection import InjectionQueue
from moneypenny.prompts import FRONT_OF_HOUSE
from moneypenny.tools import ToolHost
from moneypenny.tools.homey_adapter import HomeyAdapter
from moneypenny.tools.timers import TimerService
from moneypenny.vad import EnergyVAD

log = logging.getLogger("moneypenny")


def _load_engine(cfg: Config):
    """Runs on the engine worker: first mlx import in the process + construction."""
    from moneypenny.engine import VoiceEngine

    return VoiceEngine(
        system_prompt=FRONT_OF_HOUSE,
        voice=cfg.voice,
        quantize_bits=cfg.quantize_bits,
    )


def _load_router(cfg: Config):
    """Runs on the route worker (mlx_lm import + construction)."""
    from moneypenny.router import Router

    return Router(model_id=cfg.router_model)


def _load_asr(cfg: Config):
    """Runs on the asr worker, where add_frame/finish run (after the engine
    worker has already claimed the first mlx import)."""
    from moneypenny.asr import StreamingTranscriber

    return StreamingTranscriber(model_id=cfg.asr_model)


def _load_tts(cfg: Config):
    """Runs on the tts worker, where synthesize runs."""
    from moneypenny.tts import BriefingSynth

    return BriefingSynth(voice=cfg.briefing_voice)

STATUS_EVERY_FRAMES = 25  # ~2s of audio at 12.5 fps
CATCHUP_TRIGGER_FRAMES = 60  # ~5s behind real time: start dropping silence
CATCHUP_TARGET_FRAMES = 12   # ~1s of backlog kept after a catch-up drain


def _describe_device(kind: str) -> str:
    """One-line summary of the default device for kind ('input'/'output')."""
    try:
        d = sd.query_devices(kind=kind)
        return (f"{d['name']!r} max_{kind}_ch={d[f'max_{kind}_channels']} "
                f"default_sr={d['default_samplerate']:.0f}")
    except Exception as exc:  # diagnostics must never kill startup
        return f"<query failed: {exc!r}>"


class UtteranceState:
    """Tracks one utterance across the soft/hard VAD boundaries so a Tier 1
    tool never executes twice for the same utterance.

    gen is a generation counter bumped on every reset(): route jobs capture it
    at submit time, and the execution claim requires it to still match, so a
    job queued for utterance N can never execute after utterance N+1 has begun.
    The claim itself is executed_gen (the generation that executed a tool, -1
    for none) rather than a boolean: a stale worker that claims late can only
    write its own old gen value, which no current-generation check ever
    matches — so it can neither double-execute nor suppress the next
    utterance's execution. Single int reads/writes are atomic under the GIL;
    no lock needed."""

    def __init__(self) -> None:
        self.gen = 0
        self.classified_text: str | None = None
        self.executed_gen = -1

    def reset(self) -> None:
        self.gen += 1
        self.classified_text = None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    load_dotenv()
    cfg = Config.from_env()
    loop = asyncio.get_running_loop()

    # Pools first: MLX models must be constructed on the worker thread that
    # will run them (streams are thread-bound; see module docstring).
    engine_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
    route_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="route")
    asr_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
    tts_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    log.info("loading models (engine, router, asr, tts)...")
    # Engine FIRST: its worker must win the first-mlx-import race (import
    # affinity, module docstring) before the other pools touch mlx.
    engine = await loop.run_in_executor(engine_pool, _load_engine, cfg)
    router = await loop.run_in_executor(route_pool, _load_router, cfg)
    asr = await loop.run_in_executor(asr_pool, _load_asr, cfg)
    synth = await loop.run_in_executor(tts_pool, _load_tts, cfg)
    vad = EnergyVAD(rms_threshold=cfg.vad_rms_threshold)
    gate = AsrGate()  # hangover = vad's hard boundary by construction; see asr_gate.py
    injections = InjectionQueue()
    timers = TimerService(
        on_fire=lambda label: injections.put(compose("briefing", f"TIMER DONE {label.upper()}")),
        loop=loop,
    )
    homey = None
    homey_status = "unconfigured"
    if cfg.homey_configured:
        try:
            homey = HomeyAdapter.from_config(cfg)
        except Exception:
            homey_status = "unavailable"
            log.exception(
                "home control DISABLED: HomeyAdapter construction failed "
                "(box unreachable or bad credentials?) - continuing without it"
            )
    else:
        log.info("home control DISABLED: HOMEY_BASE_URL/HOMEY_API_KEY not set")
    host = ToolHost(cfg, homey, timers, homey_status=homey_status)
    log.info("models loaded (home control %s)", "enabled" if homey else "disabled")

    utt = UtteranceState()

    def classify_and_execute(transcript: str, t_marked: float, gen: int) -> None:
        """Runs on route_pool. Failures become log lines + (where sensible) briefings."""
        try:
            decision = router.classify(transcript)
            log.info("route %r -> %s", transcript, decision)
            if decision.tier == 1 and gen == utt.gen and utt.executed_gen != gen:
                utt.executed_gen = gen  # claim before executing: never run a tool twice
                try:
                    briefing = host.execute(decision)  # action > narration
                except Exception:
                    log.exception("tool execution failed for %r", transcript)
                    briefing = compose("briefing", "TOOL FAILED TELL USER YOU HIT A SNAG")
                if briefing:
                    injections.put(briefing)
                    log.info("briefing queued %.0fms after boundary: %r",
                             (time.perf_counter() - t_marked) * 1000, briefing)
            elif decision.tier in (2, 3):
                log.info("tier %d escalation (phase 2 wiring pending): %r",
                         decision.tier, transcript)
        except Exception:
            log.exception("router crashed on %r", transcript)

    def submit_classification(transcript: str, t_marked: float) -> None:
        utt.classified_text = transcript
        fut = route_pool.submit(classify_and_execute, transcript, t_marked, utt.gen)
        fut.add_done_callback(
            lambda f: f.exception() and log.error("route worker error: %r", f.exception())
        )

    # --- timed worker wrappers (timings feed the status line) ---

    def _asr_timed(buf: np.ndarray) -> tuple[str, float]:
        """Runs on the asr worker."""
        t0 = time.perf_counter()
        return asr.add_frame(buf), time.perf_counter() - t0

    def _step_timed(mic_frame: np.ndarray) -> tuple[tuple[np.ndarray | None, str], float]:
        """Runs on the engine worker."""
        t0 = time.perf_counter()
        return engine.step(mic_frame), time.perf_counter() - t0

    def _finish_and_reset() -> str:
        """Runs on the asr worker: ALL transcriber calls stay on that one
        thread (thread affinity, module docstring)."""
        final = asr.finish()
        asr.reset()
        return final

    def _synthesize_timed(text: str) -> tuple[np.ndarray, float]:
        """Runs on the tts worker."""
        t0 = time.perf_counter()
        return synth.synthesize(text), time.perf_counter() - t0

    def _inject_synthesized(f: "asyncio.Future") -> None:
        """Loop-thread done-callback: hand finished briefing PCM to the engine
        worker. Failures become log lines, never silent."""
        if f.exception():
            log.error("briefing tts error: %r", f.exception())
            return
        pcm, dur = f.result()
        log.info("briefing synthesized in %.0fms (%.1fs of audio); queueing inject",
                 dur * 1000, pcm.shape[-1] / Config.SAMPLE_RATE)
        inj_fut = loop.run_in_executor(engine_pool, engine.inject_audio, pcm)
        inj_fut.add_done_callback(
            lambda g: g.exception() and log.error("inject error: %r", g.exception())
        )

    log.info("audio defaults: device=%s input=%s output=%s",
             sd.default.device, _describe_device("input"), _describe_device("output"))

    # diagnostics window (reset each status report)
    win_frames = 0
    win_start = time.perf_counter()
    win_max_rms = 0.0
    win_asr_s = 0.0
    win_step_s = 0.0

    partial = ""

    with AudioIO() as audio:
        log.info("session live - speak")
        while True:
            try:
                mic = audio.mic_frames.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.002)
                continue

            # bounded catch-up: only during silence (gate off) - dropping
            # mid-utterance corrupts the transcript; dropped silence is just
            # room tone the engine never misses.
            if audio.mic_frames.qsize() > CATCHUP_TRIGGER_FRAMES and not gate.active:
                dropped = 0
                while audio.mic_frames.qsize() > CATCHUP_TARGET_FRAMES:
                    try:
                        audio.mic_frames.get_nowait()
                    except queue.Empty:
                        break
                    dropped += 1
                log.warning("catch-up: dropped %d silent mic frames (~%.1fs of drift)",
                            dropped, dropped * 0.08)

            win_max_rms = max(win_max_rms, float(np.sqrt(np.mean(mic ** 2))))

            # VAD first (cheap RMS), then the gate decides whether ASR runs
            # at all this frame; gated-off frames cost zero ASR work (P0.1
            # transcript tap now only runs around speech).
            event = vad.feed(mic)
            asr_buf = gate.feed(mic, event, vad.in_speech)

            # drain injections: briefing text -> TTS on the tts worker ->
            # finished PCM -> engine.inject_audio on the engine worker. The
            # frame loop never blocks on synthesis.
            briefing = injections.get()
            if briefing is not None:
                synth_fut = loop.run_in_executor(tts_pool, _synthesize_timed, briefing)
                synth_fut.add_done_callback(_inject_synthesized)

            # ASR (if gated on) and the model step run CONCURRENTLY on their
            # workers: per-frame cost is max(asr, step), not the sum.
            step_fut = loop.run_in_executor(engine_pool, _step_timed, mic)
            if asr_buf is not None:
                (partial, asr_s), ((audio_out, text), step_s) = await asyncio.gather(
                    loop.run_in_executor(asr_pool, _asr_timed, asr_buf), step_fut
                )
                win_asr_s += asr_s
            else:
                (audio_out, text), step_s = await step_fut
            win_step_s += step_s

            # VAD events are handled AFTER the gather so maybe_end classifies
            # on the freshest partial (it arrives one gather later than the
            # pre-overlap design, <=80ms).
            if event:
                log.info("vad %s partial=%r", event, partial)
            if event == "speech_start":
                utt.reset()
            elif event == "maybe_end" and partial.strip():
                # classify on the partial BEFORE the utterance is confirmed over
                submit_classification(partial, time.perf_counter())
            elif event == "utterance_end":
                # the gate force-flushed its buffered frame into this frame's
                # add_frame (just awaited), so finish() sees the full hangover tail
                final = await loop.run_in_executor(asr_pool, _finish_and_reset)
                grew = final.strip() and final.strip() != (utt.classified_text or "").strip()
                if grew and utt.executed_gen != utt.gen:
                    submit_classification(final, time.perf_counter())
                partial = ""

            if audio_out is not None:
                audio.speaker_frames.put_nowait(audio_out)
            if text:
                print(text, end="", flush=True)

            win_frames += 1
            if win_frames >= STATUS_EVERY_FRAMES:
                elapsed = time.perf_counter() - win_start
                log.info(
                    "status: micq=%d spkq=%d underruns=%d micRMS=%.6f vad=%s asr_on=%s "
                    "asr_len=%d fps=%.1f asr_ms=%.0f step_ms=%.0f",
                    audio.mic_frames.qsize(), audio.speaker_frames.qsize(),
                    audio.underruns,
                    win_max_rms, vad.in_speech, gate.active, len(partial),
                    win_frames / elapsed if elapsed > 0 else 0.0,
                    win_asr_s / win_frames * 1000, win_step_s / win_frames * 1000,
                )
                win_frames = 0
                win_start = time.perf_counter()
                win_max_rms = 0.0
                win_asr_s = 0.0
                win_step_s = 0.0


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
