"""Moneypenny main loop: the only file that touches real devices.

Flow per 80ms frame:
  mic -> [VAD, ASR, engine.step] ; engine PCM -> speakers
  VAD maybe_end (160ms silence)  -> classify on current ASR partial (P0.1:
                                    classification starts BEFORE the confirmed end)
  VAD utterance_end (640ms)      -> if the final transcript materially grew and
                                    nothing executed yet, re-classify on it
  RouteDecision tier 1 -> ToolHost -> briefing -> engine.inject (on the engine worker;
                                    the engine gates injection on output silence)

Threading model: engine.step AND engine.inject run on a single engine worker
thread (engine is single-threaded by design; inject's TTS stalls the frame loop
~1-2s, accepted in Phase 1); router+tools run on a second worker so
classification never stalls the frame loop. asyncio coordinates via
run_in_executor. Worker exceptions are surfaced via done-callbacks - never
silently swallowed (invariant #2).

Known accepted limitation: if the user resumes speaking after maybe_end and
changes the meaning, an already-executed read-only tool (weather) is harmless;
Homey/timer commands need the full command present in the partial to classify
Tier 1 at all, and escalation bias pushes truncated partials to Tier 2.

Known Phase 1 limitation - speaker-queue latency ratchet: each inject() TTS
stall blocks the engine worker, then the frame loop catches up by stepping
through the backlog faster than realtime, bursting the produced PCM into the
unbounded speaker_frames queue. After catch-up, production and consumption
both run at 12.5 fps again, so the queue depth gained during the stall never
drains: every stall permanently adds its duration to mouth-to-ear latency for
the rest of the session. TODO(phase2): cap/drain speaker_frames on catch-up.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import time

from dotenv import load_dotenv

from moneypenny.asr import StreamingTranscriber
from moneypenny.audio import AudioIO
from moneypenny.briefing import compose
from moneypenny.config import Config
from moneypenny.engine import VoiceEngine
from moneypenny.injection import InjectionQueue
from moneypenny.prompts import FRONT_OF_HOUSE
from moneypenny.router import Router
from moneypenny.tools import ToolHost
from moneypenny.tools.homey_adapter import HomeyAdapter
from moneypenny.tools.timers import TimerService
from moneypenny.vad import EnergyVAD

log = logging.getLogger("moneypenny")


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

    log.info("loading models (engine, router, asr)...")
    engine = VoiceEngine(system_prompt=FRONT_OF_HOUSE, voice=cfg.voice,
                         quantize_bits=cfg.quantize_bits)
    router = Router(model_id=cfg.router_model)
    asr = StreamingTranscriber(model_id=cfg.asr_model)
    vad = EnergyVAD()
    injections = InjectionQueue()
    timers = TimerService(
        on_fire=lambda label: injections.put(compose("briefing", f"TIMER DONE {label.upper()}")),
        loop=loop,
    )
    host = ToolHost(cfg, HomeyAdapter.from_config(cfg), timers)
    log.info("models loaded")

    engine_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")
    route_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="route")
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

    with AudioIO() as audio:
        log.info("session live - speak")
        while True:
            try:
                mic = audio.mic_frames.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.002)
                continue

            # parallel transcript tap (P0.1)
            partial = asr.add_frame(mic)
            event = vad.feed(mic)

            if event == "speech_start":
                utt.reset()
            elif event == "maybe_end" and partial.strip():
                # classify on the partial BEFORE the utterance is confirmed over
                submit_classification(partial, time.perf_counter())
            elif event == "utterance_end":
                final = asr.finish()
                asr.reset()
                grew = final.strip() and final.strip() != (utt.classified_text or "").strip()
                if grew and utt.executed_gen != utt.gen:
                    submit_classification(final, time.perf_counter())

            # drain injections into the engine (inject runs on the engine worker;
            # TTS there stalls the frame loop briefly - accepted phase 1 cost)
            briefing = injections.get()
            if briefing is not None:
                inj_fut = loop.run_in_executor(engine_pool, engine.inject, briefing)
                inj_fut.add_done_callback(
                    lambda f: f.exception() and log.error("inject error: %r", f.exception())
                )

            # one model step, off the event loop thread
            audio_out, text = await loop.run_in_executor(engine_pool, engine.step, mic)
            if audio_out is not None:
                audio.speaker_frames.put_nowait(audio_out)
            if text:
                print(text, end="", flush=True)


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
