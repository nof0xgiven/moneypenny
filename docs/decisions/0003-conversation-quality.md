# 0003 — Conversation quality on open speakers (warm-up, gate, restraint, AEC)

Date: 2026-06-12

Scope: four fixes landed as one bundle after the owner's first live session on
open speakers, plus the bundle's live verification in the sandbox (RØDE
NT-USB+ mic, Mac Studio Speakers, `VAD_RMS_THRESHOLD=0.03`, `ECHO_CANCEL=1`).
Each fix was reviewed and tested individually; this records the failure modes,
the mechanisms, and what the bundle verification actually showed.

## The failure script (owner's live log, speakers, no AEC)

One session produced five distinct failures:

- Session start dropped ~13s of mic frames (cold-start steps at fps ~1.5
  starved the loop; catch-up dumped the backlog) and the greeting was choppy.
- Backchannels spammed the router: "Yeah." / "Oh my god." each burned a
  ~200ms classification, ~11 router calls in 7s during one exchange.
- "I don't mind the function" routed to the timer tool with garbage args and
  injected a clarification briefing — the assistant asked "can you say it
  again?" about a request never made.
- The model's own speech leaked back through the mic as phantom user
  utterances ("I don't have that." routed tier 2); on bad trajectories it
  answered its own echo for half a minute.
- A garbled weather ask ("what the white or is it") classified tier 0 chat,
  so no tool fired and the assistant denied being able to check weather.

## The four fixes

1. **Engine warm-up in `Session.load()`** (`0c92dfd`): 25 throwaway
   `engine.step(None)` calls on the engine worker pay Metal kernel
   compilation before any live frame; `reset_session()` re-primes afterwards.
   Same commit adds router fewshot pairs for backchannel→tier 0 and
   garbled-weather→tier 1.
2. **`ClassifyGate`** (`4a0137c`, `814788c`): a pre-router transcript gate on
   the frame loop — empty/backchannel/duplicate/self-echo filters, fed every
   text piece the model speaks; blocked transcripts emit `gate` events
   ("filtered:" rows in the dashboard feed) instead of burning a router call.
   Tool keywords always pass (a silenced real command is worse than a phantom
   the router can still refuse).
3. **Clarification restraint** (`aefca15`, `625871d`): a tier-1 route whose
   args fail validation briefs "ASK USER TO REPEAT" only when the transcript
   contains a trigger word for that tool; otherwise the route drops silently
   (warning log, `dropped: true` tool event) and the execution claim is
   released so the utterance can still execute on re-classification.
   Router-copied args never count as evidence.
4. **Speex AEC on the mic path** (`5abbd17`, `feb69a4`, `74cb361`): every
   frame the output callback plays is the far-end reference; pairing is
   anchored on hardware timestamps so mic-side sample loss drops stale
   references (counted as `aecslips` in the status line) instead of going
   permanently acausal. `ECHO_CANCEL=0` is the kill switch; a missing dylib
   degrades to pass-through.

## Bundle verification (2026-06-12, sandbox)

Suites: fast `218 passed, 28 deselected` (4.9s; second pass 5.5s); slow
`28 passed, 218 deselected` (113s; second pass 128.6s), process exit 138
after the summary — the documented interpreter-shutdown artifact (decision
0002), not a failure.

Live: `moneypenny-web`, sessions driven over `/ws`, fixtures played
acoustically into the room with `afplay` through the Mac Studio Speakers.
Two independent passes: the first across volumes 65-84, the second at
volume 85 with five start/stop cycles against a single process.
Caveat on the acoustic path: Kokoro-voice fixtures through chassis speakers
are marginal for ASR at threshold 0.03 — 5 of 8 weather-question plays
fragmented or garbled at the mic in the first pass. The second pass
isolated the worst spot to the first ~1s of each play (suppression release
through speech onsets); per-frame AGC-levelled fixtures (target frame RMS
0.18) with the keyword ≥2s into the phrase got full sentences through. A
human at the mic does not have this problem — the owner's original log
transcribed whole garbled sentences. The checks below are judged on what
reached the router and what the system did with it.

| Check | Result | Evidence |
|---|---|---|
| 1. Greeting triggers no VAD | PASS | At volume 65/76: zero `vad` events through the greeting (mic peak 0.021–0.027, under the 0.03 threshold). At volume 84/85 the documented pre-convergence blip appeared: one blip utterance per session start, 1–2 fragments ("Anyway.", "Hello?", "Okay.", "Mm-hmm.") routed tier 0, the rest gate-blocked `backchannel`; zero tool actions in either pass. Post-convergence model speech held micRMS ≤0.028 with zero VAD events through full replies. |
| 2. No catch-up dump at session start | **FAIL** (resolved same day — see "Check 2 resolution" below) | Every `start()`: `catch-up: dropped 121-129 silent mic frames (~9.7-10.3s of drift)` ~11s after "session live", first status `fps=2.0 step_ms=~490`, then locked to 12.5. Recurrence is per-start, not per-process: the second pass ran five start/stop cycles against one process and every start stalled identically (121-123 frames), as did both starts of the first pass's two-session process. That rules out once-per-process compile; the leading candidate is start()-scoped state the warm-up never steps — `reset_session()`'s re-prime runs AFTER the 25 warm steps, so the first live frames after every reset (every stop/start) materialize it; the warm `step(None)` path also never touches real-PCM encode. Warm-up itself works as far as it goes: first warm step 11.3-13.3s, settled 29-37ms, all inside `load()`. |
| 3. Garbled weather ask → tier 1 → briefing → spoken facts | PARTIAL | Routing + pipeline PASS three times: "How's the weather looking?", "Uh the weather looked.", and (second pass, garbled onset) "Everyone is wondering, what is the weather looking like today?" → tier 1 weather (conf 0.95), Open-Meteo briefing `WEATHER NOW 30C THUNDERSTORM WIND 1 KMH` queued 2938ms / 1067ms / 1099ms after the boundary, synthesized in ~470-503ms (4.8s audio), injected; a same-utterance second tier-1 route was correctly blocked by the execution claim. Spoken uptake: 1 of 3 trajectories — the second pass's model first hedged invented facts ("partly cloudy... around 70 degrees"), then corrected to the briefed ones ("the 30 C wind and thunder is normal today"); the first pass's two failed (one acknowledged-then-refused — "Thanks for letting me know. I can't see the weather" — one stayed silent), plus two pre-briefing hallucinations ("It's sunny, mid 70s") when the ask never routed. Uptake is the pre-existing probabilistic risk (0001 risk 1, 0002 limitation 2), not part of this bundle. |
| 4. Backchannel fixture blocked before routing | PASS | "Yeah. Okay cool. Yeah that's right." transcribed faithfully and produced four `gate` blocks (`backchannel` ×3, duplicate re-check ×1) and **zero** route calls; the second pass reproduced it ("Yeah." / "Yeah, that's right." → `backchannel` blocks, zero routes). |
| 5. "I don't mind the function honestly." → no phantom clarification | PASS (router path) | Transcribed "I don't mind the document/fact that honestly." → tier 0 chat (conf 0.95), twice; second pass garbled it to "I don't mind that / the thing / the sound" → tier 0 (conf 0.95) all three. No tool route, no UNCLEAR briefing, no "say it again". The ToolHost restraint path was not exercised live (the router never misrouted it); it is pinned by `tests/test_toolhost.py` (timer and homey variants of the verbatim incident). |
| 6. No phantom routes of the model's own speech | PASS | Across ~14 min of cumulative session time in the first pass and five sessions in the second, every routed transcript traces to fixture playback or a pre-convergence greeting fragment; none match model speech outside that window. The one mid-reply leak ("Right." / "Yeah." during "I don't have that in front of me") was gate-blocked `backchannel`. `aecslips` ticked 0→6 (first pass) and 1-5 per session (second pass) under near+far overlap, with re-convergence each time; no echo runaway followed any slip. The `self_echo` filter itself never had to fire live (AEC + the cheaper filters caught everything first); its behavior is pinned by `tests/test_classify_gate.py`. |

Final status lines before the clean stops — first pass:
`micq=0 spkq=9 underruns=138 aecslips=6 vad=False asr_on=False fps=12.5
asr_ms=0 step_ms=64`; second pass:
`micq=0 spkq=12 underruns=137 aecslips=2 micRMS=0.000965 vad=False
asr_on=False fps=12.5 asr_ms=0 step_ms=66` — underruns flat after the
start-stall burst, fps at budget throughout.

## Check 2 resolution (2026-06-12, same day): the unevaluated re-prime graph

Measured with `tmp/start_stall_probe.py`: a real `Session.load()`, then —
without audio devices — 3 cycles of `reset_session()` + 30 timed
`engine.step(noise)` calls (real 1920-float32 PCM, the `_encode_pcm` path),
all on the engine pool.

**Root cause: MLX laziness.** `reset_session()` → `LmGen.reset_streaming()` +
`step_system_prompts()` (`personaplex_mlx/models/generate.py:312`) builds one
lazy transformer step per voice-prompt frame and system-prompt token (~500
steps) and evaluates nothing — `reset_session()` returned after ~1.7-2.3s of
pure graph BUILDING. The first live `engine.step()` after every start was the
first forced evaluation (`engine.py`, `int(out_text_token[0].item())`) and
paid the entire re-prime compute. The stall was ONE ~10.8s frame, not 25
~490ms frames: the logged `step_ms=~490` was that frame averaged over the
25-frame status window (10.8s + 24×61ms ≈ 12.3s / 25 ≈ 490ms). It recurred
per start by construction — `stop()` ends in `reset_session()` — and was
never kernel compilation (the same graph is rebuilt by every reset). The
"first warm step 11.3-13.3s" of the warm-up was the same mechanism: it forced
the prime graph that `VoiceEngine.__init__` had built lazily.

Probe, before the fix (per cycle: reset; then 30 step times):

- cycle 1: reset 2263ms; first step **10844ms**, rest 56-66ms (median 61ms)
- cycle 2: reset 1712ms; first step **10765ms**, rest 58-66ms (median 62ms)
- cycle 3: reset 1710ms; first step **10801ms**, rest 58-65ms (median 62ms)
- discriminator: `mx.eval(LM state)` straight after a fourth reset took
  **10.68s**, and the first step after it 73ms — the pending graph IS the
  stall.

**Fix** (`engine._prime`, called from `__init__` and `reset_session()`):
after `reset_streaming()` + `step_system_prompts()`, force the graph with
`mx.eval` on exactly the state the next step reads — the token cache,
provided mask, and every layer's KV cache; their dependency closure is the
whole re-prime. The cost now lands in `load()` and `stop()` on the engine
worker, never on a live frame. `mx.eval` is semantically transparent (the
evaluated state is identical to the lazy one), so the seeded regression
trajectory is unchanged.

Probe, after the fix: `reset_session()` 12.4-12.8s (pays its own re-prime);
first step **73 / 77 / 76ms** across the 3 cycles, rest 56-68ms; `mx.eval(LM
state)` after a reset is 0.00s (nothing left pending). Warm-up first step
fell 11.3s → 31ms (construction now evaluates its own prime); `load()` total
~24s → ~35s, absorbing the eval of its final reset.

Live re-verification (same sandbox, `VAD_RMS_THRESHOLD=0.03`, two
start→stop→start cycles against one process over `/ws`): **zero**
`catch-up:` lines in the whole log; first status window `fps=11.9
step_ms=68` and `fps=11.9 step_ms=66`, locked to 12.5 from the second
window; the greeting begins within the first second of "session live".
The relocated cost is visible as `stop()` taking ~14s before its ack
(teardown, serialized by the lifecycle lock — not live audio).

Suites after the fix: fast `218 passed, 29 deselected` (4.9s); slow
`29 passed, 218 deselected` (139s, exit 138 after the summary — the
documented artifact). Pinned by
`tests/test_engine_offline.py::test_reset_session_leaves_engine_hot`
(red pre-fix: first step 11460ms vs 62ms median; green post-fix: 70ms).

## Residual risks

- **Per-start cold start — RESOLVED (Check 2 resolution above).** The stall
  was the unevaluated system-prompt re-prime graph, now forced inside
  `load()`/`stop()`. Trade-off accepted: `stop()` blocks ~12-14s while the
  next conversation's prime is computed and evaluated; a future option is
  snapshotting the evaluated primed state once and restoring it per reset
  (the primed state is deterministic — all prompt-replay tokens are forced).
- **Pre-convergence blip scales with speaker volume.** Quiet/moderate
  speakers: no blip. Loud speakers: 1–2 garbled fragments leak in the first
  seconds of the first utterance and route tier 0. The gate's filler list
  absorbs most fragments but not all ASR renderings ("Mm-hmm." normalizes to
  `mmhmm`, "Hey.", "Anyway." — each costs one harmless router call). The
  gate only protects the router: the model itself still hears the leak
  (full-duplex) and may answer it conversationally ("Can I get your name and
  number?" to a leaked "Okay.").
- **Slip re-convergence.** Each `aecslips` event implies a couple of seconds
  of degraded cancellation while the filter re-anchors; under sustained CPU
  spikes during model speech this window can still leak fragments.
- **Silent-drop trade-off.** A real garbled request that keeps no trigger
  word ("kettle off please" heard without "plug"/"switch") is dropped
  silently rather than clarified; the user's natural retry is the recovery
  path.
- **Gate user-echo limitation.** A user genuinely repeating the model's words
  within the 4s echo window is dropped; tool keywords are the escape hatch.
- **ASR garble can still defeat the weather fewshot.** "How's the weather
  model?" and "How the water looked like." both classified tier 0 — the
  router only recovers garbles that keep a recognizable weather ask.
- **rustymimi 8192-position ceiling is cumulative per process** (sharpens
  0002 limitation 8): `reset_session()` re-primes the LM but not the mimi
  streaming caches, so successive start/stop cycles in one process share the
  ~5.5-minute encode budget; the crash can land mid-session on a later
  start. Restarting the process resets the budget.
