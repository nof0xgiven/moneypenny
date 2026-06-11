# 0001 — Injection mechanism (resolves spec D2)

Date: 2026-06-11

Inputs:

- Spike A (commit `5d69def`): `spikes/out/spike_text.{wav,json,asr.txt}`, `spikes/out/spike_text_muted.{wav,json,asr.txt}`
- Spike B (commit `d412908`): `spikes/out/spike_audio.{wav,json,asr.txt}`, `spikes/out/spike_audio_long.{wav,json,asr.txt}`
- Spike C (commits `fdb10dc`/`68d20b4`, seed sweep `8581deb`): `spikes/out/spike_c_{c1_long_beat,c2_distinct_voice,c3_sharp_prompt,c4_combo,c5_voice_beat}.{wav,json,asr.txt}`, `spikes/out/spike_c_{c1,c5}_seed{7,1234,20260611}.{wav,json,asr.txt}`

ASR transcripts of the assistant audio (`*.asr.txt`) are ground truth for all verdicts; model-side JSON transcripts contained token artifacts.

## Comparison

| Criterion | A-free (forced text tokens, audio free) | A-muted (forced text + forced silence audio) | B (naive audio briefing, same voice, 0.4s beat) | C-c5 (audio briefing, distinct voice, post-silence) |
|---|---|---|---|---|
| Fact uptake | FAIL — never says 31/clear; continues the BRIEFING *format* with hallucinated Fahrenheit forecast ("highs in eighties... no chance of rain today") | FAIL — zero briefing facts | FAIL — "I don't have that info right now... I don't have the temperature data yet"; in the 20s run, hallucinated briefing content ("the only thing briefed right now is that there's a new report") | PASS — "Got it. 31 Celsius and clear skies" (both facts); 3/4 seeds in sweep |
| Character (no parroting / illusion intact) | FAIL — treats injected format as a style to continue, not facts to absorb | FAIL — conversational state destroyed | Moot (nothing to parrot) | PARTIAL — residual hedge "I'll share that with you once it's confirmed" |
| Stability (no garble/desync/reset) | FAIL — speech garbles during injection ("be Iving Carly Fucked this yeah it's it's cover E if year off") | FAIL — acts as session reset; model abandons in-flight answer and loops greetings ("Thank you for calling Moneypenny. How can I help you today? Hi there. Welcome to Moneypenny...") | PASS — no garble, no reset; minor false starts while briefing played (full-duplex barge-in) | PASS — all 10+ spike C runs stable; determinism verified byte-identically |
| Injection latency | ~0ms (instant token feed) | ~0ms | 4.58s of briefing audio ≈ 58 frames must be stepped before facts can land | Same audio cost; spike used a fixed 6s beat before injection (production: output-silence gate, see below) |

Per the Phase 0 decision rule (A wins if it passes all three; otherwise B; if all fail → STOP and escalate), **all three initial spikes failed**. The owner was consulted and authorized one iteration round on the audio mechanism (Spike C).

## Spike C — authorized iteration round

Key insight from Spike B: the model is full-duplex and starts answering immediately; the briefing landed mid-hedge. It registered "a briefing happened" but extracted no facts. Spike C varied one lever at a time (seed 42424242):

| Variant | Recipe | Fact uptake | Character | Stability |
|---|---|---|---|---|
| c1_long_beat | original prompt, original voice, 6s beat | PASS — "Just wanted to confirm it's 31 degrees Celsius" | PARTIAL — mid-run turbulence ("Got it. Thanks. I updated.") | PASS |
| c2_distinct_voice | male voice (Kokoro `am_michael`), 0.4s beat | PASS — "Okay, 31 Celsius" then undermines ("recommend checking the weather app... I can't give live updates") | FAIL-ish | PASS |
| c3_sharp_prompt | explicit-mechanics prompt naming BRIEFING, 0.4s beat | FAIL — hallucinated "partly cloudy with a chance of rain"; says "briefing says" aloud (illusion break) | FAIL | PASS |
| c4_combo | sharp prompt + male voice + 6s beat | FAIL — hallucinated "sunny in most areas"; narrates "the briefings are coming through clearly" | FAIL | PASS |
| c5_voice_beat | original prompt + male voice + 6s beat | PASS — "Got it. 31 Celsius and clear skies" (only variant with BOTH facts) | PARTIAL — residual hedge ("I'll share that with you once it's confirmed") | PASS |

Hedge timing verified by RMS energy analysis: the hedge ends ~6.7s absolute, so with the 6s beat the briefing starts ~8.7s — after the model's output has gone quiet. The sharp prompt is actively harmful: naming the mechanism makes the model role-play briefings and hallucinate (c3, c4).

### Seed sweep (c1 + c5 at seeds 7, 1234, 20260611)

- **c5: uptake in 3/4 seeds** — 42424242, 1234 ("I note that it's 31 degrees Celsius"), 20260611 ("Got it, 31 degrees clear, sky's the best bet for outdoor plans"). Failure at seed 7: the model treats the briefing as user speech ("Thanks, Al.") and derails.
- **c1 (same-voice control): 2/4** — the distinct voice demonstrably helps.
- **New finding — pre-briefing hallucination:** at all three new seeds the model confidently invented weather during the 6s beat, before the briefing arrived ("75 degrees, partly cloudy" / "high of 72, low of 58" / "mostly sunny, high 70s"), despite the prompt's never-guess instruction. Seed 42424242's clean hedge was the exception. When c5 then lands the true facts, they arrive as a correction after the user has heard fiction. Hedge discipline is an unsolved prerequisite — but it is a prompt/persona + latency problem, a different lever than injection mechanics. In production the dead air is ~1s (real tool latency), not 6s, and injection should be gated on output-VAD silence rather than a fixed delay.
- All 10+ spike C runs: stability PASS (no garble/desync/reset). Determinism verified byte-identically.

**Decision:** Mechanism B — audio injection on the user channel — with the c5 recipe: (1) briefing text → TTS in a distinct synthetic voice (Kokoro `am_michael` for now; making it flatter/more synthetic is a future lever); (2) injected on the user-audio channel gated on the model's output going silent (production replaces the spike's fixed 6s beat with an output-silence gate); (3) the front-of-house prompt stays vague about the mechanism — never name BRIEFING mechanics explicitly (c3/c4 evidence). Rejected: A-free (text/audio desync plus format-continuation hallucination) and A-muted (session reset). Both structurally fight the model's text-channel semantics.

**Consequences for Phase 1:** `VoiceEngine.inject()` implements TTS → user-channel audio; the forced-text-token path is deleted, not kept as an option (Task 16). Injection draining is gated on model output silence, not a fixed delay.

**Open risks (tracked into Phase 1):**

1. **Uptake is probabilistic** (3/4 seeds for c5) — P0.5 latency work and Task 18 live tests must observe real-world uptake.
2. **Pre-briefing hallucination** during dead air — mitigated by short real-world tool latency (~1s vs the spike's 6s) and Task 12/Task 18 prompt iteration plus the fact-bait gate (P0.4).
3. **"Briefing acknowledged as correction" character pattern** ("Got it...") — prompt-iteration target.
