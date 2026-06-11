# Project Moneypenny — Instant-Feel Conversational Voice Assistant

**Owner:** Mark Fox · **Status:** Draft v0.1 · **Date:** 2026-06-11
**One-liner:** A full-duplex voice front-end that *feels* like the brain, backed by an invisible router + Hermes agent + Hindsight memory. One voice, zero seams.

---

## 1. Problem Statement

Voice assistants today either feel instant but dumb (S2S models hallucinate, no tools) or smart but laggy (ASR→LLM→TTS cascades with dead air). Nobody ships the illusion: instant conversational presence *and* real agentic capability. We separate mouth from brain and hide the seam entirely.

**Design north star:** a first-time user says "holy crap, how?"

## 2. Goals

- G1: Perceived response latency for reflex queries (weather, lights) ≤ 1 conversational beat (~1.2s from end of user utterance to substantive spoken answer).
- G2: Zero audible seams — 100% of spoken output originates from the PersonaPlex voice. No secondary TTS ever touches the speakers.
- G3: Factual queries answered from tools, not model weights — hallucinated "official voice" answers < 2% in adversarial testing.
- G4: Fire-and-forget tasks reliably tracked: 100% of async tasks land in the ledger and are deliverable (email or spontaneous callback).
- G5: Session continuity — assistant references relevant past context (via Hindsight) without being asked.

## 3. Non-Goals (v1)

- **Native function calling in audio-token space** — nobody has solved it; we route around it.
- **Multi-user / speaker identification** — single primary user (Mark). Diarization is P2.
- **Wake-word / always-on listening** — session-initiated conversations only for v1; proactive wake is P2.
- **Phone/mobile client** — Mac Studio resident only. Remote access is a separate initiative.
- **Voice customization UI** — preset voice (pick one PersonaPlex preset, commit to the character).

## 4. Architecture

```
                       ┌─────────────────────────────────────────┐
 mic ──► PersonaPlex 7B (full-duplex S2S, MLX, 8-bit) ──► speakers
                       │  text transcript stream (live)          │
                       └───────────────┬─────────────────────────┘
                                       │ partial transcript (streaming)
                                       ▼
                              ┌────────────────┐
                              │  ROUTER (fast,  │  intent + tier classification
                              │  dumb, biased   │  on partial transcript
                              │  to escalate)   │
                              └───┬────┬───┬────┘
                       Tier 1     │    │   │     Tier 3
                    (reflex) ◄────┘    │   └────► (fire & forget)
                  simple tools:        │          Hermes async job +
                  Homey, weather,      ▼          task ledger + delivery
                  timers, calendar   Tier 2
                                    (brain)
                                    Hermes Agent API (Mac Studio M3 Ultra)
                                       │
                                       ▼
                              Hindsight (vectorize.io)
                              memory: retrieval + injection + write-back

 INJECTION PATH (the illusion):
 tool/brain result ──► fast TTS (Kokoro) ──► PersonaPlex USER-AUDIO stream
 (model "hears" the briefing in its earpiece, performs it in character)
```

**Core principle:** PersonaPlex is front-of-house. Router is the control room. Hermes is the intelligence agency. Hindsight is institutional memory. The earpiece (audio injection) is the only bridge — and it is never audible to the room.

## 5. Components

| Component | Role | Tech | Notes |
| --- | --- | --- | --- |
| Voice | Full-duplex S2S conversation | PersonaPlex 7B 8-bit, MLX (Soniqo) | ~112ms/step, ~11GB RAM. Fixed system prompt per session. |
| Router | Latency-tier classification on partial transcript | Small fast LLM (local Qwen3 small / Haiku) | Single call, structured output, <300ms budget. Biased to escalate. |
| Reflex tools | Tier 1 direct execution | Homey API, weather, timers, calendar read | Router calls these itself. No Hermes round-trip. |
| Brain | Tier 2/3 reasoning + agentic tasks | Hermes agent API on Mac Studio (when enabled) | Owns research, multi-step, email, anything ambiguous. |
| Memory | Persona, context retrieval, write-back, task ledger | Hindsight (vectorize.io) | Injection at session start; retrieval per query; transcript + task write-back async. |
| Injection TTS | Briefings into the model's ears | Kokoro (fast, robotic is fine) | Output routed to PersonaPlex user-audio stream ONLY. Never speakers. |
| Audio plumbing | Mic/speaker I/O, stream mixing | Core Audio / aggregate device or in-process mixing | Mixes real mic + injection channel into user-audio input. |

## 6. Latency Tiers

| Tier | Examples | Path | Budget (utterance-end → spoken answer) |
| --- | --- | --- | --- |
| 0 — Chat | banter, opinions, stories | PersonaPlex alone, no routing action | native (~instant) |
| 1 — Reflex | "what's the weather", "lights off", "set a timer" | Router → tool → inject | ≤ 1.2s |
| 2 — Brain | "what's on my calendar conflict-wise this week", anything reasoning-y | Router → Hermes → inject | ≤ 6s, masked by natural stall |
| 3 — Fire & forget | "research X and email me", "keep an eye on Y" | Router → Hermes job + ledger → ack injected immediately | ack ≤ 1.5s; delivery async |

**Stall-and-fold:** for Tier 2, the voice model stalls naturally ("hmm, let me look at that—") while the briefing is prepared. Injection lands mid-stall; model folds the answer in. The stall is the latency mask. If budget is blown (>8s), inject a graceful deferral briefing ("TELL USER: STILL CHECKING, WILL FOLLOW UP") and convert to Tier 3.

## 7. Requirements

### P0 — Must have (the illusion doesn't exist without these)

**P0.1 — Live transcript tap.** Router receives PersonaPlex text stream incrementally (word-level), not per-utterance.
- [ ] Partial transcript available to router < 200ms behind audio
- [ ] Router can classify before user finishes speaking

**P0.2 — Tier classification.** Router classifies every user utterance into Tier 0/1/2/3 with structured output.
- [ ] Single model call, < 300ms p95
- [ ] When confidence is low → Tier 2 (escalate to Hermes). Never guess Tier 1.
- [ ] Tier 0 (chat) results in NO action — PersonaPlex just talks

**P0.3 — Audio injection channel.** Tool/brain results are TTS'd and mixed into PersonaPlex's user-audio stream, inaudible to the room.
- [ ] Injection audio never reaches speakers (hard routing guarantee, not volume tricks)
- [ ] Briefing format: terse, structured, prefixed ("BRIEFING: WEATHER 31C CLEAR")
- [ ] Model paraphrases briefing in character (verified by transcript diff vs briefing text)

**P0.4 — Front-of-house system prompt.** PersonaPlex session prompt enforces: never guess facts (home, schedule, people, current events); stall naturally when a check is implied; fold briefings in seamlessly; never mention briefings, tools, or "systems".
- [ ] Adversarial test set (≥50 fact-bait questions): hallucinated factual answer rate < 2%
- [ ] Stall phrases varied — no robotic repetition of the same filler

**P0.5 — Tier 1 reflex tools.** Homey (lights/scenes), weather, timers.
- [ ] End-to-end ≤ 1.2s p90 (utterance end → spoken confirmation begins)
- [ ] Action executes even if voice response fails (action > narration)

**P0.6 — Hermes bridge (Tier 2).** Router forwards query + Hindsight context to Hermes API; response is summarized into a briefing and injected.
- [ ] Hermes response → briefing compression (briefings ≤ ~40 words; long answers split or deferred)
- [ ] Hermes unavailable → graceful briefing ("BRAIN OFFLINE, OFFER TO HANDLE LATER"), never silence
- [ ] Timeout → auto-convert to Tier 3 with ledger entry

**P0.7 — Race-condition guard.** The model must not answer factual questions from weights before injection lands.
- [ ] System prompt stall behavior + adversarial verification (overlaps P0.4)
- [ ] If model answers early AND briefing contradicts it: inject correction briefing ("CORRECTION: ...") — model self-corrects in character

### P1 — Should have (fast follows)

**P1.1 — Hindsight session bootstrap.** Persona + recent relevant context baked into PersonaPlex system prompt at session start (only injection point for the voice model's own context).
**P1.2 — Task ledger + async delivery.** Tier 3 jobs persisted (Hindsight); results delivered by email AND queued for next-session spontaneous callback ("oh — that thing you asked about...").
**P1.3 — Transcript write-back.** Full session transcripts → Hindsight async, for retrieval and the "did you ever send that?" query.
**P1.4 — Per-query memory retrieval.** Router enriches Tier 2 queries with Hindsight retrieval before hitting Hermes.
**P1.5 — Calendar + email read tools at Tier 1.** Common enough to deserve reflex latency.

### P2 — Future (design for, don't build)

**P2.1 — Proactive wake.** Assistant initiates sessions for ledger deliveries.
**P2.2 — Speaker diarization** (multi-user households; per-person memory partitions).
**P2.3 — Barge-in tool cancellation** ("actually never mind" kills in-flight jobs).
**P2.4 — Reachy embodiment** — same stack drives the robot; injection channel becomes its inner monologue.

## 8. Key Design Decisions (locked)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Voice ↔ brain separation | Hard split | Voice = presence, brain = capability. Don't kid ourselves. |
| Seam strategy | Full illusion — audio injection only | All speech from one model/voice. No secondary TTS to speakers, ever. |
| Router disposition | Dumb, fast, escalation-biased | Clever-and-slow loses to dumb-and-instant. Hermes absorbs ambiguity. |
| Injection audio style | Terse robotic briefings | It's an earpiece, not a conversation. Optimize for model comprehension + TTS speed. |
| PersonaPlex quant | 8-bit | 4-bit degrades output; Mac Studio Ultra has headroom to spare. |
| Memory | Hindsight | Session bootstrap + retrieval + write-back + ledger in one system. |

## 9. Success Metrics

- **Illusion integrity:** blind listener cannot identify tool-backed vs native answers (informal n=5 test).
- **Reflex latency:** Tier 1 p90 ≤ 1.2s (measured utterance-end → first word of substantive answer).
- **Hallucination rate:** < 2% on adversarial fact-bait set.
- **Ledger reliability:** 0 lost Tier 3 tasks over 2 weeks of daily use.
- **The real metric:** first demo to Vicky produces an unprompted "how is it doing that?"

## 10. Open Questions

- ~~**[engineering, blocking]** Does Soniqo's PersonaPlex runtime expose a programmatic user-audio input stream mid-session?~~ **ANSWERED — see §12.** API is turn-based (`respond`/`respondStream` take a buffer), which makes injection *easier*: we own the input buffer, so briefings are appended pre-turn. No fork needed.
- ~~**[engineering, blocking]** Hermes API surface — streaming or request/response?~~ **ANSWERED — see §12.** Both, plus native Runs API (async + SSE + stop) and Jobs API (scheduled, with delivery target). Tier 3 ledger is largely native to Hermes.
- **[engineering]** Router model pick: local small Qwen3 (no network, ~free) vs Haiku (smarter, ~200ms network). Benchmark both on the intent set.
- **[design]** Stall vocabulary — how many filler variants before it feels human? Needs listening tests.
- **[engineering]** Mic + injection mixing: in-process sample mixing vs macOS aggregate device. In-process preferred (deterministic, no audio-config fragility). *Note: turn-based API (§12) may eliminate mixing entirely — briefings are buffer appends, not live mixes.*
- **[engineering]** Time-to-first-audio: `firstChunkFrames` defaults to 25 (~2s before first emission). How low can it go before quality/stability suffers? Tune on M3 Ultra with `warmUp()` compiled inference.
- **[engineering]** True continuous full-duplex (model listens while speaking) vs the demo's VAD-segmented turn loop — is a persistent session exposed, or is multi-turn context replayed per call? Affects barge-in (P2.3) and context window burn (3000-token context at 12.5Hz).
- **[product]** Session lifetime — when does a session end and Hindsight bootstrap refresh? Idle timeout vs explicit close.

## 11. Phasing

**Phase 1 — Smallest viable illusion (target: 1–2 weekends)**
Transcript tap → router → weather + Homey lights → Kokoro briefing → audio injection. No Hermes, no memory. Exit criteria: stall-and-fold feels human on Tier 1; P0.3/P0.4/P0.5 pass.

**Phase 2 — Brain online**
Hermes bridge (Tier 2), briefing compression, timeout → Tier 3 conversion, race-condition adversarial hardening. Exit: P0.6/P0.7 pass.

**Phase 3 — Memory & async**
Hindsight bootstrap, task ledger, transcript write-back, email delivery, spontaneous-callback queue. Exit: P1.1–P1.4; "did you ever send that?" works a week later.

**Phase 4 — Polish & expand**
Stall vocabulary tuning, calendar/email reflex tools, proactive wake exploration, Reachy embodiment spike.

**Scope rule:** any addition to a phase requires a removal. The parking lot is §7 P2.

---
*v0.1 — drafted with Ava, 2026-06-11. The voice is the anchor; the control room does the work.*

## 12. Research Findings (2026-06-11)

### 12.1 PersonaPlex / speech-swift — the earpiece is a buffer append, not a stream hack

The Swift API is **turn-based**: `respond(userAudio:)` / `respondStream(userAudio:)` take a complete audio buffer per turn. The PersonaPlexDemo does live conversation via VAD-segmented turns (mic → VAD end-of-utterance → buffer → respondStream), with multi-turn context. RTF ~0.94–1.4 on M2 Max; M3 Ultra + `warmUp()` compiled inference (~30%/step gain) should be comfortably faster than real-time.

**Implications (all favourable):**
- **Injection = pre-turn buffer append.** Tier 1 flow: user utterance captured → router classifies → tool runs (~300ms) → briefing TTS'd → briefing samples appended to the user buffer → ONE respondStream call. The model hears question+briefing as a single turn. No mid-stream surgery, no audio mixing, no fork.
- **Race condition (P0.7) largely dissolves for Tier 1.** The briefing is in the input before generation starts — the model physically cannot answer ahead of it. P0.7's adversarial guard still matters for Tier 0/2 (questions the router misses or slow Hermes turns).
- **Router doesn't need PersonaPlex's transcript.** The stack ships streaming ASR with end-of-utterance detection (Parakeet-EOU-120M, Nemotron streaming). Run it in parallel on the mic feed — router gets words in real time and classifies before the utterance even ends. P0.1 "transcript tap" → "parallel streaming ASR tap." PersonaPlex's own transcript becomes verification/logging.
- **Tier 2 = briefing-only turn.** Turn N: user asks; model (per system prompt) stalls. Hermes works. Turn N+1: input buffer is purely the briefing audio; model delivers the answer in character. Multi-turn context carries it.
- **Tune `firstChunkFrames` down** from 25 (~2s) for time-to-first-audio; benchmark quality floor.
- **Bonus:** wake-word model (KWS Zipformer, 26× real-time) is already in the stack — P2.1 proactive wake is cheaper than assumed. `speech-server` also exposes an OpenAI-compatible `/v1/realtime` WebSocket as an alternative integration surface.

### 12.2 Hermes Agent API — Tier 2/3 infrastructure is mostly native

OpenAI-compatible server on `127.0.0.1:8642` (bearer auth, `API_SERVER_ENABLED=true` in `~/.hermes/.env`). Relevant surfaces:

| Surface | Use in Moneypenny |
| --- | --- |
| `POST /v1/responses` + `previous_response_id` / named `conversation` | Tier 2 brain calls with server-side conversation state — Hermes remembers the thread, we don't replay history |
| `POST /v1/runs` → run_id, `GET /runs/{id}/events` (SSE), `POST /runs/{id}/stop` | Tier 2 with progress visibility + timeout→Tier 3 conversion; stop enables future barge-in cancellation (P2.3) |
| `POST /api/jobs` (prompt, schedule, **delivery target**) | Tier 3 fire-and-forget: "research X and email me" maps ~1:1 to a Hermes job with email delivery. Most of P1.2's ledger is native |
| `X-Hermes-Session-Key` header | Stable memory scope across voice sessions — pass a fixed key so Hermes's own memory persists |
| System prompt layering | Our per-request instructions stack on Hermes's core prompt — briefing-format directives ("answer in ≤40 words for voice") go here |
| `GET /health` | Graceful "BRAIN OFFLINE" briefing trigger (P0.6) |

**Spec adjustments:** P1.2 shrinks — Hindsight remains the *conversational* ledger (so the voice can reference past asks), but job execution, scheduling, and delivery live in Hermes. The architecture diagram's "task ledger" is now: Hermes jobs (execution) + Hindsight (recall).

### 12.3 Revised Phase 1 shape

Mic → Silero VAD + Parakeet-EOU streaming ASR (parallel) → router classifies on partials → Tier 1 tool fires during the tail of the utterance → Kokoro briefing appended to buffer → single PersonaPlex turn → speakers. The illusion's first version doesn't even need the stall — the answer arrives in the same turn. Stall-and-fold becomes the Tier 2 mechanic.

## references
https://github.com/soniqo/speech-swift
https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server
https://hindsight.vectorize.io

NOTE: 
- hindsight is on local already running (dont delete data)
- parakeet is already running on mac here via another app `hex` `/Users/ava/Library/Containers/com.kitlangton.Hex/Data/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v2-coreml`

