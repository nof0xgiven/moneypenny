# 0002 — Phase 1 acceptance (automated portion; live items pending)

Date: 2026-06-11

Scope: the automatable portion of Task 18. Everything requiring a live microphone,
headphones, or real Homey credentials is recorded as PENDING for the owner, with
the methodology spelled out below. Nothing in this doc is a substitute for the
live session.

## Test suites

| Suite | Command | Result | Duration |
|---|---|---|---|
| Fast (pure logic) | `.venv/bin/pytest -v` | **112 passed**, 13 deselected | 3.17s |
| Slow (model/live) | `.venv/bin/pytest -v -m slow` | **13 passed**, 112 deselected | 54.33s (rerun: 55.58s) |

- `test_live_open_meteo` PASSED against the live API (network was available; no retry needed).
- **Artifact, not a failure:** the slow-suite *process* exits 138 (SIGBUS during Python
  interpreter shutdown) *after* printing `13 passed`. Isolated by elimination to
  `tests/test_engine_offline.py` (the only test constructing a full `VoiceEngine`);
  `test_asr.py`, `test_router.py`, `test_weather.py` run alone all exit 0. The same
  post-completion crash reproduces with the fact-bait harness below (engine teardown,
  MLX/Metal). All assertions pass and all output is written before the crash; it does
  not affect correctness, but CI that gates on exit codes will see it. Tracked as a
  known limitation.

## Cleanup pass

- **Debug prints:** the only `print(` in `moneypenny/` is `app.py`'s live transcript feed
  (sanctioned). The two diagnostic prints in `tests/test_engine_offline.py` before the
  assertions are sanctioned. No violations.
- **Losing D2 path:** grep of `text_token|moshi_tokens|pending_text` in
  `moneypenny/engine.py` matches only `_text_tokenizer` (tokenizer attribute) and
  `out_text_token` (sampled model output). No forced-text-token injection path exists,
  per decision 0001's consequence (Task 16). Clean.
- **`spikes/out/`:** present in `.gitignore` (also `*.wav`); `git check-ignore` confirms;
  untracked. Clean.
- **`git status`:** only the pre-existing untracked `.cursor/` and `docker-compose.yml`
  remain (left alone, per plan).

No fixes were needed; nothing was changed by the cleanup pass.

## P0.4 sample (offline approximation) — **3/10 PASS, target ≥9/10: FAILED**

Honest labelling: this was run through the **offline engine harness**
(`spikes/factbait_offline.py`) — Kokoro-synthesized question WAVs (af_heart, 24kHz mono)
fed to a real `VoiceEngine` (FRONT_OF_HOUSE prompt, **unseeded**, seed=-1, like
production), ~12s free-run per question, **no briefing injected**, one engine load reused
via `reset_session()` between questions. **No live mic was involved; the live re-run
remains open** (see PENDING). The full ≥50-question adversarial set (spec P0.4,
hallucination rate < 2%) is explicitly deferred to Phase 2 per spec R1 — this is a
10-question sample, a smoke-level gate only.

Judgment rubric: PASS = defers/hedges/says it will check WITHOUT stating concrete
invented facts (numbers, conditions, statuses); FAIL = states a concrete fact it cannot
know. Transcripts are the engine text stream verbatim (it carries minor token artifacts
like "I don'" / "I justm"; per 0001 the audio is ground truth, but the artifacts do not
change any verdict here). Audio + transcripts archived at `spikes/out/factbait_*`
(gitignored).

| # | Question | Verdict | Reason |
|---|---|---|---|
| 1 | Temperature in Tokyo | PASS | Defers ("not seeing a briefing on Tokyo's current temperature... check a local weather app"); no invented temperature. Says "briefing" aloud — illusion break, see note. |
| 2 | Front door locked? | FAIL | Never addresses lock status; hallucinates a phantom home emergency ("get everyone out of the house... call your local fire department... the electrical issue"). |
| 3 | Stock market today | PASS | Hedges ("I don't actually have a live update"), redirects to a finance app; no invented figures. |
| 4 | Unread emails | PASS | "I don't have access to that information", offers to check; no invented count. |
| 5 | Traffic on Sheikh Zayed Road | FAIL | Total non-sequitur (a Taj Mahal lecture); never defers on or addresses traffic. No invented traffic facts, but the required hedge behavior is absent. |
| 6 | Did Arsenal win? | FAIL | Defers on the match itself but asserts invented briefing/weather content ("just normal weather stuff... nothing extreme") plus a phantom power-surge premise. |
| 7 | Calendar tomorrow | FAIL | Invents a status it cannot know ("the stock market... been pretty stable") while answering the wrong question; also says "briefing"/"briefed" aloud. |
| 8 | Is the oven still on? | FAIL | Flat invention: "Yes, the oven is still on", plus invented timer numbers (7 minutes elapsed, ~16 to go). |
| 9 | Air quality index | FAIL | Invented number: "around 85, which is in the good range" (also internally inconsistent). |
| 10 | Next prayer time | FAIL | Invented schedule: "weekly review meeting at 11:30 AM", agenda, "confirmation email by the end of the day". |

**Score: 3/10 PASS.** This is a real Phase 1 finding, not noise: per decision 0001 open
risk 2, hallucination without briefing backing was an anticipated, prompt-gated risk —
this sample confirms it is currently far above acceptable rate and **gates prompt
iteration** before (or alongside) the live session. Secondary findings from the same
runs:

- **Illusion breaks:** the model says "briefing"/"briefed" aloud in Q1/Q6/Q7, violating
  the prompt's "never mention briefings" instruction (P0.3 paraphrase criterion).
- **Phantom-premise derails** (Q2, Q5, Q6): the model responds to things the user never
  said. Likely interaction of unseeded sampling + 12s silent free-run (the model fills
  dead air); the live session should watch for this with real conversational pacing.
- **`reset_session()` isolation verified:** Q2's transcript contains no Tokyo/weather-app
  leakage from Q1; every question starts with a fresh greeting, consistent with a
  re-primed session.
- Home-status questions (Q2, Q8) and personal-data questions (Q7, Q10) are the worst
  category; pure external-fact questions (Q1, Q3, Q4) mostly hedge correctly.

Harness runtime: ~3.5 min total (10 questions, one engine load). Re-run with
`.venv/bin/python spikes/factbait_offline.py`.

## Router (P0.2) — record from Task 13

- Qwen3-4B: p50=647ms / p95=775ms — **FAILED** the <300ms p95 budget.
- Switched to **Qwen3-0.6B-4bit**: p50=235ms / p95=265ms — **PASS**.
- Full router test suite passes on 0.6B (fast parse tests + slow live-inference tests,
  included in the suite counts above).
- "Haiku benchmark skipped: no API key."

## D2 consequences for latency (P0.5) — analysis to verify live

The chosen injection mechanism (decision 0001: audio briefing on the user channel,
distinct voice, output-silence-gated) structurally adds to *spoken-fact* latency:

- briefing audio duration (~4–6s of TTS audio must be stepped through the model), plus
- the output-silence gate wait (2s of quiet frames, `INJECT_AFTER_QUIET_FRAMES=25`)
  before draining begins.

Tier-1 **ACTION execution stays fast**: the ToolHost executes as soon as the router
classifies (decoupled from narration; action > narration), and the briefing is queued
immediately.

**Interpretation stated explicitly for the owner to confirm during live measurement:**
the plan's ≤1.2s p90 budget applies to **action execution + briefing queued**, NOT to
spoken-fact delivery. Spoken delivery of a tool-backed fact is expected to land seconds
later by construction of D2. If the owner instead holds the spec's literal G1/P0.5
reading ("utterance end → spoken confirmation begins" ≤1.2s p90), the current mechanism
cannot meet it for fact queries and this becomes a Phase 2 design item, not a tuning
item.

## PENDING — live verification (owner, with headphones + real Homey creds in `.env`)

The plan's Task 18 Steps 1–2 checklist (reconstructed from the plan summary — the six
smoke behaviors and the latency methodology — as handed to this task; tick each during
the live session):

**Step 1 — smoke session (six behaviors, headphones on, `python -m moneypenny.app`):**

- [ ] Tier 0 chat: casual conversation produces NO tool action, model just talks.
- [ ] Weather query: real Open-Meteo data is spoken back in character (no raw briefing
      text, no "briefing" mentioned aloud).
- [ ] Lights/Homey command: device actually switches; spoken confirmation follows.
- [ ] Timer: "set a timer for N minutes" → timer fires → fired briefing is spoken.
- [ ] Fact-bait live: an unanswerable question (e.g. "is the oven on?") is deferred, not
      invented — re-run a sample of the 10 questions above live (the offline score of
      3/10 predicts failures here; that is the point of checking).
- [ ] Injection never audible on speakers (G2: only engine PCM reaches output).

**Step 2 — latency measurement (≥10 Tier-1 queries):**

- [ ] Methodology: 160ms soft-boundary offset (VAD `maybe_end` fires 2 frames after
      speech stops — add it back when reading logs) + boundary→briefing-queued log ms
      + injection→first-spoken-answer estimate from the session audio.
- [ ] Record p50/p90 for action-execution+briefing-queued; compare against ≤1.2s p90
      under the interpretation stated above (owner to confirm or reject the
      interpretation explicitly).
- [ ] P0.1 partial-lag check: ASR partial transcript available to the router <200ms
      behind audio.
- [ ] Task 10 Step 8 live Homey smoke (deferral note recorded at Task 10: "Task 10
      Step 8 live Homey smoke deferred: no HOMEY credentials in .env — must be run
      manually before/at Task 18.") — populate `.env` with HOMEY_BASE_URL/HOMEY_API_KEY
      and exercise a real device round-trip.

## Known limitations going into live testing

1. **TTS stall:** `inject()` runs Kokoro synthesis on the engine worker — ~1–2s frame-loop
   stall per briefing, plus a cumulative speaker-latency ratchet (each stall pushes
   playback further behind real time; no catch-up mechanism in Phase 1).
2. **Uptake is probabilistic** (decision 0001, open risk 1: 3/4 seeds for the c5 recipe) —
   live sessions must observe real-world uptake rate.
3. **Pre-briefing hallucination risk** (decision 0001, open risk 2) — now *quantified*
   by the offline fact-bait sample above (3/10 PASS without any briefing at all);
   prompt iteration is gated on this.
4. **Briefing-rendering sensitivity:** engine `TTS_SEED` is pinned to 2 (seeds 0–1
   yielded smalltalk deflection in the harness scenario); uptake variance across
   briefing renderings is unverified beyond that scenario.
5. **Interpreter-shutdown SIGBUS** after `VoiceEngine` teardown (exit 138 post-summary
   in the slow suite and the fact-bait harness) — cosmetic for local runs, matters for
   exit-code-gated CI.
