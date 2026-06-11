# Moneypenny

Moneypenny is a local voice assistant built on PersonaPlex MLX. A full-duplex
speech model carries the conversation while a tier router classifies each
utterance from streaming ASR partials; Tier 1 tool results (weather, Homey
devices, timers) are fed back to the model as synthesized "earpiece briefings"
on the user audio channel, gated on output silence, so the model speaks the
facts in character. The design and requirements live in
`docs/moneypenny-spec.md`; architecture decisions live in `docs/decisions/`.

## Setup

Requires Python 3.12 and Apple Silicon (MLX).

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# personaplex-mlx is a local editable install, not a PyPI package:
pip install -e ~/orca/personaplex-mlx
# If its declared dependencies conflict in your environment, install it with
# --no-deps instead; this repo's pyproject already declares everything the
# engine imports (rustymimi, sentencepiece, sphn, mlx-audio, ...):
#   pip install -e ~/orca/personaplex-mlx --no-deps

pip install -e ".[dev]"
```

## Fixtures

Slow tests replay synthesized WAV fixtures. Generate them once before running
the slow suite:

```bash
python spikes/make_fixtures.py
```

## Tests

```bash
pytest          # fast suite (pure logic; default, slow tests deselected)
pytest -m slow  # slow suite (loads models / hits live services)
pytest -m ""    # everything
```

## Live run

Copy `.env.example` to `.env`. `HOMEY_BASE_URL` and `HOMEY_API_KEY` are
optional — without them the app runs with home control disabled (home-control
commands get a spoken "not set up" response). Then run with headphones on —
there is no echo cancellation, so open speakers will feed the model its own
voice:

```bash
.venv/bin/moneypenny
```

## Choosing voices

Two independent voices, selected via environment variables:

- `MONEYPENNY_VOICE` (default `NATF2`) — the PersonaPlex voice prompt:
  Moneypenny's own voice, the one **you hear** on the speakers. Legal values
  (the voice prompts shipped with `nvidia/personaplex-7b-v1`): `NATF0` `NATF1`
  `NATF2` `NATF3` `NATM0` `NATM1` `NATM2` `NATM3` `VARF0` `VARF1` `VARF2`
  `VARF3` `VARF4` `VARM0` `VARM1` `VARM2` `VARM3` `VARM4` (NAT = natural,
  VAR = varied; F/M = female/male).
- `BRIEFING_VOICE` (default `am_michael`) — the Kokoro voice used to TTS tool
  briefings onto the model's user-audio channel: **only the model hears it**,
  never the speakers. `am_michael` is the spike-proven value; any other Kokoro
  voice (e.g. `af_heart`, `bm_george`, `bf_emma`, `am_adam`) is untested
  territory here. A typo'd voice fails fast: startup runs a one-word synthesis
  probe and aborts with an error naming the voice, rather than failing
  silently at the first briefing. Whatever you pick, keep it clearly distinct from the actual
  user's voice — distinctness is what stops the model treating briefings as
  user speech (see `docs/decisions/0001-injection-mechanism.md`).

Phase 1 acceptance status (test results, fact-bait scores, pending live
checklist) is recorded in `docs/decisions/0002-phase1-acceptance.md`.
