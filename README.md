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

Copy `.env.example` to `.env` and set `HOMEY_BASE_URL` and `HOMEY_API_KEY`.
Then run with headphones on — there is no echo cancellation, so open speakers
will feed the model its own voice:

```bash
.venv/bin/moneypenny
```

Phase 1 acceptance status (test results, fact-bait scores, pending live
checklist) is recorded in `docs/decisions/0002-phase1-acceptance.md`.
