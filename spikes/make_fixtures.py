"""Synthesize spike fixture WAVs: a user question and an earpiece briefing.

Output is 24kHz mono WAV (PersonaPlex/Mimi native rate). Kokoro output that
is not 24kHz is resampled here so downstream spikes never resample.
"""
from pathlib import Path

import numpy as np
import sphn
from mlx_audio.tts.generate import generate_audio

OUT = Path(__file__).parent / "out"
QUESTION_TEXT = "Hey, what's the weather looking like today?"
BRIEFING_TEXT = "BRIEFING. WEATHER TODAY. 31 CELSIUS. CLEAR SKIES."

SAMPLE_RATE = 24000


def _synth(text: str, path: Path) -> None:
    generate_audio(
        text=text,
        model="prince-canuma/Kokoro-82M",
        voice="af_heart",
        file_prefix=str(path.with_suffix("")),
        audio_format="wav",
        join_audio=True,
        play=False,
        verbose=False,
    )
    # generate_audio writes <prefix>.wav; normalize to 24kHz mono
    pcm, sr = sphn.read(str(path), sample_rate=SAMPLE_RATE)
    sphn.write_wav(str(path), pcm[0].astype(np.float32), SAMPLE_RATE)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    _synth(QUESTION_TEXT, OUT / "question.wav")
    _synth(BRIEFING_TEXT, OUT / "briefing.wav")
    for name in ("question.wav", "briefing.wav"):
        pcm, _ = sphn.read(str(OUT / name), sample_rate=SAMPLE_RATE)
        dur = pcm.shape[-1] / SAMPLE_RATE
        print(f"{name}: {dur:.2f}s")


if __name__ == "__main__":
    main()
