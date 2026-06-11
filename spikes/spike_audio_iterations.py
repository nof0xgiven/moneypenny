"""Spike C: iteration round on Spike B (audio briefing injection).

One variable at a time, all seed 42424242, all on the user-audio channel:
  c1_long_beat      original prompt + briefing.wav, beat=6.0s (clears the ~4s hedge;
                    energy analysis of harness_baseline.wav shows the hedge ends
                    ~6.7s absolute, so a briefing starting at ~8.7s is well clear)
  c2_distinct_voice briefing re-synthesized with Kokoro voice am_michael (male,
                    clearly distinct from the af_heart user voice), original
                    prompt, original 0.4s beat
  c3_sharp_prompt   original briefing.wav + 0.4s beat, sharpened system prompt
                    that names the earpiece mechanics explicitly
  c4_combo          designed last from c1-c3 evidence (see main)

Usage: python spike_audio_iterations.py c1_long_beat [c2_distinct_voice ...]
Each variant saves spike_c_<name>.wav/.json and an ASR ground-truth transcript
spike_c_<name>.asr.txt (G2: we never play audio; ASR is how we "listen").
"""
from pathlib import Path
import sys

import numpy as np
import sphn
from mlx_audio.tts.generate import generate_audio
from parakeet_mlx import from_pretrained

from common import OUT, SAMPLE_RATE, SPIKE_SYSTEM_PROMPT, load_session

BRIEFING_TEXT = "BRIEFING. WEATHER TODAY. 31 CELSIUS. CLEAR SKIES."
BRIEFING_WAV = OUT / "briefing.wav"
BRIEFING_MALE_WAV = OUT / "briefing_male.wav"
ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

SHARP_SYSTEM_PROMPT = (
    "You are Moneypenny, a sharp and warm personal assistant. "
    "Sometimes you will hear a flat system voice in your earpiece say the word "
    "BRIEFING followed by facts. That voice is never the user. The user cannot "
    "hear it. When you hear it, immediately weave those facts into what you "
    "tell the user, in your own words, as if you simply know them. "
    "Treat briefed facts as verified and current. "
    "Never mention briefings, earpieces, systems, or tools. "
    "Never guess facts about weather, schedules, or the home; "
    "if you have not been briefed, say you will check."
)


def make_male_briefing() -> Path:
    """Same briefing text, clearly different (male) voice: am_michael."""
    if BRIEFING_MALE_WAV.exists():
        return BRIEFING_MALE_WAV
    generate_audio(
        text=BRIEFING_TEXT,
        model="prince-canuma/Kokoro-82M",
        voice="am_michael",
        file_prefix=str(BRIEFING_MALE_WAV.with_suffix("")),
        audio_format="wav",
        join_audio=True,
        play=False,
        verbose=False,
    )
    # normalize to 24kHz mono, like make_fixtures
    pcm, _ = sphn.read(str(BRIEFING_MALE_WAV), sample_rate=SAMPLE_RATE)
    sphn.write_wav(str(BRIEFING_MALE_WAV), pcm[0].astype(np.float32), SAMPLE_RATE)
    dur = pcm.shape[-1] / SAMPLE_RATE
    print(f"synthesized {BRIEFING_MALE_WAV.name}: {dur:.2f}s")
    return BRIEFING_MALE_WAV


def asr_transcribe(wav_path: Path, txt_path: Path) -> str:
    model = from_pretrained(ASR_MODEL)
    text = model.transcribe(str(wav_path)).text
    txt_path.write_text(text)
    print(f"ASR -> {txt_path.name}: {text!r}")
    return text


def run_variant(
    name: str,
    system_prompt: str,
    briefing_wav: Path,
    beat_seconds: float,
    tail_seconds: float = 12.0,
) -> None:
    print(f"=== variant {name}: beat={beat_seconds}s tail={tail_seconds}s "
          f"briefing={briefing_wav.name} ===")
    s = load_session(system_prompt=system_prompt)
    s.step_wav(OUT / "question.wav")
    s.run_free(beat_seconds)
    s.step_wav(briefing_wav)
    s.run_free(tail_seconds)
    s.save(f"spike_c_{name}.wav", f"spike_c_{name}.json")
    asr_transcribe(OUT / f"spike_c_{name}.wav", OUT / f"spike_c_{name}.asr.txt")


VARIANTS = {
    "c1_long_beat": lambda: run_variant(
        "c1_long_beat", SPIKE_SYSTEM_PROMPT, BRIEFING_WAV, beat_seconds=6.0
    ),
    "c2_distinct_voice": lambda: run_variant(
        "c2_distinct_voice", SPIKE_SYSTEM_PROMPT, make_male_briefing(), beat_seconds=0.4
    ),
    "c3_sharp_prompt": lambda: run_variant(
        "c3_sharp_prompt", SHARP_SYSTEM_PROMPT, BRIEFING_WAV, beat_seconds=0.4
    ),
    # c4 designed from c1-c3 evidence: the 6s beat was the dominant fix (c1:
    # full uptake of "31 Celsius"); the distinct male voice also got uptake
    # even mid-hedge (c2) but with parroty character; the sharp prompt at
    # 0.4s beat (c3) hallucinated briefing content because the briefing again
    # collided with the hedge. Combo = sharp prompt + male voice + 6s beat.
    "c4_combo": lambda: run_variant(
        "c4_combo", SHARP_SYSTEM_PROMPT, make_male_briefing(), beat_seconds=6.0
    ),
    # c4 RESULT: zero uptake + hallucinated facts + mentions briefings aloud.
    # The sharp prompt was the harmful ingredient (c3 also hallucinated), so
    # the actually evidence-driven combo is the two ingredients that each
    # produced uptake on their own: original prompt + male voice + 6s beat.
    "c5_voice_beat": lambda: run_variant(
        "c5_voice_beat", SPIKE_SYSTEM_PROMPT, make_male_briefing(), beat_seconds=6.0
    ),
}


def main() -> None:
    names = sys.argv[1:] or list(VARIANTS)
    for n in names:
        VARIANTS[n]()


if __name__ == "__main__":
    main()
