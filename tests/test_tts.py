"""BriefingSynth tests (real Kokoro, no mocks)."""
import numpy as np
import pytest

from moneypenny.tts import SAMPLE_RATE, BriefingSynth


@pytest.mark.slow
def test_briefing_voice_probe_fails_fast_on_bad_voice():
    # generate_audio swallows a bad-voice load error (prints it, writes no
    # wav); construction must still turn that into an error naming the voice.
    with pytest.raises(RuntimeError, match="no_such_voice"):
        BriefingSynth("no_such_voice")


@pytest.mark.slow
def test_synthesize_returns_24k_mono_float32():
    synth = BriefingSynth("am_michael")  # construction probe = good-voice test
    pcm = synth.synthesize("BRIEFING: WEATHER TODAY 31 CELSIUS CLEAR SKIES")
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    # a ~4s briefing: sanity-bound the duration rather than pin a byte count
    assert SAMPLE_RATE * 1 < pcm.shape[-1] < SAMPLE_RATE * 15
