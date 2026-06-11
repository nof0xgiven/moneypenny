import numpy as np
import pytest
import sphn

from moneypenny.asr import StreamingTranscriber, resample_24k_to_16k


def test_resample_ratio():
    x = np.zeros(24000, dtype=np.float32)
    y = resample_24k_to_16k(x)
    assert abs(len(y) - 16000) <= 2


@pytest.mark.slow
def test_streaming_partials_on_fixture():
    """Feed the Kokoro question WAV in 80ms chunks; expect growing partials
    containing the word 'weather' before the audio ends."""
    pcm, _ = sphn.read("spikes/out/question.wav", sample_rate=24000)
    t = StreamingTranscriber()
    partials = []
    frame = 1920
    for i in range(0, pcm.shape[-1], frame):
        text = t.add_frame(pcm[0, i:i + frame])
        if text:
            partials.append(text)
    final = t.finish()
    assert "weather" in final.lower()
    assert any("weather" in p.lower() for p in partials), "no streaming partial contained the keyword"
