"""Spike B: inject briefing as Kokoro audio frames on the user-audio channel."""
from common import load_session, OUT


def main() -> None:
    s = load_session()
    s.step_wav(OUT / "question.wav")
    s.run_free(0.4)
    s.step_wav(OUT / "briefing.wav")
    s.run_free(8.0)
    s.save("spike_audio.wav", "spike_audio.json")


if __name__ == "__main__":
    main()
