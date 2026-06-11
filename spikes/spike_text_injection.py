"""Spike A: inject briefing as forced text tokens mid-session, two variants.

Risk being tested: forcing the text channel mid-stream may desync the model's
text/audio alignment or be parroted verbatim instead of folded in.
"""
from personaplex_mlx.persona_utils import wrap_with_system_tags

from common import load_session, OUT

BRIEFING = "BRIEFING: WEATHER TODAY 31 CELSIUS CLEAR SKIES"


def run(mute_assistant: bool, wav: str, txt: str) -> None:
    s = load_session()
    s.step_wav(OUT / "question.wav")
    # short beat, as a real tool call would take
    s.run_free(0.4)
    for tok in s.text_tokenizer.encode(wrap_with_system_tags(BRIEFING)):
        s.step_sine(text_token=int(tok), mute_assistant=mute_assistant)
    s.run_free(8.0)
    s.save(wav, txt)


def main() -> None:
    run(mute_assistant=False, wav="spike_text.wav", txt="spike_text.json")
    run(mute_assistant=True, wav="spike_text_muted.wav", txt="spike_text_muted.json")


if __name__ == "__main__":
    main()
