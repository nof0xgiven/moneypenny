import pytest

from moneypenny.briefing import compose, MAX_WORDS


def test_briefing_prefix_and_upper():
    assert compose("briefing", "weather today 31C clear") == "BRIEFING: WEATHER TODAY 31C CLEAR"


def test_correction_prefix():
    assert compose("correction", "it is 31C not 25C").startswith("CORRECTION: ")


def test_offline_prefix():
    assert compose("offline", "brain unavailable, offer to handle later").startswith("OFFLINE: ")


def test_truncates_to_max_words():
    long = "word " * 100
    out = compose("briefing", long)
    # prefix word ("BRIEFING:") + at most MAX_WORDS content words
    assert len(out.split()) <= MAX_WORDS + 1


def test_rejects_unknown_kind():
    with pytest.raises(ValueError):
        compose("poem", "x")


def test_strips_newlines_and_collapses_space():
    assert compose("briefing", "a\n  b\t c") == "BRIEFING: A B C"
