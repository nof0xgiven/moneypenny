"""ClassifyGate: pre-router transcript filters (pure logic, no models).

The quoted strings are verbatim from the owner's live session log (2026-06):
backchannel like "Yeah." burned a ~200ms route-worker classification each
(~11 router calls in 7s during one exchange), the same partial was classified
twice back-to-back, and — on speakers, no AEC yet — the model's own speech
leaked into the mic and came back as phantom "user" utterances that escalated
tier 2 and can trigger tools.
"""
import pytest

from moneypenny.classify_gate import ClassifyGate

# What the model actually spoke (its text streams from engine.step), and what
# ASR then transcribed off the speaker leak ~200ms later.
MODEL_LEAK = "I don havet that in front of me. Can you say it again?"


# --- filter 1: empty ---


def test_no_token_transcripts_block_as_empty():
    g = ClassifyGate()
    assert g.should_classify("", 0.0) == (False, "empty")
    assert g.should_classify("   ", 0.0) == (False, "empty")
    assert g.should_classify("...", 0.0) == (False, "empty")  # punctuation only


# --- filter 2: backchannel ---


@pytest.mark.parametrize("transcript", [
    "Yeah.",
    "Oh.",
    "Uh uh.",
    "Yeah, that's right.",
    "Yeah, I know.",
    "Oh my god.",
])
def test_pure_backchannel_blocks(transcript):
    g = ClassifyGate()
    assert g.should_classify(transcript, 0.0) == (False, "backchannel")


@pytest.mark.parametrize("transcript", [
    "lights off",
    "what's the weather",
    "stop the timer",
    "set a timer for five minutes",
    "uh I just wondered what the weather is",
    "Yeah, reviving your brain.",  # mid-sentence garbage: the router's job, not the gate's
])
def test_one_content_word_defeats_backchannel(transcript):
    g = ClassifyGate()
    assert g.should_classify(transcript, 0.0) == (True, "ok")


# --- filter 3: duplicate (within one utterance) ---


def test_repeat_of_last_allowed_blocks_until_reset():
    g = ClassifyGate()
    assert g.should_classify("turn the lights off", 0.0) == (True, "ok")
    # second maybe_end with no transcript change (the log's double-classify)
    assert g.should_classify("turn the lights off", 0.2) == (False, "duplicate")
    # normalized comparison: case/punctuation drift is still the same text
    assert g.should_classify("Turn the lights off.", 0.4) == (False, "duplicate")
    g.reset_utterance()
    assert g.should_classify("turn the lights off", 5.0) == (True, "ok")


def test_blocked_transcripts_do_not_become_dedupe_memory():
    g = ClassifyGate()
    assert g.should_classify("turn the lights off", 0.0) == (True, "ok")
    assert g.should_classify("Yeah.", 0.1) == (False, "backchannel")
    # the memory still holds the lights command, not the blocked backchannel
    assert g.should_classify("turn the lights off", 0.2) == (False, "duplicate")


def test_grown_transcript_in_same_utterance_is_allowed():
    g = ClassifyGate()
    assert g.should_classify("turn the lights off", 0.0) == (True, "ok")
    assert g.should_classify("turn the lights off in the kitchen", 0.3) == (True, "ok")


# --- filter 4: self-echo ---


def test_production_phantom_blocks():
    g = ClassifyGate()
    g.note_model_text(MODEL_LEAK, 100.0)
    assert g.should_classify("I don't have that.", 100.2) == (False, "self_echo")


def test_mangled_phantom_still_blocks():
    # ASR mangled the tail ("channel" for the "front of me" region); the
    # per-transcript-token overlap (3/4 here) still clears the threshold.
    g = ClassifyGate()
    g.note_model_text(MODEL_LEAK, 100.0)
    assert g.should_classify("I don't have that channel.", 100.2) == (False, "self_echo")


def test_unrelated_question_passes_despite_recent_model_speech():
    g = ClassifyGate()
    g.note_model_text(MODEL_LEAK, 100.0)
    assert g.should_classify("what's the weather like", 100.2) == (True, "ok")


def test_partial_overlap_below_threshold_passes():
    # The log's third phantom shares only "can't"/"that" with MODEL_LEAK
    # (2/5 content tokens): against THIS model text it passes — documented
    # limitation, the gate is deliberately conservative; AEC is the real fix.
    g = ClassifyGate()
    g.note_model_text(MODEL_LEAK, 100.0)
    assert g.should_classify("I can't check that thing right now.", 100.3) == (True, "ok")


def test_subword_pieces_accumulate_into_echo_evidence():
    # engine.step streams text pieces; a word can arrive split ("chan"+"nel").
    # The >=3-char prefix rule lets fragments still count as overlap.
    g = ClassifyGate()
    for i, piece in enumerate([" I", " can", "not", " find", " that", " chan", "nel"]):
        g.note_model_text(piece, 100.0 + i * 0.08)
    assert g.should_classify("Cannot find that channel.", 101.0) == (False, "self_echo")


def test_echo_window_expires():
    g = ClassifyGate(echo_window_s=10.0)
    g.note_model_text("zebra quartz banjo", 100.0)
    assert g.should_classify("zebra quartz banjo", 109.5) == (False, "self_echo")
    # same words, but the model said them >10s ago: no longer echo evidence
    assert g.should_classify("zebra quartz banjo", 110.5) == (True, "ok")


def test_default_echo_window_is_four_seconds():
    # The phantom transcribes ~200ms-2s after the model speaks; 4s covers the
    # real path with margin while quartering the false-block surface a 10s
    # window left on legitimate user speech.
    g = ClassifyGate()
    g.note_model_text("zebra quartz banjo", 100.0)
    assert g.should_classify("zebra quartz banjo", 103.9) == (False, "self_echo")
    assert g.should_classify("zebra quartz banjo", 104.5) == (True, "ok")


def test_tool_keyword_escapes_self_echo():
    """Reviewer case: the model itself just said "turn"/"lights", then the
    user gives a real command built from the same words. Tool intent must
    always reach the router — a falsely silenced command is worse than a
    phantom the router can still refuse. (Scoped to self_echo: a duplicate
    already reached the router once, so it still dedupes.)"""
    g = ClassifyGate()
    g.note_model_text("Sure, I can turn the lights off for you.", 100.0)
    assert g.should_classify("yeah turn the lights off", 100.8) == (True, "ok")


def test_tool_keyword_escape_covers_timer_echo():
    g = ClassifyGate()
    g.note_model_text("I can set a timer for five minutes if you like.", 50.0)
    assert g.should_classify("set a timer for five minutes", 51.0) == (True, "ok")


def test_echo_needs_two_content_tokens():
    g = ClassifyGate()
    g.note_model_text("weather", 0.0)
    # 1/1 overlap but a single content token is too little signal to block on
    assert g.should_classify("Weather.", 0.1) == (True, "ok")


def test_overlap_threshold_is_inclusive():
    g = ClassifyGate(echo_overlap_threshold=0.5)
    g.note_model_text("alpha gamma", 0.0)
    assert g.should_classify("alpha beta", 0.1) == (False, "self_echo")  # 1/2 == threshold
    assert g.should_classify("delta beta", 0.2) == (True, "ok")          # 0/2 < threshold


def test_overlap_consumes_model_words_as_a_multiset():
    g = ClassifyGate()
    g.note_model_text("play", 0.0)
    # two transcript "play" tokens vs ONE model "play": only one matches (1/2)
    assert g.should_classify("play play", 0.1) == (True, "ok")
    g.note_model_text("play", 0.2)
    g.reset_utterance()  # clear the dedupe memory, keep the echo window
    assert g.should_classify("play play", 0.3) == (False, "self_echo")  # 2/2 now


# --- ordering and state interaction ---


def test_backchannel_wins_over_self_echo():
    g = ClassifyGate()
    g.note_model_text("yeah right okay", 0.0)
    assert g.should_classify("Yeah, right, okay.", 0.1) == (False, "backchannel")


def test_duplicate_wins_over_self_echo():
    g = ClassifyGate()
    assert g.should_classify("play some jazz music", 0.0) == (True, "ok")
    g.note_model_text("play some jazz music", 1.0)  # model echoes the command back
    assert g.should_classify("play some jazz music", 1.5) == (False, "duplicate")


def test_reset_utterance_keeps_echo_window():
    g = ClassifyGate()
    g.note_model_text(MODEL_LEAK, 100.0)
    g.reset_utterance()  # the phantom arrives as a NEW utterance (speech_start fired)
    assert g.should_classify("I don't have that.", 100.5) == (False, "self_echo")


def test_note_model_text_tolerates_empty_pieces():
    g = ClassifyGate()
    g.note_model_text("", 0.0)
    g.note_model_text("  ", 0.1)
    assert g.should_classify("what's the weather like", 0.2) == (True, "ok")


def test_live_session_interleaving():
    """The production sequence end to end: model speaks over the speakers,
    the leak transcribes as a phantom (blocked), the user then asks a real
    question (routes once, dedupes on the repeat maybe_end)."""
    g = ClassifyGate()
    g.note_model_text("I don havet that in front of me.", 100.0)
    g.note_model_text(" Can you say it again?", 100.6)
    g.reset_utterance()  # VAD speech_start fired on the leaked audio itself
    assert g.should_classify("I don't have that.", 100.9) == (False, "self_echo")
    g.reset_utterance()  # real user speech begins
    assert g.should_classify("What's the weather like?", 104.0) == (True, "ok")
    assert g.should_classify("What's the weather like?", 104.2) == (False, "duplicate")
