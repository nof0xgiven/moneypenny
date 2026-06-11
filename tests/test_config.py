import pytest

from moneypenny.config import Config


def test_loads_from_env(monkeypatch):
    monkeypatch.setenv("HOMEY_BASE_URL", "http://h.local")
    monkeypatch.setenv("HOMEY_API_KEY", "k")
    monkeypatch.setenv("WEATHER_LAT", "25.1")
    monkeypatch.setenv("WEATHER_LON", "55.2")
    monkeypatch.delenv("BRIEFING_VOICE", raising=False)
    cfg = Config.from_env()
    assert cfg.homey_base_url == "http://h.local"
    assert cfg.weather_lat == pytest.approx(25.1)
    assert cfg.briefing_voice == "am_michael"


def test_missing_homey_yields_none_not_error(monkeypatch):
    monkeypatch.delenv("HOMEY_BASE_URL", raising=False)
    monkeypatch.delenv("HOMEY_API_KEY", raising=False)
    cfg = Config.from_env()
    assert cfg.homey_base_url is None
    assert cfg.homey_api_key is None
    assert cfg.homey_configured is False


def test_homey_configured_true_when_both_set(monkeypatch):
    monkeypatch.setenv("HOMEY_BASE_URL", "http://h.local")
    monkeypatch.setenv("HOMEY_API_KEY", "k")
    assert Config.from_env().homey_configured is True


def test_homey_configured_false_when_one_empty(monkeypatch):
    monkeypatch.setenv("HOMEY_BASE_URL", "http://h.local")
    monkeypatch.setenv("HOMEY_API_KEY", "")
    cfg = Config.from_env()
    assert cfg.homey_api_key is None
    assert cfg.homey_configured is False


def test_voice_env_vars(monkeypatch):
    monkeypatch.setenv("MONEYPENNY_VOICE", "NATM1")
    monkeypatch.setenv("BRIEFING_VOICE", "bm_george")
    cfg = Config.from_env()
    assert cfg.voice == "NATM1"
    assert cfg.briefing_voice == "bm_george"


def test_voice_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("MONEYPENNY_VOICE", raising=False)
    monkeypatch.delenv("BRIEFING_VOICE", raising=False)
    cfg = Config.from_env()
    assert cfg.voice == "NATF2"
    assert cfg.briefing_voice == "am_michael"


def test_vad_rms_threshold_env_var(monkeypatch):
    monkeypatch.setenv("VAD_RMS_THRESHOLD", "0.025")
    assert Config.from_env().vad_rms_threshold == pytest.approx(0.025)


def test_vad_rms_threshold_default(monkeypatch):
    monkeypatch.delenv("VAD_RMS_THRESHOLD", raising=False)
    assert Config.from_env().vad_rms_threshold == pytest.approx(0.01)


def test_frame_constants():
    # protocol invariants (1920 samples = one 80ms Mimi step @ 24kHz), not tunables
    assert Config.SAMPLE_RATE == 24000
    assert Config.FRAME == 1920
