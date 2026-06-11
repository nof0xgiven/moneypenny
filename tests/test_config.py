import pytest

from moneypenny.config import Config


def test_loads_from_env(monkeypatch):
    monkeypatch.setenv("HOMEY_BASE_URL", "http://h.local")
    monkeypatch.setenv("HOMEY_API_KEY", "k")
    monkeypatch.setenv("WEATHER_LAT", "25.1")
    monkeypatch.setenv("WEATHER_LON", "55.2")
    cfg = Config.from_env()
    assert cfg.homey_base_url == "http://h.local"
    assert cfg.weather_lat == pytest.approx(25.1)
    assert cfg.briefing_voice == "am_michael"


def test_missing_required_raises(monkeypatch):
    monkeypatch.delenv("HOMEY_BASE_URL", raising=False)
    monkeypatch.delenv("HOMEY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="HOMEY_BASE_URL"):
        Config.from_env()


def test_frame_constants():
    # protocol invariants (1920 samples = one 80ms Mimi step @ 24kHz), not tunables
    assert Config.SAMPLE_RATE == 24000
    assert Config.FRAME == 1920
