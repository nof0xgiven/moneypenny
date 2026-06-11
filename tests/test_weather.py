import pytest

from moneypenny.tools.weather import Weather, current_weather, format_weather


@pytest.mark.slow
def test_live_open_meteo():
    w = current_weather(lat=25.2048, lon=55.2708)
    assert -20 < w.temperature_c < 60
    assert w.condition  # non-empty human label


def test_format_weather():
    w = Weather(temperature_c=31.4, condition="clear sky", wind_kmh=12.0)
    s = format_weather(w)
    assert "31" in s and "CLEAR SKY" in s.upper()
    assert len(s.split()) <= 40
