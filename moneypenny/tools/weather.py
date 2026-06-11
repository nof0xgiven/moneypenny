"""Current conditions from Open-Meteo (free, keyless). Stdlib HTTP only."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

# WMO weather interpretation codes -> short labels (subset; default fallback)
_WMO = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


@dataclass(frozen=True)
class Weather:
    temperature_c: float
    condition: str
    wind_kmh: float


def current_weather(lat: float, lon: float, timeout_s: float = 3.0) -> Weather:
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code,wind_speed_10m",
    })
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        data = json.load(resp)
    cur = data["current"]
    return Weather(
        temperature_c=float(cur["temperature_2m"]),
        condition=_WMO.get(int(cur["weather_code"]), "unknown conditions"),
        wind_kmh=float(cur["wind_speed_10m"]),
    )


def format_weather(w: Weather) -> str:
    return f"WEATHER NOW {round(w.temperature_c)}C {w.condition.upper()} WIND {round(w.wind_kmh)} KMH"
