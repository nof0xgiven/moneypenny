"""Environment-backed configuration. One object, passed explicitly — no globals."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Config:
    SAMPLE_RATE: ClassVar[int] = 24000
    FRAME: ClassVar[int] = 1920  # samples per 80ms model step
    ASR_SAMPLE_RATE: ClassVar[int] = 16000

    homey_base_url: str
    homey_api_key: str
    weather_lat: float
    weather_lon: float
    router_model: str = "mlx-community/Qwen3-0.6B-4bit"
    asr_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    voice: str = "NATF2"
    quantize_bits: int = 8

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from os.environ (pure read, no .env loading).

        The app entry point (app.main) is responsible for calling
        dotenv.load_dotenv() before invoking this.
        """
        missing = [k for k in ("HOMEY_BASE_URL", "HOMEY_API_KEY") if not os.environ.get(k)]
        if missing:
            raise ValueError(f"missing required env: {', '.join(missing)}")
        return cls(
            homey_base_url=os.environ["HOMEY_BASE_URL"],
            homey_api_key=os.environ["HOMEY_API_KEY"],
            weather_lat=float(os.environ.get("WEATHER_LAT", "25.2048")),
            weather_lon=float(os.environ.get("WEATHER_LON", "55.2708")),
        )
