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

    weather_lat: float
    weather_lon: float
    homey_base_url: str | None = None
    homey_api_key: str | None = None
    router_model: str = "mlx-community/Qwen3-0.6B-4bit"
    asr_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    voice: str = "NATF2"
    briefing_voice: str = "am_michael"
    quantize_bits: int = 8
    # Energy-VAD speech threshold (frame RMS). Room/mic dependent: must sit
    # above the ambient noise floor or the VAD never leaves "speech" (boundary
    # events stop firing and the ASR gate never closes).
    vad_rms_threshold: float = 0.01

    @property
    def homey_configured(self) -> bool:
        """True only when both Homey vars are present and non-empty."""
        return bool(self.homey_base_url) and bool(self.homey_api_key)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from os.environ (pure read, no .env loading).

        The app entry point (app.main) is responsible for calling
        dotenv.load_dotenv() before invoking this. Homey vars are optional:
        the app runs without home control when they are unset.
        """
        return cls(
            homey_base_url=os.environ.get("HOMEY_BASE_URL") or None,
            homey_api_key=os.environ.get("HOMEY_API_KEY") or None,
            weather_lat=float(os.environ.get("WEATHER_LAT", "25.2048")),
            weather_lon=float(os.environ.get("WEATHER_LON", "55.2708")),
            voice=os.environ.get("MONEYPENNY_VOICE", "NATF2"),
            briefing_voice=os.environ.get("BRIEFING_VOICE", "am_michael"),
            vad_rms_threshold=float(os.environ.get("VAD_RMS_THRESHOLD", "0.01")),
        )
