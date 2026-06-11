# Vendored from ~/orca/AvaGeminiMultimodal/reachy/reachy_realtime/homey @ 2026-06-11.
# Framework-free modules only; the Gemini tool layer was NOT vendored
# (moneypenny/tools/homey_adapter.py replaces it).
# Spec deviation note (D6): vendored rather than installed as a package
# dependency because the source repo is unpackaged and would drag
# google-genai + the Reachy SDK into the voice process.
"""Homey integration helpers."""

from .client import HomeyClient, HomeyClientError

__all__ = ["HomeyClient", "HomeyClientError"]
