"""Small synchronous HTTP client for Homey Web API calls."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


class HomeyClientError(RuntimeError):
    """Raised when Homey cannot be reached or returns an invalid response."""


class HomeyClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_s: float = 10) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "HomeyClient":
        base_url = os.environ.get("HOMEY_BASE_URL")
        api_key = os.environ.get("HOMEY_API_KEY")
        if not base_url:
            raise HomeyClientError("HOMEY_BASE_URL is required")
        if not api_key:
            raise HomeyClientError("HOMEY_API_KEY is required")

        try:
            timeout_s = float(os.environ.get("HOMEY_TIMEOUT_S", "10"))
        except ValueError as exc:
            raise HomeyClientError("HOMEY_TIMEOUT_S must be a number") from exc

        return cls(base_url=base_url.rstrip("/"), api_key=api_key, timeout_s=timeout_s)

    def get_devices(self):
        payload = self._request_json("GET", "/api/manager/devices/device")
        if not isinstance(payload, dict):
            raise HomeyClientError("Homey devices response was not a JSON object")
        return payload

    def get_zones(self):
        payload = self._request_json("GET", "/api/manager/zones/zone")
        if not isinstance(payload, dict):
            raise HomeyClientError("Homey zones response was not a JSON object")
        return payload

    def get_capability_value(self, device_id: str, capability_id: str):
        return self._request_json(
            "GET",
            self._capability_path(device_id, capability_id),
        )

    def set_capability_value(self, device_id: str, capability_id: str, value):
        return self._request_json(
            "PUT",
            self._capability_path(device_id, capability_id),
            body={"value": value},
        )

    def _request_json(self, method: str, path: str, body=None):
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise HomeyClientError("HOMEY_API_KEY was rejected by Homey") from exc
            raise HomeyClientError(f"Homey request failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise HomeyClientError(f"Homey request failed: {_safe_reason(exc)}") from exc
        except TimeoutError as exc:
            raise HomeyClientError("Homey request timed out") from exc
        except OSError as exc:
            if "timed out" in str(exc).lower():
                raise HomeyClientError("Homey request timed out") from exc
            raise HomeyClientError("Homey request failed") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HomeyClientError("Homey returned invalid JSON") from exc

    def _capability_path(self, device_id: str, capability_id: str) -> str:
        escaped_device_id = urllib.parse.quote(device_id, safe="")
        escaped_capability_id = urllib.parse.quote(capability_id, safe="")
        return (
            "/api/manager/devices/device/"
            f"{escaped_device_id}/capability/{escaped_capability_id}"
        )


def _safe_reason(exc: urllib.error.URLError) -> str:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError):
        return "timed out"
    if reason is None:
        return "connection failed"
    return str(reason)
