"""DeepSeek transport with explicit proxy selection and bounded request retries."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable

import httpx


class DeepSeekAPI:
    def __init__(self) -> None:
        self.key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not self.key:
            raise ValueError("Fill DEEPSEEK_API_KEY in the root .env")
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        self.timeout = float(os.environ.get("DEEPSEEK_TIMEOUT_SEC", "60"))
        self.max_retries = int(os.environ.get("DEEPSEEK_MAX_RETRIES", "2"))
        trust_env = os.environ.get("DEEPSEEK_TRUST_ENV", "false").strip().lower()
        if trust_env not in ("0", "1", "false", "true", "no", "yes"):
            raise ValueError("DEEPSEEK_TRUST_ENV must be true or false")
        self.trust_env = trust_env in ("1", "true", "yes")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("DEEPSEEK_TIMEOUT_SEC must be positive and finite")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("DEEPSEEK_MAX_RETRIES must be between 0 and 5")

    def complete(self, payload: dict, *, before_attempt: Callable[[], None] | None = None) -> dict:
        """Retry only HTTP requests; callers execute tools after a successful return."""
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            # A retry must still belong to a live, unchanged simulation session.
            # The caller may also replace the observation with its latest frame.
            if before_attempt is not None:
                before_attempt()
            try:
                # A new connection also discards any broken keep-alive socket.
                # trust_env=False avoids implicit macOS system proxies; TLS
                # certificate verification remains enabled.
                with httpx.Client(
                    timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
                    trust_env=self.trust_env,
                ) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.key}"},
                        json={"model": self.model, **payload},
                    )
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {str(exc).replace(self.key, '[REDACTED]')}"
            else:
                if response.is_success:
                    try:
                        body = response.json()
                        message = body["choices"][0]["message"]
                        if not isinstance(message, dict):
                            raise TypeError("message is not an object")
                    except (ValueError, KeyError, IndexError, TypeError):
                        raise RuntimeError("DeepSeek returned an invalid chat-completion response") from None
                    return body
                detail = response.text.replace(self.key, "[REDACTED]")[:1000]
                failure = f"HTTP {response.status_code}: {detail}"
                if response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"DeepSeek {failure}")

            if attempt == attempts:
                raise RuntimeError(
                    f"DeepSeek request failed after {attempts} attempt(s): {failure}. "
                    "Run ./run_hei_robot_agent.sh --check-api to test without the simulator. "
                    "DEEPSEEK_TRUST_ENV=false uses a direct connection; true uses environment/system proxies."
                ) from None
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"[API] Request {attempt}/{attempts} failed: {failure}. "
                f"Retrying in {delay}s; no tool is executed by this retry.",
                flush=True,
            )
            time.sleep(delay)
        raise AssertionError("unreachable")
