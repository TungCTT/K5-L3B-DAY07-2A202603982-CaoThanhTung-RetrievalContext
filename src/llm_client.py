"""Minimal OpenAI-compatible LLM client (DeepSeek) using Python stdlib urllib."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    """Raised when the LLM API returns an error or invalid response."""


class DeepSeekClient:
    """OpenAI-compatible chat completion client with tool-calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("MODEL", "deepseek-chat")
        self.base_url = base_url
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY must be set in environment or passed explicitly.")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_exc: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(3):
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (2**attempt))
                    last_exc = exc
                    continue
                raise LLMError(f"HTTP {exc.code}: {body}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.5 * (2**attempt))
                    continue
                raise LLMError(f"Request failed after 3 attempts: {exc}") from exc
        else:
            raise LLMError(f"Request failed: {last_exc}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        choices = raw.get("choices") or []
        if not choices:
            raise LLMError(f"Empty choices in response: {raw}")
        return {
            "message": choices[0].get("message", {}),
            "usage": raw.get("usage", {}),
            "model": raw.get("model", self.model),
            "latency_ms": round(elapsed_ms, 2),
        }
