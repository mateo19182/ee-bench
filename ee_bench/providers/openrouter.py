"""OpenRouter provider — single adapter that reaches every model."""

from __future__ import annotations

import httpx


class OpenRouterProvider:

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        """Send a chat completion request. Returns the assistant's text."""
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def close(self):
        self._client.close()
