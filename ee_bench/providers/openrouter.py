"""OpenRouter provider — single adapter that reaches every model."""

from __future__ import annotations

import time

import httpx

# Status codes worth retrying (rate limit + server errors)
_RETRYABLE = {429, 500, 502, 503, 504}


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
        max_retries: int = 8,
        base_delay: float = 2.0,
    ) -> str:
        """Send a chat completion request with exponential backoff on rate limits."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)

                if resp.status_code in _RETRYABLE:
                    delay = _retry_delay(resp, attempt, base_delay)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                delay = base_delay * (2 ** attempt)
                time.sleep(min(delay, 120))
                continue

        raise RuntimeError(
            f"Failed after {max_retries + 1} attempts: {last_exc or 'rate limited'}"
        )

    def close(self):
        self._client.close()


def _retry_delay(resp: httpx.Response, attempt: int, base_delay: float) -> float:
    """Compute retry delay, respecting Retry-After header if present."""
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return min(base_delay * (2 ** attempt), 120)
