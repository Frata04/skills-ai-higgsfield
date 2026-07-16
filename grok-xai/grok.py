"""
grok.py — Thin, production-ready client for the xAI Grok API.

The xAI API is OpenAI-compatible, so we drive it with the official OpenAI
Python SDK pointed at xAI's base URL (https://api.x.ai/v1). The API key is
read from the environment (loaded from a local .env file) and is never
hardcoded.

Usage:
    from grok import GrokClient

    client = GrokClient()
    print(client.chat("Explain black holes in one sentence."))
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

# Default xAI configuration. The base URL is fixed by the xAI API; the model
# can be overridden per call or via the GROK_MODEL environment variable.
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4"


class GrokError(Exception):
    """Raised when a Grok API call fails in a way the caller should handle."""


class GrokClient:
    """A small wrapper around the OpenAI SDK configured for xAI Grok.

    Args:
        api_key: Optional explicit key. If omitted, it is read from the
            XAI_API_KEY environment variable (loaded from .env). Passing the
            key here is supported for tests but should not be used to hardcode
            secrets.
        model: Default model to use for requests.
        base_url: The xAI-compatible base URL. Defaults to the public endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = XAI_BASE_URL,
    ) -> None:
        # Load variables from a local .env file if present. This is a no-op
        # when the file does not exist, so it is safe in every environment.
        load_dotenv()

        key = api_key or os.getenv("XAI_API_KEY")
        if not key:
            raise GrokError(
                "Missing API key. Set XAI_API_KEY in your environment or in a "
                ".env file (see .env.example). Never hardcode the key."
            )

        self.model = model or os.getenv("GROK_MODEL") or DEFAULT_MODEL
        self._client = OpenAI(api_key=key, base_url=base_url)

    def chat(
        self,
        prompt: str,
        *,
        system: str = "You are Grok, a helpful assistant.",
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Send a single-turn prompt and return the assistant's reply text.

        Raises GrokError on any API, auth, rate-limit, or connection failure.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat_messages(
            messages, model=model, temperature=temperature, **kwargs
        )

    def chat_messages(
        self,
        messages: Iterable[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Send a full message list (multi-turn) and return the reply text."""
        try:
            response = self._client.chat.completions.create(
                model=model or self.model,
                messages=list(messages),
                temperature=temperature,
                **kwargs,
            )
        except AuthenticationError as exc:
            raise GrokError(
                "Authentication failed. Check that XAI_API_KEY is valid."
            ) from exc
        except RateLimitError as exc:
            raise GrokError(
                "Rate limit or quota exceeded. Back off and retry later."
            ) from exc
        except APIConnectionError as exc:
            raise GrokError(
                "Could not reach the xAI API. Check your network connection."
            ) from exc
        except APIStatusError as exc:
            raise GrokError(
                f"xAI API returned an error (status {exc.status_code}): {exc}"
            ) from exc

        if not response.choices:
            raise GrokError("xAI API returned no choices in the response.")

        content = response.choices[0].message.content
        if content is None:
            raise GrokError("xAI API returned an empty message.")
        return content

    def stream(
        self,
        prompt: str,
        *,
        system: str = "You are Grok, a helpful assistant.",
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """Yield reply text incrementally as the model streams tokens."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            stream = self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except AuthenticationError as exc:
            raise GrokError(
                "Authentication failed. Check that XAI_API_KEY is valid."
            ) from exc
        except RateLimitError as exc:
            raise GrokError(
                "Rate limit or quota exceeded. Back off and retry later."
            ) from exc
        except APIConnectionError as exc:
            raise GrokError(
                "Could not reach the xAI API. Check your network connection."
            ) from exc
        except APIStatusError as exc:
            raise GrokError(
                f"xAI API returned an error (status {exc.status_code}): {exc}"
            ) from exc


def main() -> None:
    """Minimal CLI: read a prompt from argv (or stdin) and print the reply."""
    import sys

    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = input("Prompt: ").strip()
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        sys.exit(1)

    try:
        client = GrokClient()
        print(client.chat(prompt))
    except GrokError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
