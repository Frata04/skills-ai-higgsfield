"""
test_grok.py — Tests for the Grok client in grok.py.

Two layers of testing:

1. Offline unit tests (default): the OpenAI SDK network call is mocked, so
   these run anywhere without an API key or internet access. They verify
   config wiring, message construction, reply parsing, and error mapping.

2. Live smoke test (opt-in): only runs when RUN_LIVE_GROK_TEST=1 and a real
   XAI_API_KEY are present. It makes one real request to the xAI API.

Run all offline tests:
    pytest test_grok.py

Run the live smoke test too:
    RUN_LIVE_GROK_TEST=1 pytest test_grok.py

You can also run this file directly for a quick manual live check:
    python test_grok.py "Say hello in five words."
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from grok import DEFAULT_MODEL, XAI_BASE_URL, GrokClient, GrokError


def _fake_completion(text: str):
    """Build an object shaped like an OpenAI ChatCompletion response."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


# --------------------------------------------------------------------------- #
# Configuration / construction
# --------------------------------------------------------------------------- #

def test_missing_key_raises(monkeypatch):
    """No key in env and none passed -> a clear GrokError, not a crash."""
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    # Stop load_dotenv from repopulating the key from a local .env file.
    with patch("grok.load_dotenv", lambda *a, **k: None):
        with pytest.raises(GrokError):
            GrokClient()


def test_client_uses_xai_base_url():
    """The underlying OpenAI client must point at the xAI endpoint."""
    with patch("grok.OpenAI") as mock_openai:
        GrokClient(api_key="test-key")
        _, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == XAI_BASE_URL
        assert kwargs["api_key"] == "test-key"


def test_default_model():
    """When no model is configured, the default is used."""
    with patch("grok.OpenAI"):
        client = GrokClient(api_key="test-key")
        assert client.model == DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_chat_returns_text():
    """chat() returns the assistant message content."""
    with patch("grok.OpenAI") as mock_openai:
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = _fake_completion("42")

        client = GrokClient(api_key="test-key")
        assert client.chat("What is 6 times 7?") == "42"


def test_chat_builds_system_and_user_messages():
    """chat() forwards a system + user message pair to the API."""
    with patch("grok.OpenAI") as mock_openai:
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = _fake_completion("ok")

        client = GrokClient(api_key="test-key")
        client.chat("hi", system="be terse")

        _, kwargs = instance.chat.completions.create.call_args
        roles = [m["role"] for m in kwargs["messages"]]
        assert roles == ["system", "user"]
        assert kwargs["messages"][0]["content"] == "be terse"
        assert kwargs["messages"][1]["content"] == "hi"


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

def test_empty_choices_raises():
    """A response with no choices is surfaced as GrokError."""
    with patch("grok.OpenAI") as mock_openai:
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = SimpleNamespace(choices=[])

        client = GrokClient(api_key="test-key")
        with pytest.raises(GrokError):
            client.chat("hello")


def test_api_status_error_is_wrapped():
    """Underlying SDK errors are translated into GrokError."""
    from openai import APIStatusError

    with patch("grok.OpenAI") as mock_openai:
        instance = mock_openai.return_value
        # APIStatusError needs a response-like object with a status code.
        fake_response = MagicMock()
        fake_response.status_code = 500
        instance.chat.completions.create.side_effect = APIStatusError(
            "boom", response=fake_response, body=None
        )

        client = GrokClient(api_key="test-key")
        with pytest.raises(GrokError):
            client.chat("hello")


# --------------------------------------------------------------------------- #
# Live smoke test (opt-in)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GROK_TEST") != "1",
    reason="Set RUN_LIVE_GROK_TEST=1 and XAI_API_KEY to run the live test.",
)
def test_live_grok_call():
    """One real round-trip to the xAI API. Requires a valid key + network."""
    client = GrokClient()
    reply = client.chat("Reply with exactly the word: pong")
    assert isinstance(reply, str) and reply.strip() != ""


if __name__ == "__main__":
    # Convenience path: `python test_grok.py "your prompt"` does a live call.
    import sys

    prompt = " ".join(sys.argv[1:]).strip() or "Say hello in five words."
    try:
        client = GrokClient()
        print(client.chat(prompt))
    except GrokError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
