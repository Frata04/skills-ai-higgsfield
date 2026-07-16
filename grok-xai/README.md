# Grok (xAI) API integration

A small, production-ready Python client for the [xAI Grok API](https://docs.x.ai/),
driven through the official OpenAI SDK pointed at `https://api.x.ai/v1`.

## Setup

```bash
cd grok-xai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set XAI_API_KEY
```

## Usage

```python
from grok import GrokClient

client = GrokClient()
print(client.chat("Explain black holes in one sentence."))
```

Or from the command line:

```bash
python grok.py "Explain black holes in one sentence."
```

## Testing

```bash
pytest test_grok.py                      # offline unit tests (no key needed)
RUN_LIVE_GROK_TEST=1 pytest test_grok.py # also runs one real API call
```

## Files

| File               | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `grok.py`          | The `GrokClient` wrapper + a small CLI entry point.            |
| `test_grok.py`     | Mocked unit tests plus an opt-in live smoke test.              |
| `requirements.txt` | Pinned runtime + dev dependencies.                             |
| `.env.example`     | Template for the `.env` file that holds your `XAI_API_KEY`.    |

The API key is read from the environment (loaded from `.env`) and is never
hardcoded. `.env` is gitignored.
