"""Thin, optional Gemini client shared by the AI features.

Everything here is lazy: PyImpact's static analysis never imports google-genai,
and the AI features fail with a clear, actionable message (``LLMError``) when the
SDK or an API key is missing instead of a stack trace.

The API key is read from ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``. If neither
variable is set at all, a ``.env`` file in the current directory is consulted
(only these keys are read from it; nothing else in the file is touched).
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_DOTENV_KEYS = (*KEY_VARS, "GEMINI_MODEL", "GEMINI_EMBED_MODEL")


class LLMError(RuntimeError):
    """A user-facing problem with the AI backend (no key, no SDK, API error)."""


def load_dotenv(path: str = ".env") -> None:
    """Populate missing Gemini settings from a ``.env`` file, if present.

    Only consulted when no key variable is set *at all* — an explicitly empty
    variable means "no key", which keeps tests and CI hermetic.
    """
    if any(var in os.environ for var in KEY_VARS) or not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key in _DOTENV_KEYS and value:
            os.environ.setdefault(key, value)


def get_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    load_dotenv()
    for var in KEY_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


def text_model(model: str | None = None) -> str:
    return model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def embed_model(model: str | None = None) -> str:
    return model or os.environ.get("GEMINI_EMBED_MODEL") or DEFAULT_EMBED_MODEL


def _client(api_key: str | None):
    key = get_api_key(api_key)
    if not key:
        raise LLMError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY), or put "
            "GEMINI_API_KEY=... in a .env file. Free keys: https://aistudio.google.com/apikey"
        )
    try:
        from google import genai
    except ImportError as exc:
        raise LLMError("The AI features need google-genai: pip install 'pyimpact-cli[ai]'") from exc
    return genai.Client(api_key=key)


def _friendly(exc: Exception, model: str) -> str:
    msg = " ".join(str(exc).split())[:300]
    low = msg.lower()
    if "404" in msg or "not found" in low:
        return (f"Gemini model '{model}' is not available ({msg}). "
                "Choose another with --model or GEMINI_MODEL.")
    if "api key" in low or "401" in msg or "403" in msg or "permission" in low:
        return f"Gemini rejected the API key ({msg})."
    if "429" in msg or "quota" in low or "rate" in low:
        return f"Gemini rate limit or quota reached ({msg}). Try again shortly."
    return f"Gemini request failed: {msg}"


def generate(prompt: str, *, api_key: str | None = None, model: str | None = None,
             json_mode: bool = False) -> str:
    """Single-turn text generation. ``json_mode`` asks for a JSON response."""
    client = _client(api_key)
    from google.genai import types

    name = text_model(model)
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain",
        temperature=0.2,
    )
    try:
        resp = client.models.generate_content(model=name, contents=prompt, config=config)
    except Exception as exc:  # the SDK raises several error types
        raise LLMError(_friendly(exc, name)) from exc
    return resp.text or ""


def embed(texts: list[str], *, api_key: str | None = None, model: str | None = None,
          task_type: str = "RETRIEVAL_DOCUMENT", dimensions: int = 768,
          batch_size: int = 100) -> list[list[float]]:
    """Embed ``texts`` in batches; returns one vector per input, in order."""
    if not texts:
        return []
    client = _client(api_key)
    from google.genai import types

    name = embed_model(model)
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dimensions)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            resp = client.models.embed_content(model=name, contents=batch, config=config)
        except Exception as exc:
            raise LLMError(_friendly(exc, name)) from exc
        vectors.extend(list(e.values) for e in resp.embeddings)
    return vectors
