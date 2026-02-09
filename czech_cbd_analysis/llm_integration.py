"""
llm_integration.py
===================

This module defines helper functions for interacting with the e INFRA CZ AI‑as‑a‑service
platform.  These helpers wrap HTTP requests to the OpenAI‑compatible API provided
by https://chat.ai.e-infra.cz.  Functions include generic chat completions,
translation, summarisation and embedding utilities.  All functions take an
explicit API key and avoid using global state.  Network errors and HTTP
status codes are propagated to the caller.

The module is designed for testability: each function delegates the actual
HTTP request to a private `_post` helper which can be patched during
testing.  Example usage:

    from czech_cbd_analysis.llm_integration import translate, summarise

    api_key = "sk-..."  # provided by the user
    english = translate("Přeložte tento text do angličtiny.", api_key=api_key)
    summary = summarise("This is a long article...", api_key=api_key)

Note that these utilities are synchronous and may block; consider running
them in a background thread if integrating into a web application.
"""

from __future__ import annotations

import os
import requests
from typing import List, Dict, Any

# Base URL for the OpenAI‑compatible API.  The default points to the
# e‟INFRA CZ chat service; override via the LLM_API_BASE_URL environment
# variable if needed (e.g., during testing).
BASE_URL = os.getenv("LLM_API_BASE_URL", "https://chat.ai.e-infra.cz/api/")


def _post(endpoint: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Internal helper to perform a POST request to the API.

    Parameters
    ----------
    endpoint: str
        The API path (starting with a slash) to call, e.g., "/v1/chat/completions".
    payload: dict
        JSON payload for the POST request.
    api_key: str
        Bearer token used to authenticate with the service.

    Returns
    -------
    dict
        Parsed JSON response.  Raises an exception on HTTP errors or
        connection failures.
    """
    if not api_key:
        raise ValueError("An API key must be provided")
    url = BASE_URL.rstrip("/") + endpoint
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def chat_completion(messages: List[Dict[str, str]], api_key: str, model: str = "gpt-oss-120b") -> str:
    """Send a chat completion request and return the model's response.

    Parameters
    ----------
    messages: list of dict
        Conversation messages following the OpenAI chat format (roles: system,
        user, assistant).  The last message should typically be from the user.
    api_key: str
        Bearer token for authentication.
    model: str, optional
        Identifier of the model to use (default: gpt oss‑120b).

    Returns
    -------
    str
        The assistant's reply content.
    """
    payload = {"model": model, "messages": messages}
    resp = _post("/v1/chat/completions", payload, api_key)
    # According to the OpenAI API spec, the response includes a list of choices
    # containing the assistant message.  We return the first choice.
    return resp.get("choices", [{}])[0].get("message", {}).get("content", "")


def translate(text: str, api_key: str, model: str = "gpt-oss-120b") -> str:
    """Translate a Czech or multilingual text to English using the chat API.

    A system prompt instructs the model to perform translation without
    embellishment.  If translation fails or the API is unreachable, an
    exception will be raised.

    Parameters
    ----------
    text: str
        The input text to translate.
    api_key: str
        Bearer token for authentication.
    model: str, optional
        Model identifier (default: gpt oss‑120b).

    Returns
    -------
    str
        The translated text.
    """
    messages = [
        {"role": "system", "content": "You are an assistant that translates text to English. Translate the user message without adding new information."},
        {"role": "user", "content": text},
    ]
    return chat_completion(messages, api_key=api_key, model=model)


def summarise(text: str, api_key: str, model: str = "gpt-oss-120b") -> str:
    """Generate a concise English summary of the provided text.

    The model is instructed via a system prompt to condense the content.

    Parameters
    ----------
    text: str
        Text to summarise.
    api_key: str
        Bearer token for authentication.
    model: str, optional
        Model identifier (default: gpt oss‑120b).

    Returns
    -------
    str
        The summary.
    """
    messages = [
        {"role": "system", "content": "Summarise the following text in a concise and neutral manner."},
        {"role": "user", "content": text},
    ]
    return chat_completion(messages, api_key=api_key, model=model)


def embed(texts: List[str], api_key: str, model: str = "qwen3-embedding-4b") -> List[Dict[str, Any]]:
    """Compute embeddings for a list of texts using an embedding model.

    Parameters
    ----------
    texts: list of str
        Input strings to embed.
    api_key: str
        Bearer token for authentication.
    model: str, optional
        Embedding model identifier (default: qwen3 embedding‑4b).

    Returns
    -------
    list of dict
        Each dict contains an embedding vector and metadata as returned by the API.
    """
    payload = {"model": model, "input": texts}
    resp = _post("/v1/embeddings", payload, api_key)
    return resp.get("data", [])
