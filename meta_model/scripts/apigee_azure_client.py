#!/usr/bin/env python
"""Small helper for Mayo Apigee Azure OpenAI chat-completions endpoint.

No tokens should be committed. Configure either:
- api_key_env: environment variable containing the bearer token; or
- api_key_file / api_key_file_env: text file containing the bearer token.

When api_key_file is used, the token is read on every request, so a long-running
job can continue after you overwrite the token file with a refreshed token.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


def read_bearer_token(model_cfg: dict[str, Any]) -> str:
    token_file = model_cfg.get("api_key_file") or None
    token_file_env = model_cfg.get("api_key_file_env") or None
    if token_file_env and os.getenv(str(token_file_env)):
        token_file = os.getenv(str(token_file_env))
    if token_file:
        token = Path(str(token_file)).read_text().strip()
        if token:
            return token
    api_key_env = model_cfg.get("api_key_env") or "APIGEE_TOKEN"
    token = os.getenv(str(api_key_env), "").strip()
    if not token:
        raise RuntimeError(
            "Missing Apigee bearer token. Set api_key_env, api_key_file, or api_key_file_env in the model config."
        )
    return token


def build_endpoint(model_cfg: dict[str, Any]) -> str:
    apigee_url = str(model_cfg.get("apigee_url") or "https://mcc.apix.mayo.edu").rstrip("/")
    deployment = str(
        model_cfg.get("deployment")
        or model_cfg.get("engine")
        or model_cfg.get("model")
        or "gpt-5.5"
    )
    api_version = str(model_cfg.get("api_version") or "2024-10-21")
    dep = quote(deployment, safe="")
    return f"{apigee_url}/llm-azure-openai/openai/deployments/{dep}/chat/completions?api-version={api_version}"


def extract_text_from_response(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected Apigee/Azure response shape: {response}") from exc
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"}:
                parts.append(str(part.get("text", "")))
        return "".join(parts).strip()
    return (content or "").strip()


def call_apigee_chat(client: Any, model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """Drop-in replacement for the repo's call_chat(client, model_cfg, messages)."""
    endpoint = build_endpoint(model_cfg)
    max_retries = int(model_cfg.get("max_retries", 3))
    retry_sleep = float(model_cfg.get("retry_sleep_seconds", 5))
    timeout = float(model_cfg.get("timeout_seconds", 120))
    max_tokens = int(model_cfg.get("max_tokens", model_cfg.get("max_completion_tokens", 4096)))
    temperature = model_cfg.get("temperature", 0)
    top_p = model_cfg.get("top_p", 1)

    payload = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            token = read_bearer_token(model_cfg)
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.request("POST", endpoint, headers=headers, json=payload, timeout=timeout)
            if not resp.text or not resp.text.strip():
                raise RuntimeError(f"Empty response from Apigee/Azure: status={resp.status_code}")
            try:
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Non-JSON response from Apigee/Azure: status={resp.status_code}; text={resp.text[:500]}"
                ) from exc
            if resp.status_code >= 400:
                raise RuntimeError(f"Apigee/Azure error status={resp.status_code}; body={data}")
            return extract_text_from_response(data)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"Apigee/Azure request failed after {max_retries} attempts: {last_error}")
