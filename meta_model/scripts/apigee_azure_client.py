#!/usr/bin/env python
"""Mayo Apigee Azure OpenAI chat-completions helper.

Configure tokens through the model YAML using one of:
- api_key_env: name of an environment variable containing the bearer token
- api_key_file: path to a text file containing the bearer token
- api_key_file_env: name of an environment variable whose value is the token-file path

When a token file is used, the file is read on every request so a long-running job
can continue after the file is overwritten with a refreshed token.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return True
    return False


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
            "Missing Apigee bearer token. Set api_key_env, api_key_file, "
            "or api_key_file_env in the model config."
        )
    return token


def build_endpoint(model_cfg: dict[str, Any]) -> str:
    apigee_url = str(model_cfg.get("apigee_url") or "https://mcc.apix.mayo.edu").rstrip("/")
    deployment = str(
        model_cfg.get("deployment")
        or model_cfg.get("engine")
        or model_cfg.get("model")
        or "gpt-5.1"
    )
    api_version = str(model_cfg.get("api_version") or "2024-10-21")
    dep = quote(deployment, safe="")
    return f"{apigee_url}/llm-azure-openai/openai/deployments/{dep}/chat/completions?api-version={api_version}"


def extract_text_from_response(response: dict[str, Any]) -> str:
    try:
        message = response["choices"][0]["message"]
        content = message.get("content", "")
    except Exception as exc:
        raise RuntimeError(f"Unexpected Apigee/Azure response shape: {response}") from exc

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                parts.append(str(part.get("text", "")))
        return "".join(parts).strip()

    return (content or "").strip()


def build_payload(model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    max_tokens = int(model_cfg.get("max_tokens", model_cfg.get("max_completion_tokens", 4096)))
    payload: dict[str, Any] = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }

    omit_temperature = bool(model_cfg.get("omit_temperature", False))
    temperature = model_cfg.get("temperature", None)
    if not omit_temperature and not _is_nullish(temperature):
        payload["temperature"] = temperature

    top_p = model_cfg.get("top_p", None)
    if not _is_nullish(top_p):
        payload["top_p"] = top_p

    reasoning_effort = model_cfg.get("reasoning_effort", None)
    if not _is_nullish(reasoning_effort):
        payload["reasoning_effort"] = reasoning_effort

    return payload


def call_apigee_chat(client: Any, model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """Drop-in replacement for call_chat(client, model_cfg, messages)."""
    endpoint = build_endpoint(model_cfg)
    max_retries = int(model_cfg.get("max_retries", 3))
    retry_sleep = float(model_cfg.get("retry_sleep_seconds", 5))
    timeout = float(model_cfg.get("timeout_seconds", 120))
    payload = build_payload(model_cfg, messages)

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

            text = extract_text_from_response(data)
            if not text:
                raise RuntimeError(
                    "Apigee/Azure returned an empty message content. "
                    f"status={resp.status_code}; finish_reason="
                    f"{data.get('choices', [{}])[0].get('finish_reason')}; body={str(data)[:1000]}"
                )
            return text
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)

    raise RuntimeError(f"Apigee/Azure request failed after {max_retries} attempts: {last_error}")
