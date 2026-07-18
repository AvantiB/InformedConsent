#!/usr/bin/env python
"""Mayo Apigee Azure OpenAI chat-completions helper.

Token options in model YAML:
- OAuth auto-refresh, recommended for long jobs:
  oauth_client_id_env, oauth_client_secret_env, optional oauth_token_url
- Static bearer token fallback:
  api_key_env, api_key_file, or api_key_file_env

Secrets must live in environment variables or untracked local files, never in git.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

_TOKEN_CACHE: dict[str, dict[str, Any]] = {}


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return True
    return False


def _cache_key(model_cfg: dict[str, Any]) -> str:
    return str(model_cfg.get("model_key") or model_cfg.get("deployment") or model_cfg.get("model") or "default")


def oauth_configured(model_cfg: dict[str, Any]) -> bool:
    return bool(model_cfg.get("oauth_client_id_env") and model_cfg.get("oauth_client_secret_env"))


def _parse_token_response(resp: requests.Response) -> tuple[str, float | None]:
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError(f"Empty OAuth token response: status={resp.status_code}")
    try:
        data = resp.json()
    except Exception:
        return text.strip('"'), None

    token = data.get("access_token") or data.get("token") or data.get("apigee_token")
    if not token:
        raise RuntimeError(f"OAuth response did not contain access_token/token: {data}")
    expires_in = data.get("expires_in") or data.get("expires")
    expires_at = None
    if expires_in is not None:
        try:
            expires_at = time.time() + max(0, int(expires_in) - 120)
        except Exception:
            expires_at = None
    return str(token).strip(), expires_at


def generate_oauth_token(model_cfg: dict[str, Any]) -> str:
    client_id_env = str(model_cfg.get("oauth_client_id_env"))
    client_secret_env = str(model_cfg.get("oauth_client_secret_env"))
    client_id = os.getenv(client_id_env, "").strip()
    client_secret = os.getenv(client_secret_env, "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing OAuth client credentials. Set environment variables named by "
            "oauth_client_id_env and oauth_client_secret_env."
        )

    token_url = str(model_cfg.get("oauth_token_url") or "https://mcc.apix.mayo.edu/oauth/token")
    timeout = float(model_cfg.get("oauth_timeout_seconds", model_cfg.get("timeout_seconds", 120)))
    resp = requests.request(
        "POST",
        token_url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OAuth token request failed: status={resp.status_code}; body={resp.text[:500]}")

    token, expires_at = _parse_token_response(resp)
    _TOKEN_CACHE[_cache_key(model_cfg)] = {"token": token, "expires_at": expires_at or (time.time() + 25 * 60)}
    return token


def read_static_bearer_token(model_cfg: dict[str, Any]) -> str:
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
            "Missing Apigee bearer token. Configure OAuth credentials or set "
            "api_key_env, api_key_file, or api_key_file_env."
        )
    return token


def read_bearer_token(model_cfg: dict[str, Any], force_refresh: bool = False) -> str:
    if oauth_configured(model_cfg):
        key = _cache_key(model_cfg)
        cached = _TOKEN_CACHE.get(key)
        if not force_refresh and cached:
            expires_at = cached.get("expires_at")
            if expires_at is None or float(expires_at) > time.time():
                return str(cached["token"])
        return generate_oauth_token(model_cfg)
    return read_static_bearer_token(model_cfg)


def build_endpoint(model_cfg: dict[str, Any]) -> str:
    apigee_url = str(model_cfg.get("apigee_url") or "https://mcc.apix.mayo.edu").rstrip("/")
    deployment = str(model_cfg.get("deployment") or model_cfg.get("engine") or model_cfg.get("model") or "gpt-5.1")
    api_version = str(model_cfg.get("api_version") or "2024-10-21")
    dep = quote(deployment, safe="")
    return f"{apigee_url}/llm-azure-openai/openai/deployments/{dep}/chat/completions?api-version={api_version}"


def extract_text_from_response(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"].get("content", "")
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
    payload: dict[str, Any] = {"messages": messages, "max_completion_tokens": max_tokens}

    if not bool(model_cfg.get("omit_temperature", False)):
        temperature = model_cfg.get("temperature", None)
        if not _is_nullish(temperature):
            payload["temperature"] = temperature

    top_p = model_cfg.get("top_p", None)
    if not _is_nullish(top_p):
        payload["top_p"] = top_p

    reasoning_effort = model_cfg.get("reasoning_effort", None)
    if not _is_nullish(reasoning_effort):
        payload["reasoning_effort"] = reasoning_effort
    return payload


def _should_force_refresh(status_code: int, data: Any) -> bool:
    if status_code in {401, 403}:
        return True
    text = str(data).lower()[:1000]
    return any(s in text for s in ["expired token", "invalid token", "unauthorized", "forbidden"])


def call_apigee_chat(client: Any, model_cfg: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """Drop-in replacement for call_chat(client, model_cfg, messages)."""
    endpoint = build_endpoint(model_cfg)
    max_retries = int(model_cfg.get("max_retries", 3))
    retry_sleep = float(model_cfg.get("retry_sleep_seconds", 5))
    timeout = float(model_cfg.get("timeout_seconds", 120))
    payload = build_payload(model_cfg, messages)
    last_error: Exception | None = None

    force_refresh = False
    for attempt in range(1, max_retries + 1):
        try:
            token = read_bearer_token(model_cfg, force_refresh=force_refresh)
            force_refresh = False
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.request("POST", endpoint, headers=headers, json=payload, timeout=timeout)
            if not resp.text or not resp.text.strip():
                if resp.status_code in {401, 403} and oauth_configured(model_cfg):
                    force_refresh = True
                    raise RuntimeError(f"Empty auth response from Apigee/Azure: status={resp.status_code}")
                raise RuntimeError(f"Empty response from Apigee/Azure: status={resp.status_code}")
            try:
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Non-JSON response from Apigee/Azure: status={resp.status_code}; text={resp.text[:500]}"
                ) from exc

            if resp.status_code >= 400:
                if _should_force_refresh(resp.status_code, data) and oauth_configured(model_cfg):
                    force_refresh = True
                raise RuntimeError(f"Apigee/Azure error status={resp.status_code}; body={data}")

            text = extract_text_from_response(data)
            if not text:
                raise RuntimeError(
                    "Apigee/Azure returned an empty message content. "
                    f"status={resp.status_code}; finish_reason={data.get('choices', [{}])[0].get('finish_reason')}; "
                    f"body={str(data)[:1000]}"
                )
            return text
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)

    raise RuntimeError(f"Apigee/Azure request failed after {max_retries} attempts: {last_error}")
