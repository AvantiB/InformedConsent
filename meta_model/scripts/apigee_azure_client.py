#!/usr/bin/env python
"""Small helper for Mayo Apigee Azure OpenAI chat-completions endpoint.

No tokens should be committed. Configure either:
- api_key_env: environment variable containing the bearer token; or
- api_key_file / api_key_file_env: text file containing the bearer token.

When api_key_file is used, the token is read on every request, so a long-running
job can continue after you overwrite the token file with a refreshed token.

Some GPT-5.x deployments do not support temperature. Set omit_temperature: true
in the model config to omit it from the request payload even when shared defaults
contain temperature: 0.
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
    token_file = model_cfg.get("api_key_file") or