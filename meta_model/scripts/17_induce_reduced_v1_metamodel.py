#!/usr/bin/env python
"""Induce a reduced V1 consent meta-model from the expert-validated corpus.

Main derivation path:
- expert-preserved round trips are positive functional evidence;
- expert-failed round trips are boundary evidence that weakens merge evidence;
- new LLM runs are validation/stress-test data, not the primary induction corpus.

Input is usually produced by 12_build_expert_roundtrip_corpus.py.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

TEXT_COL