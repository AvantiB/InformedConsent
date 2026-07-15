#!/usr/bin/env python
"""MVP meaning-preservation classifier experiments.

Runs lexical, consent-aware, optional embedding, optional NLI, and hybrid
classifiers for binary meaning-preservation labels from LLM round trips.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer