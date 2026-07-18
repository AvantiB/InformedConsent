#!/usr/bin/env python
"""Audit annotation granularity in Union V0 and individual round-trip outputs.

This script is meant to detect model/prompt behavior that can inflate backward
meaning preservation: many annotations, long clause-level spans, full-sentence-like
spans, or duplicated coverage of the same source tokens.
"""
from __future__ import annotations

import argparse
import csv
import json