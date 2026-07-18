#!/usr/bin/env python
"""Standardize Union V0 and individual information-model round-trip outputs.

This script converts heterogeneous round-trip output folders into one classifier-ready
table. It does not score anything; it validates presence of forward/backward
records, extracts reconstruction text, counts annotations, and writes comparison
metadata.
"""
from __future__ import annotations

import argparse
import csv
import json