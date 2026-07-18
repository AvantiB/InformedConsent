#!/usr/bin/env python
"""Probe the exact Union V0 forward prompt through Mayo Apigee/Azure.

Use this when a simple Apigee token test works but the Union V0 runner fails.
It calls the same Apigee helper and the same Union V0 forward prompt for one row,
then prints response length, a preview, and JSON parse diagnostics without writing
round-trip outputs.
"""
from __future__ import annotations

import argparse
import