#!/usr/bin/env python
"""Run Union V0 full-dictionary forward/backward round-trip experiments.

Forward mapping intentionally has two layers:
1. raw annotations, including same-span, overlapping, and nested labels; and
2. interpretation_units, where the LLM decides how related annotations should be
   considered together for backward reconstruction.

Backward mapping never receives