#!/usr/bin/env python
"""MVP meaning-preservation classifier experiments.

This script builds a binary meaning-preservation dataset from LLM round-trip
outputs, extracts lexical/consent-aware/optional semantic features, and evaluates
models under random, leave-sentence-out, leave-one-LLM-out, and
leave-one-information-model-out settings.

Design choice