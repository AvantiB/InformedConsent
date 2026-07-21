#!/usr/bin/env python
"""Refined informed-consent meta-model cross-validation pipeline.

Paper-facing workflow:
- create form-level cross-validation splits using stable form_key when available;
- build provenance-preserving mention evidence;
- split broad source elements into context-specific sense nodes;
- separate near-equivalence from broader/narrower, related-distinct, and complementary evidence;
- merge only strict