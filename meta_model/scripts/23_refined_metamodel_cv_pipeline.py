#!/usr/bin/env python
"""Refined informed-consent meta-model cross-validation pipeline.

Paper-facing workflow:
- create form-level cross-validation splits using stable form_key when available;
- build provenance-preserving mention evidence;
- strip annotation-format decision markers such as (permit)/(deny) from spans;
- store annotation decisions