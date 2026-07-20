#!/usr/bin/env python
"""Discover evidence for a reduced V1 consent meta-model from expert data.

This script intentionally does NOT hard-code the reduced V1 fields. It produces
evidence artifacts for audit:

1. a semantic-equivalence graph: which source-model elements appear to express
   the same semantic field and may be merge candidates;
2. a provision-bundle graph: which elements co-occur composition