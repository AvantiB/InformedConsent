#!/usr/bin/env python
"""Induce a reduced V1 consent meta-model from the expert-validated corpus.

This is the main derivation/induction path for the reduced consent meta-model.

Principle:
- Expert-preserved round trips are treated as functionally validated positive evidence:
  the forward representation contained enough information to reconstruct the original
  sentence meaning.
- Expert-failed round trips are treated as boundary evidence:
  they weaken merge evidence and flag unsafe simplifications.
-