#!/usr/bin/env python
"""Induce a reduced V1 consent meta-model from the original expert-validated corpus.

This is the main derivation path. Expert-preserved round trips are treated as
functionally validated positive evidence; expert-failed round trips are boundary
evidence that weakens merge evidence and flags unsafe simplifications. Newer LLM
runs are validation/stress-test data