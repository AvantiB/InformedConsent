#!/usr/bin/env python
"""Induce a reduced V1 consent meta-model using an evidence graph.

This script implements the data-driven part of the reduced meta-model workflow.
It does not start from a hand-written role set. Instead it:

1. builds an element profile for every span-level source-model element;
2. builds weighted element-element relationships from co-use, same-span evidence