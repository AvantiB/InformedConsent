#!/usr/bin/env python
"""Generate paper/deck-ready visualizations for meta-model development and evaluation.

This script is intentionally tolerant of partially completed runs. It creates any
plots for which the required input files are available and skips the rest with a
clear message.

Primary input families:
- crosswalk outputs from script 26
- refined/cluster seed outputs from scripts 23/24/28