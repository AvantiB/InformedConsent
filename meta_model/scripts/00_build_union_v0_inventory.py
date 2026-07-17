#!/usr/bin/env python
"""Build Union V0 source-element inventory from original full-dictionary prompts.

This script treats each original information-model prompt as an input data
dictionary. It extracts the allowed labels/elements and writes a combined Union
V0 inventory. The inventory is intentionally unreduced and can be used either as
a full-dictionary baseline or as the source for retrieval-augmented candidate
selection.

The parser is deliberately conservative: it only starts a new