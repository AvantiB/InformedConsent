#!/usr/bin/env python
"""Post-score round-trip evaluation and meta-model evidence analysis.

This script combines classifier scores, lexical/content metrics, cue preservation,
and source-element evidence for reduced meta-model development.

Important distinction:
- Sentence-level decision fields are summarized separately.
- Span/source-element evidence and co-occurrence tables