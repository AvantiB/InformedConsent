#!/usr/bin/env python
"""Run individual source-model prompt round-trip experiments.

Replication condition: new LLMs + original individual source-model prompts.

The forward step uses the original source-model forward prompt text. The backward
step uses a matching backward prompt if supplied; otherwise it uses a generic
reconstruction prompt. Backward mapping never receives the original