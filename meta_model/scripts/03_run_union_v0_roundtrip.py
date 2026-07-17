#!/usr/bin/env python
"""Run Union V0 full-dictionary forward/backward round-trip experiments.

This runner is designed for one model at a time. Open-source models can be
served with vLLM's OpenAI-compatible API, while closed-source models can use the
same OpenAI client interface. Outputs are append-only JSONL files so interrupted
runs can be resumed safely.

The