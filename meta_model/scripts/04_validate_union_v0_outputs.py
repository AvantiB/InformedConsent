#!/usr/bin/env python
"""Validate Union V0 round-trip smoke/full outputs.

The validator checks:
- JSONL readability;
- forward/backward completion;
- failed requests;
- Union V0 ID validity in primary annotations;
- IDs quarantined by the runner in invalid_annotations;
- overlap/nesting annotation counts; and
- interpretation-unit coverage.

Readiness gates:
- strict: no invalid