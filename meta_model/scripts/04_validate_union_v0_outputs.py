#!/usr/bin/env python
"""Validate Union V0 round-trip smoke/full outputs.

Checks parse completion, Union V0 ID validity, repairable ID formatting errors,
overlap/nesting annotations, and interpretation-unit coverage. The validator
separates invalid IDs that remain in primary annotations from invalid IDs that
were quarantined by the runner in invalid_annotations.

Readiness gates