#!/usr/bin/env python
"""Run provisional/audited reduced-schema round-trip smoke tests.

The prompt is intentionally close to the Union V0 prompt style:
- neutral data-dictionary wording;
- cluster IDs are treated as dictionary IDs, not named roles;
- decision is sentence/provision-level only;
- cluster membership/source examples explain the fields;
- output