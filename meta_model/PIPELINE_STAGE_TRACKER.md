# Pipeline stage tracker

This tracker records the agreed pipeline state and should be updated whenever the plan changes.

## Current status snapshot

| Phase | Status | Notes |
|---|---|---|
| P0 Reset/archive | in progress | Old leakage-contaminated outputs should be archived, not deleted. README and runbook updated. |
| P1 strict baseline rerun | planned | Reuse existing forward mappings for individual source models and Union V0; rerun backward only with strict annotation-only input. |
| P2 data-driven seeding | not started | Rebuild from strict-preserved valid annotation evidence; no co-occurrence-only clustering. |
| P3 schema generation | not started | Compare direct LLM high/low schemas vs data-driven LLM high/low schemas. |
| P4 generated schema evaluation | not started | Constant forward/backward/scoring protocol. |
| P5 expert review | not started | Expert assessment after candidate schema results are available. |

## Agreed decisions

1. Existing forward mappings for individual source-model baselines and Union V0 may be reused.
2. Backward reconstruction must be rerun with a universal strict annotation-only prompt.
3. Backward input may include only valid span-level annotations, annotation labels, canonical annotation-attached modifiers when present, and sentence-level annotations only when valid span annotations exist.
4. Backward input must exclude unmatched language, residual text, interpretation units, combined meanings, rationales, raw forward responses, previous reconstructions, and the original sentence.
5. Rows with no backward-eligible annotations are not sent to the LLM; reconstructed_sentence is intentionally blank.
6. Old outputs should be archived rather than deleted to preserve provenance.
7. The prior broad Manual V1 is no longer the main schema claim. The next schema comparison will test direct LLM induction and data-driven LLM induction at high and low granularity.
8. The data-driven pipeline must type relationships before clustering. Only near-equivalence edges should be clustered; co-occurrence/proximity edges are used for complementarity/functional bundles.

## Detailed task tracker

The editable workbook-style task list is maintained in:

```text
meta_model/PIPELINE_STAGE_TRACKER.csv
```

Update that CSV whenever a task is completed, postponed, or revised.

## Next immediate tasks

1. Archive old leakage-contaminated outputs.
2. Create `meta_model/strict_annotation_only_experiments` as the fresh output root.
3. Copy/import existing forward JSONL files into the fresh root.
4. Run strict backward only for Union V0 and individual source models.
5. Standardize, score, diagnose, and compile Phase 1 baseline results.
