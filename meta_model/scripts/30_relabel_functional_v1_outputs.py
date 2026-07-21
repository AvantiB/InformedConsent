#!/usr/bin/env python
"""Relabel Functional V1 JSONL outputs with a schema-condition name.

The round-trip prompt is intentionally constant across reduced-schema conditions;
only the schema YAML changes. This helper changes only metadata fields in the
saved outputs after generation so manual V1 and LLM-induced V1 can be scored as
separate conditions without changing the prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def relabel(path: Path, condition: str, information_model: str) -> int:
    if not path.exists():
        return 0
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                obj["condition"] = condition
                obj["information_model"] = information_model
                obj["info_model"] = information_model
                rows.append(obj)
    backup = path.with_suffix(path.suffix + ".before_relabel")
    if not backup.exists():
        backup.write_text(path.read_text())
    with path.open("w") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", required=True, help="Directory containing functional_v1_forward_mappings.jsonl and backward JSONL, e.g. .../<model>/<compact>.")
    ap.add_argument("--condition", required=True, help="Condition label, e.g. functional_v1_manual or functional_v1_llm_induced.")
    ap.add_argument("--information_model", default="Functional_V1")
    args = ap.parse_args()
    root = Path(args.output_dir)
    n1 = relabel(root / "functional_v1_forward_mappings.jsonl", args.condition, args.information_model)
    n2 = relabel(root / "functional_v1_backward_reconstructions.jsonl", args.condition, args.information_model)
    print(f"Relabeled {n1} forward and {n2} backward rows under {root}")


if __name__ == "__main__":
    main()
