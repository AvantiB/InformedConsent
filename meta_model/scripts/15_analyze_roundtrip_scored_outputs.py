#!/usr/bin/env python
"""Post-score round-trip evaluation and meta-model evidence analysis."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CUE_PATTERNS: dict[str, list[str]] = {
    "permission": [r"\bcan\b", r"\bmay\b", r"\bare allowed\b", r"\bis allowed\b", r"\bpermission\b", r"\bpermit(?:ted)?\b", r"\bauthori[sz]e(?:d)?\b", r"\bchoose\b", r"\bchoice\b", r"\bdecide\b", r"\boptional\b"],
    "obligation": [r"\bmust\b", r"\brequired\b", r"\brequirement\b", r"\bneed to\b", r"\bneeds to\b", r"\bhave to\b", r"\bhas to\b", r"\bwill need\b", r"\bwill be asked\b", r"\bwe ask\b", r"\bwe are asking\b"],
    "prohibition": [r"\bcannot\b", r"\bcan't\b", r"\bmay not\b", r"\bmust not\b", r"\bnot allowed\b", r"\bnever\b", r"\bno\b", r"\bprohibit(?:ed|ion)?\b", r"\bwithout\b"],
    "condition": [r"\bif\b", r"\bwhen\b", r"\bunless\b", r"\bonly if\b", r"\bas long as\b", r"\bin order to\b", r"\bdepending on\b"],
    "constraint_or_exception": [r"\bonly\b", r"\bexcept\b", r"\bexception\b", r"\blimit(?:ed|ation|s)?\b", r"\brestrict(?:ed|ion|s)?\b", r"\bconstraint\b", r"\bsubject to\b"],
    "time_or_duration": [r"\bat any time\b", r"\bany time\b", r"\bfuture\b", r"\blater\b", r"\bbefore\b", r"\bafter\b", r"\buntil\b", r"\bduring\b", r"\bfor at least\b", r"\byear(?:s)?\b", r"\bmonth(?:s)?\b", r"\bday(?:s)?\b", r"\bongoing\b", r"\blong-term\b"],
    "withdrawal": [r"\bwithdraw(?:al|n)?\b", r"\bquit\b", r"\bstop taking part\b", r"\bstop participating\b", r"\bleave the study\b", r"\bdrop out\b"],
    "consent_or_choice": [r"\bconsent\b", r"\bagree\b", r"\bagreement\b", r"\byes\b", r"\bno\b", r"\bjoin\b", r"\bparticipate\b", r"\btake part\b", r"\bdecide\b"],
    "data_or_specimen": [r"\bdata\b", r"\binformation\b", r"\brecords?\b", r"\bmedical records?\b", r"\bhealth records?\b", r"\bsamples?\b", r"\bspecimens?\b", r"\btissue\b", r"\bblood\b", r"\bbiospecimen\b", r"\bdna\b", r"\bgenetic\b", r"\bgenomic\b"],
    "use_or_analysis": [r"\buse\b", r"\bused\b", r"\busing\b", r"\banaly[sz]e\b", r"\banaly[sz]ed\b", r"\btest\b", r"\btesting\b", r"\bstudy\b", r"\bresearch\b"],
    "sharing_or_access": [r"\bshare\b", r"\bshared\b", r"\bsharing\b", r"\bdisclose\b", r"\bdisclosure\b", r"\brelease\b", r"\bgive\b", r"\bsend\b", r"\baccess\b", r"\bavailable to\b", r"\bprovide\b"],
    "privacy_or_identifiability": [r"\bprivate\b", r"\bprivacy\b", r"\bconfidential\b", r"\bidentify\b", r"\bidentified\b", r"\bidentifiable\b", r"\bde-?identified\b", r"\bcoded\b", r"\banonymous\b", r"\bname\b", r"\bcontact information\b"],
    "actor_or_recipient": [r"\byou\b", r"\bwe\b", r"\bresearchers?\b", r"\bscientists?\b", r"\binvestigators?\b", r"\bdoctor(?:s)?\b", r"\bstudy team\b", r"\ball of us\b", r"\bcompany\b", r"\bthird part(?:y|ies)\b"],
    "risk_or_benefit": [r"\brisk(?:s)?\b", r"\bbenefit(?:s)?\b", r"\bharm\b", r"\bincidental\b", r"\bresults?\b", r"\breturn(?:ed)?\b", r"\bfindings?\b"],
}
CRITICAL_CUES = {"permission", "obligation", "prohibition", "condition", "constraint_or_exception", "time_or_duration", "withdrawal", "sharing_or_access", "privacy_or_identifiability", "data_or_specimen"}


def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def text_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def group_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in ["llm", "condition", "information_model"] if c in df.columns]
    if "information_model" not in cols and "info_model" in df.columns:
        cols.append("info_model")
    return cols or ["llm"]


def find_matches(text: Any, patterns: list[str]) -> list[str]:
    t = norm_text(text).lower()
    return [pat for pat in patterns if re.search(pat, t, flags=re.IGNORECASE)]


def cue_hits(text: Any) -> dict[str, list[str]]:
    return {name: find_matches(text, pats) for name, pats in CUE_PATTERNS.items()}


def cue_set(hits: dict[str, list[str]]) -> set[str]:
    return {k for k, v in hits.items() if v}


def set_recall(orig: set[str], recon: set[str]) -> float:
    return 1.0 if not orig else len(orig & recon) / len(orig)


def set_precision(orig: set[str], recon: set[str]) -> float:
    return (1.0 if not orig else 0.0) if not recon else len(orig & recon) / len(recon)


def set_jaccard(orig: set[str], recon: set[str]) -> float:
    if not orig and not recon:
        return 1.0
    if not orig or not recon:
        return 0.0
    return len(orig & recon) / len(orig | recon)


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def modal_category(groups: set[str]) -> str:
    cats = [x for x in ["permission", "obligation", "prohibition"] if x in groups]
    return "+".join(cats) if cats else "none"


def add_cue_preservation(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["original_text_for_cues"] = text_col(out, "original_text", "source_text")
    out["reconstructed_text_for_cues"] = text_col(out, "reconstructed_text", "reconstructed_sentence")
    rows: list[dict[str, Any]] = []
    for _, r in out.iterrows():
        oh = cue_hits(r["original_text_for_cues"])
        rh = cue_hits(r["reconstructed_text_for_cues"])
        os, rs = cue_set(oh), cue_set(rh)
        missing, added = sorted(os - rs), sorted(rs - os)
        p, rec = set_precision(os, rs), set_recall(os, rs)
        critical_missing = sorted((os - rs) & CRITICAL_CUES)
        rows.append({
            "orig_cue_groups_json": json.dumps(sorted(os), ensure_ascii=False),
            "recon_cue_groups_json": json.dumps(sorted(rs), ensure_ascii=False),
            "missing_cue_groups_json": json.dumps(missing, ensure_ascii=False),
            "added_cue_groups_json": json.dumps(added, ensure_ascii=False),
            "orig_cue_match_patterns_json": json.dumps(oh, ensure_ascii=False),
            "recon_cue_match_patterns_json": json.dumps(rh, ensure_ascii=False),
            "cue_group_recall": rec,
            "cue_group_precision": p,
            "cue_group_f1": f1(p, rec),
            "cue_group_jaccard": set_jaccard(os, rs),
            "n_orig_cue_groups": len(os),
            "n_recon_cue_groups": len(rs),
            "n_missing_cue_groups": len(missing),
            "modal_category_original": modal_category(os),
            "modal_category_reconstruction": modal_category(rs),
            "modal_category_changed": modal_category(os) != modal_category(rs),
            "critical_missing_cue_groups_json": json.dumps(critical_missing, ensure_ascii=False),
            "n_critical_missing_cue_groups": len(critical_missing),
            "audit_critical_cue_missing": bool(critical_missing),
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    gcols = group_cols(df)
    metric_candidates = ["classifier_preservation_score", "content_token_recall", "content_token_f1", "bigram_recall", "recon_to_orig_length_ratio", "cue_group_recall", "cue_group_f1", "cue_group_jaccard", "n_missing_cue_groups"]
    bool_candidates = ["classifier_pred_0_8", "classifier_pred_0_9", "audit_high_score_low_overlap", "audit_low_content_recall", "audit_heavy_compression", "modal_category_changed", "audit_critical_cue_missing"]
    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(gcols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: keys[i] for i, c in enumerate(gcols)}
        row["n"] = len(g)
        for c in metric_candidates:
            if c in g.columns:
                row[f"mean_{c}"] = pd.to_numeric(g[c], errors="coerce").mean()
                row[f"median_{c}"] = pd.to_numeric(g[c], errors="coerce").median()
        for c in bool_candidates:
            if c in g.columns:
                row[f"pct_{c}"] = pd.to_numeric(g[c], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(gcols)


def cue_group_long_summary(df: pd.DataFrame) -> pd.DataFrame:
    gcols = group_cols(df)
    rows = []
    for keys, g in df.groupby(gcols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {c: keys[i] for i, c in enumerate(gcols)}
        for cue in CUE_PATTERNS:
            op, rp = [], []
            for _, r in g.iterrows():
                os = set(json.loads(r["orig_cue_groups_json"]))
                rs = set(json.loads(r["recon_cue_groups_json"]))
                op.append(cue in os)
                rp.append(cue in rs)
            op_arr, rp_arr = np.array(op, dtype=bool), np.array(rp, dtype=bool)
            denom = int(op_arr.sum())
            rows.append({
                **base,
                "cue_group": cue,
                "n": len(g),
                "n_original_present": denom,
                "original_prevalence": float(op_arr.mean()) if len(op_arr) else np.nan,
                "reconstruction_prevalence": float(rp_arr.mean()) if len(rp_arr) else np.nan,
                "preservation_rate_when_original_present": float((op_arr & rp_arr).sum() / denom) if denom else np.nan,
                "loss_rate_when_original_present": float((op_arr & ~rp_arr).sum() / denom) if denom else np.nan,
                "addition_rate_when_original_absent": float((~op_arr & rp_arr).sum() / max(1, int((~op_arr).sum()))),
            })
    return pd.DataFrame(rows).sort_values(gcols + ["cue_group"])


def modal_transition_summary(df: pd.DataFrame) -> pd.DataFrame:
    gcols = group_cols(df)
    trans = df.copy()
    trans["modal_transition"] = trans["modal_category_original"] + " -> " + trans["modal_category_reconstruction"]
    rows = []
    for keys, g in trans.groupby(gcols + ["modal_transition"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({
            **{c: keys[i] for i, c in enumerate(gcols + ["modal_transition"])},
            "n": len(g),
            "mean_classifier_preservation_score": pd.to_numeric(g.get("classifier_preservation_score"), errors="coerce").mean(),
            "mean_cue_group_recall": pd.to_numeric(g["cue_group_recall"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows).sort_values(gcols + ["n"], ascending=[True] * len(gcols) + [False])


def parse_jsonish(value: Any) -> Any | None:
    text = norm_text(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
    return None


def walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def row_element_records(row: pd.Series) -> list[dict[str, Any]]:
    obj = parse_jsonish(row.get("forward_mapping", ""))
    if obj is None:
        return []
    records = []
    for d in walk(obj):
        if not isinstance(d, dict) or not d.get("union_element_id"):
            continue
        records.append({
            "source_id": row.get("source_id", row.get("roundtrip_id", "")),
            "llm": row.get("llm", ""),
            "condition": row.get("condition", ""),
            "information_model": row.get("information_model", row.get("info_model", "")),
            "union_element_id": str(d.get("union_element_id")),
            "span_text": norm_text(d.get("span_text") or d.get("evidence_span_text") or d.get("value") or ""),
            "annotation_id": norm_text(d.get("annotation_id", "")),
            "score": row.get("classifier_preservation_score", np.nan),
            "content_token_recall": row.get("content_token_recall", np.nan),
            "cue_group_recall": row.get("cue_group_recall", np.nan),
            "modal_category_changed": row.get("modal_category_changed", np.nan),
            "audit_critical_cue_missing": row.get("audit_critical_cue_missing", np.nan),
            "orig_cue_groups_json": row.get("orig_cue_groups_json", "[]"),
        })
    return records


def build_element_evidence(df: pd.DataFrame, inventory_csv: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.extend(row_element_records(row))
    long = pd.DataFrame(records)
    if long.empty:
        return long, pd.DataFrame(), pd.DataFrame()

    agg_rows = []
    for uid, g in long.groupby("union_element_id"):
        spans = [s for s in g["span_text"].dropna().astype(str).tolist() if s]
        span_counts = Counter(spans)
        orig_cues = Counter()
        for x in g["orig_cue_groups_json"].dropna().astype(str):
            try:
                orig_cues.update(json.loads(x))
            except Exception:
                pass
        agg_rows.append({
            "union_element_id": uid,
            "n_mentions": len(g),
            "n_source_sentences": g["source_id"].nunique(),
            "n_llms": g["llm"].nunique(),
            "n_conditions": g["condition"].nunique(),
            "mean_classifier_preservation_score": pd.to_numeric(g["score"], errors="coerce").mean(),
            "mean_content_token_recall": pd.to_numeric(g["content_token_recall"], errors="coerce").mean(),
            "mean_cue_group_recall": pd.to_numeric(g["cue_group_recall"], errors="coerce").mean(),
            "pct_modal_category_changed": pd.to_numeric(g["modal_category_changed"], errors="coerce").mean(),
            "pct_critical_cue_missing": pd.to_numeric(g["audit_critical_cue_missing"], errors="coerce").mean(),
            "top_span_examples_json": json.dumps([x for x, _ in span_counts.most_common(8)], ensure_ascii=False),
            "top_original_cue_groups_json": json.dumps([x for x, _ in orig_cues.most_common(8)], ensure_ascii=False),
        })
    evidence = pd.DataFrame(agg_rows).sort_values(["n_source_sentences", "mean_classifier_preservation_score"], ascending=[False, False])
    if inventory_csv:
        inv = pd.read_csv(inventory_csv)
        if "union_element_id" in inv.columns:
            evidence = evidence.merge(inv, on="union_element_id", how="left")

    pair_counts: defaultdict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"n": 0, "source_ids": set(), "scores": []})
    for _, g in long.groupby(["source_id", "llm", "condition", "information_model"], dropna=False):
        ids = sorted(set(g["union_element_id"]))
        score = pd.to_numeric(g["score"], errors="coerce").mean()
        sid = str(g["source_id"].iloc[0])
        for a, b in itertools.combinations(ids, 2):
            rec = pair_counts[(a, b)]
            rec["n"] += 1
            rec["source_ids"].add(sid)
            rec["scores"].append(score)
    pairs = [{"union_element_id_a": a, "union_element_id_b": b, "n_cooccurrences": rec["n"], "n_source_sentences": len(rec["source_ids"]), "mean_classifier_preservation_score": float(np.nanmean(rec["scores"])) if rec["scores"] else np.nan} for (a, b), rec in pair_counts.items()]
    pair_df = pd.DataFrame(pairs).sort_values(["n_source_sentences", "n_cooccurrences"], ascending=[False, False])
    return long, evidence, pair_df


def write_audits(df: pd.DataFrame, out: Path) -> None:
    cols = ["source_id", "llm", "condition", "information_model", "original_text", "source_text", "reconstructed_text", "reconstructed_sentence", "classifier_preservation_score", "content_token_recall", "content_token_f1", "bigram_recall", "cue_group_recall", "cue_group_f1", "modal_category_original", "modal_category_reconstruction", "missing_cue_groups_json", "critical_missing_cue_groups_json", "missing_content_tokens_json"]
    cols = [c for c in cols if c in df.columns]
    score = pd.to_numeric(df.get("classifier_preservation_score", 0), errors="coerce")
    cue = pd.to_numeric(df["cue_group_recall"], errors="coerce")
    high_risk = df[(score >= 0.75) & (df["audit_critical_cue_missing"] | (cue < 0.70))].sort_values(["classifier_preservation_score", "cue_group_recall"], ascending=[False, True])
    high_risk[cols].to_csv(out / "high_score_cue_loss_audit.csv", index=False)
    df.sort_values(["cue_group_recall", "classifier_preservation_score"], ascending=[True, False]).head(250)[cols].to_csv(out / "lowest_cue_preservation_top250.csv", index=False)


def write_story(summary: pd.DataFrame, cue_long: pd.DataFrame, element_evidence: pd.DataFrame, pair_df: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Round-trip evaluation and meta-model evidence summary",
        "",
        "This report combines classifier-based meaning-preservation scores, lexical/content coverage, and transparent cue-preservation diagnostics.",
        "",
        "## Evaluation lens",
        "",
        "- Classifier score: proxy estimate of semantic meaning preservation.",
        "- Lexical/content coverage: guardrail for omissions and heavy compression.",
        "- Cue preservation: qualitative check that modal and consent-governance cues survive reconstruction.",
        "- Source-element evidence: empirical support for reduced meta-model dimensions and candidate merge/split decisions.",
        "",
    ]
    if not summary.empty:
        lines += ["## Condition-level summary", "", summary.head(30).to_markdown(index=False), ""]
    if not cue_long.empty:
        high_loss = cue_long.sort_values("loss_rate_when_original_present", ascending=False).head(20)
        lines += ["## Cue groups most often lost when present in the original", "", high_loss.to_markdown(index=False), ""]
    if not element_evidence.empty:
        cols = [c for c in ["union_element_id", "n_source_sentences", "n_llms", "mean_classifier_preservation_score", "mean_content_token_recall", "mean_cue_group_recall", "top_span_examples_json"] if c in element_evidence.columns]
        lines += ["## Source elements with strongest empirical support", "", element_evidence[cols].head(25).to_markdown(index=False), ""]
    if not pair_df.empty:
        lines += ["## Frequent source-element co-occurrences for candidate merge review", "", pair_df.head(25).to_markdown(index=False), ""]
    lines += [
        "## How to use this for the reduced meta-model",
        "",
        "1. Retain dimensions whose source elements are frequent across source sentences and LLMs and whose omission is associated with lower cue or lexical preservation.",
        "2. Treat high co-occurrence source-element pairs as merge candidates only when cue preservation remains high and qualitative examples do not reveal lost distinctions.",
        "3. Split candidate dimensions when rows show different modal behavior, different cue-loss patterns, or different reconstruction failures.",
        "4. Use high-score cue-loss and low-content-coverage audit tables as targeted human-review queues before freezing the reduced model.",
        "",
    ]
    (out / "meta_model_evidence_summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", required=True, help="scored_roundtrips.csv from script 09")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--inventory_csv", default=None, help="Optional Union V0 source_element_inventory.csv")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(args.scored_csv)
    if "original_text" not in scored.columns and "source_text" in scored.columns:
        scored["original_text"] = scored["source_text"]
    if "reconstructed_text" not in scored.columns and "reconstructed_sentence" in scored.columns:
        scored["reconstructed_text"] = scored["reconstructed_sentence"]
    if "information_model" not in scored.columns and "info_model" in scored.columns:
        scored["information_model"] = scored["info_model"]

    enriched = add_cue_preservation(scored)
    enriched.to_csv(out / "scored_roundtrips_with_cue_audit.csv", index=False)
    summary = aggregate_summary(enriched)
    cue_long = cue_group_long_summary(enriched)
    modal = modal_transition_summary(enriched)
    summary.to_csv(out / "evaluation_summary_by_condition.csv", index=False)
    cue_long.to_csv(out / "cue_group_preservation_long.csv", index=False)
    modal.to_csv(out / "modal_transition_summary.csv", index=False)
    write_audits(enriched, out)
    element_long, element_evidence, pair_df = build_element_evidence(enriched, args.inventory_csv)
    element_long.to_csv(out / "source_element_mentions_long.csv", index=False)
    element_evidence.to_csv(out / "source_element_evidence_summary.csv", index=False)
    pair_df.to_csv(out / "source_element_cooccurrence_pairs.csv", index=False)
    write_story(summary, cue_long, element_evidence, pair_df, out)
    metadata = {"n_rows": int(len(enriched)), "n_llms": int(enriched["llm"].nunique()) if "llm" in enriched.columns else None, "cue_groups": sorted(CUE_PATTERNS), "critical_cues": sorted(CRITICAL_CUES)}
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote post-score analysis to {out}")


if __name__ == "__main__":
    main()
