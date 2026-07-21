#!/usr/bin/env python
"""Compute holistic diagnostic metrics for round-trip meaning preservation.

This script complements the classifier score with interpretable preservation signals:
- lexical/content-word preservation
- cue/category preservation using the classifier cue dictionary when available
- modal cue/category changes
- unmatched-language rate when present in functional-schema outputs
- heuristic qualitative relationship-error flags and review samples

The qualitative flags are triage signals for manual review, not ground-truth error labels.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:  # joblib is optional unless --classifier_bundle is used
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


DEFAULT_CUE_GROUPS: dict[str, list[str]] = {
    "permission": ["may", "can", "could", "allowed", "allow", "permitted", "permit", "authorized", "agree", "consent"],
    "obligation": ["must", "should", "required", "require", "need to", "have to", "responsible", "obligated", "duty"],
    "prohibition": ["may not", "cannot", "can't", "will not", "not allowed", "not permitted", "prohibited", "restricted", "no "],
    "negation": ["not", "no", "never", "cannot", "can't", "without", "neither", "nor"],
    "condition": ["if", "when", "unless", "only if", "as long as", "provided that", "until", "before", "after", "during"],
    "exception": ["however", "except", "but", "although", "nevertheless", "already", "prior", "except that"],
    "restriction": ["only", "limited", "limit", "restriction", "restricted", "commercial", "non-commercial", "identifiable", "de-identified", "geographic", "institution", "approved", "irb", "ethics", "no expiration", "at any time"],
    "withdrawal": ["withdraw", "revoke", "quit", "stop", "withdrawal"],
    "action": ["use", "used", "store", "stored", "share", "shared", "disclose", "disclosed", "collect", "collected", "withdraw", "revoke", "destroy", "retain", "contact", "return", "access", "sell", "distribute", "retrieve", "study", "analyze", "learn"],
    "resource": ["data", "information", "health information", "medical record", "records", "dna", "sample", "samples", "specimen", "specimens", "biospecimen", "blood", "urine", "saliva", "results", "database", "databases"],
    "actor": ["researcher", "researchers", "doctor", "doctors", "study team", "institution", "sponsor", "company", "biobank", "irb", "university", "clinic", "hospital", "all of us", "mayo"],
    "purpose": ["research", "future research", "cancer", "genetic", "genomic", "commercial", "clinical care", "public health", "study", "studies"],
}

# Evaluation-only groups are not necessarily classifier features. They make the final
# diagnostic tables more interpretable for consent-language review.
EVAL_ONLY_CUE_GROUPS: dict[str, list[str]] = {
    "temporal": ["future", "at any time", "ten years", "years", "until", "after", "before", "during", "ongoing", "long as", "no expiration", "later"],
    "privacy_identifiability": ["private", "privacy", "confidential", "confidentiality", "identified", "identifiable", "de-identified", "deidentified", "anonymous", "coded"],
    "results_feedback": ["results", "return results", "feedback", "findings", "report back", "tell you", "notify"],
    "storage_lifecycle": ["store", "stored", "storage", "retain", "retained", "keep", "kept", "destroy", "destroyed", "dispose", "delete", "last", "continue", "stop"],
    "contact_recontact": ["contact", "recontact", "call", "email", "notify", "ask you", "request"],
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", "of", "to", "in", "on", "for", "with", "by", "from", "as", "at", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "it", "its", "they", "them", "their", "there", "we", "us", "our", "you", "your", "i", "me", "my", "he", "she", "his", "her", "will", "would", "can", "could", "may", "might", "should", "must", "do", "does", "did", "have", "has", "had", "not", "no", "yes", "also", "other", "any", "all", "some", "such", "than", "into", "about", "under", "over", "between", "within", "without", "because", "so", "up", "out", "off", "only",
}

TEXT_COL_CANDIDATES = {
    "original_text": ["original_text", "source_text", "canonical_full_text", "full_text_original", "sentence", "sentence_text"],
    "reconstructed_text": ["reconstructed_text", "reconstructed_sentence", "backward_mapping", "reconstruction"],
    "forward_mapping": ["forward_mapping", "annotations_serialized", "annotations_combined", "mapping"],
}

SCORE_CANDIDATES = [
    "meaning_preserved_score",
    "meaning_preservation_score",
    "classifier_score",
    "predicted_probability",
    "probability",
    "score",
    "meaning_preserved_pred_proba",
    "meaning_preserved_pred",
]


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def low(x: Any) -> str:
    return norm(x).lower()


def safe_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")


def choose_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried {candidates}; available={list(df.columns)}")
    return None


def tokens(text: Any) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", low(text))


def content_tokens(text: Any) -> list[str]:
    return [t for t in tokens(text) if len(t) > 1 and t not in STOPWORDS]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def recall(a: set[str], b: set[str]) -> float | None:
    if not a:
        return None
    return len(a & b) / len(a)


def precision(a: set[str], b: set[str]) -> float | None:
    if not b:
        return None
    return len(a & b) / len(b)


def f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def cue_set(text: Any, cues: list[str]) -> set[str]:
    t = low(text)
    found: set[str] = set()
    for cue in cues:
        c = cue.lower()
        pat = re.escape(c) if " " in c or not c.isalnum() else r"\b" + re.escape(c) + r"\b"
        if re.search(pat, t):
            found.add(c)
    return found


def parse_jsonish(text: Any) -> Any:
    s = norm(text)
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|csv|yaml)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    for l, r in [("{", "}"), ("[", "]")]:
        a, b = s.find(l), s.rfind(r)
        if a >= 0 and b > a:
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                pass
    return None


def flatten_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        if norm(obj):
            out.append(norm(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_strings(v))
    else:
        s = norm(obj)
        if s:
            out.append(s)
    return out


def find_unmatched_language(obj: Any) -> tuple[bool, list[str]]:
    keys = {
        "unmatched_language",
        "unmatched_language_spans",
        "unmatched_spans",
        "unmapped_language",
        "unmapped_spans",
        "not_annotated",
        "not_covered",
    }
    found_key = False
    values: list[str] = []

    def rec(x: Any) -> None:
        nonlocal found_key, values
        if isinstance(x, dict):
            for k, v in x.items():
                if safe_name(k) in keys:
                    found_key = True
                    values.extend(flatten_strings(v))
                rec(v)
        elif isinstance(x, list):
            for item in x:
                rec(item)

    rec(obj)
    # De-duplicate while preserving order.
    seen = set()
    deduped = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return found_key, deduped


def load_cue_groups(args: argparse.Namespace) -> tuple[dict[str, list[str]], dict[str, Any]]:
    cue_groups = {k: list(v) for k, v in DEFAULT_CUE_GROUPS.items()}
    source = {"classifier_bundle": None, "cue_dictionary": None, "used_default_classifier_groups": True, "added_eval_only_groups": sorted(EVAL_ONLY_CUE_GROUPS)}

    if args.classifier_bundle:
        if joblib is None:
            raise RuntimeError("joblib is required to read --classifier_bundle")
        bundle = joblib.load(args.classifier_bundle)
        source["classifier_bundle"] = str(args.classifier_bundle)
        cue_dict = bundle.get("cue_dictionary") if isinstance(bundle, dict) else None
        if isinstance(cue_dict, dict) and isinstance(cue_dict.get("cue_groups"), dict):
            cue_groups = {str(k): [str(x).lower() for x in v] for k, v in cue_dict["cue_groups"].items()}
            source["used_default_classifier_groups"] = False

    if args.cue_dictionary:
        with Path(args.cue_dictionary).open() as f:
            cue_dict = json.load(f)
        if "cue_groups" not in cue_dict:
            raise ValueError("Cue dictionary JSON must contain cue_groups.")
        cue_groups = {str(k): [str(x).lower() for x in v] for k, v in cue_dict["cue_groups"].items()}
        source["cue_dictionary"] = str(args.cue_dictionary)
        source["used_default_classifier_groups"] = False

    # Add evaluation-only groups unless they already exist.
    for name, cues in EVAL_ONLY_CUE_GROUPS.items():
        cue_groups.setdefault(name, cues)
    return cue_groups, source


def modal_category(permission: set[str], obligation: set[str], prohibition: set[str]) -> str:
    if prohibition:
        return "prohibition"
    if obligation:
        return "obligation"
    if permission:
        return "permission"
    return "none"


def add_row_metrics(df: pd.DataFrame, cue_groups: dict[str, list[str]], score_col: str | None) -> pd.DataFrame:
    orig_col = choose_col(df, TEXT_COL_CANDIDATES["original_text"])
    rec_col = choose_col(df, TEXT_COL_CANDIDATES["reconstructed_text"])
    fwd_col = choose_col(df, TEXT_COL_CANDIDATES["forward_mapping"], required=False)

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        orig = r.get(orig_col, "")
        rec = r.get(rec_col, "")
        ot = set(content_tokens(orig))
        rt = set(content_tokens(rec))
        p = precision(ot, rt)
        rr = recall(ot, rt)
        row: dict[str, Any] = r.to_dict()
        row.update({
            "orig_content_word_count": len(ot),
            "recon_content_word_count": len(rt),
            "content_word_overlap_count": len(ot & rt),
            "content_word_recall": rr,
            "content_word_precision": p,
            "content_word_f1": f1(p, rr),
            "content_word_jaccard": jaccard(ot, rt),
            "missing_content_word_count": len(ot - rt),
            "added_content_word_count": len(rt - ot),
            "dropped_content_word_rate": (len(ot - rt) / len(ot)) if ot else None,
            "added_content_word_rate": (len(rt - ot) / len(rt)) if rt else None,
            "missing_content_words": "; ".join(sorted(ot - rt)),
            "added_content_words": "; ".join(sorted(rt - ot)),
        })

        orig_categories = 0
        retained_categories = 0
        all_orig_cues: set[str] = set()
        all_recon_cues: set[str] = set()
        cue_sets: dict[str, tuple[set[str], set[str]]] = {}
        for name, cues in cue_groups.items():
            sname = safe_name(name)
            a, b = cue_set(orig, cues), cue_set(rec, cues)
            cue_sets[sname] = (a, b)
            all_orig_cues |= {f"{sname}:{x}" for x in a}
            all_recon_cues |= {f"{sname}:{x}" for x in b}
            if a:
                orig_categories += 1
                if b:
                    retained_categories += 1
            row[f"orig_{sname}_count"] = len(a)
            row[f"recon_{sname}_count"] = len(b)
            row[f"{sname}_recall"] = recall(a, b)
            row[f"{sname}_jaccard"] = jaccard(a, b)
            row[f"{sname}_missing_count"] = len(a - b)
            row[f"{sname}_added_count"] = len(b - a)
            row[f"{sname}_presence_preserved"] = float(bool(a) == bool(b))
            row[f"{sname}_category_retained"] = float((not a) or bool(b))
            row[f"orig_{sname}_cues"] = "; ".join(sorted(a))
            row[f"recon_{sname}_cues"] = "; ".join(sorted(b))
            row[f"{sname}_missing_cues"] = "; ".join(sorted(a - b))
            row[f"{sname}_added_cues"] = "; ".join(sorted(b - a))

        row["important_category_orig_count"] = orig_categories
        row["important_category_retained_count"] = retained_categories
        row["important_category_presence_recall"] = retained_categories / orig_categories if orig_categories else None
        row["important_cue_exact_recall"] = recall(all_orig_cues, all_recon_cues)
        row["important_cue_jaccard"] = jaccard(all_orig_cues, all_recon_cues)

        op, rp = cue_sets.get("permission", (set(), set()))
        oo, ro = cue_sets.get("obligation", (set(), set()))
        opr, rpr = cue_sets.get("prohibition", (set(), set()))
        orig_modal = op | oo | opr
        recon_modal = rp | ro | rpr
        row["modal_orig"] = modal_category(op, oo, opr)
        row["modal_recon"] = modal_category(rp, ro, rpr)
        row["modal_category_changed"] = float(row["modal_orig"] != row["modal_recon"])
        row["modal_word_jaccard"] = jaccard(orig_modal, recon_modal)
        row["modal_word_recall"] = recall(orig_modal, recon_modal)
        row["modal_word_change_ratio"] = 1.0 - jaccard(orig_modal, recon_modal) if (orig_modal or recon_modal) else 0.0
        row["modal_missing_count"] = len(orig_modal - recon_modal)
        row["modal_added_count"] = len(recon_modal - orig_modal)
        row["modal_missing_cues"] = "; ".join(sorted(orig_modal - recon_modal))
        row["modal_added_cues"] = "; ".join(sorted(recon_modal - orig_modal))

        unmatched_available = False
        unmatched_values: list[str] = []
        if fwd_col:
            parsed = parse_jsonish(r.get(fwd_col, ""))
            if parsed is not None:
                unmatched_available, unmatched_values = find_unmatched_language(parsed)
        unmatched_tokens = set()
        for x in unmatched_values:
            unmatched_tokens |= set(content_tokens(x))
        row["unmatched_language_available"] = float(unmatched_available)
        row["unmatched_language_count"] = len(unmatched_values) if unmatched_available else None
        row["unmatched_language_token_count"] = len(unmatched_tokens) if unmatched_available else None
        row["unmatched_language_rate"] = (len(unmatched_tokens) / len(ot)) if unmatched_available and ot else (0.0 if unmatched_available else None)
        row["unmatched_language_text"] = " | ".join(unmatched_values)

        flags: list[str] = []
        if row["modal_category_changed"]:
            flags.append("modal_or_permission_category_changed")
        if row.get("prohibition_missing_count", 0) > 0:
            flags.append("prohibition_cue_dropped")
        if row.get("negation_missing_count", 0) > 0:
            flags.append("negation_cue_dropped")
        if any(row.get(f"{g}_missing_count", 0) > 0 for g in ["condition", "restriction", "exception"]):
            flags.append("condition_scope_or_exception_cue_dropped")
        if row.get("withdrawal_missing_count", 0) > 0:
            flags.append("withdrawal_choice_cue_dropped")
        if row.get("action_missing_count", 0) > 0 or row.get("action_added_count", 0) > 0:
            flags.append("governed_action_changed")
        if row.get("resource_missing_count", 0) > 0 or row.get("resource_added_count", 0) > 0:
            flags.append("governed_resource_changed")
        if row.get("actor_missing_count", 0) > 0 or row.get("actor_added_count", 0) > 0:
            flags.append("actor_or_recipient_changed")
        if row.get("purpose_missing_count", 0) > 0 or row.get("purpose_added_count", 0) > 0:
            flags.append("purpose_or_use_context_changed")
        if row.get("temporal_missing_count", 0) > 0 or row.get("temporal_added_count", 0) > 0:
            flags.append("temporal_expression_changed")
        if row.get("privacy_identifiability_missing_count", 0) > 0 or row.get("privacy_identifiability_added_count", 0) > 0:
            flags.append("privacy_identifiability_changed")
        if row["content_word_recall"] is not None and row["content_word_recall"] < 0.60:
            flags.append("substantial_content_loss")
        if row["content_word_precision"] is not None and row["content_word_precision"] < 0.60:
            flags.append("substantial_added_content")
        if score_col and pd.notna(pd.to_numeric(row.get(score_col), errors="coerce")):
            try:
                if float(row.get(score_col)) < 0.50:
                    flags.append("low_classifier_score")
            except Exception:
                pass
        row["suspected_error_flags"] = "; ".join(flags)
        row["suspected_error_count"] = len(flags)
        rows.append(row)

    return pd.DataFrame(rows)


def numeric_mean(s: pd.Series) -> float:
    return pd.to_numeric(s, errors="coerce").mean()


def bool_rate(s: pd.Series) -> float:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).mean()


def summarize(df: pd.DataFrame, group_cols: list[str], score_col: str | None) -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        cols = ["__all__"]
        df = df.copy()
        df["__all__"] = "all"
    agg: dict[str, Any] = {
        "roundtrip_id": "count" if "roundtrip_id" in df.columns else lambda x: len(x),
        "content_word_recall": numeric_mean,
        "content_word_precision": numeric_mean,
        "content_word_f1": numeric_mean,
        "content_word_jaccard": numeric_mean,
        "dropped_content_word_rate": numeric_mean,
        "added_content_word_rate": numeric_mean,
        "important_category_presence_recall": numeric_mean,
        "important_cue_exact_recall": numeric_mean,
        "important_cue_jaccard": numeric_mean,
        "modal_category_changed": numeric_mean,
        "modal_word_change_ratio": numeric_mean,
        "unmatched_language_available": numeric_mean,
        "unmatched_language_rate": numeric_mean,
        "annotation_count": numeric_mean,
        "unique_element_count": numeric_mean,
        "forward_parse_ok": numeric_mean,
        "backward_parse_ok": numeric_mean,
        "suspected_error_count": numeric_mean,
    }
    if score_col:
        agg[score_col] = numeric_mean
    use_agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.groupby(cols, dropna=False).agg(use_agg).reset_index()
    rename = {
        "roundtrip_id": "n",
        "content_word_recall": "mean_content_word_recall",
        "content_word_precision": "mean_content_word_precision",
        "content_word_f1": "mean_content_word_f1",
        "content_word_jaccard": "mean_content_word_jaccard",
        "dropped_content_word_rate": "mean_dropped_content_word_rate",
        "added_content_word_rate": "mean_added_content_word_rate",
        "important_category_presence_recall": "mean_important_category_presence_recall",
        "important_cue_exact_recall": "mean_important_cue_exact_recall",
        "important_cue_jaccard": "mean_important_cue_jaccard",
        "modal_category_changed": "modal_category_change_rate",
        "modal_word_change_ratio": "mean_modal_word_change_ratio",
        "unmatched_language_available": "unmatched_language_availability_rate",
        "unmatched_language_rate": "mean_unmatched_language_rate_when_available",
        "annotation_count": "mean_annotation_count",
        "unique_element_count": "mean_unique_fields",
        "forward_parse_ok": "forward_parse_rate",
        "backward_parse_ok": "backward_parse_rate",
        "suspected_error_count": "mean_suspected_error_flags",
    }
    if score_col:
        rename[score_col] = "mean_classifier_score"
    return out.rename(columns=rename)


def cue_group_summary(df: pd.DataFrame, cue_groups: dict[str, list[str]], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cols = [c for c in group_cols if c in df.columns]
    grouped = df.groupby(cols, dropna=False) if cols else [((), df)]
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = {c: v for c, v in zip(cols, key)}
        for name in cue_groups:
            sname = safe_name(name)
            orig_col = f"orig_{sname}_count"
            recon_col = f"recon_{sname}_count"
            recall_col = f"{sname}_recall"
            missing_col = f"{sname}_missing_count"
            added_col = f"{sname}_added_count"
            retained_col = f"{sname}_category_retained"
            if orig_col not in sub.columns:
                continue
            orig_positive = pd.to_numeric(sub[orig_col], errors="coerce").fillna(0) > 0
            denom = int(orig_positive.sum())
            retained_rate = pd.to_numeric(sub.loc[orig_positive, retained_col], errors="coerce").mean() if denom else math.nan
            row = dict(base)
            row.update({
                "cue_group": sname,
                "n_rows": int(len(sub)),
                "n_rows_with_original_cue": denom,
                "original_cue_prevalence": denom / len(sub) if len(sub) else math.nan,
                "category_presence_retention_rate": retained_rate,
                "mean_exact_cue_recall_when_original_present": pd.to_numeric(sub.loc[orig_positive, recall_col], errors="coerce").mean() if denom else math.nan,
                "mean_missing_cues_per_row": pd.to_numeric(sub[missing_col], errors="coerce").mean(),
                "mean_added_cues_per_row": pd.to_numeric(sub[added_col], errors="coerce").mean(),
                "mean_original_cue_count": pd.to_numeric(sub[orig_col], errors="coerce").mean(),
                "mean_reconstructed_cue_count": pd.to_numeric(sub[recon_col], errors="coerce").mean(),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def review_sample(df: pd.DataFrame, score_col: str | None, per_condition: int) -> pd.DataFrame:
    if per_condition <= 0 or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if score_col and score_col in work.columns:
        work["_score_sort"] = pd.to_numeric(work[score_col], errors="coerce")
    else:
        work["_score_sort"] = 1.0
    work["_recall_sort"] = pd.to_numeric(work.get("content_word_recall", 1.0), errors="coerce")
    work["_err_sort"] = pd.to_numeric(work.get("suspected_error_count", 0), errors="coerce").fillna(0)
    sort_cols = ["_err_sort", "_score_sort", "_recall_sort"]
    ascending = [False, True, True]
    group_cols = [c for c in ["condition", "information_model", "llm"] if c in work.columns]
    if not group_cols:
        sample = work.sort_values(sort_cols, ascending=ascending).head(per_condition)
    else:
        sample = (
            work.sort_values(sort_cols, ascending=ascending)
            .groupby(group_cols, dropna=False)
            .head(per_condition)
        )
    keep = [
        c for c in [
            "roundtrip_id", "source_id", "condition", "information_model", "llm", score_col,
            "source_text", "original_text", "reconstructed_sentence", "reconstructed_text",
            "annotation_count", "unique_element_count", "content_word_recall", "content_word_precision",
            "missing_content_words", "added_content_words", "modal_orig", "modal_recon",
            "modal_missing_cues", "modal_added_cues", "important_category_presence_recall",
            "unmatched_language_rate", "unmatched_language_text", "suspected_error_flags",
        ] if c and c in sample.columns
    ]
    return sample[keep].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roundtrips_csv", required=True, help="Scored or standardized roundtrip CSV.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classifier_bundle", default="", help="Optional final classifier joblib bundle. Used to recover the exact classifier cue dictionary.")
    ap.add_argument("--cue_dictionary", default="", help="Optional JSON cue dictionary with cue_groups; overrides bundle/default cue groups.")
    ap.add_argument("--score_column", default="", help="Optional classifier score column name.")
    ap.add_argument("--review_sample_per_condition", type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.roundtrips_csv)
    score_col = args.score_column or next((c for c in SCORE_CANDIDATES if c in df.columns), None)
    cue_groups, cue_source = load_cue_groups(args)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    enriched = add_row_metrics(df, cue_groups, score_col)
    enriched.to_csv(out / "roundtrip_diagnostic_metrics.csv", index=False)
    summarize(enriched, ["condition"], score_col).to_csv(out / "condition_diagnostic_summary.csv", index=False)
    summarize(enriched, ["condition", "information_model", "llm"], score_col).to_csv(out / "condition_llm_diagnostic_summary.csv", index=False)
    summarize(enriched, ["condition", "information_model"], score_col).to_csv(out / "condition_information_model_diagnostic_summary.csv", index=False)
    cue_group_summary(enriched, cue_groups, ["condition"]).to_csv(out / "cue_group_retention_summary_by_condition.csv", index=False)
    cue_group_summary(enriched, cue_groups, ["condition", "information_model", "llm"]).to_csv(out / "cue_group_retention_summary_by_condition_llm.csv", index=False)
    review_sample(enriched, score_col, args.review_sample_per_condition).to_csv(out / "qualitative_relationship_error_review_sample.csv", index=False)

    dictionary_payload = {
        "description": "Cue groups used for diagnostic retention metrics. Classifier cue groups are loaded from the final classifier bundle or supplied JSON when available; evaluation-only groups are added for richer paper-facing diagnostics.",
        "source": cue_source,
        "cue_groups": cue_groups,
        "modal_priority": ["prohibition", "obligation", "permission", "none"],
        "qualitative_flags_are_heuristic": True,
    }
    (out / "evaluation_dictionary_used.json").write_text(json.dumps(dictionary_payload, indent=2, ensure_ascii=False))

    print(f"Wrote row-level diagnostics to {out / 'roundtrip_diagnostic_metrics.csv'}")
    print(f"Wrote condition summary to {out / 'condition_diagnostic_summary.csv'}")
    print(f"Wrote qualitative review sample to {out / 'qualitative_relationship_error_review_sample.csv'}")


if __name__ == "__main__":
    main()
