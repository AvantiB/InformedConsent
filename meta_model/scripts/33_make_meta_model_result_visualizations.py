#!/usr/bin/env python
"""Generate paper/deck-ready visualizations for meta-model development and evaluation.

This script is tolerant of partially completed runs. It creates any plots for
which the required input files are available and skips the rest with a clear
message.

Input families:
- crosswalk outputs from script 26
- refined/cluster seed outputs from scripts 23/24/28
- classifier training outputs from script 08
- scored/diagnostic outputs from scripts 09/32
- optional LLM-induced consensus mapping outputs

The generated plots are meant for figures or for insertion into the final visual
presentation deck.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


SCORE_CANDIDATES = [
    "mean_classifier_score",
    "meaning_preserved_score",
    "meaning_preservation_score",
    "classifier_score",
    "predicted_probability",
    "probability",
    "score",
    "meaning_preserved_pred_proba",
    "meaning_preserved_pred",
]

CONDITION_ORDER = [
    "individual_source_model_json",
    "union_v0_full_dictionary",
    "functional_v1_manual",
    "functional_v1_llm_induced",
    "functional_v1_llm_induced_consensus",
]

METRIC_LABELS = {
    "mean_classifier_score": "Classifier score",
    "content_word_recall": "Content recall",
    "mean_content_word_recall": "Content recall",
    "important_category_presence_recall": "Cue-category recall",
    "mean_important_category_presence_recall": "Cue-category recall",
    "important_cue_exact_recall": "Exact cue recall",
    "mean_important_cue_exact_recall": "Exact cue recall",
    "modal_category_changed": "Modal category changed",
    "modal_category_change_rate": "Modal category changed",
    "modal_word_change_ratio": "Modal word change",
    "mean_modal_word_change_ratio": "Modal word change",
    "unmatched_language_rate": "Unmatched language",
    "mean_unmatched_language_rate_when_available": "Unmatched language",
    "annotation_count": "Annotations",
    "mean_annotation_count": "Annotations",
    "unique_element_count": "Unique fields",
    "mean_unique_fields": "Unique fields",
    "forward_parse_ok": "Forward parse",
    "forward_parse_rate": "Forward parse",
    "backward_parse_ok": "Backward parse",
    "backward_parse_rate": "Backward parse",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def exists(path: str | Path | None) -> bool:
    return bool(path) and Path(path).exists()


def read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if not exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log(f"SKIP: could not read {path}: {exc}")
        return None


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def label(x: Any, width: int = 24) -> str:
    s = norm(x).replace("_", " ")
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def score_col(df: pd.DataFrame) -> str | None:
    return next((c for c in SCORE_CANDIDATES if c in df.columns), None)


def order_conditions(values: Iterable[Any]) -> list[str]:
    vals = [str(v) for v in values]
    known = [c for c in CONDITION_ORDER if c in vals]
    rest = sorted([v for v in vals if v not in known])
    return known + rest


def savefig(path: Path, title: str | None = None) -> None:
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    log(f"Wrote {path}")


def bar_plot(df: pd.DataFrame, x: str, y: str, out: Path, title: str, xlabel: str = "", ylabel: str = "") -> None:
    work = df[[x, y]].dropna().copy()
    if work.empty:
        return
    fig_w = max(7, min(14, 0.55 * len(work) + 4))
    plt.figure(figsize=(fig_w, 4.8))
    plt.bar(range(len(work)), pd.to_numeric(work[y], errors="coerce"))
    plt.xticks(range(len(work)), [label(v, 18) for v in work[x]], rotation=45, ha="right")
    plt.ylabel(ylabel or METRIC_LABELS.get(y, y))
    plt.xlabel(xlabel)
    savefig(out, title)


def grouped_bar(df: pd.DataFrame, group: str, metrics: list[str], out: Path, title: str) -> None:
    metrics = [m for m in metrics if m in df.columns]
    if not metrics or group not in df.columns:
        return
    work = df[[group] + metrics].drop_duplicates(subset=[group]).copy()
    conds = order_conditions(work[group].astype(str).unique())
    work[group] = work[group].astype(str)
    work = work.set_index(group).reindex(conds).reset_index()
    x = list(range(len(work)))
    width = 0.8 / max(1, len(metrics))
    plt.figure(figsize=(max(8, len(work) * 1.35), 5.2))
    for i, m in enumerate(metrics):
        vals = pd.to_numeric(work[m], errors="coerce")
        offsets = [xx - 0.4 + width / 2 + i * width for xx in x]
        plt.bar(offsets, vals, width=width, label=METRIC_LABELS.get(m, m))
    plt.xticks(x, [label(v, 18) for v in work[group]], rotation=35, ha="right")
    plt.ylabel("Mean value")
    plt.legend(loc="best", fontsize=9)
    savefig(out, title)


def heatmap(table: pd.DataFrame, out: Path, title: str, xlabel: str = "", ylabel: str = "") -> None:
    if table.empty:
        return
    vals = table.apply(pd.to_numeric, errors="coerce").fillna(0).values
    h = max(4.5, min(14, 0.35 * len(table.index) + 2.5))
    w = max(6.5, min(14, 0.65 * len(table.columns) + 3.5))
    plt.figure(figsize=(w, h))
    im = plt.imshow(vals, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(table.columns)), [label(c, 14) for c in table.columns], rotation=45, ha="right")
    plt.yticks(range(len(table.index)), [label(i, 28) for i in table.index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    savefig(out, title)


def plot_condition_scores(diag: pd.DataFrame | None, out_dir: Path) -> None:
    if diag is None or "condition" not in diag.columns:
        return
    sc = score_col(diag)
    if not sc:
        return
    if {"condition", "llm"}.issubset(diag.columns):
        piv = diag.pivot_table(index="condition", columns="llm", values=sc, aggfunc="mean")
        piv = piv.reindex(order_conditions(piv.index))
        heatmap(piv, out_dir / "01_condition_x_llm_classifier_score_heatmap.png", "Meaning-preservation classifier score by condition and LLM", "LLM", "Schema condition")
    overall = diag.groupby("condition", dropna=False)[sc].mean().reset_index()
    overall["condition"] = pd.Categorical(overall["condition"], order_conditions(overall["condition"].astype(str).unique()), ordered=True)
    overall = overall.sort_values("condition")
    bar_plot(overall, "condition", sc, out_dir / "02_classifier_score_by_condition.png", "Mean classifier score by schema condition", ylabel="Mean classifier score")


def plot_diagnostic_metrics(diag: pd.DataFrame | None, out_dir: Path) -> None:
    if diag is None or "condition" not in diag.columns:
        return
    metrics = [
        "content_word_recall",
        "important_category_presence_recall",
        "important_cue_exact_recall",
        "modal_word_change_ratio",
        "unmatched_language_rate",
    ]
    condition_summary = diag.groupby("condition", dropna=False).agg({m: "mean" for m in metrics if m in diag.columns}).reset_index()
    grouped_bar(condition_summary, "condition", [m for m in metrics if m in condition_summary.columns], out_dir / "03_holistic_metrics_by_condition.png", "Holistic preservation diagnostics by schema condition")

    burden = diag.groupby("condition", dropna=False).agg({m: "mean" for m in ["annotation_count", "unique_element_count"] if m in diag.columns}).reset_index()
    grouped_bar(burden, "condition", [m for m in ["annotation_count", "unique_element_count"] if m in burden.columns], out_dir / "04_annotation_burden_by_condition.png", "Annotation burden by schema condition")

    if "suspected_error_flags" in diag.columns:
        rows = []
        for _, r in diag.iterrows():
            cond = r.get("condition", "")
            for flag in str(r.get("suspected_error_flags", "")).split(";"):
                flag = flag.strip()
                if flag:
                    rows.append({"condition": cond, "flag": flag})
        if rows:
            flags = pd.DataFrame(rows)
            top = flags.groupby("flag").size().sort_values(ascending=False).head(10).index
            piv = flags[flags["flag"].isin(top)].pivot_table(index="flag", columns="condition", aggfunc="size", fill_value=0)
            piv = piv.reindex(columns=order_conditions(piv.columns))
            heatmap(piv, out_dir / "05_qualitative_error_flag_heatmap.png", "Heuristic relationship-error flags by condition", "Schema condition", "Flag")


def plot_cue_retention(cue_summary: pd.DataFrame | None, out_dir: Path) -> None:
    if cue_summary is None:
        return
    needed = {"condition", "cue_group", "category_presence_retention_rate"}
    if not needed.issubset(cue_summary.columns):
        return
    piv = cue_summary.pivot_table(index="cue_group", columns="condition", values="category_presence_retention_rate", aggfunc="mean")
    piv = piv.reindex(columns=order_conditions(piv.columns))
    heatmap(piv, out_dir / "06_cue_category_retention_heatmap.png", "Cue-category retention by schema condition", "Schema condition", "Cue group")


def plot_crosswalk(crosswalk_csv: str | Path | None, matrix_csv: str | Path | None, out_dir: Path) -> None:
    mat = read_csv(matrix_csv)
    if mat is not None:
        # Accept either already-wide matrix or long matrix.
        id_cols = [c for c in ["field_id", "functional_field", "v1_field", "field"] if c in mat.columns]
        if id_cols:
            idx = id_cols[0]
            value_cols = [c for c in mat.columns if c != idx and pd.api.types.is_numeric_dtype(mat[c])]
            if value_cols:
                table = mat.set_index(idx)[value_cols]
                heatmap(table, out_dir / "07_functional_role_x_source_model_heatmap.png", "Source-model support across functional roles", "Source model", "Functional role")
                coverage = (table > 0).sum(axis=1).value_counts().sort_index().reset_index()
                coverage.columns = ["n_source_models", "n_roles"]
                bar_plot(coverage, "n_source_models", "n_roles", out_dir / "08_role_source_model_overlap_distribution.png", "How many source models support each role?", xlabel="Number of source models", ylabel="Number of roles")
                return

    cw = read_csv(crosswalk_csv)
    if cw is None:
        return
    src_col = next((c for c in ["source_model", "information_model", "model", "Source model"] if c in cw.columns), None)
    field_col = next((c for c in ["proposed_v1_field", "v1_field", "functional_field", "field_id", "field"] if c in cw.columns), None)
    if not src_col or not field_col:
        log("SKIP: crosswalk CSV lacks source-model or field columns")
        return
    table = cw.pivot_table(index=field_col, columns=src_col, aggfunc="size", fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    heatmap(table, out_dir / "07_functional_role_x_source_model_heatmap.png", "Source-model support across functional roles", "Source model", "Functional role")
    coverage = (table > 0).sum(axis=1).value_counts().sort_index().reset_index()
    coverage.columns = ["n_source_models", "n_roles"]
    bar_plot(coverage, "n_source_models", "n_roles", out_dir / "08_role_source_model_overlap_distribution.png", "How many source models support each role?", xlabel="Number of source models", ylabel="Number of roles")


def plot_seed_cluster_evidence(selected_csv: str | Path | None, stability_csv: str | Path | None, out_dir: Path) -> None:
    sel = read_csv(selected_csv)
    if sel is not None:
        fold_col = next((c for c in ["fold", "fold_id"] if c in sel.columns), None)
        tier_col = next((c for c in ["selection_tier", "tier", "status"] if c in sel.columns), None)
        if fold_col and tier_col:
            table = sel.pivot_table(index=tier_col, columns=fold_col, aggfunc="size", fill_value=0)
            heatmap(table, out_dir / "09_selected_seed_cluster_counts_by_fold.png", "Selected data-driven seed fields by fold", "Fold", "Selection tier")
        support_col = next((c for c in ["positive_mentions", "total_positive_mentions", "n_positive_mentions", "support"] if c in sel.columns), None)
        name_col = next((c for c in ["stability_group_id", "candidate_field_id", "field_id", "schema_field_id"] if c in sel.columns), None)
        if support_col and name_col:
            top = sel[[name_col, support_col]].dropna().copy()
            top[support_col] = pd.to_numeric(top[support_col], errors="coerce")
            top = top.groupby(name_col, as_index=False)[support_col].max().sort_values(support_col, ascending=False).head(20)
            bar_plot(top.sort_values(support_col), name_col, support_col, out_dir / "10_top_seed_clusters_by_support.png", "Top data-driven seed clusters by support", ylabel="Support")

    stab = read_csv(stability_csv)
    if stab is not None:
        fold_count_col = next((c for c in ["n_folds", "fold_count", "recurrence_folds"] if c in stab.columns), None)
        if fold_count_col:
            dist = pd.to_numeric(stab[fold_count_col], errors="coerce").value_counts().sort_index().reset_index()
            dist.columns = ["fold_recurrence", "n_groups"]
            bar_plot(dist, "fold_recurrence", "n_groups", out_dir / "11_cross_fold_seed_recurrence_distribution.png", "Cross-fold recurrence of induced seed groups", xlabel="Number of folds", ylabel="Number of groups")


def plot_llm_consensus(consensus_fields_csv: str | Path | None, fold_mapping_csv: str | Path | None, support_csv: str | Path | None, out_dir: Path) -> None:
    support = read_csv(support_csv)
    if support is not None:
        field_col = next((c for c in ["consensus_field", "consensus_field_id", "field", "name"] if c in support.columns), None)
        folds_col = next((c for c in ["n_folds", "fold_count", "recurrence_folds"] if c in support.columns), None)
        if field_col and folds_col:
            top = support[[field_col, folds_col]].copy()
            top[folds_col] = pd.to_numeric(top[folds_col], errors="coerce")
            top = top.sort_values(folds_col, ascending=False).head(30)
            bar_plot(top.sort_values(folds_col), field_col, folds_col, out_dir / "12_llm_consensus_field_recurrence.png", "LLM-induced consensus fields by fold recurrence", ylabel="Number of folds")

    fmap = read_csv(fold_mapping_csv)
    if fmap is not None:
        cons_col = next((c for c in ["consensus_field", "consensus_field_id", "consensus_name"] if c in fmap.columns), None)
        fold_col = next((c for c in ["fold", "fold_id"] if c in fmap.columns), None)
        if cons_col and fold_col:
            table = fmap.pivot_table(index=cons_col, columns=fold_col, aggfunc="size", fill_value=0)
            heatmap(table, out_dir / "13_llm_fold_field_to_consensus_heatmap.png", "Fold-specific LLM fields mapped to consensus roles", "Fold", "Consensus field")

    fields = read_csv(consensus_fields_csv)
    if fields is not None:
        tier_col = next((c for c in ["tier", "status", "selection_tier"] if c in fields.columns), None)
        if tier_col:
            dist = fields[tier_col].value_counts().reset_index()
            dist.columns = ["tier", "n_fields"]
            bar_plot(dist, "tier", "n_fields", out_dir / "14_llm_consensus_fields_by_tier.png", "Consensus LLM-induced fields by tier", ylabel="Number of fields")


def plot_classifier_insights(training_features_csv: str | Path | None, classifier_bundle: str | Path | None, out_dir: Path) -> None:
    feats = read_csv(training_features_csv)
    # Try to merge labels from roundtrip_dataset.csv in the same classifier folder.
    if feats is not None:
        dataset_path = Path(training_features_csv).with_name("roundtrip_dataset.csv")
        ds = read_csv(dataset_path)
        if ds is not None and {"roundtrip_id", "meaning_preserved"}.issubset(ds.columns) and "roundtrip_id" in feats.columns:
            work = feats.merge(ds[["roundtrip_id", "meaning_preserved"]], on="roundtrip_id", how="inner")
            candidate_metrics = [
                "token_jaccard",
                "tfidf_cosine",
                "length_ratio",
                "modal_category_changed",
                "permission_missing_count",
                "obligation_missing_count",
                "prohibition_missing_count",
                "negation_missing_count",
                "condition_missing_count",
                "restriction_missing_count",
                "withdrawal_missing_count",
                "action_missing_count",
                "resource_missing_count",
                "actor_missing_count",
                "purpose_missing_count",
            ]
            rows = []
            for m in candidate_metrics:
                if m in work.columns:
                    tmp = work.groupby("meaning_preserved")[m].mean()
                    if 0 in tmp.index and 1 in tmp.index:
                        rows.append({"feature": m, "mean_not_preserved": tmp.loc[0], "mean_preserved": tmp.loc[1], "difference_preserved_minus_not": tmp.loc[1] - tmp.loc[0]})
            if rows:
                diffs = pd.DataFrame(rows).sort_values("difference_preserved_minus_not")
                plt.figure(figsize=(8, max(5, 0.32 * len(diffs) + 2)))
                plt.barh(range(len(diffs)), diffs["difference_preserved_minus_not"])
                plt.yticks(range(len(diffs)), [label(x, 28) for x in diffs["feature"]])
                plt.xlabel("Mean difference: preserved minus not preserved")
                savefig(out_dir / "15_classifier_feature_mean_differences.png", "Classifier feature patterns by human meaning-preservation label")
                diffs.to_csv(out_dir / "classifier_feature_mean_differences.csv", index=False)

    if classifier_bundle and exists(classifier_bundle) and joblib is not None:
        try:
            bundle = joblib.load(classifier_bundle)
            pipe = bundle.get("model") if isinstance(bundle, dict) else None
            if pipe is not None and hasattr(pipe, "named_steps") and "clf" in pipe.named_steps:
                importances = getattr(pipe.named_steps["clf"], "feature_importances_", None)
                if importances is not None:
                    try:
                        names = pipe.named_steps["pre"].get_feature_names_out()
                    except Exception:
                        names = bundle.get("feature_columns", [f"feature_{i}" for i in range(len(importances))])
                    imp = pd.DataFrame({"feature": list(names), "importance": importances}).sort_values("importance", ascending=False).head(25)
                    imp["feature"] = imp["feature"].astype(str).str.replace(r"^(num|cat)__", "", regex=True)
                    bar_plot(imp.sort_values("importance"), "feature", "importance", out_dir / "16_classifier_feature_importance.png", "Top classifier feature importances", ylabel="Feature importance")
                    imp.to_csv(out_dir / "classifier_feature_importance_top25.csv", index=False)
        except Exception as exc:
            log(f"SKIP: could not plot classifier importances: {exc}")


def write_manifest(out_dir: Path) -> None:
    rows = []
    for p in sorted(out_dir.glob("*.png")):
        rows.append({"plot_file": p.name, "path": str(p)})
    pd.DataFrame(rows).to_csv(out_dir / "plot_manifest.csv", index=False)
    log(f"Wrote {out_dir / 'plot_manifest.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root", default="meta_model/functional_v1_experiments")
    ap.add_argument("--output_dir", default="", help="Defaults to <out_root>/plots")
    ap.add_argument("--diagnostic_csv", default="", help="Defaults to <out_root>/diagnostics/roundtrip_diagnostic_metrics.csv")
    ap.add_argument("--cue_summary_csv", default="", help="Defaults to <out_root>/diagnostics/cue_group_retention_summary_by_condition.csv")
    ap.add_argument("--crosswalk_csv", default="meta_model/functional_v1/crosswalk/functional_v1_crosswalk.csv")
    ap.add_argument("--crosswalk_matrix_csv", default="meta_model/functional_v1/crosswalk/functional_v1_model_field_matrix.csv")
    ap.add_argument("--selected_fields_csv", default="meta_model/refined_cv/field_selection_strict/selected_fields_long.csv")
    ap.add_argument("--field_stability_csv", default="meta_model/refined_cv/field_selection_strict/selected_field_stability_summary.csv")
    ap.add_argument("--llm_consensus_fields_csv", default="")
    ap.add_argument("--llm_consensus_support_csv", default="")
    ap.add_argument("--llm_fold_mapping_csv", default="")
    ap.add_argument("--training_features_csv", default="meta_model/outputs/final_classifier/training_features.csv")
    ap.add_argument("--classifier_bundle", default="meta_model/outputs/final_classifier/final_meaning_preservation_classifier.joblib")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_dir = Path(args.output_dir) if args.output_dir else out_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_csv = args.diagnostic_csv or str(out_root / "diagnostics" / "roundtrip_diagnostic_metrics.csv")
    cue_summary_csv = args.cue_summary_csv or str(out_root / "diagnostics" / "cue_group_retention_summary_by_condition.csv")

    diag = read_csv(diagnostic_csv)
    cue_summary = read_csv(cue_summary_csv)

    plot_condition_scores(diag, out_dir)
    plot_diagnostic_metrics(diag, out_dir)
    plot_cue_retention(cue_summary, out_dir)
    plot_crosswalk(args.crosswalk_csv, args.crosswalk_matrix_csv, out_dir)
    plot_seed_cluster_evidence(args.selected_fields_csv, args.field_stability_csv, out_dir)
    plot_llm_consensus(args.llm_consensus_fields_csv, args.llm_fold_mapping_csv, args.llm_consensus_support_csv, out_dir)
    plot_classifier_insights(args.training_features_csv, args.classifier_bundle, out_dir)
    write_manifest(out_dir)


if __name__ == "__main__":
    main()
