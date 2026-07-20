#!/usr/bin/env python
"""Visualize empirical Reduced V1 discovery outputs.

This script is meant to support audit and manuscript figures. It keeps the two
interpretations separate:

1. semantic-equivalence evidence: candidate field merges;
2. provision-bundle evidence: compositional co-occurrence, not merge evidence.

Input is the output directory from 17_induce_reduced_v1_metamodel.py.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def load_json_list(x: Any) -> list[str]:
    s = norm(x)
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [norm(a) for a in v if norm(a)]
    except Exception:
        pass
    return [s]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("") if path.exists() else pd.DataFrame()


def source_model_from_id(uid: str) -> str:
    return str(uid).split("::", 1)[0] if "::" in str(uid) else "unknown"


def compute_cluster_support(clusters: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clusters = clusters.copy()
    clusters["source_model_from_id"] = clusters["union_element_id"].map(source_model_from_id)
    for cid, g in clusters.groupby("semantic_cluster_id"):
        elements = g["union_element_id"].astype(str).tolist()
        models = sorted(g["source_model_from_id"].astype(str).unique().tolist())
        m = mentions[mentions["union_element_id"].astype(str).isin(elements)].copy()
        if m.empty:
            pm = m
            nm = m
        else:
            pm = m[m["expert_meaning_preserved"].astype(bool)]
            nm = m[~m["expert_meaning_preserved"].astype(bool)]
        sentence_col = "sentence_key" if "sentence_key" in pm.columns else "source_id"
        span_counts = Counter([s for s in pm.get("span_text", pd.Series(dtype=str)).astype(str) if s and s.lower() != "nan"])
        rows.append({
            "semantic_cluster_id": cid,
            "n_elements": len(elements),
            "source_models": ", ".join(models),
            "n_source_models_in_cluster": len(models),
            "positive_source_sentences": pm[sentence_col].nunique() if sentence_col in pm.columns else 0,
            "positive_contexts": pm["context_id"].nunique() if "context_id" in pm.columns else 0,
            "positive_mentions": len(pm),
            "negative_contexts": nm["context_id"].nunique() if "context_id" in nm.columns else 0,
            "expert_positive_rate": len(pm) / max(1, len(m)),
            "top_elements": "; ".join(elements[:12]),
            "top_spans": "; ".join([x for x, _ in span_counts.most_common(10)]),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["positive_source_sentences", "n_source_models_in_cluster"], ascending=[False, False])


def plot_cluster_support(cluster_support: pd.DataFrame, output_png: Path) -> None:
    if cluster_support.empty:
        return
    df = cluster_support.head(12).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["semantic_cluster_id"], df["positive_source_sentences"])
    ax.invert_yaxis()
    ax.set_xlabel("Expert-preserved source sentences")
    ax.set_ylabel("Semantic cluster")
    ax.set_title("Semantic clusters by expert-preserved sentence coverage")
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(row.positive_source_sentences, i, f"  {int(row.positive_source_sentences)} sent.; {int(row.n_source_models_in_cluster)} models", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_source_model_heatmap(clusters: pd.DataFrame, cluster_support: pd.DataFrame, output_png: Path) -> None:
    if clusters.empty or cluster_support.empty:
        return
    df = clusters.copy()
    df["source_model_from_id"] = df["union_element_id"].map(source_model_from_id)
    models = sorted(df["source_model_from_id"].astype(str).unique().tolist())
    heat = df.groupby(["semantic_cluster_id", "source_model_from_id"]).size().unstack(fill_value=0)
    heat = heat.reindex(index=cluster_support["semantic_cluster_id"], columns=models, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(heat))))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Source information model")
    ax.set_ylabel("Semantic cluster")
    ax.set_title("Source-model composition of semantic clusters")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = int(heat.values[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Number of source elements")
    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def aggregate_semantic_cluster_edges(edges: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or clusters.empty:
        return pd.DataFrame()
    cmap = dict(zip(clusters["union_element_id"].astype(str), clusters["semantic_cluster_id"].astype(str)))
    bucket: dict[tuple[str, str], list[float]] = {}
    for _, r in edges.iterrows():
        ca = cmap.get(str(r.get("union_element_id_a", "")))
        cb = cmap.get(str(r.get("union_element_id_b", "")))
        if not ca or not cb or ca == cb:
            continue
        key = tuple(sorted((ca, cb)))
        bucket.setdefault(key, []).append(float(r.get("semantic_edge_weight", 0)))
    rows = []
    for (a, b), weights in bucket.items():
        rows.append({
            "cluster_a": a,
            "cluster_b": b,
            "n_semantic_edges": len(weights),
            "mean_weight": sum(weights) / max(1, len(weights)),
            "max_weight": max(weights),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["n_semantic_edges", "mean_weight"], ascending=[False, False])


def plot_network(edge_df: pd.DataFrame, cluster_support: pd.DataFrame, output_png: Path, title: str, weight_col: str) -> None:
    if nx is None or cluster_support.empty:
        return
    graph = nx.Graph()
    for _, r in cluster_support.iterrows():
        graph.add_node(str(r["semantic_cluster_id"]), size=float(r["positive_source_sentences"]), label=f"{r['semantic_cluster_id']}\n{int(r['positive_source_sentences'])} sent.")
    for _, r in edge_df.iterrows():
        a = str(r.iloc[0])
        b = str(r.iloc[1])
        w = float(r.get(weight_col, 1))
        graph.add_edge(a, b, weight=w)
    if graph.number_of_nodes() == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=7, weight="weight", k=0.9)
    sizes = [max(300, min(2500, graph.nodes[n].get("size", 1) * 7)) for n in graph.nodes]
    weights = [graph.edges[e].get("weight", 1) for e in graph.edges]
    max_w = max(weights) if weights else 1
    widths = [max(0.8, min(6, 6 * w / max_w)) for w in weights]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, alpha=0.35)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, alpha=0.85)
    nx.draw_networkx_labels(graph, pos, labels={n: graph.nodes[n].get("label", n) for n in graph.nodes}, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def write_markdown_report(out: Path, cluster_support: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Reduced V1 discovery visual audit report",
        "",
        "This report distinguishes semantic-equivalence evidence from provision-bundle evidence.",
        "Semantic-equivalence clusters are candidate field merges. Provision-bundle edges describe composition and are not merge evidence.",
        "",
        "## Diagnostics",
        f"- Raw source-element mentions: {len(inputs['mentions']):,}",
        f"- Unique source elements: {inputs['profiles']['union_element_id'].nunique() if not inputs['profiles'].empty else 0:,}",
        f"- Semantic-equivalence edges: {len(inputs['semantic_edges']):,}",
        f"- Semantic clusters: {inputs['clusters']['semantic_cluster_id'].nunique() if not inputs['clusters'].empty else 0:,}",
        f"- Provision-bundle edges: {len(inputs['bundle_edges']):,}",
        "",
        "## Corrected cluster support",
        "The table recomputes cluster-level source-model support from cluster membership. This is important because each source element may come from one information-model condition, while an empirical cluster can still span several source models.",
        "",
    ]
    if not cluster_support.empty:
        cols = ["semantic_cluster_id", "n_elements", "n_source_models_in_cluster", "source_models", "positive_source_sentences", "positive_contexts", "positive_mentions", "expert_positive_rate", "top_elements", "top_spans"]
        lines.append(cluster_support[cols].to_markdown(index=False))
    lines += [
        "",
        "## Visuals",
        "",
        "![Semantic cluster support](semantic_cluster_support.png)",
        "",
        "![Source-model heatmap](semantic_cluster_source_model_heatmap.png)",
        "",
        "![Semantic cluster network](semantic_cluster_network.png)",
        "",
        "![Provision-bundle network](provision_bundle_cluster_network.png)",
        "",
        "## Interpretation guide",
        "",
        "- Semantic cluster support defends that candidate fields are induced from expert-preserved annotation behavior.",
        "- Source-model composition shows whether a candidate field has cross-model evidence.",
        "- Semantic cluster networks show merge pressure among clusters.",
        "- Provision-bundle networks show how fields combine in consent provisions; these edges support provision structure, not merging.",
    ]
    (out / "v1_discovery_visual_audit_report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    discovery = Path(args.discovery_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    inputs = {
        "mentions": read_csv_if_exists(discovery / "expert_element_mentions_long.csv"),
        "profiles": read_csv_if_exists(discovery / "expert_element_profiles.csv"),
        "clusters": read_csv_if_exists(discovery / "semantic_equivalence_clusters.csv"),
        "semantic_edges": read_csv_if_exists(discovery / "semantic_equivalence_edges.csv"),
        "bundle_edges": read_csv_if_exists(discovery / "provision_bundle_edges.csv"),
        "bundle_by_cluster": read_csv_if_exists(discovery / "provision_bundle_summary_by_semantic_cluster.csv"),
    }

    support = compute_cluster_support(inputs["clusters"], inputs["mentions"])
    support.to_csv(out / "cluster_support_corrected.csv", index=False)

    sem_cluster_edges = aggregate_semantic_cluster_edges(inputs["semantic_edges"], inputs["clusters"])
    sem_cluster_edges.to_csv(out / "semantic_cluster_edge_summary.csv", index=False)
    inputs["semantic_edges"].head(50).to_csv(out / "top_semantic_equivalence_edges.csv", index=False)
    inputs["bundle_edges"].head(50).to_csv(out / "top_provision_bundle_edges.csv", index=False)

    plot_cluster_support(support, out / "semantic_cluster_support.png")
    plot_source_model_heatmap(inputs["clusters"], support, out / "semantic_cluster_source_model_heatmap.png")
    plot_network(sem_cluster_edges, support, out / "semantic_cluster_network.png", "Semantic-equivalence relationships between clusters", "mean_weight")
    bundle_cluster = inputs["bundle_by_cluster"].copy()
    if not bundle_cluster.empty:
        bundle_cluster = bundle_cluster.rename(columns={"semantic_cluster_id_a": "cluster_a", "semantic_cluster_id_b": "cluster_b"})
        plot_network(bundle_cluster[["cluster_a", "cluster_b", "positive_cooccurrence_contexts"]].head(25), support, out / "provision_bundle_cluster_network.png", "Provision-bundle co-occurrence between semantic clusters", "positive_cooccurrence_contexts")

    write_markdown_report(out, support, inputs)
    print(f"Wrote V1 discovery visualization report to {out}")


if __name__ == "__main__":
    main()
