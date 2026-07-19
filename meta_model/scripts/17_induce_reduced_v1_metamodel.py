#!/usr/bin/env python
"""Induce a reduced V1 consent meta-model using an evidence graph.

This is the data-driven induction step. It does not select fields only from a
hand-written role list. It first constructs an element relationship graph, then
clusters source-model elements, then selects core/context/audit fields from the
cluster evidence.

Inputs are outputs from 15_analyze_roundtrip_scored_outputs.py after sentence-
level decision fields have been separated from span-level evidence.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from exc

ROLE_KEYWORDS: dict[str, str] = {
    "action": "action verb operation collect use store share disclose access analyze contact withdraw destroy return provide send release",
    "resource": "resource asset object data information record specimen sample tissue blood dna genetic genomic image audio video contact identifier",
    "actor_or_party": "actor agent party grantor grantee assigner assignee performer researcher doctor investigator institution team participant subject",
    "purpose": "purpose research study future commercial clinical care objective disease cancer genetic genomic",
    "condition_or_governance": "condition if when unless approval governance irb law precondition require allowed review committee",
    "constraint_or_prohibition": "constraint restriction exception limitation limited only except not without prohibit prohibition deny refuse",
    "temporal_scope": "time temporal duration future after before during until year month day ongoing long-term indefinite period",
    "privacy_identifiability": "privacy identifiable identifier identified de-identified deidentified coded anonymous confidential name contact",
    "choice_or_consent": "choice consent agree decline yes no optional join participate withdraw permission decision",
    "lifecycle_or_results": "retain destroy delete withdrawal effect continue return result finding incidental benefit risk harm",
    "residual_or_other": "residual unmatched other note rationale",
}
CORE_ROLE_HINTS = {"action", "resource", "actor_or_party", "purpose", "condition_or_governance", "constraint_or_prohibition", "temporal_scope", "privacy_identifiability"}


def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return " ".join(str(x).split())


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x == "":
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def safe_json_list(x: Any) -> list[Any]:
    try:
        val = json.loads(norm(x))
        return val if isinstance(val, list) else []
    except Exception:
        return []


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def element_text(row: pd.Series) -> str:
    fields = [
        "union_element_id", "source_model", "source_element_id", "source_element_label",
        "source_element_definition", "top_span_examples_json", "top_original_cue_groups_json",
    ]
    bits = [norm(row.get(c, "")) for c in fields]
    for c in ["top_span_examples_json", "top_original_cue_groups_json"]:
        bits.extend(str(v) for v in safe_json_list(row.get(c, ""))[:8])
    return " ".join(bits).lower()


def field_name_from_text(text: str) -> tuple[str, float, str]:
    toks = token_set(text)
    scores = {}
    for role, words in ROLE_KEYWORDS.items():
        wset = token_set(words)
        scores[role] = len(toks & wset) / max(1, len(wset))
    role = max(scores, key=scores.get)
    score = scores[role]
    if score <= 0:
        return "residual_or_other", 0.0, "no keyword evidence"
    matched = sorted(token_set(ROLE_KEYWORDS[role]) & toks)[:8]
    return role, float(score), ",".join(matched)


def load_inputs(analysis_dir: Path, inventory_csv: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evidence = pd.read_csv(analysis_dir / "source_element_evidence_summary.csv").fillna("")
    mentions_path = analysis_dir / "source_element_mentions_long.csv"
    pairs_path = analysis_dir / "source_element_cooccurrence_pairs.csv"
    decisions_path = analysis_dir / "sentence_level_decision_summary.csv"
    mentions = pd.read_csv(mentions_path).fillna("") if mentions_path.exists() else pd.DataFrame()
    pairs = pd.read_csv(pairs_path).fillna("") if pairs_path.exists() else pd.DataFrame()
    decisions = pd.read_csv(decisions_path).fillna("") if decisions_path.exists() else pd.DataFrame()
    if inventory_csv:
        inv = pd.read_csv(inventory_csv).fillna("")
        if "union_element_id" in inv.columns:
            drop_cols = [c for c in inv.columns if c in evidence.columns and c != "union_element_id"]
            evidence = evidence.drop(columns=drop_cols, errors="ignore").merge(inv, on="union_element_id", how="left").fillna("")
    return evidence, mentions, pairs, decisions


def build_nodes(evidence: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mention_stats: dict[str, dict[str, Any]] = {}
    if not mentions.empty and "union_element_id" in mentions.columns:
        for uid, g in mentions.groupby("union_element_id"):
            mention_stats[str(uid)] = {
                "mention_rows": len(g),
                "mention_llms": g["llm"].nunique() if "llm" in g.columns else 0,
                "mention_conditions": g["condition"].nunique() if "condition" in g.columns else 0,
                "top_spans_from_mentions_json": json.dumps([x for x, _ in Counter(g.get("span_text", pd.Series([], dtype=str)).dropna().astype(str)).most_common(8)], ensure_ascii=False),
            }
    for _, r in evidence.iterrows():
        uid = norm(r.get("union_element_id"))
        text = element_text(r)
        role, role_score, role_reason = field_name_from_text(text)
        source_model = norm(r.get("source_model")) or uid.split("::", 1)[0]
        d = r.to_dict()
        d.update({
            "union_element_id": uid,
            "source_model_inferred": source_model,
            "element_profile_text": text,
            "profile_tokens_json": json.dumps(sorted(token_set(text)), ensure_ascii=False),
            "candidate_functional_label": role,
            "functional_label_score": role_score,
            "functional_label_reason": role_reason,
            "evidence_weight": safe_float(r.get("n_source_sentences")) * (1 + 0.15 * safe_float(r.get("n_llms"))) * max(0.2, safe_float(r.get("mean_classifier_preservation_score"))) * max(0.2, safe_float(r.get("mean_cue_group_recall"))) * max(0.2, safe_float(r.get("mean_content_token_recall"))),
        })
        d.update(mention_stats.get(uid, {}))
        rows.append(d)
    return pd.DataFrame(rows)


def exact_span_pair_counts(mentions: pd.DataFrame) -> dict[tuple[str, str], int]:
    out: defaultdict[tuple[str, str], int] = defaultdict(int)
    if mentions.empty or not {"source_id", "union_element_id", "span_text"}.issubset(mentions.columns):
        return {}
    group_cols = [c for c in ["source_id", "llm", "condition", "information_model", "span_text"] if c in mentions.columns]
    m = mentions.copy()
    m["span_norm"] = m["span_text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    group_cols = [c for c in group_cols if c != "span_text"] + ["span_norm"]
    for _, g in m[m["span_norm"].astype(bool)].groupby(group_cols, dropna=False):
        ids = sorted(set(g["union_element_id"].astype(str)))
        for a, b in itertools.combinations(ids, 2):
            out[(a, b)] += 1
    return dict(out)


def build_edges(nodes: pd.DataFrame, pairs: pd.DataFrame, mentions: pd.DataFrame, min_edge_weight: float) -> pd.DataFrame:
    node_ids = nodes["union_element_id"].astype(str).tolist()
    counts = {str(r["union_element_id"]): safe_float(r.get("n_source_sentences")) for _, r in nodes.iterrows()}
    text_by_id = {str(r["union_element_id"]): norm(r.get("element_profile_text")) for _, r in nodes.iterrows()}
    token_by_id = {uid: token_set(txt) for uid, txt in text_by_id.items()}
    exact_span = exact_span_pair_counts(mentions)
    pair_lookup = {}
    if not pairs.empty:
        for _, p in pairs.iterrows():
            a, b = sorted([norm(p.get("union_element_id_a")), norm(p.get("union_element_id_b"))])
            if a and b:
                pair_lookup[(a, b)] = p
    rows = []
    for a, b in itertools.combinations(sorted(node_ids), 2):
        p = pair_lookup.get((a, b), {})
        shared = safe_float(p.get("n_source_sentences")) if isinstance(p, pd.Series) or isinstance(p, dict) else 0.0
        denom = max(1.0, counts.get(a, 0) + counts.get(b, 0) - shared)
        cooc_j = shared / denom
        cooc_score = min(1.0, cooc_j * 2.0)
        lexical_sim = jaccard(token_by_id.get(a, set()), token_by_id.get(b, set()))
        same_span_n = exact_span.get((a, b), 0)
        same_span_score = min(1.0, same_span_n / max(1.0, shared if shared else 1.0))
        cross_source_bonus = 0.05 if a.split("::", 1)[0] != b.split("::", 1)[0] else 0.0
        edge_weight = 0.45 * cooc_score + 0.35 * lexical_sim + 0.15 * same_span_score + cross_source_bonus
        reasons = []
        if cooc_j > 0: reasons.append(f"cooccurrence_jaccard={cooc_j:.3f}")
        if lexical_sim > 0: reasons.append(f"profile_token_jaccard={lexical_sim:.3f}")
        if same_span_n: reasons.append(f"same_exact_span_n={same_span_n}")
        if cross_source_bonus: reasons.append("cross_source_model_pair")
        if edge_weight >= min_edge_weight:
            rows.append({
                "union_element_id_a": a,
                "union_element_id_b": b,
                "edge_weight": edge_weight,
                "cooccurrence_jaccard": cooc_j,
                "n_shared_source_sentences": shared,
                "profile_token_jaccard": lexical_sim,
                "same_exact_span_n": same_span_n,
                "same_exact_span_score": same_span_score,
                "edge_evidence_reason": "; ".join(reasons),
            })
    return pd.DataFrame(rows).sort_values("edge_weight", ascending=False) if rows else pd.DataFrame()


def cluster_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    ids = nodes["union_element_id"].astype(str).tolist()
    try:
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(ids)
        if not edges.empty:
            for _, e in edges.iterrows():
                graph.add_edge(str(e["union_element_id_a"]), str(e["union_element_id_b"]), weight=float(e["edge_weight"]))
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
        assignments = {}
        for i, comm in enumerate(sorted(communities, key=lambda c: (-len(c), sorted(c)[0]))):
            for uid in comm:
                assignments[uid] = f"C{i+1:02d}"
        method = "networkx_greedy_modularity"
    except Exception:
        parent = {x: x for x in ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        if not edges.empty:
            for _, e in edges.iterrows():
                union(str(e["union_element_id_a"]), str(e["union_element_id_b"]))
        comps: defaultdict[str, list[str]] = defaultdict(list)
        for uid in ids:
            comps[find(uid)].append(uid)
        assignments = {}
        for i, comp in enumerate(sorted(comps.values(), key=lambda c: (-len(c), sorted(c)[0]))):
            for uid in comp:
                assignments[uid] = f"C{i+1:02d}"
        method = "threshold_connected_components"
    out = nodes.copy()
    out["cluster_id"] = out["union_element_id"].map(assignments)
    out["clustering_method"] = method
    return out


def summarize_clusters(clustered: pd.DataFrame, edges: pd.DataFrame, decisions: pd.DataFrame, min_core_sentences: int) -> pd.DataFrame:
    rows = []
    for cid, g in clustered.groupby("cluster_id"):
        label_counts = Counter(g["candidate_functional_label"].astype(str))
        functional_label, _ = label_counts.most_common(1)[0]
        source_models = sorted(set(g.get("source_model", g.get("source_model_inferred", pd.Series())).astype(str))) if "source_model" in g.columns else sorted(set(g["source_model_inferred"].astype(str)))
        n_source_sentences_max = pd.to_numeric(g.get("n_source_sentences"), errors="coerce").max()
        n_llms_max = pd.to_numeric(g.get("n_llms"), errors="coerce").max()
        ew = pd.to_numeric(g.get("evidence_weight"), errors="coerce").sum()
        mean_score = pd.to_numeric(g.get("mean_classifier_preservation_score"), errors="coerce").mean()
        mean_cue = pd.to_numeric(g.get("mean_cue_group_recall"), errors="coerce").mean()
        mean_content = pd.to_numeric(g.get("mean_content_token_recall"), errors="coerce").mean()
        if n_source_sentences_max >= min_core_sentences and len(source_models) >= 2 and n_llms_max >= 2 and functional_label in CORE_ROLE_HINTS:
            selection = "core_shared"
        elif n_source_sentences_max >= max(3, min_core_sentences // 3) and n_llms_max >= 2:
            selection = "context_module"
        else:
            selection = "audit_or_extension"
        rows.append({
            "cluster_id": cid,
            "candidate_field_name": functional_label,
            "selection_category": selection,
            "n_source_elements": len(g),
            "n_source_models": len(source_models),
            "source_models_json": json.dumps(source_models, ensure_ascii=False),
            "n_source_sentences_max": n_source_sentences_max,
            "n_llms_max": n_llms_max,
            "evidence_weight_sum": ew,
            "mean_classifier_preservation_score": mean_score,
            "mean_content_token_recall": mean_content,
            "mean_cue_group_recall": mean_cue,
            "top_source_elements_json": json.dumps(g.sort_values("evidence_weight", ascending=False)["union_element_id"].head(12).tolist(), ensure_ascii=False),
            "top_span_examples_json": json.dumps([x for v in g.get("top_span_examples_json", pd.Series()).head(8) for x in safe_json_list(v)[:2]][:10], ensure_ascii=False),
            "functional_label_votes_json": json.dumps(label_counts.most_common(), ensure_ascii=False),
        })
    out = pd.DataFrame(rows).sort_values(["selection_category", "evidence_weight_sum"], ascending=[True, False])
    return out


def role_cooccurrence(clustered: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
    if edge_df.empty:
        return pd.DataFrame()
    c_by_id = dict(zip(clustered["union_element_id"], clustered["cluster_id"]))
    rows = []
    for _, e in edge_df.iterrows():
        ca, cb = c_by_id.get(e["union_element_id_a"]), c_by_id.get(e["union_element_id_b"])
        if ca and cb and ca != cb:
            a, b = sorted([ca, cb])
            rows.append({"cluster_a": a, "cluster_b": b, "edge_weight": e["edge_weight"], "n_shared_source_sentences": e["n_shared_source_sentences"]})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return raw.groupby(["cluster_a", "cluster_b"], as_index=False).agg(
        mean_edge_weight=("edge_weight", "mean"),
        max_edge_weight=("edge_weight", "max"),
        n_intercluster_edges=("edge_weight", "size"),
        n_shared_source_sentences_sum=("n_shared_source_sentences", "sum"),
    ).sort_values(["max_edge_weight", "n_intercluster_edges"], ascending=[False, False])


def make_schema(cluster_summary: pd.DataFrame, clustered: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, Any]:
    fields = [{
        "name": "decision",
        "status": "core",
        "source": "sentence_level_decision_fields",
        "description": "Sentence/provision rule type derived from DUO.decision, ICO.decision, ODRL Rule_TestSentence, and FHIR Consent.provision.type.",
        "values": ["permit", "deny", "obligation", "mixed", "unclear"],
    }]
    keep = cluster_summary[cluster_summary["selection_category"].isin(["core_shared", "context_module"])].copy()
    used_names: Counter[str] = Counter()
    for _, c in keep.iterrows():
        base = str(c["candidate_field_name"])
        used_names[base] += 1
        name = base if used_names[base] == 1 else f"{base}_{used_names[base]}"
        support = clustered[clustered["cluster_id"].eq(c["cluster_id"])].sort_values("evidence_weight", ascending=False)
        fields.append({
            "name": name,
            "cluster_id": c["cluster_id"],
            "status": "core" if c["selection_category"] == "core_shared" else "context_module",
            "description": f"Data-induced cluster labeled as {base}; selected as {c['selection_category']} from graph evidence.",
            "value_type": "normalized_value_with_evidence",
            "allow_multiple": True,
            "selection_evidence": {
                "n_source_elements": int(c["n_source_elements"]),
                "n_source_models": int(c["n_source_models"]),
                "n_source_sentences_max": float(c["n_source_sentences_max"]),
                "n_llms_max": float(c["n_llms_max"]),
                "mean_classifier_preservation_score": float(c["mean_classifier_preservation_score"]),
                "mean_cue_group_recall": float(c["mean_cue_group_recall"]),
            },
            "source_element_support": support["union_element_id"].head(12).tolist(),
        })
    fields += [
        {"name": "residual_important_content", "status": "audit", "description": "Short meaning-critical content not captured by selected fields.", "value_type": "short_evidence_phrase"},
        {"name": "provenance", "status": "audit", "description": "Source sentence, evidence spans, selected clusters, and source elements used for audit.", "value_type": "audit_metadata"},
    ]
    return {
        "meta_model_id": "reduced_consent_metamodel_v1_graph_induced_candidate",
        "version": "0.2",
        "status": "graph_induced_candidate_requires_audit_and_validation",
        "design_goal": "Reduced functional consent representation induced from source-element evidence, co-occurrence, same-span usage, and preservation metrics.",
        "selection_method": {
            "node_input": "span-level source elements after sentence-level decision fields are removed",
            "edge_evidence": ["source-sentence co-occurrence", "same exact evidence span", "label/definition/span/cue profile similarity", "cross-source-model support"],
            "clustering": "weighted graph community detection with fallback to threshold connected components",
            "field_selection": "core_shared clusters require cross-source support, multi-LLM support, minimum sentence coverage, and functional label support; lower-coverage clusters become context modules or audit extensions",
        },
        "evaluation_variants": {
            "compact_evidence": "same schema; short evidence phrases; no full-clause copying",
            "permissive_evidence": "same schema; longer evidence allowed when needed to preserve condition, exception, temporal, or privacy meaning",
        },
        "fields": fields,
        "provision_structure": {
            "rule_type": "decision",
            "selected_cluster_fields": [f["name"] for f in fields if f.get("cluster_id")],
            "audit_fields": ["residual_important_content", "provenance"],
        },
    }


def write_methodology(out: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Reduced V1 graph-induction methodology",
        "",
        "The reduced V1 candidate is induced from corrected round-trip evidence, not selected only by manual role recommendations.",
        "",
        "## Inputs",
        "",
        "- `source_element_evidence_summary.csv`: one row per span-level source-model element with frequency, LLM support, preservation metrics, cue metrics, and span examples.",
        "- `source_element_mentions_long.csv`: element mentions by sentence/model/LLM, used to detect same-span evidence.",
        "- `source_element_cooccurrence_pairs.csv`: corrected span-level co-occurrence pairs, with sentence-level decision fields removed.",
        "- `sentence_level_decision_summary.csv`: DUO/ICO/FHIR/ODRL decision fields, summarized separately as the V1 `decision` field.",
        "",
        "## Graph construction",
        "",
        "Each source-model element is a node. Edges are weighted using source-sentence co-occurrence, same-span evidence, profile similarity from names/definitions/span examples/cue groups, and a small cross-source-model bonus.",
        "",
        "## Clustering",
        "",
        "The script uses weighted graph community detection when NetworkX is available and falls back to thresholded connected components otherwise. UMAP is not used for grouping; it can be added later only for visualization.",
        "",
        "## Field selection",
        "",
        "Clusters are selected as `core_shared`, `context_module`, or `audit_or_extension` using coverage across source sentences, support across LLMs, support across source models, preservation metrics, and functional-label evidence from cluster text. Human review is limited to audit/naming/unsafe-merge checks.",
        "",
        f"Parameters: min_edge_weight={args.min_edge_weight}, min_core_sentences={args.min_core_sentences}",
        "",
    ]
    (out / "reduced_v1_graph_induction_methodology.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", required=True)
    ap.add_argument("--inventory_csv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_edge_weight", type=float, default=0.22)
    ap.add_argument("--min_core_sentences", type=int, default=15)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    evidence, mentions, pairs, decisions = load_inputs(Path(args.analysis_dir), args.inventory_csv)
    nodes = build_nodes(evidence, mentions)
    edges = build_edges(nodes, pairs, mentions, args.min_edge_weight)
    clustered = cluster_graph(nodes, edges)
    cluster_summary = summarize_clusters(clustered, edges, decisions, args.min_core_sentences)
    cluster_cooc = role_cooccurrence(clustered, edges)
    schema = make_schema(cluster_summary, clustered, decisions)

    nodes.to_csv(out / "element_nodes.csv", index=False)
    edges.to_csv(out / "element_relationship_edges.csv", index=False)
    clustered.to_csv(out / "element_clusters.csv", index=False)
    cluster_summary.to_csv(out / "cluster_evidence_summary.csv", index=False)
    cluster_cooc.to_csv(out / "cluster_cooccurrence_summary.csv", index=False)
    decisions.to_csv(out / "sentence_level_decision_fields.csv", index=False)
    (out / "reduced_metamodel_v1_candidate.yaml").write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True))
    (out / "reduced_metamodel_v1_candidate.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2))

    md = ["# Candidate Reduced V1 Consent Meta-Model", "", "This candidate is induced from a weighted source-element evidence graph.", "", "## Selected fields"]
    for f in schema["fields"]:
        md += [f"### {f['name']} ({f.get('status', '')})", "", f.get("description", ""), ""]
        if f.get("selection_evidence"):
            md += ["Selection evidence:", "", "```json", json.dumps(f["selection_evidence"], indent=2), "```", ""]
        if f.get("source_element_support"):
            md += ["Source-element support: " + ", ".join(f["source_element_support"][:10]), ""]
    md += ["## Cluster evidence summary", "", cluster_summary.to_markdown(index=False), ""]
    (out / "reduced_metamodel_v1_candidate.md").write_text("\n".join(md))
    write_methodology(out, args)
    print(f"Wrote graph-induced V1 candidate to {out}")


if __name__ == "__main__":
    main()
