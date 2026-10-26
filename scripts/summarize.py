#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Regenerate every results table in README.md from raw outputs.

    uv run scripts/summarize.py \
        --geometry corpus/geometry_report_30k.json \
        --matrix results/run_30k/results.jsonl \
        --fourway results/run_30k_fourway_p50/results.jsonl
"""

import argparse
import collections
import json

TIERS = ["p07", "p3", "p6", "p15", "p50"]
TREATMENTS = [
    (("auto", 40, "off"), "(a) planner default"),
    (("force_exact", 40, "off"), "(b) force exact"),
    (("auto", 40, "relaxed_order"), "(c) iterative scan"),
    (("auto", 200, "off"), "(d) ef_search=200"),
    (("auto", 400, "off"), "(d') ef_search=400"),
]


def load(path):
    return [json.loads(l) for l in open(path)]


def mean(xs):
    return sum(xs) / len(xs)


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def group(r):
    if r["dispersion"] == "clustered":
        return "clustered/" + r["side"]
    return "scattered"


def geometry(rep):
    print("### Geometry gate (B.3)\n")
    print("| # | check | value | criterion | pass |")
    print("|---|---|---|---|---|")
    for c in rep["checks"]:
        v = c["value"]
        if c["check"] == 5:
            v = f"worst non-exception {max(r for d, r in v['per_domain'].items() if d not in v['exceptions']):.4f}; " \
                + ", ".join(f"{d} {v['per_domain'][d]:.4f}" for d in v["exceptions"])
        elif c["check"] == 7:
            v = "see GLS table"
        print(f"| {c['check']} | {c['name']} | {v} | {c['criterion']} | {'yes' if c['pass'] else 'no'} |")
    print()
    gls = next(c for c in rep["checks"] if c["check"] == 7)["value"]
    print("### GLS by tier (pooled over queries)\n")
    print("| tier | σ_g | clustered enriched r | clustered enriched σ_l | clustered cross σ_l | scattered r (3 seeds) |")
    print("|---|---|---|---|---|---|")
    for t in gls:
        e, x = t["clustered_enriched"], t["clustered_cross"]
        sc = " / ".join(f"{s['pooled_r']:.2f}" for s in t["scattered_by_seed"])
        print(f"| {t['tier']} | {t['sigma_g']} | {e['pooled_r']:.2f} (n={e['n_queries']}) | {e['sigma_l']:.3f} | {x['sigma_l']:.3f} | {sc} |")
    print()


def plan_map(rows):
    print("### Planner choice (matrix, PlanMode=auto)\n")
    cell = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["config_set"] == "matrix":
            cell[(r["tier"], r["ef_search"], r["iterative_scan"])][r["plan_type"]] += 1
    head = [(ef, it) for it in ["off", "relaxed_order"] for ef in [40, 100, 200, 400]]
    print("| tier | " + " | ".join(f"ef{ef} {'iter' if it != 'off' else 'off'}" for ef, it in head) + " |")
    print("|---|" + "---|" * len(head))
    for t in TIERS:
        print(f"| {t} | " + " | ".join("/".join(sorted(cell[(t, ef, it)])) for ef, it in head) + " |")
    disagree = {(r["config_set"], r["tier"], r["dispersion"], r["scatter_seed"], r["ef_search"],
                 r["iterative_scan"]) for r in rows if not r["plan_agree"]}
    print(f"\nConfigs with disagreeing representative EXPLAINs: {len(disagree)}\n")


def recall_tiers(rows):
    print("### Recall@50 by tier (matrix, ef_search=40, iterative off)\n")
    agg = collections.defaultdict(list)
    for r in rows:
        if r["config_set"] == "matrix" and r["ef_search"] == 40 and r["iterative_scan"] == "off":
            agg[(r["tier"], group(r))].append(r["recall_at_k"])
    gs = ["clustered/enriched", "clustered/cross", "scattered"]
    print("| tier | " + " | ".join(gs) + " |")
    print("|---|---|---|---|")
    for t in TIERS:
        print(f"| {t} | " + " | ".join(f"{mean(agg[(t, g)]):.3f}" for g in gs) + " |")
    print()


def p50_grid(rows):
    print("### p50: recall@50 vs ef_search and iterative scan\n")
    agg = collections.defaultdict(list)
    lat = collections.defaultdict(list)
    for r in rows:
        if r["config_set"] == "matrix" and r["tier"] == "p50":
            k = (r["ef_search"], r["iterative_scan"])
            agg[k + (group(r),)].append(r["recall_at_k"])
            lat[k].append(r["latency_ms"])
    gs = ["clustered/enriched", "clustered/cross", "scattered"]
    print("| iterative | ef_search | " + " | ".join(gs) + " | p50 latency ms |")
    print("|---|---|---|---|---|---|")
    for it in ["off", "relaxed_order"]:
        for ef in [40, 100, 200, 400]:
            print(f"| {it} | {ef} | " + " | ".join(f"{mean(agg[(ef, it, g)]):.3f}" for g in gs)
                  + f" | {pct(lat[(ef, it)], 0.5):.1f} |")
    print()


def sigma_prediction(rows):
    print("### σ_l prediction vs measurement (p50, ef_search=40, iterative off, k=50)\n")
    print("| group | σ_l (construction) | predicted recall min(1, ef·σ_l/k) | measured recall |")
    print("|---|---|---|---|")
    agg = collections.defaultdict(list)
    sig = collections.defaultdict(list)
    for r in rows:
        if (r["config_set"] == "matrix" and r["tier"] == "p50" and r["ef_search"] == 40
                and r["iterative_scan"] == "off"):
            agg[group(r)].append(r["recall_at_k"])
            sig[group(r)].append(r["sigma_l_construct"])
    for g in ["clustered/enriched", "clustered/cross", "scattered"]:
        s = mean(sig[g])
        print(f"| {g} | {s:.3f} | {min(1, 40 * s / 50):.3f} | {mean(agg[g]):.3f} |")
    print()
    print("Same model across ef_search (iterative off): predicted / measured\n")
    agg = collections.defaultdict(list)
    for r in rows:
        if (r["config_set"] == "matrix" and r["tier"] == "p50"
                and r["iterative_scan"] == "off"):
            agg[(r["ef_search"], group(r))].append(r["recall_at_k"])
    gs = ["clustered/enriched", "clustered/cross", "scattered"]
    print("| ef_search | " + " | ".join(gs) + " |")
    print("|---|---|---|---|")
    for ef in [40, 100, 200, 400]:
        print(f"| {ef} | " + " | ".join(
            f"{min(1, ef * mean(sig[g]) / 50):.3f} / {mean(agg[(ef, g)]):.3f}" for g in gs) + " |")
    print()


def fourway(rows):
    print("### Four-way comparison at the danger tier (p50)\n")
    agg = collections.defaultdict(list)
    lat = collections.defaultdict(list)
    plan = {}
    for r in rows:
        k = (r["plan_mode"], r["ef_search"], r["iterative_scan"])
        agg[(k, group(r))].append(r["recall_at_k"])
        lat[k].append(r["latency_ms"])
        plan[k] = r["plan_type"]
    gs = ["clustered/enriched", "clustered/cross", "scattered"]
    print("| treatment | plan | " + " | ".join(gs) + " | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|")
    for k, name in TREATMENTS:
        print(f"| {name} | {plan[k]} | " + " | ".join(f"{mean(agg[(k, g)]):.3f}" for g in gs)
              + f" | {pct(lat[k], 0.5):.1f} | {pct(lat[k], 0.95):.1f} |")
    print()


def auxiliary(rows):
    print("### Auxiliary\n")
    tie = mean([1 if r["gt_boundary_tie"] else 0 for r in rows])
    print(f"- Rows whose ground truth has an ε-tie at the k-th boundary: {tie:.2%}")
    # Only rows where the query's source record is guaranteed to pass the filter
    # (clustered, enriched side) can retrieve it; elsewhere a miss is by construction.
    hit = mean([1 if r["expected_id_hit"] else 0 for r in rows
                if r["config_set"] == "matrix" and r["plan_type"] != "hnsw_index"
                and r["dispersion"] == "clustered" and r["side"] == "enriched"])
    print(f"- Constructed-GT `expected_id_hit`, exact plans, source eligible under the filter "
          f"(clustered/enriched): {hit:.2%} — auxiliary sanity check, not a recall metric")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--fourway", required=True)
    a = ap.parse_args()
    geometry(json.load(open(a.geometry)))
    m = load(a.matrix)
    plan_map(m)
    recall_tiers(m)
    p50_grid(m)
    sigma_prediction(m)
    fourway(load(a.fourway))
    auxiliary(m)


if __name__ == "__main__":
    main()
