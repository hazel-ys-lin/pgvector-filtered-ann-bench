# pgvector filtered-ANN bench

A reproducible measurement study of filtered approximate nearest-neighbour (ANN) search in PostgreSQL + pgvector.

## 1. TL;DR

**For filtered vector search, the query plan often matters more than the index — and today the plan is picked automatically, with only a manual `EXPLAIN` to catch a bad choice.**

On a 30k-record corpus with a btree index on the filter column:

- **The planner draws a sharp line.** For filters that keep 0.7%–15% of rows it always chooses an exact plan (bitmap scan on the filter index), and recall@50 is 1.000. For a filter that keeps 50% of rows it switches to the HNSW index at every `ef_search` and iterative-scan setting, and recall drops to **0.165–0.616** with default settings.
- **At that 50% tier, the fix matters more than the index:**

  | treatment | recall (worst group) | p50 latency |
  |---|---|---|
  | planner default (ef_search=40) | 0.165 | 0.7 ms |
  | force exact plan | 1.000 | 22.2 ms |
  | `hnsw.iterative_scan = relaxed_order` | 0.958 | 1.2 ms |
  | ef_search = 400 | 0.866 | 2.5 ms |

  Raising `ef_search` does not rescue queries whose neighbourhood is poor in filter-passing rows. Iterative scan does, at close to the default latency.
- **Filter–cluster correlation is a causal variable, and it can be predicted.** The vectors are identical and the filter keeps the same fraction of rows. Only *which* rows carry the label changes, and recall at the default setting ranges from 0.165 to 0.616. A statistic computed from the corpus alone (σ_l, below) predicted all three measured recalls to within 0.011.
- **The synthetic corpus is validated against real captions.** Its LLM-generated photo captions match the embedding geometry of real COCO captions: pairwise-distance median 0.2895 vs 0.2931. Getting there needed constructive controls, because asking the model to "vary" made duplication worse (§4).

## 2. Motivation

Production filtered-ANN workloads combine a vector `ORDER BY embedding <=> $q LIMIT k` with an ordinary `WHERE` clause. PostgreSQL's cost-based planner decides per query whether to:

- walk the HNSW index and filter afterwards (fast; can return fewer than `k` matching rows), or
- use an exact path: scan the rows that pass the filter and sort them (always correct; cost grows with the number of passing rows).

When the planner picks the index for a filter whose passing rows are scarce near the query, recall silently drops. The usual remedy is a person reading `EXPLAIN`. This study measures where that failure happens, what drives it, and how the available knobs compare.

## 3. Related work

> **Citations to be completed by the author.** The positioning below follows the project design notes (`docs/DESIGN_v1.1.md` §0).

- **Amanbayev et al. 2026** — FANNS benchmark. Treats query–filter correlation as an observed monitoring signal (GLS). Reports that at low selectivity the optimizer often picks an approximate scan and misses an exact path that would score full recall, with forcing the exact path as the remedy.
- **Shi et al. 2025** — standardized index-tuning protocol. This study instead fixes the index at defaults (§5).
- **Li et al. 2025** — selectivity sweep design.
- **Iff et al. 2025** — FANNS benchmark.
- **Xia et al. 2026** — cost estimation.

This work adds three things:

1. **A controlled causal test of dispersion.** Two label sets on identical vectors: one correlated with clusters, one random. Both are quantified with GLS.
2. **A planner right/wrong boundary map** built from full `EXPLAIN` capture on every configuration (§6.2).
3. **A method for validating a synthetic corpus** against a real-text reference, with the failure modes and fixes documented (§4).

## 4. Synthetic corpus and how it was validated

### 4.1 Generation

- **Text.** 30 everyday photo domains (sports events, weddings, wildlife, street markets, …). For each domain, `gemini-3.5-flash` writes 2–4 sentence captions of a single photograph. The full prompts are in `corpus/PROMPTS.md`.
- **Mix.** The domain shares are set so that domain groups form filter tiers keeping 0.7%, 3%, 6%, 15% and 50% of rows. Six "background" domains belong to no tier (`configs/domains_v1.toml`).
- **Embedding.** Every text is embedded with `gemini-embedding-001` at 768 dimensions (`RETRIEVAL_DOCUMENT` for corpus text, `RETRIEVAL_QUERY` for queries). Every vector is L2-normalized; the 768-d truncation leaves raw norms around 0.59.

### 4.2 The reference distribution

A synthetic corpus is only useful if its geometry looks like real data. Gaussian-noise vectors, for example, are nearly orthogonal (pairwise cosine distance ≈ 0.85), which makes HNSW behave unrealistically.

- **Reference sample.** 1,000 real captions from the COCO 2017 val split. One caption per distinct image, fixed seed. Source zip sha256 `113a836d…0268`.
- **Same conditions.** The sample is embedded with the same model, task type and normalization as the corpus.
- **Gate.** Seven checks must pass before any benchmark runs:

| # | check | 30k corpus | reference / criterion | pass |
|---|---|---|---|---|
| 1 | pairwise distance median | 0.2895 | COCO 0.2931, window ±0.05 | yes |
| 2 | distance spread (std/mean) | 0.1201 | ≥ 70% of COCO 0.1289 | yes |
| 3 | near-duplicate rate (NN distance < 0.05) | 2.68% | ≤ 3% | yes |
| 4 | intra/cross-domain distance ratio | 0.831 | 0.6–0.9 | yes |
| 5 | per-domain near-duplicate rate | worst 4.70% (3 documented exceptions, §8) | < 5% (exceptions < 10%) | yes |
| 6 | end-to-end recall, no filter, ef=200, k=50 | 0.9988 | ≥ 0.99, HNSW plan asserted | yes |
| 7 | GLS construction | see §5.3 | clustered tiers enriched; random labels r ∈ [0.85, 1.15] | yes |

Notes on the thresholds:

- **Check 3** originally also had a 1% floor. It was removed: the failure the check guards against (ties breaking recall) exists only on the high side. Real COCO captions themselves sit at 4.5% on this metric.
- **Checks 3–5** started as initial values and were calibrated after the first 5k run. Every change is recorded in `docs/DECISIONS.md`.

### 4.3 What it took to pass: monitor → plead → construct

Duplication was the hard part. Each fix below is recorded in `docs/DECISIONS.md` together with the attempts that failed.

| step | intervention | wildlife near-dup |
|---|---|---|
| 5k pilot | plain prompt; **monitoring** (embedding-space check 5) catches the problem | 8.98% |
| plead | prompt hint: "vary species widely, avoid iconic scenes" | **34.73%** (worse) |
| construct | fixed list of 332 species, one assigned to each record; the model has no choice of subject | 0.0% at 5k |
| scale to 30k | each species now comes up ~3 times; every batch is a separate API call with no memory of earlier ones | 20.0% |
| more thinking | `thinking_budget` raised 512 → 1024 → 2048 | 20% → 28.5% → **36.0%** (worse) |
| construct | when a species repeats, its earlier caption is put into the prompt as the thing to differ from | **3.8%** |

Two more findings point the same way:

- **Thinking budget.** Setting `thinking_budget=0` to save cost made near-duplicates 6.7× more common across the whole corpus (6.53% vs 0.98% in a same-size pool). A cost pilot of one batch per domain could not reveal this, because duplicates only appear *across* batches. A moderate budget (512) restored diversity.
- **Examples without a forced subject.** Showing the model recent captions from a domain, with no per-record subject, made one domain much worse (work scenes 5.9% → 26.2%). The concrete examples anchored the model toward similar text. This attempt was reverted.

The common lesson: telling the model to "vary" does not hold, but constraining the generation does. The planner result has the same shape — automatic plan choice does not hold, explicit control does.

## 5. Experiment setup

### 5.1 Environment

| | |
|---|---|
| hardware | Apple M4, 24 GB; Docker VM with 10 CPUs and ~2 GB RAM |
| database | `pgvector/pgvector:0.8.1-pg16`, pinned by digest (`docker-compose.yml`); `shm_size: 2gb` |
| index | HNSW `vector_cosine_ops`, `m=16`, `ef_construction=64` (pgvector defaults); build with `maintenance_work_mem=2GB` took 9.6 s for 30k rows |
| filter indexes | btree on `label_clustered` and on each `label_scattered_s{1,2,3}` column |
| statistics | `ANALYZE` after loading and index build |
| session | every configuration starts with `RESET ALL` and `plan_cache_mode = force_custom_plan` (prepared statements would otherwise switch to generic plans mid-run and could ignore GUC changes) |
| plan forcing | `force_exact` sets `enable_indexscan = off`; each `EXPLAIN` is checked to show an exact plan |
| harness | Go + pgx; no psql text parsing |

**Index parameters.** The index is deliberately left at pgvector defaults. The question here is how plans are chosen, not how to tune the index. Defaults match most production deployments, and tuning is left to Shi et al.'s protocol.

### 5.2 Queries and ground truth

- **Query set.** 324 search queries written by the LLM from sampled corpus captions: 3 seeds × 100, sampled in proportion to domain shares, plus 24 extra queries so that each filter tier's matching side has at least 18.
- **No label leakage.** The generator sees only the caption text, never its domain or tier. No query contains a domain name; this is checked automatically and none failed.
- **Judge.** Recall@50 is measured against an exact top-k computed in Go on float32 vectors, independent of the database. Before each run, Go and pgvector distances are compared and must agree within 1e-5; the worst observed gap was 1.6e-7. A returned row within ε = 1e-5 of the k-th distance counts as a tie-equivalent hit. When fewer than `k` rows pass the filter, recall is divided by the number that do.
- **Auxiliary check.** Each query also records whether its source caption was retrieved. Among exact-plan rows where the source passes the filter, it was retrieved 100% of the time. This is a sanity check on the queries, not a recall metric.

### 5.3 Filters: selectivity × dispersion

Each tier is queried with two label sets on the **same vectors**:

- **clustered** — the label is the domain's tier, so passing rows form clusters in vector space.
- **scattered** — labels are assigned at random with exactly the same per-tier counts (three seeds).

Queries are split into two sides for the clustered labels:

- **enriched** — the query came from a domain inside the filter's tier;
- **cross** — it came from a domain outside it.

GLS measures how filter-dense a query's neighbourhood is:

- **σ_g** — the share of all rows that pass the filter.
- **σ_l** — the share of the query's exact unfiltered top-50 that passes the filter.
- **r = σ_l / σ_g** — pooled over queries.

| tier | σ_g | clustered enriched r | clustered enriched σ_l | clustered cross σ_l | scattered r (3 seeds) |
|---|---|---|---|---|---|
| p07 | 0.007 | 26.35 | 0.184 | 0.002 | 0.86 / 0.88 / 1.05 |
| p3 | 0.03 | 22.11 | 0.663 | 0.005 | 0.99 / 0.96 / 0.94 |
| p6 | 0.06 | 13.48 | 0.809 | 0.023 | 1.01 / 1.00 / 0.97 |
| p15 | 0.15 | 5.32 | 0.799 | 0.056 | 1.00 / 1.04 / 1.00 |
| p50 | 0.5 | 1.51 | 0.756 | 0.216 | 0.99 / 1.00 / 0.99 |

The construction works as intended:

- **Clustered labels** make one side's neighbourhood far richer than the global rate (enriched) and the other far poorer (cross).
- **Random labels** stay at r ≈ 1.

### 5.4 Configurations

- **Main matrix.** 5 tiers × {clustered, scattered ×3 seeds} × `ef_search` {40, 100, 200, 400} × `iterative_scan` {off, relaxed_order}, with `PlanMode=auto`: 160 configurations, all 324 queries each, `k=50`.
- **Per configuration.** 5 unrecorded warm-up queries, then the measured run.
- **EXPLAIN.** `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` is saved for 3 representative queries and parsed into a plan type. The three plan types must agree; they agreed in all 160 configurations.
- **Latency** is client round-trip on the same machine.

### 5.5 Reproduce

The commands come in two tiers:

```sh
# Tier 1 — reproduce every number here. No API key needed.
# Put the released JSONL files into corpus/ first.
make reproduce-30k        # geometry gate, main matrix, four-way at p50, then print all tables

# Tier 2 — regenerate the corpus. Needs GEMINI_API_KEY in .env.
make corpus-30k embed-30k queries-30k
```

Tier 2 produces a corpus made by the same method, not the identical text, because LLM output varies between runs; see §8 for how the released corpus was produced. Every table in §5–§6 is printed by `scripts/summarize.py` (`make summary-30k`).

**Data.** Corpus text, vectors, queries and the COCO reference sample are distributed as release assets, not in git. Their manifests (source sha256, model, task type, dimensions, commit) are in `corpus/*.manifest.json`.

## 6. Results

### 6.1 Four-way comparison at the danger tier (p50)

| treatment | plan | clustered/enriched | clustered/cross | scattered | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| (a) planner default, ef=40 | hnsw_index | 0.616 | 0.165 | 0.399 | 0.7 | 1.3 |
| (b) force exact | bitmap_meta | 1.000 | 1.000 | 1.000 | 22.2 | 26.6 |
| (c) iterative scan, ef=40 | hnsw_index | 0.985 | 0.958 | 0.978 | 1.2 | 2.4 |
| (d) ef_search=200 | hnsw_index | 0.993 | 0.674 | 0.990 | 1.5 | 2.2 |
| (d') ef_search=400 | hnsw_index | 0.998 | 0.866 | 0.997 | 2.5 | 3.5 |

The design expected the danger tier to be p3/p6. At 30k those tiers never use the index, so the comparison was run at p50, where the planner does.

### 6.2 Where the planner switches

Planner choice in the main matrix (`PlanMode=auto`; clustered and scattered labels agree):

| tier | every ef_search, iterative off | every ef_search, iterative on |
|---|---|---|
| p07, p3, p6, p15 | bitmap_meta (exact) | bitmap_meta (exact) |
| p50 | hnsw_index | hnsw_index |

Recall@50 at ef_search=40, iterative off:

| tier | clustered/enriched | clustered/cross | scattered |
|---|---|---|---|
| p07 – p15 | 1.000 | 1.000 | 1.000 |
| p50 | 0.616 | 0.165 | 0.399 |

At this scale the planner is never wrong on restrictive filters: it picks the exact path and gets full recall. It is wrong at the permissive tier, where the index looks cheaper but cannot supply enough passing candidates.

The boundary also moves with table size. All points below come from this study:

| rows | data | observation |
|---|---|---|
| 2k | Gaussian random vectors (smoke test) | sequential scan even without a filter |
| 5k | this corpus (pilot) | all 200 configurations use exact plans; recall 1.000 everywhere |
| 10k | Gaussian random vectors (smoke test) | no filter: HNSW at ef=40, sequential scan at ef=400 |
| 30k | this corpus | exact up to the 15% tier, HNSW at the 50% tier at every ef |

The 2k and 10k points use random vectors and are not directly comparable to the corpus runs. The ef-driven switch to sequential scan seen at 10k did not occur at 30k.

### 6.3 Dispersion is causal, and σ_l predicts the loss

At ef_search=40 with iterative scan off, pgvector returns at most `ef` candidates before filtering. About `ef · σ_l` of them should pass the filter, so a first-order prediction is recall ≈ min(1, ef · σ_l / k):

| group | σ_l (from the corpus) | predicted recall | measured recall |
|---|---|---|---|
| clustered/enriched | 0.756 | 0.605 | 0.616 |
| clustered/cross | 0.216 | 0.173 | 0.165 |
| scattered | 0.498 | 0.398 | 0.399 |

The same prediction across `ef_search` (predicted / measured):

| ef_search | clustered/enriched | clustered/cross | scattered |
|---|---|---|---|
| 40 | 0.605 / 0.616 | 0.173 / 0.165 | 0.398 / 0.399 |
| 100 | 1.000 / 0.945 | 0.431 / 0.413 | 0.995 / 0.942 |
| 200 | 1.000 / 0.993 | 0.863 / 0.674 | 1.000 / 0.990 |
| 400 | 1.000 / 0.998 | 1.000 / 0.866 | 1.000 / 0.997 |

- **Where the model holds.** It is accurate while candidates are scarce (ef · σ_l < k).
- **Where it overpredicts.** Closer to saturation it predicts too high, most of all on the cross side. HNSW's candidate list is an approximate neighbourhood, and on the cross side it holds fewer passing rows than the exact top-k does.
- **The causal result.** Same vectors, same filter share, different label placement: recall 0.165 / 0.399 / 0.616. Placement of the labels alone moves recall by 3.7×.

### 6.4 Iterative scan vs deeper ef_search (p50)

| iterative | ef_search | clustered/enriched | clustered/cross | scattered | p50 ms |
|---|---|---|---|---|---|
| off | 40 | 0.616 | 0.165 | 0.399 | 0.6 |
| off | 100 | 0.945 | 0.413 | 0.942 | 1.0 |
| off | 200 | 0.993 | 0.674 | 0.990 | 1.6 |
| off | 400 | 0.998 | 0.866 | 0.997 | 2.4 |
| relaxed_order | 40 | 0.985 | 0.958 | 0.978 | 1.2 |
| relaxed_order | 100 | 0.989 | 0.964 | 0.979 | 1.3 |
| relaxed_order | 200 | 0.995 | 0.970 | 0.990 | 1.5 |
| relaxed_order | 400 | 0.998 | 0.980 | 0.997 | 2.4 |

Deeper `ef_search` mostly helps queries whose neighbourhood is already rich in passing rows. Iterative scan keeps pulling candidates until `k` pass, so it helps the cross side as well.

## 7. Practical guidance

Scope: one system, one synthetic corpus, 30k rows.

| situation | what we measured | suggestion |
|---|---|---|
| restrictive filter (≤ 15% of rows) with a btree on the filter column | planner picks the exact path; recall 1.000 | nothing to do; check `EXPLAIN` once |
| permissive filter where `EXPLAIN` shows the HNSW index | recall 0.165–0.616 at defaults | enable `hnsw.iterative_scan = relaxed_order` (≥ 0.958 recall at ~1.2 ms) |
| filter labels correlated with content (e.g. category, tenant) | cross-category queries are worst (0.165) | judge recall by cross-category queries, not the average |
| need guaranteed recall | forcing the exact path gives 1.000 at ~30× latency | use it for correctness-critical paths, not as a default |
| tempted to only raise `ef_search` | cross side still 0.866 at ef=400 | not enough on its own when filter and content are correlated |

Forcing the exact path matches the remedy reported by Amanbayev et al. (to verify against their text). Iterative scan, the `ef_search` comparison and the dispersion-based reading of recall are additions from this study.

## 8. Limitations and future work

- **Synthetic corpus.** Validated against COCO captions on seven geometric checks, but it is still not a production dataset.
- **Near-duplicate exceptions (check 5).** Three domains are allowed up to 10% instead of 5% (`docs/DECISIONS.md` #32):
  - **religious ceremonies (7.91%)** — a background domain, never used as a filter value;
  - **work scenes (5.9%)** and **architecture & landmarks (5.8%)** — both in the p50 tier. Their duplicates may make p50 recall slightly optimistic.

  The architecture & landmarks duplicates follow the wildlife pattern (the same landmark written twice as its postcard scene) and could be fixed with the species-list mechanism. The author chose to accept them instead.
- **Mixed generation settings.** In the released corpus, ids ≤ 5,780 were generated with the model's default thinking budget and batches of 30; the rest with `thinking_budget=512` and batches of 50. Wildlife uses the fixed species list plus the repeat-history prompt (§4.3). An earlier `thinking_budget=0` batch failed the geometry gate and was discarded.
- **Fewer ties than real data.** The corpus's near-duplicate rate (2.68%) is below real COCO captions (4.5%), so effects that depend on ties may be understated. 2.38% of measured rows have an ε-tie at the k-th ground-truth boundary.
- **Queries are LLM-written** from corpus captions. The source-caption check is auxiliary; recall is always judged by exact search.
- **Scale and scope.** 30k rows, one machine, one database (pgvector 0.8.1 on PostgreSQL 16), HNSW only (IVFFlat not tested), always with a btree on the filter column. The case without a filter index is untested.
- **Latency** is client round-trip on a laptop; compare treatments relative to each other, not as absolute numbers.
- **Future work.** Larger scales (the boundary moves with size); comparing with other implementations; using σ_l to choose the plan automatically, since it already predicts the scarce-candidate regime.

## Attribution

The COCO reference sample uses COCO 2017 caption annotations, © COCO Consortium, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See `corpus/coco_ref_1k.ATTRIBUTION.md`.
