#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=1.0"]
# ///
"""querygen — query 集生成(DESIGN B.2;DECISIONS #21)。

構造性 GT 紅線落實:
  Q3 洩漏判定 = 分類法字串:query 含 30 域名精確字串即重生(上限 3 次),
     命中計數入報告;生成 prompt 只餵語料筆 text,不含 domain/tier 資訊。
  Q4 分層:源筆按域配比抽,stratum 欄(檔位名)入輸出;100 條/seed × 3 seeds。
  Q5 欄位相容 embed loader({id, text, ...}),task_type=query 另跑 embed。

用法:
    uv run cmd/corpusgen/querygen.py --corpus corpus/corpus_5k.jsonl \
        --config configs/domains_v1.toml --out corpus/queries_5k.jsonl \
        --report corpus/querygen_report.json --per-seed 100 --seeds 1 2 3
"""

import argparse
import json
import os
import pathlib
import random
import sys
import time
import tomllib

QUERY_SYSTEM = """You convert photo captions into the search query a real user would type to find that photo. Output JSON Lines, one per input caption:
{"idx": <input number>, "query": "..."}
Rules:
- 3-10 words, natural search phrasing (keywords or a short phrase, not a full sentence)
- describe the visible content a stranger would search for
- never copy person names, business names, or invented place names from the caption
- do not mention datasets, categories, collections, or labels"""

QUERY_USER = """Captions:
{numbered}"""

RETRY_SUFFIX = "\nFor idx {idx}: avoid the exact phrase \"{phrase}\"."


def load_env_key():
    if os.environ.get("GEMINI_API_KEY"):
        return
    env_file = pathlib.Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def taxonomy_strings(tiers: dict) -> list:
    return sorted({d.lower() for ds in tiers.values() for d in ds})


def leaked(query: str, taxonomy: list) -> str | None:
    """Q3 決定性判定:回傳命中的分類法字串,無則 None。"""
    q = query.lower()
    for t in taxonomy:
        if t in q:
            return t
    return None


def domain_to_stratum(tiers: dict) -> dict:
    return {d: tier for tier, ds in tiers.items() for d in ds}


def domain_targets(proportions: dict, total: int) -> dict:
    s = sum(proportions.values())
    raw = {d: p / s * total for d, p in proportions.items()}
    floors = {d: int(v) for d, v in raw.items()}
    rem = total - sum(floors.values())
    for d in sorted(raw, key=lambda d: raw[d] - floors[d], reverse=True)[:rem]:
        floors[d] += 1
    return floors


def stratified_sample(records: list, proportions: dict, n: int, seed: int) -> list:
    by_domain = {}
    for r in records:
        by_domain.setdefault(r["domain"], []).append(r)
    rng = random.Random(seed)
    targets = domain_targets(proportions, n)
    out = []
    for d, cnt in targets.items():
        pool = by_domain.get(d, [])
        if cnt > len(pool):
            sys.exit(f"stratum {d}: want {cnt}, pool {len(pool)}")
        out.extend(rng.sample(pool, cnt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/corpus_5k.jsonl")
    ap.add_argument("--config", default="configs/domains_v1.toml")
    ap.add_argument("--out", default="corpus/queries_5k.jsonl")
    ap.add_argument("--report", default="corpus/querygen_report.json")
    ap.add_argument("--prompts-md", default="corpus/PROMPTS_queries.md")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--per-seed", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    load_env_key()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set")
    from google import genai
    from google.genai import types
    client = genai.Client(http_options=types.HttpOptions(timeout=120_000))

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    taxonomy = taxonomy_strings(cfg["tiers"])
    strata = domain_to_stratum(cfg["tiers"])
    records = [json.loads(l) for l in open(args.corpus)]

    out_rows, leak_events, retry_fail = [], [], 0
    next_id = 1
    for seed in args.seeds:
        sources = stratified_sample(records, cfg["proportions"], args.per_seed, seed)
        for i in range(0, len(sources), args.batch_size):
            batch = sources[i:i + args.batch_size]
            numbered = "\n".join(f"{j+1}. {r['text']}" for j, r in enumerate(batch))
            queries = gen_batch(client, types, args.model, numbered, len(batch))
            # Q3:洩漏 → 針對該 idx 重生,上限 3 次
            for attempt in range(3):
                bad = {j: leaked(q, taxonomy) for j, q in queries.items() if leaked(q, taxonomy)}
                if not bad:
                    break
                for j, phrase in bad.items():
                    leak_events.append({"seed": seed, "source_id": batch[j - 1]["id"],
                                        "query": queries[j], "hit": phrase, "attempt": attempt + 1})
                suffix = "".join(RETRY_SUFFIX.format(idx=j, phrase=p) for j, p in bad.items())
                regen = gen_batch(client, types, args.model, numbered + suffix, len(batch),
                                  only_idx=set(bad))
                queries.update(regen)
            else:
                retry_fail += sum(1 for q in queries.values() if leaked(q, taxonomy))
            for j, r in enumerate(batch, start=1):
                q = queries.get(j)
                if not q or leaked(q, taxonomy):
                    continue  # 重生 3 次仍洩漏或缺漏:丟棄(計數在 retry_fail/missing)
                out_rows.append({"id": next_id, "text": q, "qseed": seed,
                                 "source_id": r["id"], "stratum": strata[r["domain"]]})
                next_id += 1
            print(f"[seed {seed}] {len(out_rows)} queries accumulated")

    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "model": args.model, "per_seed": args.per_seed, "seeds": args.seeds,
        "total_queries": len(out_rows),
        "leak_events": leak_events, "leak_count": len(leak_events),
        "dropped_after_retries": retry_fail,
        "taxonomy_strings": taxonomy,
    }
    pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    pathlib.Path(args.prompts_md).write_text(f"""# PROMPTS_queries.md — query 集生成全文(B.2;DECISIONS #21)

- 模型:`{args.model}`;{args.per_seed} 條/seed × seeds {args.seeds}
- 源筆分層:按域配比(configs/domains_v1.toml),stratum 欄入輸出
- 洩漏判定(Q3):分類法字串精確匹配(30 域名),命中重生上限 3 次
- 生成輸入只含語料筆 text,不含 domain/tier/標籤資訊

## System prompt

```
{QUERY_SYSTEM}
```

## User prompt 模板

```
{QUERY_USER}
```
""")
    print(f"{len(out_rows)} queries → {args.out}; leaks {len(leak_events)}, dropped {retry_fail}")


def gen_batch(client, types, model, user_text, n, only_idx=None):
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model, contents=QUERY_USER.format(numbered=user_text),
                config=types.GenerateContentConfig(
                    system_instruction=QUERY_SYSTEM, temperature=0.7))
            break
        except Exception as e:
            if "PerDay" in str(e) or "credits are depleted" in str(e):
                sys.exit(f"quota/credits exhausted: {str(e)[:200]}")
            time.sleep(15 * (attempt + 1))
    else:
        sys.exit("generation failed after retries")
    out = {}
    for line in (resp.text or "").splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith(("```", "json")):
            continue
        try:
            obj = json.loads(line)
            idx, q = int(obj["idx"]), str(obj["query"]).strip()
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        if 1 <= idx <= n and q and (only_idx is None or idx in only_idx):
            out[idx] = q
    return out


def selftest():
    tiers = {"p50": ["wildlife", "sports events"], "background": ["religious ceremonies"]}
    tax = taxonomy_strings(tiers)
    assert leaked("wildlife photo of a rare bird", tax) == "wildlife"
    assert leaked("Religious Ceremonies at dawn", tax) == "religious ceremonies"
    assert leaked("a bird on a branch at dawn", tax) is None
    strata = domain_to_stratum(tiers)
    assert strata["wildlife"] == "p50" and strata["religious ceremonies"] == "background"
    recs = [{"id": i, "domain": "wildlife" if i % 2 else "sports events", "text": "t"}
            for i in range(100)]
    s = stratified_sample(recs, {"wildlife": 0.5, "sports events": 0.5}, 10, 7)
    assert len(s) == 10
    assert stratified_sample(recs, {"wildlife": 0.5, "sports events": 0.5}, 10, 7) == s
    print("selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
