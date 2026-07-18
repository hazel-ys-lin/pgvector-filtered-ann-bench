#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["google-genai>=1.0"]
# ///
"""corpusgen — LLM 語料生成(DESIGN B.1/B.2;DECISIONS #13 Gemini / #14 Python / #15 配比)。

用法(uv 管理,PEP 723 inline deps):
    GEMINI_API_KEY=... uv run cmd/corpusgen/corpusgen.py \
        --total 5000 --config configs/domains_v1.toml \
        --out corpus/corpus_5k.jsonl --report corpus/gen_report_5k.json

    uv run cmd/corpusgen/corpusgen.py --selftest   # 無 API 的邏輯自檢

輸出:
  - JSONL 語料(欄位:id, text, domain, scene, objects, time_of_day)
  - gen_report.json:per-domain 統計 + 三項監控(B.2 去重段 + 5k 首批監控拍板)
  - corpus/PROMPTS.md:prompt 與參數全文(B.4,model id 於此釘定)

監控三項(2026-07-18 拍板照列):
  1. per-domain 文字層近重複率(B.3 檢查 5 的文字層預警;正式檢查在 embedding 空間)
  2. 跨批人名/地名塌縮監控(英文生成傾向,B.2 補句;>2% 即報)
  3. 每批 Jaccard 丟棄率曲線

可續跑:--out 已存在時讀入既有記錄,續補各域缺額。
"""

import argparse
import collections
import json
import os
import pathlib
import random
import re
import sys
import time
import tomllib

# ---- B.2 定稿 prompt(勿改;變更須過 DECISIONS)----

SYSTEM_PROMPT = """You are a dataset generator. Output JSON Lines, one record per line:
{"text": "...", "domain": "<batch domain>", "scene": "...", "objects": [...], "time_of_day": "..."}
Rules:
- "text" is a 2-4 sentence concrete scene description, written like a caption for a single photograph
- Every record must differ in participants, location, objects, actions, and time
- Never reuse a person name, place name, or business name that appeared earlier
- Vary sentence structure: do not open consecutive records with the same pattern
- No abstract commentary; describe only what is visible in the frame"""

USER_TEMPLATE = """Domain: {domain}
Generate {n} records. Diversity axes (vary at least three per record):
number and age of participants / indoor vs outdoor / weather / time of day /
mood / camera distance (close-up to wide shot) / primary objects"""

REQUIRED_KEYS = {"text", "domain", "scene", "objects", "time_of_day"}
JACCARD_THRESHOLD = 0.6   # B.2:同批 > 0.6 丟棄
CROSS_BATCH_WARN = 0.02   # B.2:跨批抽樣重複率 > 2% 即報
CROSS_SAMPLE_SIZE = 200   # 跨批比對的抽樣視窗
NAME_REPEAT_MIN = 3       # 人名/地名出現 ≥3 筆視為塌縮候選

# ---- 純函式(selftest 覆蓋)----

def trigrams(text: str) -> set:
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


TITLE_WORDS = {"Dr", "Mr", "Mrs", "Ms", "St", "Prof", "Fr", "Rev"}


def extract_names(text: str) -> set:
    """粗抽專有名:非句首的大寫詞串。啟發式監控用,非 NER。
    濾誤報:稱謂單獨出現不算名;純大寫縮寫(LED、AI)不算名。"""
    names = set()
    # 稱謂縮寫的句點會騙過句子切分("Dr. Chen" 斷成兩句),先去掉
    for t in TITLE_WORDS:
        text = text.replace(f"{t}. ", f"{t} ")
    for sent in re.split(r"(?<=[.!?])\s+", text):
        words = sent.split()
        i = 1  # 跳過句首詞
        while i < len(words):
            w = words[i].strip(",.;:!?\"'()")
            if w and w[0].isupper() and w.isalpha():
                run = [w]
                j = i + 1
                while j < len(words):
                    nw = words[j].strip(",.;:!?\"'()")
                    if nw and nw[0].isupper() and nw.isalpha():
                        run.append(nw)
                        j += 1
                    else:
                        break
                # 丟棄:單獨稱謂、純大寫縮寫、稱謂開頭但只剩稱謂的串
                run = [w for w in run if not (w.isupper() and len(w) <= 4)]
                if run and not all(w in TITLE_WORDS for w in run):
                    names.add(" ".join(run))
                i = j
            else:
                i += 1
    return names


def domain_targets(proportions: dict, total: int) -> dict:
    """正規化配比 → 每域目標筆數(最大餘數法,總和精確 = total)。"""
    s = sum(proportions.values())
    raw = {d: p / s * total for d, p in proportions.items()}
    floors = {d: int(v) for d, v in raw.items()}
    remainder = total - sum(floors.values())
    by_frac = sorted(raw, key=lambda d: raw[d] - floors[d], reverse=True)
    for d in by_frac[:remainder]:
        floors[d] += 1
    return floors


def parse_batch(raw_text: str, batch_domain: str) -> tuple[list, int]:
    """解析模型輸出的 JSONL;回傳 (合法記錄, 格式丟棄數)。domain 欄位強制覆寫為本批域。"""
    records, rejected = [], 0
    for line in raw_text.splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith(("```", "json")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            rejected += 1
            continue
        if not isinstance(obj, dict) or not REQUIRED_KEYS.issubset(obj):
            rejected += 1
            continue
        if not isinstance(obj["text"], str) or len(obj["text"].split()) < 10:
            rejected += 1
            continue
        obj["domain"] = batch_domain
        records.append(obj)
    return records, rejected


# ---- 監控 ----

class Monitors:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.accepted_grams = []          # (domain, trigram set) 全部保留(5k 規模可承受)
        self.name_counts = collections.Counter()
        self.records_with_names = 0
        self.records_total = 0
        self.cross_dup = 0
        self.cross_checked = 0
        self.batch_discard_curve = []     # (batch_no, domain, in_batch_discard, format_rejects)

    def check_cross_batch(self, grams: set) -> bool:
        """對既有記錄抽樣比對;回傳是否為跨批重複(僅計數,不丟棄——B.2 是監控訊號)。"""
        if not self.accepted_grams:
            return False
        sample = self.rng.sample(self.accepted_grams,
                                 min(CROSS_SAMPLE_SIZE, len(self.accepted_grams)))
        self.cross_checked += 1
        if any(jaccard(grams, g) > JACCARD_THRESHOLD for _, g in sample):
            self.cross_dup += 1
            return True
        return False

    def note_accept(self, domain: str, grams: set, text: str):
        self.accepted_grams.append((domain, grams))
        self.records_total += 1
        names = extract_names(text)
        if names:
            self.records_with_names += 1
        self.name_counts.update(names)

    def per_domain_dup_rate(self) -> dict:
        """域內文字層近重複率:與同域其他記錄任一 Jaccard > 閾值的比例。"""
        by_domain = collections.defaultdict(list)
        for d, g in self.accepted_grams:
            by_domain[d].append(g)
        out = {}
        for d, gs in by_domain.items():
            dup = sum(
                1 for i, g in enumerate(gs)
                if any(jaccard(g, h) > JACCARD_THRESHOLD for j, h in enumerate(gs) if j != i)
            )
            out[d] = round(dup / len(gs), 4) if gs else 0.0
        return out

    def report(self, watchlist: list) -> dict:
        collapsed = {n: c for n, c in self.name_counts.items() if c >= NAME_REPEAT_MIN}
        cross_rate = self.cross_dup / self.cross_checked if self.cross_checked else 0.0
        per_domain = self.per_domain_dup_rate()
        return {
            "records_total": self.records_total,
            "cross_batch_dup_rate": round(cross_rate, 4),
            "cross_batch_dup_alarm": cross_rate > CROSS_BATCH_WARN,
            "name_collapse": {
                "repeated_names_top20": dict(
                    sorted(collapsed.items(), key=lambda kv: -kv[1])[:20]),
                "records_with_name_fraction": round(
                    self.records_with_names / max(self.records_total, 1), 4),
            },
            "per_domain_text_dup_rate": per_domain,
            "watchlist": {
                d: {"text_dup_rate": per_domain.get(d, 0.0),
                    "note": "B.3 檢查 5 預先觀察名單(文字層預警;正式判定在 embedding 空間)"}
                for d in watchlist
            },
            "batch_discard_curve": self.batch_discard_curve,
        }


# ---- 生成主流程 ----

def load_existing(out_path: pathlib.Path, monitors: Monitors) -> collections.Counter:
    counts = collections.Counter()
    if not out_path.exists():
        return counts
    with out_path.open() as f:
        for line in f:
            obj = json.loads(line)
            counts[obj["domain"]] += 1
            monitors.note_accept(obj["domain"], trigrams(obj["text"]), obj["text"])
    return counts


def write_prompts_md(path: pathlib.Path, model: str, args):
    path.write_text(f"""# PROMPTS.md — 語料生成方法全文(B.4;方法可重現本身是賣點)

- 生成模型(DECISIONS #13,首批釘定):`{model}`
- temperature:每批於 [0.9, 1.1] 均勻抽樣(B.2)
- 批大小:{args.batch_size}(B.2:20–40)
- 同批去重:3-gram Jaccard > {JACCARD_THRESHOLD} 丟棄重生
- 跨批監控:抽樣 {CROSS_SAMPLE_SIZE} 筆比對,重複率 > {CROSS_BATCH_WARN:.0%} 告警
- 域配比:configs/domains_v1.toml(DECISIONS #15)
- seed(域輪換與抽樣):{args.seed}

## System prompt(每批固定)

```
{SYSTEM_PROMPT}
```

## User prompt 模板(每批變動)

```
{USER_TEMPLATE}
```
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=5000)
    ap.add_argument("--config", default="configs/domains_v1.toml")
    ap.add_argument("--out", default="corpus/corpus_5k.jsonl")
    ap.add_argument("--report", default="corpus/gen_report_5k.json")
    ap.add_argument("--prompts-md", default="corpus/PROMPTS.md")
    # 2.5-flash 對新帳號已下架(2026-07-18 實測 404);3.5-flash 為現行穩定款,
    # 避開 -latest floating alias 與 preview(釘版紀律)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = 不限,>0 供小額試跑")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not os.environ.get("GEMINI_API_KEY"):
        # 直接 uv run(不經 make)時自行讀 repo 根目錄 .env
        env_file = pathlib.Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set(.env 或環境變數;#9 執行注記:個人帳號,配額先查)")
    from google import genai
    from google.genai import types
    # timeout 必設:無 timeout 時單一 stalled 請求會讓整個生成靜默卡死
    client = genai.Client(http_options=types.HttpOptions(timeout=120_000))  # ms

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    proportions = cfg["proportions"]
    tiers = cfg["tiers"]
    all_domains = [d for ds in tiers.values() for d in ds]
    missing = set(all_domains) ^ set(proportions)
    if missing:
        sys.exit(f"config mismatch between tiers and proportions: {missing}")

    rng = random.Random(args.seed)
    monitors = Monitors(rng)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = load_existing(out_path, monitors)
    if counts:
        print(f"[resume] {sum(counts.values())} existing records loaded")

    targets = domain_targets(proportions, args.total)
    next_id = sum(counts.values()) + 1
    batch_no = 0
    write_prompts_md(pathlib.Path(args.prompts_md), args.model, args)

    with out_path.open("a") as out:
        while True:
            pending = [d for d in targets if counts[d] < targets[d]]
            if not pending or (args.max_batches and batch_no >= args.max_batches):
                break
            # 缺額最大者優先,同缺額隨機——近似 B.2 的批間輪換
            domain = max(pending, key=lambda d: (targets[d] - counts[d], rng.random()))
            n = min(args.batch_size, targets[domain] - counts[domain])
            batch_no += 1
            temp = rng.uniform(0.9, 1.1)
            try:
                resp = client.models.generate_content(
                    model=args.model,
                    contents=USER_TEMPLATE.format(domain=domain, n=n),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT, temperature=temp),
                )
            except Exception as e:  # 配額/暫時性錯誤
                msg = str(e)
                if "PerDay" in msg:
                    # 日配額打滿:重試無意義(2026-07-18 實測 free tier 20 req/天/模型)。
                    # 明確退出;輸出已逐筆 flush,之後同指令續跑。
                    print(f"[batch {batch_no}] daily quota exhausted "
                          f"({sum(counts.values())}/{args.total} done, resume-safe). "
                          "升 tier 或明日再跑。", file=sys.stderr)
                    break
                m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay': '(\d+)s", msg)
                wait = max(float(m.group(1)), 10.0) + 1 if m else 30.0
                print(f"[batch {batch_no}] API error (transient): backoff {wait:.0f}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue

            records, fmt_rejects = parse_batch(resp.text or "", domain)
            in_batch_discard = 0
            batch_grams = []
            for rec in records:
                if counts[domain] >= targets[domain]:
                    break
                g = trigrams(rec["text"])
                if any(jaccard(g, h) > JACCARD_THRESHOLD for h in batch_grams):
                    in_batch_discard += 1
                    continue
                monitors.check_cross_batch(g)  # 監控訊號,不丟棄
                batch_grams.append(g)
                rec_out = {"id": next_id, **{k: rec[k] for k in REQUIRED_KEYS}}
                out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                out.flush()
                monitors.note_accept(domain, g, rec["text"])
                counts[domain] += 1
                next_id += 1
            monitors.batch_discard_curve.append(
                [batch_no, domain, in_batch_discard, fmt_rejects])
            done = sum(counts.values())
            print(f"[batch {batch_no}] {domain}: +{len(batch_grams)} "
                  f"(dup-discard {in_batch_discard}, fmt-reject {fmt_rejects}) "
                  f"total {done}/{args.total}")

    report = {
        "config": args.config,
        "model": args.model,
        "seed": args.seed,
        "targets": targets,
        "counts": dict(counts),
        **monitors.report(watchlist=["religious ceremonies"]),
    }
    pathlib.Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report → {args.report}")
    if report["cross_batch_dup_alarm"]:
        print("WARNING: cross-batch dup rate > 2% — 調溫或縮批(B.2)", file=sys.stderr)


# ---- selftest(無 API)----

def selftest():
    a = trigrams("A red dog runs across the sunny park meadow")
    b = trigrams("A red dog runs across the sunny park meadow today")
    assert jaccard(a, a) == 1.0
    assert 0.5 < jaccard(a, b) < 1.0
    assert jaccard(a, trigrams("Completely different words entirely here now")) == 0.0

    names = extract_names("The vendor Maria Lopez waves near Blue Harbor Cafe. Children play.")
    assert "Maria Lopez" in names and "Blue Harbor Cafe" in names
    assert "The" not in names and "Children" not in names  # 句首詞跳過
    noise = extract_names("A worker installs LED panels while Dr. Chen and St. Mary volunteers watch.")
    assert "LED" not in noise and "Dr" not in noise and "St" not in noise
    assert any("Chen" in n for n in noise)  # 稱謂後的真名保留

    t = domain_targets({"a": 0.5, "b": 0.3, "c": 0.2}, 10)
    assert sum(t.values()) == 10 and t["a"] == 5

    recs, rej = parse_batch(
        '{"text": "One two three four five six seven eight nine ten eleven", '
        '"domain": "x", "scene": "s", "objects": ["o"], "time_of_day": "noon"}\n'
        'not json\n'
        '{"text": "too short", "domain": "x", "scene": "s", "objects": [], "time_of_day": "n"}',
        "weddings")
    assert len(recs) == 1 and rej == 2 and recs[0]["domain"] == "weddings"

    print("selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
