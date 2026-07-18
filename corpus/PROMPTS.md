# PROMPTS.md — 語料生成方法全文(B.4;方法可重現本身是賣點)

- 生成模型(DECISIONS #13,首批釘定):`gemini-3.5-flash`
- temperature:每批於 [0.9, 1.1] 均勻抽樣(B.2)
- 批大小:30(B.2:20–40)
- 同批去重:3-gram Jaccard > 0.6 丟棄重生
- 跨批監控:抽樣 200 筆比對,重複率 > 2% 告警
- 域配比:configs/domains_v1.toml(DECISIONS #15)
- seed(域輪換與抽樣):42

## System prompt(每批固定)

```
You are a dataset generator. Output JSON Lines, one record per line:
{"text": "...", "domain": "<batch domain>", "scene": "...", "objects": [...], "time_of_day": "..."}
Rules:
- "text" is a 2-4 sentence concrete scene description, written like a caption for a single photograph
- Every record must differ in participants, location, objects, actions, and time
- Never reuse a person name, place name, or business name that appeared earlier
- Vary sentence structure: do not open consecutive records with the same pattern
- No abstract commentary; describe only what is visible in the frame
```

## User prompt 模板(每批變動)

```
Domain: {domain}
Generate {n} records. Diversity axes (vary at least three per record):
number and age of participants / indoor vs outdoor / weather / time of day /
mood / camera distance (close-up to wide shot) / primary objects
```
