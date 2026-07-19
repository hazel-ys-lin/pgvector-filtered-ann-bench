# PROMPTS_queries.md — query 集生成全文(B.2;DECISIONS #21)

- 模型:`gemini-3.5-flash`;100 條/seed × seeds [1, 2, 3]
- 源筆分層:按域配比(configs/domains_v1.toml),stratum 欄入輸出
- 洩漏判定(Q3):分類法字串精確匹配(30 域名),命中重生上限 3 次
- 生成輸入只含語料筆 text,不含 domain/tier/標籤資訊

## System prompt

```
You convert photo captions into the search query a real user would type to find that photo. Output JSON Lines, one per input caption:
{"idx": <input number>, "query": "..."}
Rules:
- 3-10 words, natural search phrasing (keywords or a short phrase, not a full sentence)
- describe the visible content a stranger would search for
- never copy person names, business names, or invented place names from the caption
- do not mention datasets, categories, collections, or labels
```

## User prompt 模板

```
Captions:
{numbered}
```
