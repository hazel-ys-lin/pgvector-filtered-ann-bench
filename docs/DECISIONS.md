# 決策紀錄(Decision Log)

規格本體在 `DESIGN_v1.1.md`;此檔記錄動工後的拍板,設計文件同步小改處會標註日期。

## 2026-07-18 — 順序 2 動工前拍板(author: hazel-ys-lin)

| # | 決定點 | 拍板 |
|---|---|---|
| 1 | Plan 強制選型 | `enable_*` GUC 先行,pg_hint_plan 列 future work。`PlanMode` 枚舉移除 `force_index`,只剩 `auto` \| `force_exact`。`force_exact` 生效判準放寬:plan 類型為 `seq_scan` **或** `bitmap_meta` 皆算成功(兩者都是精確路徑);出現 `hnsw_index` 即 error。實作:`SET enable_indexscan = off`(HNSW 只能走 Index Scan node,關掉即保證精確;btree 仍可走 bitmap。冒煙實證:force_exact 下未出現 hnsw_index;「HNSW 無 bitmap 路徑」為 pgvector 0.8.1 實作層主張,升級版本時重驗)。 |
| 2 | PlanType 判定 | **優先序判定**:走訪整棵 plan 樹,`hnsw_index` > `bitmap_meta` > `seq_scan` > `other`(樹中任一 node 命中高優先級即定案)。索引以名稱匹配(harness 自建索引,名稱已知)。**平行掃描入映射**:`Parallel Seq Scan`(JSON 中為 `Seq Scan` + Gather/`Parallel Aware`)歸 `seq_scan`;平行 bitmap 同理歸 `bitmap_meta`。 |
| 3 | Docker image | `pgvector/pgvector:0.8.1-pg16` **釘 digest**:`sha256:33198da2828a14c30348d2ccb4750833d5ed9a44c88d840a0e523d7417120337`(2026-07-18 自 Docker Hub 查得)。啟動後仍 assert extversion = 0.8.1 雙保險。 |
| 4 | EXPLAIN 格式 | `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`,解析走 JSON;`ExplainResult.Raw` 存 JSON 全文供事後補分類。 |
| 5 | Schema | 維度參數化進 config(`vector(N)` 由 config 決定,不寫死)。欄位**機制命名**:`label_clustered` / `label_scattered`(依散布機制命名,不用領域語義名)。表名 `items`,id 為 `bigint`。 |
| 6 | HNSW 建索引參數 | **定案**(非 placeholder):`m=16, ef_construction=64`(pgvector 預設)。README 第 5 節對 Shi 2025 調參協定的自圓其說以此為基準。論證(2026-07-18 補,DESIGN A.5 加註):本工作命題為 plan 選擇行為而非索引調參,索引固定於預設對應多數生產部署實態,調參維度讓給 Shi 2025 協定。 |
| 7 | 延遲計法 | `SearchANN` 延遲 = client round-trip(harness 與 DB 同機 docker,網路成本可忽略);`EXPLAIN ANALYZE` 的 execution time 另存於 `ExplainResult`,不混用。**新增坑**:pgx extended protocol + prepared statement:①重複執行後 custom→generic 切換,plan 於量測中無聲變化;②generic plan 快取可能無視後續 session GUC 變更 → `ApplySession` 一律 `SET plan_cache_mode = force_custom_plan`,已補入 A.6 坑表。 |

## 2026-07-18 — 語料與參數拍板(順序 3 前,author: hazel-ys-lin)

| # | 決定點 | 拍板 |
|---|---|---|
| 8 | 語料語言 | **英文**。理由:公開 repo 的可審查性(讀者可抽查語料)、PROMPTS.md 方法重現價值、英文公開授權參照資料集可得性。代價:中文錨點降為僅示意(原本即標注僅量級參考,可承受)。B.1 域表/B.2 prompt/query 集已同步英文化。 |
| 9 | Embedding 模型 | **gemini-embedding-001, 768d**。理由:索引尺寸合理、POC 全程行為熟悉、錨點連續性。dims 已參數化,換模型成本 = 重建參照分佈。執行注記:embed loader 併發與配額依**個人帳號**現行限制查定,不沿用公司帳號印象(**待辦**)。 |
| 10 | 真實文字參照樣本 | **COCO Captions 抽 1000 筆**(與待驗語料同文體 photo caption,參照效度依據)。**待辦:CC BY 4.0 授權現況查證,完成前不動工 validate 參照分佈部分。** |
| 11 | 冒煙 recall 門檻 | **0.95 裁決通過**(高斯向量為 B.3 定義之病態幾何,0.99 屬順序 3 真語料之檢查 6,不變)。附帶條件:recall 量測 query 斷言 PlanType = hnsw_index,結構性保證測的是 ANN 路徑(DESIGN 動工順序 2 加註)。 |
| 12 | force_index 活註記 | 順序 2 發現 ef-cost 翻轉現象(cost 隨 ef_search 線性放大,10k 實測 ef=400 時 seq+sort 勝出),force_index 可能因處置 (d')(= force_index + 深 ef,拆分演算法效果與含 planner 反應的實際效果)復活。**100k 實跑後裁決是否恢復枚舉**(DESIGN A.5b 加註)。 |
