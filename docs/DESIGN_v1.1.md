# pgvector-filtered-ann-bench v1.1 — 動工前設計文件

**目的**:公開、可重現的 pgvector filtered-ANN 行為量測研究(measurement study)。
**v1.1 改版(2026-07)**:文獻查核後重新定位——納入 Amanbayev et al. 2026 的發現,新增危險區四方對照實驗、EXPLAIN 全量記錄與 plan 解析、GLS 指標計算。
**語言**:Go(harness)+ Python(語料生成腳本,可選)
**時程錨點**:8 月中 harness 完成 → 8 月底 README 成稿 → 9 月初發佈

---

## 0\. 定位與文獻脈絡(v1.1 新增,決定一切取捨)

近 18 個月已有四篇 FANNS benchmark(Amanbayev 2026 / Shi 2025 / Li 2025 / Iff 2025)加一篇成本估計(Xia 2026)。單純的「選擇性 × recall 掃描」已無新意。本 repo 收窄為三個仍空著的縫:

1. **散布的受控因果實驗**:既有工作把 query-filter 相關性當「觀測後的監控訊號」(Amanbayev 的 GLS、guideline 8);本工作在**同構向量上建構**聚集/散開兩種標籤,做受控對照——從觀測升級為因果。並採用 GLS 作為量化語言(對兩套標籤各計算 GLS,接上既有詞彙體系)。

2. **Planner 對錯邊界地圖**:Amanbayev 發現「低選擇性時 optimizer 常錯選 approximate scan、錯過可拿滿分的精確路徑」;本 POC 前作觀察到「極低選擇性時 planner 正確切 Seq Scan、天然免疫」。兩者不矛盾——同一個 cost model 在不同條件下有對有錯,**對錯邊界由什麼決定(選擇性?schema/denormalize?統計時機?)沒有人畫過完整地圖**。本 repo 以全量 EXPLAIN 記錄開始畫這張圖。ef 本身是 plan 邊界的移動變因(README 第 6 節素材)。

3. **語料幾何方法論**(升為共同主角):四篇 benchmark 全部依賴固定的真實資料集,沒有人回答「想在自己的規模與領域上量測的人,怎麼確保合成語料量出的數字可信」。本 repo 的 LLM 語料 + 幾何驗證守門是這一塊的第一份系統性方法。

核心敘事一句話(README 與 statement 共用):**對 filtered search 而言,plan 往往比 index 更重要——而 plan 的自動選擇目前只能靠人工 EXPLAIN 補救,本工作量測這個缺口,為自動化(研究提案)鋪路。**

範圍外但必讀的文獻義務:Shi 2025 的標準化調參協定(v1 參數設定需能對其自圓其說)、Li 2025 的選擇性掃描設計(確認不重複)、Xia 2026(題目二 statement 前第一必讀)。

---

## Part A — Harness 模組架構

### A.0 設計原則

1. **Harness 不是被測物**:所有量測發生在 Postgres 內或以 DB 回應計時,harness 只負責編排與記錄。

2. **一切可重現**:每個結果列可回溯到 (corpus_id, config, seed, git commit);README 每張表附重現指令。

3. **Sanity check 內建**:recall ≤ 1 斷言、ef_search ≥ k 檢查、ground truth 與 ANN 用同一份 query 向量。

4. **用 driver 不用 psql 文字輸出**:全部走 pgx,杜絕解析事故。

5. **(v1.1 新增)EXPLAIN 是一級資料**:每個配置必存 EXPLAIN (ANALYZE, BUFFERS) 並解析 plan 類型入結果表——planner 邊界地圖是正式研究問題,不是附錄。

### A.1 目錄與模組切分

```
pgvector-filtered-ann-bench/
├── cmd/
│   ├── bench/          # 主入口:讀 config → 跑 sweep → 出報表
│   ├── corpusgen/      # LLM 生成文字語料
│   ├── embed/          # 併發 embed loader(文字 → 向量 → JSONL)
│   └── validate/       # 語料幾何驗證 + GLS 計算(Part B)
├── internal/
│   ├── config/         # 實驗矩陣定義(TOML)與展開
│   ├── corpus/         # JSONL 讀寫、語料 metadata
│   ├── db/             # schema、索引、session 設定、plan 強制、EXPLAIN 擷取
│   ├── filters/        # 選擇性 × 散布的標籤指派 + GLS 量測
│   ├── query/          # query 集、exact ground truth、ANN 查詢執行
│   ├── metrics/        # recall@k、延遲百分位、上界斷言、plan 解析
│   └── report/         # CSV/JSON 原始結果 + Markdown 表生成
├── configs/
│   ├── v1_matrix.toml  # 主掃描:5 選擇性 × 2 散布 × 4 ef × 2 iterative
│   └── v1_fourway.toml # 危險區四方對照(A.5b)
├── docker-compose.yml  # postgres:16 + pgvector ≥ 0.8.1,shm_size 已設
├── Makefile            # make corpus / make bench / make fourway / make report
└── README.md           # 技術報告(骨架見 B.5)
```

### A.2 資料流

```
[corpusgen] 文字 JSONL
    → [embed] 向量 JSONL(worker pool 併發呼叫 embedding API)
    → [validate] 幾何六項檢查 + 兩套標籤的 GLS 計算(不過 = 停)
    → [db] 灌庫 + 建 HNSW + 建 metadata 索引(見 A.4 注記)
    → [filters] 指派標籤(聚集型 = 領域欄位;散開型 = seed 隨機)
    → [query] 每 (filter, query) 算 exact ground truth(一次,快取)
    → [bench] 主掃描 + 四方對照 → 每配置記 recall/延遲/EXPLAIN plan
    → [report] results/*.csv + README 用 Markdown 表
```

### A.3 核心介面(Go 簽名草案)

```go
// internal/config
type SweepConfig struct {
    CorpusID      string
    Selectivity   string   // "p07" | "p3" | "p6" | "p15" | "p50"
    Dispersion    string   // "clustered" | "scattered"
    EfSearch      int      // 40 | 100 | 200 | 400
    IterativeScan string   // "off" | "relaxed_order"
    PlanMode      string   // v1.1: "auto" | "force_exact"(2026-07-18 拍板移除 force_index,見 DECISIONS.md #1)
    K             int      // 50
    Seed          int64
}

// internal/db
type DB interface {
    LoadCorpus(ctx, records []corpus.Record) error
    BuildHNSW(ctx, opts IndexOpts) (BuildStats, error)
    BuildMetaIndexes(ctx) error                    // v1.1: filter 欄位 btree 索引
    ApplySession(ctx, cfg SweepConfig) error       // SET ef/iterative + plan 強制
    SearchANN(ctx, q Query, cfg SweepConfig) (ids []int64, lat time.Duration, err error)
    ExplainANN(ctx, q Query, cfg SweepConfig) (ExplainResult, error) // v1.1: 每配置必跑
}
// plan 強制實作(2026-07-18 拍板,DECISIONS.md #1):enable_* GUC 先行,pg_hint_plan 列 future work;
// force_exact = SET enable_indexscan=off,生效判準 = plan 為 seq_scan 或 bitmap_meta(兩種精確路徑皆可)。
// README 註明 force 僅為實驗對照,非生產建議。

// internal/metrics
func RecallAtK(gt, got []int64, k int) (float64, error)   // 內含 ≤1 斷言
type ExplainResult struct {
    Raw      string
    PlanType string   // v1.1 解析:"hnsw_index" | "seq_scan" | "bitmap_meta" | "other"
}

// internal/filters — v1.1 新增 GLS 量測
// 對每 (query, filter):σ_l = 精確 top-k 中通過 filter 的比例(由 GT 免費得出)
//                     σ_g = filter 全域選擇性
// 輸出 per-query 比值 r = σ_l/σ_g 與語料級平均,寫入 geometry_report.json
func MeasureGLS(gt GroundTruth, f Filter) GLSStats
```

### A.4 filters 模組——散布軸(本 benchmark 的因果核心)

- **聚集型(clustered)**:filter 欄位 = 語料生成時的領域標籤。同領域文字天然成叢 → 通過集在向量空間聚集 → 預期 GLS 顯著偏離 1(query 落在同域時富集、跨域時耗竭)。

- **散開型(scattered)**:filter 欄位 = 固定 seed 隨機指派,與內容無關 → 預期 GLS ≈ 1(局部 ≈ 全域)。

同一份向量、兩套標籤欄位:幾何完全同構,唯一變因是 filter 與叢集的相關性。GLS 實測值作為「建構成功」的操作型驗證(散開型 r 應緊貼 1,聚集型應顯著偏離),寫進 README——這讓散布軸從直覺概念變成有量化憑證的受控變因。

**Metadata 索引注記(v1.1)**:Amanbayev guideline 6 指出 metadata 索引的存在改變 planner 的可選 plan 集合。v1 預設**建立** filter 欄位的 btree 索引(貼近生產實務,也讓 planner 邊界地圖涵蓋「有 attribute-index 可選」的情境);「無 metadata 索引」情境列 future work。

### A.5 主掃描與量測紀律

主矩陣:5 選擇性 × 2 散布 × 4 ef(40/100/200/400)× 2 iterative(off/relaxed)= 80 配置,PlanMode 固定 auto。每配置:ApplySession → 暖機 → 正式跑 query 集(50–100 條,全配置共用)→ 對 GT 算 recall → **每配置對至少 3 條代表 query 存 EXPLAIN 並解析 plan 類型入結果表**。filter 指派與 query 抽樣各 3 seed,報告附 std。延遲只計 DB 端(向量預先算好)。索引每語料建一次,配置切換只動 session。HNSW 建構參數定案(2026-07-18 拍板,DECISIONS.md #6):m=16 / ef_construction=64(pgvector 預設)。論證(README 第 5 節對 Shi 2025 的說明用):本工作命題為 plan 選擇行為而非索引調參,索引固定於預設對應多數生產部署實態,調參維度讓給 Shi 2025 協定。

### A.5b 危險區四方對照(v1.1 新增,README 的主打表)

在危險區選擇性檔位(預期 p3/p6,實跑後微調),同一組 query 比四種處置:

| 處置 | 設定 | 對應誰的主張 | 
|---|---|---|
| (a) optimizer 預設 | PlanMode=auto, iterative=off, ef=40 | 現狀基線 | 
| (b) 強制精確路徑 | PlanMode=force_exact | Amanbayev §7.2.8 的救法 | 
| (c) iterative 救援 | auto, iterative=relaxed, ef=40 | pgvector 官方救援 | 
| (d) 加深 search budget | auto, iterative=off, ef=200/400 | 本 POC 前作的發現 | 

每格報 recall@50 + p50/p95 延遲 + plan 類型。此表無任何既有論文完整擁有((b) 他們有、(c)(d) 他們無;POC 前作有 (c)(d) 無 (b)),同時是使用者決策表與「自適應機制該選哪條路」的第一份四方實證。

**(d) 註記(2026-07-18,DECISIONS.md #12)**:順序 2 冒煙觀測到 pgvector cost 隨 ef_search 線性放大,深 ef 可能將 planner 推離 index(10k 實測 ef=400 時 seq+sort 勝出)。(d) 維持 auto 不變(量測使用者實際可得的行為);若 100k 危險區實跑確認翻轉發生,增設 (d') = force_index + 深 ef,拆分「演算法效果」與「含 planner 反應的實際效果」。

### A.6 環境已知坑(寫進 docker-compose / Makefile)

| 坑 | 對策 | 
|---|---|
| Docker 預設 /dev/shm=64MB,平行 build 失敗 | compose 設 shm_size: 2gb | 
| maintenance_work_mem 預設 64MB → 磁碟建構病態慢 | build 前 SET 2GB | 
| iterative_scan 需 pgvector ≥ 0.8 | image 釘版本,啟動 assert extversion | 
| enable_seqscan 等 GUC 是 session 級 | ApplySession 每配置重設,防洩漏到下一配置 | 
| planner 統計影響 plan 選擇 | 灌庫與建索引後必跑 ANALYZE,並記錄於重現指令 | 
| (2026-07-18 拍板)pgx extended protocol + prepared statement:①重複執行後 custom→generic 切換,plan 於量測中無聲變化;②generic plan 快取可能無視後續 session GUC 變更 | ApplySession 一律 SET plan_cache_mode = force_custom_plan | 

---

## Part B — 語料生成 + 幾何驗證 + GLS

### B.0 方法定位(README 第 3 節的論證)

檢索向量來自文字——不需要真實照片,需要多樣的真實文字。LLM 生成 → 真實 embedding 模型轉向量 → 幾何與真實語料同形、規模任意。此節同時回應文獻缺口:四篇既有 benchmark 均綁定固定真實資料集,本方法回答「自有規模/領域的量測如何確保效度」。兩個失敗模式(prompt 設計即在防):描述雷同 → 近重複 ties → recall 假崩;描述空泛 → 無叢集 → 不像真實資料。

### B.1 領域輪換表(通用公開版,30 域)

sports events / music performances / weddings / graduation ceremonies / street markets / hiking & mountaineering / beach activities / cooking & cuisine / coffee & desserts / museum exhibitions / animals & pets / wildlife / city streetscapes / architecture & landmarks / transportation / gardening & plants / work scenes / medical scenes / classrooms & courses / children's play / festival parades / religious ceremonies / agriculture & fishing / construction & manufacturing / tech expos / esports & board games / dance rehearsals / studio photography / camping & picnics / volunteering & charity

(2026-07-18 拍板英文 30 域,DECISIONS.md #8;「農事漁業」合併譯為 agriculture & fishing,配比結構不動。)"religious ceremonies" 注記:生成時多樣性軸須涵蓋跨宗教場景,防止塌向單一文化預設;此域列為 B.3 檢查 5(單域近重複率)的預先觀察名單。

每域生成量 = 該域目標選擇性 × 總量(聚集型 filter 的選擇性由此控制)。

### B.2 Prompt 模板

**System(每批固定;2026-07-18 拍板英文定稿,DECISIONS.md #8)**:

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

**User(每批變動)**:

```
Domain: {domain}
Generate {N} records. Diversity axes (vary at least three per record):
number and age of participants / indoor vs outdoor / weather / time of day /
mood / camera distance (close-up to wide shot) / primary objects
```

**生成參數**:temperature 0.9–1.1;每批 20–40 筆;批間輪換領域;total 100k(約 3,000 批,embed loader 併發吞掉)。
**批後去重**:同批 3-gram Jaccard > 0.6 丟棄重生;跨批抽樣監控重複率 > 2% 即調溫或縮批。英文生成對預設人名/地名的塌縮傾向較強,首批 5k 後檢視實際重複率,必要時收緊 2% 閾值。
**Query 集同步生成**:對隨機抽出的語料筆生成「使用者會怎麼搜這張」的查詢並記錄該筆 id 作為 expected(構造性 GT,limitations 註明)。核心指標 id-recall 的裁判是精確掃描,不依賴此 GT。query 隨語料改為英文。

### B.3 幾何驗證檢查清單(cmd/validate 自動跑,全過才進 benchmark)

| \# | 檢查 | 方法 | 通過準則 | 失敗含意 | 
|---|---|---|---|---|
| 1 | 兩兩距離中位數 | 抽 5k 對,cosine | 同模型真實文字參照樣本中位數 ±0.05 | 過大→近正交;過小→擠團 | 
| 2 | 距離集中度 | 分佈 std/mean | ≥ 參照樣本的 70% | 過窄 → HNSW 分不出鄰居 | 
| 3 | 近重複率 | 最近鄰距離 < 0.05 比例 | 1–3% | 過高 → ties 崩 recall | 
| 4 | 叢集結構存在 | 域內/跨域平均距離比 | 0\.6–0.9 | ≈1 → 無叢集,聚集型實驗失效 | 
| 5 | 無退化叢 | 單域近重複率 | 每域 < 5% | 該域 prompt 塌縮,重生 | 
| 6 | End-to-end 冒煙 | no-filter ef=200 recall@50 vs exact | ≥ 0.99 | 幾何或索引有問題,先修 | 
| 7 | (v1.1)GLS 建構驗證 | MeasureGLS 兩套標籤 | 散開型 r ≈ 1(±0.15);聚集型顯著偏離 | 散布軸建構失敗,調域配比 | 

檢查 3/4/5 閾值為沿用先前工作的初始值(結構比值、語言敏感度低,但檢查 4 的域內/跨域距離比依賴模型與域切分粒度),首個 5k 語料實跑後校準,調整記入 DECISIONS。

**校準錨點(2026-07-18 改寫,DECISIONS.md #8/#10)**:v1 語料為英文。原中文錨點(0.166/0.177/0.195)降級為**僅示意,出自先前中文工作**,不得作為本 repo 通過準則。正式參照分佈依本節規定重建:真實文字參照樣本採 **COCO Captions**(CC BY 4.0;動工前完成授權現況查證,查證待辦記入 DECISIONS.md #10),抽 1000 筆 caption 以 gemini-embedding-001 768d 建立參照分佈。參照樣本與待驗語料同文體(photo caption),此為參照效度依據,README 第 4 節引用。壞合成對照(高斯噪聲 0.85–0.87)維持量級參考。檢查 1、2 的參照分佈檔案路徑於 validate config 參數化,不寫死數字。

### B.4 產出物

- corpus/corpus_100k.jsonl、corpus/queries.jsonl

- corpus/geometry_report.json(七項檢查 + 兩套標籤 GLS,README 引用)

- corpus/PROMPTS.md(prompt 與參數全文——方法可重現本身是賣點)

### B.5 README 章節骨架(v1.1 改版)

1. **TL;DR**——主句:「對 filtered search,plan 往往比 index 更重要」;三發現:危險區四方對照的結論、散布(GLS)的因果效應、合成語料效度方法;附四方對照主圖。

2. **Motivation**——生產場景的 filtered-ANN 痛點;現有最佳實踐是人工 EXPLAIN(引 Amanbayev §7.2.8)。

3. **Related work(v1.1 新增,必寫)**——Amanbayev 2026 / Shi 2025 / Li 2025 / Iff 2025 / Xia 2026 各一句定位 + 本工作的三個縫(§0)。建設性銜接:選擇性主掃描定位為「與 Amanbayev 對齊的重現基線」,並評估直接掛接其 ANN-Benchmarks 擴充。

4. **Why synthetic corpora fail**——距離集中根因 + LLM 語料方法 + 七項守門(距離分佈對比圖)。

5. **Experiment setup**——環境、語料、GLS 值、指標、重現指令;參數設定對 Shi 2025 調參協定的說明;向量 JSONL 隨 release assets 發佈(自產,無智財問題),重現分兩層——重現量測零 API 依賴,重現語料生成才需 API key(可重現性設計的一部分)。

6. **Results**——四方對照表(主打)、三段 planner 地圖(附 plan 類型統計)、散布 × 選擇性交互表、iterative vs ef 取捨。

7. **Practical guidance**——決策表;明標哪幾條與 Amanbayev guidelines 一致、哪幾條是本工作新增((c)(d) 處置、散布判讀)。

8. **Limitations & future work**——規模依賴、單系統、構造性 GT、IVFFlat 未測、v2(跨實作驗證 + Rust 白箱)。

---

## 動工順序(v1.1)

1. (前置,非 harness)精讀 Amanbayev 全文 + 掃 Shi/Li/Iff/Xia 摘要與結論,確認 §0 三縫無誤——**此步在寫任何程式前完成**

2. docker-compose + db 模組(含 plan 強制與 EXPLAIN 解析)+ 冒煙(1 週末)
   (2026-07-18 拍板,DECISIONS.md #11:冒煙 no-filter recall 門檻 0.95——高斯向量為 B.3 定義之病態幾何,0.99 屬順序 3 真語料之檢查 6,不變;附帶條件:recall 量測 query 斷言 PlanType = hnsw_index,結構性保證測的是 ANN 路徑)

3. corpusgen + embed + validate(含 GLS),5k 小語料過七項檢查(1 週末)

4. filters + query + GT 快取 + bench runner,小語料跑通主矩陣與四方對照(1 週末)

5. 100k 全量生成與正式 sweep(掛機)

6. report + README(含 related work 段)(1–2 週末)