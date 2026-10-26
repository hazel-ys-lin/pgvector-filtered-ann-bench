// bench — 主掃描 + 四方對照 runner(DESIGN A.5/A.5b;DECISIONS #21/#24/#25)。
//
// 5k 定位 = 管線彩排,非現象量測(動工順序 4 注記②):滿屏 recall 1.0
// 與大量精確 plan 是小規模的預期行為,不是 bug。
//
// 用法:
//	DATABASE_URL=... go run ./cmd/bench \
//	    --corpus corpus/corpus_5k.jsonl --vectors corpus/vectors_5k.jsonl \
//	    --queries corpus/queries_5k.jsonl --query-vectors corpus/queries_5k_vec.jsonl \
//	    --geometry corpus/geometry_report.json --out-dir results/run_5k
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/config"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/corpus"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/db"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/filters"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/metrics"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/query"
)

const warmupN = 5 // #25 規則③:每配置 warmup 不落表

// resultRow 是凍結的輸出 schema(DECISIONS #25)。
type resultRow struct {
	// 回溯鏈
	RunID      string `json:"run_id"`
	GitCommit  string `json:"git_commit"`
	CorpusSHA  string `json:"corpus_sha"`
	QueriesSHA string `json:"queries_sha"`
	GTCacheKey string `json:"gt_cache_key"`
	// 配置全參
	ConfigSet     string `json:"config_set"` // matrix | fourway
	Tier          string `json:"tier"`
	Dispersion    string `json:"dispersion"`
	ScatterSeed   int    `json:"scatter_seed"` // 0 = N/A 哨兵
	PlanMode      string `json:"plan_mode"`
	EfSearch      int    `json:"ef_search"`
	IterativeScan string `json:"iterative_scan"`
	K             int    `json:"k"`
	// query 身分
	QueryID   int64  `json:"query_id"`
	QSeed     int    `json:"qseed"`
	Stratum   string `json:"stratum"`
	QuotaFill bool   `json:"quota_fill"`
	Side      string `json:"side"` // enriched | cross
	// 量測值
	RecallAtK     float64 `json:"recall_at_k"`
	GTSize        int     `json:"gt_size"`
	NReturned     int     `json:"n_returned"`
	LatencyMS     float64 `json:"latency_ms"`
	ExpectedIDHit bool    `json:"expected_id_hit"`
	// plan
	PlanType    string `json:"plan_type"`
	Explained   bool   `json:"explained"`
	ExplainFile string `json:"explain_file,omitempty"`
	PlanAgree   bool   `json:"plan_agree"`
	// 建構預言 + tie
	SigmaLConstruct float64 `json:"sigma_l_construct"`
	GTBoundaryTie   bool    `json:"gt_boundary_tie"`
	TieWidth        int     `json:"tie_width"`
}

type benchConfig struct {
	set  string // matrix | fourway
	spec query.FilterSpec
	cfg  config.SweepConfig
}

type queryMeta struct {
	ID        int64  `json:"id"`
	QSeed     int    `json:"qseed"`
	SourceID  int64  `json:"source_id"`
	Stratum   string `json:"stratum"`
	QuotaFill bool   `json:"quota_fill"`
}

func main() {
	corpusPath := flag.String("corpus", "corpus/corpus_5k.jsonl", "")
	vecPath := flag.String("vectors", "corpus/vectors_5k.jsonl", "")
	queriesPath := flag.String("queries", "corpus/queries_5k.jsonl", "")
	qvecPath := flag.String("query-vectors", "corpus/queries_5k_vec.jsonl", "")
	filtersConfig := flag.String("filters-config", "configs/domains_v1.toml", "")
	geometryPath := flag.String("geometry", "corpus/geometry_report.json", "σ_l 建構值來源")
	outDir := flag.String("out-dir", "results/run", "")
	k := flag.Int("k", 50, "")
	skipLoad := flag.Bool("skip-load", false, "跳過灌庫/建索引(表已就緒時)")
	sets := flag.String("sets", "matrix,fourway", "跑哪些配置集")
	fourwayTiers := flag.String("fourway-tiers", "p3,p6", "四方對照檔位(A.5b:預期 p3/p6,實跑後微調)")
	flag.Parse()

	ctx := context.Background()
	must(os.MkdirAll(filepath.Join(*outDir, "explains"), 0o755))

	// ---- 載入與標籤 ----
	domains, qmeta := loadCorpusMeta(*corpusPath), loadQueryMeta(*queriesPath)
	corpusIDs, corpusVecs := loadVec32(*vecPath)
	queryIDs, queryVecs := loadVec32(*qvecPath)
	spec, err := filters.LoadSpec(*filtersConfig)
	must(err)
	clustered, err := spec.AssignClustered(domains)
	must(err)
	counts := filters.CountByLabel(clustered)
	scattered := map[int]map[int64]int16{}
	for _, s := range []int{1, 2, 3} {
		scattered[s] = filters.AssignScattered(corpusIDs, counts, int64(s))
	}
	labelOf := func(fs query.FilterSpec, id int64) int16 {
		if fs.Dispersion == "clustered" {
			return clustered[id]
		}
		return scattered[fs.ScatterSeed][id]
	}

	// ---- GT 快取(#21 Q2)----
	corpusSHA, queriesSHA := fileSHA(*vecPath), fileSHA(*qvecPath)
	gtKey := query.CacheKey(corpusSHA, queriesSHA, *k, "specs_v1")
	allSpecs := filterSpecs()
	gt, hit, err := query.BuildOrLoadGT(filepath.Join(*outDir, "gt_cache.json"), gtKey, *k,
		corpusIDs, corpusVecs, queryIDs, queryVecs, allSpecs, labelOf)
	must(err)
	fmt.Printf("GT cache %s (hit=%v, key=%s)\n", filepath.Join(*outDir, "gt_cache.json"), hit, gtKey)

	// ---- DB ----
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = "postgres://bench:bench@localhost:5433/bench"
	}
	dims := len(corpusVecs[0])
	d, err := db.Connect(ctx, url, dims)
	must(err)
	defer d.Close(ctx)
	if !*skipLoad {
		loadDB(ctx, d, corpusIDs, corpusVecs, clustered, scattered)
	}

	// ---- 距離對齊測試(#21 Q2:Go 裁判 vs DB <=>,ε=1e-5)----
	alignmentTest(ctx, d, corpusIDs, corpusVecs, queryVecs)

	// ---- σ_l 建構值(#23/#25)----
	sigmaL := loadSigmaL(*geometryPath)

	// ---- 配置集 ----
	var configs []benchConfig
	if strings.Contains(*sets, "matrix") {
		configs = append(configs, matrixConfigs(*k)...)
	}
	if strings.Contains(*sets, "fourway") {
		configs = append(configs, fourwayConfigs(*k, strings.Split(*fourwayTiers, ","))...)
	}
	fmt.Printf("%d configs × %d queries\n", len(configs), len(queryIDs))

	// ---- 主迴圈 ----
	runID := time.Now().UTC().Format("20060102T150405Z")
	commit := gitCommit()
	outF, err := os.Create(filepath.Join(*outDir, "results.jsonl"))
	must(err)
	defer outF.Close()
	w := bufio.NewWriter(outF)
	defer w.Flush()

	idxOf := map[int64]int{}
	for i, id := range corpusIDs {
		idxOf[id] = i
	}
	qIdxOf := map[int64]int{}
	for i, id := range queryIDs {
		qIdxOf[id] = i
	}

	for ci, bc := range configs {
		must(d.ApplySession(ctx, bc.cfg))
		entries := gt.Entries[bc.spec.Key()]

		// warmup(#25 規則③,不落表)
		for wi := 0; wi < warmupN && wi < len(qmeta); wi++ {
			q := mkQuery(qmeta[wi], queryVecs[qIdxOf[qmeta[wi].ID]], bc.spec)
			if _, _, err := d.SearchANN(ctx, q, bc.cfg); err != nil {
				must(fmt.Errorf("warmup: %w", err))
			}
		}

		// 代表 EXPLAIN:每 qseed 第一條(決定性),共 3;#25 規則② plan_agree
		repIDs := representativeQueries(qmeta)
		planTypes := map[int64]string{}
		for _, qid := range repIDs {
			pt := explainOne(ctx, d, qmeta, qIdxOf, queryVecs, bc, qid, *outDir, ci)
			planTypes[qid] = pt
		}
		agree := true
		var configPlan string
		for _, pt := range planTypes {
			if configPlan == "" {
				configPlan = pt
			} else if pt != configPlan {
				agree = false
			}
		}
		if !agree {
			// 邊界配置活體樣本:explained 覆蓋擴到全 query
			fmt.Printf("config %d PLAN DISAGREEMENT(邊界配置,縫 2 樣本): %v — expanding explains\n", ci, planTypes)
			for _, qm := range qmeta {
				if _, ok := planTypes[qm.ID]; !ok {
					planTypes[qm.ID] = explainOne(ctx, d, qmeta, qIdxOf, queryVecs, bc, qm.ID, *outDir, ci)
				}
			}
		}

		for _, qm := range qmeta {
			qv := queryVecs[qIdxOf[qm.ID]]
			q := mkQuery(qm, qv, bc.spec)
			got, lat, err := d.SearchANN(ctx, q, bc.cfg)
			must(err)
			e := entries[qm.ID]
			distOf := func(id int64) (float64, bool) {
				i, ok := idxOf[id]
				if !ok {
					return 0, false
				}
				return query.Dist32(corpusVecs[i], qv), true
			}
			rec, err := metrics.RecallTie(e.IDs, e.DK, e.GTSize, got, distOf, query.TieEps)
			must(err)
			expHit := false
			for _, id := range got {
				if id == qm.SourceID {
					expHit = true
					break
				}
			}
			side := "cross"
			if qm.Stratum == bc.spec.Tier {
				side = "enriched"
			}
			pt, explained := planTypes[qm.ID]
			if !explained {
				pt = configPlan
			}
			ef := ""
			if explained {
				ef = explainPath(*outDir, ci, qm.ID)
			}
			row := resultRow{
				RunID: runID, GitCommit: commit, CorpusSHA: corpusSHA,
				QueriesSHA: queriesSHA, GTCacheKey: gtKey,
				ConfigSet: bc.set, Tier: bc.spec.Tier, Dispersion: bc.spec.Dispersion,
				ScatterSeed: bc.spec.ScatterSeed, PlanMode: bc.cfg.PlanMode,
				EfSearch: bc.cfg.EfSearch, IterativeScan: bc.cfg.IterativeScan, K: bc.cfg.K,
				QueryID: qm.ID, QSeed: qm.QSeed, Stratum: qm.Stratum,
				QuotaFill: qm.QuotaFill, Side: side,
				RecallAtK: rec, GTSize: e.GTSize, NReturned: len(got),
				LatencyMS: float64(lat.Microseconds()) / 1000, ExpectedIDHit: expHit,
				PlanType: pt, Explained: explained, ExplainFile: ef, PlanAgree: agree,
				SigmaLConstruct: sigmaL[sigKey(bc.spec, side)],
				GTBoundaryTie:   e.BoundaryTie, TieWidth: e.TieWidth,
			}
			b, _ := json.Marshal(row)
			w.Write(append(b, '\n'))
		}
		if (ci+1)%20 == 0 || ci == len(configs)-1 {
			fmt.Printf("config %d/%d done\n", ci+1, len(configs))
		}
	}
	fmt.Printf("results → %s\n", filepath.Join(*outDir, "results.jsonl"))
}

// ---- 配置展開 ----

var tierList = []string{"p07", "p3", "p6", "p15", "p50"}

func specFor(dispersion string, seed int, tier string) query.FilterSpec {
	col := "label_clustered"
	if dispersion == "scattered" {
		col = fmt.Sprintf("label_scattered_s%d", seed)
	}
	return query.FilterSpec{Dispersion: dispersion, ScatterSeed: seed, Tier: tier,
		Column: col, Value: filters.TierOrdinal[tier]}
}

func filterSpecs() []query.FilterSpec {
	var out []query.FilterSpec
	for _, t := range tierList {
		out = append(out, specFor("clustered", 0, t))
		for _, s := range []int{1, 2, 3} {
			out = append(out, specFor("scattered", s, t))
		}
	}
	return out
}

// matrixConfigs:A.5 主矩陣——5 tier × 2 散布 × 4 ef × 2 iterative,auto;
// scattered 展開 3 filter seeds。
func matrixConfigs(k int) []benchConfig {
	var out []benchConfig
	for _, tier := range tierList {
		for _, ef := range []int{40, 100, 200, 400} {
			for _, iter := range []string{"off", "relaxed_order"} {
				cfg := config.SweepConfig{EfSearch: ef, IterativeScan: iter,
					PlanMode: "auto", K: k}
				out = append(out, benchConfig{"matrix", specFor("clustered", 0, tier), cfg})
				for _, s := range []int{1, 2, 3} {
					out = append(out, benchConfig{"matrix", specFor("scattered", s, tier), cfg})
				}
			}
		}
	}
	return out
}

// fourwayConfigs:A.5b 危險區四方——tier {p3, p6},處置 (a)(b)(c)(d200)(d400)。
func fourwayConfigs(k int, tiers []string) []benchConfig {
	treatments := []config.SweepConfig{
		{EfSearch: 40, IterativeScan: "off", PlanMode: "auto", K: k},           // (a)
		{EfSearch: 40, IterativeScan: "off", PlanMode: "force_exact", K: k},    // (b)
		{EfSearch: 40, IterativeScan: "relaxed_order", PlanMode: "auto", K: k}, // (c)
		{EfSearch: 200, IterativeScan: "off", PlanMode: "auto", K: k},          // (d)
		{EfSearch: 400, IterativeScan: "off", PlanMode: "auto", K: k},          // (d')
	}
	var out []benchConfig
	for _, tier := range tiers {
		for _, cfg := range treatments {
			out = append(out, benchConfig{"fourway", specFor("clustered", 0, tier), cfg})
			for _, s := range []int{1, 2, 3} {
				out = append(out, benchConfig{"fourway", specFor("scattered", s, tier), cfg})
			}
		}
	}
	return out
}

// ---- helpers ----

func mkQuery(qm queryMeta, vec []float32, fs query.FilterSpec) query.Query {
	return query.Query{ID: qm.ID, Vector: vec,
		Filter: &query.Filter{Column: fs.Column, Value: fs.Value}}
}

// representativeQueries:每 qseed 第一條(id 升冪決定性)。
func representativeQueries(qmeta []queryMeta) []int64 {
	seen := map[int]int64{}
	for _, q := range qmeta {
		if cur, ok := seen[q.QSeed]; !ok || q.ID < cur {
			seen[q.QSeed] = q.ID
		}
	}
	var out []int64
	for _, id := range seen {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func explainPath(outDir string, ci int, qid int64) string {
	return filepath.Join(outDir, "explains", fmt.Sprintf("c%03d_q%d.json", ci, qid))
}

func explainOne(ctx context.Context, d *db.DB, qmeta []queryMeta, qIdxOf map[int64]int,
	queryVecs [][]float32, bc benchConfig, qid int64, outDir string, ci int) string {
	var qm queryMeta
	for _, m := range qmeta {
		if m.ID == qid {
			qm = m
			break
		}
	}
	q := mkQuery(qm, queryVecs[qIdxOf[qid]], bc.spec)
	er, err := d.ExplainANN(ctx, q, bc.cfg)
	must(err) // force 不生效在此即中止(#25 規則②之前的既有紀律)
	must(os.WriteFile(explainPath(outDir, ci, qid), []byte(er.Raw), 0o644))
	return er.PlanType
}

func loadDB(ctx context.Context, d *db.DB, ids []int64, vecs [][]float32,
	clustered map[int64]int16, scattered map[int]map[int64]int16) {
	recs := make([]corpus.Record, len(ids))
	for i, id := range ids {
		recs[i] = corpus.Record{ID: id, Embedding: vecs[i],
			LabelClustered: clustered[id],
			LabelScattered: [3]int16{scattered[1][id], scattered[2][id], scattered[3][id]}}
	}
	must(d.EnsureSchema(ctx))
	must(d.LoadCorpus(ctx, recs))
	start := time.Now()
	stats, err := d.BuildHNSW(ctx, db.IndexOpts{M: 16, EfConstruction: 64})
	must(err)
	must(d.BuildMetaIndexes(ctx))
	must(d.Analyze(ctx))
	fmt.Printf("db loaded: hnsw %v (mwm=%s), total %v\n", stats.Duration, stats.MaintenanceWorkMem, time.Since(start))
}

// alignmentTest:#21 Q2——抽 20 對 (query, doc),Go float32 距離 vs DB <=>,
// |Δ| ≥ 1e-5 即中止(裁判與被測系統的距離定義漂移)。
func alignmentTest(ctx context.Context, d *db.DB, corpusIDs []int64,
	corpusVecs, queryVecs [][]float32) {
	worst := 0.0
	for i := 0; i < 20; i++ {
		qi := (i * 7919) % len(queryVecs)
		ciX := (i * 104729) % len(corpusVecs)
		goD := query.Dist32(corpusVecs[ciX], queryVecs[qi])
		dbD, err := d.DistanceTo(ctx, corpusIDs[ciX], queryVecs[qi])
		must(err)
		diff := goD - dbD
		if diff < 0 {
			diff = -diff
		}
		if diff > worst {
			worst = diff
		}
	}
	if worst >= query.TieEps {
		must(fmt.Errorf("distance alignment FAILED: worst |Go−DB| = %g ≥ %g", worst, query.TieEps))
	}
	fmt.Printf("distance alignment OK: worst |Go−DB| = %.2e (< %g)\n", worst, query.TieEps)
}

func loadSigmaL(path string) map[string]float64 {
	out := map[string]float64{}
	b, err := os.ReadFile(path)
	must(err)
	var rep struct {
		Checks []struct {
			Check int             `json:"check"`
			Value json.RawMessage `json:"value"`
		} `json:"checks"`
	}
	must(json.Unmarshal(b, &rep))
	for _, c := range rep.Checks {
		if c.Check != 7 {
			continue
		}
		var tiers []struct {
			Tier            string `json:"tier"`
			ClusteredEnrich struct {
				SigmaL float64 `json:"sigma_l"`
			} `json:"clustered_enriched"`
			ClusteredCross struct {
				SigmaL float64 `json:"sigma_l"`
			} `json:"clustered_cross"`
			ScatteredBySeed []struct {
				SigmaL float64 `json:"sigma_l"`
			} `json:"scattered_by_seed"`
		}
		must(json.Unmarshal(c.Value, &tiers))
		for _, t := range tiers {
			out["clustered/s0/"+t.Tier+"/enriched"] = t.ClusteredEnrich.SigmaL
			out["clustered/s0/"+t.Tier+"/cross"] = t.ClusteredCross.SigmaL
			for si, s := range t.ScatteredBySeed {
				// scattered 無幾何意義的側,兩側同值(pooled)
				out[fmt.Sprintf("scattered/s%d/%s/enriched", si+1, t.Tier)] = s.SigmaL
				out[fmt.Sprintf("scattered/s%d/%s/cross", si+1, t.Tier)] = s.SigmaL
			}
		}
	}
	if len(out) == 0 {
		must(fmt.Errorf("no check-7 sigma_l in %s(檢查 7 是 bench 前閘門,先跑 validate)", path))
	}
	return out
}

func sigKey(fs query.FilterSpec, side string) string {
	return fmt.Sprintf("%s/s%d/%s/%s", fs.Dispersion, fs.ScatterSeed, fs.Tier, side)
}

func loadCorpusMeta(path string) map[int64]string {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	out := map[int64]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r struct {
			ID     int64  `json:"id"`
			Domain string `json:"domain"`
		}
		must(json.Unmarshal(sc.Bytes(), &r))
		out[r.ID] = r.Domain
	}
	must(sc.Err())
	return out
}

func loadQueryMeta(path string) []queryMeta {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	var out []queryMeta
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var q queryMeta
		must(json.Unmarshal(sc.Bytes(), &q))
		out = append(out, q)
	}
	must(sc.Err())
	return out
}

func loadVec32(path string) ([]int64, [][]float32) {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	var ids []int64
	var vecs [][]float32
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r struct {
			ID        int64     `json:"id"`
			Embedding []float64 `json:"embedding"`
		}
		must(json.Unmarshal(sc.Bytes(), &r))
		v := make([]float32, len(r.Embedding))
		for i, x := range r.Embedding {
			v[i] = float32(x)
		}
		ids = append(ids, r.ID)
		vecs = append(vecs, v)
	}
	must(sc.Err())
	return ids, vecs
}

func fileSHA(path string) string {
	f, err := os.Open(path)
	must(err)
	defer f.Close()
	h := sha256.New()
	_, err = io.Copy(h, f)
	must(err)
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func gitCommit() string {
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "bench:", err)
		os.Exit(1)
	}
}
