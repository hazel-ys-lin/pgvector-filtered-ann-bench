// validate — 語料幾何驗證,B.3 檢查 1–6(檢查 7 GLS 需 GT,屬順序 4)。
//
// 檢查 1/2 以 COCO 參照分佈為準(DECISIONS #10/#18),參照與語料同嵌入條件
// (#16)。檢查 3/4/5 閾值為沿用初始值,5k 實跑後校準(B.3 註記)。
// 檢查 6 需 DATABASE_URL(docker compose 的 Postgres),量測 ANN 路徑
// recall 並斷言 PlanType = hnsw_index(#11 紀律)。
//
// 用法:
//	DATABASE_URL=... go run ./cmd/validate \
//	    --corpus corpus/corpus_5k.jsonl --vectors corpus/vectors_5k.jsonl \
//	    --ref corpus/coco_ref_1k_vec.jsonl --report corpus/geometry_report.json
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/config"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/corpus"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/db"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/filters"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/query"
)

type checkResult struct {
	Check     int     `json:"check"`
	Name      string  `json:"name"`
	Value     any     `json:"value"`
	Criterion string  `json:"criterion"`
	Pass      bool    `json:"pass"`
	Note      string  `json:"note,omitempty"`
}

type report struct {
	GeneratedAt string        `json:"generated_at"`
	CorpusFile  string        `json:"corpus_file"`
	VectorsFile string        `json:"vectors_file"`
	RefFile     string        `json:"ref_file"`
	Seed        int64         `json:"seed"`
	PairSample  int           `json:"pair_sample"`
	Checks      []checkResult `json:"checks"`
	AllPass     bool          `json:"all_pass"`
}

func main() {
	corpusPath := flag.String("corpus", "corpus/corpus_5k.jsonl", "文字語料(id→domain)")
	vecPath := flag.String("vectors", "corpus/vectors_5k.jsonl", "語料向量")
	refPath := flag.String("ref", "corpus/coco_ref_1k_vec.jsonl", "COCO 參照向量")
	reportPath := flag.String("report", "corpus/geometry_report.json", "報表輸出")
	seed := flag.Int64("seed", 42, "抽樣 seed")
	pairs := flag.Int("pairs", 5000, "檢查 1/2 抽樣對數")
	nearDup := flag.Float64("near-dup", 0.05, "檢查 3/5 近重複距離閾值")
	skipDB := flag.Bool("skip-db", false, "跳過檢查 6(無 DB 環境)")
	queriesPath := flag.String("queries", "", "queries JSONL;給定即跑檢查 7(GLS)")
	queryVecPath := flag.String("query-vectors", "", "query 向量 JSONL(檢查 7)")
	filtersConfig := flag.String("filters-config", "configs/domains_v1.toml", "檔位映射(檢查 7)")
	flag.Parse()

	ids, vecs, err := loadVectors(*vecPath)
	must(err)
	domains, err := loadDomains(*corpusPath)
	must(err)
	_, refVecs, err := loadVectors(*refPath)
	must(err)
	fmt.Printf("corpus %d vecs, ref %d vecs\n", len(vecs), len(refVecs))

	rng := rand.New(rand.NewSource(*seed))
	var checks []checkResult

	// 參照分佈(檢查 1/2 的準繩)
	refMed, refStdMean := pairStats(refVecs, *pairs, rng)
	corpMed, corpStdMean := pairStats(vecs, *pairs, rng)

	// 檢查 1:距離中位數 ∈ 參照 ±0.05
	c1 := checkResult{Check: 1, Name: "pairwise distance median",
		Value:     map[string]float64{"corpus": round4(corpMed), "ref": round4(refMed)},
		Criterion: "corpus median within ref ±0.05",
		Pass:      math.Abs(corpMed-refMed) <= 0.05}
	checks = append(checks, c1)

	// 檢查 2:距離集中度 std/mean ≥ 參照的 70%
	c2 := checkResult{Check: 2, Name: "distance concentration (std/mean)",
		Value:     map[string]float64{"corpus": round4(corpStdMean), "ref": round4(refStdMean)},
		Criterion: "corpus std/mean >= 0.7 × ref",
		Pass:      corpStdMean >= 0.7*refStdMean}
	checks = append(checks, c2)

	// 全語料最近鄰距離(檢查 3/5 共用;brute force,goroutine 平行)
	nn := nearestNeighborDist(vecs)

	// 檢查 3:近重複率 1–3%(初始閾值,校準候選)
	dupCount := 0
	for _, d := range nn {
		if d < *nearDup {
			dupCount++
		}
	}
	dupRate := float64(dupCount) / float64(len(nn))
	c3 := checkResult{Check: 3, Name: "near-duplicate rate (NN dist < 0.05)",
		Value: round4(dupRate), Criterion: "<= 3%(2026-07-18 校準撤下限,DECISIONS #20)",
		Pass: dupRate <= 0.03,
		Note: "機制單側:失敗模式僅在過高側(ties 崩 recall)。實測語料 0.76% vs " +
			"COCO 參照 4.50%——語料 tie 密度低於真實 caption,README limitations 揭露"}
	checks = append(checks, c3)

	// 檢查 4:域內/跨域平均距離比 0.6–0.9
	ratio := intraCrossRatio(vecs, ids, domains, *pairs, rng)
	c4 := checkResult{Check: 4, Name: "intra/cross domain distance ratio",
		Value: round4(ratio), Criterion: "0.6–0.9(沿用初始值,模型/域粒度敏感,可校準)",
		Pass: ratio >= 0.6 && ratio <= 0.9}
	checks = append(checks, c4)

	// 檢查 5:單域近重複率 < 5%(watchlist: religious ceremonies)
	perDomain := perDomainDupRate(vecs, ids, domains, *nearDup)
	worstDomain, worstRate := "", 0.0
	for d, r := range perDomain {
		if r > worstRate {
			worstDomain, worstRate = d, r
		}
	}
	c5 := checkResult{Check: 5, Name: "per-domain near-duplicate rate",
		Value: map[string]any{
			"worst_domain": worstDomain, "worst_rate": round4(worstRate),
			"per_domain":   roundMap(perDomain),
		},
		Criterion: "every domain < 5%",
		Pass:      worstRate < 0.05}
	checks = append(checks, c5)

	// 檢查 6:end-to-end,no-filter ef=200 recall@50 ≥ 0.99,斷言 hnsw_index
	if *skipDB {
		checks = append(checks, checkResult{Check: 6, Name: "end-to-end recall (SKIPPED)",
			Criterion: "no-filter ef=200 recall@50 >= 0.99", Pass: false,
			Note: "--skip-db;此檢查未跑,報表不得視為全過"})
	} else {
		c6, err := checkEndToEnd(ids, vecs, domains, rng)
		must(err)
		checks = append(checks, c6)
	}

	// 檢查 7:GLS 建構驗證(#15 分檔位準則;bench runner 前閘門)
	if *queriesPath != "" {
		c7, err := checkGLS(*queriesPath, *queryVecPath, *filtersConfig, ids, vecs, domains)
		must(err)
		checks = append(checks, c7)
	}

	all := true
	for _, c := range checks {
		fmt.Printf("check %d [%s] pass=%v value=%v\n", c.Check, c.Name, c.Pass, c.Value)
		all = all && c.Pass
	}
	rep := report{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		CorpusFile:  *corpusPath, VectorsFile: *vecPath, RefFile: *refPath,
		Seed: *seed, PairSample: *pairs, Checks: checks, AllPass: all,
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	must(os.WriteFile(*reportPath, append(b, '\n'), 0o644))
	fmt.Printf("report → %s (all_pass=%v)\n", *reportPath, all)
	if !all {
		os.Exit(1)
	}
}

// ---- 載入 ----

func loadVectors(path string) ([]int64, [][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	var ids []int64
	var vecs [][]float64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r struct {
			ID        int64     `json:"id"`
			Embedding []float64 `json:"embedding"`
		}
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, nil, fmt.Errorf("%s: %w", path, err)
		}
		ids = append(ids, r.ID)
		vecs = append(vecs, r.Embedding)
	}
	return ids, vecs, sc.Err()
}

func loadDomains(path string) (map[int64]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[int64]string{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r struct {
			ID     int64  `json:"id"`
			Domain string `json:"domain"`
		}
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		out[r.ID] = r.Domain
	}
	return out, sc.Err()
}

// ---- 幾何 ----

func dist(a, b []float64) float64 {
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return 1 - dot // 向量已正規化(manifest normalized=true)
}

func pairStats(vecs [][]float64, n int, rng *rand.Rand) (median, stdMean float64) {
	ds := make([]float64, 0, n)
	for len(ds) < n {
		i, j := rng.Intn(len(vecs)), rng.Intn(len(vecs))
		if i == j {
			continue
		}
		ds = append(ds, dist(vecs[i], vecs[j]))
	}
	sort.Float64s(ds)
	median = ds[len(ds)/2]
	var sum, sq float64
	for _, d := range ds {
		sum += d
	}
	mean := sum / float64(len(ds))
	for _, d := range ds {
		sq += (d - mean) * (d - mean)
	}
	std := math.Sqrt(sq / float64(len(ds)-1))
	return median, std / mean
}

// nearestNeighborDist 全對暴力,行分片平行。
func nearestNeighborDist(vecs [][]float64) []float64 {
	n := len(vecs)
	nn := make([]float64, n)
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	chunk := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, min((w+1)*chunk, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				best := math.Inf(1)
				for j := 0; j < n; j++ {
					if i == j {
						continue
					}
					if d := dist(vecs[i], vecs[j]); d < best {
						best = d
					}
				}
				nn[i] = best
			}
		}(lo, hi)
	}
	wg.Wait()
	return nn
}

func intraCrossRatio(vecs [][]float64, ids []int64, domains map[int64]string, n int, rng *rand.Rand) float64 {
	byDomain := map[string][]int{}
	for idx, id := range ids {
		byDomain[domains[id]] = append(byDomain[domains[id]], idx)
	}
	var intraSum float64
	intraCount := 0
	for _, idxs := range byDomain {
		if len(idxs) < 2 {
			continue
		}
		per := n / len(byDomain)
		for k := 0; k < per; k++ {
			i, j := idxs[rng.Intn(len(idxs))], idxs[rng.Intn(len(idxs))]
			if i == j {
				continue
			}
			intraSum += dist(vecs[i], vecs[j])
			intraCount++
		}
	}
	var crossSum float64
	crossCount := 0
	for crossCount < n {
		i, j := rng.Intn(len(vecs)), rng.Intn(len(vecs))
		if i == j || domains[ids[i]] == domains[ids[j]] {
			continue
		}
		crossSum += dist(vecs[i], vecs[j])
		crossCount++
	}
	return (intraSum / float64(intraCount)) / (crossSum / float64(crossCount))
}

func perDomainDupRate(vecs [][]float64, ids []int64, domains map[int64]string, thresh float64) map[string]float64 {
	byDomain := map[string][]int{}
	for idx, id := range ids {
		byDomain[domains[id]] = append(byDomain[domains[id]], idx)
	}
	out := map[string]float64{}
	for d, idxs := range byDomain {
		dup := 0
		for _, i := range idxs {
			best := math.Inf(1)
			for _, j := range idxs {
				if i == j {
					continue
				}
				if dd := dist(vecs[i], vecs[j]); dd < best {
					best = dd
				}
			}
			if best < thresh {
				dup++
			}
		}
		out[d] = float64(dup) / float64(len(idxs))
	}
	return out
}

// ---- 檢查 6:end-to-end ----

func checkEndToEnd(ids []int64, vecs [][]float64, domains map[int64]string, rng *rand.Rand) (checkResult, error) {
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = "postgres://bench:bench@localhost:5433/bench"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	dims := len(vecs[0])
	d, err := db.Connect(ctx, url, dims)
	if err != nil {
		return checkResult{}, err
	}
	defer d.Close(ctx)

	// domain → 序數作 label_clustered 佔位;label_scattered = 0。
	// 真正的檔位標籤指派屬 internal/filters(順序 4),檢查 6 不用 filter。
	domOrd := map[string]int16{}
	var domSorted []string
	for _, dm := range domains {
		if _, ok := domOrd[dm]; !ok {
			domOrd[dm] = 0
			domSorted = append(domSorted, dm)
		}
	}
	sort.Strings(domSorted)
	for i, dm := range domSorted {
		domOrd[dm] = int16(i)
	}
	recs := make([]corpus.Record, len(ids))
	for i, id := range ids {
		f32 := make([]float32, dims)
		for j, x := range vecs[i] {
			f32[j] = float32(x)
		}
		recs[i] = corpus.Record{ID: id, Embedding: f32,
			LabelClustered: domOrd[domains[id]], LabelScattered: 0}
	}

	if err := d.EnsureSchema(ctx); err != nil {
		return checkResult{}, err
	}
	if err := d.LoadCorpus(ctx, recs); err != nil {
		return checkResult{}, err
	}
	if _, err := d.BuildHNSW(ctx, db.IndexOpts{M: 16, EfConstruction: 64}); err != nil {
		return checkResult{}, err
	}
	if err := d.BuildMetaIndexes(ctx); err != nil {
		return checkResult{}, err
	}
	if err := d.Analyze(ctx); err != nil {
		return checkResult{}, err
	}

	cfg := config.SweepConfig{EfSearch: 200, IterativeScan: "off", PlanMode: "auto", K: 50, Seed: 42}
	if err := d.ApplySession(ctx, cfg); err != nil {
		return checkResult{}, err
	}

	// 50 條 seeded query(取語料向量);先斷言 plan = hnsw_index(#11)
	qIdx := rng.Perm(len(vecs))[:50]
	probe := query.Query{ID: ids[qIdx[0]], Vector: recs[qIdx[0]].Embedding}
	er, err := d.ExplainANN(ctx, probe, cfg)
	if err != nil {
		return checkResult{}, err
	}
	if er.PlanType != "hnsw_index" {
		return checkResult{Check: 6, Name: "end-to-end recall",
			Value: er.PlanType, Criterion: "plan must be hnsw_index",
			Pass: false, Note: "recall 未量測:plan 不是 ANN 路徑,量了也是假的"}, nil
	}

	var recallSum float64
	for _, qi := range qIdx {
		gt := exactTopK(vecs, vecs[qi], 50)
		got, _, err := d.SearchANN(ctx, query.Query{ID: ids[qi], Vector: recs[qi].Embedding}, cfg)
		if err != nil {
			return checkResult{}, err
		}
		recallSum += overlap(gt, got, ids)
	}
	mean := recallSum / float64(len(qIdx))
	return checkResult{Check: 6, Name: "end-to-end recall (no-filter, ef=200, k=50)",
		Value: round4(mean), Criterion: ">= 0.99, plan asserted hnsw_index",
		Pass: mean >= 0.99}, nil
}

func exactTopK(vecs [][]float64, q []float64, k int) []int {
	type pair struct {
		idx int
		d   float64
	}
	ps := make([]pair, len(vecs))
	for i, v := range vecs {
		ps[i] = pair{i, dist(v, q)}
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].d < ps[b].d })
	out := make([]int, k)
	for i := range out {
		out[i] = ps[i].idx
	}
	return out
}

func overlap(gtIdx []int, got []int64, ids []int64) float64 {
	gtSet := map[int64]bool{}
	for _, i := range gtIdx {
		gtSet[ids[i]] = true
	}
	hit := 0
	for _, id := range got {
		if gtSet[id] {
			hit++
		}
	}
	return float64(hit) / float64(len(gtIdx))
}

// ---- 檢查 7:GLS 建構驗證 ----

// glsSideStats 依 #15 分檔位準則組織:聚集型分富集/跨域側,散開型 per seed。
type glsTierStats struct {
	Tier             string             `json:"tier"`
	SigmaG           float64            `json:"sigma_g"`
	ClusteredEnrich  filters.GLSStats   `json:"clustered_enriched"`
	ClusteredCross   filters.GLSStats   `json:"clustered_cross"`
	ScatteredBySeed  []filters.GLSStats `json:"scattered_by_seed"`
	Gate             string             `json:"gate"` // "hard" | "observe" | "record"
	Pass             bool               `json:"pass"`
}

var scatterSeeds = []int64{1, 2, 3}

// checkGLS:σ_l = 精確(無 filter)top-k 中通過 filter 的比例,由 GT 免費得出;
// σ_g = 全域選擇性;pooled r = σ_l/σ_g 的合併估計(A.3、#15)。
// 距離計算以 float32 量化(#21 Q2,與 DB 儲存精度一致)。
func checkGLS(queriesPath, queryVecPath, filtersConfig string,
	corpusIDs []int64, corpusVecs [][]float64, domains map[int64]string) (checkResult, error) {

	spec, err := filters.LoadSpec(filtersConfig)
	if err != nil {
		return checkResult{}, err
	}
	clustered, err := spec.AssignClustered(domains)
	if err != nil {
		return checkResult{}, err
	}
	counts := filters.CountByLabel(clustered)
	scattered := make([]map[int64]int16, len(scatterSeeds))
	for i, s := range scatterSeeds {
		scattered[i] = filters.AssignScattered(corpusIDs, counts, s)
	}

	// query 載入:stratum 決定聚集型的富集/跨域側
	type qrec struct {
		ID      int64  `json:"id"`
		Stratum string `json:"stratum"`
	}
	var qmeta []qrec
	f, err := os.Open(queriesPath)
	if err != nil {
		return checkResult{}, err
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var q qrec
		if err := json.Unmarshal(sc.Bytes(), &q); err != nil {
			f.Close()
			return checkResult{}, err
		}
		qmeta = append(qmeta, q)
	}
	f.Close()
	qids, qvecs, err := loadVectors(queryVecPath)
	if err != nil {
		return checkResult{}, err
	}
	if len(qids) != len(qmeta) {
		return checkResult{}, fmt.Errorf("check7: %d query vectors vs %d query records", len(qids), len(qmeta))
	}
	stratumOf := map[int64]string{}
	for _, q := range qmeta {
		stratumOf[q.ID] = q.Stratum
	}

	// float32 量化後算每條 query 的精確 top-k(無 filter)
	k := 50
	cv32 := quantize(corpusVecs)
	qv32 := quantize(qvecs)
	topk := make([][]int, len(qv32)) // corpus index lists
	for i := range qv32 {
		topk[i] = exactTopK32(cv32, qv32[i], k)
	}

	n := float64(len(corpusIDs))
	var tiers []glsTierStats
	allPass := true
	for _, tier := range []string{"p07", "p3", "p6", "p15", "p50"} {
		ord := filters.TierOrdinal[tier]
		sigmaG := float64(counts[ord]) / n
		var enrich, cross []int
		scatterPass := make([][]int, len(scatterSeeds))
		for qi, qid := range qids {
			cCount := 0
			sCount := make([]int, len(scatterSeeds))
			for _, ci := range topk[qi] {
				id := corpusIDs[ci]
				if clustered[id] == ord {
					cCount++
				}
				for si := range scatterSeeds {
					if scattered[si][id] == ord {
						sCount[si]++
					}
				}
			}
			if stratumOf[qid] == tier {
				enrich = append(enrich, cCount)
			} else {
				cross = append(cross, cCount)
			}
			for si := range scatterSeeds {
				scatterPass[si] = append(scatterPass[si], sCount[si])
			}
		}
		ts := glsTierStats{
			Tier: tier, SigmaG: round4(sigmaG),
			ClusteredEnrich: filters.PooledGLS(enrich, k, sigmaG),
			ClusteredCross:  filters.PooledGLS(cross, k, sigmaG),
		}
		for si := range scatterSeeds {
			ts.ScatteredBySeed = append(ts.ScatteredBySeed, filters.PooledGLS(scatterPass[si], k, sigmaG))
		}
		// 準則(#15):聚集型 p07/p3/p6 硬閘(富集側 pooled r ≥ 2 = 顯著偏離);
		// p15 觀察、p50 記錄(域聯集稀釋為量測對象,不設閘)。
		// 散開型全檔位 r ∈ [0.85, 1.15] 硬閘——偏離即指派程式 bug 訊號。
		switch tier {
		case "p07", "p3", "p6":
			ts.Gate = "hard"
			ts.Pass = ts.ClusteredEnrich.PooledR >= 2.0
		case "p15":
			ts.Gate = "observe"
			ts.Pass = true
		case "p50":
			ts.Gate = "record"
			ts.Pass = true
		}
		for _, ss := range ts.ScatteredBySeed {
			if ss.PooledR < 0.85 || ss.PooledR > 1.15 {
				ts.Pass = false
				ts.Gate += "+scattered_FAIL"
			}
		}
		allPass = allPass && ts.Pass
		tiers = append(tiers, ts)
	}
	return checkResult{Check: 7, Name: "GLS construction validation",
		Value:     tiers,
		Criterion: "clustered enriched pooled r >= 2 (p07/p3/p6 hard; p15 observe; p50 record); scattered all tiers r in [0.85, 1.15]",
		Pass:      allPass,
		Note:      "σ_l 出自 float32 量化的無 filter 精確 top-50;scattered 偏離 = 指派程式 bug 訊號"}, nil
}

func quantize(vecs [][]float64) [][]float32 {
	out := make([][]float32, len(vecs))
	for i, v := range vecs {
		f := make([]float32, len(v))
		for j, x := range v {
			f[j] = float32(x)
		}
		out[i] = f
	}
	return out
}

func exactTopK32(vecs [][]float32, q []float32, k int) []int {
	type pair struct {
		idx int
		d   float64
	}
	ps := make([]pair, len(vecs))
	for i, v := range vecs {
		var dot float64
		for j := range q {
			dot += float64(v[j]) * float64(q[j])
		}
		ps[i] = pair{i, 1 - dot}
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].d < ps[b].d })
	out := make([]int, k)
	for i := range out {
		out[i] = ps[i].idx
	}
	return out
}

func round4(x float64) float64 { return math.Round(x*1e4) / 1e4 }

func roundMap(m map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(m))
	for k, v := range m {
		out[k] = round4(v)
	}
	return out
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "validate:", err)
		os.Exit(1)
	}
}
