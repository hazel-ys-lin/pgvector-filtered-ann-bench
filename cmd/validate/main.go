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
		Value: round4(dupRate), Criterion: "1%–3%(沿用初始值,B.3 註記可校準)",
		Pass: dupRate >= 0.01 && dupRate <= 0.03,
		Note: "低於 1% 屬『好的一側』越界——ties 崩 recall 的風險僅在過高側"}
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
