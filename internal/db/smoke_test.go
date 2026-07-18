//go:build smoke

// 順序 2 冒煙(make smoke):對真 Postgres 驗證 db 模組全鏈路。
// 驗收對照:版本 assert(2)、灌庫/建索引/ANALYZE(3)、no-filter recall(4)、
// 三種 plan 誘發與解析(5)、GUC 洩漏(6)、force 生效驗證(7)。
// 冒煙用固定 seed 隨機向量,非語料——語料與幾何守門屬順序 3。
package db

import (
	"context"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/config"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/corpus"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/query"
)

// smokeN 注意:2k 列時 planner 連 no-filter 都(正確地)選 seq_scan——
// 掃小表比走 HNSW 便宜,冒煙階段即observed的 planner 邊界。取 10k 讓
// index 路徑划算,冒煙才能驗證 hnsw_index 的解析。
const (
	smokeDims = 64
	smokeN    = 10000
	smokeK    = 50
	smokeSeed = 42
	// label_scattered ∈ [0,20) 均勻 → 單值選擇性 ~5%,給 bitmap plan 誘發用
	smokeScatteredCard = 20
)

func smokeURL(t *testing.T) string {
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		t.Skip("DATABASE_URL not set; run via `make smoke`")
	}
	return url
}

func genRecords(rng *rand.Rand) []corpus.Record {
	recs := make([]corpus.Record, smokeN)
	for i := range recs {
		v := make([]float32, smokeDims)
		var norm float64
		for j := range v {
			x := rng.NormFloat64()
			v[j] = float32(x)
			norm += x * x
		}
		norm = math.Sqrt(norm)
		for j := range v {
			v[j] /= float32(norm)
		}
		recs[i] = corpus.Record{
			ID:             int64(i + 1),
			Embedding:      v,
			LabelClustered: int16(i * 10 / smokeN),
			LabelScattered: int16(rng.Intn(smokeScatteredCard)),
		}
	}
	return recs
}

// exactTopK 在 Go 端算精確 top-k(cosine 距離),獨立於 DB 作裁判。
func exactTopK(recs []corpus.Record, qv []float32, k int) []int64 {
	type pair struct {
		id   int64
		dist float64
	}
	ps := make([]pair, len(recs))
	for i, r := range recs {
		var dot float64
		for j := range qv {
			dot += float64(r.Embedding[j]) * float64(qv[j])
		}
		ps[i] = pair{r.ID, 1 - dot} // 向量已正規化,cosine 距離 = 1 - dot
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].dist < ps[b].dist })
	ids := make([]int64, k)
	for i := range ids {
		ids[i] = ps[i].id
	}
	return ids
}

func recallAt(gt, got []int64) float64 {
	set := make(map[int64]bool, len(gt))
	for _, id := range gt {
		set[id] = true
	}
	hit := 0
	for _, id := range got {
		if set[id] {
			hit++
		}
	}
	return float64(hit) / float64(len(gt))
}

func TestSmoke(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// 驗收 2:Connect 內含 extversion assert,連得上即版本正確
	d, err := Connect(ctx, smokeURL(t), smokeDims)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer d.Close(ctx)

	rng := rand.New(rand.NewSource(smokeSeed))
	recs := genRecords(rng)

	// 驗收 3:schema → 灌庫 → HNSW(內驗 maintenance_work_mem)→ meta 索引 → ANALYZE
	if err := d.EnsureSchema(ctx); err != nil {
		t.Fatalf("schema: %v", err)
	}
	if err := d.LoadCorpus(ctx, recs); err != nil {
		t.Fatalf("load: %v", err)
	}
	stats, err := d.BuildHNSW(ctx, IndexOpts{M: 16, EfConstruction: 64})
	if err != nil {
		t.Fatalf("build hnsw: %v", err)
	}
	t.Logf("hnsw built in %v, maintenance_work_mem=%s", stats.Duration, stats.MaintenanceWorkMem)
	if err := d.BuildMetaIndexes(ctx); err != nil {
		t.Fatalf("meta indexes: %v", err)
	}
	if err := d.Analyze(ctx); err != nil {
		t.Fatalf("analyze: %v", err)
	}

	qv := recs[7].Embedding // 用庫內向量當 query,GT 必含自身,穩定
	q := query.Query{ID: 1, Vector: qv}
	// ef=200:實測此規模下 planner 選 HNSW;ef=400 時 cost model(隨 ef
	// 線性放大)會讓 seq+sort 勝出——見下方邊界觀測區塊
	auto := config.SweepConfig{
		EfSearch: 200, IterativeScan: "off", PlanMode: "auto", K: smokeK, Seed: smokeSeed,
	}

	// 驗收 4:no-filter recall vs Go 端精確裁判
	if err := d.ApplySession(ctx, auto); err != nil {
		t.Fatalf("apply auto: %v", err)
	}
	gt := exactTopK(recs, qv, smokeK)
	got, lat, err := d.SearchANN(ctx, q, auto)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(got) != smokeK {
		t.Fatalf("got %d ids, want %d", len(got), smokeK)
	}
	if lat <= 0 {
		t.Errorf("latency not measured: %v", lat)
	}
	// 門檻 0.95 非 B.3 檢查 6 的 0.99:隨機高斯向量正是 B.3 所述病態幾何
	// (距離集中、無叢集),對 HNSW 是最壞情況;0.99 門檻屬順序 3 真語料。
	// 實測本配置 recall = 0.98。
	if r := recallAt(gt, got); r < 0.95 {
		t.Errorf("no-filter recall@%d = %.3f, want >= 0.95", smokeK, r)
	} else {
		t.Logf("no-filter recall@%d = %.3f, round-trip = %v", smokeK, r, lat)
	}

	// 驗收 5a:auto no-filter → hnsw_index
	er, err := d.ExplainANN(ctx, q, auto)
	if err != nil {
		t.Fatalf("explain auto: %v", err)
	}
	if er.PlanType != "hnsw_index" {
		t.Errorf("auto no-filter plan = %q, want hnsw_index; raw:\n%s", er.PlanType, er.Raw)
	}
	if er.ExecutionMS <= 0 {
		t.Errorf("ExecutionMS not parsed")
	}

	// 邊界觀測(不設斷言,planner 的自由選擇本身是研究對象):pgvector
	// cost 隨 ef_search 放大,10k 列時 ef=40 → hnsw、ef=400 → seq+sort。
	// 此即 A.5b 處置 (d) 的方法論注意點:加深 ef 可能把 planner 推離 index。
	for _, ef := range []int{40, 400} {
		cfgEf := auto
		cfgEf.EfSearch = ef
		if err := d.ApplySession(ctx, cfgEf); err != nil {
			t.Fatalf("apply ef=%d: %v", ef, err)
		}
		bres, err := d.ExplainANN(ctx, q, cfgEf)
		if err != nil {
			t.Fatalf("explain ef=%d: %v", ef, err)
		}
		t.Logf("boundary observation: auto no-filter ef=%d → %s", ef, bres.PlanType)
	}

	// 驗收 5b + 7:force_exact no-filter → seq_scan(ExplainANN 內建 force 生效驗證,不匹配即 error)
	forced := auto
	forced.PlanMode = "force_exact"
	if err := d.ApplySession(ctx, forced); err != nil {
		t.Fatalf("apply force_exact: %v", err)
	}
	er, err = d.ExplainANN(ctx, q, forced)
	if err != nil {
		t.Fatalf("explain force_exact: %v", err)
	}
	if er.PlanType != "seq_scan" {
		t.Errorf("force_exact no-filter plan = %q, want seq_scan; raw:\n%s", er.PlanType, er.Raw)
	}

	// 驗收 5c:force_exact + ~5% filter,期望 bitmap_meta;小數據誘發不出
	// 則記錄待順序 4 驗證,不硬湊(seq_scan 亦屬 force_exact 合法精確路徑)
	qf := q
	qf.Filter = &query.Filter{Column: "label_scattered", Value: 3}
	er, err = d.ExplainANN(ctx, qf, forced)
	if err != nil {
		t.Fatalf("explain force_exact filtered: %v", err)
	}
	switch er.PlanType {
	case "bitmap_meta":
		t.Logf("bitmap_meta induced as expected")
	case "seq_scan":
		t.Logf("NOTE: bitmap_meta not induced on smoke-size data (got seq_scan); recheck at 順序 4 scale")
	default:
		t.Errorf("force_exact filtered plan = %q, want bitmap_meta or seq_scan; raw:\n%s", er.PlanType, er.Raw)
	}

	// 驗收 6:GUC 洩漏——force_exact 後 enable_indexscan=off,切回 auto 必須回 on
	if v, _ := d.ShowGUC(ctx, "enable_indexscan"); v != "off" {
		t.Errorf("enable_indexscan = %q under force_exact, want off", v)
	}
	if err := d.ApplySession(ctx, auto); err != nil {
		t.Fatalf("apply auto again: %v", err)
	}
	if v, _ := d.ShowGUC(ctx, "enable_indexscan"); v != "on" {
		t.Errorf("GUC leak: enable_indexscan = %q after switching back to auto, want on", v)
	}
	// plan_cache_mode 每配置必為 force_custom_plan(A.6 新增坑)
	if v, _ := d.ShowGUC(ctx, "plan_cache_mode"); v != "force_custom_plan" {
		t.Errorf("plan_cache_mode = %q, want force_custom_plan", v)
	}
}
