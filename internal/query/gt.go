// GT 快取(DESIGN A.2;DECISIONS #21 Q2、#25)。
//
// 裁判獨立於被測系統:exact top-k 在 Go 端以 float32 量化計算(與 DB 儲存
// 精度一致);快取 key 綁 (corpus sha, queries sha, k, ε, filter spec 版本),
// 語料或 query 集變動即失效重算。tie 邊界資訊(d_k、boundary_tie、tie_width)
// 一併入快取,供 #25 的 tie 欄與 tie 容差 recall。
package query

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// TieEps 為 #21 的 tie 容差與距離對齊測試共用 ε。
const TieEps = 1e-5

// FilterSpec 描述一個 filter 配置格的過濾條件。
type FilterSpec struct {
	Dispersion  string `json:"dispersion"`   // "clustered" | "scattered"
	ScatterSeed int    `json:"scatter_seed"` // clustered = 0(N/A 哨兵,#25)
	Tier        string `json:"tier"`
	Column      string `json:"column"` // db 欄名(白名單於 db 層再驗)
	Value       int16  `json:"value"`
}

// Key 為 filter 在快取內的識別字串。
func (f FilterSpec) Key() string {
	return fmt.Sprintf("%s/s%d/%s", f.Dispersion, f.ScatterSeed, f.Tier)
}

// GTEntry 是單一 (query, filter) 的精確裁判。
type GTEntry struct {
	IDs         []int64 `json:"ids"` // 依距離升冪
	DK          float64 `json:"d_k"` // 第 |IDs| 名距離(tie 容差基準)
	GTSize      int     `json:"gt_size"`
	BoundaryTie bool    `json:"boundary_tie"` // 邊界 ε 內存在可互換排除項
	TieWidth    int     `json:"tie_width"`    // 通過集內 |d - d_k| ≤ ε 筆數
}

type GTCache struct {
	SchemaVersion int                            `json:"schema_version"`
	Key           string                         `json:"key"`
	K             int                            `json:"k"`
	Entries       map[string]map[int64]GTEntry   `json:"entries"` // filterKey → qid → GT
}

// CacheKey 綁定所有影響 GT 的輸入(A.0 原則 2 的回溯錨點)。
func CacheKey(corpusSHA, queriesSHA string, k int, specVersion string) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("%s|%s|k=%d|eps=%g|f32|%s",
		corpusSHA, queriesSHA, k, TieEps, specVersion)))
	return hex.EncodeToString(h[:])[:16]
}

// BuildOrLoadGT:cache 檔存在且 key 相符即載入,否則全算後落盤。
func BuildOrLoadGT(path, key string, k int,
	corpusIDs []int64, corpusVecs [][]float32,
	queryIDs []int64, queryVecs [][]float32,
	specs []FilterSpec, labelOf func(FilterSpec, int64) int16) (*GTCache, bool, error) {

	if b, err := os.ReadFile(path); err == nil {
		var c GTCache
		if err := json.Unmarshal(b, &c); err == nil && c.Key == key && c.K == k {
			return &c, true, nil
		}
	}
	c := &GTCache{SchemaVersion: 1, Key: key, K: k,
		Entries: map[string]map[int64]GTEntry{}}
	for _, spec := range specs {
		pass := make([]int, 0, len(corpusIDs))
		for i, id := range corpusIDs {
			if labelOf(spec, id) == spec.Value {
				pass = append(pass, i)
			}
		}
		m := make(map[int64]GTEntry, len(queryIDs))
		for qi, qid := range queryIDs {
			m[qid] = exactGT(corpusIDs, corpusVecs, pass, queryVecs[qi], k)
		}
		c.Entries[spec.Key()] = m
	}
	b, err := json.Marshal(c)
	if err != nil {
		return nil, false, err
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return nil, false, fmt.Errorf("gt cache write: %w", err)
	}
	return c, false, nil
}

func exactGT(ids []int64, vecs [][]float32, passIdx []int, q []float32, k int) GTEntry {
	type pair struct {
		idx int
		d   float64
	}
	ps := make([]pair, len(passIdx))
	for i, ci := range passIdx {
		ps[i] = pair{ci, Dist32(vecs[ci], q)}
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].d < ps[b].d })
	n := min(k, len(ps))
	e := GTEntry{IDs: make([]int64, n), GTSize: n}
	for i := 0; i < n; i++ {
		e.IDs[i] = ids[ps[i].idx]
	}
	if n == 0 {
		return e
	}
	e.DK = ps[n-1].d
	for _, p := range ps {
		if p.d >= e.DK-TieEps && p.d <= e.DK+TieEps {
			e.TieWidth++
		}
	}
	// boundary tie:GT 滿 k 且存在被排除者與第 k 名 ε 內互換
	if n == k && len(ps) > k && ps[k].d <= e.DK+TieEps {
		e.BoundaryTie = true
	}
	return e
}

// Dist32:float32 量化的 cosine 距離(#21 Q2;向量已正規化)。
func Dist32(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return 1 - dot
}
