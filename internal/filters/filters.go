// Package filters 實作選擇性 × 散布的標籤指派與 GLS 量測(DESIGN A.4;
// DECISIONS #15)。
//
// 聚集型:標籤 = 檔位群組 id(域 → 檔位映射來自 configs/domains_v1.toml,
// 同一 TOML 供 corpusgen 使用——單一事實來源)。背景域 = 0,不屬任何檔位。
// 散開型:固定 seed 隨機指派,各檔位筆數與聚集型完全一致(選擇性精確相同,
// 唯一變因是 filter 與向量叢集的相關性)。
package filters

import (
	"fmt"
	"math/rand"
	"sort"

	"github.com/BurntSushi/toml"
)

// 檔位 → 標籤值。0 保留給背景(不屬任何檔位)。
var TierOrdinal = map[string]int16{
	"background": 0,
	"p07":        1,
	"p3":         2,
	"p6":         3,
	"p15":        4,
	"p50":        5,
}

// OrdinalTier 反向映射(報表用)。
func OrdinalTier(o int16) string {
	for t, v := range TierOrdinal {
		if v == o {
			return t
		}
	}
	return fmt.Sprintf("unknown(%d)", o)
}

type Spec struct {
	Tiers  map[string][]string `toml:"tiers"`
	tierOf map[string]string
}

func LoadSpec(path string) (*Spec, error) {
	var s Spec
	if _, err := toml.DecodeFile(path, &s); err != nil {
		return nil, fmt.Errorf("filters: load %s: %w", path, err)
	}
	s.tierOf = map[string]string{}
	for tier, domains := range s.Tiers {
		if _, ok := TierOrdinal[tier]; !ok {
			return nil, fmt.Errorf("filters: unknown tier %q in %s", tier, path)
		}
		for _, d := range domains {
			if prev, dup := s.tierOf[d]; dup {
				return nil, fmt.Errorf("filters: domain %q in tiers %q and %q", d, prev, tier)
			}
			s.tierOf[d] = tier
		}
	}
	return &s, nil
}

// TierOf 回傳域所屬檔位;未知域回 error(語料與 config 脫鉤的訊號)。
func (s *Spec) TierOf(domain string) (string, error) {
	t, ok := s.tierOf[domain]
	if !ok {
		return "", fmt.Errorf("filters: domain %q not in spec", domain)
	}
	return t, nil
}

// AssignClustered:id → 檔位標籤(域決定,無隨機)。
func (s *Spec) AssignClustered(domains map[int64]string) (map[int64]int16, error) {
	out := make(map[int64]int16, len(domains))
	for id, d := range domains {
		t, err := s.TierOf(d)
		if err != nil {
			return nil, err
		}
		out[id] = TierOrdinal[t]
	}
	return out, nil
}

// AssignScattered:seed 隨機指派,各標籤筆數與 counts 完全一致。
// 決定性:id 排序 → seeded shuffle → 依序切塊。
func AssignScattered(ids []int64, counts map[int16]int, seed int64) map[int64]int16 {
	sorted := make([]int64, len(ids))
	copy(sorted, ids)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(sorted), func(i, j int) { sorted[i], sorted[j] = sorted[j], sorted[i] })

	// 標籤值排序後依序切塊(map 迭代序不可靠,排序保決定性)
	labels := make([]int16, 0, len(counts))
	for l := range counts {
		labels = append(labels, l)
	}
	sort.Slice(labels, func(i, j int) bool { return labels[i] < labels[j] })

	out := make(map[int64]int16, len(sorted))
	pos := 0
	for _, l := range labels {
		for i := 0; i < counts[l]; i++ {
			out[sorted[pos]] = l
			pos++
		}
	}
	return out
}

// CountByLabel 統計標籤分佈(散開型指派的 counts 輸入,亦供 σ_g 計算)。
func CountByLabel(assign map[int64]int16) map[int16]int {
	out := map[int16]int{}
	for _, l := range assign {
		out[l]++
	}
	return out
}

// GLSStats 是單一 (檔位, 散布, 分層側) 的 pooled GLS(DESIGN A.3)。
// Pooled r = Σσ_l / (n·k·σ_g):對低選擇性檔位,per-query σ_l 是離散小數,
// pooled 估計比 per-query 均值穩健;分母期望數入報表供讀者評估功率。
type GLSStats struct {
	NQueries      int     `json:"n_queries"`
	SigmaG        float64 `json:"sigma_g"`
	PassTotal     int     `json:"pass_total"`
	ExpectedCount float64 `json:"expected_count_if_r1"`
	PooledR       float64 `json:"pooled_r"`
	// SigmaL = PooledR × SigmaG = query 鄰域內的通過率(2026-07-19 author 指出,
	// DECISIONS #23)。r 的上限是 1/σ_g,各檔位滿分不同故 raw r 不可比;σ_l 單調
	// 可比,且直接是 post-filter 情境下 ANN 候選存活率的定量預言
	// (期望存活數 = ef × σ_l),由建構端算出、由 bench 量測端驗證。
	SigmaL float64 `json:"sigma_l"`
}

// PooledGLS:passCounts[i] = 第 i 條 query 的精確 top-k 中通過 filter 的筆數。
func PooledGLS(passCounts []int, k int, sigmaG float64) GLSStats {
	total := 0
	for _, c := range passCounts {
		total += c
	}
	expected := float64(len(passCounts)*k) * sigmaG
	r := 0.0
	if expected > 0 {
		r = float64(total) / expected
	}
	return GLSStats{
		NQueries:      len(passCounts),
		SigmaG:        sigmaG,
		PassTotal:     total,
		ExpectedCount: expected,
		PooledR:       r,
		SigmaL:        r * sigmaG,
	}
}
