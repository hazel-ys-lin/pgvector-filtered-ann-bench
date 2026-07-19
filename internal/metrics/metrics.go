// Package metrics 實作 recall 與 tie 容差(DESIGN A.3;DECISIONS #21/#25)。
package metrics

import "fmt"

// RecallTie 依 #21 tie 容差計 recall:got 中的 id 若在 GT 集內,或其距離
// ≤ dK + eps(與第 k 名 tie 等價),計為 hit。分母 = gtSize(低選擇性
// filter 下 GT < k,#25 凍結說明)。內含 recall ≤ 1 斷言(A.0 原則 3)。
func RecallTie(gtIDs []int64, dK float64, gtSize int, got []int64,
	distOf func(int64) (float64, bool), eps float64) (float64, error) {

	if gtSize == 0 {
		return 0, fmt.Errorf("metrics: empty ground truth")
	}
	gtSet := make(map[int64]bool, len(gtIDs))
	for _, id := range gtIDs {
		gtSet[id] = true
	}
	hit := 0
	for _, id := range got {
		if gtSet[id] {
			hit++
			continue
		}
		if d, ok := distOf(id); ok && d <= dK+eps {
			hit++ // tie 等價類:與第 k 名距離差 < ε,視同正確
		}
	}
	r := float64(hit) / float64(gtSize)
	if r > 1+1e-9 {
		return 0, fmt.Errorf("metrics: recall %v > 1(hit %d / gt %d)——GT 或 tie 邏輯有 bug", r, hit, gtSize)
	}
	if r > 1 {
		r = 1
	}
	return r, nil
}
