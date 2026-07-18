// Package config 定義實驗矩陣(TOML)與單一配置點。順序 2 只放 SweepConfig
// 與其驗證;TOML 展開屬順序 4。
package config

import "fmt"

// SweepConfig 是實驗矩陣展開後的單一配置點(DESIGN A.3)。
type SweepConfig struct {
	CorpusID      string
	Selectivity   string // "p07" | "p3" | "p6" | "p15" | "p50"
	Dispersion    string // "clustered" | "scattered"
	EfSearch      int    // 40 | 100 | 200 | 400
	IterativeScan string // "off" | "relaxed_order"
	PlanMode      string // "auto" | "force_exact"(DECISIONS.md #1)
	K             int    // 50
	Seed          int64
}

// Validate 檢查枚舉值。注意:EfSearch < K 且 iterative off 是合法配置
// (A.5b 處置 (a) 刻意如此,recall 上限受 ef 截斷即為量測對象),不在此擋。
func (c SweepConfig) Validate() error {
	switch c.IterativeScan {
	case "off", "relaxed_order":
	default:
		return fmt.Errorf("config: unknown IterativeScan %q", c.IterativeScan)
	}
	switch c.PlanMode {
	case "auto", "force_exact":
	default:
		return fmt.Errorf("config: unknown PlanMode %q", c.PlanMode)
	}
	if c.EfSearch <= 0 || c.K <= 0 {
		return fmt.Errorf("config: EfSearch/K must be positive, got %d/%d", c.EfSearch, c.K)
	}
	return nil
}
