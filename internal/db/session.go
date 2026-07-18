package db

import (
	"context"
	"fmt"

	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/config"
)

// ApplySession 套用單一配置的 session 設定。順序固定:
//
//  1. RESET ALL — 防上一配置的 GUC 洩漏(A.6 坑 4);
//  2. plan_cache_mode = force_custom_plan — 防 pgx prepared statement 的
//     generic plan 快取無視本次 GUC(A.6 坑表 2026-07-18 新增列);
//  3. ef / iterative_scan / plan 強制。
//
// force_exact = enable_indexscan off:HNSW 只能經 Index Scan node 進 plan,
// 關掉即保證精確;btree 仍可走 bitmap(DECISIONS.md #1)。
// 「HNSW 無 bitmap 路徑」為 pgvector 0.8.1 實作層主張,升級版本時重驗;
// ExplainANN 的 force 生效驗證是每次量測的運行時保險。
// GUC 值不吃 bind 參數,一律經 Validate 白名單後才進 Sprintf。
func (d *DB) ApplySession(ctx context.Context, cfg config.SweepConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	stmts := []string{
		"RESET ALL",
		"SET plan_cache_mode = force_custom_plan",
		fmt.Sprintf("SET hnsw.ef_search = %d", cfg.EfSearch),
		fmt.Sprintf("SET hnsw.iterative_scan = %s", cfg.IterativeScan),
	}
	if cfg.PlanMode == "force_exact" {
		stmts = append(stmts, "SET enable_indexscan = off")
	}
	for _, s := range stmts {
		if _, err := d.conn.Exec(ctx, s); err != nil {
			return fmt.Errorf("db: apply session %q: %w", s, err)
		}
	}
	return nil
}

// ShowGUC 回讀 session GUC 現值,供洩漏測試與紀錄(冒煙驗收 6)。
func (d *DB) ShowGUC(ctx context.Context, name string) (string, error) {
	var v string
	if err := d.conn.QueryRow(ctx, "SELECT current_setting($1)", name).Scan(&v); err != nil {
		return "", fmt.Errorf("db: show %s: %w", name, err)
	}
	return v, nil
}
