package db

import (
	"context"
	"fmt"
	"time"

	"github.com/pgvector/pgvector-go"

	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/config"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/query"
)

// filterColumns 是 Filter.Column 白名單:欄名進 SQL 前必經此表。
var filterColumns = map[string]bool{
	"label_clustered":    true,
	"label_scattered_s1": true,
	"label_scattered_s2": true,
	"label_scattered_s3": true,
}

// buildSearchSQL 組出 ANN 查詢。GT 的精確版本由 force_exact session 決定
// plan,SQL 本身同一條(A.0 原則 3:GT 與 ANN 用同一份 query)。
func buildSearchSQL(q query.Query, k int) (sql string, args []any, err error) {
	vec := pgvector.NewVector(q.Vector)
	if q.Filter == nil {
		return fmt.Sprintf("SELECT id FROM %s ORDER BY embedding <=> $1 LIMIT %d", TableName, k),
			[]any{vec}, nil
	}
	if !filterColumns[q.Filter.Column] {
		return "", nil, fmt.Errorf("db: filter column %q not allowed", q.Filter.Column)
	}
	return fmt.Sprintf("SELECT id FROM %s WHERE %s = $2 ORDER BY embedding <=> $1 LIMIT %d",
		TableName, q.Filter.Column, k), []any{vec, q.Filter.Value}, nil
}

// SearchANN 執行查詢並回傳 id 與 client round-trip 延遲(DECISIONS.md #7)。
// 回傳筆數可能 < k(iterative off 且 ef < k 時的截斷即量測對象),不在此擋。
func (d *DB) SearchANN(ctx context.Context, q query.Query, cfg config.SweepConfig) ([]int64, time.Duration, error) {
	sql, args, err := buildSearchSQL(q, cfg.K)
	if err != nil {
		return nil, 0, err
	}
	start := time.Now()
	rows, err := d.conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, 0, fmt.Errorf("db: search: %w", err)
	}
	ids := make([]int64, 0, cfg.K)
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return nil, 0, fmt.Errorf("db: scan: %w", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("db: rows: %w", err)
	}
	return ids, time.Since(start), nil
}
