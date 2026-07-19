package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"

	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/corpus"
)

// IndexOpts 為 HNSW 建構參數。v1 定案 m=16, ef_construction=64
// (DECISIONS.md #6),由呼叫端顯式傳入以利記錄於結果列。
type IndexOpts struct {
	M              int
	EfConstruction int
}

// BuildStats 記錄建構耗時與生效的 maintenance_work_mem(A.6 坑 2 的驗證憑據)。
type BuildStats struct {
	Duration           time.Duration
	MaintenanceWorkMem string
}

// EnsureSchema 重建 items 表。標籤欄位依散布機制命名(DECISIONS.md #5)。
func (d *DB) EnsureSchema(ctx context.Context) error {
	stmts := []string{
		fmt.Sprintf("DROP TABLE IF EXISTS %s", TableName),
		fmt.Sprintf(`CREATE TABLE %s (
			id bigint PRIMARY KEY,
			embedding vector(%d) NOT NULL,
			label_clustered smallint NOT NULL,
			label_scattered_s1 smallint NOT NULL,
			label_scattered_s2 smallint NOT NULL,
			label_scattered_s3 smallint NOT NULL
		)`, TableName, d.dims),
	}
	for _, s := range stmts {
		if _, err := d.conn.Exec(ctx, s); err != nil {
			return fmt.Errorf("db: schema: %w", err)
		}
	}
	return nil
}

// LoadCorpus 以 COPY 批次灌庫。
func (d *DB) LoadCorpus(ctx context.Context, records []corpus.Record) error {
	_, err := d.conn.CopyFrom(ctx,
		[]string{TableName},
		[]string{"id", "embedding", "label_clustered",
			"label_scattered_s1", "label_scattered_s2", "label_scattered_s3"},
		pgx.CopyFromSlice(len(records), func(i int) ([]any, error) {
			r := records[i]
			if len(r.Embedding) != d.dims {
				return nil, fmt.Errorf("db: record %d has %d dims, want %d", r.ID, len(r.Embedding), d.dims)
			}
			return []any{r.ID, pgvector.NewVector(r.Embedding), r.LabelClustered,
				r.LabelScattered[0], r.LabelScattered[1], r.LabelScattered[2]}, nil
		}),
	)
	if err != nil {
		return fmt.Errorf("db: load corpus: %w", err)
	}
	return nil
}

// BuildHNSW 建向量索引。建前 SET maintenance_work_mem = '2GB'(A.6 坑 2)
// 並以 SHOW 回讀驗證,生效值進 BuildStats。
func (d *DB) BuildHNSW(ctx context.Context, opts IndexOpts) (BuildStats, error) {
	if opts.M <= 0 || opts.EfConstruction <= 0 {
		return BuildStats{}, fmt.Errorf("db: invalid IndexOpts %+v", opts)
	}
	if _, err := d.conn.Exec(ctx, "SET maintenance_work_mem = '2GB'"); err != nil {
		return BuildStats{}, fmt.Errorf("db: set maintenance_work_mem: %w", err)
	}
	var mwm string
	if err := d.conn.QueryRow(ctx, "SHOW maintenance_work_mem").Scan(&mwm); err != nil {
		return BuildStats{}, fmt.Errorf("db: show maintenance_work_mem: %w", err)
	}
	if mwm != "2GB" {
		return BuildStats{}, fmt.Errorf("db: maintenance_work_mem = %s, want 2GB", mwm)
	}
	if _, err := d.conn.Exec(ctx, fmt.Sprintf("DROP INDEX IF EXISTS %s", HNSWIndexName)); err != nil {
		return BuildStats{}, fmt.Errorf("db: drop old hnsw index: %w", err)
	}
	start := time.Now()
	_, err := d.conn.Exec(ctx, fmt.Sprintf(
		"CREATE INDEX %s ON %s USING hnsw (embedding vector_cosine_ops) WITH (m = %d, ef_construction = %d)",
		HNSWIndexName, TableName, opts.M, opts.EfConstruction))
	if err != nil {
		return BuildStats{}, fmt.Errorf("db: build hnsw: %w", err)
	}
	return BuildStats{Duration: time.Since(start), MaintenanceWorkMem: mwm}, nil
}

// BuildMetaIndexes 建 filter 欄位 btree(A.4 注記:v1 預設建立,
// 讓 planner 邊界地圖涵蓋「有 attribute-index 可選」情境)。
func (d *DB) BuildMetaIndexes(ctx context.Context) error {
	stmts := []string{
		fmt.Sprintf("CREATE INDEX IF NOT EXISTS %s ON %s (label_clustered)", ClusteredIndexName, TableName),
	}
	for i, col := range ScatteredColumns {
		stmts = append(stmts, fmt.Sprintf("CREATE INDEX IF NOT EXISTS %s ON %s (%s)",
			ScatteredIndexNames[i], TableName, col))
	}
	for _, s := range stmts {
		if _, err := d.conn.Exec(ctx, s); err != nil {
			return fmt.Errorf("db: meta index: %w", err)
		}
	}
	return nil
}

// Analyze 更新 planner 統計(A.6 坑 5:灌庫與建索引後必跑)。
func (d *DB) Analyze(ctx context.Context) error {
	if _, err := d.conn.Exec(ctx, fmt.Sprintf("ANALYZE %s", TableName)); err != nil {
		return fmt.Errorf("db: analyze: %w", err)
	}
	return nil
}
