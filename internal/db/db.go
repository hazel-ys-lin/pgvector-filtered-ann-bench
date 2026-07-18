// Package db 封裝 schema、索引、session 設定、plan 強制與 EXPLAIN 擷取解析
// (DESIGN A.3)。全部走 pgx,不解析 psql 文字輸出(A.0 原則 4)。
//
// 刻意用單一 *pgx.Conn 而非連線池:ApplySession 的 GUC 是 session 級,
// 連線池會讓「哪條連線帶著哪組 GUC」不可控(A.6 坑 4)。
package db

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// RequiredPgvectorVersion 與 docker-compose.yml 釘定的 image 版本雙保險
// (A.6 坑 3;DECISIONS.md #3)。
const RequiredPgvectorVersion = "0.8.1"

const (
	TableName          = "items"
	HNSWIndexName      = "items_embedding_hnsw"
	ClusteredIndexName = "items_label_clustered_btree"
	ScatteredIndexName = "items_label_scattered_btree"
)

// DB 持有單一 session 連線與 schema 參數(維度參數化,DECISIONS.md #5)。
type DB struct {
	conn *pgx.Conn
	dims int
}

// Connect 建立連線、確保 extension 存在並 assert 版本,失敗即 fail fast。
func Connect(ctx context.Context, url string, dims int) (*DB, error) {
	if dims <= 0 {
		return nil, fmt.Errorf("db: dims must be positive, got %d", dims)
	}
	conn, err := pgx.Connect(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("db: connect: %w", err)
	}
	if _, err := conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		conn.Close(ctx)
		return nil, fmt.Errorf("db: create extension: %w", err)
	}
	var ver string
	if err := conn.QueryRow(ctx,
		"SELECT extversion FROM pg_extension WHERE extname = 'vector'").Scan(&ver); err != nil {
		conn.Close(ctx)
		return nil, fmt.Errorf("db: read extversion: %w", err)
	}
	if ver != RequiredPgvectorVersion {
		conn.Close(ctx)
		return nil, fmt.Errorf("db: pgvector extversion = %s, require %s (image 與 assert 需同步改)",
			ver, RequiredPgvectorVersion)
	}
	if err := pgxvec.RegisterTypes(ctx, conn); err != nil {
		conn.Close(ctx)
		return nil, fmt.Errorf("db: register pgvector types: %w", err)
	}
	return &DB{conn: conn, dims: dims}, nil
}

func (d *DB) Close(ctx context.Context) error {
	return d.conn.Close(ctx)
}
