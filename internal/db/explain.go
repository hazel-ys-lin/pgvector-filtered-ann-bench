package db

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/config"
	"github.com/hazel-ys-lin/pgvector-filtered-ann-bench/internal/query"
)

// ExplainResult 是一級資料(A.0 原則 5)。ExecutionMS 來自 EXPLAIN ANALYZE,
// 與 SearchANN 的 round-trip 延遲分開存、不混用(DECISIONS.md #7)。
type ExplainResult struct {
	Raw         string  // EXPLAIN FORMAT JSON 全文,供事後補分類
	PlanType    string  // "hnsw_index" | "seq_scan" | "bitmap_meta" | "other"
	ExecutionMS float64
}

// planNode 對應 EXPLAIN FORMAT JSON 的 plan 樹節點(只取判定所需欄位)。
type planNode struct {
	NodeType     string     `json:"Node Type"`
	IndexName    string     `json:"Index Name"`
	RelationName string     `json:"Relation Name"`
	Plans        []planNode `json:"Plans"`
}

type explainRoot struct {
	Plan          planNode `json:"Plan"`
	ExecutionTime float64  `json:"Execution Time"`
}

// ExplainANN 對同一條查詢 SQL 跑 EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON),
// 解析 plan 類型並驗證 plan 強制是否生效——force 不生效即 error 非 warning
// (冒煙驗收 7),杜絕默默採信。
func (d *DB) ExplainANN(ctx context.Context, q query.Query, cfg config.SweepConfig) (ExplainResult, error) {
	sql, args, err := buildSearchSQL(q, cfg.K)
	if err != nil {
		return ExplainResult{}, err
	}
	var raw string
	if err := d.conn.QueryRow(ctx,
		"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "+sql, args...).Scan(&raw); err != nil {
		return ExplainResult{}, fmt.Errorf("db: explain: %w", err)
	}
	res, err := parseExplain(raw)
	if err != nil {
		return ExplainResult{}, err
	}
	if err := verifyPlanMode(cfg.PlanMode, res.PlanType); err != nil {
		return res, err
	}
	return res, nil
}

func parseExplain(raw string) (ExplainResult, error) {
	var roots []explainRoot
	if err := json.Unmarshal([]byte(raw), &roots); err != nil {
		return ExplainResult{}, fmt.Errorf("db: parse explain json: %w", err)
	}
	if len(roots) == 0 {
		return ExplainResult{}, fmt.Errorf("db: empty explain output")
	}
	return ExplainResult{
		Raw:         raw,
		PlanType:    classifyPlan(roots[0].Plan),
		ExecutionMS: roots[0].ExecutionTime,
	}, nil
}

// classifyPlan 依優先序判定(DECISIONS.md #2):走訪整棵樹,
// hnsw_index > bitmap_meta > seq_scan > other。
// 平行變體(Gather 底下的 Seq Scan / Parallel Bitmap Heap Scan)因走訪
// 全樹且以 Node Type + 索引名判定,自然落入對應類別。
func classifyPlan(root planNode) string {
	var hasHNSW, hasBitmapMeta, hasSeqScan bool
	var walk func(n planNode)
	walk = func(n planNode) {
		switch n.NodeType {
		case "Index Scan", "Index Only Scan":
			if n.IndexName == HNSWIndexName {
				hasHNSW = true
			}
		case "Bitmap Index Scan":
			if isMetaIndex(n.IndexName) {
				hasBitmapMeta = true
			}
		case "Seq Scan":
			if n.RelationName == TableName {
				hasSeqScan = true
			}
		}
		for _, c := range n.Plans {
			walk(c)
		}
	}
	walk(root)
	switch {
	case hasHNSW:
		return "hnsw_index"
	case hasBitmapMeta:
		return "bitmap_meta"
	case hasSeqScan:
		return "seq_scan"
	default:
		return "other"
	}
}

// verifyPlanMode:force_exact 的生效判準 = seq_scan 或 bitmap_meta
// (兩種精確路徑皆可,DECISIONS.md #1);auto 不設限。
func isMetaIndex(name string) bool {
	if name == ClusteredIndexName {
		return true
	}
	for _, n := range ScatteredIndexNames {
		if name == n {
			return true
		}
	}
	return false
}

func verifyPlanMode(planMode, planType string) error {
	if planMode != "force_exact" {
		return nil
	}
	if planType == "seq_scan" || planType == "bitmap_meta" {
		return nil
	}
	return fmt.Errorf("db: force_exact not effective: got plan type %q", planType)
}
