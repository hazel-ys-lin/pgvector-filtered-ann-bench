// Package query 負責 query 集、exact ground truth 與 ANN 查詢執行(順序 4)。
// 順序 2 只定義 db 執行查詢所需的最小型別。
package query

// Filter 指定 metadata 過濾條件。Column 只能是 db 層機制命名的標籤欄位
// ("label_clustered" / "label_scattered"),由 db 層白名單校驗。
type Filter struct {
	Column string
	Value  int16
}

// Query 是單一查詢:向量 + 可選 filter。GT 與 ANN 共用同一份(A.0 原則 3)。
type Query struct {
	ID     int64
	Vector []float32
	Filter *Filter // nil = no-filter
}
