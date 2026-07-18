// Package corpus 負責 JSONL 讀寫與語料 metadata(順序 3)。順序 2 只定義
// db 灌庫所需的最小 Record。
package corpus

// Record 是灌庫的最小單位。標籤欄位依散布機制命名(DECISIONS.md #5):
// LabelClustered 來自領域標籤,LabelScattered 來自固定 seed 隨機指派。
type Record struct {
	ID             int64
	Embedding      []float32
	LabelClustered int16
	LabelScattered int16
}
