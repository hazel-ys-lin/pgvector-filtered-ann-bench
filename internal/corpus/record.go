// Package corpus 負責 JSONL 讀寫與語料 metadata(順序 3)。順序 2 只定義
// db 灌庫所需的最小 Record。
package corpus

// Record 是灌庫的最小單位。標籤欄位依散布機制命名(DECISIONS.md #5):
// LabelClustered 來自領域標籤(檔位群組 id);LabelScattered 為 3 個
// filter-seed 的隨機指派,一次灌庫避免 UPDATE 造成索引膨脹(#24)。
type Record struct {
	ID             int64
	Embedding      []float32
	LabelClustered int16
	LabelScattered [3]int16 // seeds 1/2/3(A.5:filter 指派 3 seed)
}
