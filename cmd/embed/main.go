// embed — 併發 embedding loader(A.1;DECISIONS #9/#16/#17)。
//
// 文字 JSONL → gemini-embedding-001 768d → 向量 JSONL {id, embedding} + manifest。
// task_type 非對稱(#16):語料/COCO 參照 = RETRIEVAL_DOCUMENT,query = RETRIEVAL_QUERY。
// 768d 截斷輸出非單位範數:一律正規化並斷言(#16 附帶驗收)。
//
// 用法:
//	go run ./cmd/embed --in corpus/corpus_5k.jsonl --out corpus/vectors_5k.jsonl \
//	    --manifest corpus/vectors_5k.manifest.json --task-type document
//
// GEMINI_API_KEY 取自環境或 repo 根 .env。可續跑:--out 既有 id 跳過。
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	model    = "gemini-embedding-001"
	endpoint = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":batchEmbedContents"
)

type textRecord struct {
	ID   int64  `json:"id"`
	Text string `json:"text"`
}

type vecRecord struct {
	ID        int64     `json:"id"`
	Embedding []float64 `json:"embedding"`
}

type manifest struct {
	SchemaVersion        int     `json:"schema_version"`
	GeneratedAt          string  `json:"generated_at"`
	SourceFile           string  `json:"source_file"`
	SourceSHA256         string  `json:"source_sha256"`
	Records              int     `json:"records"`
	Model                string  `json:"model"`
	TaskType             string  `json:"task_type"`
	OutputDimensionality int     `json:"output_dimensionality"`
	Normalized           bool    `json:"normalized"`
	PreNormMin           float64 `json:"pre_norm_min"`
	PreNormMax           float64 `json:"pre_norm_max"`
	PreNormMean          float64 `json:"pre_norm_mean"`
	GitCommit            string  `json:"git_commit"`
}

// ---- 純函式(單元測試覆蓋)----

// normalize 就地正規化;回傳 pre-norm L2。零向量回 error(API 異常訊號)。
func normalize(v []float64) (float64, error) {
	var ss float64
	for _, x := range v {
		ss += x * x
	}
	n := math.Sqrt(ss)
	if n < 1e-9 {
		return 0, fmt.Errorf("zero-norm embedding")
	}
	for i := range v {
		v[i] /= n
	}
	// #16 驗收:正規化後斷言
	var check float64
	for _, x := range v {
		check += x * x
	}
	if math.Abs(math.Sqrt(check)-1) > 1e-6 {
		return 0, fmt.Errorf("post-normalize norm = %v, want 1", math.Sqrt(check))
	}
	return n, nil
}

func taskTypeOf(flagVal string) (string, error) {
	switch flagVal {
	case "document":
		return "RETRIEVAL_DOCUMENT", nil
	case "query":
		return "RETRIEVAL_QUERY", nil
	}
	return "", fmt.Errorf("unknown --task-type %q (document|query)", flagVal)
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ---- API ----

type batchReq struct {
	Requests []embedReq `json:"requests"`
}
type embedReq struct {
	Model                string  `json:"model"`
	Content              content `json:"content"`
	TaskType             string  `json:"taskType"`
	OutputDimensionality int     `json:"outputDimensionality"`
}
type content struct {
	Parts []part `json:"parts"`
}
type part struct {
	Text string `json:"text"`
}
type batchResp struct {
	Embeddings []struct {
		Values []float64 `json:"values"`
	} `json:"embeddings"`
}

func embedBatch(client *http.Client, key, taskType string, dims int, texts []string) ([][]float64, error) {
	reqs := make([]embedReq, len(texts))
	for i, t := range texts {
		reqs[i] = embedReq{
			Model:                "models/" + model,
			Content:              content{Parts: []part{{Text: t}}},
			TaskType:             taskType,
			OutputDimensionality: dims,
		}
	}
	body, _ := json.Marshal(batchReq{Requests: reqs})
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(10*(1<<attempt)) * time.Second)
		}
		req, _ := http.NewRequest("POST", endpoint, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-goog-api-key", key)
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		rb, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 200 {
			var br batchResp
			if err := json.Unmarshal(rb, &br); err != nil {
				return nil, fmt.Errorf("parse response: %w", err)
			}
			if len(br.Embeddings) != len(texts) {
				return nil, fmt.Errorf("got %d embeddings for %d texts", len(br.Embeddings), len(texts))
			}
			out := make([][]float64, len(texts))
			for i, e := range br.Embeddings {
				if len(e.Values) != dims {
					return nil, fmt.Errorf("embedding %d has %d dims, want %d", i, len(e.Values), dims)
				}
				out[i] = e.Values
			}
			return out, nil
		}
		msg := string(rb)
		// 不可重試型(corpusgen 同款分類):日配額 / prepay credits
		if resp.StatusCode == 429 && (strings.Contains(msg, "PerDay") ||
			strings.Contains(msg, "credits are depleted") ||
			strings.Contains(msg, "spending cap")) {
			return nil, fmt.Errorf("quota/credits exhausted (resume-safe): %.200s", msg)
		}
		lastErr = fmt.Errorf("HTTP %d: %.200s", resp.StatusCode, msg)
	}
	return nil, fmt.Errorf("giving up after retries: %w", lastErr)
}

// ---- 主流程 ----

func loadEnvKey() string {
	if k := os.Getenv("GEMINI_API_KEY"); k != "" {
		return k
	}
	for _, dir := range []string{".", "..", "../.."} {
		b, err := os.ReadFile(filepath.Join(dir, ".env"))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if k, v, ok := strings.Cut(line, "="); ok && strings.TrimSpace(k) == "GEMINI_API_KEY" {
				return strings.TrimSpace(v)
			}
		}
	}
	return ""
}

func gitCommit() string {
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func main() {
	in := flag.String("in", "", "輸入文字 JSONL(需 id, text 欄位)")
	out := flag.String("out", "", "輸出向量 JSONL")
	manifestPath := flag.String("manifest", "", "manifest 輸出路徑(預設 <out>.manifest.json)")
	taskFlag := flag.String("task-type", "", "document | query(#16 非對稱,必填)")
	dims := flag.Int("dims", 768, "output dimensionality")
	workers := flag.Int("workers", 4, "併發 worker 數(Tier 1 保守預設,實跑再調)")
	batchSize := flag.Int("batch", 20, "每請求筆數")
	limit := flag.Int("limit", 0, "只處理前 N 筆(0 = 全部;試跑用)")
	flag.Parse()

	if *in == "" || *out == "" || *taskFlag == "" {
		flag.Usage()
		os.Exit(2)
	}
	taskType, err := taskTypeOf(*taskFlag)
	if err != nil {
		fatal(err)
	}
	key := loadEnvKey()
	if key == "" {
		fatal(fmt.Errorf("GEMINI_API_KEY not set(環境或 .env)"))
	}
	if *manifestPath == "" {
		*manifestPath = *out + ".manifest.json"
	}

	records, err := readTexts(*in, *limit)
	if err != nil {
		fatal(err)
	}
	done, err := readDoneIDs(*out)
	if err != nil {
		fatal(err)
	}
	var todo []textRecord
	for _, r := range records {
		if !done[r.ID] {
			todo = append(todo, r)
		}
	}
	fmt.Printf("input %d records, %d already embedded, %d to go\n", len(records), len(done), len(todo))

	outF, err := os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		fatal(err)
	}
	defer outF.Close()

	type job struct{ recs []textRecord }
	jobs := make(chan job)
	var mu sync.Mutex // 保護 outF 與 norm 統計
	normMin, normMax, normSum := math.Inf(1), math.Inf(-1), 0.0
	written := 0
	client := &http.Client{Timeout: 120 * time.Second}
	var wg sync.WaitGroup
	errCh := make(chan error, *workers)

	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				texts := make([]string, len(j.recs))
				for i, r := range j.recs {
					texts[i] = r.Text
				}
				vecs, err := embedBatch(client, key, taskType, *dims, texts)
				if err != nil {
					errCh <- err
					return
				}
				mu.Lock()
				for i, v := range vecs {
					pre, err := normalize(v)
					if err != nil {
						mu.Unlock()
						errCh <- fmt.Errorf("record %d: %w", j.recs[i].ID, err)
						return
					}
					normMin = math.Min(normMin, pre)
					normMax = math.Max(normMax, pre)
					normSum += pre
					line, _ := json.Marshal(vecRecord{ID: j.recs[i].ID, Embedding: v})
					outF.Write(append(line, '\n'))
					written++
				}
				outF.Sync() // 逐批 flush,續跑安全
				fmt.Printf("progress: %d/%d\n", written, len(todo))
				mu.Unlock()
			}
		}()
	}

	go func() {
		for i := 0; i < len(todo); i += *batchSize {
			end := min(i+*batchSize, len(todo))
			jobs <- job{recs: todo[i:end]}
		}
		close(jobs)
	}()
	wg.Wait()
	close(errCh)
	if err := <-errCh; err != nil {
		fatal(err)
	}

	sha, err := fileSHA256(*in)
	if err != nil {
		fatal(err)
	}
	total := len(done) + written
	m := manifest{
		SchemaVersion:        1,
		GeneratedAt:          time.Now().UTC().Format(time.RFC3339),
		SourceFile:           *in,
		SourceSHA256:         sha,
		Records:              total,
		Model:                model,
		TaskType:             taskType,
		OutputDimensionality: *dims,
		Normalized:           true,
		PreNormMin:           normMin,
		PreNormMax:           normMax,
		PreNormMean:          normSum / math.Max(float64(written), 1),
		GitCommit:            gitCommit(),
	}
	mb, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(*manifestPath, mb, 0o644); err != nil {
		fatal(err)
	}
	fmt.Printf("done: %d vectors → %s, manifest → %s\n", total, *out, *manifestPath)
}

func readTexts(path string, limit int) ([]textRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []textRecord
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r textRecord
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, len(out)+1, err)
		}
		if r.Text == "" {
			return nil, fmt.Errorf("%s: record %d has empty text", path, r.ID)
		}
		out = append(out, r)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out, sc.Err()
}

func readDoneIDs(path string) (map[int64]bool, error) {
	done := map[int64]bool{}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return done, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r vecRecord
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("corrupt output %s(截斷?刪掉重跑或修最後一行): %w", path, err)
		}
		done[r.ID] = true
	}
	return done, sc.Err()
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "embed:", err)
	os.Exit(1)
}
