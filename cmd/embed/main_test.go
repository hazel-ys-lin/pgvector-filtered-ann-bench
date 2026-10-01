package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNormalize(t *testing.T) {
	v := []float64{3, 4}
	pre, err := normalize(v)
	if err != nil {
		t.Fatal(err)
	}
	if pre != 5 {
		t.Errorf("pre-norm = %v, want 5", pre)
	}
	if math.Abs(v[0]-0.6) > 1e-12 || math.Abs(v[1]-0.8) > 1e-12 {
		t.Errorf("normalized = %v, want [0.6 0.8]", v)
	}
	if _, err := normalize([]float64{0, 0, 0}); err == nil {
		t.Error("want error on zero vector")
	}
}

func TestTaskTypeOf(t *testing.T) {
	for flagVal, want := range map[string]string{
		"document": "RETRIEVAL_DOCUMENT",
		"query":    "RETRIEVAL_QUERY",
	} {
		got, err := taskTypeOf(flagVal)
		if err != nil || got != want {
			t.Errorf("taskTypeOf(%q) = %q, %v; want %q", flagVal, got, err, want)
		}
	}
	if _, err := taskTypeOf("SEMANTIC_SIMILARITY"); err == nil {
		t.Error("want error on unpinned task type")
	}
}

func writeManifest(t *testing.T, path, sourceFile, sha string) {
	t.Helper()
	m := manifest{SourceFile: sourceFile, SourceSHA256: sha}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCheckResumeSafety(t *testing.T) {
	dir := t.TempDir()
	inA := filepath.Join(dir, "a.jsonl")
	inB := filepath.Join(dir, "b.jsonl")
	manifestPath := filepath.Join(dir, "out.manifest.json")
	os.WriteFile(inA, []byte("content A"), 0o644)
	os.WriteFile(inB, []byte("content B"), 0o644)
	shaA, err := fileSHA256(inA)
	if err != nil {
		t.Fatal(err)
	}

	// 無 manifest:全新輸出,放行
	if err := checkResumeSafety(manifestPath, inA); err != nil {
		t.Errorf("no manifest should be safe, got %v", err)
	}

	writeManifest(t, manifestPath, inA, shaA)

	// 同一份 --in:sha 相符,放行(正常續跑)
	if err := checkResumeSafety(manifestPath, inA); err != nil {
		t.Errorf("matching source should be safe, got %v", err)
	}

	// 換了 --in 內容:sha 不符,id 不保證對得上內容,必須拒絕
	// (2026-10-01 實測踩到:querygen 每次重抽 id 都從 1 起算,
	// 換語料後舊 manifest 的 sha 比對是唯一防線)
	if err := checkResumeSafety(manifestPath, inB); err == nil {
		t.Error("mismatched source sha must be rejected, got nil error")
	}

	// 壞掉/讀不出的 manifest:不卡死正常流程
	os.WriteFile(manifestPath, []byte("not json"), 0o644)
	if err := checkResumeSafety(manifestPath, inA); err != nil {
		t.Errorf("corrupt manifest should not block, got %v", err)
	}
}
