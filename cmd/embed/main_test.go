package main

import (
	"math"
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
