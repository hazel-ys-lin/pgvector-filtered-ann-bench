package filters

import (
	"os"
	"path/filepath"
	"testing"
)

func specFixture(t *testing.T) *Spec {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "d.toml")
	os.WriteFile(p, []byte(`
[tiers]
p07 = ["a"]
p3 = ["b"]
background = ["c", "d"]
`), 0o644)
	s, err := LoadSpec(p)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestAssignClustered(t *testing.T) {
	s := specFixture(t)
	got, err := s.AssignClustered(map[int64]string{1: "a", 2: "b", 3: "c", 4: "d"})
	if err != nil {
		t.Fatal(err)
	}
	want := map[int64]int16{1: 1, 2: 2, 3: 0, 4: 0}
	for id, l := range want {
		if got[id] != l {
			t.Errorf("id %d label = %d, want %d", id, got[id], l)
		}
	}
	if _, err := s.AssignClustered(map[int64]string{9: "nope"}); err == nil {
		t.Error("want error on unknown domain")
	}
}

func TestAssignScatteredExactCountsAndDeterminism(t *testing.T) {
	ids := make([]int64, 100)
	for i := range ids {
		ids[i] = int64(i + 1)
	}
	counts := map[int16]int{0: 70, 1: 10, 2: 20}
	a := AssignScattered(ids, counts, 7)
	b := AssignScattered(ids, counts, 7)
	c := AssignScattered(ids, counts, 8)
	if len(a) != 100 {
		t.Fatalf("assigned %d, want 100", len(a))
	}
	gotCounts := CountByLabel(a)
	for l, n := range counts {
		if gotCounts[l] != n {
			t.Errorf("label %d count = %d, want %d(選擇性必須精確一致)", l, gotCounts[l], n)
		}
	}
	same, diff := 0, 0
	for id := range a {
		if a[id] != b[id] {
			t.Fatalf("same seed must be deterministic")
		}
		if a[id] != c[id] {
			diff++
		} else {
			same++
		}
	}
	if diff == 0 {
		t.Error("different seeds gave identical assignment")
	}
}

func TestPooledGLS(t *testing.T) {
	// 100 queries × k=50, σ_g=0.1 → 期望 500;實得 500 → r=1
	pass := make([]int, 100)
	for i := range pass {
		pass[i] = 5
	}
	g := PooledGLS(pass, 50, 0.1)
	if g.PooledR != 1.0 {
		t.Errorf("pooled r = %v, want 1.0", g.PooledR)
	}
	if g.ExpectedCount != 500 {
		t.Errorf("expected = %v, want 500", g.ExpectedCount)
	}
	// 富集:實得倍增 → r=2
	for i := range pass {
		pass[i] = 10
	}
	if r := PooledGLS(pass, 50, 0.1).PooledR; r != 2.0 {
		t.Errorf("enriched r = %v, want 2.0", r)
	}
}
