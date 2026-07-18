package db

import "testing"

func TestParseExplainClassify(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "hnsw index scan",
			raw: `[{"Plan":{"Node Type":"Limit","Plans":[
				{"Node Type":"Index Scan","Index Name":"items_embedding_hnsw","Relation Name":"items"}
			]},"Execution Time":1.23}]`,
			want: "hnsw_index",
		},
		{
			name: "plain seq scan under sort",
			raw: `[{"Plan":{"Node Type":"Limit","Plans":[
				{"Node Type":"Sort","Plans":[
					{"Node Type":"Seq Scan","Relation Name":"items"}
				]}
			]},"Execution Time":4.5}]`,
			want: "seq_scan",
		},
		{
			name: "parallel seq scan under gather merge",
			raw: `[{"Plan":{"Node Type":"Limit","Plans":[
				{"Node Type":"Gather Merge","Plans":[
					{"Node Type":"Sort","Plans":[
						{"Node Type":"Seq Scan","Relation Name":"items","Parallel Aware":true}
					]}
				]}
			]},"Execution Time":2.0}]`,
			want: "seq_scan",
		},
		{
			name: "bitmap on meta btree",
			raw: `[{"Plan":{"Node Type":"Limit","Plans":[
				{"Node Type":"Sort","Plans":[
					{"Node Type":"Bitmap Heap Scan","Relation Name":"items","Plans":[
						{"Node Type":"Bitmap Index Scan","Index Name":"items_label_scattered_btree"}
					]}
				]}
			]},"Execution Time":0.8}]`,
			want: "bitmap_meta",
		},
		{
			name: "parallel bitmap heap scan",
			raw: `[{"Plan":{"Node Type":"Limit","Plans":[
				{"Node Type":"Gather Merge","Plans":[
					{"Node Type":"Sort","Plans":[
						{"Node Type":"Bitmap Heap Scan","Relation Name":"items","Parallel Aware":true,"Plans":[
							{"Node Type":"Bitmap Index Scan","Index Name":"items_label_clustered_btree"}
						]}
					]}
				]}
			]},"Execution Time":0.9}]`,
			want: "bitmap_meta",
		},
		{
			name: "priority: hnsw wins over seq scan elsewhere in tree",
			raw: `[{"Plan":{"Node Type":"Nested Loop","Plans":[
				{"Node Type":"Index Scan","Index Name":"items_embedding_hnsw","Relation Name":"items"},
				{"Node Type":"Seq Scan","Relation Name":"items"}
			]},"Execution Time":3.0}]`,
			want: "hnsw_index",
		},
		{
			name: "priority: bitmap_meta wins over seq scan",
			raw: `[{"Plan":{"Node Type":"Append","Plans":[
				{"Node Type":"Bitmap Heap Scan","Relation Name":"items","Plans":[
					{"Node Type":"Bitmap Index Scan","Index Name":"items_label_clustered_btree"}
				]},
				{"Node Type":"Seq Scan","Relation Name":"items"}
			]},"Execution Time":3.0}]`,
			want: "bitmap_meta",
		},
		{
			name: "btree index scan on label is not hnsw",
			raw: `[{"Plan":{"Node Type":"Limit","Plans":[
				{"Node Type":"Sort","Plans":[
					{"Node Type":"Index Scan","Index Name":"items_label_clustered_btree","Relation Name":"items"}
				]}
			]},"Execution Time":1.0}]`,
			want: "other",
		},
		{
			name: "seq scan on foreign table ignored",
			raw: `[{"Plan":{"Node Type":"Seq Scan","Relation Name":"pg_stats"},"Execution Time":0.1}]`,
			want: "other",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := parseExplain(tc.raw)
			if err != nil {
				t.Fatalf("parseExplain: %v", err)
			}
			if res.PlanType != tc.want {
				t.Errorf("PlanType = %q, want %q", res.PlanType, tc.want)
			}
			if res.Raw != tc.raw {
				t.Errorf("Raw not preserved")
			}
			if res.ExecutionMS == 0 {
				t.Errorf("ExecutionMS not parsed")
			}
		})
	}
}

func TestParseExplainErrors(t *testing.T) {
	if _, err := parseExplain("not json"); err == nil {
		t.Error("want error on invalid json")
	}
	if _, err := parseExplain("[]"); err == nil {
		t.Error("want error on empty explain output")
	}
}

func TestVerifyPlanMode(t *testing.T) {
	cases := []struct {
		mode, plan string
		wantErr    bool
	}{
		{"auto", "hnsw_index", false},
		{"auto", "seq_scan", false},
		{"force_exact", "seq_scan", false},
		{"force_exact", "bitmap_meta", false}, // 拍板 #1:兩種精確 plan 皆算生效
		{"force_exact", "hnsw_index", true},
		{"force_exact", "other", true},
	}
	for _, tc := range cases {
		err := verifyPlanMode(tc.mode, tc.plan)
		if (err != nil) != tc.wantErr {
			t.Errorf("verifyPlanMode(%q, %q) err = %v, wantErr %v", tc.mode, tc.plan, err, tc.wantErr)
		}
	}
}
