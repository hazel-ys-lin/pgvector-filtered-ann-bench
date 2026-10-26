# .env(gitignored)存本機秘密,如 GEMINI_API_KEY=...;存在即自動載入並外傳給子行程
-include .env
export

DATABASE_URL ?= postgres://bench:bench@localhost:5433/bench

.PHONY: up down smoke shm-check test

up:
	docker compose up -d --wait

down:
	docker compose down -v

# A.6 坑1 驗證:容器內 /dev/shm 必須是 2GB(compose shm_size 生效)
shm-check:
	docker compose exec -T db df -h /dev/shm | grep -E '\b2\.0G\b'

test:
	go test ./...

# 順序 2 驗收:一鍵冒煙(環境起動 → shm 驗證 → db 模組整合測試)
smoke: up shm-check
	DATABASE_URL=$(DATABASE_URL) go test -tags smoke -count=1 -v ./internal/db/ -run TestSmoke

# 順序 3:5k 首批語料(需 GEMINI_API_KEY;Python 由 uv 管理,PEP 723 inline deps)
corpus-pilot:
	uv run cmd/corpusgen/corpusgen.py --total 5000 \
		--config configs/domains_v1.toml \
		--out corpus/corpus_5k.jsonl --report corpus/gen_report_5k.json

corpusgen-selftest:
	uv run cmd/corpusgen/corpusgen.py --selftest

# 順序 5:30k 生成(#27/#29/#30:thinking_budget=512 + batch 50,可續跑)。
# 注意:發佈的 corpus_30k 前 5,780 id 是舊配置(thinking 預設、batch 30)生成,
# 從零重跑只能得到「同方法」的語料,不是逐字相同(LLM 非決定性)。
corpus-30k:
	uv run cmd/corpusgen/corpusgen.py --total 30000 --batch-size 50 --thinking-budget 512 \
		--config configs/domains_v1.toml \
		--out corpus/corpus_30k.jsonl --report corpus/gen_report_30k.json

# 順序 3:5k 語料 embed(#16 task_type=document;#17 manifest 自動出)
embed-5k:
	go run ./cmd/embed --in corpus/corpus_5k.jsonl --out corpus/vectors_5k.jsonl \
		--task-type document

# 順序 3:COCO 參照抽樣(#18;COCO_ZIP= 指向官方 annotations_trainval2017.zip)
coco-sample:
	uv run scripts/coco_sample.py --zip $(COCO_ZIP) --out corpus/coco_ref_1k.jsonl \
		--n 1000 --seed 42

embed-coco:
	go run ./cmd/embed --in corpus/coco_ref_1k.jsonl --out corpus/coco_ref_1k_vec.jsonl \
		--task-type document

# 順序 3:幾何檢查 1–7(檢查 6 需 DB;up 先。檢查 7 是 bench 前閘門)
validate: up
	DATABASE_URL=$(DATABASE_URL) go run ./cmd/validate \
		--corpus corpus/corpus_5k.jsonl --vectors corpus/vectors_5k.jsonl \
		--ref corpus/coco_ref_1k_vec.jsonl \
		--queries corpus/queries_5k.jsonl --query-vectors corpus/queries_5k_vec.jsonl \
		--report corpus/geometry_report.json

# 順序 4:5k 彩排(管線驗證,非現象量測——動工順序 4 注記②)
bench-5k: up
	DATABASE_URL=$(DATABASE_URL) go run ./cmd/bench --out-dir results/run_5k

# ---- 30k 正式量測 ----
# 需 API key 的步驟:corpus-30k、embed-30k、queries-30k(重現語料生成)。
# 零 API 的步驟:reproduce-30k(從發佈的 JSONL 重現所有量測數字)。
C30 = --corpus corpus/corpus_30k.jsonl --vectors corpus/vectors_30k.jsonl \
	--queries corpus/queries_30k.jsonl --query-vectors corpus/queries_30k_vec.jsonl

embed-30k:
	go run ./cmd/embed --in corpus/corpus_30k.jsonl --out corpus/vectors_30k.jsonl \
		--task-type document

queries-30k:
	uv run cmd/corpusgen/querygen.py --corpus corpus/corpus_30k.jsonl \
		--config configs/domains_v1.toml --out corpus/queries_30k.jsonl \
		--report corpus/querygen_report_30k.json
	go run ./cmd/embed --in corpus/queries_30k.jsonl --out corpus/queries_30k_vec.jsonl \
		--task-type query

validate-30k: up
	DATABASE_URL=$(DATABASE_URL) go run ./cmd/validate $(C30) \
		--ref corpus/coco_ref_1k_vec.jsonl --report corpus/geometry_report_30k.json

# 主矩陣(160 配置)+ 預設檔位的四方對照
bench-30k: up
	DATABASE_URL=$(DATABASE_URL) go run ./cmd/bench $(C30) \
		--geometry corpus/geometry_report_30k.json --out-dir results/run_30k

# 四方對照移到實測危險檔位 p50(#33;沿用 bench-30k 已灌好的資料庫)
fourway-30k: up
	DATABASE_URL=$(DATABASE_URL) go run ./cmd/bench $(C30) \
		--geometry corpus/geometry_report_30k.json --out-dir results/run_30k_fourway_p50 \
		--sets fourway --fourway-tiers p50 --skip-load

summary-30k:
	uv run scripts/summarize.py --geometry corpus/geometry_report_30k.json \
		--matrix results/run_30k/results.jsonl \
		--fourway results/run_30k_fourway_p50/results.jsonl

reproduce-30k: validate-30k bench-30k fourway-30k summary-30k
