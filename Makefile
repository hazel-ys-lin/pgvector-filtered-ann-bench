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

# 順序 5:30k 分段生成(#27/#28:thinking=0 + batch 50,可續跑)
corpus-30k:
	uv run cmd/corpusgen/corpusgen.py --total 30000 --batch-size 50 --thinking-budget 0 \
		--config configs/domains_v1.toml \
		--out corpus/corpus_5k.jsonl --report corpus/gen_report_30k.json

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
