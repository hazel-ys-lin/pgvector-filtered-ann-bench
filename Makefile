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
