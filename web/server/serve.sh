#!/usr/bin/env bash
# Демо «Матчасть» на арендованном сервере, рядом с идущими замерами.
#
#   bash web/server/serve.sh          поднять (или перезапустить) tmux-сессию demo
#   bash web/server/serve.sh --stop   остановить только её
#
# Ничего чужого не трогает: модели не переключает, службы не перезапускает,
# порт берёт свой (DEMO_PORT, по умолчанию 8090) и слушает только 127.0.0.1 —
# смотреть через SSH-туннель. Конфигурация поиска — принятая продуктовая
# (BASE_ENV дня 4: library_ru, граф v4 из файла, top-16, маршрут всегда в граф).

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
PORT="${DEMO_PORT:-8090}"
SESSION=demo
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }

if [ "${1:-}" = "--stop" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "demo остановлен" || echo "demo не был запущен"
    exit 0
fi

# Порт занят не нами — не отбираем.
tmux kill-session -t "$SESSION" 2>/dev/null
sleep 1
if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
    echo "порт $PORT занят другим процессом, задайте DEMO_PORT" >&2
    exit 1
fi

ENV_VARS=(QDRANT_COLLECTION=library_ru QDRANT_SPARSE_LANGUAGE=russian
          CHUNKER_RESPECT_FORMULAS=true CHUNK_SIZE=1200 CHUNK_OVERLAP=180
          GRAPH_ENABLED=true GRAPH_RETRIEVAL_ENABLED=true
          GRAPH_BACKEND=memory GRAPH_FILE="$REPO_DIR/artifacts/graphs/v4.json.gz"
          GRAPH_RANKER=walk GRAPH_WALK=comention
          GRAPH_EXPANSION_HOPS=1 GRAPH_SEED_ENTITY_LIMIT=20 GRAPH_PASSAGE_LIMIT=30
          GRAPH_MAX_ENTITY_DEGREE=64 GRAPH_HOP_DECAY=0.5 GRAPH_PASSAGE_IDF_ENABLED=false
          GRAPH_SEED_MODE=both GRAPH_SEED_PASSAGES=3 GRAPH_EXPANSION_REL_TYPES=RELATES
          GRAPH_WEIGHT=0.4 GRAPH_EXTRACTION_PROMPT_VERSION=v4
          RETRIEVAL_DENSE_CANDIDATES=40 RETRIEVAL_SPARSE_CANDIDATES=40
          RETRIEVAL_ROUTER_ENABLED=true RETRIEVAL_ROUTER_MODE=always
          RETRIEVAL_TOP_K=16 RETRIEVAL_TOP_K_LINKING=16 RETRIEVAL_RRF_K=60
          RETRIEVAL_DEDUP_ENABLED=true RETRIEVAL_DEDUP_SIMILARITY=0.92
          RETRIEVAL_MIN_GRAPH_DOCS=0 RETRIEVAL_GRAPH_CANDIDATE_QUOTA=6
          RETRIEVAL_DIVERSITY_MODE=off RETRIEVAL_SELECTION=off
          RETRIEVAL_QUERY_REWRITE_ENABLED=true RETRIEVAL_DECOMPOSE_ENABLED=false
          RERANKER_ENABLED=true RERANKER_MODE=always RERANKER_BLEND_ALPHA=1.0
          RERANKER_TOP_N=8 RERANKER_CANDIDATES=30
          LLM_REASONING_EFFORT=none DEMO_MAX_CONCURRENT=1)

LOG="$REPO_DIR/web/demo.log"
tmux new-session -d -s "$SESSION" \
    "cd '$REPO_DIR' && env ${ENV_VARS[*]} .venv/bin/python web/server/run.py --repo '$REPO_DIR' --mode server --host 127.0.0.1 --port $PORT 2>&1 | tee '$LOG'; exec bash"

for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/info" >/dev/null 2>&1 && break
    sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/api/info" >/dev/null 2>&1; then
    echo "demo не ответил за 2 минуты, хвост журнала:" >&2
    tail -n 30 "$LOG" >&2
    exit 1
fi
echo "demo поднят на 127.0.0.1:$PORT"
