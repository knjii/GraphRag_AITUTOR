#!/usr/bin/env bash
# Межкнижная проверка «не хуже» серии S (docs/HYPOTHESES.md, S1 SetR и S4 SEAL),
# не сделанная 2026-09-30, и строка с цепочкой Laya, если серия L её допустила.
# Набор — goldset-x test (60 межкнижных вопросов), корпус — коллекция учебника
# library_ru, граф — файл v4.json.gz; мерят те же скрипты, что MuSiQue.
#
#   bash deploy/crossbook-bench.sh --list
#   bash deploy/crossbook-bench.sh            всё по порядку, с места остановки
#   bash deploy/crossbook-bench.sh --only A1  один шаг заново
#
# Запускать в tmux ПОСЛЕ deploy/laya-bench.sh (карта одна):
#   tmux has-session -t xbook 2>/dev/null || tmux new-session -d -s xbook \
#       'bash deploy/crossbook-bench.sh; exec bash'

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

BUNDLE="$REPO_DIR/artifacts/bench/goldset-x"
RUN="$REPO_DIR/artifacts/runs/bench/goldset-x"
OURS="$RUN/ours"
ANSWERS="$RUN/answers"
LOG="$RUN/crossbook-bench.log"
DONE="$RUN/done"
LAYA_MODEL="$REPO_DIR/artifacts/laya/model"
LAYA_HELD="$REPO_DIR/artifacts/runs/bench/musique-300/laya/heldout-ft.json"
LAYA_PY="$REPO_DIR/.venv-laya/bin/python"
K_ANSWER=5
WORKERS=8
mkdir -p "$DONE" "$OURS" "$ANSWERS"

# Конфигурация учебника — BASE_ENV дня 4 (deploy/day4.sh), окно 16 для отборщиков,
# как у ours-current на MuSiQue.
CURRENT_ENV=(QDRANT_COLLECTION=library_ru QDRANT_SPARSE_LANGUAGE=russian
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
             EVAL_K_VALUES=1,3,5,8,12,16
             LLM_REASONING_EFFORT=none)

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*" | tee -a "$LOG"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
die()  { printf '\n\033[1;31mСТОП: %s\033[0m\n' "$*" | tee -a "$LOG" >&2; exit 1; }
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications|HTTP Request: POST'; local rc=$?; [ "$rc" -le 1 ]; }
run() {
    local desc="$1"; shift
    "$@" 2>&1 | clean | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die "$desc — код возврата $rc (журнал: $LOG)"
}

ensure_4b() {
    if curl -sf http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q 'Qwen3.5-4B' \
        && curl -sf -o /dev/null http://127.0.0.1:7997/health \
        && ! docker ps --format '{{.Names}}' | grep -qxE 'judge|generator'; then
        ok "4B и поиск уже подняты"
    else
        docker rm -f judge generator >/dev/null 2>&1
        run "переключение на 4B" bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval
        run "запуск служб" bash deploy/services.sh up
    fi
}

# Цепочка Laya на межкнижном — только если серия L признала классификатор
# обученным (AUC связи ≥ 0.70 на отложенных вопросах MuSiQue).
laya_ok() {
    [ -s "$LAYA_MODEL/model.safetensors" ] && [ -x "$LAYA_PY" ] && [ -s "$LAYA_HELD" ] \
        && "$LAYA_PY" -c "import json,sys; sys.exit(0 if json.load(open('$LAYA_HELD'))['pair']['auc'] >= 0.70 else 1)"
}

step_C0() {  # набор цел, граф учебника на месте
    [ -s "$REPO_DIR/artifacts/graphs/v4.json.gz" ] || die "нет artifacts/graphs/v4.json.gz"
    [ -s evaluation/goldsets/goldset-x.json ] || die "нет evaluation/goldsets/goldset-x.json"
    sha256sum -c --quiet evaluation/goldsets/goldset-x.accepted || die "goldset-x.json отличается от замороженного"
    curl -sf -o /dev/null "http://127.0.0.1:6333/collections/library_ru" || die "нет коллекции library_ru в Qdrant"
    ok "goldset-x заморожен, library_ru на месте"
}

step_X0() {  # набор в формате бенча: 60 вопросов test + все фрагменты library_ru
    run "набор goldset-x" env QDRANT_COLLECTION=library_ru uv run python scripts/crossbook_bundle.py \
        --out "$BUNDLE" --split test
}

rank() {  # система, окружение…
    local system="$1"; shift
    ensure_4b
    run "выдача $system" env "$@" uv run python scripts/bench_ours.py rank --bundle "$BUNDLE" \
        --out "$OURS" --system "$system" --workers "$WORKERS"
}

step_R1() { rank x-current "${CURRENT_ENV[@]}"; }

step_R2() {  # SEAL-RAG с буфером k=5, как на MuSiQue
    rank x-current+seal "${CURRENT_ENV[@]}" RETRIEVAL_SELECTION=seal \
        RETRIEVAL_TOP_K="$K_ANSWER" RETRIEVAL_TOP_K_LINKING="$K_ANSWER"
}

step_S1() {  # SetR по окну 20 поверх x-current
    ensure_4b
    run "отбор setr" env LLM_REASONING_EFFORT=none uv run python scripts/bench_select.py \
        --bundle "$BUNDLE" --rankings "$OURS/rankings-x-current.jsonl" \
        --mode setr --pool 20 --top-k 16 --max-tokens 4096 --workers "$WORKERS"
}

step_L1() {  # цепочки Laya (объясняющие строки), если классификатор допущен
    if ! laya_ok; then
        warn "классификатор Laya не допущен серией L — строки с цепочкой пропущены"
        return 0
    fi
    docker rm -f judge generator >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.vllm.yml --profile sglang stop sglang >/dev/null 2>&1
    run "цепочка" "$LAYA_PY" scripts/laya_eval.py chain --model "$LAYA_MODEL" --bundle "$BUNDLE" \
        --rankings "$OURS/rankings-x-current.jsonl" --out "$OURS/rankings-x-current+chain.jsonl"
    run "setr-fill" "$LAYA_PY" scripts/laya_eval.py setrfill \
        --rankings "$OURS/rankings-x-current+setr.jsonl" --out "$OURS/rankings-x-current+setr-fill.jsonl"
    run "цепочка от SetR" "$LAYA_PY" scripts/laya_eval.py chain --model "$LAYA_MODEL" --bundle "$BUNDLE" \
        --seed selected --rankings "$OURS/rankings-x-current+setr.jsonl" \
        --out "$OURS/rankings-x-current+setr+chain.jsonl"
}

GEN_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
GEN_MODEL=/models/qwen9b/Qwen3.5-9B-UD-Q4_K_XL.gguf
ensure_9b_alone() {
    local slots=16 slot_tokens=8192 config
    config="$slots x $slot_tokens"
    if docker ps --format '{{.Names}}' | grep -qx generator \
        && [ "$(docker inspect -f '{{index .Config.Labels "slots"}}' generator 2>/dev/null)" = "$config" ] \
        && curl -sf -o /dev/null http://127.0.0.1:8001/health; then
        ok "9B-Q4 уже поднята ($config)"
        return
    fi
    docker rm -f judge generator rag-textbook-sglang-1 >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.vllm.yml --profile sglang stop sglang >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.yml stop infinity ollama >/dev/null 2>&1
    run "запуск llama.cpp (9B-Q4, $config)" docker run -d --name generator --gpus all \
        --label "slots=$config" \
        -v rag-textbook_gguf_models:/models -p 127.0.0.1:8001:8000 "$GEN_IMAGE" \
        -m "$GEN_MODEL" --host 0.0.0.0 --port 8000 \
        -c $((slot_tokens * slots)) -np "$slots" -ngl 999 --jinja \
        --reasoning off --reasoning-effort minimal
    local ready=0
    for _ in $(seq 1 60); do
        curl -sf -o /dev/null http://127.0.0.1:8001/health && { ready=1; break; }
        sleep 5
    done
    [ "$ready" = 1 ] || { docker logs --tail 25 generator 2>&1 | tee -a "$LOG"; die "llama.cpp не поднялся"; }
}

step_A1() {  # ответы 9B-Q4 при k=5, по-русски; база — x-current
    ensure_9b_alone
    local runs=(--run x-current="$OURS/rankings-x-current.jsonl"
                --run x-current+setr="$OURS/rankings-x-current+setr.jsonl:selected"
                --run x-current+seal="$OURS/rankings-x-current+seal.jsonl")
    [ -s "$OURS/rankings-x-current+chain.jsonl" ] && runs+=(
                --run x-current+chain="$OURS/rankings-x-current+chain.jsonl"
                --run x-current+setr-fill="$OURS/rankings-x-current+setr-fill.jsonl"
                --run x-current+setr+chain="$OURS/rankings-x-current+setr+chain.jsonl")
    run "ответы" env LLM_MODEL=Qwen3.5-9B-UD-Q4_K_XL LLM_CHAT_MODEL= LLM_REASONING_EFFORT=none \
        LLM_MAX_CONCURRENCY=16 uv run python scripts/bench_answers.py --bundle "$BUNDLE" \
        --k "$K_ANSWER" --baseline x-current --workers 16 --max-tokens 1536 --prompt ru \
        --out "$ANSWERS" "${runs[@]}"
    cp "$ANSWERS/answers-report.json" "$RUN/answers-crossbook.json"
    uv run python - "$ANSWERS" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
for path in sorted(Path(sys.argv[1]).glob("answers-x-*.jsonl")):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    no_answer = sum("answer:" not in (r["raw"] or "").lower() for r in rows)
    errors = sum(bool(r.get("error")) for r in rows)
    print(f"{path.name}: ответов {len(rows)}, без строки Answer {no_answer}, ошибок {errors}")
    if rows:
        print("  образец", rows[0]["qid"], repr((rows[0]["raw"] or "")[-300:]))
PY
    run "вердикт «не хуже»" uv run python scripts/crossbook_verdict.py --bundle "$BUNDLE" \
        --answers "$ANSWERS" --base x-current --out "$RUN/crossbook-verdict.json"
}

step_Z1() {  # результаты одним архивом
    tar -czf "$REPO_DIR/artifacts/crossbook-results.tgz" -C "$RUN" ours answers answers-crossbook.json \
        crossbook-verdict.json \
        crossbook-bench.log -C "$BUNDLE" manifest.json questions.jsonl \
        || die "не упаковать результаты"
    ok "результаты: $REPO_DIR/artifacts/crossbook-results.tgz"
}

STEPS=(C0 X0 R1 R2 S1 L1 A1 Z1)
declare -A DESC=(
    [C0]="goldset-x заморожен, library_ru и граф учебника на месте"
    [X0]="набор goldset-x test в формате бенча"
    [R1]="выдача x-current (конфигурация учебника, окно 16)"
    [R2]="выдача x-current+seal (k=$K_ANSWER)"
    [S1]="SetR по окну 20"
    [L1]="цепочки Laya, если серия L допустила классификатор"
    [A1]="ответы 9B-Q4 по-русски при k=$K_ANSWER"
    [Z1]="упаковка результатов"
)
ALWAYS=" C0 Z1 "

only="" from=""
case "${1:-}" in
    --list) for s in "${STEPS[@]}"; do printf '%s  %s\n' "$s" "${DESC[$s]}"; done; exit 0 ;;
    --only) only="$2" ;;
    --from) from="$2" ;;
esac

if [ -n "$only" ]; then
    say "$only — ${DESC[$only]:-?} ($(date '+%H:%M:%S'))"
    declare -F "step_$only" >/dev/null || die "нет шага $only"
    [ "$only" = C0 ] || step_C0
    "step_$only"
    [[ "$ALWAYS" == *" $only "* ]] || touch "$DONE/$only"
    ok "готово: $only"
    exit 0
fi

started=""
[ -z "$from" ] && started=1
for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && started=1
    [ -n "$started" ] || continue
    if [ -f "$DONE/$s" ] && [[ "$ALWAYS" != *" $s "* ]]; then
        ok "$s уже сделан"
        continue
    fi
    say "$s — ${DESC[$s]} ($(date '+%H:%M:%S'))"
    "step_$s"
    [[ "$ALWAYS" == *" $s "* ]] || touch "$DONE/$s"
done
ok "готово: $RUN"
