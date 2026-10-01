#!/usr/bin/env bash
# День 5: К11 (рёбра-синонимы), допуск гибридного судьи, данные и проба обучения 9B.
# Каждый шаг отмечается в artifacts/runs/day5/done, повторный запуск продолжает.
#
#   bash deploy/day5.sh --list
#   bash deploy/day5.sh                 всё по порядку
#   bash deploy/day5.sh --only P1       один шаг заново
#   bash deploy/day5.sh --from Q2       начиная с шага
#
# Критерии записаны до замера в docs/HYPOTHESES.md (К11, M6 r1).
# Модели названы явно (правило владельца): 4B — поиск и переписывание запроса
# в замерах К11 и слепок эпизодов; 9B — всё, что генерирует ответы или
# вопросы (Q1, D1), и обучаемая модель (P1); 27B — судья (J2).
#
# Порядок задан одной картой: 4B+поиск (S1–KV) → 9B один (Q1, D1) →
# 4B+поиск (Q2) → 27B (J2) → пустая карта (P1). Лист D1 скачивается
# сразу после D1 и оценивается владельцем, пока идут Q2–P1.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

RUN="$REPO_DIR/artifacts/runs/day5"
RUN3="$REPO_DIR/artifacts/runs/day3"
RUN4="$REPO_DIR/artifacts/runs/day4"
RUN1="$REPO_DIR/artifacts/runs/day1"   # окружения RL и отметки их сборки — общие с днём 1
GRAPHS="$REPO_DIR/artifacts/graphs"
OUT="$REPO_DIR/artifacts/rl"
LOG="$RUN/day5.log"
mkdir -p "$RUN/done" "$OUT" "$RUN1"

GOLD_X=evaluation/goldsets/goldset-x.json
GOLD_V2=evaluation/goldsets/goldset-v2.json
SYN_GRAPH="$GRAPHS/v4syn-080.json.gz"
MML_DOC=0690bb81b7e3c831
PROMPT_V4=deploy/prompts/qa-v4.txt
V1_CALIB="$RUN3/judged/calibration/calibration.json"
MODEL_9B=Qwen/Qwen3.5-9B
COLLECTION=library_ru

# Конвейер — буква в букву BASE_ENV дней 3–4: К11 меняет только граф и типы рёбер.
CORPUS_ENV=(QDRANT_COLLECTION="$COLLECTION" QDRANT_SPARSE_LANGUAGE=russian
            CHUNKER_RESPECT_FORMULAS=true CHUNK_SIZE=1200 CHUNK_OVERLAP=180)
BASE_ENV=("${CORPUS_ENV[@]}" GRAPH_ENABLED=true GRAPH_RETRIEVAL_ENABLED=true
          GRAPH_BACKEND=memory GRAPH_RANKER=walk GRAPH_WALK=comention
          GRAPH_EXPANSION_HOPS=1 GRAPH_SEED_ENTITY_LIMIT=20 GRAPH_PASSAGE_LIMIT=30
          GRAPH_MAX_ENTITY_DEGREE=64 GRAPH_HOP_DECAY=0.5 GRAPH_PASSAGE_IDF_ENABLED=false
          GRAPH_SEED_MODE=both GRAPH_SEED_PASSAGES=3 GRAPH_EXPANSION_REL_TYPES=RELATES
          GRAPH_WEIGHT=0.4 GRAPH_EXTRACTION_PROMPT_VERSION=v4
          RETRIEVAL_DENSE_CANDIDATES=40 RETRIEVAL_SPARSE_CANDIDATES=40
          RETRIEVAL_ROUTER_ENABLED=true RETRIEVAL_ROUTER_MODE=always
          RETRIEVAL_TOP_K=8 RETRIEVAL_TOP_K_LINKING=16 RETRIEVAL_RRF_K=60
          RETRIEVAL_DEDUP_ENABLED=true RETRIEVAL_DEDUP_SIMILARITY=0.92
          RETRIEVAL_MIN_GRAPH_DOCS=0 RETRIEVAL_GRAPH_CANDIDATE_QUOTA=6
          RETRIEVAL_DIVERSITY_MODE=off RETRIEVAL_SELECTION=off
          RETRIEVAL_QUERY_REWRITE_ENABLED=true RETRIEVAL_DECOMPOSE_ENABLED=false
          RERANKER_ENABLED=true RERANKER_MODE=always RERANKER_BLEND_ALPHA=1.0
          RERANKER_TOP_N=8 RERANKER_CANDIDATES=30
          EVAL_K_VALUES=1,3,5,8,12,16)

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*" | tee -a "$LOG"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
die()  { printf '\n\033[1;31mСТОП: %s\033[0m\n' "$*" | tee -a "$LOG" >&2; exit 1; }
clean() { grep -v -E ' INFO |pymorphy|neo4j\.notifications'; local rc=$?; [ "$rc" -le 1 ]; }
stamp() { date '+%H:%M:%S'; }

run() {
    local desc="$1"; shift
    "$@" 2>&1 | clean | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die "$desc — код возврата $rc (журнал: $LOG)"
}
LAST_OK=1
try() {
    local desc="$1"; shift
    "$@" 2>&1 | clean | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = 0 ]; then LAST_OK=1; else LAST_OK=0; warn "$desc — код возврата $rc"; fi
}

# ---------------------------------------------------------------- модели
ensure_4b() {
    if grep -q '^LLM_MODEL=Qwen/Qwen3.5-4B$' .env 2>/dev/null \
        && curl -sf http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q 'Qwen3.5-4B' \
        && curl -sf -o /dev/null http://127.0.0.1:7997/health \
        && ! docker ps --format '{{.Names}}' | grep -qxE 'judge|generator'; then
        ok "4B и поиск уже подняты"
    else
        docker rm -f judge generator >/dev/null 2>&1
        run "переключение на 4B" bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval
        run "запуск служб" bash deploy/services.sh up
    fi
}
# 9B для генерации — квант UD-Q4_K_XL на llama.cpp, как Q0 дня 2 (решение
# владельца 2026-09-22). bf16 на SGLang не стартует при доле 0.7: веса 18.87 ГБ
# больше 0.7 × 24 ГБ (день 5, первый запуск Q1 упал ровно на этом).
GEN_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
GEN_MODEL=/models/qwen9b/Qwen3.5-9B-UD-Q4_K_XL.gguf
# Слоты под задачу: вопросы (Q1) короткие — 16 × 8192; эпизоды RL (D1) несут
# контекст до окна rl_dataset 16384 плюс ответ 768 — 8 × 18432 (день 5:
# разминка D1 упала на промпте 8453 при слоте 8192).
ensure_9b_alone() {
    local slots="${1:-16}" slot_tokens="${2:-8192}" config
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
    ok "видеопамять: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    # Ответ читается до долгого прогона: пустой ответ или размышление вместо текста.
    run "пробный ответ 9B-Q4" uv run python - <<'PY'
import json, urllib.request
body = {"model": "qwen9b", "max_tokens": 200, "messages": [
    {"role": "user", "content": "Сформулируй одним предложением, что такое собственный вектор матрицы."}]}
req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]
text = (msg.get("content") or "").strip()
print("ответ:", text[:300])
print("размышление:", len(msg.get("reasoning_content") or ""), "знаков")
raise SystemExit(0 if text and "<think>" not in text and len(msg.get("reasoning_content") or "") < 200 else 1)
PY
}
all_off() {
    docker rm -f judge generator >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.vllm.yml --profile sglang stop sglang >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.yml stop infinity ollama >/dev/null 2>&1
    ok "карта свободна: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
hf_cache_dir() {
    docker volume inspect rag-textbook_hf_cache --format '{{.Mountpoint}}' 2>/dev/null \
        || { docker volume create rag-textbook_hf_cache >/dev/null \
             && docker volume inspect rag-textbook_hf_cache --format '{{.Mountpoint}}'; }
}
metrics_dir() { uv run python -c 'from rag_textbook.config import Settings; print(Settings().paths.metrics_dir)'; }
run_file() {
    local metrics file
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    file=$(ls -t "$metrics"/retrieval_eval_"$1"_*.json 2>/dev/null | head -1)
    [ -n "$file" ] || die "нет прогона с меткой $1 — сначала его шаг"
    printf '%s' "$file"
}

# Окружения обучения — те же, что готовил день 1 (задача 018).
rl_env_dir()  { [ "$1" = vllm ] && echo .venv-rl-vllm || echo .venv-rl; }
rl_env_reqs() { [ "$1" = vllm ] && echo deploy/requirements-rl-vllm.txt || echo deploy/requirements-rl.txt; }
rl_env_hash() { sha256sum "$(rl_env_reqs "$1")" deploy/requirements-rl-app.txt | sha256sum | cut -d' ' -f1; }
rl_env_done() {
    [ -x "$(rl_env_dir "$1")/bin/python" ] \
        && [ "$(cat "$RUN1/rl-env-$1.ok" 2>/dev/null)" = "$(rl_env_hash "$1")" ]
}
rl_env_install() {
    local stack="$1" dir want
    dir="$(rl_env_dir "$stack")"; want="$(rl_env_hash "$stack")"
    rl_env_done "$stack" && { ok "окружение $stack уже собрано"; return; }
    if [ -f "$RUN1/rl-env-$stack.pid" ] && kill -0 "$(cat "$RUN1/rl-env-$stack.pid")" 2>/dev/null; then
        ok "окружение $stack уже ставится"; return
    fi
    rm -f "$RUN1/rl-env-$stack.ok"
    (
        uv venv --python 3.11 "$dir" >/dev/null 2>&1 \
            && uv pip install -p "$dir" -r "$(rl_env_reqs "$stack")" \
            && uv pip install -p "$dir" -r deploy/requirements-rl-app.txt \
            && echo "$want" > "$RUN1/rl-env-$stack.ok"
    ) > "$RUN1/rl-env-$stack.log" 2>&1 &
    echo $! > "$RUN1/rl-env-$stack.pid"
    ok "окружение $stack ставится фоном: $RUN1/rl-env-$stack.log"
}
rl_env_ready() {
    local stack="$1" pid
    rl_env_done "$stack" && return 0
    rl_env_install "$stack"
    pid="$(cat "$RUN1/rl-env-$stack.pid" 2>/dev/null)"
    for _ in $(seq 1 90); do
        rl_env_done "$stack" && return 0
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || break
        sleep 30
    done
    rl_env_done "$stack"
}

# ------------------------------------------------------------------- план
PLAN=$(cat <<'PLAN'
E0|Проверки: эталоны заморожены, граф v4, судья v1 дня 3, эпизоды MML; фоном — окружения RL и веса 9B
S1|К11: граф v4 + рёбра SYNONYM по bge-m3 (порог 0.80, k 10) — 4B+поиск
X1|К11: прогоны база/K11 на goldset-x и goldset-v2 — 4B+поиск
KV|Вердикт К11 по критериям, записанным до замера
Q1|Вопросы для обучения по русскому комплекту (без MML) — 9B
D1|Д1+Д2: группы генераций 9B (20 × 8) и слепой лист для владельца — 9B
Q2|Аудит вопросов, слепок поиска, эпизоды RL (MML уходит в тест) — 4B+поиск
J2|Допуск судьи: v2 на within (27B), затем гибрид r1 по критерию M6
R0|Окружения RL: импорты, карта видна, отпечаток награды
P1|Проба GRPO 9B: Unsloth 4 бит, Unsloth bf16, TRL+vLLM; правило probe_compare.py
Z9|Сводка и архив
PLAN
)
CODES=$(printf '%s\n' "$PLAN" | cut -d'|' -f1)

ONLY=""; FROM=""; LIST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --list) LIST=1; shift ;;
        --only) ONLY="${2:-}"; [ -n "$ONLY" ] || die "--only без имени шага"; shift 2 ;;
        --from) FROM="${2:-}"; [ -n "$FROM" ] || die "--from без имени шага"; shift 2 ;;
        *) die "неизвестный аргумент $1" ;;
    esac
done
if [ "$LIST" = 1 ]; then
    printf '%s\n' "$PLAN" | awk -F'|' '{printf "  %-3s %s\n", $1, $2}'
    exit 0
fi
for chosen in "$ONLY" "$FROM"; do
    [ -z "$chosen" ] && continue
    printf '%s\n' "$CODES" | grep -qx "$chosen" || die "нет шага $chosen; список — --list"
done
STARTED=$([ -z "$FROM" ] && echo 1 || echo 0)
should_run() {
    local code="$1"
    [ -n "$ONLY" ] && { [ "$code" = "$ONLY" ]; return; }
    [ "$STARTED" = 0 ] && [ "$code" = "$FROM" ] && STARTED=1
    [ "$STARTED" = 1 ] || return 1
    [ -f "$RUN/done/$code" ] && { ok "$code уже выполнен"; return 1; }
    return 0
}
done_mark() { date '+%F %T' > "$RUN/done/$1"; ok "$1 готов ($(stamp))"; }

# ---------------------------------------------------------------- шаги
step_E0() {
    say "E0. Проверка перед стартом"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
    df -h "$REPO_DIR" | tail -1 | tee -a "$LOG"
    sha256sum -c --quiet evaluation/goldsets/goldset-x.accepted || die "goldset-x отличается от замороженного"
    sha256sum -c --quiet evaluation/goldsets/goldset-v2.accepted || die "goldset-v2 отличается от принятого"
    for required in "$GRAPHS/v4.json.gz" "$V1_CALIB" "$RUN4/judge-v2/across-r0.fingerprint" \
                    "$OUT/mml-test.jsonl" "$OUT/reward-fingerprint.json" "$PROMPT_V4" \
                    deploy/requirements-rl.txt deploy/requirements-rl-vllm.txt deploy/requirements-rl-app.txt; do
        [ -s "$required" ] || die "нет или пуст $required"
    done
    [ -e "$RUN4/judge-v2/within-final" ] && die "within уже израсходована — допуск M6 ставится один раз"
    [ -e "$SYN_GRAPH" ] && die "$SYN_GRAPH уже есть — К11 строится один раз, до замера"
    run "отпечаток награды (окружение сервиса)" uv run python scripts/reward_fingerprint.py \
        --dataset "$OUT/mml-test.jsonl" --check "$OUT/reward-fingerprint.json"
    run "тесты новых частей" uv run python -m pytest -q tests/test_graph_synonyms.py \
        tests/test_judge_hybrid.py
    for stack in vllm unsloth; do rl_env_install "$stack"; done
    command -v hf >/dev/null 2>&1 || uv tool install "huggingface_hub[cli]" >/dev/null 2>&1
    ( HF_HOME="$(hf_cache_dir)"; export HF_HOME
      hf download "$MODEL_9B" --exclude "*.gguf" >/dev/null && echo "загрузка весов завершена"
    ) > "$RUN/weights.log" 2>&1 &
    echo $! > "$RUN/weights.pid"
    ok "веса 9B для обучения качаются фоном: $RUN/weights.log"
    ensure_4b
    done_mark E0
}

step_S1() {
    say "S1. К11: рёбра-синонимы"
    ensure_4b
    run "граф с синонимами" env "${CORPUS_ENV[@]}" uv run python scripts/graph_synonyms.py \
        --graph "$GRAPHS/v4.json.gz" --out "$SYN_GRAPH" --threshold 0.80 --k 10 --max-degree 64
    done_mark S1
}

variant() {
    local goldset="$1" label="$2"; shift 2
    run "прогон $label" env "${BASE_ENV[@]}" "$@" \
        uv run rag-textbook eval run --goldset "$goldset" --label "$label" \
        --trace "$RUN/trace-$label.jsonl"
}

step_X1() {
    say "X1. К11: прогоны"
    ensure_4b
    [ -s "$SYN_GRAPH" ] || die "нет $SYN_GRAPH — шаг S1"
    # База переснимается в тот же день тем же кодом: сравнение без дрейфа.
    variant "$GOLD_X"  x5-base  GRAPH_FILE="$GRAPHS/v4.json.gz"
    variant "$GOLD_X"  x5-K11   GRAPH_FILE="$SYN_GRAPH" GRAPH_EXPANSION_REL_TYPES=RELATES,SYNONYM
    variant "$GOLD_V2" v5-base  GRAPH_FILE="$GRAPHS/v4.json.gz"
    variant "$GOLD_V2" v5-K11   GRAPH_FILE="$SYN_GRAPH" GRAPH_EXPANSION_REL_TYPES=RELATES,SYNONYM
    rm -f "$RUN/done/KV"
    done_mark X1
}

step_KV() {
    say "KV. Вердикт К11"
    run "вердикт К11" uv run python scripts/k11_verdict.py \
        --x-goldset "$GOLD_X" --x-base "$(run_file x5-base)" --x-cand "$(run_file x5-K11)" \
        --v2-goldset "$GOLD_V2" --v2-base "$(run_file v5-base)" --v2-cand "$(run_file v5-K11)" \
        --json "$RUN/k11-verdict.json"
    # Граф для слепка эпизодов (Q2) выбирается этим вердиктом, правилом до замера:
    # принята — продукт идёт с синонимами, отвергнута — с v4.
    if grep -q '"decision": "принята"' "$RUN/k11-verdict.json"; then
        echo "$SYN_GRAPH" > "$RUN/graph-choice.txt"; ok "К11 принята: эпизоды — по графу с синонимами"
    else
        echo "$GRAPHS/v4.json.gz" > "$RUN/graph-choice.txt"; warn "К11 отвергнута: эпизоды — по графу v4"
    fi
    done_mark KV
}

step_Q1() {
    say "Q1. Вопросы для обучения (9B)"
    ensure_9b_alone
    # С запасом: эпизоды, чей контекст задел MML, уйдут в тест (Q2).
    run "генерация вопросов" env "${CORPUS_ENV[@]}" GRAPH_ENABLED=false LLM_MAX_CONCURRENCY=16 LLM_REASONING_EFFORT=none \
        uv run rag-textbook goldset build --single 2800 --multihop 700 --workers 16 \
        --exclude-doc "$MML_DOC" --output "$OUT/train-ru.json" --seed 20260929
    run "вопросы читаются" uv run python - "$OUT/train-ru.json" <<'PY'
import sys
from pathlib import Path
from rag_textbook.evaluation.goldset import load_goldset
questions = load_goldset(Path(sys.argv[1]))
print(f"вопросов: {len(questions)}")
raise SystemExit(0 if len(questions) >= 1500 else 1)
PY
    done_mark Q1
}

step_D1() {
    say "D1. Группы генераций 9B"
    ensure_9b_alone 8 18432
    rm -f "$RUN/smoke.summary.json"
    run "разминка генерации" uv run python scripts/sample_groups.py \
        --dataset "$OUT/mml-test.jsonl" --questions 2 --n 2 --out "$RUN/smoke.jsonl"
    run "разминка: ворота" uv run python - "$RUN/smoke.summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
gated = {k: v for k, v in summary["ворота"].items() if k != "прошёл"}
print("разминка:", summary["ворота"])
raise SystemExit(1 if sum(gated.values()) == summary["ответов"] else 0)
PY
    rm -f "$OUT/groups-9b.summary.json"
    run "группы генераций" uv run python scripts/sample_groups.py \
        --dataset "$OUT/mml-test.jsonl" --questions 20 --n 8 \
        --out "$OUT/groups-9b.jsonl" --sheet "$OUT/groups-9b"
    run "Д2: сигнал для GRPO" uv run python - "$OUT/groups-9b.summary.json" <<'PY'
import json, sys
share = json.load(open(sys.argv[1], encoding="utf-8"))["доля групп без разброса"]
print(f"Д2: доля групп без разброса {share} — {'ок' if share <= 0.5 else 'НЕТ СИГНАЛА'}")
PY
    warn "ЛИСТ ДЛЯ ВЛАДЕЛЬЦА: $OUT/groups-9b-sheet.md (ключ $OUT/groups-9b-key.json не открывать до оценки)"
    done_mark D1
}

step_Q2() {
    say "Q2. Аудит, слепок, эпизоды (4B + поиск)"
    ensure_4b
    local graph
    graph="$(cat "$RUN/graph-choice.txt" 2>/dev/null)"
    [ -s "$graph" ] || die "граф для слепка не выбран — шаг KV"
    run "общий корпус фрагментов" uv run python - "$RUN/library-chunks.json" <<'PY'
import json, sys
from pathlib import Path
from rag_textbook.config import Settings
from rag_textbook.rl.env import load_chunks
chunks = load_chunks(Path(Settings().paths.parsed_dir))
Path(sys.argv[1]).write_text(json.dumps([c.model_dump(mode="json") for c in chunks.values()],
                                        ensure_ascii=False), encoding="utf-8")
print(f"фрагментов в корпусе: {len(chunks)}")
PY
    run "аудит вопросов" uv run rag-textbook goldset audit --path "$OUT/train-ru.json" \
        --chunks "$RUN/library-chunks.json" --write "$OUT/train-ru-clean.json"
    # Эпизоды — тем же конвейером, что продукт (BASE_ENV, граф по вердикту К11).
    local rel=RELATES
    [ "$graph" = "$SYN_GRAPH" ] && rel=RELATES,SYNONYM
    run "слепок поиска" env "${BASE_ENV[@]}" GRAPH_FILE="$graph" GRAPH_EXPANSION_REL_TYPES="$rel" \
        EVAL_TRACE_RERANK_ALL=true uv run rag-textbook eval run --goldset "$OUT/train-ru-clean.json" \
        --trace "$OUT/train-ru-trace.jsonl" --label day5-train-ru
    rm -f "$OUT/lib-ru-train.jsonl" "$OUT/lib-ru-test.jsonl"
    run "эпизоды RL" uv run python scripts/rl_dataset.py --trace "$OUT/train-ru-trace.jsonl" \
        --goldset "$OUT/train-ru-clean.json" --prompt "$PROMPT_V4" \
        --out "$OUT/lib-ru" --test-docs "$MML_DOC"
    local episodes
    episodes=$(wc -l < "$OUT/lib-ru-train.jsonl")
    [ "$episodes" -gt 0 ] || die "эпизодов для обучения нет"
    ok "в тест ушло (контекст задел MML): $(wc -l < "$OUT/lib-ru-test.jsonl")"
    if [ "$episodes" -ge 2000 ]; then ok "Д5: эпизодов $episodes"; else warn "Д5: эпизодов $episodes < 2000"; fi
    done_mark Q2
}

step_J2() {
    say "J2. Допуск судьи (M6 r1: гибрид)"
    # Прогон v2 на within — шаг J2 дня 4 на замороженной r0 (код судьи не менялся).
    [ -f "$RUN4/done/J2" ] || JUDGE_FROZEN=r0 bash deploy/day4.sh --only J2 2>&1 | tee -a "$LOG"
    [ -s "$RUN4/judge-v2/within-final/calibration.json" ] || die "v2 на within не снят — журнал дня 4"
    warn "v2 на within — только для сведения; допуск решает гибрид (записано до замера)"
    run "гибрид r1 на within" uv run python scripts/judge_hybrid.py --v1 "$V1_CALIB" \
        --v2 "$RUN4/judge-v2/within-final/calibration.json" --group within \
        --json "$RUN/judge-hybrid-within.json"
    if grep -q '"accepted": true' "$RUN/judge-hybrid-within.json"; then
        ok "СУДЬЯ ДОПУЩЕН: гибрид r1 (v1 + атомарные проверки v2, 27B)"
    else
        warn "гибрид r1 не допущен — судья 27B закрывается (M6)"
    fi
    docker rm -f judge >/dev/null 2>&1
    done_mark J2
}

step_R0() {
    say "R0. Окружения RL"
    local stacks=()
    for stack in unsloth vllm; do
        if ! rl_env_ready "$stack"; then
            tail -20 "$RUN1/rl-env-$stack.log" 2>/dev/null; warn "окружение $stack не собралось"; continue
        fi
        local dir; dir="$(rl_env_dir "$stack")"
        try "импорты и карта ($stack)" "$dir/bin/python" -c \
            "import torch, trl, peft; assert torch.cuda.is_available(); print('torch', torch.__version__, 'trl', trl.__version__)"
        [ "$LAST_OK" = 1 ] || continue
        try "отпечаток награды ($stack)" "$dir/bin/python" scripts/reward_fingerprint.py \
            --dataset "$OUT/mml-test.jsonl" --check "$OUT/reward-fingerprint.json"
        [ "$LAST_OK" = 1 ] && stacks+=("$stack")
    done
    [ "${#stacks[@]}" -gt 0 ] || die "ни одно окружение обучения не собралось"
    printf '%s\n' "${stacks[@]}" > "$RUN/stacks.txt"
    ok "к пробе готовы: ${stacks[*]}"
    done_mark R0
}

# Одна проба: вариант, число генераций (4, при OOM 2), короткая проба ради доли генерации.
probe_one() {
    local name="$1" stack="$2"; shift 2
    local dir status gens label
    dir="$(rl_env_dir "$stack")"
    for gens in 4 2; do
        label="$name-g$gens"
        rm -rf "runs/probe-9b-$label"
        "$dir/bin/python" scripts/train_grpo.py --dataset "$OUT/lib-ru-train.jsonl" \
            --model "$MODEL_9B" --probe --out "runs/probe-9b-$label" \
            --num-generations "$gens" --grad-accum "$gens" "$@" \
            2>&1 | tee "$RUN/probe-$label.log" | tail -25 | tee -a "$LOG"
        status=${PIPESTATUS[0]}
        [ "$status" = 0 ] && break
        grep -q -i -E "out of memory|outofmemoryerror|cuda error: out" "$RUN/probe-$label.log" || break
        warn "$name: OOM при $gens генерациях"
    done
    [ "$status" = 0 ] || { warn "$name: проба не прошла — $RUN/probe-$label.log"; return; }
    cat "runs/probe-9b-$label/probe.json" | tee -a "$LOG"
    FRESH+=("runs/probe-9b-$label/probe.json")
    rm -rf "runs/probe-9b-$label-short"
    if "$dir/bin/python" scripts/train_grpo.py --dataset "$OUT/lib-ru-train.jsonl" \
        --model "$MODEL_9B" --probe --out "runs/probe-9b-$label-short" \
        --num-generations "$gens" --grad-accum "$gens" --max-completion-length 96 "$@" \
        > "$RUN/probe-$label-short.log" 2>&1; then
        FRESH+=("runs/probe-9b-$label-short/probe.json")
    else
        warn "$name: короткая проба не прошла, доля генерации не посчитается"
    fi
}

step_P1() {
    say "P1. Проба GRPO 9B"
    [ -s "$OUT/lib-ru-train.jsonl" ] || die "нет эпизодов — шаг Q2"
    all_off
    if [ -f "$RUN/weights.pid" ] && kill -0 "$(cat "$RUN/weights.pid")" 2>/dev/null; then
        warn "жду загрузку весов 9B"; wait "$(cat "$RUN/weights.pid")" 2>/dev/null
    fi
    HF_HOME="$(hf_cache_dir)"; export HF_HOME
    local stacks=(); mapfile -t stacks < "$RUN/stacks.txt" 2>/dev/null
    [ "${#stacks[@]}" -gt 0 ] || die "нет $RUN/stacks.txt — шаг R0"
    FRESH=()
    rm -f "$RUN"/probe-*.log   # вывод о памяти — только по пробам этого запуска
    # Обучение 9B — в кванте (решение владельца 2026-09-28): bf16-связки
    # на одной карте не помещаются (vLLM: веса 17 ГБ, первый запуск P1),
    # поэтому по умолчанию пробуется только QLoRA. PROBES="unsloth-4bit unsloth vllm" — все.
    local probe
    for probe in ${PROBES:-unsloth-4bit}; do
        case "$probe" in
            unsloth-4bit) probe_one unsloth-4bit unsloth --load-in-4bit --micro-batch 1 ;;
            unsloth)      probe_one unsloth unsloth ;;
            vllm)         probe_one vllm vllm --backend hf --vllm --vllm-memory 0.3 ;;
        esac
    done
    if [ "${#FRESH[@]}" = 0 ]; then
        # Вывод о второй карте — только если отказ был по памяти: первый запуск
        # P1 упал на ошибке кода, а сценарий объявил нехватку карты.
        if grep -l -i -E "out of memory|outofmemoryerror|less than desired GPU memory" \
                "$RUN"/probe-*.log >/dev/null 2>&1; then
            die "ни одна связка 9B не поместилась на одной карте — нужна вторая карта (см. $RUN/probe-*.log)"
        fi
        die "пробы 9B упали не по памяти — ошибка кода или окружения, см. $RUN/probe-*.log"
    fi
    run "разбор проб (правило записано до замера)" "$(rl_env_dir "${stacks[0]}")/bin/python" \
        scripts/probe_compare.py "${FRESH[@]}"
    done_mark P1
}

step_Z9() {
    say "Z9. Архив"
    local archive metrics
    archive="$HOME/day5-results-$(date +%Y%m%d-%H%M).tar.gz"
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    local paths=(artifacts/runs/day5 artifacts/runs/day4/judge-v2 "$metrics")
    for f in "$SYN_GRAPH" "${SYN_GRAPH%.json.gz}.report.json" "$OUT/groups-9b-sheet.md" \
             "$OUT/groups-9b-key.json" "$OUT/groups-9b.summary.json" "$OUT/train-ru-clean.json"; do
        [ -e "$f" ] && paths+=("${f#"$REPO_DIR"/}")
    done
    [ -d runs ] && paths+=(runs)
    tar czf "$archive" -C "$REPO_DIR" --exclude='runs/*/checkpoint-*' "${paths[@]}" 2>>"$LOG" \
        || die "архив не создан: см. $LOG"
    ok "архив: $archive ($(du -h "$archive" | cut -f1))"
    done_mark Z9
}

for code in $CODES; do
    if should_run "$code"; then
        "step_$code"
    fi
done
say "Готово ($(stamp)). Журнал: $LOG"
