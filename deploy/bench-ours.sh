#!/usr/bin/env bash
# Наша система (старая и текущая конфигурация) и этап 2 на публичном наборе —
# рядом с HippoRAG 2 и обычным RAG из deploy/hipporag-bench.sh, затем ответы 9B.
#
#   bash deploy/bench-ours.sh --list
#   bash deploy/bench-ours.sh                      всё по порядку, с места остановки
#   bash deploy/bench-ours.sh --only A1            один шаг заново
#   bash deploy/bench-ours.sh --until R3           до шага включительно (bench-all.sh)
#   BENCH=musique-300 bash deploy/bench-ours.sh    набор (по умолчанию musique-300)
#
# Запускать внутри tmux ПОСЛЕ hipporag-bench.sh (нужны его выдачи dense и
# hipporag2 — без них шаги S и A сравнивают только наши системы):
#   tmux has-session -t ours 2>/dev/null || tmux new-session -d -s ours \
#       'bash deploy/bench-ours.sh; exec bash'
#
# Протокол — как доказывали авторы статей этапа 2 (docs/HYPOTHESES.md, серия S):
# главная метрика — EM/F1 ответа при k=5 (SetR, SEAL), один генератор на всех
# (Qwen3.5-9B UD-Q4_K_XL на llama.cpp, температура 0), значимость — McNemar
# и парный t-тест с поправкой Холма (SEAL). Метрики поиска — объясняющие:
# P/R@1,3,5, NDCG@10, MRR@10, all@16.
#
# Модели. Граф строит и отбирает (SetR, SEAL) Qwen3.5-4B на SGLang —
# та же, что строила граф HippoRAG 2 и наш граф v4: иначе сравниваем модели.
# Отвечает 9B, и она гасит поиск (Infinity) — поэтому ответы идут последними.
#
# Neo4j Community держит одну базу: граф учебника снимается. Все опыты с дня 3
# идут по файлу artifacts/graphs/v4.json.gz (GRAPH_BACKEND=memory), поэтому
# шаг C0 требует этот файл, а возврат графа в Neo4j (N1) — только по --only.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

BENCH="${BENCH:-musique-300}"
BUNDLE="$REPO_DIR/artifacts/bench/$BENCH"
RUN="$REPO_DIR/artifacts/runs/bench/$BENCH"
OURS="$RUN/ours"
HIPPO="$RUN/hipporag"
ANSWERS="$RUN/answers"
GRAPHS="$REPO_DIR/artifacts/graphs"
BENCH_GRAPHS="$GRAPHS/bench-$BENCH"
TEXTBOOK_GRAPH="$GRAPHS/v4.json.gz"
# Синонимы ответа MuSiQue (answer_aliases) из исходного musique.json HippoRAG 2,
# вынесены рядом с набором (sha256 набора не меняется);
# нет файла — EM/F1 считаются по одному ответу, это пишется в журнал.
ALIASES="${ALIASES:-$REPO_DIR/artifacts/bench/$BENCH-aliases.json}"
LOG="$RUN/bench-ours.log"
# Отметки отдельно от hipporag-bench.sh: у обоих есть шаг M1.
DONE="$RUN/done-ours"
mkdir -p "$DONE" "$OURS" "$ANSWERS" "$BENCH_GRAPHS"

COLLECTION="bench_${BENCH//-/_}"
K_ANSWER=5
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-0}"
# Чистый замер задержки (P1/P2): один запрос за раз, ничего чужого на карте.
CLEAN_N=60
# P1/P2: clean — кэши как есть (вопросы уже прогонялись: извлечение сущностей
# вопроса и эмбеддинги берутся из кэша, это «тёплый» замер); cold — кэши
# эмбеддингов и извлечения выключены, как у нового вопроса в работе.
CLEAN_TAG="${CLEAN_TAG:-clean}"
COLD_ENV=()
[ "$CLEAN_TAG" = cold ] && COLD_ENV=(EMBEDDING_CACHE_ENABLED=false GRAPH_EXTRACTION_CACHE_ENABLED=false)

# Текущая конфигурация — BASE_ENV дня 2 (точка отсчёта серии К), граф файлом.
# Язык BM25 — английский: набор английский. TOP_K=16 — выдача длиннее k ответа,
# отборщикам и метрикам нужно окно; ответ всё равно читает первые K_ANSWER.
CORE_ENV=(QDRANT_COLLECTION="$COLLECTION" QDRANT_SPARSE_LANGUAGE=english
          GRAPH_ENABLED=true GRAPH_RETRIEVAL_ENABLED=true
          GRAPH_RANKER=walk GRAPH_WALK=comention
          GRAPH_EXPANSION_HOPS=1 GRAPH_SEED_ENTITY_LIMIT=20 GRAPH_PASSAGE_LIMIT=30
          GRAPH_MAX_ENTITY_DEGREE=64 GRAPH_HOP_DECAY=0.5 GRAPH_PASSAGE_IDF_ENABLED=false
          GRAPH_SEED_PASSAGES=3 GRAPH_EXPANSION_REL_TYPES=RELATES
          GRAPH_WEIGHT=0.4 GRAPH_EXTRACTION_PROMPT_VERSION=en
          RETRIEVAL_DENSE_CANDIDATES=40 RETRIEVAL_SPARSE_CANDIDATES=40
          RETRIEVAL_ROUTER_ENABLED=true RETRIEVAL_ROUTER_MODE=always
          RETRIEVAL_TOP_K=16 RETRIEVAL_TOP_K_LINKING=16 RETRIEVAL_RRF_K=60
          RETRIEVAL_DEDUP_ENABLED=true RETRIEVAL_DEDUP_SIMILARITY=0.92
          RETRIEVAL_MIN_GRAPH_DOCS=0 RETRIEVAL_GRAPH_CANDIDATE_QUOTA=6
          RETRIEVAL_DIVERSITY_MODE=off RETRIEVAL_SELECTION=off
          RETRIEVAL_QUERY_REWRITE_ENABLED=true RETRIEVAL_DECOMPOSE_ENABLED=false
          RERANKER_ENABLED=true RERANKER_MODE=always RERANKER_BLEND_ALPHA=1.0
          RERANKER_TOP_N=8 RERANKER_CANDIDATES=30
          EVAL_TRACE_POOL=100 EVAL_TRACE_RERANK_ALL=true
          LLM_REASONING_EFFORT=none)
# Построение: кросс-фрагментные связи — как у графа учебника v4 (.env сервера).
CURRENT_BUILD=(GRAPH_CROSS_CHUNK_ENABLED=true GRAPH_CROSS_CHUNK_MAX_ENTITIES=250
               GRAPH_CROSS_CHUNK_MIN_CHUNKS=3)
# «Старая» — основа системы коммита b8a74dc (2026-04-08, код src/) на нашем
# движке. Её собственный стек (Chroma, Ollama, извлечение ~39 с на фрагмент)
# на 4657 фрагментах занял бы сутки карты — по решению владельца 2026-09-30
# берётся основа: извлечение моделью, рёбра совместной встречаемости без
# фильтра, обход RELATES и CO_OCCURS на один шаг, затравки из вопроса,
# гибрид плотный + BM25 через RRF с окнами 8 и 8, граф с весом 0.35, без
# реранкера, без переписывания вопроса, без обрезки хабов.
# Не перенесено: эмбеддер qwen3-embedding 0.6B (здесь bge-m3 у всех систем),
# канал ключевых слов и KET (отвергнуты при переписывании, MIGRATION-PLAN).
OLD_BUILD=(GRAPH_CROSS_CHUNK_ENABLED=false
           GRAPH_COOCCURRENCE_ENABLED=true GRAPH_COOCCURRENCE_MIN_PMI=-1000
           GRAPH_COOCCURRENCE_MIN_COUNT=1)
CURRENT_ENV=("${CORE_ENV[@]}" GRAPH_BACKEND=memory GRAPH_FILE="$BENCH_GRAPHS/current.json.gz"
             GRAPH_SEED_MODE=both)
# Промпт извлечения: основные графы строятся промптом en (решение владельца
# 2026-09-30: русский промпт v4 под учебник математики не находит сущностей
# в 55% английских фрагментов, и сравнение мерило бы промпт, а не граф).
# Тот же текущий граф с v4 — дополнительная строка: цена чужого промпта.
V4_BUILD=("${CURRENT_BUILD[@]}" GRAPH_EXTRACTION_PROMPT_VERSION=v4)
V4_ENV=("${CURRENT_ENV[@]}" GRAPH_FILE="$BENCH_GRAPHS/current-v4.json.gz"
        GRAPH_EXTRACTION_PROMPT_VERSION=v4)
# Переменные справа перекрывают CORE_ENV (env берёт последнее значение).
OLD_ENV=("${CORE_ENV[@]}" GRAPH_BACKEND=memory GRAPH_FILE="$BENCH_GRAPHS/old.json.gz"
         GRAPH_SEED_MODE=query GRAPH_EXPANSION_REL_TYPES=RELATES,CO_OCCURS
         GRAPH_SEED_ENTITY_LIMIT=30 GRAPH_PASSAGE_LIMIT=30 GRAPH_WEIGHT=0.35
         GRAPH_MAX_ENTITY_DEGREE=0 GRAPH_HOP_DECAY=0.5
         RETRIEVAL_DENSE_CANDIDATES=8 RETRIEVAL_SPARSE_CANDIDATES=8
         RETRIEVAL_GRAPH_CANDIDATE_QUOTA=0 RETRIEVAL_DEDUP_ENABLED=false
         RETRIEVAL_QUERY_REWRITE_ENABLED=false RERANKER_ENABLED=false RERANKER_MODE=off)

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

# 9B-Q4 на llama.cpp — как ensure_9b_alone дня 5 (решение владельца 2026-09-22).
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
    run "пробный ответ 9B-Q4" uv run python - <<'PY'
import json, urllib.request
body = {"model": "qwen9b", "max_tokens": 200, "temperature": 0, "messages": [
    {"role": "user", "content": "Who wrote 'War and Peace'? End with 'Answer: <short answer>'."}]}
req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]
text = (msg.get("content") or "").strip()
print("ответ:", text[:300])
raise SystemExit(0 if "Answer:" in text and "Tolstoy" in text and "<think>" not in text else 1)
PY
}

answers() {  # имя шага, база, затем --run …
    local label="$1" base="$2"; shift 2
    local extra=()
    [ -f "$ALIASES" ] && extra=(--aliases "$ALIASES") || warn "нет $ALIASES — EM/F1 без синонимов ответа"
    run "ответы $label" env LLM_MODEL=Qwen3.5-9B-UD-Q4_K_XL LLM_CHAT_MODEL= LLM_REASONING_EFFORT=none \
        LLM_MAX_CONCURRENCY=16 uv run python scripts/bench_answers.py --bundle "$BUNDLE" \
        --k "$K_ANSWER" --baseline "$base" --workers 16 --max-tokens 1536 "${extra[@]}" \
        --out "$ANSWERS" "$@"
    cp "$ANSWERS/answers-report.json" "$RUN/answers-$label.json"
}

# Выдачи HippoRAG 2 годны только после полного прогона (F1): проба пишет
# файлы с теми же именами на 30 вопросах, и без этой проверки они сошли бы
# за полные — в том числе после провала шлюза.
have_hippo() { [ -f "$RUN/done/F1" ] && [ -s "$HIPPO/rankings-hipporag2.jsonl" ]                && [ -s "$HIPPO/rankings-dense.jsonl" ]; }
# Пока идёт HippoRAG 2 (флаг ставит его tmux-сессия), шаги, которым нужны
# его выдачи, ждут. Флаг снимается и при провале — тогда идём без него.
wait_hippo() {
    local waited=0
    while [ -f "$RUN/hippo.running" ]; do
        [ $((waited % 30)) = 0 ] && warn "жду HippoRAG 2 ($waited мин; журнал $RUN/hipporag-bench.out)"
        sleep 60; waited=$((waited + 1))
    done
}

# ---------------------------------------------------------------- шаги
step_C0() {  # проверки: набор цел, файл графа учебника есть, выдачи HippoRAG 2
    [ -f "$BUNDLE/manifest.json" ] || die "нет набора $BUNDLE — он едет в архиве с кодом"
    run "сверка набора" uv run python - "$BUNDLE" <<'PY'
import hashlib, json, sys
from pathlib import Path
b = Path(sys.argv[1]); m = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
for name, want in m["sha256"].items():
    got = hashlib.sha256((b / name).read_bytes()).hexdigest()
    assert got == want, f"{name}: {got} != {want}"
print(f"набор цел: {m['chunks']} фрагментов, {m['questions']} вопросов")
PY
    # Граф учебника в Neo4j будет снят: без его файла опыты дней 3–5 не повторить.
    [ -s "$TEXTBOOK_GRAPH" ] || die "нет $TEXTBOOK_GRAPH — выгрузить граф учебника до его снятия (day2.sh X0)"
    have_hippo && ok "выдачи HippoRAG 2 и dense на месте" \
        || warn "нет выдач HippoRAG 2/dense ($HIPPO) — сначала deploy/hipporag-bench.sh; шаги S и A без них неполны"
}

build_graph() {  # вариант, окружение построения…
    local variant="$1"; shift
    ensure_4b
    run "граф снят" env "${CORE_ENV[@]}" GRAPH_BACKEND=neo4j uv run rag-textbook graph drop --yes
    run "индекс и граф $variant" env "${CORE_ENV[@]}" GRAPH_BACKEND=neo4j "$@" LLM_MAX_CONCURRENCY=32 \
        uv run python scripts/bench_ours.py index --bundle "$BUNDLE" --graph \
        --report "$OURS/index-$variant.json"
    run "статистика графа $variant" env "${CORE_ENV[@]}" GRAPH_BACKEND=neo4j uv run rag-textbook graph stats
    run "выгрузка графа $variant" env "${CORE_ENV[@]}" GRAPH_BACKEND=neo4j \
        uv run python scripts/graph_export.py --out "$BENCH_GRAPHS/$variant.json.gz" --variant "bench-$variant"
}

step_I1() {  # текущий граф: векторы, BM25 и граф с кросс-фрагментными связями
    build_graph current "${CURRENT_BUILD[@]}"
}

step_I2() {  # старый граф: извлечение из кэша I1, без кросс-фрагментных связей
    build_graph old "${OLD_BUILD[@]}"
}

step_I3() {  # текущий граф с промптом v4 (учебник) — дополнительная строка
    build_graph current-v4 "${V4_BUILD[@]}"
}

rank() {  # система, окружение…
    local system="$1"; shift
    ensure_4b
    run "выдача $system" env "$@" uv run python scripts/bench_ours.py rank --bundle "$BUNDLE" \
        --out "$OURS" --system "$system" --workers "$WORKERS" --limit "$LIMIT"
}

step_R1() { rank ours-current "${CURRENT_ENV[@]}"; }
step_R2() { rank ours-old "${OLD_ENV[@]}"; }

step_R3() {  # SEAL-RAG: буфер фиксированной ёмкости k=5, как в статье
    rank ours-current+seal "${CURRENT_ENV[@]}" RETRIEVAL_SELECTION=seal \
        RETRIEVAL_TOP_K="$K_ANSWER" RETRIEVAL_TOP_K_LINKING="$K_ANSWER"
}

step_R4() { rank ours-current-v4 "${V4_ENV[@]}"; }

# Context-Picker — метод обучения (GRPO): без него это лишь подсказка, и ставить
# её рядом со статьёй несправедливо. В прогон войдёт после обучения (S3).
step_S1() {  # SetR по окну 20 — поверх нашей выдачи и HippoRAG 2
    wait_hippo
    local sources=("$OURS/rankings-ours-current.jsonl")
    have_hippo && sources+=("$HIPPO/rankings-hipporag2.jsonl")
    ensure_4b
    for src in "${sources[@]}"; do
        for mode in setr; do
            run "отбор $mode: $(basename "$src")" env LLM_REASONING_EFFORT=none \
                uv run python scripts/bench_select.py --bundle "$BUNDLE" --rankings "$src" \
                --mode "$mode" --pool 20 --top-k 16 --max-tokens 4096 --workers "$WORKERS"
        done
    done
}

step_M1() {  # метрики поиска: all/recall@k и бутстрап разности (объясняющие)
    wait_hippo
    local runs=(--run ours-old="$OURS/rankings-ours-old.jsonl"
                --run ours-current="$OURS/rankings-ours-current.jsonl"
                --run ours-current+seal="$OURS/rankings-ours-current+seal.jsonl"
                --run ours-current-v4="$OURS/rankings-ours-current-v4.jsonl")
    have_hippo && runs+=(--run dense="$HIPPO/rankings-dense.jsonl"
                         --run hipporag2="$HIPPO/rankings-hipporag2.jsonl")
    local pair=ours-old,ours-current
    have_hippo && pair=hipporag2,ours-current
    run "метрики поиска" uv run python scripts/bench_metrics.py --bundle "$BUNDLE" "${runs[@]}" \
        --k 1,3,5,10,16 --pair "$pair" --out "$RUN/metrics-all.json"
}

step_A1() {  # этап 1: ответы 9B при k=5, база — обычный RAG
    wait_hippo
    ensure_9b_alone
    local runs=(--run ours-old="$OURS/rankings-ours-old.jsonl"
                --run ours-current="$OURS/rankings-ours-current.jsonl"
                --run ours-current-v4="$OURS/rankings-ours-current-v4.jsonl")
    local base=ours-old
    if have_hippo; then
        runs=(--run dense="$HIPPO/rankings-dense.jsonl" --run hipporag2="$HIPPO/rankings-hipporag2.jsonl"
              "${runs[@]}")
        base=dense
    fi
    answers stage1 "$base" "${runs[@]}"
}

step_A2() {  # этап 2: отборщики против своей базы — нашей и HippoRAG 2
    ensure_9b_alone
    answers stage2-ours ours-current \
        --run ours-current="$OURS/rankings-ours-current.jsonl" \
        --run ours-current+setr="$OURS/rankings-ours-current+setr.jsonl:selected" \
        --run ours-current+seal="$OURS/rankings-ours-current+seal.jsonl"
    if have_hippo; then
        answers stage2-hippo hipporag2 \
            --run hipporag2="$HIPPO/rankings-hipporag2.jsonl" \
            --run hipporag2+setr="$HIPPO/rankings-hipporag2+setr.jsonl:selected"
    fi
}

# Задержка S4 (×8.5) снята, пока рядом шёл HippoRAG 2 и 8 запросов разом.
# Перезамер: первые CLEAN_N вопросов по одному, на карте только 4B и Infinity.
# Выдачи пишутся под новыми именами (@clean) и сверяются с прежними.
step_P1() {  # (только --only) чистая задержка поиска: ours-current и ours-current+seal
    local others
    others=$(tmux ls -F '#S' 2>/dev/null | grep -vx "${CLEAN_SESSION:-laya}" | tr '\n' ' ')
    [ -z "$others" ] || warn "есть другие сессии tmux ($others) — проверяю, что в них ничего не считается"
    pgrep -af "[b]ench_ours|[b]ench_answers|[b]ench_select|[h]ipporag|[l]aya_train|[l]aya_eval|[r]ag-textbook ingest" \
        && die "идёт другой расчёт (список выше) — замер не будет чистым"
    ensure_4b
    ok "на карте: $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
    curl -sf -o /dev/null "http://127.0.0.1:6333/collections/$COLLECTION" \
        || die "нет коллекции $COLLECTION в Qdrant — выдачи не повторить"
    WORKERS=1 LIMIT=$CLEAN_N rank "ours-current@$CLEAN_TAG" "${CURRENT_ENV[@]}" "${COLD_ENV[@]}"
    WORKERS=1 LIMIT=$CLEAN_N rank "ours-current+seal@$CLEAN_TAG" "${CURRENT_ENV[@]}" "${COLD_ENV[@]}" RETRIEVAL_SELECTION=seal \
        RETRIEVAL_TOP_K="$K_ANSWER" RETRIEVAL_TOP_K_LINKING="$K_ANSWER"
    run "сводка задержки" uv run python - "$OURS" "$CLEAN_TAG" <<'PY'
import json, statistics as st, sys
from pathlib import Path
ours, tag = Path(sys.argv[1]), sys.argv[2]
def rows(name):
    with (ours / f"rankings-{name}.jsonl").open(encoding="utf-8") as handle:
        return {r["qid"]: r for r in map(json.loads, handle)}
report = {}
for name in ("ours-current", "ours-current+seal"):
    clean, old = rows(f"{name}@{tag}"), rows(name)
    lat = [r["latency_ms"] / 1000 for r in clean.values()]
    old_lat = [old[q]["latency_ms"] / 1000 for q in clean]
    same = sum(clean[q]["ranked"][:5] == old[q]["ranked"][:5] for q in clean) / len(clean)
    report[name] = {"n": len(clean), "median_s": round(st.median(lat), 2), "mean_s": round(st.fmean(lat), 2),
                    "p90_s": round(sorted(lat)[int(0.9 * len(lat))], 2),
                    "old_median_s": round(st.median(old_lat), 2), "same_top5": round(same, 3)}
    print(name, report[name])
ratio = report["ours-current+seal"]["median_s"] / report["ours-current"]["median_s"]
report["seal_over_base_median"] = round(ratio, 2)
print(f"SEAL / база по медиане: ×{ratio:.2f} (было ×8.5 под нагрузкой)")
report["tag"] = tag
(ours.parent / f"latency-{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

step_P2() {  # (только --only) время ответа 9B-Q4 при k=5, по одному, на тех же CLEAN_N вопросах
    [ -s "$OURS/rankings-ours-current+seal@$CLEAN_TAG.jsonl" ] || die "нет выдач P1 (@$CLEAN_TAG) — сначала --only P1"
    ensure_9b_alone
    local out="$RUN/answers-$CLEAN_TAG"
    run "ответы по одному" env LLM_MODEL=Qwen3.5-9B-UD-Q4_K_XL LLM_CHAT_MODEL= LLM_REASONING_EFFORT=none \
        LLM_MAX_CONCURRENCY=1 uv run python scripts/bench_answers.py --bundle "$BUNDLE" \
        --k "$K_ANSWER" --baseline ours-current --workers 1 --max-tokens 1536 --limit "$CLEAN_N" \
        --out "$out" \
        --run ours-current="$OURS/rankings-ours-current@$CLEAN_TAG.jsonl" \
        --run ours-current+seal="$OURS/rankings-ours-current+seal@$CLEAN_TAG.jsonl"
    run "сводка полного пути" uv run python - "$out" "$RUN/latency-$CLEAN_TAG.json" <<'PY'
import json, statistics as st, sys
from pathlib import Path
out, path = Path(sys.argv[1]), Path(sys.argv[2])
report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
for name in ("ours-current", "ours-current+seal"):
    with (out / f"answers-{name}.jsonl").open(encoding="utf-8") as handle:
        ms = [r.get("ms", 0) / 1000 for r in map(json.loads, handle)]
    entry = report.setdefault(name, {})
    entry["answer_median_s"] = round(st.median(ms), 2)
    if "median_s" in entry:
        entry["end_to_end_median_s"] = round(entry["median_s"] + entry["answer_median_s"], 2)
    print(name, entry)
if all("end_to_end_median_s" in report.get(n, {}) for n in ("ours-current", "ours-current+seal")):
    ratio = report["ours-current+seal"]["end_to_end_median_s"] / report["ours-current"]["end_to_end_median_s"]
    report["seal_over_base_end_to_end"] = round(ratio, 2)
    print(f"полный путь SEAL / база: ×{ratio:.2f}")
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

step_N1() {  # граф учебника обратно в Neo4j (по --only): пересборка из кэша извлечения
    ensure_4b
    run "граф снят" uv run rag-textbook graph drop --yes
    local env=(QDRANT_COLLECTION=library_ru QDRANT_SPARSE_LANGUAGE=russian CHUNKER_RESPECT_FORMULAS=true
               CHUNK_SIZE=1200 CHUNK_OVERLAP=180 GRAPH_ENABLED=true GRAPH_EXTRACTION_PROMPT_VERSION=v4
               LLM_REASONING_EFFORT=none MINERU_LANG=east_slavic)
    run "граф pdf_docs" env "${env[@]}" PDF_DIR=documents/pdf_docs MINERU_METHOD=auto \
        uv run rag-textbook ingest --stages graph --no-monitor
    run "граф ru" env "${env[@]}" PDF_DIR=documents/library/ru MINERU_METHOD=auto \
        uv run rag-textbook ingest --stages graph --no-monitor
    run "граф ru-ocr" env "${env[@]}" PDF_DIR=documents/library/ru-ocr MINERU_METHOD=ocr \
        uv run rag-textbook ingest --stages graph --no-monitor
    run "статистика графа" uv run rag-textbook graph stats
}

STEPS=(C0 I1 I2 R1 R2 R3 I3 R4 S1 M1 A1 A2)
declare -A DESC=(
    [C0]="набор цел, файл графа учебника есть, выдачи HippoRAG 2 на месте"
    [I1]="коллекция $COLLECTION и текущий граф (кросс-фрагментные связи, промпт en) → файл"
    [I2]="граф основы b8a74dc (встречаемость без фильтра, извлечение из кэша, en) → файл"
    [R1]="выдача ours-current (затравки both)"
    [R2]="выдача ours-old (основа b8a74dc: без реранкера и переписывания)"
    [R3]="выдача ours-current+seal (k=$K_ANSWER)"
    [I3]="текущий граф с промптом v4 (учебник) → файл, дополнительная строка"
    [R4]="выдача ours-current-v4"
    [S1]="SetR по окну 20: ours-current и hipporag2"
    [M1]="метрики поиска всех выдач"
    [A1]="ответы 9B-Q4 при k=$K_ANSWER: dense, hipporag2, ours-old, ours-current, ours-current-v4"
    [A2]="ответы 9B-Q4: отборщики против своей базы"
    [N1]="(только --only) граф учебника обратно в Neo4j из кэша"
    [P1]="(только --only) чистая задержка поиска: база и SEAL, по одному, $CLEAN_N вопросов"
    [P2]="(только --only) время ответа 9B-Q4 по одному на тех же вопросах"
)
# C0 — проверка: выполняется при каждом запуске, отметку не ставит.
ALWAYS=" C0 "

only="" from="" until=""
case "${1:-}" in
    --until) until="$2" ;;
    --list) for s in "${STEPS[@]}" N1 P1 P2; do printf '%s  %s\n' "$s" "${DESC[$s]}"; done; exit 0 ;;
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
        [ "$s" = "$until" ] && exit 0
        continue
    fi
    say "$s — ${DESC[$s]} ($(date '+%H:%M:%S'))"
    "step_$s"
    [[ "$ALWAYS" == *" $s "* ]] || touch "$DONE/$s"
    [ "$s" = "$until" ] && { ok "остановка после $until"; exit 0; }
done
ok "готово: $RUN"
