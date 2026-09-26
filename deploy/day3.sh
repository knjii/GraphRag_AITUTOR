#!/usr/bin/env bash
# День аренды №3: серия К онлайн (как строим граф, как отбираем), ответы
# финалистов, судья 27B. План и критерии — docs/SERVER-DAY-2.md, раздел
# «День 3». Каждый шаг отмечается в artifacts/runs/day3/done, повторный
# запуск продолжает с несделанного.
#
#   bash deploy/day3.sh --list          шаги
#   bash deploy/day3.sh                 всё по порядку
#   bash deploy/day3.sh --only K4       один шаг заново
#   bash deploy/day3.sh --from V0       начиная с шага
#
# Предусловия (проверяет E0): день 2 прошёл до конца, стенд graph_fidelity
# годен (artifacts/runs/day2/fidelity.ok), эталон v2 принят владельцем
# (evaluation/goldsets/goldset-v2.accepted — sha256 набора после ручных
# правок). Без приёмки серия меряла бы разметку, а не граф
# (goldset-must-be-accepted-not-grown).
#
# Все варианты идут на файлах графа (GRAPH_BACKEND=memory): режимы отбора
# К6–К8 и обход по зависимостям работают только там, а базовая линия —
# тот же v4 файлом, поэтому различие между прогонами — ровно в варианте.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

RUN="$REPO_DIR/artifacts/runs/day3"
RUN2="$REPO_DIR/artifacts/runs/day2"
GRAPHS="$REPO_DIR/artifacts/graphs"
LOG="$RUN/day3.log"

GOLD_V2=evaluation/goldsets/goldset-v2.json
GOLD_V2_ACCEPTED=evaluation/goldsets/goldset-v2.accepted
COLLECTION=library_ru
PROMPT_FILE=deploy/prompts/qa-v4.txt
JUDGE_DIR=qwen27b            # каталог в томе rag-textbook_gguf_models (models-fetch.sh)
JUDGE_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

# Состав — как в day2.sh (слепок 2026-08-19), только граф файлом.
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
# WARNING не вырезаем: так пропадали бы отказы графа, PPR и судьи (ревизия 031).
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

ensure_4b() {
    if grep -q '^LLM_MODEL=Qwen/Qwen3.5-4B$' .env 2>/dev/null \
        && curl -sf http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q 'Qwen3.5-4B' \
        && curl -sf -o /dev/null http://127.0.0.1:7997/health \
        && ! docker ps --format '{{.Names}}' | grep -qxE 'judge|generator'; then
        ok "4B и поиск уже подняты"
    else
        # Порт 8001 делят SGLang и llama.cpp (судья, 9B дня 2): /health
        # ответит и чужая модель, поэтому сверяем имя и снимаем оба контейнера.
        docker rm -f judge generator >/dev/null 2>&1
        run "переключение на 4B" bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval
        run "запуск служб" bash deploy/services.sh up
    fi
}
metrics_dir() { uv run python -c 'from rag_textbook.config import Settings; print(Settings().paths.metrics_dir)'; }
# Файл последнего прогона с меткой: имена — retrieval_eval_<метка>_<время>.json.
run_file() {
    local metrics file
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    file=$(ls -t "$metrics"/retrieval_eval_"$1"_*.json 2>/dev/null | head -1)
    [ -n "$file" ] || die "нет прогона с меткой $1 — сначала его шаг"
    printf '%s' "$file"
}

# ------------------------------------------------------------------- план
PLAN=$(cat <<'PLAN'
E0|Проверка: день 2 завершён, стенд годен, графы серии К на месте, эталон v2 принят; фоном — веса 27B
S0|Базовая линия на goldset-v2: граф v4 файлом, обход по совместному упоминанию
K1|К1: структурный граф (A) и объединение со структурным (C)
K3|К3: обозначения на всю книгу и на раздел
K2|К2: обход по зависимостям «использует → определяет» (граф v4 с ролями)
K4|К4: PPR против обхода на лучшем графе (выбор — на части dev, правило ниже)
K6|К6: условный отбор, жадный (a) и парами (b)
K7|К7: распространение балла по зависимостям внутри пула
K8|К8: замыкание контекста местами определений
V0|Вердикт серии К: test принимает, dev — только выбор варианта
A0|Ответы 4B по слепкам: базовая линия и принятые варианты (без судьи)
J0|Судья 27B (llama.cpp): сверка с ручными оценками, затем оценка ответов
Z9|Сводка и архив для скачивания
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
# Каталоги — после --list: список шагов не должен ничего создавать.
mkdir -p "$RUN/done" "$RUN/answers" "$RUN/judged"
# Окружение — один раз и с MinerU. Голый `uv run` приводит venv к pyproject
# без extra и снимает parsing: L1 2026-09-22 упал на «No module named
# mineru». Дальше `uv run` окружение не трогает.
uv sync --extra parsing --extra dev -q || die "uv sync не прошёл"
export UV_NO_SYNC=1
for chosen in "$ONLY" "$FROM"; do
    [ -z "$chosen" ] && continue
    printf '%s\n' "$CODES" | grep -qx "$chosen" || die "нет шага $chosen; список — --list"
done

STARTED=$([ -z "$FROM" ] && echo 1 || echo 0)
# Явный --from X — повтор X и всего после него. Без этого отметка done/X
# молча пропускала сам X (день 2: P0 с прежней квотой).
if [ -n "$FROM" ]; then
    seen=0
    for code in $CODES; do
        [ "$code" = "$FROM" ] && seen=1
        [ "$seen" = 1 ] && rm -f "$RUN/done/$code"
    done
fi
should_run() {
    local code="$1"
    [ -n "$ONLY" ] && { [ "$code" = "$ONLY" ]; return; }
    [ "$STARTED" = 0 ] && [ "$code" = "$FROM" ] && STARTED=1
    [ "$STARTED" = 1 ] || return 1
    [ -f "$RUN/done/$code" ] && { ok "$code уже выполнен"; return 1; }
    return 0
}
done_mark() { date '+%F %T' > "$RUN/done/$1"; ok "$1 готов ($(stamp))"; }
# Базовая линия — опора всех сравнений: её повтор снимает отметки
# вердикта, ответов и судьи. Прогоны вариантов независимы друг от друга.
invalidate_verdict() {
    local code
    for code in V0 A0 J0 Z9; do
        [ -f "$RUN/done/$code" ] && { rm -f "$RUN/done/$code"; warn "$code: отметка снята — прогоны изменились"; }
    done
}

# Один вариант: прогон по эталону v2 и слепок (из него потом ответы).
# Слепок не меняет метрик прогона (capture.sh), поэтому снимается всегда.
variant() {
    local name="$1"; shift
    say "  $name"
    run "прогон $name" env "${BASE_ENV[@]}" "$@" \
        uv run rag-textbook eval run --goldset "$GOLD_V2" --label "k-$name" \
        --trace "$RUN/trace-$name.jsonl"
    invalidate_verdict
    # Выбор К4 читает base, K1 и K3: их повтор снимает и К4.
    case "$name" in
        base|K1*|K3*) [ -f "$RUN/done/K4" ] && { rm -f "$RUN/done/K4"; warn "K4: отметка снята — изменился вход выбора"; } ;;
    esac
}

# ---------------------------------------------------------------- шаги
step_E0() {
    say "E0. Проверка перед стартом"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
    df -h "$REPO_DIR" | tail -1 | tee -a "$LOG"
    local code
    for code in L2 G1 B0 X0 Q0 K0; do
        [ -f "$RUN2/done/$code" ] || die "день 2 не завершён: нет отметки $code (bash deploy/day2.sh)"
    done
    # Допуск стенда записан до замера: без него серия на файлах не ставится
    # (graph-file-stand-2026-09-22, offline-conclusions-need-admission).
    [ -f "$RUN2/fidelity.ok" ] || die "стенд graph_fidelity не годен — серию К на файлах не ставить (day2.sh --only X0)"
    for required in "$GRAPHS/v4.json.gz" "$GRAPHS/k1a.json.gz" "$GRAPHS/k1c.json.gz" \
                    "$GRAPHS/k3-book.json.gz" "$GRAPHS/k3-sec.json.gz" "$GOLD_V2" "$PROMPT_FILE"; do
        [ -s "$required" ] || die "нет или пуст $required"
    done
    [ -f "$GOLD_V2_ACCEPTED" ] || die "эталон v2 не принят: нет $GOLD_V2_ACCEPTED (docs/SERVER-DAY-2.md, «Между днями»)"
    sha256sum -c --quiet "$GOLD_V2_ACCEPTED" || die "goldset-v2.json отличается от принятого владельцем"
    ok "эталон v2 принят: $(cut -c1-12 "$GOLD_V2_ACCEPTED")…"
    run "срезы эталона" uv run python - "$GOLD_V2" <<'PY'
import sys
from collections import Counter
from pathlib import Path
from rag_textbook.evaluation.goldset import load_goldset

questions = load_goldset(Path(sys.argv[1]))
cells = Counter((q.slice or q.question_type, q.split) for q in questions)
print(f"вопросов {len(questions)}; срез × часть:", dict(sorted(cells.items())))
test_linking = sum(1 for q in questions if q.split == "test" and q.slice in ("linking", "cross_book"))
# Было 60. Владелец 2026-09-25 снизил до 44: столько связывающих в test после
# фильтрации голосованием, и больше взять негде. Минимальный различимый эффект
# серии вырос примерно в sqrt(60/44) ≈ 1.17 раза — это учитывать в вердикте.
raise SystemExit(0 if test_linking >= 44 else 1)
PY
    # 27B нужна только к J0 — качаем фоном, пока идут прогоны.
    (bash deploy/models-fetch.sh "$JUDGE_DIR") > "$RUN/weights-27b.log" 2>&1 &
    echo $! > "$RUN/weights.pid"
    ok "веса 27B качаются фоном: $RUN/weights-27b.log"
    ensure_4b
    done_mark E0
}

step_S0() {
    say "S0. Базовая линия"
    ensure_4b
    variant base GRAPH_FILE="$GRAPHS/v4.json.gz"
    done_mark S0
}

step_K1() {
    say "K1. Структурный граф"
    ensure_4b
    variant K1A GRAPH_FILE="$GRAPHS/k1a.json.gz"
    variant K1C GRAPH_FILE="$GRAPHS/k1c.json.gz"
    done_mark K1
}

step_K3() {
    say "K3. Обозначения"
    ensure_4b
    variant K3book GRAPH_FILE="$GRAPHS/k3-book.json.gz"
    variant K3sec  GRAPH_FILE="$GRAPHS/k3-sec.json.gz"
    done_mark K3
}

step_K2() {
    say "K2. Обход по зависимостям"
    ensure_4b
    variant K2 GRAPH_FILE="$GRAPHS/v4.json.gz" GRAPH_WALK=dependency
    done_mark K2
}

step_K4() {
    say "K4. PPR против обхода"
    ensure_4b
    # Правило выбора записано до прогонов: граф с наибольшим recall@16
    # на связывающих вопросах части dev среди base, K1A, K1C, K3book,
    # K3sec; при равенстве до третьего знака — первый в этом списке
    # (более простой). Часть test в выборе не участвует.
    local choice
    choice=$(uv run python - "$GOLD_V2" \
        "base=$(run_file k-base)" "K1A=$(run_file k-K1A)" "K1C=$(run_file k-K1C)" \
        "K3book=$(run_file k-K3book)" "K3sec=$(run_file k-K3sec)" 2>>"$LOG" <<'PY'
import sys
from pathlib import Path
from rag_textbook.evaluation.goldset import load_goldset
from rag_textbook.evaluation.metrics import recall_at_k
from rag_textbook.evaluation.runner import load_outcomes

questions = load_goldset(Path(sys.argv[1]))
linking_types = {"graph_linked", "multi_hop", "relation"}
ids = {q.id for q in questions if q.split == "dev"
       and (q.slice in ("linking", "cross_book") or q.question_type in linking_types)}
best, best_value = None, -1.0
for item in sys.argv[2:]:
    name, _, path = item.partition("=")
    outcomes = [o for o in load_outcomes(Path(path))[1] if o.question_id in ids]
    value = round(sum(recall_at_k(o.retrieved, o.relevant, 16) for o in outcomes) / max(1, len(outcomes)), 3)
    print(f"{name}: recall@16 dev-linking {value} (n={len(outcomes)})", file=sys.stderr)
    # Пустая выборка дала бы нули у всех и молча выбрала бы base.
    # Было 20. Владелец 2026-09-25 снизил до 17: столько связывающих в dev после
    # фильтрации голосованием. Выбор по 17 шумный, ничья до третьего знака
    # отдаёт более простой граф.
    if len(outcomes) < 17:
        sys.exit(f"{name}: связывающих dev в прогоне {len(outcomes)} < 17")
    if value > best_value:
        best, best_value = name, value
print(best)
PY
    ) || die "выбор графа для К4 не удался (журнал: $LOG)"
    tail -5 "$LOG"
    local graph
    case "$choice" in
        base)   graph=v4 ;;
        K1A)    graph=k1a ;;
        K1C)    graph=k1c ;;
        K3book) graph=k3-book ;;
        K3sec)  graph=k3-sec ;;
        *) die "неожиданный выбор К4: '$choice'" ;;
    esac
    printf '%s\n' "$choice" > "$RUN/k4-choice.txt"
    ok "К4 на графе $choice ($graph.json.gz); обход — это прогон k-$choice"
    variant K4ppr GRAPH_FILE="$GRAPHS/$graph.json.gz" GRAPH_RANKER=ppr
    done_mark K4
}

step_K6() {
    say "K6. Условный отбор"
    ensure_4b
    variant K6a GRAPH_FILE="$GRAPHS/v4.json.gz" RETRIEVAL_SELECTION=conditional
    variant K6b GRAPH_FILE="$GRAPHS/v4.json.gz" RETRIEVAL_SELECTION=pairs
    done_mark K6
}

step_K7() {
    say "K7. Распространение по зависимостям"
    ensure_4b
    variant K7 GRAPH_FILE="$GRAPHS/v4.json.gz" RETRIEVAL_SELECTION=diffusion \
        RETRIEVAL_SELECTION_LINKS=dependency
    done_mark K7
}

step_K8() {
    say "K8. Замыкание определениями"
    ensure_4b
    variant K8 GRAPH_FILE="$GRAPHS/v4.json.gz" RETRIEVAL_SELECTION=closure \
        RETRIEVAL_SELECTION_LINKS=dependency
    done_mark K8
}

verdict_args() {
    local name choice
    printf -- '--run base=%s ' "$(run_file k-base)"
    for name in K1A K1C K3book K3sec K2 K6a K6b K7 K8; do
        printf -- '--run %s=%s ' "$name" "$(run_file "k-$name")"
    done
    choice=$(cat "$RUN/k4-choice.txt")
    printf -- '--run K4walk=%s --run K4ppr=%s' "$(run_file "k-$choice")" "$(run_file k-K4ppr)"
}

step_V0() {
    say "V0. Вердикт серии К"
    # Новый вердикт меняет состав финалистов: ответы и судья — заново.
    rm -f "$RUN/done/A0" "$RUN/done/J0" "$RUN/done/Z9"
    [ -f "$RUN/k4-choice.txt" ] || die "нет выбора К4 — сначала шаг K4"
    local args
    args=$(verdict_args) || die "не все прогоны серии на месте"
    # shellcheck disable=SC2086  # имена файлов без пробелов, собраны выше
    run "вердикт на test" uv run python scripts/series_k_verdict.py --goldset "$GOLD_V2" \
        --split test $args --json "$RUN/verdict-test.json"
    # shellcheck disable=SC2086
    run "картина на dev" uv run python scripts/series_k_verdict.py --goldset "$GOLD_V2" \
        --split dev $args --json "$RUN/verdict-dev.json"
    done_mark V0
}

step_A0() {
    say "A0. Ответы финалистов"
    ensure_4b
    [ -f "$RUN/verdict-test.json" ] || die "нет вердикта — сначала V0"
    # Ответы — только для базы и вариантов, прошедших все свои проверки
    # на test: ответ стоит полчаса карты, а у отвергнутых он ничего не решает.
    local finalists
    finalists=$(uv run python - "$RUN/verdict-test.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))["tests"]
by_candidate = {}
for row in rows:
    by_candidate.setdefault(row["candidate"], []).append(row["status"])
passed = [c for c, statuses in by_candidate.items() if all(s == "выполнено" for s in statuses)]
print(" ".join(["base", *passed]))
PY
    ) || die "не прочитан вердикт"
    ok "финалисты: $finalists"
    # Список пишется заново: иначе J0 судил бы и финалистов прежнего вердикта.
    : > "$RUN/answers/traces.tsv"
    local name trace
    for name in $finalists; do
        trace="$RUN/trace-$name.jsonl"
        [ "$name" = K4walk ] && trace="$RUN/trace-$(cat "$RUN/k4-choice.txt").jsonl"
        [ -s "$trace" ] || die "нет слепка $trace"
        # Судья — отдельной моделью в J0, поэтому --no-judge. Промпт — v4,
        # победитель по формулам (prompt-beats-window-for-formulas).
        run "ответы $name" env "${CORPUS_ENV[@]}" QA_SYSTEM_PROMPT="$(cat "$PROMPT_FILE")" \
            PROMPT_VERSION=v4 uv run rag-textbook eval answers --goldset "$GOLD_V2" \
            --from-trace "$trace" --no-judge --label "k-$name"
        cp "$(metrics_dir)/answers_k-$name.json" "$RUN/answers/" || die "нет файла ответов k-$name"
        printf '%s\t%s\n' "$name" "$trace" >> "$RUN/answers/traces.tsv"
    done
    # Прочитать три ответа глазами до судьи (read-generations-before-measuring).
    run "три ответа базы" uv run python - "$RUN/answers/answers_k-base.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for row in data["outcomes"][:3]:
    print(f"\n--- {row['question_id']}\n{str(row.get('answer', ''))[:600]}")
print(json.dumps(data["summary"].get("всего", {}), ensure_ascii=False, indent=2))
PY
    rm -f "$RUN/done/J0" "$RUN/done/Z9"
    done_mark A0
}

step_J0() {
    say "J0. Судья 27B"
    rm -f "$RUN/done/Z9"  # новые оценки — новый архив
    [ -f "$RUN/answers/traces.tsv" ] || die "нет ответов — сначала A0"
    local volume path
    volume=$(docker volume inspect rag-textbook_gguf_models --format '{{.Mountpoint}}' 2>/dev/null) \
        || die "нет тома rag-textbook_gguf_models"
    if [ -f "$RUN/weights.pid" ] && kill -0 "$(cat "$RUN/weights.pid")" 2>/dev/null; then
        warn "веса 27B ещё качаются — жду"
        while kill -0 "$(cat "$RUN/weights.pid")" 2>/dev/null; do sleep 20; done
    fi
    path=$(find "$volume/$JUDGE_DIR" -name '*.gguf' -printf '%f\n' 2>/dev/null | sort | head -1)
    [ -n "$path" ] || die "нет весов 27B в $volume/$JUDGE_DIR (bash deploy/models-fetch.sh $JUDGE_DIR)"
    # 27B в четырёх битах — 17 ГБ: рядом ни 4B, ни поиск не помещаются.
    docker compose --env-file .env -f docker/docker-compose.yml stop infinity ollama >/dev/null 2>&1
    docker rm -f rag-textbook-sglang-1 generator judge >/dev/null 2>&1
    # Размышление гасится на стороне движка: клиент chat_template_kwargs
    # не передаёт (задача 029), а шаблон Qwen слушает enable_thinking.
    # Даже если не погасится, llama.cpp кладёт его в reasoning_content,
    # а предел 2048 не даст ему съесть ответ (llamacpp-reasoning-controls).
    run "запуск судьи" docker run -d --name judge --gpus all \
        -v rag-textbook_gguf_models:/models -p 127.0.0.1:8001:8000 "$JUDGE_IMAGE" \
        -m "/models/$JUDGE_DIR/$path" --host 0.0.0.0 --port 8000 \
        -c $((16384 * 2)) -np 2 -ngl 999 --jinja --reasoning off --reasoning-effort minimal \
        --chat-template-kwargs '{"enable_thinking": false}'
    local ready=0
    for _ in $(seq 1 90); do
        curl -sf -o /dev/null http://127.0.0.1:8001/health && { ready=1; break; }
        sleep 10
    done
    [ "$ready" = 1 ] || { docker logs --tail 25 judge 2>&1 | tee -a "$LOG"; die "судья не поднялся за 15 минут"; }
    ok "судья готов: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    local judge_env=(LLM_BASE_URL=http://127.0.0.1:8001/v1 LLM_JUDGE_BASE_URL=http://127.0.0.1:8001/v1
                     LLM_MODEL="$JUDGE_DIR" LLM_JUDGE_MODEL="$JUDGE_DIR" LLM_MAX_TOKENS=2048)
    # Допуск записан до запуска (scripts/judge_saved.py): согласие пар
    # внутри вопроса ≥ 0.75 при нижней границе > 0.5 по ручным оценкам
    # 2026-09-17. Не допущен — оценки ответов не снимаются вовсе.
    rm -rf "$RUN/judged/calibration"
    try "сверка судьи с человеком" env "${judge_env[@]}" uv run python scripts/judge_saved.py \
        --calibrate --goldset capture/goldset.json --trace capture/session-0819/trace-always.jsonl \
        --chunks capture/0690bb81b7e3c831_chunks.json --output-dir "$RUN/judged/calibration" --workers 2
    if [ "$LAST_OK" != 1 ]; then
        warn "СУДЬЯ НЕ ДОПУЩЕН: ответы не оцениваются; вывод по серии — только recall и объективные меры"
        docker rm -f judge >/dev/null 2>&1
        done_mark J0
        return
    fi
    local name trace
    while IFS=$'\t' read -r name trace; do
        rm -f "$RUN/judged/answers_k-${name}_judged.json"
        run "судья: $name" env "${judge_env[@]}" uv run python scripts/judge_saved.py \
            "$RUN/answers/answers_k-$name.json" --goldset "$GOLD_V2" --trace "$trace" \
            --chunks artifacts/parsed --output-dir "$RUN/judged" --workers 2
    done < "$RUN/answers/traces.tsv"
    docker rm -f judge >/dev/null 2>&1
    done_mark J0
}

step_Z9() {
    say "Z9. Архив"
    local archive metrics
    archive="$HOME/day3-results-$(date +%Y%m%d-%H%M).tar.gz"
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    local paths=(artifacts/runs/day3)
    [ -e "$metrics" ] && paths+=("$metrics")
    tar czf "$archive" -C "$REPO_DIR" "${paths[@]}" 2>>"$LOG" || die "архив не создан: см. $LOG"
    ok "архив: $archive ($(du -h "$archive" | cut -f1))"
    warn "скачать: scp -i <ключ> root@<ip>:$archive ."
    warn "затем сервер можно выключать"
    done_mark Z9
}

for code in $CODES; do
    if should_run "$code"; then
        "step_$code"
    fi
done
say "Готово ($(stamp)). Журнал: $LOG"
