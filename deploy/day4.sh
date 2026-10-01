#!/usr/bin/env bash
# День аренды №4: межкнижный эталон (К10) и судья v2 (M6). Критерии записаны
# до замера в docs/HYPOTHESES.md (К10, M6). Каждый шаг отмечается
# в artifacts/runs/day4/done, повторный запуск продолжает с несделанного.
#
#   bash deploy/day4.sh --list          шаги
#   bash deploy/day4.sh                 X0…XV, затем J1 (across, редакция r0)
#   JUDGE_REV=r1 bash deploy/day4.sh --only J1      следующая редакция судьи
#   JUDGE_FROZEN=r1 bash deploy/day4.sh --only J2   допуск на within — один раз
#   bash deploy/day4.sh --only Z9       архив
#
# Состояние сервера — после дня 3: коллекция library_ru, граф v4 файлом,
# веса 27B в томе. Межкнижный эталон собран дома (scripts/build_crossbook_goldset.py)
# и заморожен до замера: evaluation/goldsets/goldset-x.accepted — sha256 файла.
#
# Серия на межкнижных повторяет день 3 (К10): база, K2 и K4 с прежними
# критериями series_k_verdict.py. К4 — на графе v4: выбор графа сделан на дне 3
# (artifacts/runs/day3/k4-choice.txt = base) по правилу, записанному до
# прогонов, и здесь не пересматривается.
#
# Судья v2 (M6): подбор только на группе across, не больше трёх редакций
# (r1–r3) после r0; между редакциями — разбор результатов дома. Затем ровно
# один прогон within на замороженной редакции (J2 требует JUDGE_FROZEN).

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

RUN="$REPO_DIR/artifacts/runs/day4"
RUN3="$REPO_DIR/artifacts/runs/day3"
GRAPHS="$REPO_DIR/artifacts/graphs"
LOG="$RUN/day4.log"

GOLD_X=evaluation/goldsets/goldset-x.json
GOLD_X_ACCEPTED=evaluation/goldsets/goldset-x.accepted
JUDGE_FACTS=evaluation/judge_facts/calibration.json
COLLECTION=library_ru
JUDGE_DIR=qwen27b
JUDGE_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
JUDGE_REV="${JUDGE_REV:-r0}"
JUDGE_FROZEN="${JUDGE_FROZEN:-}"

# Состав — буква в букву как BASE_ENV дня 3: сравнение с его вердиктом
# имеет смысл только при том же конвейере.
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
metrics_dir() { uv run python -c 'from rag_textbook.config import Settings; print(Settings().paths.metrics_dir)'; }
run_file() {
    local metrics file
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    file=$(ls -t "$metrics"/retrieval_eval_"$1"_*.json 2>/dev/null | head -1)
    [ -n "$file" ] || die "нет прогона с меткой $1 — сначала его шаг"
    printf '%s' "$file"
}

# ------------------------------------------------------------------- план
PLAN=$(cat <<'PLAN'
E0|Проверка: состояние дня 3, граф v4, межкнижный эталон заморожен и сходится с нарезкой сервера
X0|База на межкнижном эталоне (граф v4 файлом, обход по совместному упоминанию)
X2|K2 на межкнижных: обход по зависимостям
X4|K4 на межкнижных: PPR на графе v4 (обход — прогон X0)
XV|Вердикт К10: test принимает, dev — только картина
J1|Судья v2 на группе across, редакция JUDGE_REV (по умолчанию r0)
J2|Судья v2: допуск на within — один раз, на замороженной редакции JUDGE_FROZEN
Z9|Сводка и архив для скачивания
PLAN
)
CODES=$(printf '%s\n' "$PLAN" | cut -d'|' -f1)
# J1 и J2 не идут в прогон «всё по порядку» дальше r0: между редакциями нужен разбор.
MANUAL="J2"

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
mkdir -p "$RUN/done" "$RUN/judge-v2"
uv sync --extra parsing --extra dev -q || die "uv sync не прошёл"
export UV_NO_SYNC=1
for chosen in "$ONLY" "$FROM"; do
    [ -z "$chosen" ] && continue
    printf '%s\n' "$CODES" | grep -qx "$chosen" || die "нет шага $chosen; список — --list"
done

STARTED=$([ -z "$FROM" ] && echo 1 || echo 0)
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
    case " $MANUAL " in *" $code "*) return 1 ;; esac
    [ "$STARTED" = 0 ] && [ "$code" = "$FROM" ] && STARTED=1
    [ "$STARTED" = 1 ] || return 1
    [ -f "$RUN/done/$code" ] && { ok "$code уже выполнен"; return 1; }
    return 0
}
done_mark() { date '+%F %T' > "$RUN/done/$1"; ok "$1 готов ($(stamp))"; }
invalidate_verdict() {
    [ -f "$RUN/done/XV" ] && { rm -f "$RUN/done/XV"; warn "XV: отметка снята — прогоны изменились"; }
    rm -f "$RUN/done/Z9"
}

variant() {
    local name="$1"; shift
    say "  $name"
    run "прогон $name" env "${BASE_ENV[@]}" "$@" \
        uv run rag-textbook eval run --goldset "$GOLD_X" --label "x-$name" \
        --trace "$RUN/trace-$name.jsonl"
    invalidate_verdict
}

check_goldset_x() {
    [ -s "$GOLD_X" ] || die "нет $GOLD_X (дома: scripts/build_crossbook_goldset.py --write)"
    [ -f "$GOLD_X_ACCEPTED" ] || die "межкнижный эталон не заморожен: нет $GOLD_X_ACCEPTED"
    sha256sum -c --quiet "$GOLD_X_ACCEPTED" || die "goldset-x.json отличается от замороженного"
}

# ---------------------------------------------------------------- шаги
step_E0() {
    say "E0. Проверка перед стартом"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
    df -h "$REPO_DIR" | tail -1 | tee -a "$LOG"
    [ -f "$RUN3/done/S0" ] || die "нет отметки S0 дня 3 — состояние сервера не то (bash deploy/day3.sh)"
    [ "$(cat "$RUN3/k4-choice.txt" 2>/dev/null)" = base ] \
        || die "выбор К4 дня 3 не base — граф для X4 надо решать заново"
    for required in "$GRAPHS/v4.json.gz" "$JUDGE_FACTS"; do
        [ -s "$required" ] || die "нет или пуст $required"
    done
    check_goldset_x
    ok "межкнижный эталон заморожен: $(cut -c1-12 "$GOLD_X_ACCEPTED")…"
    # Идентификаторы эталона взяты из нарезки дня 2 (tasks/035/chunks ←
    # artifacts/runs/day2/ru-parsed). Разойдись нарезка — эталон мерил бы
    # пустоту: recall 0 у всех вариантов без единой ошибки (force-flag-starves-stages).
    run "эталон сходится с нарезкой и графом" uv run python - "$GOLD_X" "$GRAPHS/v4.json.gz" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from rag_textbook.config import Settings
from rag_textbook.evaluation.goldset import load_goldset
from rag_textbook.stores.graph_file import GraphFile

questions = load_goldset(Path(sys.argv[1]))
chunks = {item["id"]: item.get("text_hash") for path in Path(Settings().paths.parsed_dir).glob("*_chunks.json")
          for item in json.loads(path.read_text(encoding="utf-8"))}
graph = GraphFile.load(sys.argv[2])
gold = {cid for q in questions for cid in q.gold_chunk_ids}
missing = sorted(gold - set(chunks))
unlinked = sorted(cid for cid in gold if cid not in graph.passages)
# Отпечаток текстов снят дома при сборке: совпадение id ещё не значит совпадение текста.
hashes = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("chunk_hashes") or {}
changed = sorted(cid for cid, h in hashes.items() if cid in chunks and chunks[cid] != h)
print(f"тексты сверены по {len(hashes)} отпечаткам, расходится {len(changed)}", changed[:10])
cells = Counter((q.slice, q.split) for q in questions)
print(f"вопросов {len(questions)}; срез × часть: {dict(sorted(cells.items()))}")
print(f"эталонных фрагментов {len(gold)}: нет в нарезке {len(missing)}, нет в графе {len(unlinked)}")
if missing:
    print("нет в нарезке:", missing[:10])
test = sum(1 for q in questions if q.split == "test" and q.slice == "cross_book")
# Порог К10 записан до замера (решение владельца 2026-09-27).
raise SystemExit(0 if not missing and not changed and hashes and test >= 60 else 1)
PY
    ensure_4b
    done_mark E0
}

step_X0() {
    say "X0. База на межкнижных"
    ensure_4b
    check_goldset_x
    variant base GRAPH_FILE="$GRAPHS/v4.json.gz"
    done_mark X0
}

step_X2() {
    say "X2. K2 на межкнижных"
    ensure_4b
    check_goldset_x
    variant K2 GRAPH_FILE="$GRAPHS/v4.json.gz" GRAPH_WALK=dependency
    done_mark X2
}

step_X4() {
    say "X4. K4 на межкнижных"
    ensure_4b
    check_goldset_x
    variant K4ppr GRAPH_FILE="$GRAPHS/v4.json.gz" GRAPH_RANKER=ppr
    done_mark X4
}

step_XV() {
    say "XV. Вердикт К10"
    check_goldset_x
    local base args
    base=$(run_file x-base)
    args="--run base=$base --run K2=$(run_file x-K2) --run K4walk=$base --run K4ppr=$(run_file x-K4ppr)"
    # На межкнижном наборе срезы linking и cross_book совпадают: К2 проходит
    # обе свои проверки (≥+0.02 и ≥+0.03) на одних и тех же вопросах.
    # shellcheck disable=SC2086
    run "вердикт на test" uv run python scripts/series_k_verdict.py --goldset "$GOLD_X" \
        --split test $args --json "$RUN/verdict-test.json"
    # shellcheck disable=SC2086
    run "картина на dev" uv run python scripts/series_k_verdict.py --goldset "$GOLD_X" \
        --split dev $args --json "$RUN/verdict-dev.json"
    rm -f "$RUN/done/Z9"
    done_mark XV
}

judge_up() {
    local volume path
    volume=$(docker volume inspect rag-textbook_gguf_models --format '{{.Mountpoint}}' 2>/dev/null) \
        || die "нет тома rag-textbook_gguf_models"
    path=$(find "$volume/$JUDGE_DIR" -name '*.gguf' -printf '%f\n' 2>/dev/null | sort | head -1)
    if [ -z "$path" ]; then
        run "веса 27B" bash deploy/models-fetch.sh "$JUDGE_DIR"
        path=$(find "$volume/$JUDGE_DIR" -name '*.gguf' -printf '%f\n' 2>/dev/null | sort | head -1)
        [ -n "$path" ] || die "нет весов 27B в $volume/$JUDGE_DIR"
    fi
    if docker ps --format '{{.Names}}' | grep -qx judge && curl -sf -o /dev/null http://127.0.0.1:8001/health; then
        ok "судья уже поднят"
        return
    fi
    docker compose --env-file .env -f docker/docker-compose.yml stop infinity ollama >/dev/null 2>&1
    docker rm -f rag-textbook-sglang-1 generator judge >/dev/null 2>&1
    # Те же параметры, что в J0 дня 3: меняется только инструкция (M6).
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
}
JUDGE_ENV=(LLM_BASE_URL=http://127.0.0.1:8001/v1 LLM_JUDGE_BASE_URL=http://127.0.0.1:8001/v1
           LLM_MODEL="$JUDGE_DIR" LLM_JUDGE_MODEL="$JUDGE_DIR" LLM_MAX_TOKENS=2048)
judge_v2() {
    # Команда — из tasks/034-report.md; каталог результата обязан быть новым.
    run "судья v2: $1 → $2" env "${JUDGE_ENV[@]}" uv run python scripts/judge_saved.py \
        --calibrate --judge-version v2 --facts "$JUDGE_FACTS" \
        --goldset capture/goldset.json --trace capture/session-0819/trace-always.jsonl \
        --chunks capture --checks evaluation/reward_checks/2026-09-17 \
        --answers-dir capture/session-0903 --group "$1" \
        --output-dir "$RUN/judge-v2/$2" --workers 2
}

step_J1() {
    say "J1. Судья v2 на across, редакция $JUDGE_REV"
    case "$JUDGE_REV" in r0|r1|r2|r3) ;; *) die "JUDGE_REV=$JUDGE_REV: подбор — r0 и не больше трёх редакций" ;; esac
    [ -e "$RUN/judge-v2/within-final" ] && die "допуск на within уже снят — подбор закрыт (M6)"
    local out="$RUN/judge-v2/across-$JUDGE_REV"
    [ -e "$out" ] && die "$out уже есть: новая редакция — новое имя (JUDGE_REV=r1…)"
    # Отпечаток инструкции и фактов — чтобы редакцию можно было сверить дома.
    uv run python - "$JUDGE_FACTS" > "$RUN/judge-v2/across-$JUDGE_REV.fingerprint" <<'PY'
import hashlib, sys
from pathlib import Path
from rag_textbook.evaluation import answers
print("prompt", hashlib.sha256(answers.JUDGE_PROMPT_V2.encode("utf-8")).hexdigest())
print("facts ", hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
    cat "$RUN/judge-v2/across-$JUDGE_REV.fingerprint" | tee -a "$LOG"
    judge_up
    judge_v2 across "across-$JUDGE_REV"
    run "сводка across" uv run python - "$out/calibration.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(data["calibration"], ensure_ascii=False, indent=2))
PY
    rm -f "$RUN/done/Z9"
    warn "разбор дома: следующая редакция — JUDGE_REV=r1 --only J1; выбор — JUDGE_FROZEN=rN --only J2"
    done_mark J1
}

step_J2() {
    say "J2. Судья v2: допуск на within"
    [ -n "$JUDGE_FROZEN" ] || die "не задан JUDGE_FROZEN — какая редакция across заморожена"
    local fp="$RUN/judge-v2/across-$JUDGE_FROZEN.fingerprint"
    [ -s "$fp" ] || die "нет прогона across-$JUDGE_FROZEN — допуск без подбора не ставится"
    [ -e "$RUN/judge-v2/within-final" ] && die "within-final уже снят: допуск ставится один раз (M6)"
    # Замороженная редакция — та, что в коде сейчас: сверяем отпечаток.
    uv run python - "$JUDGE_FACTS" > "$RUN/judge-v2/within-final.fingerprint" <<'PY'
import hashlib, sys
from pathlib import Path
from rag_textbook.evaluation import answers
print("prompt", hashlib.sha256(answers.JUDGE_PROMPT_V2.encode("utf-8")).hexdigest())
print("facts ", hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
    cmp -s "$fp" "$RUN/judge-v2/within-final.fingerprint" \
        || die "код судьи не совпадает с редакцией $JUDGE_FROZEN (см. $fp)"
    judge_up
    try "допуск судьи v2" env "${JUDGE_ENV[@]}" uv run python scripts/judge_saved.py \
        --calibrate --judge-version v2 --facts "$JUDGE_FACTS" \
        --goldset capture/goldset.json --trace capture/session-0819/trace-always.jsonl \
        --chunks capture --checks evaluation/reward_checks/2026-09-17 \
        --answers-dir capture/session-0903 --group within \
        --output-dir "$RUN/judge-v2/within-final" --workers 2
    if [ "$LAST_OK" = 1 ]; then
        ok "СУДЬЯ v2 ДОПУЩЕН (редакция $JUDGE_FROZEN)"
    else
        warn "СУДЬЯ v2 НЕ ДОПУЩЕН: оценки ответов судьёй не ведутся (критерий отказа M6)"
    fi
    run "сводка within" uv run python - "$RUN/judge-v2/within-final/calibration.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(data["calibration"], ensure_ascii=False, indent=2))
PY
    docker rm -f judge >/dev/null 2>&1
    rm -f "$RUN/done/Z9"
    done_mark J2
}

step_Z9() {
    say "Z9. Архив"
    local archive metrics
    archive="$HOME/day4-results-$(date +%Y%m%d-%H%M).tar.gz"
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    local paths=(artifacts/runs/day4)
    [ -e "$metrics" ] && paths+=("$metrics")
    tar czf "$archive" -C "$REPO_DIR" "${paths[@]}" 2>>"$LOG" || die "архив не создан: см. $LOG"
    ok "архив: $archive ($(du -h "$archive" | cut -f1))"
    warn "скачать: scp -i <ключ> root@<ip>:$archive ."
    done_mark Z9
}

for code in $CODES; do
    if should_run "$code"; then
        "step_$code"
    fi
done
say "Готово ($(stamp)). Журнал: $LOG"
