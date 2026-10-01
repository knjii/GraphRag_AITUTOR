#!/usr/bin/env bash
# День аренды №2: этапы 2–4 порядка работ (перенарезка русского блока,
# граф v4, эталоны, точка отсчёта). План и критерии — docs/SERVER-DAY-2.md.
# Каждый шаг отмечается в artifacts/runs/day2/done, повторный запуск
# продолжает с несделанного.
#
#   bash deploy/day2.sh --list          шаги и что они проверяют
#   bash deploy/day2.sh                 всё по порядку
#   bash deploy/day2.sh --only B0       один шаг заново
#   bash deploy/day2.sh --from G1       начиная с шага
#
# День 1 (deploy/day1.sh) не запускался и по порядку работ 2026-09-22
# заменён этим: разбор библиотеки и нарезка переехали сюда (только русский
# блок), приёмка награды и проба GRPO — на этап 6.
#
# Порядок задан видеопамятью одной карты: разбор (MinerU один), нарезка
# и векторы (поиск), граф (4B + поиск), замеры (4B + поиск), вопросы (9B
# без поиска), в конце — работа без карты.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

RUN="$REPO_DIR/artifacts/runs/day2"
GRAPHS="$REPO_DIR/artifacts/graphs"
V2="$REPO_DIR/artifacts/goldset-v2"
LOG="$RUN/day2.log"

MML_DOC=0690bb81b7e3c831
MML_PDF=documents/pdf_docs/Dayzenrot_Feyzal_On_Matematika_v_mashinnom_obuchen_241126_230954.pdf
MML_CHUNKS="artifacts/parsed/${MML_DOC}_chunks.json"
MML_REFERENCE="artifacts/goldset-r2/${MML_DOC}_chunks.json"
GOLD_R2=evaluation/goldsets/goldset-r2.json
GOLD_R2_REPORT=evaluation/goldsets/goldset-r2.migration.json
GOLD_V2=evaluation/goldsets/goldset-v2.json
COLLECTION=library_ru
GRAPH_V4="$GRAPHS/v4.json.gz"

# Корпус один: русский блок и MML в одной коллекции и одном графе.
# Этап 4 — новая точка отсчёта, а эталон v2 межкнижный: MML отдельно
# от библиотеки мерила бы другую систему.
CORPUS_ENV=(QDRANT_COLLECTION="$COLLECTION" QDRANT_SPARSE_LANGUAGE=russian
            CHUNKER_RESPECT_FORMULAS=true CHUNK_SIZE=1200 CHUNK_OVERLAP=180)
# Библиотека — без описаний картинок: модель зрения в .env ходит на порт
# SGLang, погашенного на этих шагах (как в day1.sh). У MML описания
# берутся из кэша обогащения — без них не сойдётся отпечаток нарезки.
LIB_ENV=("${CORPUS_ENV[@]}" GRAPH_ENABLED=false GRAPH_RETRIEVAL_ENABLED=false
         CHUNKER_ENRICH_ENABLED=false)
MML_ENV=("${CORPUS_ENV[@]}" GRAPH_ENABLED=false GRAPH_RETRIEVAL_ENABLED=false
         CHUNKER_ENRICH_ENABLED=true CHUNKER_ENRICH_CACHE_ENABLED=true)
# Извлечение v4: роли и обозначения (К2, К3). Модель и размышление —
# как у графа v3: 4B, reasoning_effort=none (reasoning-effort-breaks-extraction).
GRAPH_ENV=("${CORPUS_ENV[@]}" GRAPH_ENABLED=true GRAPH_EXTRACTION_PROMPT_VERSION=v4
           LLM_REASONING_EFFORT=none)
# Существующая система — ровно слепок 2026-08-19 (capture/session-0819,
# заголовок trace-always.jsonl): маршрут always, 8 мест и 16 для
# связывающих. Задаётся явно, а не .env: переменные окружения старше .env,
# и точка отсчёта не зависит от того, что там осталось с прошлых сессий.
BASE_ENV=("${CORPUS_ENV[@]}" GRAPH_ENABLED=true GRAPH_RETRIEVAL_ENABLED=true
          GRAPH_BACKEND=neo4j GRAPH_RANKER=walk GRAPH_WALK=comention
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
# Слепок расширяет оценивание, а не отбор (capture.sh): метрики прогона
# с ним те же, поэтому точка отсчёта и слепок снимаются одним прогоном.
TRACE_ENV=(EVAL_TRACE_POOL=100 EVAL_TRACE_RERANK_ALL=true)

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*" | tee -a "$LOG"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
die()  { printf '\n\033[1;31mСТОП: %s\033[0m\n' "$*" | tee -a "$LOG" >&2; exit 1; }
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; local rc=$?; [ "$rc" -le 1 ]; }
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

model_4b_with_search() {
    # llama.cpp с 9B держит тот же порт 8001, что и SGLang.
    docker rm -f generator >/dev/null 2>&1 || true
    run "переключение на 4B" bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval
    run "запуск служб" bash deploy/services.sh up
}
ensure_4b() {
    if grep -q '^LLM_MODEL=Qwen/Qwen3.5-4B$' .env 2>/dev/null \
        && curl -sf -o /dev/null http://127.0.0.1:8001/health \
        && curl -sf -o /dev/null http://127.0.0.1:7997/health; then
        ok "4B и поиск уже подняты"
    else
        model_4b_with_search
    fi
}
# 9B — в том кванте, на котором мерили качество ответов (2026-09-09,
# deploy/models-ab.sh): UD-Q4_K_XL на llama.cpp, те же ключи размышления.
# bf16 на SGLang занимал 18.9 ГБ весов, и движок сжимал батч до одного
# запроса (журнал, запись 39): эталон собирался час вместо минут.
GEN_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
GEN_MODEL=/models/qwen9b/Qwen3.5-9B-UD-Q4_K_XL.gguf
GEN_SLOTS=16
GEN_SLOT_TOKENS=8192
model_9b_alone() {
    say "9B-Q4 на llama.cpp ($GEN_SLOTS слотов по $GEN_SLOT_TOKENS)"
    docker rm -f rag-textbook-sglang-1 generator >/dev/null 2>&1 || true
    run "запуск llama.cpp" docker run -d --name generator --gpus all \
        -v rag-textbook_gguf_models:/models -p 127.0.0.1:8001:8000 "$GEN_IMAGE" \
        -m "$GEN_MODEL" --host 0.0.0.0 --port 8000 \
        -c $((GEN_SLOT_TOKENS * GEN_SLOTS)) -np "$GEN_SLOTS" -ngl 999 --jinja \
        --reasoning off --reasoning-effort minimal
    local ready=0
    for _ in $(seq 1 60); do
        curl -sf -o /dev/null http://127.0.0.1:8001/health && { ready=1; break; }
        sleep 5
    done
    [ "$ready" = 1 ] || { docker logs --tail 25 generator 2>&1 | tee -a "$LOG"; die "llama.cpp не поднялся"; }
    ok "видеопамять: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    # Прочитать ответ до долгого прогона: пустой ответ или размышление
    # вместо текста — ровно то, что погубило ячейку Muse.
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
llm_off() {
    run "остановка SGLang" docker compose --env-file .env \
        -f docker/docker-compose.vllm.yml --profile sglang stop sglang
    ok "видеопамять: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
search_off() {
    run "остановка служб поиска" docker compose --env-file .env \
        -f docker/docker-compose.yml stop infinity ollama
    ok "видеопамять: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
search_on() { run "запуск служб" bash deploy/services.sh up; }
hf_cache_dir() {
    docker volume inspect rag-textbook_hf_cache --format '{{.Mountpoint}}' 2>/dev/null \
        || { docker volume create rag-textbook_hf_cache >/dev/null \
             && docker volume inspect rag-textbook_hf_cache --format '{{.Mountpoint}}'; }
}
metrics_dir() { uv run python -c 'from rag_textbook.config import Settings; print(Settings().paths.metrics_dir)'; }

# ------------------------------------------------------------------- план
PLAN=$(cat <<'PLAN'
E0|Проверка: карта, диск, книги, goldset-r2 и эталонная нарезка MML, кэш обогащения; фоном — веса 4B и 9B
L1|Разбор русского блока MinerU: ru (auto), ru-ocr (Гельфанд, распознавание)
L2|Нарезка с CHUNKER_RESPECT_FORMULAS: MML (сверка отпечатка с goldset-r2), ru, ru-ocr; векторы в library_ru
G0|Проба извлечения v4 на 40 фрагментах MML: откаты, заполненность ролей, доля defines
G1|Граф v4 по всему русскому блоку (4B); журнал отказов
B0|Точка отсчёта на goldset-r2 и слепок (существующая система, Neo4j)
X0|Выгрузка графа в файл и допуск стенда graph_fidelity --live (≥ 99%); прогон по файлу
P0|Пары для эталона v2 из пяти источников (векторы из коллекции)
Q0|Эталон v2 моделью 9B с абляцией двухшаговости
K0|Графы серии К без карты: структурный (К1 A), объединение (К1 C), обозначения (К3)
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
mkdir -p "$RUN/done" "$GRAPHS" "$V2"
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
should_run() {
    local code="$1"
    [ -n "$ONLY" ] && { [ "$code" = "$ONLY" ]; return; }
    [ "$STARTED" = 0 ] && [ "$code" = "$FROM" ] && STARTED=1
    [ "$STARTED" = 1 ] || return 1
    [ -f "$RUN/done/$code" ] && { ok "$code уже выполнен"; return 1; }
    return 0
}
done_mark() { date '+%F %T' > "$RUN/done/$1"; ok "$1 готов ($(stamp))"; }
# Повтор шага снимает отметки зависящих от него: иначе они считались бы
# выполненными на старых данных (урок train-day.sh, задача 027).
invalidate_after() {
    local code found=0
    rm -f "$RUN/done/$1"
    for code in $CODES; do
        [ "$found" = 1 ] && [ -f "$RUN/done/$code" ] && { rm -f "$RUN/done/$code"; warn "$code: отметка снята — зависит от $1"; }
        [ "$code" = "$1" ] && found=1
    done
}

ingest_group() {
    local dir="$1" method="$2" stages="$3"; shift 3
    say "  $dir ($stages, MinerU $method)"
    run "индексация $dir ($stages)" env "$@" PDF_DIR="$dir" \
        MINERU_METHOD="$method" MINERU_LANG=east_slavic \
        uv run rag-textbook ingest --stages "$stages" --monitor
}

# ---------------------------------------------------------------- шаги
step_E0() {
    say "E0. Проверка перед стартом"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
    df -h "$REPO_DIR" | tail -1 | tee -a "$LOG"
    [ -f documents/library/SHA256SUMS ] || die "нет documents/library — upload.ps1 -WithLibrary"
    (cd documents/library && sha256sum -c --quiet SHA256SUMS) || die "книги не совпадают с проверенными"
    ok "русский блок: $(find documents/library/ru documents/library/ru-ocr -name '*.pdf' | wc -l) PDF"
    for required in "$MML_PDF" "$GOLD_R2" "$GOLD_R2_REPORT" "$MML_REFERENCE" \
                    artifacts/cache/enrichment.sqlite3; do
        [ -s "$required" ] || die "нет или пуст $required — см. «До аренды» в docs/SERVER-DAY-2.md"
    done
    # В pdf_docs индексируется всё; чужой PDF попал бы в коллекцию и граф.
    local others
    others=$(find documents/pdf_docs -name '*.pdf' ! -path "$MML_PDF" | wc -l)
    [ "$others" = 0 ] || die "в documents/pdf_docs кроме MML ещё $others PDF — убрать"
    # Разбор MML должен приехать из кэша: повторный MinerU даст иной текст.
    ls -d artifacts/parsed/Dayzenrot_*__* >/dev/null 2>&1 || die "нет разбора MML в artifacts/parsed — upload.ps1 -WithCaches"
    run "эталонная нарезка совпадает с переносом" uv run rag-textbook goldset check-chunks \
        --report "$GOLD_R2_REPORT" --chunks "$MML_REFERENCE"
    # Ключи, от которых зависят кэши и нарезка. Остальное задаётся явно.
    local key want actual bad=0
    for want in LLM_REASONING_EFFORT=none MINERU_BACKEND=pipeline; do
        key="${want%%=*}"
        actual=$(grep -E "^${key}=" .env 2>/dev/null | head -1)
        [ "$actual" = "$want" ] || { warn "в .env: '${actual:-нет}', ожидалось '$want'"; bad=1; }
    done
    [ "$bad" = 0 ] || die "поправьте .env (deploy/baseline-env.sh и model-swap.sh) и повторите E0"
    command -v hf >/dev/null 2>&1 || uv tool install "huggingface_hub[cli]" >/dev/null 2>&1
    (
        HF_HOME="$(hf_cache_dir)"
        export HF_HOME
        for model in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
            hf download "$model" --exclude "*.gguf" >/dev/null && echo "готово: $model"
        done
        echo "загрузка весов завершена"
    ) > "$RUN/weights.log" 2>&1 &
    echo $! > "$RUN/weights.pid"
    ok "веса 4B и 9B качаются фоном: $RUN/weights.log"
    done_mark E0
}

step_L1() {
    say "L1. Разбор русского блока"
    # MinerU получает карту целиком.
    llm_off
    search_off
    ingest_group documents/library/ru     auto parse "${LIB_ENV[@]}"
    ingest_group documents/library/ru-ocr ocr  parse "${LIB_ENV[@]}"
    run "проверка текста книг" uv run python scripts/check_parsed_text.py
    run "все книги русского блока разобраны" uv run python - <<'PY'
from pathlib import Path
from rag_textbook.config import Settings

expected = {p.stem for d in ("ru", "ru-ocr") for p in Path("documents/library", d).glob("*.pdf")}
parsed = {d.parent.name.split("__", 1)[0] for d in Path(Settings().paths.parsed_dir).glob("*/blocks.json")}
missing = sorted(expected - parsed)
print(f"книг {len(expected)}, разобрано {len(expected) - len(missing)}")
if missing:
    print("не разобраны:", ", ".join(missing))
    raise SystemExit(1)
PY
    invalidate_after L1
    done_mark L1
}

step_L2() {
    say "L2. Нарезка и векторы"
    search_on
    # Манифест помнит «нарезано / векторизовано / в графе» по файлу, а не по
    # настройкам и коллекции: без сброса MML осталась бы на старой нарезке,
    # а в library_ru не попала бы вовсе (ловушка force-flag-starves-stages).
    run "сброс отметок стадий" bash deploy/reset-stages.sh chunked embedded graphed
    ingest_group documents/pdf_docs auto chunk "${MML_ENV[@]}"
    try "отпечаток нарезки MML" uv run rag-textbook goldset check-chunks \
        --report "$GOLD_R2_REPORT" --chunks "$MML_CHUNKS"
    if [ "$LAST_OK" != 1 ]; then
        # Сервер нарезал иначе (обычно — промах кэша обогащения). Эталон
        # перенесён на эталонную нарезку; ставим её, серверную храним
        # для разбора. Отметка «нарезано» уже стоит — векторизуется файл.
        cp "$MML_CHUNKS" "$RUN/mml-chunks-server.json"
        cp "$MML_REFERENCE" "$MML_CHUNKS"
        warn "нарезка MML разошлась с переносом — поставлена эталонная; серверная: $RUN/mml-chunks-server.json"
        run "отпечаток после замены" uv run rag-textbook goldset check-chunks \
            --report "$GOLD_R2_REPORT" --chunks "$MML_CHUNKS"
    fi
    ingest_group documents/pdf_docs       auto embed       "${MML_ENV[@]}"
    ingest_group documents/library/ru     auto chunk,embed "${LIB_ENV[@]}"
    ingest_group documents/library/ru-ocr ocr  chunk,embed "${LIB_ENV[@]}"
    run "коллекция собрана целиком" env "${CORPUS_ENV[@]}" uv run python - "$GOLD_R2" <<'PY'
import json, sys
from pathlib import Path
from rag_textbook.config import Settings
from rag_textbook.evaluation.goldset import load_goldset
from rag_textbook.stores.vector_store import build_vector_store

settings = Settings()
files = sorted(Path(settings.paths.parsed_dir).glob("*_chunks.json"))
expected = {item["id"] for path in files for item in json.loads(path.read_text(encoding="utf-8"))}
stored = {chunk.id for chunk in build_vector_store(settings.vector_store).iter_chunks()}
print(f"{settings.vector_store.collection}: файлов нарезки {len(files)}, фрагментов {len(expected)}, в коллекции {len(stored)}")
gold = {cid for q in load_goldset(Path(sys.argv[1])) for cid in q.gold_chunk_ids}
print(f"эталон r2: фрагментов {len(gold)}, нет в коллекции {len(gold - stored)}")
problems = []
# Пустой файл нарезки проверку выше проходит: 2026-09-22 все 15 книг
# нарезались в [], а «коллекция собрана целиком» ответила да.
empty = [p.name for p in files if not json.loads(p.read_text(encoding="utf-8"))]
if empty:
    problems.append(f"пустая нарезка у {len(empty)} файлов: {', '.join(empty[:5])}")
books = {p.stem for d in ("ru", "ru-ocr") for p in Path("documents/library", d).glob("*.pdf")}
if len(files) < len(books) + 1:
    problems.append(f"файлов нарезки {len(files)} при {len(books)} книгах и MML")
if expected != stored:
    problems.append(f"расхождение: нет в коллекции {len(expected - stored)}, лишних {len(stored - expected)}")
if gold - stored:
    problems.append("эталон ссылается на отсутствующие фрагменты")
if problems:
    print("; ".join(problems))
    raise SystemExit(1)
PY
    invalidate_after L2
    done_mark L2
}

step_G0() {
    say "G0. Проба извлечения v4"
    ensure_4b
    run "40 фрагментов MML" env "${GRAPH_ENV[@]}" uv run python scripts/extraction_smoke.py \
        --chunks "$MML_CHUNKS" --sample 40 --json "$RUN/smoke-v4.json"
    warn "прочитать примеры выше: роли должны быть осмысленны, а не просто заполнены"
    done_mark G0
}

step_G1() {
    say "G1. Граф v4"
    ensure_4b
    run "граф снят" env "${GRAPH_ENV[@]}" uv run rag-textbook graph drop --yes
    local journal=artifacts/state/extraction_failures.jsonl
    [ -f "$journal" ] && mv "$journal" "$RUN/extraction_failures.before-$(date +%H%M%S).jsonl"
    # Сначала MML: её граф нужен первым (точка отсчёта на r2), и сбой
    # схемы v4 выйдет наружу на 40 минутах, а не на трёх часах.
    ingest_group documents/pdf_docs       auto graph "${GRAPH_ENV[@]}"
    ingest_group documents/library/ru     auto graph "${GRAPH_ENV[@]}"
    ingest_group documents/library/ru-ocr ocr  graph "${GRAPH_ENV[@]}"
    run "статистика графа" env "${GRAPH_ENV[@]}" uv run rag-textbook graph stats
    run "доля откатов извлечения" uv run python - "$journal" <<'PY'
import json, sys
from pathlib import Path
from rag_textbook.config import Settings

total = sum(len(json.loads(p.read_text(encoding="utf-8")))
            for p in Path(Settings().paths.parsed_dir).glob("*_chunks.json"))
journal = Path(sys.argv[1])
failed = sum(1 for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()) if journal.exists() else 0
share = failed / max(1, total)
print(f"фрагментов {total}, откатов к правилам {failed} ({share:.1%})")
# Порог записан до прогона: у графа v3 после починки обрезки откатов 0.
raise SystemExit(1 if share > 0.05 else 0)
PY
    # Счётчик на выходе, а не согласованность: 2026-09-22 пустая нарезка
    # прошла все проверки согласованности (журнал, запись 38).
    run "граф покрывает корпус" env "${GRAPH_ENV[@]}" uv run python - <<'PY'
import json
from pathlib import Path
from rag_textbook.config import Settings
from rag_textbook.context import build_context

settings = Settings()
total = sum(len(json.loads(p.read_text(encoding="utf-8")))
            for p in Path(settings.paths.parsed_dir).glob("*_chunks.json"))
context = build_context(settings)
try:
    stats = context.graph_store.stats()
finally:
    context.close()
print(f"фрагментов {total}, в графе {stats}")
raise SystemExit(0 if stats.get("passages", 0) >= 0.95 * total else 1)
PY
    invalidate_after G1
    done_mark G1
}

step_B0() {
    say "B0. Точка отсчёта на goldset-r2"
    ensure_4b
    run "прогон и слепок" env "${BASE_ENV[@]}" "${TRACE_ENV[@]}" \
        uv run rag-textbook eval run --goldset "$GOLD_R2" --label d2-base-r2 \
        --trace "$RUN/trace-r2.jsonl"
    run "слепок снят с той конфигурацией" uv run python - "$RUN/trace-r2.jsonl" <<'PY'
import json, sys
header = json.loads(open(sys.argv[1], encoding="utf-8").readline())
snap = {**header["settings_snapshot"], **header["ordering_snapshot"]}
want = {"graph.seed_mode": "both", "graph.max_entity_degree": 64, "graph.hop_decay": 0.5,
        "graph.passage_idf_enabled": False, "retrieval.router_mode": "always",
        "retrieval.top_k": 8, "retrieval.top_k_linking": 16, "graph.backend": "neo4j",
        "graph.walk": "comention", "llm.model": "Qwen/Qwen3.5-4B"}
bad = {k: (snap.get(k), v) for k, v in want.items() if snap.get(k) != v}
print("состав слепка:", "совпадает" if not bad else bad)
raise SystemExit(1 if bad else 0)
PY
    invalidate_after B0
    done_mark B0
}

step_X0() {
    say "X0. Граф файлом и допуск стенда"
    ensure_4b
    run "выгрузка графа" env "${BASE_ENV[@]}" uv run python scripts/graph_export.py \
        --out "$GRAPH_V4" --variant v4
    try "допуск стенда" env "${BASE_ENV[@]}" uv run python scripts/graph_fidelity.py \
        --graph-file "$GRAPH_V4" --trace "$RUN/trace-r2.jsonl" --live --json "$RUN/fidelity.json"
    if [ "$LAST_OK" = 1 ]; then
        echo ok > "$RUN/fidelity.ok"
        ok "стенд годен: серия К на файлах разрешена"
    else
        rm -f "$RUN/fidelity.ok"
        warn "СТЕНД НЕ ГОДЕН: опыты К1–К5 на файлах не ставить (день 3 это проверит)"
    fi
    # Сквозная проверка: вся цепочка на файле против цепочки на Neo4j.
    run "прогон по файлу" env "${BASE_ENV[@]}" GRAPH_BACKEND=memory GRAPH_FILE="$GRAPH_V4" \
        uv run rag-textbook eval run --goldset "$GOLD_R2" --label d2-base-file-r2
    local metrics
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    try "файл против Neo4j" env "${BASE_ENV[@]}" uv run rag-textbook eval compare \
        "$(ls -t "$metrics"/retrieval_eval_d2-base-r2_*.json | head -1)" \
        "$(ls -t "$metrics"/retrieval_eval_d2-base-file-r2_*.json | head -1)"
    done_mark X0
}

step_P0() {
    say "P0. Пары для эталона v2"
    # Пары не отбираются ни одним из сравниваемых графов (урок R8).
    run "пары" env "${CORPUS_ENV[@]}" uv run rag-textbook goldset pairs \
        --out "$V2/pairs.jsonl" --per-source 200 --vectors-from-store --seed 20260922
    run "межкнижные пары есть" uv run python - "$V2/pairs.jsonl" <<'PY'
import json, sys
from collections import Counter
pairs = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
by_source = Counter(p["source"] for p in pairs)
print("пар по источникам:", dict(by_source))
raise SystemExit(0 if by_source.get("cross_book", 0) >= 25 else 1)
PY
    invalidate_after P0
    done_mark P0
}

step_Q0() {
    say "Q0. Эталон v2 (9B)"
    search_off
    model_9b_alone
    run "вопросы и абляция" env "${CORPUS_ENV[@]}" LLM_MAX_CONCURRENCY=16 LLM_REASONING_EFFORT=none \
        uv run rag-textbook goldset build-v2 --pairs "$V2/pairs.jsonl" --out "$GOLD_V2" \
        --single 100 --formula 50 --ablate --workers 16 --seed 20260922         --journal "$V2/llm-journal.jsonl"
    run "состав эталона" uv run rag-textbook goldset stats --path "$GOLD_V2"
    run "двухшаговых достаточно" uv run python - "$GOLD_V2" <<'PY'
import sys
from collections import Counter
from pathlib import Path
from rag_textbook.evaluation.goldset import load_goldset

questions = load_goldset(Path(sys.argv[1]))
cells = Counter((q.slice or q.question_type, q.split) for q in questions)
print("срез × часть:", dict(sorted(cells.items())))
two_hop = sum(1 for q in questions if q.expected_hops > 1)
cross = sum(1 for q in questions if q.slice == "cross_book")
print(f"вопросов {len(questions)}, двухшаговых после абляции {two_hop}, межкнижных {cross}")
# Порог записан до прогона: меньше 100 двухшаговых — интервал шире эффекта.
raise SystemExit(0 if two_hop >= 100 and cross >= 20 else 1)
PY
    warn "владельцу: выборка вопросов v2 на приёмку (M4) — до дня 3"
    invalidate_after Q0
    done_mark Q0
}

step_K0() {
    say "K0. Графы серии К (без карты)"
    local parsed="$RUN/ru-parsed"
    rm -rf "$parsed" && mkdir -p "$parsed"
    cp artifacts/parsed/*_chunks.json "$parsed/"
    run "К1 A: структурный" uv run python scripts/graph_structural.py --parsed "$parsed" \
        --out "$GRAPHS/k1a.json.gz" --report "$RUN/k1a-report.json"
    run "К1 C: объединение с v4" uv run python scripts/graph_structural.py --parsed "$parsed" \
        --out "$GRAPHS/k1c.json.gz" --merge-with "$GRAPH_V4"
    run "К3: обозначения на книгу" uv run python scripts/graph_structural.py --parsed "$parsed" \
        --out "$GRAPHS/k3-book.json.gz" --merge-with "$GRAPH_V4" --notation --notation-scope book
    run "К3: обозначения на раздел" uv run python scripts/graph_structural.py --parsed "$parsed" \
        --out "$GRAPHS/k3-sec.json.gz" --merge-with "$GRAPH_V4" --notation --notation-scope section
    run "роли в графе v4" uv run python - "$GRAPH_V4" <<'PY'
import sys
from collections import Counter
from rag_textbook.stores.graph_file import GraphFile

graph = GraphFile.load(sys.argv[1])
roles = Counter(role for entities in graph.mentions.values() for _, role in entities.values())
kinds = Counter(str(entity.get("kind") or "concept") for entity in graph.entities.values())
defined = sum(1 for entities in graph.mentions.values() if any(r == "defines" for _, r in entities.values()))
print("роли упоминаний:", dict(roles), "| типы узлов:", dict(kinds))
print(f"фрагментов с определением: {defined} из {len(graph.passages)}")
raise SystemExit(0 if roles.get("defines", 0) > 0 else 1)
PY
    done_mark K0
}

step_Z9() {
    say "Z9. Архив"
    local archive metrics
    archive="$HOME/day2-results-$(date +%Y%m%d-%H%M).tar.gz"
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    local paths=(artifacts/runs/day2 artifacts/graphs artifacts/goldset-v2)
    for extra in "$GOLD_V2" "$GOLD_V2.ablation.jsonl" artifacts/state/extraction_failures.jsonl "$metrics"; do
        [ -e "$extra" ] && paths+=("$extra")
    done
    tar czf "$archive" -C "$REPO_DIR" "${paths[@]}" 2>>"$LOG" || die "архив не создан: см. $LOG"
    ok "архив: $archive ($(du -h "$archive" | cut -f1)), внутри: ${paths[*]}"
    warn "скачать: scp -i <ключ> root@<ip>:$archive ."
    warn "кэши разбора и извлечения — deploy/backup.ps1; затем сервер можно выключать"
    done_mark Z9
}

for code in $CODES; do
    if should_run "$code"; then
        "step_$code"
    fi
done
say "Готово ($(stamp)). Журнал: $LOG"
