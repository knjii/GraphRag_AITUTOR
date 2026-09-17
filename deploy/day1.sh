#!/usr/bin/env bash
# День аренды №1 (спринт 2): приёмка награды, библиотека, вопросы, проба GRPO.
# План и критерии — docs/SERVER-DAY-1.md. Каждый шаг отмечается в
# artifacts/runs/day1/done, повторный запуск продолжает с несделанного.
#
#   bash deploy/day1.sh --list          шаги и что они проверяют
#   bash deploy/day1.sh                 всё по порядку
#   bash deploy/day1.sh --only D1       один шаг заново
#   bash deploy/day1.sh --from L1       начиная с шага
#
# Порядок задан видеопамятью одной карты: сначала всё, что работает с 4B
# и поиском, потом разбор книг (модель и поиск погашены), потом 9B без
# поиска, поиск без модели и в конце обучение на пустой карте.
#
# Каждая команда, которая что-то производит, проверяется по коду возврата:
# шаг не отмечается выполненным после ошибки. `set -e` не включён намеренно —
# P1 разбирает неуспешный запуск сам (OOM) и повторяет с меньшим числом
# генераций.
#
# Приёмка награды (Д1) — ручная: лист и ключ скачиваются и оцениваются
# на ноутбуке, пока идут остальные шаги.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
# bootstrap.sh кладёт рабочее зеркало PyPI в .bashrc; неинтерактивный bash его не читает.
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

RUN="$REPO_DIR/artifacts/runs/day1"
OUT="$REPO_DIR/artifacts/rl"
LOG="$RUN/day1.log"
mkdir -p "$RUN/done" "$OUT"

MML_DOC=0690bb81b7e3c831
TRACE_MML=capture/session-0819/trace-always.jsonl
PROMPT_V4=deploy/prompts/qa-v4.txt
# Обогащение картинок моделью зрения для библиотеки выключено осознанно:
# в .env адрес модели зрения наследует порт SGLang, который на этих шагах
# погашен, и описания молча терялись бы, тратя время на повторы.
LIB_ENV=(CHUNKER_RESPECT_FORMULAS=true GRAPH_ENABLED=false GRAPH_RETRIEVAL_ENABLED=false
         CHUNKER_ENRICH_ENABLED=false)

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*" | tee -a "$LOG"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
die()  { printf '\n\033[1;31mСТОП: %s\033[0m\n' "$*" | tee -a "$LOG" >&2; exit 1; }
# grep возвращает 1, когда отфильтровал всё; это не ошибка шага.
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; local rc=$?; [ "$rc" -le 1 ]; }
stamp() { date '+%H:%M:%S'; }

# Команда с журналом: падение останавливает шаг, а не отмечает его готовым.
run() {
    local desc="$1"; shift
    "$@" 2>&1 | clean | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die "$desc — код возврата $rc (журнал: $LOG)"
}

# Как run, но без остановки дня: нужен там, где сбой одной связки
# не должен отменять вторую (задача 018, находка 3).
LAST_OK=1
try() {
    local desc="$1"; shift
    "$@" 2>&1 | clean | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = 0 ]; then LAST_OK=1; else LAST_OK=0; warn "$desc — код возврата $rc"; fi
}

# Модель и службы. model-swap.sh правит .env и ждёт готовности.
model_4b_with_search() {
    run "переключение на 4B" bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval
    run "запуск служб" bash deploy/services.sh up
}
# 4B и поиск уже подняты — не перезапускать (перезапуск SGLang стоит минут карты).
ensure_4b() {
    if grep -q '^LLM_MODEL=Qwen/Qwen3.5-4B$' .env 2>/dev/null \
        && curl -sf -o /dev/null http://127.0.0.1:8001/health \
        && curl -sf -o /dev/null http://127.0.0.1:7997/health; then
        ok "4B и поиск уже подняты"
    else
        model_4b_with_search
    fi
}
model_9b_alone() { run "переключение на 9B" bash deploy/model-swap.sh Qwen/Qwen3.5-9B 0.85; }
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
E0|Проверка: карта, диск, книги по SHA256SUMS, эпизоды, отпечаток награды; фоном — окружение RL и веса
D1|Д1+Д2: группы генераций 4B (20 вопросов × 8) и слепой лист для ручной приёмки награды
D4|Д4: ответы 4B по слепку с удвоенной разметкой и без неё; перенос формул v1/v2
G4|Г4: затравки графа «вопрос + верх выдачи» (eval ab graph_seed_both)
L1|Разбор библиотеки: ru (auto), ru-ocr (Гельфанд, распознавание), en (язык en)
L2|Проверка разбора на мусор кодировки; нарезка и векторы в library_ru / library_en
Q1|Д5: вопросы для обучения по русскому комплекту моделью 9B (параллельно)
Q2|Д5: аудит вопросов по всему корпусу, слепок поиска по library_ru, эпизоды RL
R0|Окружения RL (vLLM и Unsloth): импорты, карта видна, отпечаток награды
P1|Д3: проба GRPO 4B + LoRA на обеих связках (TRL+vLLM и Unsloth): память, секунды на шаг, доля генерации
Z9|Сводка и архив результатов для скачивания
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
    [ -f documents/library/SHA256SUMS ] || die "нет documents/library — upload.ps1 -WithLibrary"
    (cd documents/library && sha256sum -c --quiet SHA256SUMS) || die "книги не совпадают с проверенными"
    ok "книг: $(find documents/library -name '*.pdf' | wc -l)"
    # Входы дорогих шагов проверяются здесь, а не там, где уже потрачена карта.
    for required in "$OUT/mml-test.jsonl" "$OUT/reward-fingerprint.json" "$TRACE_MML" \
                    "$PROMPT_V4" deploy/requirements-rl.txt \
                    deploy/requirements-rl-vllm.txt deploy/requirements-rl-app.txt; do
        [ -s "$required" ] || die "нет или пуст $required — см. «До аренды»"
    done
    run "отпечаток награды (окружение сервиса)" uv run python scripts/reward_fingerprint.py \
        --dataset "$OUT/mml-test.jsonl" --check "$OUT/reward-fingerprint.json"
    # Кэш разбора MML нужен D4: без него замер по слепку посчитается по пустому корпусу.
    run "фрагменты MML на месте" uv run python - <<'PY'
import json
from pathlib import Path
from rag_textbook.config import Settings
from rag_textbook.evaluation.goldset import load_goldset

settings = Settings()
have = set()
for path in Path(settings.paths.parsed_dir).glob("*_chunks.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["chunks"] if isinstance(payload, dict) else payload
    have.update(str(item["id"]) for item in items)
need = {cid for q in load_goldset(settings.evaluation.goldset_path) for cid in q.gold_chunk_ids}
missing = need - have
print(f"эталонных фрагментов {len(need)}, нет в выгрузке разбора: {len(missing)}")
raise SystemExit(1 if len(missing) > len(need) // 2 else 0)
PY
    # Окружение RL не трогает карту — ставится фоном, пока идут D1–Q2.
    # Два окружения: быстрое (TRL + vLLM) и экономное по памяти (Unsloth).
    # Выбор связки открыт — задача 017; P1 меряет оба.
    for stack in vllm unsloth; do
        rl_env_install "$stack"
    done
    # Веса 9B (~18 ГБ) качаются фоном сейчас, а не в Q1 на оплаченной
    # простаивающей карте. Том тот же, что у SGLang и у обучения.
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

# Окружение обучения: vllm — TRL с vLLM без Unsloth, unsloth — экономное по памяти.
rl_env_dir()  { [ "$1" = vllm ] && echo .venv-rl-vllm || echo .venv-rl; }
rl_env_reqs() { [ "$1" = vllm ] && echo deploy/requirements-rl-vllm.txt || echo deploy/requirements-rl.txt; }
rl_env_hash() { sha256sum "$(rl_env_reqs "$1")" deploy/requirements-rl-app.txt | sha256sum | cut -d' ' -f1; }
# Маркер годен, только если окружение на месте: удалённый или битый
# каталог иначе считался бы собранным (задача 018, находка 1).
rl_env_done() {
    local stack="$1"
    [ -x "$(rl_env_dir "$stack")/bin/python" ] \
        && [ "$(cat "$RUN/rl-env-$stack.ok" 2>/dev/null)" = "$(rl_env_hash "$stack")" ]
}
rl_env_install() {
    local stack="$1" dir want
    dir="$(rl_env_dir "$stack")"
    want="$(rl_env_hash "$stack")"
    if rl_env_done "$stack"; then
        ok "окружение $stack уже собрано под текущие requirements"
        return
    fi
    # Уже ставится (после --only/--from сценарий мог перезапуститься) —
    # второй установщик в тот же каталог сломал бы первый.
    if [ -f "$RUN/rl-env-$stack.pid" ] && kill -0 "$(cat "$RUN/rl-env-$stack.pid")" 2>/dev/null; then
        ok "окружение $stack уже ставится (pid $(cat "$RUN/rl-env-$stack.pid"))"
        return
    fi
    rm -f "$RUN/rl-env-$stack.ok"
    (
        uv venv --python 3.11 "$dir" >/dev/null 2>&1 \
            && uv pip install -p "$dir" -r "$(rl_env_reqs "$stack")" \
            && uv pip install -p "$dir" -r deploy/requirements-rl-app.txt \
            && echo "$want" > "$RUN/rl-env-$stack.ok"
    ) > "$RUN/rl-env-$stack.log" 2>&1 &
    echo $! > "$RUN/rl-env-$stack.pid"
    ok "окружение $stack ставится фоном: $RUN/rl-env-$stack.log"
}
# Готово ли окружение; ждёт по своему pid, а не по совпадению строки
# в списке процессов (задача 018, находка 2: «.venv-rl» совпадает
# и с «.venv-rl-vllm», а между двумя установками процесса нет вовсе).
rl_env_ready() {
    local stack="$1" pid
    rl_env_done "$stack" && return 0
    # Шаг R0 может выполняться отдельно (--only R0): там установки ещё не было.
    rl_env_install "$stack"
    pid="$(cat "$RUN/rl-env-$stack.pid" 2>/dev/null)"
    for _ in $(seq 1 90); do
        rl_env_done "$stack" && return 0
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || break
        sleep 30
    done
    rl_env_done "$stack"
}

step_D1() {
    say "D1. Группы генераций 4B"
    ensure_4b
    # Дешёвая проверка до основного прогона: если размышление не погашено,
    # все ответы уйдут за ворота, и двадцать минут карты пропадут.
    rm -f "$RUN/smoke.summary.json"
    run "разминка генерации" uv run python scripts/sample_groups.py \
        --dataset "$OUT/mml-test.jsonl" --questions 2 --n 2 --out "$RUN/smoke.jsonl"
    run "разминка: ворота" uv run python - "$RUN/smoke.summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
gated = {k: v for k, v in summary["ворота"].items() if k != "прошёл"}
print("разминка:", summary["ворота"])
if sum(gated.values()) == summary["ответов"]:
    print("все ответы за воротами — проверьте, погашено ли размышление модели")
    raise SystemExit(1)
PY
    rm -f "$OUT/groups-4b.summary.json"
    run "группы генераций" uv run python scripts/sample_groups.py \
        --dataset "$OUT/mml-test.jsonl" --questions 20 --n 8 \
        --out "$OUT/groups-4b.jsonl" --sheet "$OUT/groups-4b"
    run "Д2: сигнал для GRPO" uv run python - "$OUT/groups-4b.summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
share = summary["доля групп без разброса"]
print(f"Д2: доля групп без разброса {share} — {'ок' if share <= 0.5 else 'НЕТ СИГНАЛА: см. план'}")
PY
    warn "Д1 — ручная: скачать $OUT/groups-4b-sheet.md И $OUT/groups-4b-key.json"
    warn "  оценить лист вслепую (ключ не открывать), сложить оценки в groups-4b-grades.json"
    warn "  вида {\"1.1\": 3, \"1.2\": 0}, затем на ноутбуке:"
    warn "  python scripts/reward_agreement.py --key groups-4b-key.json --grades groups-4b-grades.json"
    done_mark D1
}

step_D4() {
    say "D4. Удвоенная разметка по слепку"
    ensure_4b
    local common=(QA_SYSTEM_PROMPT="$(cat "$PROMPT_V4")" PROMPT_VERSION=v4 LLM_CONTEXT_WINDOW=16384)
    # Обе ветки задают флаг явно: иначе значение из .env сравнивало бы
    # одинаковые режимы и разница вышла бы нулевой «по построению».
    run "ответы без нормализации" env "${common[@]}" CONTEXT_NORMALIZE_MATH_DELIMITERS=false \
        uv run rag-textbook eval answers --from-trace "$TRACE_MML" --no-judge --label day1-4b-base
    run "ответы с нормализацией" env "${common[@]}" CONTEXT_NORMALIZE_MATH_DELIMITERS=true \
        uv run rag-textbook eval answers --from-trace "$TRACE_MML" --no-judge --label day1-4b-mathnorm
    local metrics
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    run "перенос формул по обеим ячейкам" uv run python scripts/rl_score_saved.py \
        "$metrics/answers_day1-4b-base.json" "$metrics/answers_day1-4b-mathnorm.json" \
        --trace "$TRACE_MML" --json "$RUN/d4-score.json"
    done_mark D4
}

step_G4() {
    say "G4. Затравки графа (попутно)"
    ensure_4b
    run "сравнение затравок" uv run rag-textbook eval ab --experiment graph_seed_both
    warn "критерий: связывающие recall@16 ≥ +0.010, интервал без нуля, одношаговые не хуже −0.005"
    done_mark G4
}

ingest_group() {
    local dir="$1" collection="$2" sparse="$3" method="$4" lang="$5" stages="$6"
    say "  $dir → $collection ($stages, MinerU $method/$lang)"
    run "индексация $dir ($stages)" env "${LIB_ENV[@]}" PDF_DIR="documents/library/$dir" \
        QDRANT_COLLECTION="$collection" QDRANT_SPARSE_LANGUAGE="$sparse" \
        MINERU_METHOD="$method" MINERU_LANG="$lang" \
        uv run rag-textbook ingest --stages "$stages" --monitor
}

step_L1() {
    say "L1. Разбор библиотеки"
    # MinerU получает карту целиком: и SGLang, и Infinity гасятся.
    llm_off
    search_off
    ingest_group ru     library_ru russian auto east_slavic parse
    ingest_group ru-ocr library_ru russian ocr  east_slavic parse
    ingest_group en     library_en english auto en          parse
    done_mark L1
}

step_L2() {
    say "L2. Проверка разбора, нарезка, векторы"
    run "проверка текста книг" uv run python scripts/check_parsed_text.py
    # Все книги манифеста должны быть разобраны: пропущенную проверка текста
    # не заметит, потому что смотрит только на найденное.
    run "все книги разобраны" uv run python - <<'PY'
import json
from pathlib import Path
from rag_textbook.config import Settings

expected = {p.stem for p in Path("documents/library").rglob("*.pdf")}
parsed = {d.name.split("__", 1)[0] for d in Path(Settings().paths.parsed_dir).glob("*/blocks.json")}
missing = sorted(expected - {p for p in parsed})
print(f"книг в каталоге {len(expected)}, разобрано {len(expected) - len(missing)}")
if missing:
    print("не разобраны:", ", ".join(missing))
    raise SystemExit(1)
PY
    # Векторизация обращается к Infinity — он остановлен на L1.
    search_on
    ingest_group ru     library_ru russian auto east_slavic chunk,embed
    ingest_group ru-ocr library_ru russian ocr  east_slavic chunk,embed
    ingest_group en     library_en english auto en          chunk,embed
    run "коллекции библиотеки собраны" env "${LIB_ENV[@]}" uv run python - <<'PY'
from rag_textbook.config import Settings
from qdrant_client import QdrantClient

settings = Settings()
client = QdrantClient(url=settings.vector_store.url, api_key=settings.vector_store.api_key.get_secret_value() or None)
for name in ("library_ru", "library_en"):
    info = client.get_collection(name)
    sparse = list((info.config.params.sparse_vectors or {}).keys())
    print(f"{name}: точек {info.points_count}, разреженные векторы {sparse or 'НЕТ'}")
    assert info.points_count > 0, f"{name} пуста"
    assert sparse, f"{name} без разреженных векторов — лексический канал не заработает"
PY
    done_mark L2
}

step_Q1() {
    say "Q1. Вопросы для обучения (9B)"
    search_off
    model_9b_alone
    # Предел параллельных запросов клиента — отдельная настройка, иначе
    # --workers упрётся в 4.
    run "генерация вопросов" env "${LIB_ENV[@]}" QDRANT_COLLECTION=library_ru LLM_MAX_CONCURRENCY=16 \
        uv run rag-textbook goldset build --single 2000 --multihop 500 --workers 16 \
        --exclude-doc "$MML_DOC" --output "$OUT/train-ru.json" --seed 20260928
    run "вопросы читаются" uv run python - "$OUT/train-ru.json" <<'PY'
import sys
from pathlib import Path
from rag_textbook.evaluation.goldset import load_goldset

questions = load_goldset(Path(sys.argv[1]))
print(f"вопросов: {len(questions)}")
if len(questions) < 500:
    print("слишком мало — смотреть разбор исходов в журнале выше")
    raise SystemExit(1)
PY
    done_mark Q1
}

step_Q2() {
    say "Q2. Аудит, слепок, эпизоды"
    # Поиск с поднятой 4B, как в сервисе: маршрут и переписывание запроса
    # могут обращаться к модели.
    ensure_4b
    # Аудит по умолчанию берёт ОДИН файл фрагментов и объявил бы остальные
    # вопросы битыми. Собираем общий корпус.
    run "общий корпус фрагментов" uv run python - "$RUN/library-chunks.json" <<'PY'
import json, sys
from pathlib import Path
from rag_textbook.config import Settings
from rag_textbook.rl.env import load_chunks

chunks = load_chunks(Path(Settings().paths.parsed_dir))
Path(sys.argv[1]).write_text(
    json.dumps([c.model_dump(mode="json") for c in chunks.values()], ensure_ascii=False),
    encoding="utf-8",
)
print(f"фрагментов в корпусе: {len(chunks)}")
PY
    run "аудит вопросов" uv run rag-textbook goldset audit --path "$OUT/train-ru.json" \
        --chunks "$RUN/library-chunks.json" --write "$OUT/train-ru-clean.json"
    run "слепок поиска" env "${LIB_ENV[@]}" QDRANT_COLLECTION=library_ru EVAL_TRACE_RERANK_ALL=true \
        uv run rag-textbook eval run --goldset "$OUT/train-ru-clean.json" \
        --trace "$OUT/train-ru-trace.jsonl" --label day1-train-ru
    rm -f "$OUT/lib-ru-train.jsonl" "$OUT/lib-ru-test.jsonl"
    run "эпизоды RL" uv run python scripts/rl_dataset.py --trace "$OUT/train-ru-trace.jsonl" \
        --goldset "$OUT/train-ru-clean.json" --prompt "$PROMPT_V4" \
        --out "$OUT/lib-ru" --test-docs "$MML_DOC"
    local episodes
    episodes=$(wc -l < "$OUT/lib-ru-train.jsonl")
    [ "$episodes" -gt 0 ] || die "эпизодов для обучения нет"
    [ "$(wc -l < "$OUT/lib-ru-test.jsonl")" -eq 0 ] \
        || warn "в обучающие слепки попала MML — эпизоды ушли в тест; разобраться до спринта 3"
    if [ "$episodes" -ge 2000 ]; then ok "Д5: эпизодов $episodes"; else warn "Д5: эпизодов $episodes < 2000 — добрать во второй день"; fi
    done_mark Q2
}

step_R0() {
    say "R0. Окружения RL"
    STACKS=()
    for stack in vllm unsloth; do
        if ! rl_env_ready "$stack"; then
            tail -20 "$RUN/rl-env-$stack.log" 2>/dev/null
            warn "окружение $stack не собралось — пропускаю его в P1"
            continue
        fi
        local dir
        dir="$(rl_env_dir "$stack")"
        if [ "$stack" = unsloth ]; then
            try "импорты ($stack)" "$dir/bin/python" - <<'PY'
import unsloth  # noqa: F401  — до trl, как в train_grpo.py
import datasets, peft, torch, transformers, trl
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "trl", trl.__version__,
      "transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
assert torch.cuda.is_available(), "карта не видна из окружения обучения"
PY
        else
            try "импорты ($stack)" "$dir/bin/python" - <<'PY'
import datasets, peft, torch, transformers, trl, vllm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "trl", trl.__version__,
      "transformers", transformers.__version__, "vllm", vllm.__version__,
      "peft", peft.__version__, "datasets", datasets.__version__)
assert torch.cuda.is_available(), "карта не видна из окружения обучения"
PY
        fi
        [ "$LAST_OK" = 1 ] || { warn "связка $stack выбывает: импорты"; continue; }
        try "отпечаток награды ($stack)" "$dir/bin/python" scripts/reward_fingerprint.py \
            --dataset "$OUT/mml-test.jsonl" --check "$OUT/reward-fingerprint.json"
        [ "$LAST_OK" = 1 ] || { warn "связка $stack выбывает: награда считает иначе"; continue; }
        STACKS+=("$stack")
    done
    [ "${#STACKS[@]}" -gt 0 ] || die "ни одно окружение обучения не собралось"
    printf '%s\n' "${STACKS[@]}" > "$RUN/stacks.txt"
    ok "к пробе готовы: ${STACKS[*]}"
    done_mark R0
}

step_P1() {
    say "P1. Проба GRPO"
    llm_off
    search_off
    # Веса могут ещё качаться фоном — обучение ждать их не должно.
    if [ -f "$RUN/weights.pid" ] && kill -0 "$(cat "$RUN/weights.pid")" 2>/dev/null; then
        warn "жду загрузку весов (фон): $RUN/weights.log"
        wait "$(cat "$RUN/weights.pid")" 2>/dev/null
    fi
    grep -q "загрузка весов завершена" "$RUN/weights.log" 2>/dev/null \
        || warn "веса могли не докачаться — обучение скачает недостающее само"
    HF_HOME="$(hf_cache_dir)"
    export HF_HOME
    local stacks=() any=0 FRESH=()
    mapfile -t stacks < "$RUN/stacks.txt" 2>/dev/null
    # Пустой или отсутствующий список — это не «возьмём unsloth молча»:
    # связку должен был назвать R0 (задача 018, находка 4).
    [ "${#stacks[@]}" -gt 0 ] || die "нет $RUN/stacks.txt — выполните шаг R0"
    # Сначала быстрая связка: время шага решает бюджет спринтов 3, 5 и 6.
    for stack in "${stacks[@]}"; do
        local dir extra=() gens status=1
        dir="$(rl_env_dir "$stack")"
        [ "$stack" = vllm ] && extra=(--backend hf --vllm --vllm-memory 0.3)
        for attempt in 4 2; do
            gens=$attempt
            local label="$stack-g$gens"
            rm -rf "runs/probe-4b-$label"
            "$dir/bin/python" scripts/train_grpo.py --dataset "$OUT/lib-ru-train.jsonl" \
                --probe --out "runs/probe-4b-$label" --num-generations "$gens" \
                --grad-accum "$gens" "${extra[@]}" \
                2>&1 | tee "$RUN/probe-$label.log" | tail -30 | tee -a "$LOG"
            status=${PIPESTATUS[0]}
            [ "$status" = 0 ] && break
            grep -q -i -E "out of memory|outofmemoryerror|cuda error: out" \
                "$RUN/probe-$label.log" || break
            [ "$gens" = 2 ] && break
            warn "$stack: OOM при $gens генерациях — пробую меньше"
        done
        if [ "$status" != 0 ]; then
            warn "$stack: проба не прошла — $RUN/probe-$stack-g$gens.log"
            continue
        fi
        any=1
        ok "$stack, $gens генераций:"
        cat "runs/probe-4b-$stack-g$gens/probe.json" | tee -a "$LOG"
        warn "прочитать runs/probe-4b-$stack-g$gens/samples.jsonl до выводов"
        FRESH+=("runs/probe-4b-$stack-g$gens/probe.json")
        # Та же проба с коротким ответом: разность времён шага показывает,
        # сколько в шаге стоит генерация. Без этого числа нельзя решить
        # ни про связку, ни про вторую карту — только гадать.
        rm -rf "runs/probe-4b-$stack-g$gens-short"
        if "$dir/bin/python" scripts/train_grpo.py --dataset "$OUT/lib-ru-train.jsonl" \
            --probe --out "runs/probe-4b-$stack-g$gens-short" --num-generations "$gens" \
            --grad-accum "$gens" --max-completion-length 96 "${extra[@]}" \
            > "$RUN/probe-$stack-g$gens-short.log" 2>&1
        then
            FRESH+=("runs/probe-4b-$stack-g$gens-short/probe.json")
        else
            warn "$stack: короткая проба не прошла, доля генерации не посчитается"
        fi
    done
    [ "$any" = 1 ] || die "ни одна связка не прошла пробу — журналы в $RUN"
    # Правило выбора записано до аренды (scripts/probe_compare.py).
    # Передаются только пробы этого захода: маска runs/probe-4b-*/ подобрала бы
    # прошлые связки и прошлые числа генераций (задача 018, находка 6).
    run "разбор проб" "$(rl_env_dir "${stacks[0]}")/bin/python" scripts/probe_compare.py \
        "${FRESH[@]}"
    done_mark P1
}

step_Z9() {
    say "Z9. Архив"
    local archive metrics
    archive="$HOME/day1-results-$(date +%Y%m%d-%H%M).tar.gz"
    metrics="$(metrics_dir)" || die "не удалось прочитать METRICS_DIR"
    local paths=(artifacts/rl artifacts/runs/day1)
    for extra in runs "$metrics"; do
        [ -d "$extra" ] && paths+=("$extra")
    done
    tar czf "$archive" -C "$REPO_DIR" "${paths[@]}" 2>>"$LOG" || die "архив не создан: см. $LOG"
    ok "архив: $archive ($(du -h "$archive" | cut -f1)), внутри: ${paths[*]}"
    warn "скачать: scp -i <ключ> root@<ip>:$archive ."
    warn "кэши разбора библиотеки — deploy/backup.ps1; затем сервер можно выключать"
    done_mark Z9
}

for code in $CODES; do
    if should_run "$code"; then
        "step_$code"
    fi
done
say "Готово ($(stamp)). Журнал: $LOG"
