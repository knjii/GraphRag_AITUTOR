#!/usr/bin/env bash
# Дни аренды спринта 3: обучение GRPO с наградой R6 и двумя контролями.
# План и критерии — docs/SERVER-SPRINT-3.md, критерии отказа — раздел 4.5
# docs/RESEARCH-2026-09.md. Каждый шаг отмечается в artifacts/runs/sprint3/done,
# повторный запуск продолжает с несделанного.
#
#   bash deploy/train-day.sh --list
#   bash deploy/train-day.sh --stack vllm --generations 4
#   bash deploy/train-day.sh --only T2
#
# Связка и число генераций берутся из дня №1 (scripts/probe_compare.py),
# а не выбираются здесь заново: правило было записано до того замера.
#
# Порядок задан одним: обучение и сервис не делят карту. Сначала база
# меряется на 4B (шаг B0), потом карта отдаётся обучению (T1–T3), потом
# адаптеры вплавляются и меряются по очереди (M1). Контроли обязательны:
# без них прирост верной награды не с чем сравнивать, и результат
# не принимается (2506.10947).

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

RUN="$REPO_DIR/artifacts/runs/sprint3"
OUT="$REPO_DIR/artifacts/rl"
LOG="$RUN/sprint3.log"
mkdir -p "$RUN/done" "$OUT"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
TRACE_MML=capture/session-0819/trace-always.jsonl
TRAIN_SET="$OUT/lib-ru-train.jsonl"
TEST_SET="$OUT/mml-test.jsonl"
STEPS="${STEPS:-300}"
STACK=""; GENERATIONS=""

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

rl_env_dir() { [ "$1" = vllm ] && echo .venv-rl-vllm || echo .venv-rl; }
llm_off() {
    run "остановка SGLang" docker compose --env-file .env \
        -f docker/docker-compose.vllm.yml --profile sglang stop sglang
    ok "видеопамять: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
search_off() {
    run "остановка служб поиска" docker compose --env-file .env \
        -f docker/docker-compose.yml stop infinity ollama
}
serve() {
    run "подъём модели $1" bash deploy/model-swap.sh "$1" 0.75
}

# ------------------------------------------------------------------- план
PLAN=$(cat <<'PLAN'
S0|Проверка: связка, наборы, отпечаток награды, место на диске, адаптеры не поверх старых
B0|Точка отсчёта: базовая 4B на тестовых эпизодах по слепку — «хотя бы одна формула» заново
T1|Обучение с верной наградой R6
T2|Контроль: случайная награда
T3|Контроль: только ворота формата
M1|Вплавить три адаптера в веса и замерить каждый на тех же эпизодах
V1|Вердикт по критериям 4.5: парный бутстрап, контроли, доля взлома
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
        --stack) STACK="${2:-}"; shift 2 ;;
        --generations) GENERATIONS="${2:-}"; shift 2 ;;
        --steps) STEPS="${2:-}"; shift 2 ;;
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

# Связка и число генераций не угадываются: их назвал замер дня №1.
[ -n "$STACK" ] || STACK="$(cat "$RUN/stack" 2>/dev/null)"
[ -n "$GENERATIONS" ] || GENERATIONS="$(cat "$RUN/generations" 2>/dev/null)"
[ -n "$STACK" ] && [ -n "$GENERATIONS" ] \
    || die "укажите --stack и --generations по выводу scripts/probe_compare.py (день №1, шаг P1)"
case "$STACK" in vllm|unsloth) ;; *) die "связка: vllm или unsloth, получено '$STACK'" ;; esac
printf '%s' "$STACK" > "$RUN/stack"; printf '%s' "$GENERATIONS" > "$RUN/generations"

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
step_S0() {
    say "S0. Проверка перед стартом"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
    df -h "$REPO_DIR" | tail -1 | tee -a "$LOG"
    ok "связка $STACK, генераций $GENERATIONS, шагов $STEPS"
    for required in "$TRAIN_SET" "$TEST_SET" "$OUT/reward-fingerprint.json" "$TRACE_MML"; do
        [ -s "$required" ] || die "нет или пуст $required"
    done
    local dir; dir="$(rl_env_dir "$STACK")"
    [ -x "$dir/bin/python" ] || die "нет окружения $dir — соберите его как в дне №1 (шаг E0)"
    # Тот же отпечаток, что проверен вручную: иначе награда считает иначе,
    # чем то, с чем согласился человек (например, без лемматизации).
    run "отпечаток награды" "$dir/bin/python" scripts/reward_fingerprint.py \
        --dataset "$TEST_SET" --check "$OUT/reward-fingerprint.json"
    # Три прогона обязаны стартовать с одной точки: адаптер поверх старого
    # каталога дал бы дообучение, а не контроль.
    for kind in main random format; do
        [ -e "runs/grpo-$kind" ] && die "runs/grpo-$kind уже есть — уберите или переименуйте"
    done
    # Обучение примерно на 12 ГиБ весов и три вплавленные копии по ~8 ГиБ.
    local free; free=$(df -BG --output=avail "$REPO_DIR" | tail -1 | tr -dc '0-9')
    [ "${free:-0}" -ge 60 ] || die "на диске ${free} ГиБ — трём вплавленным моделям нужно ≥ 60"
    done_mark S0
}

step_B0() {
    say "B0. Точка отсчёта на тестовых эпизодах"
    # 0.305 снята другими настройками генерации; критерий спринта задан
    # приростом, поэтому база пересчитывается здесь и сейчас.
    search_off
    measure base-4b "$BASE_MODEL"
    done_mark B0
}

step_T1() { say "T1. Обучение: верная награда"; llm_off; search_off; train_one main; done_mark T1; }
step_T2() { say "T2. Контроль: случайная награда"; llm_off; train_one random; done_mark T2; }
step_T3() { say "T3. Контроль: только формат"; llm_off; train_one format; done_mark T3; }

step_M1() {
    say "M1. Вплавление адаптеров и замер"
    local dir; dir="$(rl_env_dir "$STACK")"
    for kind in main random format; do
        [ -d "runs/grpo-$kind/merged" ] || run "вплавление ($kind)" \
            "$dir/bin/python" scripts/merge_adapter.py \
            --adapter "runs/grpo-$kind/adapter" --out "runs/grpo-$kind/merged"
        measure "rl-$kind" "/runs/grpo-$kind/merged"
    done
    done_mark M1
}

step_V1() {
    say "V1. Вердикт по критериям, записанным до прогона"
    run "сравнение" uv run python scripts/sprint3_verdict.py \
        --metrics "$(uv run python -c 'from rag_textbook.config import Settings; print(Settings().paths.metrics_dir)')" \
        --base base-4b --main rl-main --controls rl-random rl-format \
        --out "$RUN/verdict.json"
    done_mark V1
}

step_Z9() {
    say "Z9. Архив"
    local archive="$RUN/sprint3-results.tar.gz"
    tar czf "$archive" \
        --exclude='*/merged' --exclude='*/checkpoint-*' \
        runs/grpo-main runs/grpo-random runs/grpo-format \
        "$RUN/verdict.json" "$LOG" 2>/dev/null
    ok "скачать: $archive ($(du -h "$archive" | cut -f1))"
    warn "карту можно гасить только после того, как архив скачан"
    done_mark Z9
}

# Помощники держатся ниже шагов: bash читает весь файл до первого вызова,
# и порядок чтения тогда совпадает с порядком выполнения — сначала S0
# проверяет входы, потом их кто-то использует.
#
# Обучение одной наградой. Контроли отличаются только --reward, всё
# остальное обязано совпадать: иначе сравнение сравнивает не награды.
train_one() {
    local kind="$1" dir out
    dir="$(rl_env_dir "$STACK")"
    out="runs/grpo-$kind"
    local extra=()
    [ "$STACK" = vllm ] && extra=(--backend hf --vllm --vllm-memory 0.3)
    run "обучение ($kind)" "$dir/bin/python" scripts/train_grpo.py \
        --dataset "$TRAIN_SET" --model "$BASE_MODEL" --reward "$kind" \
        --steps "$STEPS" --num-generations "$GENERATIONS" --grad-accum "$GENERATIONS" \
        --out "$out" --sample-every 10 "${extra[@]}"
    [ -d "$out/adapter" ] || die "$kind: адаптер не сохранён"
    # Правило проекта: читать генерации до замера, а не после.
    warn "прочитать $out/samples.jsonl — три ответа глазами, до любых чисел"
}

# Замер обученной модели на тестовых эпизодах MML по слепку контекста:
# поиск не выполняется, сравниваются именно генераторы.
measure() {
    local label="$1" model="$2"
    serve "$model"
    run "замер ответов ($label)" uv run rag-textbook eval answers \
        --label "$label" --from-trace "$TRACE_MML" --no-judge
}

for code in $CODES; do
    should_run "$code" && "step_$code"
done
ok "готово ($(stamp))"
