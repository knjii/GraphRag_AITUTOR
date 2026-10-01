#!/usr/bin/env bash
# Серия L (docs/HYPOTHESES.md): классификатор решений laya-multilingual,
# дообученный на MuSiQue train, собирает цепочку фрагментов поверх выдачи
# на MuSiQue-300 — A1 поверх ours-current, A2 поверх множества SetR
# (контроль — SetR, добитый реранкером до k). Гейт Б1 пишет вероятность
# «хватает» по пятёрке; смесь с SEAL считается на ноутбуке после того, как
# владелец выберет критерий Б1 (файл гейта этого выбора не подсказывает).
# P1/P2 — чистый перезамер задержки SEAL и базы (deploy/bench-ours.sh).
#
#   bash deploy/laya-bench.sh --list
#   bash deploy/laya-bench.sh                 всё по порядку, с места остановки
#   bash deploy/laya-bench.sh --only T1       один шаг заново
#
# Запускать в tmux:
#   tmux has-session -t laya 2>/dev/null || tmux new-session -d -s laya \
#       'bash deploy/laya-bench.sh; exec bash'
#
# Нужно на сервере (приезжает архивом): выдачи и ответы серии S в
# artifacts/runs/bench/musique-300, набор artifacts/bench/musique-300,
# данные artifacts/laya/data (scripts/laya_data.py, sha256 в manifest.json).
#
# Карта одна (RTX 3090): P1 идёт первым на 4B (ничего чужого на карте),
# классификатор учится и работает без 4B и 9B, 9B поднимается на A3 и P2.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi

BENCH=musique-300
BUNDLE="$REPO_DIR/artifacts/bench/$BENCH"
RUN="$REPO_DIR/artifacts/runs/bench/$BENCH"
OURS="$RUN/ours"
ANSWERS="$RUN/answers"
LAYA="$REPO_DIR/artifacts/laya"
DATA="$LAYA/data"
MODEL="$LAYA/model"
OUT="$RUN/laya"
ALIASES="$REPO_DIR/artifacts/bench/$BENCH-aliases.json"
LOG="$OUT/laya-bench.log"
DONE="$OUT/done"
VENV="$REPO_DIR/.venv-laya"
PY="$VENV/bin/python"
BASE_MODEL=convaiinnovations/laya-multilingual
LAYA_VERSION=0.3.22
mkdir -p "$DONE" "$OUT"

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*" | tee -a "$LOG"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
die()  { printf '\n\033[1;31mСТОП: %s\033[0m\n' "$*" | tee -a "$LOG" >&2; exit 1; }
run() {
    local desc="$1"; shift
    "$@" 2>&1 | grep -v -E 'Warning: You are sending unauthenticated|HTTP Request:' | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die "$desc — код возврата $rc (журнал: $LOG)"
}

# Карта нужна классификатору целиком: генератор и 4B снимаются.
free_gpu() {
    docker rm -f judge generator >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.vllm.yml --profile sglang stop sglang >/dev/null 2>&1
    docker compose --env-file .env -f docker/docker-compose.yml stop infinity ollama >/dev/null 2>&1
    true
}

step_C0() {  # проверки входа: набор, выдачи, ответы серии S, данные L
    for f in "$BUNDLE/questions.jsonl" "$BUNDLE/chunks.jsonl" \
             "$OURS/rankings-ours-current.jsonl" "$OURS/rankings-ours-current+seal.jsonl" \
             "$OURS/rankings-ours-current+setr.jsonl" \
             "$ANSWERS/answers-ours-current.jsonl" "$ANSWERS/answers-ours-current+seal.jsonl" \
             "$DATA/train.jsonl" "$DATA/heldout.jsonl" "$DATA/manifest.json"; do
        [ -s "$f" ] || die "нет $f"
    done
    local want got
    for part in train heldout; do
        want=$(python3 -c "import json;print(json.load(open('$DATA/manifest.json'))['$part']['sha256'])")
        got=$(sha256sum "$DATA/$part.jsonl" | cut -d' ' -f1)
        [ "$want" = "$got" ] || die "данные $part не совпали по sha256"
    done
    [ "$(wc -l < "$OURS/rankings-ours-current.jsonl")" = 300 ] || die "выдача ours-current не на 300 вопросов"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"
    ok "вход на месте, данные сверены"
}

step_E0() {  # окружение .venv-laya: laya, transformers 5; torch — из .venv-rl/.venv или с зеркала
    if [ -x "$PY" ] && "$PY" -c "import laya, torch; assert laya.__version__ == '$LAYA_VERSION'; assert torch.cuda.is_available()" 2>/dev/null; then
        ok "окружение уже собрано"
    else
        [ -n "${UV_DEFAULT_INDEX:-}" ] || die "UV_DEFAULT_INDEX не задан: сначала deploy/bootstrap.sh"
        rm -rf "$VENV"
        run "venv" uv venv --python 3.11 "$VENV"
        run "laya и зависимости" uv pip install -p "$VENV" "laya==$LAYA_VERSION" "transformers>=5,<6" \
            safetensors "huggingface_hub[cli]>=1" numpy
        local site linked=""
        site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
        if ! "$PY" -c 'import torch' 2>/dev/null; then
            for other in .venv-rl .venv; do
                [ -x "$other/bin/python" ] || continue
                local other_site
                other_site=$("$other/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null) || continue
                [ -d "$other_site/torch" ] || continue
                echo "$other_site" > "$site/zz-torch-from-$(basename "$other").pth"
                if "$PY" -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
                    linked="$other"; break
                fi
                rm -f "$site/zz-torch-from-$(basename "$other").pth"
            done
            [ -n "$linked" ] && ok "torch взят из $linked" \
                || run "torch с зеркала" uv pip install -p "$VENV" torch
        fi
    fi
    run "проверка окружения" "$PY" -c "
import torch, transformers, laya
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'laya', laya.__version__)
assert torch.cuda.is_available()"
    run "веса $BASE_MODEL" "$VENV/bin/hf" download "$BASE_MODEL" --quiet
}

step_T0() {  # контроль без дообучения: AUC базовой модели на 500 отложенных вопросах
    free_gpu
    run "проверка базовой" "$PY" scripts/laya_eval.py heldout --model "$BASE_MODEL" \
        --data "$DATA" --out "$OUT/heldout-base.json"
}

step_T1() {  # дообучение laya-multilingual (RLCD авторов, 3 эпохи, температура на отложенной части)
    free_gpu
    # Проба на 256 строках: путь bf16 на карте, память и сохранение — за пару минут,
    # а не через полчаса полного обучения.
    run "проба обучения" "$PY" scripts/laya_train.py --data "$DATA" --out "$LAYA/model-smoke" \
        --limit 256 --epochs 1
    run "проба загрузки" "$PY" -c "import laya; a = laya.load('$LAYA/model-smoke', device='cuda'); print('загружается')"
    run "дообучение" "$PY" scripts/laya_train.py --data "$DATA" --out "$MODEL"
}

step_T2() {  # действительность: AUC дообученной на отложенных вопросах ≥ 0.70 (записано до замера)
    run "проверка дообученной" "$PY" scripts/laya_eval.py heldout --model "$MODEL" \
        --data "$DATA" --out "$OUT/heldout-ft.json"
    "$PY" - "$OUT/heldout-ft.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
report = json.load(open(sys.argv[1]))
auc = report["pair"]["auc"]
print(f"AUC L1 {auc:.3f}, AUC L2 {report['enough']['auc']:.3f}")
raise SystemExit(0 if auc >= 0.70 else 3)
PY
    [ "${PIPESTATUS[0]}" = 0 ] || die "AUC L1 < 0.70 — замер недействителен (не обучилось), дальше не идём"
}

# P1/P2 не связаны с Laya: их сбой пишется в журнал и не останавливает серию L.
# Повтор — bash deploy/laya-bench.sh --only P1 (или P2).
optional() {  # шаг, описание, команда…
    local step="$1" desc="$2"; shift 2
    "$@" 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" = 0 ]; then
        rm -f "$OUT/$step.failed"
    else
        touch "$OUT/$step.failed"
        warn "$desc не удался — серия L идёт дальше; повтор: --only $step"
    fi
}

step_P1() {  # чистая задержка поиска: база и SEAL по одному, на карте только 4B и Infinity
    optional P1 "перезамер задержки" env CLEAN_SESSION=laya bash deploy/bench-ours.sh --only P1
}

chain_ms() {  # медиана времени цепочки на вопрос (предел 2 с, записан до замера)
    "$PY" - "$1" <<'PY' 2>&1 | tee -a "$LOG"
import json, statistics, sys
ms = [json.loads(l)["chain_ms"] for l in open(sys.argv[1], encoding="utf-8")]
median = statistics.median(ms) / 1000
print(f"цепочка: медиана {median:.2f} с на вопрос, 90-й перцентиль {sorted(ms)[int(0.9 * len(ms))] / 1000:.2f} с"
      + ("" if median <= 2.0 else " — ПРЕДЕЛ 2 с ПРЕВЫШЕН"))
PY
}

step_C1() {  # A1: цепочка поверх ours-current и отсев ctx_all@5 ≥ 0.430
    free_gpu
    run "цепочка" "$PY" scripts/laya_eval.py chain --model "$MODEL" --bundle "$BUNDLE" \
        --rankings "$OURS/rankings-ours-current.jsonl" --out "$OURS/rankings-ours-current+chain.jsonl"
    run "отсев" "$PY" scripts/laya_offline.py coverage --bundle "$BUNDLE" \
        --run base="$OURS/rankings-ours-current.jsonl" \
        --run chain="$OURS/rankings-ours-current+chain.jsonl" --out "$OUT/coverage.json"
    chain_ms "$OURS/rankings-ours-current+chain.jsonl"
}

step_C2() {  # A2: цепочка от множества SetR против SetR, добитого реранкером (без отсева)
    free_gpu
    run "контроль setr-fill" "$PY" scripts/laya_eval.py setrfill \
        --rankings "$OURS/rankings-ours-current+setr.jsonl" --out "$OURS/rankings-ours-current+setr-fill.jsonl"
    run "цепочка от SetR" "$PY" scripts/laya_eval.py chain --model "$MODEL" --bundle "$BUNDLE" --seed selected \
        --rankings "$OURS/rankings-ours-current+setr.jsonl" --out "$OURS/rankings-ours-current+setr+chain.jsonl"
    run "покрытие A2" "$PY" scripts/laya_offline.py coverage --bundle "$BUNDLE" \
        --run setr-fill="$OURS/rankings-ours-current+setr-fill.jsonl" \
        --run setr+chain="$OURS/rankings-ours-current+setr+chain.jsonl" --out "$OUT/coverage-setr.json"
    chain_ms "$OURS/rankings-ours-current+setr+chain.jsonl"
}

step_X1() {  # объясняющая: AUC связи на парах из нашего пула по разметке теста
    run "пары теста" "$PY" scripts/laya_eval.py testpairs --model "$MODEL" --bundle "$BUNDLE" \
        --rankings "$OURS/rankings-ours-current.jsonl" --out "$OUT/testpairs.json"
}

screen_passed() {
    "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/coverage.json'))['chain']['ctx_all'] >= 0.430 else 1)"
}

# 9B-Q4 на llama.cpp — тот же запуск, что в deploy/bench-ours.sh (A1/A2).
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
    free_gpu
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

answers() {  # имя отчёта, база, затем --run …; проверка ошибок генератора по последней системе
    local label="$1" base="$2"; shift 2
    ensure_9b_alone
    run "ответы $label" env LLM_MODEL=Qwen3.5-9B-UD-Q4_K_XL LLM_CHAT_MODEL= LLM_REASONING_EFFORT=none \
        LLM_MAX_CONCURRENCY=16 uv run python scripts/bench_answers.py --bundle "$BUNDLE" \
        --k 5 --baseline "$base" --workers 16 --max-tokens 1536 --aliases "$ALIASES" \
        --out "$ANSWERS" "$@"
    cp "$ANSWERS/answers-report.json" "$RUN/answers-$label.json"
    local last="${!#}"; last="${last#*=}"
    uv run python - "$ANSWERS/answers-$(basename "$last" .jsonl | sed 's/^rankings-//').jsonl" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
no_answer = sum("answer:" not in (r["raw"] or "").lower() for r in rows)
errors = sum(bool(r.get("error")) for r in rows)
print(f"{sys.argv[1]}: ответов {len(rows)}, без строки Answer {no_answer}, ошибок {errors}")
for r in rows[:3]:
    print("  образец", r["qid"], repr((r["raw"] or "")[-200:]))
raise SystemExit(0 if errors <= 0.02 * len(rows) else 4)
PY
    [ "${PIPESTATUS[0]}" = 0 ] || die "ошибок генератора больше 2% — замер недействителен"
}

step_A3() {  # ответы 9B-Q4 при k=5: A1 (если отсев пройден) и A2 (всегда)
    if screen_passed; then
        answers L-A1 ours-current \
            --run ours-current="$OURS/rankings-ours-current.jsonl" \
            --run ours-current+chain="$OURS/rankings-ours-current+chain.jsonl"
    else
        warn "отсев A1 не пройден — ответы A1 не генерируются, A1 отвергнута на поиске"
    fi
    answers L-A2 ours-current+setr-fill \
        --run ours-current+setr-fill="$OURS/rankings-ours-current+setr-fill.jsonl" \
        --run ours-current+setr+chain="$OURS/rankings-ours-current+setr+chain.jsonl"
}

step_G1() {  # Б1: вероятность «хватает» по пятёрке ours-current (смесь — на ноутбуке)
    free_gpu
    run "гейт" "$PY" scripts/laya_eval.py gate --model "$MODEL" --bundle "$BUNDLE" \
        --rankings "$OURS/rankings-ours-current.jsonl" --out "$OUT/gate-ours-current.jsonl"
}

step_P2() {  # время ответа 9B-Q4 по одному — полный путь для решения о пределе задержки
    optional P2 "время ответа" env CLEAN_SESSION=laya bash deploy/bench-ours.sh --only P2
}

step_Z1() {  # всё для оценки на ноутбуке — одним архивом (одно подключение на выгрузку)
    local files=()
    for f in laya answers-L-A1.json answers-L-A2.json latency-clean.json answers-clean \
             ours/rankings-ours-current+chain.jsonl ours/rankings-ours-current+setr-fill.jsonl \
             ours/rankings-ours-current+setr+chain.jsonl ours/rankings-ours-current@clean.jsonl \
             ours/rankings-ours-current+seal@clean.jsonl \
             answers/answers-ours-current+chain.jsonl answers/answers-ours-current+setr-fill.jsonl \
             answers/answers-ours-current+setr+chain.jsonl; do
        [ -e "$RUN/$f" ] && files+=("$f")
    done
    tar -czf "$REPO_DIR/artifacts/laya-results.tgz" -C "$RUN" --exclude "laya/done" "${files[@]}" \
        -C "$LAYA" model/train_log.json model/rl_agent_config.json \
        || die "не упаковать результаты"
    ok "результаты: $REPO_DIR/artifacts/laya-results.tgz ($(du -h "$REPO_DIR/artifacts/laya-results.tgz" | cut -f1))"
}

STEPS=(C0 E0 P1 T0 T1 T2 C1 C2 X1 G1 A3 P2 Z1)
declare -A DESC=(
    [C0]="проверка входа и sha256 данных"
    [E0]="окружение .venv-laya и веса $BASE_MODEL"
    [P1]="чистая задержка поиска: база и SEAL по одному (4B, ничего чужого)"
    [T0]="контроль: базовая модель на отложенных вопросах"
    [T1]="дообучение на MuSiQue train (L1 и L2)"
    [T2]="действительность: AUC L1 ≥ 0.70"
    [C1]="A1: цепочка поверх ours-current, отсев ctx_all@5 ≥ 0.430, время цепочки"
    [C2]="A2: цепочка от SetR против SetR, добитого реранкером"
    [X1]="объясняющая: AUC связи на парах из нашего пула"
    [G1]="Б1: вероятность «хватает» по пятёрке ours-current"
    [A3]="ответы 9B-Q4: A1 (если отсев пройден) и A2"
    [P2]="время ответа 9B-Q4 по одному на 60 вопросах"
    [Z1]="упаковка результатов в artifacts/laya-results.tgz"
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
ok "готово: $OUT"
