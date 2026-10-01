#!/usr/bin/env bash
# Весь этап 1 + этап 2 на публичном наборе одной командой, без простоя карты.
#
#   tmux has-session -t bench 2>/dev/null || tmux new-session -d -s bench \
#       'bash deploy/bench-all.sh; exec bash'
#
# Порядок:
#   1. Поднять 4B и Infinity (одни на всех до ответов).
#   2. Параллельно: deploy/hipporag-bench.sh (HippoRAG 2 и dense; карту не
#      трогает — CUDA_VISIBLE_DEVICES="") и deploy/bench-ours.sh до R3
#      (наши графы и выдачи). Оба ходят в одну 4B: SGLang держит очередь сам.
#   3. Дождаться HippoRAG 2, затем bench-ours.sh с S1: SetR поверх обеих
#      выдач, метрики, переключение на 9B и ответы.
#
# Повторный запуск продолжает с места: у обоих сценариев свои отметки.
# Провал шлюза HippoRAG 2 не останавливает наши шаги — они пойдут без его
# выдач, это видно в журнале и в сводке (базой станет ours-old).

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
# Закрытый PyPI: индекс из ~/.bashrc (bootstrap.sh), нужен окружению HippoRAG.
if [ -z "${UV_DEFAULT_INDEX:-}" ]; then
    UV_DEFAULT_INDEX="$(sed -n 's/^export UV_DEFAULT_INDEX="\(.*\)"$/\1/p' "$HOME/.bashrc" 2>/dev/null | tail -1)"
    [ -n "$UV_DEFAULT_INDEX" ] && export UV_DEFAULT_INDEX
fi
export BENCH="${BENCH:-musique-300}"
RUN="$REPO_DIR/artifacts/runs/bench/$BENCH"
mkdir -p "$RUN"
LOG="$RUN/bench-all.log"

say() { printf '\n\033[1;35m##### %s (%s)\033[0m\n' "$*" "$(date '+%F %H:%M:%S')" | tee -a "$LOG"; }

say "0. тесты на заглушках (ошибка кода не должна стоить часов карты)"
if ! uv run pytest -q -x tests/test_bench_answers.py tests/test_bench_metrics.py \
        tests/test_set_selection_and_seal.py 2>&1 | tail -3 | tee -a "$LOG"; then
    echo "тесты упали — стоп" | tee -a "$LOG"; exit 1
fi

say "1. 4B и Infinity"
if curl -sf http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q 'Qwen3.5-4B' \
    && curl -sf -o /dev/null http://127.0.0.1:7997/health; then
    echo "уже подняты" | tee -a "$LOG"
else
    docker rm -f judge generator >/dev/null 2>&1
    bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval 2>&1 | tail -5 | tee -a "$LOG"
    bash deploy/services.sh up 2>&1 | tail -5 | tee -a "$LOG"
fi

say "2. параллельно: HippoRAG 2 и наши графы/выдачи (до R3)"
bash deploy/hipporag-bench.sh > "$RUN/hipporag-bench.out" 2>&1 &
hippo=$!
bash deploy/bench-ours.sh --until R3
ours_rc=$?
say "наши выдачи: код $ours_rc; жду HippoRAG 2 (журнал $RUN/hipporag-bench.out)"
wait "$hippo"
hippo_rc=$?
say "HippoRAG 2: код $hippo_rc"
tail -15 "$RUN/hipporag-bench.out" | tee -a "$LOG"
[ "$ours_rc" = 0 ] || { echo "наши шаги упали — стоп до разбора (журнал $RUN/bench-ours.log)" | tee -a "$LOG"; exit 1; }

say "3. SetR, метрики, ответы 9B"
bash deploy/bench-ours.sh
rc=$?
say "готово, код $rc: сводки $RUN/metrics-all.json, $RUN/answers-stage*.json"
exit "$rc"
