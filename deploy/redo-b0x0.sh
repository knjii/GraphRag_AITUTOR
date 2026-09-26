#!/usr/bin/env bash
# Пересъёмка B0 и X0 после правки стенда (журнал, запись 39).
# B0 снимает отметки всех следующих шагов; P0, Q0, K0 от правки не зависят —
# их отметки возвращаются, Z9 пересобирает архив с новыми B0/X0.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/rag_textbook
DONE=artifacts/runs/day2/done
RUN=artifacts/runs/day2
keep=$(cd "$DONE" && ls P0* Q0* K0* 2>/dev/null || true)
[ -f "$RUN/trace-r2.jsonl" ] && cp "$RUN/trace-r2.jsonl" "$RUN/trace-r2.pre-tiefix.jsonl"
[ -f "$RUN/fidelity.json" ] && cp "$RUN/fidelity.json" "$RUN/fidelity.pre-tiefix.json"
bash deploy/day2.sh --only B0
bash deploy/day2.sh --only X0
for mark in $keep; do touch "$DONE/$mark"; done
[ -f "$RUN/fidelity.ok" ] || { echo "СТОП: стенд по-прежнему не годен"; exit 1; }
bash deploy/day2.sh --only Z9
echo "REDO ГОТОВО"
