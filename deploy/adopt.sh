#!/usr/bin/env bash
# Закрепляет настройки, подтверждённые измерением, и снимает итоговый прогон.
#
#   bash deploy/adopt.sh                     принять подтверждённое
#   bash deploy/adopt.sh --degree 40         заодно сменить порог отсечения хабов
#
# Принимается только то, что показало значимый прирост на парном сравнении.
# Остальное остаётся в значениях точки отсчёта — включая настройки, которые
# хорошо выглядели на офлайн-замере: он меряет графовый канал в изоляции,
# и его положительные предсказания на продукт не переносятся.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

DEGREE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --degree) DEGREE="$2"; shift 2 ;;
        *) echo "Неизвестный аргумент: $1" >&2; exit 1 ;;
    esac
done

clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; }

set_key() {
    if grep -qE "^$1=" .env; then
        sed -i "s|^$1=.*|$1=$2|" .env
    else
        printf '%s=%s\n' "$1" "$2" >> .env
    fi
    printf '    %-34s %s\n' "$1" "$2"
}

printf '\n\033[1;34m=== принимаю подтверждённое ===\033[0m\n'
# Замер: recall 0.859 → 0.880, значимо, пять вопросов лучше и ни одного хуже.
# Выигрывают не только многошаговые: эвристика загоняла в граф и простые
# вопросы, добавляя им шума, поэтому у одношаговых прирост наибольший.
set_key RETRIEVAL_ROUTER_MODE llm

if [ -n "$DEGREE" ]; then
    set_key GRAPH_MAX_ENTITY_DEGREE "$DEGREE"
    printf '\n\033[1;34m=== пересобираю граф под новый порог ===\033[0m\n'
    bash deploy/reset-stages.sh graphed | clean
    uv run rag-textbook ingest --stages graph --no-monitor 2>&1 | clean
    uv run rag-textbook graph stats 2>&1 | clean
fi

printf '\n\033[1;34m=== отклонено измерением, оставлено как было ===\033[0m\n'
grep -E '^(RETRIEVAL_MIN_GRAPH_DOCS|RETRIEVAL_GRAPH_CANDIDATE_QUOTA|GRAPH_HOP_DECAY|GRAPH_PASSAGE_IDF_ENABLED)=' .env | sed 's/^/    /'

printf '\n\033[1;34m=== итоговый прогон ===\033[0m\n'
uv run rag-textbook eval run --label adopted 2>&1 | clean
