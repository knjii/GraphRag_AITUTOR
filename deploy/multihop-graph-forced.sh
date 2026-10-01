#!/usr/bin/env bash
# Повторный замер на MultiHop-RAG с принудительным маршрутом в граф.
#
#   bash deploy/multihop-graph-forced.sh
#
# Зачем понадобился повтор. Первый прогон дал «граф не меняет ничего»
# (recall@16 0.608 против 0.611), но сравнивать было нечего: маршрутизатор
# направил в граф лишь 6.6% вопросов, и графовый канал дал 2.4% контекста.
# Эвристика маршрутизатора построена на русских приметах и на английских
# вопросах молчит. Опыт был негодным, а не результат отрицательным.
#
# Здесь маршрут принудительный: граф спрашивается на каждом вопросе.
# Тогда разница с базовой линией 0.611 действительно означает вклад графа.
#
# Дёшево это стало возможно потому, что кэш извлечения пережил первый
# прогон: чужой граф пересобирается из него за минуты, без обращений
# к модели. Заново платить два часа за извлечение не нужно.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

CORPUS=evaluation/public/multihop-rag/corpus.json
QUESTIONS=evaluation/public/multihop-rag/MultiHopRAG.json

say()   { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; }

export QDRANT_COLLECTION=multihop_rag

say "Шаг 1. Снимаю граф учебника"
uv run rag-textbook graph stats 2>&1 | clean | tail -10
uv run rag-textbook graph drop --yes 2>&1 | clean | tail -4

say "Шаг 2. Чужой граф из кэша и замер с принудительным маршрутом"
# RETRIEVAL_ROUTER_MODE=always — единственное отличие от первого прогона.
RETRIEVAL_ROUTER_MODE=always \
    uv run rag-textbook eval public \
        --corpus "$CORPUS" --questions "$QUESTIONS" \
        --label multihop-graph-forced --graph 2>&1 | clean

say "Шаг 3. Статистика чужого графа"
uv run rag-textbook graph stats 2>&1 | clean | tail -10

say "Шаг 4. Возвращаю граф учебника"
uv run rag-textbook graph drop --yes 2>&1 | clean | tail -3
bash deploy/reset-stages.sh graphed 2>&1 | clean | tail -3
QDRANT_COLLECTION=textbook_chunks \
    uv run rag-textbook ingest --stages graph --no-monitor 2>&1 \
    | grep -oE "извлечение=\{[^}]*\}|Связи между фрагментами: получено [0-9]+"
QDRANT_COLLECTION=textbook_chunks uv run rag-textbook graph stats 2>&1 | clean | tail -10

say "Шаг 5. Проверка, что учебник не пострадал"
# Прогон по собственному эталону: recall@5 обязан остаться около 0.754.
# Без этой проверки мы бы узнали о повреждении графа учебника только
# в следующей сессии и не связали бы это с чужим бенчмарком.
QDRANT_COLLECTION=textbook_chunks \
    uv run rag-textbook eval run --label after-multihop 2>&1 | clean | tail -22

say "ЗАМЕР_ЗАВЕРШЁН"
cat <<'NOTE'

    Главное число: recall@16 с принудительным маршрутом против 0.611
    без графа. И строка «Граф: маршрутизировано ...» — она обязана
    показывать долю около 100%, иначе опыт снова негодный.
NOTE
