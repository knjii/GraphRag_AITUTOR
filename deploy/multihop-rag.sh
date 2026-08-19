#!/usr/bin/env bash
# Замер на публичном наборе MultiHop-RAG. Одна серверная сессия.
#
#   bash deploy/multihop-rag.sh
#
# Зачем. Собственный набор отвечает, стало ли лучше, чем вчера. На вопрос
# «как это выглядит рядом с другими системами» он не отвечает никак.
# MultiHop-RAG размечен дословными цитатами свидетельств, поэтому по нему
# считается полнота поиска — та величина, вокруг которой идёт весь проект.
#
# Что делается и в каком порядке:
#
#   1. Корпус индексируется в ОТДЕЛЬНУЮ коллекцию Qdrant, без графа.
#      Замер на нём — базовая линия: плотный канал, лексический, реранкер.
#   2. Граф учебника снимается, по чужому корпусу строится свой граф,
#      замер повторяется. Разница между 1 и 2 и есть ответ на вопрос,
#      помогает ли наш граф на многошаговых вопросах.
#   3. Граф учебника пересобирается из кэша извлечения.
#
# Почему граф учебника приходится снимать: Neo4j Community держит ровно одну
# базу данных, отдельную под чужой корпус не создать. Потеря восполнима —
# кэш извлечения на месте, пересборка идёт минуты и не обращается к модели.
#
# Коллекция Qdrant при этом отдельная, поэтому векторы учебника не страдают
# вовсе: снимается только граф.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

CORPUS=evaluation/public/multihop-rag/corpus.json
QUESTIONS=evaluation/public/multihop-rag/MultiHopRAG.json
LIMIT=${LIMIT:-0}

say()   { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; }

if [ ! -f "$CORPUS" ] || [ ! -f "$QUESTIONS" ]; then
    echo "нет файлов набора: $CORPUS / $QUESTIONS"
    exit 1
fi

# Отдельная коллекция обязательна: иначе чужой корпус смешается с учебником
# и обесценит все прежние замеры. Команда сама откажется работать
# с коллекцией по умолчанию, но проверить стоит и здесь.
export QDRANT_COLLECTION=multihop_rag

say "Шаг 1. Базовая линия без графа"
GRAPH_ENABLED=false GRAPH_RETRIEVAL_ENABLED=false \
    uv run rag-textbook eval public \
        --corpus "$CORPUS" --questions "$QUESTIONS" \
        --label multihop-nograph --limit "$LIMIT" --no-graph 2>&1 | clean

say "Шаг 2. Снимаю граф учебника"
# Состояние ДО очистки печатается намеренно: «удалено 0» означало бы, что
# снимать было нечего, то есть что-то уже пошло не так.
uv run rag-textbook graph stats 2>&1 | clean | tail -12
uv run rag-textbook graph drop --yes 2>&1 | clean | tail -6

say "Шаг 3. Граф по чужому корпусу и замер с ним"
uv run rag-textbook eval public \
    --corpus "$CORPUS" --questions "$QUESTIONS" \
    --label multihop-graph --limit "$LIMIT" --graph 2>&1 | clean

say "Шаг 4. Статистика чужого графа"
uv run rag-textbook graph stats 2>&1 | clean | tail -12

say "Шаг 5. Возвращаю граф учебника"
uv run rag-textbook graph drop --yes 2>&1 | clean | tail -4
bash deploy/reset-stages.sh graphed 2>&1 | clean | tail -4
# Коллекция здесь снова своя: пересборка графа читает чанки учебника.
QDRANT_COLLECTION=textbook_chunks \
    uv run rag-textbook ingest --stages graph --no-monitor 2>&1 \
    | grep -oE "извлечение=\{[^}]*\}|Связи между фрагментами: получено [0-9]+"
QDRANT_COLLECTION=textbook_chunks uv run rag-textbook graph stats 2>&1 | clean | tail -12

say "ЗАМЕР_ЗАВЕРШЁН"
cat <<'NOTE'

    Что читать. Разница между multihop-nograph и multihop-graph — это ответ
    на вопрос, помогает ли граф на многошаговых вопросах чужого корпуса.
    Разбивка по типам набора (inference / comparison / temporal) печатается
    отдельно: опубликованные числа даны именно в ней.

    Чего это НЕ говорит: корпус новостной и английский. На русский учебник
    математики выводы не переносятся — это внешняя точка сравнения.

    Проверьте, что граф учебника вернулся: passages=1151, entities около 4059.
NOTE
