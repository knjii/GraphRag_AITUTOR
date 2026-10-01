#!/usr/bin/env bash
# Расширение эталонного набора с приёмкой. Одна серверная сессия.
#
#   bash deploy/goldset-expand.sh
#
# Зачем это делается именно так. Набор из 388 вопросов сгенерирован моделью
# и вручную вычитан на 29 вопросах. Проверка показала, что половине
# «связывающих» вопросов хватает одного фрагмента, а 17.7% опираются
# на оглавление или страницу упражнений. То есть измеритель наполовину
# мерит не то, что заявлено, и наращивать его в прежнем виде бессмысленно:
# вырастет только точность измерения смещённой величины.
#
# Поэтому порядок такой:
#
#   1. Разметить абляцией то, что уже есть, и сверить с ручной проверкой.
#      Совпадение с ручными вердиктами — единственная доступная мера доверия
#      к машинной разметке. Не сойдётся — дальше идти нельзя.
#   2. Дописать новые вопросы с приёмкой: вопрос попадает в набор, только
#      если ответ НЕ получается по одному фрагменту.
#   3. Проверить арифметикой то, что получилось.
#   4. Замерить качество поиска на расширенном наборе — новая точка отсчёта.
#
# Чего этот сценарий НЕ делает: не трогает прежний набор и прежние вердикты.
# Ручные вердикты не затираются машинными, старые вопросы остаются на месте,
# поэтому прогоны остаются сравнимыми.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

say()   { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; }

# Сколько дописать. Приёмка отбраковывает заметную долю, поэтому просят
# больше, чем нужно получить: цель — около тысячи вопросов суммарно.
SINGLE=${SINGLE:-300}
MULTIHOP=${MULTIHOP:-500}
SEED=${SEED:-20260818}

say "Что есть сейчас"
uv run rag-textbook goldset stats 2>&1 | clean | tail -12
uv run rag-textbook goldset audit 2>&1 | clean | tail -24

say "Шаг 1. Разметка абляцией того, что уже есть"
# Только связывающие: изъян сосредоточен там, а вызовов втрое меньше.
uv run rag-textbook goldset label --only-linked 2>&1 | clean | tail -24

say "Шаг 2. Дозапись с приёмкой"
uv run rag-textbook goldset build \
    --append --verify --seed "$SEED" \
    --single "$SINGLE" --multihop "$MULTIHOP" 2>&1 | clean | tail -20

say "Шаг 3. Аудит расширенного набора"
uv run rag-textbook goldset stats 2>&1 | clean | tail -12
uv run rag-textbook goldset audit --write evaluation/goldsets/goldset_clean.json 2>&1 \
    | clean | tail -26

say "Шаг 4. Новая точка отсчёта"
uv run rag-textbook eval run --label expanded 2>&1 | clean | tail -26

say "Шаг 5. Она же на очищенном наборе"
uv run rag-textbook eval run \
    --goldset evaluation/goldsets/goldset_clean.json --label expanded-clean 2>&1 \
    | clean | tail -26

say "Готово"
cat <<'NOTE'

    Что забрать локально:
      evaluation/goldsets/goldset.json        расширенный набор
      evaluation/goldsets/goldset_clean.json  он же без вопросов с изъянами
      evaluation/goldsets/verdicts.json       вердикты, ручные и машинные
      artifacts/metrics/retrieval_eval_expanded*.json

    Что смотреть в первую очередь: совпадение машинной разметки с ручной
    из шага 1. Если оно ниже 0.7, машинным вердиктам верить нельзя, и число
    «доля одношаговых среди связывающих» остаётся оценкой по 20 вопросам.
NOTE
