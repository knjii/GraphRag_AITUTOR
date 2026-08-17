#!/usr/bin/env bash
# План проверок на арендованной карте, по убыванию ожидаемой отдачи.
#
#   bash deploy/experiments.sh              весь план по порядку
#   bash deploy/experiments.sh --list       что именно будет проверено и зачем
#   bash deploy/experiments.sh --only Р1    один шаг
#   bash deploy/experiments.sh --from Г2    начиная с шага
#
# Каждый шаг отмечается в artifacts/runs/<прогон>/done, поэтому повторный запуск
# не переделывает сделанное. Прерывать можно в любой момент.
#
# Порядок не произвольный. Сначала идут проверки, целящие в измеренные потери
# после поиска (пул кандидатов содержит нужный фрагмент в 87–95% случаев,
# а до контекста доходит 75%), потом настройки обхода графа, потом то, что
# требует пересборки индекса и стоит времени карты.

set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
if ! cd "$REPO_DIR"; then
    echo "Каталог проекта не найден: $REPO_DIR" >&2
    echo "Задайте его явно: REPO_DIR=/путь bash $0" >&2
    exit 1
fi
export PATH="$HOME/.local/bin:$PATH"

RUN_ID="${RUN_ID:-exp_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$REPO_DIR/artifacts/runs/$RUN_ID"
mkdir -p "$RUN_DIR/done"

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; }

# ---------------------------------------------------------------- план
#
# Формат: код | эксперимент eval ab | гипотеза и на чём она основана
#
PLAN=$(cat <<'PLAN'
Р1|min_graph_docs|Реранкер — узкое место многошаговых вопросов. Измерено на полном корпусе: с реранкером многошаговые 0.576, без него 0.703, при этом вопросы с формулами 0.966 против 0.915. Кросс-энкодер оценивает фрагмент против вопроса поодиночке, а второй фрагмент пары сам по себе на вопрос не отвечает. Обязательная квота мест защищает находки графа, не отменяя реранкер остальным.
Р2|graph_quota_wide|Резерв в пуле кандидатов равен 6, а граф держит нужный фрагмент в первых тридцати в 81% случаев. Шести мест мало.
Р3|router|Маршрут в граф получают лишь 66.7% вопросов типа graph_linked: эвристика по ключевым словам по умолчанию отвечает «не нужно». Те же 33% заодно теряют расширенную выдачу — top_k_linking применяется только при маршруте в граф.
Г2|hop_decay|Затухание веса соседа при обходе. Подобрано офлайн на локальном кэше: MRR второго шага 0.225 против 0.253. Здесь проверяется на продукте.
Г3|graph_idf|Вклад сущности, взвешенный редкостью по корпусу. Офлайн: доля попаданий второго фрагмента в первую восьмёрку 0.404 против 0.449.
К1|reranker|Повторная проверка реранкера на нынешнем наборе, с разбором по типам вопросов. Прежнее решение «принято» было принято по среднему, а размен между типами в среднем не виден.
PLAN
)

# Шаги, требующие пересборки индекса, идут отдельно: они стоят времени карты.
REBUILD_PLAN=$(cat <<'PLAN'
Г1|Порог отсечения хабов 64 против 40. Главный рычаг: при 64 обход достигает 461 фрагмента из 1151, то есть канал не ищет, а перечисляет. Офлайн-замер даёт MRR второго шага +0.075 и подтверждён на отложенной половине пар. Требует пересборки графа, потому что отсечение выполняется при записи.
PLAN
)

ONLY=""; FROM=""; LIST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --list) LIST=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        *) printf 'Неизвестный аргумент: %s\n' "$1" >&2; exit 1 ;;
    esac
done

if [ "$LIST" = "1" ]; then
    printf '\n\033[1mПроверки без пересборки индекса (примерно 6 минут каждая)\033[0m\n\n'
    while IFS='|' read -r code experiment reason; do
        [ -n "$code" ] || continue
        printf '  \033[1;36m%s\033[0m  eval ab --experiment %s\n' "$code" "$experiment"
        printf '      %s\n\n' "$reason"
    done <<< "$PLAN"
    printf '\033[1mТребует пересборки графа (около 5 минут из кэша)\033[0m\n\n'
    while IFS='|' read -r code reason; do
        [ -n "$code" ] || continue
        printf '  \033[1;36m%s\033[0m\n      %s\n\n' "$code" "$reason"
    done <<< "$REBUILD_PLAN"
    exit 0
fi

STARTED=$(date +%s)
exec > >(tee -a "$RUN_DIR/experiments.log") 2>&1

say "Прогон $RUN_ID"
printf '    Результаты: %s\n' "$RUN_DIR"

# Точка отсчёта: без неё нечего сравнивать с итогом.
if [ ! -f "$RUN_DIR/done/baseline" ]; then
    say "Точка отсчёта"
    uv run rag-textbook eval run --label "$RUN_ID-baseline" 2>&1 | clean
    touch "$RUN_DIR/done/baseline"
else
    ok "точка отсчёта уже снята"
fi

REACHED=0
[ -z "$FROM" ] && REACHED=1

while IFS='|' read -r code experiment reason; do
    [ -n "$code" ] || continue
    [ "$code" = "$FROM" ] && REACHED=1
    [ "$REACHED" = "1" ] || continue
    [ -n "$ONLY" ] && [ "$ONLY" != "$code" ] && continue

    if [ -f "$RUN_DIR/done/$code" ]; then
        ok "$code уже выполнен, пропускаю"
        continue
    fi

    say "$code · $experiment"
    printf '    Гипотеза: %s\n' "$reason"
    if uv run rag-textbook eval ab --experiment "$experiment" 2>&1 | clean; then
        touch "$RUN_DIR/done/$code"
    else
        warn "$code завершился с ошибкой — остальные шаги продолжаю"
    fi
done <<< "$PLAN"

# ------------------------------------------------- шаг с пересборкой графа
if { [ -z "$ONLY" ] || [ "$ONLY" = "Г1" ]; } && [ ! -f "$RUN_DIR/done/Г1" ]; then
    say "Г1 · порог отсечения хабов 64 против 40"
    printf '    Требует двух сборок графа: отсечение выполняется при записи в Neo4j.\n'

    CURRENT=$(grep -E '^GRAPH_MAX_ENTITY_DEGREE=' .env | head -1 | cut -d= -f2)
    for degree in 64 40; do
        say "    граф при пороге $degree"
        sed -i "s|^GRAPH_MAX_ENTITY_DEGREE=.*|GRAPH_MAX_ENTITY_DEGREE=$degree|" .env
        # Отметку о стадии снимаем, иначе конвейер сочтёт граф собранным.
        uv run python - <<'PY' 2>&1 | clean
import glob, json
from rag_textbook.config import Settings
from rag_textbook.context import build_context

settings = Settings()
context = build_context(settings)
try:
    with context.graph_store._session() as session:
        context.graph_store._run(session, "MATCH (n) DETACH DELETE n")
finally:
    context.close()

def strip(node):
    if isinstance(node, dict):
        node.pop("graphed", None)
        for value in node.values():
            strip(value)
    elif isinstance(node, list):
        for item in node:
            strip(item)

for path in glob.glob("artifacts/manifests/*.json"):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    strip(data)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
print("граф очищен, отметка о стадии снята")
PY
        # Без --force. С ним конвейер не читает готовые чанки с диска, когда
        # стадия чанкинга не выбрана, и граф собирается из нуля фрагментов:
        # прогон отрабатывает без ошибок, а метрики выходят как без графа.
        # Именно так этот шаг и соврал при первом запуске.
        bash deploy/reset-stages.sh graphed | clean
        uv run rag-textbook ingest --stages graph --no-monitor 2>&1 | clean

        STATS=$(uv run rag-textbook graph stats 2>&1 | clean)
        printf '%s\n' "$STATS"
        if printf '%s' "$STATS" | grep -qE 'entities *\| *0 '; then
            warn "граф пуст — замер при пороге $degree бессмысленен, шаг прерван"
            break
        fi
        uv run rag-textbook eval run --label "$RUN_ID-degree-$degree" 2>&1 | clean
    done

    # Сравнение по файлам: две конфигурации порога не существуют одновременно,
    # поэтому eval ab здесь неприменим — он переключает настройки на лету.
    say "    Г1 · парное сравнение двух прогонов"
    BASE_RUN=$(ls -t artifacts/metrics/retrieval_eval_"$RUN_ID"-degree-64_*.json 2>/dev/null | head -1)
    CAND_RUN=$(ls -t artifacts/metrics/retrieval_eval_"$RUN_ID"-degree-40_*.json 2>/dev/null | head -1)
    if [ -n "$BASE_RUN" ] && [ -n "$CAND_RUN" ]; then
        uv run rag-textbook eval compare "$BASE_RUN" "$CAND_RUN" 2>&1 | clean
    else
        warn "не нашёл оба файла прогонов, сравнение пропущено"
    fi

    printf '    Прежнее значение в .env было %s, сейчас стоит 40.\n' "$CURRENT"
    touch "$RUN_DIR/done/Г1"
fi

ELAPSED=$(( ($(date +%s) - STARTED) / 60 ))
say "План пройден за ${ELAPSED} мин"
cat <<NEXT
    Метрики каждого прогона: artifacts/metrics/retrieval_eval_*.json
    Журнал: $RUN_DIR/experiments.log

    Смотрите разбор по типам вопросов, а не только среднее. Именно среднее
    в прошлый раз скрыло, что реранкер помогает вопросам с формулами
    и вредит многошаговым.
NEXT
