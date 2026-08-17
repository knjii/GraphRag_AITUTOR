#!/usr/bin/env bash
# Восстановление рабочего индекса из кэшей на свежем сервере.
#
#   bash deploy/restore.sh              восстановить и проверить
#   bash deploy/restore.sh --rebuild    пересобрать граф с нуля (72 мин)
#
# Дорогая часть индексации — вызовы модели — уже оплачена, её результат лежит
# в artifacts/cache. Разбор PDF лежит в artifacts/parsed. Поэтому восстановление
# стоит минут, а не часа: замер прошлых прогонов — 2.8, 3.8 и 6.1 минуты против
# 71.7 минуты полной сборки графа.
#
# Скрипт идемпотентен: прерывайте и запускайте заново.

set -uo pipefail

REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
if ! cd "$REPO_DIR"; then
    echo "Каталог проекта не найден: $REPO_DIR" >&2
    echo "Задайте его явно: REPO_DIR=/путь bash $0" >&2
    exit 1
fi
export PATH="$HOME/.local/bin:$PATH"

REBUILD=0
[ "${1:-}" = "--rebuild" ] && REBUILD=1

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m    %s\033[0m\n' "$*"; exit 1; }

# Логи моделей многословны, а нужны две строки из сотни.
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; }

STARTED=$(date +%s)

say "Проверяю, что кэши доехали"
for path in artifacts/parsed artifacts/cache/extraction.sqlite3 evaluation/goldsets/goldset.json; do
    [ -e "$path" ] || die "нет $path — загрузите с ключом: .\\deploy\\upload.ps1 -ServerIp <ip> -WithCaches"
    ok "$path — $(du -sh "$path" | cut -f1)"
done

say "Сверяю настройки, от которых зависит попадание в кэш"
# Ключ кэша извлечения включает модель, режим размышления, версию промпта
# и лимиты. Разойдётся любое — и стадия графа посчитается заново за 72 минуты
# вместо четырёх, молча.
EXPECTED="LLM_MODEL=Qwen/Qwen3.5-4B
LLM_REASONING_EFFORT=none
GRAPH_EXTRACTION_PROMPT_VERSION=v3
GRAPH_MAX_ENTITIES_PER_CHUNK=12
GRAPH_MAX_RELATIONS_PER_CHUNK=12
CHUNK_SIZE=1200
CHUNK_OVERLAP=180
MINERU_BACKEND=pipeline
MINERU_LANG=east_slavic"
MISMATCH=0
while IFS= read -r line; do
    key="${line%%=*}"
    actual=$(grep -E "^${key}=" .env 2>/dev/null | head -1)
    if [ "$actual" != "$line" ]; then
        warn "$key: в .env «${actual#*=}», кэш собран при «${line#*=}»"
        MISMATCH=1
    fi
done <<< "$EXPECTED"
if [ "$MISMATCH" = "1" ]; then
    warn "Расхождение означает пересчёт вместо кэша. Эталон: deploy/measured/server-env-measured.txt"
    printf '    Продолжить всё равно? [y/N] '
    read -r answer
    [ "$answer" = "y" ] || exit 1
else
    ok "все ключевые настройки совпадают с теми, при которых собран кэш"
fi

say "Поднимаю сервисы"
bash deploy/services.sh up || die "сервисы не поднялись"

say "Жду готовности зависимостей"
for attempt in $(seq 1 30); do
    if uv run rag-textbook health >/dev/null 2>&1; then
        ok "все компоненты отвечают (попытка $attempt)"
        break
    fi
    [ "$attempt" = "30" ] && die "health не проходит, смотрите deploy/services.sh logs"
    sleep 20
done

say "Снимаю отметки о стадиях, которые пишут в хранилища"
# Именно снимаю отметки, а не запускаю с --force. Разница существенная:
# --force переделал бы и разбор, а MinerU на том же PDF стоит пять минут
# и не гарантирует посимвольно тот же текст. Текст фрагмента входит в ключ
# кэша извлечения, поэтому сдвиг на символ обесценивает 72 минуты работы
# модели — молча. Хранилища же на новой машине пусты по-настоящему:
# коллекция Qdrant и граф Neo4j остались в томах старого сервера.
if [ "$REBUILD" = "1" ]; then
    bash deploy/reset-stages.sh parsed chunked embedded graphed
else
    bash deploy/reset-stages.sh embedded graphed
fi

say "Восстанавливаю векторы"
uv run rag-textbook ingest --stages embed --no-monitor 2>&1 | clean

if [ "$REBUILD" = "1" ]; then
    say "Пересобираю граф с нуля (ожидаемо около часа)"
    uv run rag-textbook ingest --stages graph 2>&1 | clean
else
    say "Собираю граф из кэша извлечения"
    uv run rag-textbook ingest --stages graph --no-monitor 2>&1 | clean
fi

say "Что получилось"
uv run rag-textbook graph stats 2>&1 | clean

ELAPSED=$(( ($(date +%s) - STARTED) / 60 ))
say "Готово за ${ELAPSED} мин"
cat <<'NEXT'
    Дальше:
        bash deploy/experiments.sh          прогнать план проверок
        bash deploy/experiments.sh --list   посмотреть, что именно будет проверено
NEXT
