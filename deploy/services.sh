#!/usr/bin/env bash
# Управление сервисами: Qdrant, Neo4j, Infinity, Ollama, Phoenix.
#
#   bash deploy/services.sh up       поднять и дождаться готовности
#   bash deploy/services.sh status   что запущено и в каком состоянии
#   bash deploy/services.sh logs     хвост логов всех сервисов
#   bash deploy/services.sh down     остановить (данные сохраняются)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
# --env-file обязателен: каталогом проекта Compose считает каталог compose-файла
# (docker/), а .env лежит в корне репозитория. Без этого подстановка переменных
# падает с "required variable NEO4J_PASSWORD is missing a value".
COMPOSE="docker compose --env-file $REPO_DIR/.env -f $REPO_DIR/docker/docker-compose.yml"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
# .env НЕЛЬЗЯ подключать через `source`: это исполнение файла как кода shell.
# У нас там тексты промптов с пробелами и кавычками, и строка вида
# `PROMPT=Найди ...` превращается в попытку запустить команду «Найди».
# Читаем только нужные ключи, ничего не исполняя.
env_get() {
    [ -f .env ] || return 0
    sed -n "s/^$1=//p" .env | head -1
}

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m    %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mОШИБКА: %s\033[0m\n' "$*" >&2; exit 1; }

wait_http() {
    local name="$1" url="$2" tries="${3:-60}"
    printf '    жду %s' "$name"
    for _ in $(seq 1 "$tries"); do
        if curl -sf -o /dev/null "$url"; then printf ' готов\n'; return 0; fi
        printf '.'; sleep 5
    done
    printf '\n'
    die "$name не поднялся за $((tries * 5)) с. Логи: bash deploy/services.sh logs"
}

case "${1:-up}" in

up)
    [ -n "$(env_get NEO4J_PASSWORD)" ] || die "NEO4J_PASSWORD пуст в .env"

    say "Поднимаю сервисы"
    $COMPOSE up -d

    wait_http "Qdrant"   "http://127.0.0.1:6333/readyz"       60
    wait_http "Neo4j"    "http://127.0.0.1:7474"              60
    wait_http "Ollama"   "http://127.0.0.1:11434/api/version" 60
    # Infinity скачивает веса при первом старте — ждём дольше остальных.
    wait_http "Infinity" "http://127.0.0.1:7997/health"       180

    say "Загружаю модели в Ollama"
    LLM_MODEL_NAME="$(env_get LLM_MODEL)"
    LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3.5:4b}"
    VISION_MODEL_NAME="$(env_get LLM_VISION_MODEL)"
    VISION_MODEL_NAME="${VISION_MODEL_NAME:-qwen2.5vl:3b}"

    # Основная модель живёт в Ollama не всегда. При стеке с SGLang она
    # обслуживается на порту 8001 и называется по-хаггингфейсовски
    # (Qwen/Qwen3.5-4B); `ollama pull` с таким именем падает, а вместе с ним
    # из-за set -e падает весь запуск сервисов. Модель зрения при этом
    # остаётся на Ollama в любом случае — её и качаем.
    MODELS_TO_PULL="$VISION_MODEL_NAME"
    case "$(env_get LLM_BASE_URL)" in
        *11434*) MODELS_TO_PULL="$LLM_MODEL_NAME $VISION_MODEL_NAME" ;;
        *)       ok "основная модель обслуживается вне Ollama, пропускаю $LLM_MODEL_NAME" ;;
    esac

    for model in $MODELS_TO_PULL; do
        if $COMPOSE exec -T ollama ollama list | grep -q "^${model%%:*}"; then
            ok "$model уже загружена"
        else
            printf '    качаю %s...\n' "$model"
            $COMPOSE exec -T ollama ollama pull "$model"
            ok "$model готова"
        fi
    done

    say "Проверяю связность через приложение"
    uv run rag-textbook health

    say "Занятая видеопамять"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    ;;

status)
    $COMPOSE ps
    printf '\n'
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    ;;

logs)
    $COMPOSE logs --tail=80 "${2:-}"
    ;;

down)
    say "Останавливаю сервисы (данные в томах сохраняются)"
    $COMPOSE down
    ok "Сервисы остановлены"
    ;;

*)
    die "Неизвестная команда: $1. Доступно: up | status | logs | down"
    ;;
esac
