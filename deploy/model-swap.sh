#!/usr/bin/env bash
# Смена модели генерации на карте, где всё не помещается одновременно.
#
#   bash deploy/model-swap.sh Qwen/Qwen3.5-9B 0.80
#   bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval
#
# Зачем отдельный скрипт. На RTX 3090 (24 ГБ) вместе с моделью живут Infinity
# (bge-m3 плюс реранкер, около 4.5 ГБ) и ollama с моделью зрения. При доле
# памяти 0.75 под 4B этого хватает. Под 9B в bf16 — нет: только веса просят
# около 18 ГБ. Поэтому перед тяжёлой моделью службы поиска гасятся, а замер
# ведётся по замороженному контексту (eval answers --from-trace), где поиск
# не выполняется вовсе.
#
# Порядок обязателен: сначала погасить, потом поднимать. Иначе SGLang падает
# при выделении памяти, и падение выглядит как несовместимость модели.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
COMPOSE="docker compose --env-file $REPO_DIR/.env -f $REPO_DIR/docker/docker-compose.vllm.yml"
SERVICES="docker compose --env-file $REPO_DIR/.env -f $REPO_DIR/docker/docker-compose.yml"
cd "$REPO_DIR" || exit 1
export PATH="$HOME/.local/bin:$PATH"

MODEL="${1:-}"
FRACTION="${2:-0.80}"
WITH_RETRIEVAL=0
[ "${3:-}" = "--with-retrieval" ] && WITH_RETRIEVAL=1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mОШИБКА: %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "$MODEL" ] || die "укажите модель: bash deploy/model-swap.sh <модель> [доля памяти]"

# .env правится по ключам, а не через source: там лежат тексты промптов
# с пробелами и кавычками, и исполнение файла как кода их ломает.
set_key() {
    local key="$1" value="$2"
    if grep -q "^$key=" .env; then
        sed -i "s|^$key=.*|$key=$value|" .env
    else
        printf '%s=%s\n' "$key" "$value" >> .env
    fi
}

vram() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }

say "Память до: $(vram)"

if [ "$WITH_RETRIEVAL" = "0" ]; then
    say "Гашу службы поиска — их память нужна модели"
    $SERVICES stop infinity ollama >/dev/null 2>&1
    # Qdrant и Neo4j остаются: они на процессоре и памяти карты не занимают.
fi

say "Останавливаю SGLang"
$COMPOSE --profile sglang stop sglang >/dev/null 2>&1
$COMPOSE --profile sglang rm -f sglang >/dev/null 2>&1

set_key SGLANG_MODEL "$MODEL"
set_key SGLANG_GPU_FRACTION "$FRACTION"
set_key LLM_MODEL "$MODEL"

say "Поднимаю $MODEL (доля памяти $FRACTION)"
$COMPOSE --profile sglang up -d sglang || die "SGLang не запустился"

printf '    жду готовности'
for _ in $(seq 1 90); do
    if curl -sf -o /dev/null http://127.0.0.1:8001/health; then printf ' готов\n'; break; fi
    printf '.'; sleep 10
done
curl -sf -o /dev/null http://127.0.0.1:8001/health || {
    printf '\n'
    docker logs --tail 40 "$($COMPOSE --profile sglang ps -q sglang)" 2>&1 | tail -40
    die "модель не поднялась за 15 минут. Чаще всего это нехватка памяти: уменьшите долю или возьмите 8-битную сборку"
}

say "Память после: $(vram)"
say "Проверяю, что модель отвечает"
curl -s http://127.0.0.1:8001/v1/models | head -c 400; printf '\n'

cat <<'NOTE'

    Дальше — замер по замороженному контексту:

        uv run rag-textbook eval answers --from-trace capture/trace.jsonl \
            --no-judge --label 9b

    Он не обращается к поиску, поэтому службы могут оставаться погашенными,
    а сравнение с прежней моделью честное: контекст у обеих один и тот же.

    Вернуть рабочую конфигурацию:

        bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval
        bash deploy/services.sh up
NOTE
