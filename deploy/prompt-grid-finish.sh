#!/usr/bin/env bash
# Достроить сетку «промпт x окно» до полной, на одном движке и одной модели.
#
#   nohup bash deploy/prompt-grid-finish.sh > /tmp/grid.log 2>&1 &
#
# Зачем именно так. Прежние ячейки сетки снимались на Qwen3.5-4B в bf16
# под SGLang, а он этих моделей больше не поднимает и веса bf16 утрачены
# вместе с пересозданной машиной. Поэтому сетка пересобирается заново
# целиком на llama.cpp с 4B в четырёх битах: три недостающие ячейки плюс
# та, что уже есть (v4 при 16384 — это ячейка модели qwen4b).
#
# Сравнивать между собой можно только ячейки одного движка и кванта,
# поэтому старые числа сюда не подмешиваются, а служат проверкой порядка
# величин: v4 при 16384 дал 0.315 на bf16/SGLang и 0.305 на Q4/llama.cpp,
# разница внутри шума в 0.010.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
MODEL=/models/qwen4b/Qwen3.5-4B-UD-Q4_K_XL.gguf
TRACE=capture/session-0819/trace-always.jsonl

say() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mОСТАНОВ: %s\033[0m\n' "$*" >&2; exit 1; }
clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; }

[ -f "$TRACE" ] || die "нет слепка $TRACE"

# Освобождаем карту и порт. Предыдущая очередь в конце возвращает рабочую
# конфигурацию и поднимает SGLang — он занимает и память, и порт 8001,
# после чего наш движок молча не стартует.
say "Гашу службы, занимающие карту и порт"
docker compose --env-file "$PWD/.env" -f "$PWD/docker/docker-compose.yml" \n    stop infinity ollama >/dev/null 2>&1
docker rm -f rag-textbook-sglang-1 generator >/dev/null 2>&1
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

say "Поднимаю 4B"
docker rm -f generator >/dev/null 2>&1
docker run -d --name generator --gpus all \
    -v rag-textbook_gguf_models:/models \
    -p 127.0.0.1:8001:8000 "$IMAGE" \
    -m "$MODEL" --host 0.0.0.0 --port 8000 \
    -c 98304 -np 6 -ngl 999 --jinja \
    --reasoning off --reasoning-effort minimal >/dev/null || die "движок не запустился"

printf '    жду готовности'
for _ in $(seq 1 60); do
    curl -sf -o /dev/null http://127.0.0.1:8001/health && { printf ' готов\n'; break; }
    printf '.'; sleep 10
done
curl -sf -o /dev/null http://127.0.0.1:8001/health || die "движок не поднялся"

say "Проверяю три ответа"
uv run python scripts/smoke_generation.py --count 3 > /tmp/smoke-grid.log 2>&1
code=$?
clean < /tmp/smoke-grid.log | tail -20
[ "$code" = "0" ] || die "ответы негодны, сетку считать нельзя"

# Ячейка v4 при 16384 уже снята под именем модели qwen4b — повторять незачем.
for pair in "v4 8192" "v5 8192" "v5 16384"; do
    set -- $pair
    version="$1"; window="$2"
    out="artifacts/metrics/answers_prompt-${version}-w${window}.json"
    if [ -e "$out" ]; then
        say "$version при окне $window уже посчитан, пропускаю"
        continue
    fi
    say "промпт $version, окно $window"
    env QA_SYSTEM_PROMPT="$(cat deploy/prompts/qa-${version}.txt)" \
        PROMPT_VERSION="$version" LLM_CONTEXT_WINDOW="$window" \
        uv run rag-textbook eval answers \
            --from-trace "$TRACE" --no-judge \
            --label "prompt-${version}-w${window}" 2>&1 | clean | tail -18
done

say "СЕТКА_ДОСТРОЕНА"
docker rm -f generator >/dev/null 2>&1
ls -la artifacts/metrics/answers_prompt-*.json 2>/dev/null
