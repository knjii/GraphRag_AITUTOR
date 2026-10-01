#!/usr/bin/env bash
# Подобрать ключ, которым у модели гасится размышление.
#
#   bash deploy/probe-reasoning.sh /models/muse-30b/Muse-Glimmer-30B-UD-Q4_K_XL.gguf
#
# Зачем. Muse Glimmer тратит 1200-1700 токенов на ответ, из которых сам
# ответ — 60-200. Остальное уходит в рассуждение: llama.cpp кладёт его
# в отдельное поле, поэтому ответ приходит чистым, но время карты платится
# за полный объём. При 44 токенах в секунду это 36 секунд на вопрос
# и три с половиной часа на замер.
#
# Ключи проверяются по одному, потому что они друг другу мешают: в первой
# попытке я задал --reasoning off вместе с --reasoning-budget 0, ничего
# не сработало, и было непонятно, который из них виноват.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

MODEL="${1:-/models/muse-30b/Muse-Glimmer-30B-UD-Q4_K_XL.gguf}"
IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda

say() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }

try() {
    local name="$1"; shift
    say "Проверяю: $name"
    docker rm -f probe >/dev/null 2>&1
    docker run -d --name probe --gpus all \
        -v rag-textbook_gguf_models:/models \
        -p 127.0.0.1:8001:8000 "$IMAGE" \
        -m "$MODEL" --host 0.0.0.0 --port 8000 \
        -c 16384 -np 1 -ngl 999 --jinja "$@" >/dev/null

    for _ in $(seq 1 60); do
        curl -sf -o /dev/null http://127.0.0.1:8001/health && break
        sleep 5
    done
    if ! curl -sf -o /dev/null http://127.0.0.1:8001/health; then
        echo "    не поднялся"
        docker logs --tail 6 probe 2>&1 | tail -6
        return 1
    fi
    uv run python scripts/raw_probe.py --max-tokens 600 2>&1 | tail -7
    docker rm -f probe >/dev/null 2>&1
}

# «Как есть» проверяется первым: без него не с чем сравнивать остальные.
try "как есть, без ключей"
try "--reasoning off" --reasoning off
try "--reasoning-effort minimal" --reasoning-effort minimal
try "--reasoning-budget 0" --reasoning-budget 0
try "--reasoning off + budget 0" --reasoning off --reasoning-budget 0
try "kwargs reasoning_effort low" --chat-template-kwargs '{"reasoning_effort": "low"}'

say "Готово"
cat <<'NOTE'

    Читать строку «reasoning_content». Пусто или коротко — ключ работает.
    Смотреть надо и на content: если гашение размышления ломает ответ,
    такой ключ не годится, даже если он быстрее.
NOTE
