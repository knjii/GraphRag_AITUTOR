#!/usr/bin/env bash
# Остановить очередь моделей сразу после Qwen3.8-27B, не начиная Muse.
#
#   nohup bash deploy/stop-after-qwen27b.sh > /tmp/stop27.log 2>&1 &
#
# Muse Glimmer отложена до того, как разберёмся с гашением рассуждения:
# у неё оно встроено в шаблон, ни один ключ движка его не убирает, только
# сокращает вдвое, и ячейка стоит два-три часа против пятнадцати минут
# у прочих. Мерить её в таком виде — платить за то, чего мы всё равно
# не примем в продукт.
#
# Сетка промптов при этом не теряется: скрипт досчитывает её сам, уже
# после возврата рабочей модели.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

TARGET=artifacts/metrics/answers_model-qwen27b-w16384.json
say() { printf '\n\033[1;35m### %s (%s)\033[0m\n' "$*" "$(date +%H:%M)"; }

say "Жду закрытия ячейки qwen27b"
until [ -e "$TARGET" ]; do sleep 20; done
say "Ячейка закрыта, останавливаю очередь до Muse"

# Сначала управляющие скрипты, потом контейнер: иначе очередь успеет
# поднять следующую модель, пока мы гасим текущую.
pkill -f "run-rest2" >/dev/null 2>&1
pkill -f "models-ab" >/dev/null 2>&1
sleep 2
docker rm -f generator >/dev/null 2>&1

say "Возвращаю рабочую модель для сетки промптов"
docker compose --env-file "$PWD/.env" -f "$PWD/docker/docker-compose.vllm.yml" \
    --profile sglang up -d sglang >/dev/null 2>&1
for _ in $(seq 1 90); do
    curl -sf -o /dev/null http://127.0.0.1:8001/health && break
    sleep 10
done
bash deploy/services.sh up 2>&1 | tail -4

# Сетка промптов меряет формулировку, а не модель, поэтому идёт на той же
# 4B, что и прежние ячейки, иначе числа несравнимы.
for version in v4 v5; do
    say "Достраиваю сетку промптов: $version, окно 8192"
    WINDOW=8192 bash deploy/prompt-search.sh "$version" > "/tmp/grid-$version.log" 2>&1
done

say "ОСТАНОВЛЕНО ПЕРЕД MUSE. Что посчитано:"
ls -la artifacts/metrics/answers_model-*.json
ls -la artifacts/metrics/answers_prompt-*.json
