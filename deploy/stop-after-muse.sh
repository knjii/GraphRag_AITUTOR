#!/usr/bin/env bash
# Остановить очередь сразу после того, как закроется ячейка muse30b.
#
#   nohup bash deploy/stop-after-muse.sh > /tmp/stop.log 2>&1 &
#
# Нужен, чтобы не начинать следующую ячейку на час: замеры ставятся
# на паузу до завтра. Остановка идёт по появлению файла с результатом,
# а не по времени — файл пишется атомарно в конце замера, значит на этот
# момент ячейка уже сохранена целиком.
#
# Продолжение завтра ничего не переделывает: cell() в models-ab.sh
# пропускает ячейку, для которой файл уже есть. Достаточно снова запустить
# deploy/run-queue.sh.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1

TARGET=artifacts/metrics/answers_model-muse30b-w16384.json
say() { printf '\n\033[1;35m### %s (%s)\033[0m\n' "$*" "$(date +%H:%M)"; }

say "Жду закрытия ячейки muse30b"
until [ -e "$TARGET" ]; do sleep 30; done
say "Ячейка закрыта, останавливаю очередь"

# Порядок важен: сначала снимаем управляющий скрипт, иначе он поднимет
# следующую модель, пока мы гасим текущую.
pkill -f "run-queue"  >/dev/null 2>&1
pkill -f "models-ab"  >/dev/null 2>&1
sleep 2
docker rm -f generator >/dev/null 2>&1

say "Возвращаю рабочую конфигурацию, чтобы завтра начать с готового"
docker compose --env-file "$PWD/.env" -f "$PWD/docker/docker-compose.vllm.yml" \
    --profile sglang up -d sglang >/dev/null 2>&1
bash deploy/services.sh up 2>&1 | tail -4

say "ОСТАНОВЛЕНО. Что посчитано:"
ls -la artifacts/metrics/answers_model-*.json
ls -la artifacts/metrics/answers_prompt-*.json 2>/dev/null
