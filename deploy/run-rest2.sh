#!/usr/bin/env bash
# Досчитать остаток после того, как закроется ячейка muse30b.
#
#   nohup bash deploy/run-rest2.sh > /tmp/rest2.log 2>&1 &
#
# Порядок: остальные четыре модели при окне 16384 (muse уже посчитана
# и будет пропущена сама), затем достройка сетки промптов при 8192.
#
# Окно 8192 для МОДЕЛЕЙ исключено: на продукте оно не используется.
# Для промптов оставлено — там оно нужно, чтобы разложить выигрыш
# формулировки на её собственный вклад и вклад окна.
#
# Имена скриптов, которых ждём, собираются из кусков: строка, переданная
# в ssh, содержала бы их целиком, и pgrep находил бы сам себя.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

WAIT_FOR="models""-ab"
say() { printf '\n\033[1;35m### %s (%s)\033[0m\n' "$*" "$(date +%H:%M)"; }

say "Жду окончания ячейки muse30b"
while pgrep -f "$WAIT_FOR" | grep -q . ; do sleep 60; done

say "Остальные модели при окне 16384"
WINDOW=16384 PROMPT_FILE=deploy/prompts/qa-v4.txt \
    bash "deploy/${WAIT_FOR}.sh" > /tmp/models-w16384.log 2>&1
say "Модели закончены"

say "Жду готовности движка после возврата рабочей конфигурации"
for _ in $(seq 1 90); do
    curl -sf -o /dev/null http://127.0.0.1:8001/health && break
    sleep 10
done

# Сетка промптов меряет формулировку, а не модель, поэтому идёт на той же
# 4B, что и прежние ячейки, иначе числа несравнимы.
for version in v4 v5; do
    say "Достраиваю сетку промптов: $version, окно 8192"
    WINDOW=8192 bash deploy/prompt-search.sh "$version" > "/tmp/grid-$version.log" 2>&1
done

say "ВСЁ_ЗАКОНЧЕНО"
ls -la artifacts/metrics/answers_model-*.json
ls -la artifacts/metrics/answers_prompt-*.json
