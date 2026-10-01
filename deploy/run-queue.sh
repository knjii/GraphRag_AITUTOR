#!/usr/bin/env bash
# Очередь на остаток вечера. Запускать один раз:
#
#   nohup bash deploy/run-queue.sh > /tmp/queue.log 2>&1 &
#
# Требование симметрии: если что-то проверено при окне 8192, оно обязано
# быть проверено и при 16384. Первый проход по моделям идёт при 16384;
# здесь добавляется второй, при 8192, и достраиваются недостающие ячейки
# сетки промптов.
#
# Помимо симметрии у второго прохода есть смысл сам по себе. Гипотеза B2
# утверждает, что маленькие модели хуже используют длинный контекст. Если
# это так, прирост от окна обязан расти с размером модели — а при одном
# окне такая зависимость невидима вовсе.
#
# Имена скриптов, которых ждём, собираются из кусков: строка, переданная
# в ssh, содержала бы их целиком, и pgrep нашёл бы сам себя. Дважды уже
# наступал, три осиротевших цикла потом висели на сервере.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

MODELS="models""-ab"
say() { printf '\n\033[1;35m### %s (%s)\033[0m\n' "$*" "$(date +%H:%M)"; }

wait_engine() {
    for _ in $(seq 1 90); do
        curl -sf -o /dev/null http://127.0.0.1:8001/health && return 0
        sleep 10
    done
    return 1
}

# Ячейки первого захода на SGLang (qwen4b-bf16, qwen9b-bf16) остаются
# как есть: они сняты другим движком и подписаны отдельно. Сравнение с ними
# несёт в себе и смену движка, и это оговорено в отчёте, а не замолчано.
say "Помечаю прежние файлы, чтобы не смешать с новыми"
for file in artifacts/metrics/answers_model-qwen4b-bf16.json artifacts/metrics/answers_model-qwen9b-bf16.json; do
    [ -e "$file" ] && mv "$file" "${file%.json}-sglang.json"
done
ls artifacts/metrics/answers_model-*.json 2>/dev/null

say "Первый проход по моделям, окно 16384"
WINDOW=16384 PROMPT_FILE=deploy/prompts/qa-v4.txt     bash "deploy/${MODELS}.sh" > /tmp/models-w16384.log 2>&1
say "Первый проход закончен"

# Второй проход при окне 8192 отменён: на продукте это окно не работает,
# и платить за него аренду незачем.

say "Жду готовности движка после возврата рабочей модели"
wait_engine || say "движок не поднялся, сетка промптов пропущена"

# Сетка промптов меряет формулировку, а не модель, поэтому обязана идти
# на той же 4B, что и прежние ячейки. К этому моменту сравнение моделей
# уже вернуло рабочую конфигурацию.
for version in v4 v5; do
    say "Достраиваю сетку: промпт $version, окно 8192"
    WINDOW=8192 bash deploy/prompt-search.sh "$version" > "/tmp/grid-$version.log" 2>&1
done

say "ОЧЕРЕДЬ_ЗАКОНЧЕНА"
ls -la artifacts/metrics/answers_model-*.json artifacts/metrics/answers_prompt-*.json
