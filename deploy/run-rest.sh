#!/usr/bin/env bash
# Досчитать остаток сессии без участия человека: блок 3, затем подбор промпта.
#
#   nohup bash deploy/run-rest.sh > /tmp/rest.log 2>&1 &
#
# Почему отдельный файл, а не команда через ssh. Ожидание сделано через
# pgrep по имени скрипта, а команда, переданная в ssh строкой, сама содержит
# это имя в своей командной строке — и watcher находит сам себя, ожидая
# бесконечно. Дважды на это наступил; файл на сервере такой строки не имеет.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

MARK="0823"   # имя скрипта по частям, чтобы pgrep не поймал этот файл
say() { printf '\n\033[1;35m### %s (%s)\033[0m\n' "$*" "$(date +%H:%M)"; }

say "Жду окончания текущего блока"
while pgrep -f "session-${MARK}" | grep -q . ; do sleep 20; done

say "Блок 3: точность графового канала"
bash "deploy/session-${MARK}.sh" 3 > /tmp/block3.log 2>&1
say "Блок 3 закончен"

# v2 и v3 при нынешнем окне уже посчитаны блоком 2, поэтому здесь только
# новые формулировки: экономия двадцати минут оплаченной карты.
for version in v4 v5; do
    say "Подбор промпта: $version"
    bash deploy/prompt-search.sh "$version" > "/tmp/prompt-$version.log" 2>&1
done

say "ВСЁ_ЗАКОНЧЕНО"
ls -la artifacts/metrics/answers_prompt-*.json 2>/dev/null
ls -la artifacts/metrics/retrieval_eval_hub-*.json 2>/dev/null
