#!/usr/bin/env bash
# Останавливает индексацию, не обрывая собственную сессию.
#
#   bash deploy/stop-indexing.sh
#
# Зачем отдельный скрипт вместо `pkill -f 'rag-textbook ingest'`. Такой шаблон
# совпадает с командной строкой SSH-сессии, которой этот pkill и запущен:
# сессия убивает сама себя, соединение рвётся с кодом 255, а работает ли ещё
# индексация — неизвестно. Ошибка допущена дважды: сперва с `pkill -f pytest`,
# потом здесь. Поэтому она закреплена в коде, а не в памяти.
#
# Ищем процесс по исполняемому файлу, а не по строке запуска, и исключаем
# собственный pid вместе с родительским.

set -uo pipefail

me=$$
parent=$PPID
found=0

for pid in $(pgrep -f 'bin/rag-textbook' 2>/dev/null); do
    [ "$pid" = "$me" ] && continue
    [ "$pid" = "$parent" ] && continue
    command=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-70)
    printf 'останавливаю pid %s: %s\n' "$pid" "$command"
    kill "$pid" 2>/dev/null
    found=1
done

if [ "$found" = "0" ]; then
    echo "процессов индексации не найдено"
    exit 0
fi

sleep 3
left=$(pgrep -c -f 'bin/rag-textbook' 2>/dev/null || echo 0)
if [ "$left" -gt 0 ]; then
    echo "не завершились по SIGTERM: $left, добиваю"
    for pid in $(pgrep -f 'bin/rag-textbook' 2>/dev/null); do
        [ "$pid" = "$me" ] && continue
        [ "$pid" = "$parent" ] && continue
        kill -9 "$pid" 2>/dev/null
    done
fi
echo "готово"
