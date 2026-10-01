#!/usr/bin/env bash
# Снимает отметки о выполненных стадиях, не трогая артефакты.
#
#   bash deploy/reset-stages.sh embedded graphed
#
# Зачем это вместо `ingest --force`. На новой машине артефакты приезжают
# с прежней (разбор, чанки, кэши вызовов модели), а хранилища пустые: коллекции
# Qdrant и графа Neo4j живут в томах Docker и остаются на старом сервере.
# Нужно выполнить заново ровно две стадии — запись в хранилища.
#
# `--force` для этого не годится: он переделывает и разбор тоже. MinerU
# запускается заново на том же PDF, тратит пять минут и, что важнее, может
# дать чуть иной текст. Текст фрагмента входит в ключ кэша извлечения — сдвиг
# на один символ обесценивает 72 минуты работы модели, и происходит это молча.
#
# Проверено на практике: прерванный повторный разбор оставил от 257 МБ
# артефактов 77 МБ.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1

if [ $# -eq 0 ]; then
    echo "Укажите стадии: parsed, chunked, embedded, graphed" >&2
    echo "Пример: bash deploy/reset-stages.sh embedded graphed" >&2
    exit 1
fi

python3 - "$@" <<'PY'
import glob
import json
import sys

stages = set(sys.argv[1:])
known = {"parsed", "chunked", "embedded", "graphed"}
unknown = stages - known
if unknown:
    print(f"Неизвестные стадии: {', '.join(sorted(unknown))}", file=sys.stderr)
    raise SystemExit(1)

paths = glob.glob("artifacts/manifests/*.json")
if not paths:
    print("Манифестов нет — снимать нечего")
    raise SystemExit(0)

for path in paths:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    removed = []
    for doc_id, document in (data.get("documents") or {}).items():
        marks = document.get("stages") or {}
        for stage in stages:
            if marks.pop(stage, None) is not None:
                removed.append(f"{doc_id[:12]}:{stage}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    print(f"{path}: снято {len(removed)} отметок")
    for item in removed:
        print(f"    {item}")
PY

echo
echo "Теперь запускайте индексацию БЕЗ --force: выполнятся только снятые стадии,"
echo "а разбор и чанки возьмутся из artifacts/parsed."
