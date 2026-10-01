#!/usr/bin/env bash
# Оценка разбора документов на OmniDocBench.
#
#   bash deploy/omnidocbench.sh            весь бенчмарк (1651 страница)
#   bash deploy/omnidocbench.sh --pages 200   быстрая проба
#
# Зачем это нужно. Разбор PDF — единственная стадия, качество которой мы
# не измеряли ни разу. Известно лишь, что LaTeX доходит до индекса и что
# 763 фрагмента из 1151 содержат формулы; насколько верно они распознаны,
# неизвестно. Для продукта про математическую литературу это пробел
# в основании: ошибка разбора не лечится ничем ниже по конвейеру.
#
# Чего эта оценка НЕ даёт. В OmniDocBench нет русского подмножества —
# только английский, китайский и смешанный текст. Поэтому оценки по тексту
# и порядку чтения к нашему корпусу переносятся плохо. А вот **формулы
# и таблицы измеряются осмысленно**: LaTeX и HTML языконезависимы, и именно
# они у нас ключевые. Смотреть в первую очередь на них.
#
# Оценка запускается в контейнере разработчиков бенчмарка: она требует
# TeX Live, ImageMagick и Ghostscript, ставить которые на рабочий сервер
# ради одного замера незачем.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

PAGES=0
BACKEND_ARG=""
EFFORT="medium"
GROUP=0
ALLOW_PARTIAL=0
# Пересчёт метрик по уже разобранным страницам: разбор занимает считанные
# минуты, но требует остановки SGLang, а оценка — нет.
EVAL_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --pages) PAGES="$2"; shift 2 ;;
        --backend) BACKEND_ARG="$2"; shift 2 ;;
        --effort) EFFORT="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --allow-partial) ALLOW_PARTIAL=1; shift ;;
        --eval-only) EVAL_ONLY=1; shift ;;
        *) echo "Неизвестный аргумент: $1" >&2; exit 1 ;;
    esac
done

WORK="$HOME/omnidocbench"
DATA="$WORK/dataset"
# Результаты разных движков не должны перетирать друг друга: их сравнение
# и есть цель замера.
TAG="${BACKEND_ARG:-from-env}"
PRED="$WORK/predictions-$TAG"
OUT="$WORK/result-$TAG"
[ "$EVAL_ONLY" = "1" ] || rm -rf "$PRED"
mkdir -p "$DATA" "$PRED" "$OUT"

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m    %s\033[0m\n' "$*" >&2; exit 1; }

say "Проверяю место на диске"
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
printf '    свободно %s ГБ\n' "$free_gb"
[ "$free_gb" -lt 20 ] && die "нужно хотя бы 20 ГБ: датасет плюс образ оценки"

say "Скачиваю датасет"
if [ -f "$DATA/OmniDocBench.json" ] || [ -f "$DATA/gt.json" ]; then
    ok "датасет уже на месте"
else
    uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="opendatalab/OmniDocBench",
    repo_type="dataset",
    local_dir=os.path.expanduser("~/omnidocbench/dataset"),
)
print(f"скачано в {path}")
PY
    [ $? -eq 0 ] || die "не удалось скачать датасет"
fi

say "Что скачалось"
find "$DATA" -maxdepth 2 -type d | head -10
find "$DATA" -maxdepth 2 -name "*.json" | head -5
IMAGES_DIR=$(find "$DATA" -maxdepth 3 -type d -name "images" | head -1)
[ -n "$IMAGES_DIR" ] || die "каталог с изображениями не найден, посмотрите структуру выше"
total=$(find "$IMAGES_DIR" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
ok "изображений: $total"

# Движок определяется до ветвления: он нужен и в заголовке результатов,
# когда разбор пропущен.
if [ -n "$BACKEND_ARG" ]; then
    BACKEND="$BACKEND_ARG"
else
    BACKEND=$(sed -n 's/^MINERU_BACKEND=//p' .env | head -1 | tr -d '\r' | tr -d ' ')
    BACKEND="${BACKEND:-pipeline}"
fi

if [ "$EVAL_ONLY" = "1" ]; then
    say "Пересчёт метрик по готовым предсказаниям"
    count=$(find "$PRED" -name '*.md' | wc -l)
    [ "$count" -gt 0 ] || die "в $PRED нет предсказаний, сначала прогоните разбор"
    ok "предсказаний: $count"
else

say "Освобождаю видеопамять под разбор"
# MinerU требует около 7 ГБ, а SGLang держит 18 из 24 постоянно: доля памяти
# резервируется при старте и не отдаётся. Стадии разнесены по времени
# намеренно — на одной карте они не уживаются.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
printf '    свободно %s МиБ\n' "$FREE_MIB"
SGLANG_WAS_UP=0
# Решение принимается по наличию контейнера, а не по свободной памяти.
# Замер по памяти уже подвёл: контейнер был в состоянии health:starting
# и ещё не занял свою долю, проверка увидела 19 ГБ свободными и оставила
# его работать — а затем он забрал 17.6 ГБ посреди разбора, и все пакетные
# задачи упали. Мгновенный снимок памяти не описывает намерений соседа.
if docker ps --format '{{.Names}}' | grep -q sglang; then
    warn "останавливаю SGLang на время разбора"
    docker compose --env-file .env -f docker/docker-compose.vllm.yml \
        --profile sglang stop sglang >/dev/null 2>&1
    SGLANG_WAS_UP=1
    sleep 10
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    ok "стало свободно $FREE_MIB МиБ"
fi
[ "$FREE_MIB" -lt 7000 ] && die "видеопамяти всё ещё мало: $FREE_MIB МиБ, нужно 7000"

restore_sglang() {
    if [ "$SGLANG_WAS_UP" = "1" ]; then
        say "Возвращаю SGLang"
        docker compose --env-file .env -f docker/docker-compose.vllm.yml \
            --profile sglang start sglang >/dev/null 2>&1
        ok "запущен, готовность займёт пару минут"
    fi
}
trap restore_sglang EXIT

say "Разбираю страницы теми же настройками, что и рабочий корпус"
# Настройки берём из .env, чтобы мерить наш разбор, а не абстрактный MinerU.
# tr -d '\r' при чтении обязателен: .env приходит с машины разработки с концами
# строк CRLF, и без очистки значение уезжает в MinerU как "pipeline\r" — он
# отвергает его как неизвестный движок, а разбор молча заканчивается нулём.
# Язык бенчмарка — не русский, поэтому east_slavic здесь неуместен:
# он ухудшил бы распознавание латиницы и исказил замер не в нашу пользу.
LANG_ARG="en"
printf '    backend=%s, lang=%s\n' "$BACKEND" "$LANG_ARG"

SOURCE="$IMAGES_DIR"
if [ "$PAGES" -gt 0 ]; then
    SOURCE="$WORK/sample"
    rm -rf "$SOURCE"; mkdir -p "$SOURCE"
    find "$IMAGES_DIR" -type f \( -name '*.jpg' -o -name '*.png' \) \
        | sort | head -"$PAGES" | while read -r file; do cp "$file" "$SOURCE/"; done
    ok "проба на $(ls "$SOURCE" | wc -l) страницах"
fi

MINERU_OUT="$WORK/mineru_out-$TAG"
rm -rf "$MINERU_OUT"

# Ключ -l относится только к pipeline: у движков vlm-* и hybrid-* язык
# определяет сама модель, и передача ключа была бы ошибкой запуска.
# Наоборот, --effort понимают только hybrid-*.
ARGS=(-p "$SOURCE" -o "$MINERU_OUT" -b "$BACKEND")
case "$BACKEND" in
    pipeline)   ARGS+=(-l "$LANG_ARG") ;;
    hybrid-*)   ARGS+=(--effort "$EFFORT"); printf '    effort=%s\n' "$EFFORT" ;;
esac

started=$(date +%s)
if [ "$GROUP" -gt 0 ]; then
    # MinerU обрывает весь прогон, когда падает хотя бы одна задача: движок
    # hybrid-engine спотыкается на отдельных страницах (несовпадение размеров
    # тензора), и из-за восьми страниц терялись остальные сто девяносто две.
    # Разбиение на группы ограничивает потерю размером группы.
    mkdir -p "$MINERU_OUT"
    batch="$WORK/batch-$TAG"
    files=$(find "$SOURCE" -type f \( -name '*.jpg' -o -name '*.png' \) | sort)
    total_groups=$(( ($(echo "$files" | wc -l) + GROUP - 1) / GROUP ))
    index=0
    while [ "$index" -lt "$total_groups" ]; do
        rm -rf "$batch"; mkdir -p "$batch"
        echo "$files" | tail -n +$(( index * GROUP + 1 )) | head -"$GROUP" \
            | while read -r file; do cp "$file" "$batch/"; done
        printf '    группа %s/%s\n' "$((index + 1))" "$total_groups"
        uv run mineru "${ARGS[@]/$SOURCE/$batch}" > /dev/null 2>&1 \
            || warn "группа $((index + 1)) упала целиком"
        index=$((index + 1))
    done
    rm -rf "$batch"
else
    uv run mineru "${ARGS[@]}" 2>&1 | tail -8
fi
elapsed=$(( ($(date +%s) - started) ))
printf '    разбор занял %s мин %s с\n' "$((elapsed / 60))" "$((elapsed % 60))"

# Без этой проверки провал разбора уходит дальше по конвейеру: оценщик
# честно отработает на пустом каталоге и выдаст нули, которые легко
# принять за результат замера.
parsed=$(find "$MINERU_OUT" -name '*.md' | wc -l)
[ "$parsed" -eq 0 ] && die "разбор не дал ни одного файла, смотрите вывод MinerU выше"
expected=$(find "$SOURCE" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
ok "разобрано документов: $parsed из $expected"
# Частичный провал искажает замер молча: непрочитанные страницы просто
# не попадут в сравнение, и средние окажутся выше правды.
if [ "$parsed" -lt "$expected" ]; then
    [ "$ALLOW_PARTIAL" = "1" ] || \
        die "разобраны не все страницы, оценка на неполном наборе не имеет смысла"
    warn "движок не осилил $((expected - parsed)) страниц"
    warn "сравнивать такой прогон с другим движком можно только на общих страницах"
fi

say "Готовлю предсказания в формате бенчмарка"
# Оценщик ждёт по одному .md на изображение, имя совпадает с именем картинки.
uv run python - "$MINERU_OUT" "$PRED" <<'PY'
import shutil
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
copied = 0
for markdown in source.rglob("*.md"):
    # MinerU кладёт результат в подкаталог с именем документа.
    name = markdown.parent.name if markdown.stem in {"full", markdown.parent.name} else markdown.stem
    shutil.copyfile(markdown, target / f"{name}.md")
    copied += 1
print(f"подготовлено файлов: {copied}")
PY

fi  # конец ветки разбора

say "Запускаю оценку в контейнере бенчмарка"
GT_FULL=$(find "$DATA" -maxdepth 2 -name "OmniDocBench.json" -o -maxdepth 2 -name "gt.json" | head -1)
[ -n "$GT_FULL" ] || die "файл эталона не найден"

# Эталон описывает 1651 страницу, а в выгрузке лежит 496 изображений, и разбираем
# мы обычно ещё меньше. Оценщик считает страницу без предсказания полным
# промахом, поэтому на полном эталоне средние получаются не «качеством разбора»,
# а «долей страниц, которые мы вообще подали». Первый прогон дал по тексту
# расстояние правки 0.88 именно из-за этого. Эталон сужается до тех страниц,
# по которым предсказание есть.
GT="$WORK/gt-$TAG.json"
uv run python - "$GT_FULL" "$PRED" "$GT" <<'PY'
import json
import sys
from pathlib import Path

source, predictions, target = (Path(p) for p in sys.argv[1:4])
have = {path.stem for path in predictions.glob("*.md")}
full = json.loads(source.read_text())
kept = [item for item in full if Path(item["page_info"]["image_path"]).stem in have]
target.write_text(json.dumps(kept, ensure_ascii=False))
print(f"страниц в эталоне: {len(full)}, с предсказанием: {len(kept)}")
missing = have - {Path(i["page_info"]["image_path"]).stem for i in full}
if missing:
    print(f"предсказаний без эталона: {len(missing)} — они в оценку не войдут")
PY
[ -s "$GT" ] || die "не удалось сузить эталон"
ok "эталон: $GT"

# Готовая configs/end2end.yaml указывает на демонстрационный набор внутри
# образа, поэтому свои пути нужно передать своей конфигурацией. Точка входа
# образа — уже сам pdf_validation.py, аргументы к ней добавляются как есть.
CFG="$WORK/config-$TAG.yaml"
cat > "$CFG" <<'YAML'
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist
      - CDM
    table:
      metric:
      - TEDS
      - Edit_dist
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./gt/gt.json
    prediction:
      data_path: ./data_md/predictions
    match_method: quick_match
    match_workers: 4
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
YAML

docker run --rm \
    -v "$GT:/workspace/gt/gt.json:ro" \
    -v "$PRED:/workspace/data_md/predictions:ro" \
    -v "$OUT:/workspace/result" \
    -v "$CFG:/workspace/configs/ours.yaml:ro" \
    ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 \
    --config configs/ours.yaml 2>&1 | tail -40

say "Результаты"
uv run python - "$OUT" "$BACKEND" <<'PY'
import json
import sys
from pathlib import Path

out, backend = Path(sys.argv[1]), sys.argv[2]
found = sorted(out.glob("*metric_result.json"))
if not found:
    print("    файл с метриками не найден")
    raise SystemExit(0)

data = json.loads(found[0].read_text())
print(f"    движок: {backend}\n")
print(f"    {'раздел':<18}{'метрика':<12}{'значение':>10}")
print(f"    {'-' * 40}")
for section in ("text_block", "display_formula", "table", "reading_order"):
    block = data.get(section, {}).get("all", {})
    for metric, values in block.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    label = f"{metric}/{key}" if key != "all" else metric
                    print(f"    {section:<18}{label:<12}{value:>10.4f}"
                          if len(label) <= 11 else
                          f"    {section:<18}{label}\n{'':<30}{value:>10.4f}")
PY

cat <<'NOTE'

    Edit_dist — расстояние правки, МЕНЬШЕ значит лучше (0 — точное совпадение).
    CDM и TEDS — доли совпадения, БОЛЬШЕ значит лучше.
NOTE
find "$OUT" -type f | head -10
cat <<'NOTE'

    Смотрите в первую очередь метрики по формулам (CDM) и таблицам (TEDS):
    они языконезависимы и переносятся на наш корпус. Оценки по тексту
    и порядку чтения получены на английском и китайском — для русского
    учебника они справочные, а не показательные.
NOTE
