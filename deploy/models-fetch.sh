#!/usr/bin/env bash
# Заранее скачать веса кандидатов, пока карта занята другим замером.
#
#   bash deploy/models-fetch.sh            все кандидаты
#   bash deploy/models-fetch.sh qwen27b    один
#
# Смысл в порядке действий. Файлы по 4-17 ГБ, всего около 40 ГБ. Скачивать
# их в начале опыта — значит платить за простой карты: она арендована
# и считает время, пока идёт загрузка. Поэтому загрузка запускается фоном
# во время блоков 0-2 сессии, а опыт с моделями начинается с готовых файлов.
#
# Тома docker переживают пересборку контейнера, поэтому повторно скачивать
# не придётся даже после смены образа SGLang.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1
export PATH="$HOME/.local/bin:$PATH"

WHICH="${1:-all}"
say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mОШИБКА: %s\033[0m\n' "$*" >&2; exit 1; }

# Каталог тома, смонтированного в контейнер SGLang как /models.
VOLUME=$(docker volume inspect rag-textbook_gguf_models --format '{{.Mountpoint}}' 2>/dev/null)
if [ -z "$VOLUME" ]; then
    docker volume create rag-textbook_gguf_models >/dev/null
    VOLUME=$(docker volume inspect rag-textbook_gguf_models --format '{{.Mountpoint}}')
fi
say "Том для весов: $VOLUME"

# Команда называется hf. Прежнее имя huggingface-cli в свежих выпусках
# huggingface_hub не просто устарело, а не работает вовсе: печатает подсказку
# и возвращает ненулевой код.
command -v hf >/dev/null 2>&1 || uv tool install "huggingface_hub[cli]" >/dev/null 2>&1
HF=$(command -v hf || echo "$HOME/.local/bin/hf")
[ -x "$HF" ] || die "hf не установился"

# Кандидаты: репозиторий, маска файла, ожидаемый размер.
# Веса берутся только из официальных репозиториев unsloth и google.
# Это не педантизм: CVE-2026-5760 — исполнение произвольного кода при
# загрузке подготовленного GGUF в SGLang. Наш сервер слушает только
# 127.0.0.1, но источник файлов всё равно обязан быть доверенным.
fetch() {
    local key="$1" repo="$2" pattern="$3" size="$4"
    if [ "$WHICH" != "all" ] && [ "$WHICH" != "$key" ]; then return 0; fi
    say "$key: $repo ($pattern, около $size)"
    sudo -n true 2>/dev/null && SUDO=sudo || SUDO=
    $SUDO "$HF" download "$repo" --include "$pattern" \
        --local-dir "$VOLUME/$key" || die "не скачалось: $repo"
    $SUDO du -sh "$VOLUME/$key"
}

# Веса bf16 качаются не в том же томе: SGLang сам ищет их в кэше Hugging Face.
# Тянем их контейнером того же образа — в нём huggingface_hub уже есть,
# и права на файлы получаются те же, с какими их будет читать движок.
fetch_bf16() {
    local key="$1" repo="$2" size="$3"
    if [ "$WHICH" != "all" ] && [ "$WHICH" != "$key" ]; then return 0; fi
    say "$key: $repo (bf16, около $size)"
    # Точкой входа задаётся сама команда hf, без обёртки bash -c: обёртка
    # съедала кавычки вокруг масок и роняла загрузку с невнятным сообщением.
    docker run --rm         -v rag-textbook_hf_cache:/root/.cache/huggingface         -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}"         --entrypoint hf lmsysorg/sglang:v0.5.17-cu130         download "$repo" --exclude "*.pth" || die "не скачалось: $repo"
}

# Точка отсчёта в bf16 и та же модель вдвое больше — тоже в bf16.
# Пара 4B/9B в одной точности весов даёт шаг размера без примеси квантования,
# а пара 9B bf16 / 9B в 4 битах — цену самого квантования.
fetch_bf16 qwen9b-bf16 Qwen/Qwen3.5-9B "17.2 ГБ"

# Лестница размеров внутри одного семейства — самая чистая часть опыта.
# 4B работает сейчас, 9B и 27B берутся в ОДНОМ И ТОМ ЖЕ кванте UD-Q4_K_XL,
# поэтому в паре «9B против 27B» меняется только размер.
fetch qwen9b    unsloth/Qwen3.5-9B-GGUF          "*UD-Q4_K_XL*.gguf" "5.5 ГБ"
fetch qwen27b   unsloth/Qwen3.8-27B-GGUF         "*UD-Q4_K_XL*.gguf" "17.6 ГБ"

# Другое семейство и другой способ квантования: Gemma обучена с его учётом.
fetch gemma-12b unsloth/gemma-4-12B-it-qat-GGUF  "*UD-Q4_K_XL*.gguf" "6.72 ГБ"

# Meta, август 2026. Плотная 30B с отдельным зрительным кодировщиком,
# заявлена как «для местных агентов на потребительском железе».
fetch muse-30b  unsloth/Muse-Glimmer-30B-GGUF    "*UD-Q4_K_XL*.gguf" "15.9 ГБ"

# Запасные кванты на случай, если кэш при окне 16k окажется больше расчётного.
# Скачиваются заранее: подбирать квант на оплаченной карте — потерянный час.
fetch qwen27b-m unsloth/Qwen3.8-27B-GGUF         "*UD-Q4_K_M*.gguf"  "16.5 ГБ"
fetch muse-30b-3 unsloth/Muse-Glimmer-30B-GGUF   "*UD-Q3_K_XL*.gguf" "13.4 ГБ"

say "Что скачано"
du -sh "$VOLUME"/* 2>/dev/null || true

cat <<'NOTE'

    Дальше: bash deploy/models-ab.sh

    Если UD-Q4_K_M не поместится вместе с окном 16k — переключиться
    на UD-Q3_K_XL, он уже лежит рядом. Это единственная замена, которую
    можно делать посреди опыта: она меняет точность весов, поэтому такую
    ячейку надо помечать отдельно, а не подставлять молча вместо прежней.
NOTE
