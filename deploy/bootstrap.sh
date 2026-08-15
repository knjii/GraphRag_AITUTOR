#!/usr/bin/env bash
# Разворачивание окружения на свежем арендованном сервере.
#
# Запускается ОДИН раз на новой машине. Скрипт идемпотентный: повторный запуск
# ничего не ломает и пропускает уже сделанное.
#
# После успешного выполнения обязательно сделайте снапшот сервера в панели
# провайдера — следующие сессии будут стартовать за три минуты вместо часа.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
LOG_DIR="$REPO_DIR/artifacts/deploy-logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/bootstrap_$(date +%Y%m%d_%H%M%S).log") 2>&1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m    %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mОШИБКА: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- проверки

say "Проверяю видеокарту"
command -v nvidia-smi >/dev/null || die "nvidia-smi не найден — образ без драйвера NVIDIA"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
ok "GPU доступна"

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 20000 ]; then
    die "Нужно минимум 20 ГБ видеопамяти, найдено ${VRAM_MB} МБ"
fi

# ---------------------------------------------------------------- система

say "Ставлю системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    git curl ca-certificates gnupg jq \
    build-essential pkg-config \
    libgl1 libglib2.0-0 \
    tmux htop
ok "Системные пакеты готовы"

# ------------------------------------------------------------------ Docker

if ! command -v docker >/dev/null; then
    say "Ставлю Docker"
    install -m 0755 -d /etc/apt/keyrings
    # --batch --yes: иначе при повторном запуске gpg пытается спросить про
    # перезапись существующего файла и падает без терминала.
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    ok "Docker установлен"
else
    ok "Docker уже установлен"
fi

# Проброс GPU внутрь контейнеров: без него Infinity и Ollama не увидят карту.
if ! docker info 2>/dev/null | grep -qi nvidia; then
    say "Ставлю NVIDIA Container Toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq

    # Образы некоторых провайдеров держат ВСЕ пакеты NVIDIA в состоянии hold,
    # чтобы обновление не сломало драйвер. Удерживаемый пакет нельзя и
    # установить: apt отвечает невнятным "held broken packages".
    # Снимаем удержание только с контейнерных пакетов, драйвер не трогаем.
    NVIDIA_CONTAINER_PKGS="libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit nvidia-container-toolkit-base"
    # Список читается ОДИН раз в переменную. Писать `apt-mark showhold | grep -qx`
    # нельзя: grep -q закрывает канал после первого совпадения, apt-mark получает
    # SIGPIPE, и при `set -o pipefail` статусом конвейера становится 141 —
    # то есть найденное совпадение выглядит как ненайденное.
    ALL_HELD=$(apt-mark showhold)
    HELD=""
    for pkg in $NVIDIA_CONTAINER_PKGS; do
        if printf '%s\n' "$ALL_HELD" | grep -qx "$pkg"; then
            HELD="$HELD $pkg"
        fi
    done
    if [ -n "$HELD" ]; then
        say "Снимаю удержание с контейнерных пакетов:$HELD"
        # shellcheck disable=SC2086
        apt-mark unhold $HELD >/dev/null
    fi

    apt-get install -y -qq nvidia-container-toolkit

    # Возвращаем удержание — политика образа сохраняется.
    if [ -n "$HELD" ]; then
        # shellcheck disable=SC2086
        apt-mark hold $HELD >/dev/null
    fi

    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    ok "GPU проброшена в Docker"
else
    ok "NVIDIA Container Toolkit уже настроен"
fi

say "Проверяю доступ к GPU из контейнера"
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi -L \
    || die "Контейнеры не видят GPU"
ok "Контейнеры видят GPU"

# --------------------------------------------------------------------- uv

if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    say "Ставлю uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
ok "uv $(uv --version)"

# ------------------------------------------------------------- зависимости

[ -d "$REPO_DIR" ] || die "Каталог $REPO_DIR не найден — сначала загрузите код (deploy/upload.ps1)"
cd "$REPO_DIR"

# У части хостинг-провайдеров индекс pypi.org недоступен, хотя зеркала работают.
# Проверяем и при необходимости переключаемся — иначе установка падает на
# "tls handshake eof", что выглядит как проблема сети вообще.
say "Выбираю индекс пакетов"
# Восемь проб по разным пакетам, все обязаны пройти. На каналах с вмешательством
# в трафик обрывы случайны: одиночный запрос проскакивает, а установка из сотен
# запросов рвётся с "received corrupt message of type InvalidContentType".
# Проверять одним запросом бесполезно — выберем заведомо нерабочий индекс.
index_alive() {
    for pkg in numpy uvicorn pydantic httpx typer rich qdrant-client pymorphy3-dicts-ru; do
        curl -s -o /dev/null --max-time 12 -f "${1}${pkg}/" || return 1
    done
    return 0
}

PYPI_CANDIDATES="
https://pypi.org/simple/
https://repo.huaweicloud.com/repository/pypi/simple/
https://mirrors.aliyun.com/pypi/simple/
https://pypi.tuna.tsinghua.edu.cn/simple/
"
CHOSEN_INDEX=""
# Ручное переопределение: PYPI_INDEX=https://... bash deploy/bootstrap.sh
if [ -n "${PYPI_INDEX:-}" ]; then
    PYPI_CANDIDATES="$PYPI_INDEX"
fi
for candidate in $PYPI_CANDIDATES; do
    if index_alive "$candidate"; then
        CHOSEN_INDEX="$candidate"
        break
    fi
    printf '    недоступен: %s\n' "$candidate"
done
[ -n "$CHOSEN_INDEX" ] || die "Ни один индекс пакетов недоступен — проверьте сеть сервера"

export UV_DEFAULT_INDEX="$CHOSEN_INDEX"
grep -q 'UV_DEFAULT_INDEX' "$HOME/.bashrc" 2>/dev/null \
    || echo "export UV_DEFAULT_INDEX=\"$CHOSEN_INDEX\"" >> "$HOME/.bashrc"
ok "Индекс: $CHOSEN_INDEX"

say "Ставлю зависимости проекта"
# parsing тянет MinerU, dev — pytest и линтер для прогона тестов на месте.
uv sync --extra parsing --extra dev
ok "Зависимости установлены"

say "Проверяю, что PyTorch видит CUDA"
uv run python -c "
import torch
assert torch.cuda.is_available(), 'PyTorch не видит CUDA'
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}')
"
ok "PyTorch работает с GPU"

# ------------------------------------------------------------------- .env

if [ ! -f .env ]; then
    say "Создаю .env"
    cp .env.example .env
    # Пароль генерируется на сервере и никуда не передаётся.
    NEO4J_PASS=$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 24)
    sed -i "s|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=${NEO4J_PASS}|" .env
    printf '\n\033[1;33mПароль Neo4j сгенерирован и записан в .env\033[0m\n'
    printf 'Посмотреть при необходимости: grep NEO4J_PASSWORD %s/.env\n' "$REPO_DIR"
    ok ".env создан"
else
    ok ".env уже существует, не трогаю"
fi

say "Прогоняю тесты (без внешних сервисов, ~10 секунд)"
uv run pytest -q || die "Тесты не прошли — разбираемся до запуска пайплайна"
ok "Тесты пройдены"

say "Скачиваю образы контейнеров"
# --env-file обязателен: каталогом проекта Compose считает каталог compose-файла
# (docker/), а .env лежит в корне репозитория. Без этого подстановка переменных
# падает с "required variable NEO4J_PASSWORD is missing a value".
docker compose --env-file "$REPO_DIR/.env" -f docker/docker-compose.yml pull
ok "Образы загружены"

cat <<'FINAL'

===============================================================================
  Развёртывание завершено.

  Дальше:
    bash deploy/services.sh up      поднять сервисы и скачать модели
    bash deploy/pilot.sh            запустить пилотный прогон

  ВАЖНО: сделайте снапшот сервера в панели провайдера прямо сейчас —
  следующие сессии будут стартовать за минуты вместо часа.
===============================================================================

FINAL
