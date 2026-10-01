#!/usr/bin/env bash
# Окружение HippoRAG 2 на сервере: .venv-hippo, без сети к pypi.org.
#
#   bash deploy/hipporag-env.sh          собрать (повторный запуск — проверка)
#
# Почему отдельное окружение. Пакет hipporag 2.0.0a4 закрепляет torch 2.5.1,
# vLLM 0.6.6 и transformers 4.45; с нашими окружениями он конфликтует,
# а на сервере с закрытым PyPI и удержанными пакетами NVIDIA полная
# установка не встанет. Нам из него нужен только код графа и поиска:
# модель и эмбеддер вызываются по HTTP (scripts/hipporag_run.py).
#
# Что делает:
#   1. uv venv --python 3.11 .venv-hippo;
#   2. зависимости из deploy/requirements-hippo.txt (сверены холостым
#      прогоном на ноутбуке) через индекс UV_DEFAULT_INDEX из bootstrap.sh;
#   3. hipporag — из колеса в deploy/wheels, с --no-deps и сверкой sha256;
#   4. torch: сперва подключение уже установленного из .venv-rl или .venv
#      (файл .pth — ничего не качаем), иначе CPU-сборка с download.pytorch.org.
#      HippoRAG нужен torch только для поиска соседей при рёбрах-синонимах:
#      процесс запускается с CUDA_VISIBLE_DEVICES="" и карту не трогает;
#   5. проверка импорта всего, что тянет HippoRAG.
# Маркер .venv-hippo/.ok хранит sha256 требований и колеса: сменились —
# окружение пересобирается, нет — повторный запуск ничего не делает.
set -euo pipefail
cd "$(dirname "$0")/.."

DIR=.venv-hippo
REQS=deploy/requirements-hippo.txt
WHEEL=deploy/wheels/hipporag-2.0.0a4-py3-none-any.whl
WHEEL_SHA=eee80804299cd37b485c7d09caa0b03a1e00609aaf884759e4ba3cdbd4a99a7a

say()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m!!! %s\033[0m\n' "$*" >&2; exit 1; }

want=$(cat "$REQS" "$WHEEL" | sha256sum | cut -d' ' -f1)
if [ -x "$DIR/bin/python" ] && [ "$(cat "$DIR/.ok" 2>/dev/null)" = "$want" ]; then
    ok "окружение HippoRAG уже собрано"
    exit 0
fi

echo "$WHEEL_SHA  $WHEEL" | sha256sum -c --quiet - || die "колесо hipporag не совпало по sha256"
[ -f "$HOME/.bashrc" ] && eval "$(grep -E '^export UV_DEFAULT_INDEX=' "$HOME/.bashrc" || true)"
[ -n "${UV_DEFAULT_INDEX:-}" ] || die "UV_DEFAULT_INDEX не задан: сначала deploy/bootstrap.sh"

say "Собираю $DIR (индекс $UV_DEFAULT_INDEX)"
rm -rf "$DIR"
uv venv --python 3.11 "$DIR" >/dev/null
uv pip install -p "$DIR" -r "$REQS"
uv pip install -p "$DIR" --no-deps "$WHEEL"

say "Подключаю torch"
site=$("$DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
linked=""
for other in .venv-rl .venv; do
    [ -x "$other/bin/python" ] || continue
    other_site=$("$other/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null) || continue
    [ -d "$other_site/torch" ] || continue
    # .pth дописывает путь в конец sys.path: свои пакеты окружения
    # (transformers <5, numpy) остаются первыми, из чужого берётся только
    # то, чего у нас нет, — torch и его библиотеки.
    echo "$other_site" > "$site/zz-torch-from-$(basename "$other").pth"
    if CUDA_VISIBLE_DEVICES="" "$DIR/bin/python" -c 'import torch; torch.zeros(2) @ torch.zeros(2)' 2>/dev/null; then
        linked="$other"
        break
    fi
    rm -f "$site/zz-torch-from-$(basename "$other").pth"
done
if [ -n "$linked" ]; then
    ok "torch взят из $linked"
else
    uv pip install -p "$DIR" torch --index https://download.pytorch.org/whl/cpu \
        || die "torch не подключился и не скачался: нет .venv-rl/.venv с torch и нет доступа к download.pytorch.org"
    ok "torch: CPU-сборка"
fi

say "Проверяю импорт"
CUDA_VISIBLE_DEVICES="" "$DIR/bin/python" - <<'EOF'
import sys
sys.argv = ["check"]
sys.path.insert(0, "scripts")
import hipporag_run
hipporag_run._stub_optional_modules()
import hipporag, torch, transformers, igraph, openai
from hipporag.HippoRAG import HippoRAG  # noqa: F401
print(f"hipporag ok: torch {torch.__version__}, transformers {transformers.__version__}, "
      f"openai {openai.__version__}, igraph {igraph.__version__}")
EOF
echo "$want" > "$DIR/.ok"
ok "окружение HippoRAG готово"
