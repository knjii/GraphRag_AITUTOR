#!/usr/bin/env bash
# Пилотный прогон: индексация, эталонный набор, метрики, A/B по графу.
#
#   bash deploy/pilot.sh                    полный прогон
#   bash deploy/pilot.sh --skip-ingest      только оценка (индекс уже собран)
#   bash deploy/pilot.sh --questions 60     набор поменьше, для быстрой проверки
#
# Скрипт можно прерывать и запускать заново: индексация возобновляема по манифесту,
# результаты вызовов моделей лежат в кэше. Повторный запуск не платит за сделанное.
#
# Всё время выполнения пишется утилизация GPU — по ней будем решать,
# потянем ли модель побольше или MoE.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"

SKIP_INGEST=0
QUESTIONS_SINGLE=100
QUESTIONS_MULTI=50
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-ingest) SKIP_INGEST=1; shift ;;
        --questions)   QUESTIONS_SINGLE=$(( $2 * 2 / 3 )); QUESTIONS_MULTI=$(( $2 / 3 )); shift 2 ;;
        *) printf 'Неизвестный аргумент: %s\n' "$1" >&2; exit 1 ;;
    esac
done

RUN_ID="pilot_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_DIR/artifacts/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/pilot.log") 2>&1

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }

STARTED=$(date +%s)
stage_start() { STAGE_T0=$(date +%s); }
stage_end() {
    local seconds=$(( $(date +%s) - STAGE_T0 ))
    printf '%s\t%s\n' "$1" "$seconds" >> "$RUN_DIR/stages.tsv"
    ok "$1: $((seconds / 60)) мин $((seconds % 60)) с"
}

say "Проверка зависимостей"
uv run rag-textbook health

# ------------------------------------------------------ пропускная способность

say "Замер пропускной способности сервера инференса"
warn "Показывает, батчит ли движок запросы и какой параллелизм ставить."
stage_start
uv run rag-textbook bench --requests 12 || warn "Замер не удался, продолжаю"
stage_end "bench_llm"

# --------------------------------------------------------------- индексация

if [ "$SKIP_INGEST" -eq 0 ]; then
    say "Индексация корпуса"
    warn "Самая долгая стадия. Прежний baseline на локальной машине — 14.5 ч на учебник."
    warn "Ресурсы пишутся с разметкой по стадиям — потом покажем, во что упирались."
    stage_start
    uv run rag-textbook ingest --monitor --monitor-interval 2
    stage_end "ingest"

    say "Разбор узких мест индексации"
    MONITOR_DIR=$(ls -1dt "$REPO_DIR"/artifacts/metrics/monitor_* 2>/dev/null | head -1 || true)
    if [ -n "$MONITOR_DIR" ]; then
        uv run rag-textbook bottlenecks "$MONITOR_DIR"
        uv run rag-textbook bottlenecks "$MONITOR_DIR" --json \
            > "$RUN_DIR/bottlenecks.json" 2>/dev/null || true
        cp -r "$MONITOR_DIR" "$RUN_DIR/monitor" 2>/dev/null || true
    else
        warn "Каталог с замерами не найден"
    fi
else
    warn "Индексация пропущена по флагу --skip-ingest"
fi

say "Состояние хранилищ"
uv run rag-textbook graph stats || warn "Граф недоступен — проверьте Neo4j"

# ---------------------------------------------------------- эталонный набор

GOLDSET="$REPO_DIR/evaluation/goldsets/goldset.json"
if [ -f "$GOLDSET" ]; then
    ok "Эталонный набор уже существует, использую его"
else
    say "Сборка эталонного набора"
    stage_start
    uv run rag-textbook goldset build \
        --single "$QUESTIONS_SINGLE" \
        --multihop "$QUESTIONS_MULTI"
    stage_end "goldset"
fi
uv run rag-textbook goldset stats

# -------------------------------------------------------------------- оценка

say "Базовые метрики поиска"
stage_start
uv run rag-textbook eval run --label baseline
stage_end "eval_baseline"

say "A/B: даёт ли граф прирост"
stage_start
uv run rag-textbook eval ab --experiment graph
stage_end "ab_graph"

say "A/B: даёт ли реранкер прирост"
stage_start
uv run rag-textbook eval ab --experiment reranker
stage_end "ab_reranker"

# ------------------------------------------------------------------- сводка

say "Сбор результатов"
cp -r "$REPO_DIR/artifacts/metrics" "$RUN_DIR/metrics" 2>/dev/null || true
cp "$GOLDSET" "$RUN_DIR/" 2>/dev/null || true
cp "$REPO_DIR/.env" "$RUN_DIR/env_snapshot.txt" 2>/dev/null || true
# Пароль в снимок конфигурации не попадает.
sed -i 's|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=<скрыт>|' "$RUN_DIR/env_snapshot.txt" 2>/dev/null || true

uv run rag-textbook bottlenecks "$RUN_DIR/monitor" --json > "$RUN_DIR/summary.json" 2>/dev/null || true

TOTAL=$(( $(date +%s) - STARTED ))
ARCHIVE="$REPO_DIR/artifacts/${RUN_ID}.tar.gz"
tar -czf "$ARCHIVE" -C "$REPO_DIR/artifacts/runs" "$RUN_ID"

cat <<FINAL

===============================================================================
  Пилотный прогон завершён за $((TOTAL / 3600)) ч $(((TOTAL % 3600) / 60)) мин

  Результаты:  $RUN_DIR
  Архив:       $ARCHIVE

  Забрать к себе (выполнить на своём компьютере):
      scp -i \$env:USERPROFILE\\.ssh\\intelion_ed25519 \\
          root@СЕРВЕР:$ARCHIVE .

  Не забудьте остановить сервер, когда закончите:
      bash deploy/services.sh down && shutdown -h now
===============================================================================

FINAL
