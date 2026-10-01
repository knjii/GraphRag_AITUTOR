#!/usr/bin/env bash
# HippoRAG 2 и обычный RAG на публичном наборе этапа 1 — с пробой и остановкой.
#
#   bash deploy/hipporag-bench.sh --list
#   bash deploy/hipporag-bench.sh                      всё: окружение → проба → полный
#   bash deploy/hipporag-bench.sh --only F1            один шаг заново
#   BENCH=musique-300 bash deploy/hipporag-bench.sh    набор (по умолчанию musique-300)
#
# Запускать внутри tmux (сессия hippo), повторный запуск продолжает с места:
#   tmux has-session -t hippo 2>/dev/null || tmux new-session -d -s hippo \
#       'bash deploy/hipporag-bench.sh; exec bash'
#
# Модель — Qwen/Qwen3.5-4B на SGLang (:8001), та же, что строила наш граф v4:
# иначе сравниваем модели, а не графы. Эмбеддер — BAAI/bge-m3 в Infinity (:7997).
# Процесс HippoRAG карту не трогает (CUDA_VISIBLE_DEVICES=""), поэтому рядом
# могут идти замеры наших конфигураций на тех же службах.
#
# Деньги. Проба (P1) берёт небольшой срез набора и пишет в тот же каталог, что
# полный прогон: кэш ответов модели и файл OpenIE переживают и пробу, и обрыв.
# После пробы — шлюз (G1) по порогам ниже; не прошёл — сценарий стоит, карта
# не тратится на полный прогон с негодным графом.

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/rag_textbook}"
cd "$REPO_DIR" || { echo "нет $REPO_DIR" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

BENCH="${BENCH:-musique-300}"
BUNDLE="$REPO_DIR/artifacts/bench/$BENCH"
OUT="$REPO_DIR/artifacts/runs/bench/$BENCH/hipporag"
RUN="$REPO_DIR/artifacts/runs/bench/$BENCH"
LOG="$RUN/hipporag.log"
PY="$REPO_DIR/.venv-hippo/bin/python"
mkdir -p "$RUN/done" "$OUT"

LLM_URL=http://127.0.0.1:8001/v1
LLM_MODEL=Qwen/Qwen3.5-4B
EMBED_URL=http://127.0.0.1:7997
EMBED_MODEL=BAAI/bge-m3
WORKERS=32            # SGLang держит очередь сам; предел — чтобы не душить соседние замеры
TOP_K=200

# Проба: вопросы, их эталонные фрагменты и отвлекающие — около десятой части корпуса.
PROBE_Q=30
PROBE_EXTRA=400

# Пороги шлюза после пробы. ПРЕДЛОЖЕНЫ, не утверждены: владелец записывает их
# в docs/HYPOTHESES.md до запуска, после пробы не двигаются.
MAX_NO_TRIPLES=0.15   # доля фрагментов без единого триплета: без рёбер фактов граф их не видит
MAX_FILTER_EMPTY=0.50 # доля вопросов, где фильтр фактов пуст → HippoRAG 2 отдаёт плотную выдачу
MAX_FILTER_ERRORS=0   # исключения фильтра: пакет глотает их молча
MAX_UNMAPPED=0        # выданный текст, которого нет в наборе, — сломано сопоставление

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*" | tee -a "$LOG"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*" | tee -a "$LOG"; }
die()  { printf '\n\033[1;31mСТОП: %s\033[0m\n' "$*" | tee -a "$LOG" >&2; exit 1; }
run() {
    local desc="$1"; shift
    "$@" 2>&1 | grep -v -E ' INFO |HTTP Request: POST' | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die "$desc — код возврата $rc (журнал: $LOG)"
}

hippo() {
    CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "$PY" scripts/hipporag_run.py \
        --bundle "$BUNDLE" --out "$OUT" \
        --llm-base-url "$LLM_URL" --llm-model "$LLM_MODEL" \
        --embed-base-url "$EMBED_URL" --embed-model "$EMBED_MODEL" \
        --workers "$WORKERS" --top-k "$TOP_K" "$@"
}

# ---------------------------------------------------------------- шаги
step_E1() {  # окружение .venv-hippo (маркер: повторно не собирается)
    run "окружение HippoRAG" bash deploy/hipporag-env.sh
}

step_E2() {  # набор на месте и не изменён; службы отвечают
    [ -f "$BUNDLE/manifest.json" ] || die "нет набора $BUNDLE — он едет в архиве с кодом"
    run "сверка набора" "$PY" - "$BUNDLE" <<'PY'
import hashlib, json, sys
from pathlib import Path
b = Path(sys.argv[1]); m = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
for name, want in m["sha256"].items():
    got = hashlib.sha256((b / name).read_bytes()).hexdigest()
    assert got == want, f"{name}: {got} != {want}"
print(f"набор цел: {m['chunks']} фрагментов, {m['questions']} вопросов")
PY
    curl -sf "$LLM_URL/models" | grep -q 'Qwen3.5-4B' \
        || die "на :8001 нет Qwen3.5-4B — поднять: bash deploy/model-swap.sh Qwen/Qwen3.5-4B 0.75 --with-retrieval"
    curl -sf -o /dev/null "$EMBED_URL/health" || die "Infinity на :7997 не отвечает — bash deploy/services.sh up"
    # Ответ и вектор читаются до долгого прогона: пустой ответ, размышление
    # вместо текста или NaN в векторе видно здесь, а не через час.
    run "пробный вызов модели и эмбеддера" "$PY" - "$LLM_URL" "$LLM_MODEL" "$EMBED_URL" "$EMBED_MODEL" <<'PY'
import math, sys
from openai import OpenAI
llm_url, llm, emb_url, emb = sys.argv[1:5]
c = OpenAI(base_url=llm_url, api_key="sk-local")
r = c.chat.completions.create(model=llm, max_tokens=200, temperature=0, extra_body={"reasoning_effort": "none"},
    messages=[{"role": "user", "content": 'Extract named entities as JSON {"named_entities": [...]}: '
               '"Marie Curie won the Nobel Prize in Physics in 1903."'}])
text = (r.choices[0].message.content or "").strip()
print("модель:", text[:200])
assert text and "<think>" not in text and "Curie" in text, "пустой или негодный ответ модели"
e = OpenAI(base_url=emb_url, api_key="sk-local").embeddings.create(
    model=emb, input=["Marie Curie", "Nobel Prize"], encoding_format="float")
v = e.data[0].embedding
assert len(v) == 1024 and all(math.isfinite(x) for x in v), "вектор не 1024 или с NaN"
print("эмбеддер: 1024, конечные")
PY
}

step_P1() {  # проба на срезе: извлечение, граф, обе выдачи
    run "проба HippoRAG 2" hippo --probe-questions "$PROBE_Q" --probe-extra "$PROBE_EXTRA"
    cp "$OUT/run.json" "$RUN/run-probe.json"
    for s in hipporag2 dense; do cp "$OUT/rankings-$s.jsonl" "$RUN/probe-rankings-$s.jsonl"; done
}

step_G1() {  # шлюз по пробе: пороги выше, записаны до запуска
    run "шлюз пробы" "$PY" - "$RUN/run-probe.json" "$MAX_NO_TRIPLES" "$MAX_FILTER_EMPTY" \
        "$MAX_FILTER_ERRORS" "$MAX_UNMAPPED" <<'PY'
import json, sys
run = json.load(open(sys.argv[1], encoding="utf-8"))
no_tr, f_empty, f_err, unmapped = map(float, sys.argv[2:6])
ie, h, d = run["openie"], run["hipporag2"], run["dense"]
checks = [
    ("фрагментов без триплетов", ie["no_triples_share"], no_tr),
    ("вопросов с пустым фильтром", h["filter_empty_share"], f_empty),
    ("исключений фильтра", h["filter_errors"], f_err),
    ("несопоставленных выдач", h["unmapped_docs"] + d["unmapped_docs"], unmapped),
]
print(f"извлечение: {json.dumps(ie, ensure_ascii=False)}")
print(f"граф: {json.dumps(run['graph'], ensure_ascii=False)}")
failed = [name for name, value, limit in checks if value > limit]
for name, value, limit in checks:
    print(f"  {'ok ' if value <= limit else 'ПРОВАЛ'} {name}: {value} (порог {limit})")
per_chunk = run["index_seconds"] / max(1, run["chunks"])
print(f"индекс: {per_chunk:.2f} с/фрагмент → полный набор ≈ оценка в F1")
sys.exit(1 if failed else 0)
PY
    local chunks
    chunks=$("$PY" -c "import json; print(json.load(open('$BUNDLE/manifest.json'))['chunks'])")
    "$PY" -c "import json; r=json.load(open('$RUN/run-probe.json')); \
print(f'оценка полного индекса: {r[\"index_seconds\"]/r[\"chunks\"]*$chunks/3600:.1f} ч (верхняя: проба платила и за старт)')" \
        | tee -a "$LOG"
}

step_F1() {  # полный прогон: граф заново, извлечение пробы переиспользуется из кэша
    run "полный прогон HippoRAG 2" hippo --rebuild-graph
    cp "$OUT/run.json" "$RUN/run-full.json"
}

step_M1() {  # метрики: all@k, recall@k, место последнего, бутстрап разности
    run "метрики" "$PY" scripts/bench_metrics.py --bundle "$BUNDLE" \
        --run dense="$OUT/rankings-dense.jsonl" --run hipporag2="$OUT/rankings-hipporag2.jsonl" \
        --k 2,5,16,50 --pair dense,hipporag2 --out "$RUN/metrics-hipporag.json"
}

STEPS=(E1 E2 P1 G1 F1 M1)
declare -A DESC=(
    [E1]="окружение .venv-hippo"
    [E2]="набор цел, 4B и Infinity отвечают осмысленно"
    [P1]="проба: $PROBE_Q вопросов + $PROBE_EXTRA отвлекающих"
    [G1]="шлюз: без триплетов ≤ $MAX_NO_TRIPLES, пустой фильтр ≤ $MAX_FILTER_EMPTY, исключений ≤ $MAX_FILTER_ERRORS"
    [F1]="полный набор $BENCH, граф заново"
    [M1]="метрики dense против hipporag2"
)
# E2 и G1 — проверки: выполняются при каждом запуске, отметку не ставят.
ALWAYS=" E2 "

only="" from=""
case "${1:-}" in
    --list) for s in "${STEPS[@]}"; do printf '%s  %s\n' "$s" "${DESC[$s]}"; done; exit 0 ;;
    --only) only="$2" ;;
    --from) from="$2" ;;
esac

started=""
[ -z "$from" ] && started=1
for s in "${STEPS[@]}"; do
    [ "$s" = "$from" ] && started=1
    [ -n "$started" ] || continue
    [ -n "$only" ] && [ "$s" != "$only" ] && continue
    if [ -z "$only" ] && [ -f "$RUN/done/$s" ] && [[ "$ALWAYS" != *" $s "* ]]; then
        ok "$s уже сделан"
        continue
    fi
    say "$s — ${DESC[$s]} ($(date '+%H:%M:%S'))"
    "step_$s"
    [[ "$ALWAYS" == *" $s "* ]] || touch "$RUN/done/$s"
done
ok "готово: $RUN"
