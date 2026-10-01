#!/usr/bin/env bash
# Приводит .env к состоянию, от которого начинается план проверок.
#
#   bash deploy/baseline-env.sh
#
# Зачем отдельный скрипт. Часть настроек в .env.example уже стоит в значениях,
# подобранных офлайн (порог хабов 40, затухание 0.8, вес редкости). Для плана
# это неверная отправная точка: сравнивать новое надо с тем, на чём сняты
# прежние метрики, иначе «прирост» окажется сравнением нового с новым.
#
# Отдельно выставляются настройки движка: bootstrap создаёт .env из примера,
# а замеры пропускной способности в примере не отражены.

set -uo pipefail
cd "${REPO_DIR:-$HOME/rag_textbook}" || exit 1

set_key() {
    local key="$1" value="$2"
    if grep -qE "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        printf '%s=%s\n' "$key" "$value" >> .env
    fi
    printf '    %-32s %s\n' "$key" "$value"
}

printf '\n\033[1;34m=== движок инференса ===\033[0m\n'
# Замер bench на RTX 3090: 2050 фрагментов в час против 2011 при 4 и 551 при 1.
set_key LLM_MAX_CONCURRENCY 16
# Батч упирается в слоты состояния mamba, а не в KV-кэш: 0.75 удваивает их число.
set_key SGLANG_GPU_FRACTION 0.75

printf '\n\033[1;34m=== точка отсчёта для плана проверок ===\033[0m\n'
# Значения, на которых сняты метрики прошлой сессии. Кандидаты сравниваются с ними.
set_key GRAPH_MAX_ENTITY_DEGREE 64
set_key GRAPH_HOP_DECAY 0.5
set_key GRAPH_PASSAGE_IDF_ENABLED 0
set_key RETRIEVAL_MIN_GRAPH_DOCS 0
set_key RETRIEVAL_GRAPH_CANDIDATE_QUOTA 6
set_key RETRIEVAL_ROUTER_MODE heuristic

printf '\n\033[1;34m=== проверка ===\033[0m\n'
grep -E '^(LLM_MODEL|LLM_BASE_URL|LLM_REASONING_EFFORT|LLM_MAX_CONCURRENCY|SGLANG_GPU_FRACTION|GRAPH_MAX_ENTITY_DEGREE|GRAPH_HOP_DECAY|GRAPH_PASSAGE_IDF_ENABLED|GRAPH_EXTRACTION_PROMPT_VERSION|GRAPH_MAX_ENTITIES_PER_CHUNK|GRAPH_MAX_RELATIONS_PER_CHUNK|CHUNK_SIZE|CHUNK_OVERLAP|MINERU_BACKEND|MINERU_LANG|RETRIEVAL_MIN_GRAPH_DOCS|RETRIEVAL_GRAPH_CANDIDATE_QUOTA|RETRIEVAL_ROUTER_MODE|RETRIEVAL_TOP_K_LINKING)=' .env | sort
