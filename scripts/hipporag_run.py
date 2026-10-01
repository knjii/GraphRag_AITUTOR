"""HippoRAG 2 и обычный RAG из его же кода на общем наборе этапа 1.

Зачем свой запускатель, а не ``reproduce/main.py``. Пакет ``hipporag``
2.0.0a4 закрепляет torch 2.5.1, vLLM 0.6.6 и transformers 4.45 — на сервере
с закрытым PyPI и удержанными пакетами NVIDIA такая установка не встанет,
а нашу модель он всё равно вызывает по OpenAI-совместимому протоколу. Здесь:

* пакет ставится без зависимостей (``deploy/hipporag-env.sh``), а модули,
  которые нужны только необязательным путям (vLLM, GritLM, Bedrock,
  Cohere), подменяются пустышками до импорта;
* эмбеддер — bge-m3 через наш Infinity (OpenAI ``/embeddings``): в пакете
  встроен только выбор по имени, и bge-m3 он не знает;
* извлечение (NER + триплеты) и фильтр фактов при запросе идут в ту же
  модель, которой строился наш граф, — иначе сравниваем модели, а не графы;
* размышление гасится ``reasoning_effort: none`` (так же, как в нашем
  клиенте; без этого Qwen3.5 тратит лимит на рассуждение и отдаёт пустоту).

Что считается. Для каждого вопроса — ранжированный список идентификаторов
фрагментов длиной ``--top-k`` у двух систем:

``hipporag2``  полный HippoRAG 2: факты → фильтр моделью → PPR;
``dense``      ``retrieve_dpr`` того же объекта: плотный поиск bge-m3 по тем
               же векторам — «обычный RAG», как его берут в статьях.

Метрики считает ``scripts/bench_metrics.py`` одинаково для всех систем.

Деньги. Каждый ответ модели HippoRAG кладёт в sqlite-кэш (``llm_cache``),
векторы — в свои хранилища; повторный запуск после обрыва не повторяет
ни одного вызова. Поэтому ``--probe-questions`` для пробы и полный прогон пишут
в один ``--out``: проба оплачивает начало полного прогона.

    python scripts/hipporag_run.py --bundle artifacts/bench/musique-300 \\
        --out runs/bench/musique-300/hipporag \\
        --llm-base-url http://127.0.0.1:8001/v1 --llm-model Qwen/Qwen3.5-4B \\
        --embed-base-url http://127.0.0.1:7997 --embed-model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np


def _stub_optional_modules() -> None:
    """Пустышки для модулей, которые пакет импортирует безусловно.

    Их классы используются только на путях, которые мы не включаем
    (офлайн-vLLM, GritLM, Bedrock, Cohere). Если путь всё-таки сработает,
    обращение к пустышке упадёт сразу и громко, а не молча.
    """
    def module(name: str, **attrs) -> None:
        if name in sys.modules:
            return
        try:
            __import__(name)
            return
        except ImportError:
            pass
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        # Без спецификации importlib.util.find_spec(name) падает ValueError:
        # так на сервере упал accelerate из .venv-rl (проверка boto3), 2026-09-30.
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod

    class _Absent:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("необязательная зависимость HippoRAG не установлена")

    module("vllm", LLM=_Absent, SamplingParams=_Absent)
    module("vllm.lora")
    module("vllm.lora.request", LoRARequest=_Absent)
    module("gritlm", GritLM=_Absent)
    module("litellm", completion=_Absent)
    module("boto3", client=_Absent)
    module("botocore")
    module("botocore.exceptions", ClientError=Exception)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_embedding_class(base_cls, *, base_url: str, model: str, batch: int, retries: int = 5):
    """bge-m3 через OpenAI-совместимый ``/embeddings`` (Infinity, Ollama).

    Наследуем от встроенного OpenAI-класса только ради общих полей и
    нормализации; ``batch_encode`` переписан: в оригинале на ошибке запроса
    стоит ``import ipdb; ipdb.set_trace()`` — на сервере без терминала это
    зависание или падение посреди индексации.
    """
    from openai import OpenAI

    class Bge(base_cls):
        def __init__(self, global_config=None, embedding_model_name=None):
            super().__init__(global_config=global_config, embedding_model_name=embedding_model_name)
            self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "sk-local"))

        def encode(self, texts):
            texts = [t.replace("\n", " ") or " " for t in texts]
            for attempt in range(retries):
                try:
                    response = self.client.embeddings.create(input=texts, model=model, encoding_format="float")
                    return np.array([item.embedding for item in response.data], dtype=np.float32)
                except Exception as exc:  # noqa: BLE001 — сеть, перегрузка службы
                    if attempt == retries - 1:
                        raise
                    wait = 2 ** attempt
                    print(f"эмбеддер: {exc!r}, повтор через {wait} с", file=sys.stderr)
                    time.sleep(wait)
            raise AssertionError("недостижимо")

        def batch_encode(self, texts, **kwargs):
            # bge-m3 обучен без инструкций: инструкцию запроса, которую
            # HippoRAG подставляет для NV-Embed, намеренно не передаём.
            if isinstance(texts, str):
                texts = [texts]
            parts = [self.encode(texts[i:i + batch]) for i in range(0, len(texts), batch)]
            result = np.concatenate(parts) if parts else np.zeros((0, 1024), dtype=np.float32)
            if self.embedding_config.norm:
                result = result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)
            return result

    return Bge


def patch_hipporag(args) -> type:
    _stub_optional_modules()
    os.environ.setdefault("OPENAI_API_KEY", "sk-local")

    import hipporag  # noqa: F401 — инициализация пакета
    hr_module = sys.modules["hipporag.HippoRAG"]
    from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
    from hipporag.llm.openai_gpt import CacheOpenAI

    bge = build_embedding_class(OpenAIEmbeddingModel, base_url=args.embed_base_url.rstrip("/"),
                                model=args.embed_model, batch=args.embed_batch)
    original = hr_module._get_embedding_model_class

    def pick(embedding_model_name: str = ""):
        if embedding_model_name == args.embed_model:
            return bge
        return original(embedding_model_name=embedding_model_name)

    hr_module._get_embedding_model_class = pick

    init_config = CacheOpenAI._init_llm_config

    def init_with_reasoning(self) -> None:
        init_config(self)
        if args.reasoning_effort:
            self.llm_config.generate_params["extra_body"] = {"reasoning_effort": args.reasoning_effort}

    CacheOpenAI._init_llm_config = init_with_reasoning

    # Извлечение шлёт запросы пулом потоков без предела (по умолчанию
    # min(32, ядра+4)). SGLang держит очередь сам, а Ollama обслуживает
    # по одному, и хвост очереди ловит пятиминутный таймаут клиента.
    if args.workers:
        import functools
        from concurrent.futures import ThreadPoolExecutor

        openie_module = sys.modules["hipporag.information_extraction.openie_openai"]
        openie_module.ThreadPoolExecutor = functools.partial(ThreadPoolExecutor, max_workers=args.workers)
    return hr_module.HippoRAG


def make_config(args):
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig()
    config.save_dir = str(args.out)
    config.llm_name = args.llm_model
    config.llm_base_url = args.llm_base_url
    config.embedding_model_name = args.embed_model
    config.embedding_base_url = args.embed_base_url
    config.embedding_batch_size = args.embed_batch
    config.max_new_tokens = args.max_new_tokens
    config.retrieval_top_k = args.top_k
    config.openie_mode = "online"
    # Граф пересобирается с нуля только по --rebuild-graph: кэш ответов
    # модели и файл OpenIE при этом переиспользуются, так что пересборка
    # после пробы стоит минут эмбеддинга, а не часов извлечения.
    config.force_index_from_scratch = bool(args.rebuild_graph)
    config.force_openie_from_scratch = False
    # Параметры графа и поиска — по умолчанию HippoRAG 2 (порог синонимов
    # 0.8, linking_top_k 5, damping 0.5, passage_node_weight 0.05): мы
    # сравниваемся с опубликованным методом, а не с его подстройкой.
    return config


def content_index(chunks: list[dict]) -> dict[str, list[str]]:
    """Текст → идентификаторы. HippoRAG сливает одинаковые тексты в один узел."""
    index: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        index[chunk["text"]].append(chunk["chunk_id"])
    return index


def to_ids(docs: list[str], index: dict[str, list[str]]) -> tuple[list[str], int]:
    ranked: list[str] = []
    unknown = 0
    for text in docs:
        ids = index.get(text)
        if ids is None:
            unknown += 1
            continue
        ranked.extend(i for i in ids if i not in ranked)
    return ranked, unknown


def openie_stats(out: Path, llm_model: str, texts: list[str]) -> dict:
    """Качество извлечения по файлу OpenIE — главный признак негодного графа.

    Фрагмент без триплетов не получает ни одного ребра фактов, и HippoRAG 2
    находит его только через плотную часть. Если таких много, мы меряем
    не HippoRAG 2, а плотный поиск с шумом, — это видно здесь, до ответов.
    """
    path = out / f"openie_results_ner_{llm_model.replace('/', '_')}.json"
    if not path.exists():
        return {"file": str(path), "missing": True}
    wanted = set(texts)
    docs = [d for d in json.loads(path.read_text(encoding="utf-8"))["docs"] if d["passage"] in wanted]
    n = max(1, len(docs))
    triples = [len(d.get("extracted_triples") or []) for d in docs]
    entities = [len(d.get("extracted_entities") or []) for d in docs]
    # Триплет не из трёх частей HippoRAG отбрасывает при построении графа.
    malformed = sum(1 for d in docs for t in (d.get("extracted_triples") or []) if len(t) != 3)
    return {
        "docs": len(docs), "of_chunks": len(wanted),
        "no_entities_share": round(sum(1 for e in entities if e == 0) / n, 4),
        "no_triples_share": round(sum(1 for t in triples if t == 0) / n, 4),
        "triples_mean": round(sum(triples) / n, 2),
        "entities_mean": round(sum(entities) / n, 2),
        "malformed_triples": malformed,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--embed-base-url", required=True)
    parser.add_argument("--embed-model", default="BAAI/bge-m3")
    parser.add_argument("--embed-batch", type=int, default=32)
    parser.add_argument("--reasoning-effort", default="none",
                        help="передаётся модели; пустая строка — не передавать")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=200, help="длина сохраняемой выдачи")
    parser.add_argument("--probe-questions", type=int, default=0,
                        help="проба: первые N вопросов, их эталонные фрагменты и --probe-extra отвлекающих")
    parser.add_argument("--probe-extra", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0,
                        help="параллельных запросов извлечения (0 — как в пакете)")
    parser.add_argument("--systems", default="hipporag2,dense")
    parser.add_argument("--index-only", action="store_true", help="только построение графа")
    parser.add_argument("--rebuild-graph", action="store_true",
                        help="граф заново (после пробы); кэш модели и OpenIE сохраняются")
    args = parser.parse_args()

    chunks = load_jsonl(args.bundle / "chunks.jsonl")
    questions = load_jsonl(args.bundle / "questions.jsonl")
    if args.probe_questions:
        questions = questions[: args.probe_questions]
        gold = {g for q in questions for g in q["gold_chunk_ids"]}
        extra = [c for c in chunks if c["chunk_id"] not in gold][: args.probe_extra]
        chunks = [c for c in chunks if c["chunk_id"] in gold] + extra
        print(f"проба: {len(questions)} вопросов, {len(chunks)} фрагментов ({len(extra)} отвлекающих)")
    args.out.mkdir(parents=True, exist_ok=True)

    HippoRAG = patch_hipporag(args)
    rag = HippoRAG(global_config=make_config(args))

    started = time.time()
    rag.index([c["text"] for c in chunks])
    index_seconds = round(time.time() - started, 1)
    graph_info = rag.get_graph_info()
    print(f"индекс за {index_seconds} с: {json.dumps(graph_info, ensure_ascii=False)}")

    run = {
        "bundle": str(args.bundle), "chunks": len(chunks), "questions": len(questions),
        "bundle_sha256": {n: sha256(args.bundle / n) for n in ("chunks.jsonl", "questions.jsonl")},
        "llm": {"model": args.llm_model, "base_url": args.llm_base_url,
                "reasoning_effort": args.reasoning_effort},
        "embedding": {"model": args.embed_model, "base_url": args.embed_base_url},
        "index_seconds": index_seconds, "graph": graph_info,
        "openie": openie_stats(args.out, args.llm_model, [c["text"] for c in chunks]),
    }
    print(f"извлечение: {json.dumps(run['openie'], ensure_ascii=False)}")
    if args.index_only:
        (args.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    index = content_index(chunks)
    texts = [q["question"] for q in questions]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    # Сколько вопросов фильтр фактов оставил пустыми: тогда HippoRAG 2 молча
    # отдаёт плотную выдачу, и «HippoRAG 2» по этим вопросам — просто DPR.
    # На малой модели это может быть заметная доля; без счётчика не видно.
    # Отдельно — исключения: пакет их глотает и тоже уходит в плотный поиск,
    # так что сбой модели выглядел бы как «фильтр ничего не выбрал».
    fallback = {"n": 0, "errors": 0}
    rerank = rag.rerank_facts

    def counted(query, scores):
        result = rerank(query, scores)
        if len(result[1]) == 0:
            fallback["n"] += 1
            if result[2].get("error"):
                fallback["errors"] += 1
        return result

    rag.rerank_facts = counted

    for system in systems:
        started = time.time()
        if system == "hipporag2":
            solutions = rag.retrieve(texts, num_to_retrieve=args.top_k)
        elif system == "dense":
            solutions = rag.retrieve_dpr(texts, num_to_retrieve=args.top_k)
        else:
            raise SystemExit(f"неизвестная система: {system}")
        seconds = round(time.time() - started, 1)
        unknown_total = 0
        path = args.out / f"rankings-{system}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for q, solution in zip(questions, solutions):
                ranked, unknown = to_ids(list(solution.docs), index)
                unknown_total += unknown
                handle.write(json.dumps({"qid": q["qid"], "ranked": ranked,
                                         "scores": [round(float(s), 6) for s in solution.doc_scores]},
                                        ensure_ascii=False) + "\n")
        run[system] = {"seconds": seconds, "per_question_ms": round(1000 * seconds / max(1, len(texts)), 1),
                       "unmapped_docs": unknown_total}
        if system == "hipporag2":
            run[system]["filter_empty_fallback_to_dense"] = fallback["n"]
            run[system]["filter_empty_share"] = round(fallback["n"] / max(1, len(texts)), 4)
            run[system]["filter_errors"] = fallback["errors"]
        print(f"{system}: {json.dumps(run[system], ensure_ascii=False)}")

    (args.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
