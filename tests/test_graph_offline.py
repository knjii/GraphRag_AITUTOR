"""Проверки офлайн-оценки графового канала.

Смысл модуля в том, чтобы перебирать варианты обхода по локальному кэшу,
не поднимая ни Neo4j, ни сервер инференса. Поэтому проверяется не «функция
что-то вернула», а три свойства, ради которых он написан: восстановленный
граф совпадает с тем, что записалось бы в Neo4j; отсечение хабов действительно
сужает достижимое множество; расхождение настроек с кэшем видно сразу,
а не превращается в тихо неверные числа.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from rag_textbook.evaluation.graph_offline import (
    linked_pairs,
    rank_from_passage,
    reconstruct,
    second_hop_ranks,
    summarize,
)
from rag_textbook.models import content_hash

MODEL = "test-model"
EFFORT = "none"


def _entity(name: str) -> dict[str, object]:
    return {"id": f"e:{name}", "name": name, "canonical": name, "aliases": [], "count": 1}


def _write_corpus(settings, chunks: list[dict], extractions: dict[str, dict]) -> None:
    """Кладёт разбор и кэш извлечения туда, где их ищет восстановление."""
    settings.paths.parsed_dir.mkdir(parents=True, exist_ok=True)
    (settings.paths.parsed_dir / "doc_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    settings.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.paths.cache_dir / "extraction.sqlite3"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache_entries (namespace TEXT NOT NULL, key TEXT NOT NULL, "
        "value TEXT NOT NULL, created_at REAL NOT NULL DEFAULT 0, PRIMARY KEY (namespace, key))"
    )
    graph = settings.graph
    for chunk in chunks:
        key = content_hash(
            chunk["text_hash"],
            MODEL,
            EFFORT,
            graph.extraction_prompt_version,
            str(graph.max_entities_per_chunk),
            str(graph.max_relations_per_chunk),
        )
        payload = extractions.get(chunk["id"])
        if payload is None:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO cache_entries (namespace, key, value) VALUES (?, ?, ?)",
            ("extraction", key, json.dumps(payload, ensure_ascii=False)),
        )
    conn.commit()
    conn.close()


def _chunk(index: int, text: str) -> dict[str, object]:
    return {
        "id": f"doc:{index:05d}",
        "doc_id": "doc",
        "doc_name": "doc",
        "source_path": "doc.pdf",
        "ordinal": index,
        "text": text,
        "text_hash": content_hash(text),
        "pages": [index],
    }


@pytest.fixture
def corpus(settings):
    """Три фрагмента: два связаны редким понятием, третий — частым.

    Такая расстановка отделяет сигнал от шума: частое понятие есть везде
    и потому ничего не различает, редкое встречается ровно в нужной паре.
    """
    chunks = [_chunk(index, f"фрагмент {index}") for index in range(3)]
    common, rare, bridge = _entity("матрица"), _entity("холецкий"), _entity("разложение")
    extractions = {
        "doc:00000": {
            "entities": [common, rare],
            "relations": [],
            "status": "ok",
        },
        "doc:00001": {
            "entities": [common, bridge],
            "relations": [],
            "status": "ok",
        },
        "doc:00002": {
            "entities": [common, rare],
            "relations": [],
            "status": "ok",
        },
    }
    _write_corpus(settings, chunks, extractions)
    return settings


def test_reconstructed_graph_matches_extraction_cache(corpus):
    graph = reconstruct(corpus, model=MODEL, reasoning_effort=EFFORT)

    assert graph.cache_hits == 3
    assert graph.cache_misses == 0
    assert graph.entities == 3
    assert graph.mentions["doc:00000"].keys() == {"e:матрица", "e:холецкий"}


def test_rare_entity_outranks_common_one(corpus):
    """Ради этого и добавлен IDF: без него частое понятие тянет всё подряд."""
    graph = reconstruct(corpus, model=MODEL, reasoning_effort=EFFORT)

    with_idf = rank_from_passage(graph, "doc:00000", hop_decay=0.0, use_idf=True)
    without_idf = rank_from_passage(graph, "doc:00000", hop_decay=0.0, use_idf=False)

    # Второй фрагмент делит с первым только редкое понятие, третий — только частое.
    assert with_idf[0] == "doc:00002"
    # Без веса редкости оба конкурента набирают одинаково и порядок ничем не задан.
    assert set(without_idf) == {"doc:00001", "doc:00002"}


def test_hub_pruning_shrinks_what_traversal_reaches(settings):
    """Порог отсечения — главный рычаг: он задаёт, сколько вообще достижимо."""
    chunks = [_chunk(index, f"фрагмент {index}") for index in range(6)]
    hub = _entity("хаб")
    extractions = {}
    for index, chunk in enumerate(chunks):
        own = _entity(f"своё{index}")
        # Хаб связан со всеми: ровно то, что делает обход перебором.
        extractions[chunk["id"]] = {
            "entities": [hub, own],
            "relations": [
                {
                    "source_id": hub["id"],
                    "target_id": own["id"],
                    "label": "используется_в",
                    "chunk_id": chunk["id"],
                    "doc_id": "doc",
                    "weight": 1.0,
                }
            ],
            "status": "ok",
        }
    _write_corpus(settings, chunks, extractions)

    wide = reconstruct(settings, max_entity_degree=0, model=MODEL, reasoning_effort=EFFORT)
    narrow = reconstruct(settings, max_entity_degree=4, model=MODEL, reasoning_effort=EFFORT)

    reachable_wide = rank_from_passage(wide, "doc:00000", hop_decay=0.8, use_idf=True)
    reachable_narrow = rank_from_passage(narrow, "doc:00000", hop_decay=0.8, use_idf=True)
    assert len(reachable_wide) == 5
    assert len(reachable_narrow) < len(reachable_wide)
    assert narrow.pruned_entities == 1


def test_hop_decay_zero_disables_expansion(settings):
    """Нулевое затухание должно означать «без расширения», а не «вес ноль»."""
    chunks = [_chunk(index, f"фрагмент {index}") for index in range(2)]
    left, right = _entity("левое"), _entity("правое")
    _write_corpus(
        settings,
        chunks,
        {
            "doc:00000": {"entities": [left], "relations": [
                {"source_id": left["id"], "target_id": right["id"], "label": "обобщает",
                 "chunk_id": "doc:00000", "doc_id": "doc", "weight": 1.0}
            ], "status": "ok"},
            "doc:00001": {"entities": [right], "relations": [], "status": "ok"},
        },
    )
    graph = reconstruct(settings, model=MODEL, reasoning_effort=EFFORT)

    assert rank_from_passage(graph, "doc:00000", hop_decay=0.8, use_idf=True) == ["doc:00001"]
    assert rank_from_passage(graph, "doc:00000", hop_decay=0.0, use_idf=True) == []


def test_settings_mismatch_is_reported_not_silently_wrong(corpus):
    """Кэш переживает смену движка, и записи разных прогонов лежат рядом."""
    with pytest.raises(RuntimeError, match="кэше извлечения"):
        reconstruct(corpus, model="другая-модель", reasoning_effort=EFFORT)


def test_second_hop_measures_both_directions(corpus):
    graph = reconstruct(corpus, model=MODEL, reasoning_effort=EFFORT)
    ranks = second_hop_ranks(
        graph, [("doc:00000", "doc:00002")], hop_decay=0.0, use_idf=True
    )

    assert len(ranks) == 2
    summary = summarize(ranks)
    assert summary["measurements"] == 2
    assert summary["mrr"] == pytest.approx(1.0)
    assert summary["hit@8"] == pytest.approx(1.0)


def test_pairs_come_from_goldset(corpus):
    from rag_textbook.evaluation.goldset import save_goldset
    from rag_textbook.models import GoldQuestion

    save_goldset(
        [
            GoldQuestion(
                id="q1",
                question="вопрос о связи",
                gold_chunk_ids=["doc:00000", "doc:00002"],
                gold_doc_ids=["doc"],
                answer="ответ",
                question_type="graph_linked",
                expected_hops=2,
            ),
            GoldQuestion(
                id="q2",
                question="вопрос об одном фрагменте",
                gold_chunk_ids=["doc:00001"],
                gold_doc_ids=["doc"],
                answer="ответ",
                question_type="single_chunk",
                expected_hops=1,
            ),
        ],
        corpus.paths.goldset_dir / "goldset.json",
    )
    graph = reconstruct(corpus, model=MODEL, reasoning_effort=EFFORT)

    # Одношаговый вопрос не годится: у него нет второго фрагмента.
    assert linked_pairs(corpus, graph) == [("doc:00000", "doc:00002")]
