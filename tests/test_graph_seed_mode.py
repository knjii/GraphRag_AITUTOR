"""От чего отталкивается обход графа.

Замер на 348 фрагментах показал, что графовый канал, стартующий от терминов
вопроса, вырождается в ослабленный лексический поиск: его уникальный вклад —
2.3 процентных пункта recall, а вносимый шум выше. Причина в замысле: термины
вопроса — тот же сигнал, по которому уже работает BM25.

Режим ``passages`` отталкивается от сущностей **уже найденных** фрагментов.
Это другой сигнал: не «что похоже на вопрос», а «что связано с найденным».
Здесь проверяется именно это различие.
"""

from __future__ import annotations

from rag_textbook.clients.llm import FakeLLMClient
from rag_textbook.config import GraphSettings
from rag_textbook.graph.builder import GraphBuilder
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.models import Chunk, content_hash
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from tests.fakes import InMemoryGraphStore

# Корпус, в котором ответ на связывающий вопрос лежит НЕ там, где термины вопроса.
# Фрагмент 0 отвечает на вопрос про SVD напрямую. Фрагмент 20 содержит связь
# SVD с методом главных компонент, но слова «сингулярное разложение» в вопросе
# ведут прежде всего к фрагменту 0.
CORPUS: list[tuple[int, str]] = [
    (0, "Сингулярное разложение матрицы раскладывает её на три множителя."),
    (1, "Ортогональные матрицы сохраняют длины векторов при умножении."),
    (2, "Ковариационная матрица описывает совместную изменчивость признаков."),
    (20, "Метод главных компонент выражается через сингулярное разложение."),
]

EXTRACTIONS: dict[str, str] = {
    "d:00000": (
        '{"entities": [{"name": "сингулярное разложение"}], '
        '"relations": []}'
    ),
    "d:00001": '{"entities": [{"name": "ортогональная матрица"}], "relations": []}',
    "d:00002": '{"entities": [{"name": "ковариационная матрица"}], "relations": []}',
    "d:00020": (
        '{"entities": [{"name": "метод главных компонент"}, '
        '{"name": "сингулярное разложение"}], '
        '"relations": [{"source": "метод главных компонент", '
        '"relation": "вычисляется_по", "target": "сингулярное разложение"}]}'
    ),
}


class _ScriptedLLM(FakeLLMClient):
    def __init__(self, mapping: dict[str, str]) -> None:
        super().__init__()
        self._mapping = mapping

    def chat(
        self, messages, *, purpose="chat", json_schema=None, max_tokens=None, temperature=None
    ):
        if json_schema is not None:
            prompt = messages[-1].content
            for chunk_id, payload in self._mapping.items():
                marker = dict(CORPUS)[int(chunk_id.split(":")[1])][:40]
                if marker in prompt:
                    return payload
            return '{"entities": [], "relations": []}'
        return super().chat(messages, purpose=purpose, json_schema=json_schema)


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id=f"d:{ordinal:05d}",
            doc_id="d",
            doc_name="Линейная алгебра",
            source_path="/linal.pdf",
            ordinal=ordinal,
            text=text,
            pages=[ordinal + 1],
            text_hash=content_hash(text),
        )
        for ordinal, text in CORPUS
    ]


def _settings(**overrides) -> GraphSettings:
    defaults = {
        "enabled": True,
        "retrieval_enabled": True,
        "extractor": "llm",
        "extraction_cache_enabled": False,
        "max_entity_degree": 0,
        "cooccurrence_enabled": False,
        "expansion_hops": 1,
    }
    defaults.update(overrides)
    return GraphSettings(**defaults)


def _store(settings: GraphSettings) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    builder = GraphBuilder(
        settings, EntityExtractor(settings, llm=_ScriptedLLM(EXTRACTIONS)), store, max_workers=1
    )
    builder.build(
        _chunks(), doc_id="d", doc_name="Линейная алгебра", source_path="/linal.pdf", write=True
    )
    return store


def test_passage_mode_reaches_connected_material() -> None:
    """Опора на найденный фрагмент выводит на связанный с ним, а не на него же."""
    settings = _settings(seed_mode="passages", seed_passages=1)
    retriever = GraphRetriever(settings, _store(settings))

    found = retriever.retrieve("Что такое сингулярное разложение?", seed_chunk_ids=["d:00000"])
    ids = [item.chunk.id for item in found]

    assert "d:00020" in ids, "фрагмент, связанный с опорным через граф, должен найтись"


def test_passage_mode_excludes_the_seeds_themselves() -> None:
    """Канал обязан приносить новое, а не возвращать уже найденное.

    Иначе он раздувает долю графа в контексте, ничего к нему не добавляя, —
    ровно так выглядела статистика «22.8% контекста от графа» при нулевом
    приросте метрик.
    """
    settings = _settings(seed_mode="passages", seed_passages=2)
    retriever = GraphRetriever(settings, _store(settings))

    seeds = ["d:00000", "d:00020"]
    found = retriever.retrieve("сингулярное разложение", seed_chunk_ids=seeds)

    assert not ({item.chunk.id for item in found} & set(seeds))


def test_query_mode_ignores_the_seeds() -> None:
    """Прежнее поведение сохраняется без изменений."""
    settings = _settings(seed_mode="query")
    retriever = GraphRetriever(settings, _store(settings))

    with_seeds = retriever.retrieve("сингулярное разложение", seed_chunk_ids=["d:00000"])
    without = retriever.retrieve("сингулярное разложение")

    assert [item.chunk.id for item in with_seeds] == [item.chunk.id for item in without]


def test_both_mode_combines_sources() -> None:
    settings = _settings(seed_mode="both", seed_passages=1)
    retriever = GraphRetriever(settings, _store(settings))

    found = retriever.retrieve("ковариационная матрица", seed_chunk_ids=["d:00000"])
    ids = {item.chunk.id for item in found}

    # Термин вопроса ведёт к фрагменту 2, опорный фрагмент — к фрагменту 20.
    assert "d:00002" in ids
    assert "d:00020" in ids


def test_passage_mode_without_seeds_returns_nothing() -> None:
    """Без опорных фрагментов режим не должен молча падать в поиск по вопросу."""
    settings = _settings(seed_mode="passages")
    retriever = GraphRetriever(settings, _store(settings))

    assert retriever.retrieve("сингулярное разложение") == []


def test_rare_entities_outweigh_common_ones() -> None:
    """Сущность, встречающаяся всюду, не должна задавать направление обхода."""
    settings = _settings(seed_mode="passages")
    store = _store(settings)

    rows = store.entities_of_passages(["d:00020"], limit=10)

    def find(marker: str) -> dict:
        # Канонические формы лемматизированы («сингулярный разложение»),
        # поэтому ищем по корню, а не по точному совпадению.
        matches = [row for row in rows if marker in row["canonical"]]
        assert matches, f"сущность «{marker}» не найдена среди {[r['canonical'] for r in rows]}"
        return matches[0]

    common = find("разложение")
    rare = find("главн")
    assert common["document_frequency"] > rare["document_frequency"]
    assert rare["weight"] > common["weight"], (
        "редкая сущность должна весить больше частотной при равном числе упоминаний"
    )
