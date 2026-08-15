"""Отбор многошаговых пар по связям графа, а не по общим словам.

Ограничение прежнего отбора обнаружилось при разборе результатов: пары
брались по числу общих содержательных терминов (не менее трёх), то есть
были лексически похожи. На таких парах векторный и лексический каналы
справляются сами, и вопрос «даёт ли граф прирост» на них неразрешим —
измеритель по построению слеп к тому, что должен измерять.
"""

from __future__ import annotations

from rag_textbook.clients.llm import FakeLLMClient
from rag_textbook.evaluation.goldset import GoldsetBuilder
from rag_textbook.models import Chunk, content_hash


class _StubGraph:
    """Отдаёт заданные пары; повторяет контракт боевого хранилища."""

    def __init__(self, pairs: list[tuple[str, str]], fail: bool = False) -> None:
        self._pairs = pairs
        self._fail = fail
        self.requests: list[tuple[int, int]] = []

    def linked_passage_pairs(self, limit: int, min_distance: int = 10):
        self.requests.append((limit, min_distance))
        if self._fail:
            raise RuntimeError("граф недоступен")
        return [
            {"left": left, "right": right, "links": 3} for left, right in self._pairs[:limit]
        ]


def _chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(
        id=f"d:{ordinal:05d}",
        doc_id="d",
        doc_name="Учебник",
        source_path="/book.pdf",
        ordinal=ordinal,
        text=text,
        pages=[ordinal + 1],
        text_hash=content_hash(text),
    )


# Пара 0-40 связана структурно, но почти не пересекается по словам.
# Пара 0-41 наоборот: те же слова, что и в первом фрагменте.
CHUNKS = [
    _chunk(0, "Сингулярное разложение раскладывает матрицу на три множителя, "
              "среди которых диагональная матрица сингулярных чисел."),
    _chunk(40, "Сжатие изображений опирается на отбрасывание наименее значимых "
               "компонент представления и оценку потери информации."),
    _chunk(41, "Сингулярное разложение матрицы даёт три множителя, включая "
               "диагональную матрицу сингулярных чисел выборки."),
]


def _builder(graph, **kwargs) -> GoldsetBuilder:
    return GoldsetBuilder(FakeLLMClient(), graph_store=graph, **kwargs)


def test_structurally_linked_pair_is_selected() -> None:
    graph = _StubGraph([("d:00000", "d:00040")])
    pairs = _builder(graph)._select_graph_linked_pairs(CHUNKS, count=5)

    assert [(left.id, right.id) for left, right in pairs] == [("d:00000", "d:00040")]


def test_lexically_similar_pair_is_rejected() -> None:
    """Ради этого отбор и появился: похожая по словам пара ничего не проверяет."""
    graph = _StubGraph([("d:00000", "d:00041")])
    pairs = _builder(graph)._select_graph_linked_pairs(CHUNKS, count=5)

    assert pairs == []


def test_threshold_is_configurable() -> None:
    """Порог пересечения — настройка, а не зашитое число."""
    graph = _StubGraph([("d:00000", "d:00041")])
    pairs = _builder(graph, max_lexical_overlap=1.0)._select_graph_linked_pairs(CHUNKS, count=5)

    assert len(pairs) == 1


def test_missing_graph_falls_back_silently() -> None:
    """Без графа сборка обязана работать по-прежнему."""
    assert _builder(None)._select_graph_linked_pairs(CHUNKS, count=5) == []


def test_graph_failure_does_not_break_the_build() -> None:
    """Отказ графа не должен ронять сборку набора."""
    graph = _StubGraph([], fail=True)
    assert _builder(graph)._select_graph_linked_pairs(CHUNKS, count=5) == []


def test_unknown_chunk_ids_are_skipped() -> None:
    """Граф может знать фрагменты, которых нет в переданном списке."""
    graph = _StubGraph([("d:00000", "d:99999")])
    assert _builder(graph)._select_graph_linked_pairs(CHUNKS, count=5) == []


def test_graph_linked_is_a_valid_question_type() -> None:
    """Тип должен быть объявлен в модели, иначе сборка падает при сохранении.

    Регрессия: отбор пар работал, а сборка набора падала на валидации
    первого же вопроса — тип `graph_linked` не был добавлен в перечисление.
    """
    from rag_textbook.models import GoldQuestion

    question = GoldQuestion(
        id="x",
        question="вопрос",
        gold_chunk_ids=["a", "b"],
        question_type="graph_linked",
        expected_hops=2,
    )
    assert question.question_type == "graph_linked"


def test_minimum_distance_is_passed_through() -> None:
    """Соседние фрагменты перекрываются, связь между ними ничего не доказывает."""
    graph = _StubGraph([])
    _builder(graph, min_ordinal_distance=25)._select_graph_linked_pairs(CHUNKS, count=5)

    assert graph.requests and graph.requests[0][1] == 25
