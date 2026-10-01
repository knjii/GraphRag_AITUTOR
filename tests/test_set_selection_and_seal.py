"""Этап 2, серия S: отбор множеством (SetR, Context-Picker) и SEAL.

Модель подменяется заготовленными ответами. Проверяется то, что при сбое
было бы незаметно: отказ разбора обязан остаться порядком реранкера со
статусом fallback, SEAL не вправе ни расширить выдачу, ни вытеснить опорное.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

import pytest

from rag_textbook.clients.llm import FakeLLMClient
from rag_textbook.config import RetrievalSettings
from rag_textbook.evaluation.metrics import QueryOutcome, selection_metrics
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval import seal, selection, set_selection


def _item(chunk_id: str, score: float | None = None, text: str | None = None) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id,
        doc_id="d",
        doc_name="Книга",
        source_path="",
        ordinal=0,
        text=text or f"текст {chunk_id}",
        pages=[1],
    )
    return ScoredChunk(chunk=chunk, rerank_score=score, channels=["dense"])


def _settings(**values) -> RetrievalSettings:
    return RetrievalSettings().model_copy(update=values)


def _ids(items: Sequence[ScoredChunk]) -> list[str]:
    return [item.chunk.id for item in items]


# --- разбор строки выбора ---------------------------------------------------


def test_parse_takes_last_marker_and_first_line() -> None:
    raw = (
        "Формат: ### Final Selection: [9]\n"
        "Step 3...\n### Final Selection: [3] [1] [3] [7]\nи ещё [2]"
    )
    # Последняя метка, повтор [3] снят, [7] вне пула из 5, [2] на следующей строке.
    assert set_selection.parse_selection(raw, 5) == [2, 0]


def test_parse_distinguishes_missing_marker_from_empty_choice() -> None:
    assert set_selection.parse_selection("рассуждение оборвалось", 5) is None
    assert set_selection.parse_selection("### Final Selection:", 5) == []


# --- отбор множеством -------------------------------------------------------


def test_select_set_puts_chosen_first_and_keeps_pool() -> None:
    items = [_item(f"c{i}", 1.0 - i / 10) for i in range(6)]
    llm = FakeLLMClient(["Step 1 ... ### Final Selection: [4] [2]"])
    outcome = set_selection.select_set(items, "вопрос", llm, _settings(), top_k=16, mode="setr")
    assert outcome.status == "ok"
    assert _ids(outcome.ordered) == ["c3", "c1", "c0", "c2", "c4", "c5"]
    assert outcome.chosen == ["c3", "c1"]
    assert set_selection.selected_ids(outcome.ordered) == ["c3", "c1"]
    # Подсказка SetR ушла дословно: шаги IRI и формат вывода.
    prompt = llm.calls[0][-1].content
    assert "Step 1. Please list up the information requirements" in prompt
    assert "[1] текст c0" in prompt


def test_select_set_window_limits_what_model_sees() -> None:
    items = [_item(f"c{i}") for i in range(10)]
    llm = FakeLLMClient(["### Final Selection: [9]"])
    outcome = set_selection.select_set(
        items, "вопрос", llm, _settings(selection_llm_pool=4), top_k=16, mode="picker"
    )
    # Номер 9 вне окна из четырёх: строка есть, годных номеров нет.
    assert outcome.status == "empty"
    assert _ids(outcome.ordered) == _ids(items)
    assert "[5]" not in llm.calls[0][-1].content


def test_select_set_fallback_keeps_reranker_order() -> None:
    items = [_item(f"c{i}") for i in range(4)]
    outcome = set_selection.select_set(
        items, "вопрос", FakeLLMClient(["без итоговой строки"]), _settings(), top_k=16, mode="setr"
    )
    assert outcome.status == "fallback"
    assert _ids(outcome.ordered) == _ids(items)
    assert set_selection.selected_ids(outcome.ordered) == []


def test_select_set_caps_choice_at_top_k() -> None:
    items = [_item(f"c{i}") for i in range(6)]
    llm = FakeLLMClient(["### Final Selection: [1] [2] [3] [4]"])
    outcome = set_selection.select_set(items, "вопрос", llm, _settings(), top_k=2, mode="picker")
    assert outcome.chosen == ["c0", "c1"]


def test_reorder_counts_statuses_and_requires_llm() -> None:
    items = [_item(f"c{i}") for i in range(4)]
    settings = _settings(selection_mode="setr")
    with pytest.raises(ValueError, match="языковую модель"):
        selection.check_ready(settings, None, None, None)
    stats: Counter = Counter()
    ordered = selection.reorder(
        items, "вопрос", settings, 16, llm=FakeLLMClient(["### Final Selection: [2]"]), stats=stats
    )
    assert _ids(ordered)[0] == "c1"
    assert stats == Counter({"ok": 1})


# --- SEAL -------------------------------------------------------------------


def _ledger(found: dict[int, str], missing: Sequence[str] = (), sufficient: bool = False) -> str:
    return json.dumps(
        {
            "found": [{"fact": fact, "passages": [number]} for number, fact in found.items()],
            "missing": [{"need": query, "query": query} for query in missing],
            "sufficient": sufficient,
        }
    )


def _score_by_prefix(query: str, texts: Sequence[str]) -> list[float]:
    # Балл против микрозапроса — совпадение слова запроса в тексте.
    return [1.0 if query in text else 0.0 for text in texts]


def test_seal_replaces_worst_non_supporting_keeps_size() -> None:
    initial = [_item("a", 0.9), _item("b", 0.5), _item("c", 0.1), _item("d", 0.3)]
    outside = [_item("x", text="про теорему Штольца"), _item("y", text="шум")]
    llm = FakeLLMClient(
        [
            _ledger({1: "факт из a"}, missing=["Штольца"]),
            _ledger({1: "факт", 2: "второй"}, sufficient=True),
        ]
    )
    result = seal.run(
        "вопрос",
        initial,
        search=lambda query: outside,
        score=_score_by_prefix,
        llm=llm,
        settings=_settings(seal_candidates_per_gap=1),
        top_k=4,
    )
    # Вытеснен c — худший неопорный; x стал вторым, за опорным a.
    assert _ids(result.final) == ["a", "x", "b", "d"]
    assert result.added == ["x"]
    assert seal.SEAL_CHANNEL in result.final[1].channels
    assert result.status == "ok" and result.loops == 2


def test_seal_never_evicts_supporting() -> None:
    initial = [_item("a", 0.1), _item("b", 0.2)]
    llm = FakeLLMClient([_ledger({1: "a", 2: "b"}, missing=["нечто"])])
    result = seal.run(
        "вопрос",
        initial,
        search=lambda query: [_item("x", text="нечто")],
        score=_score_by_prefix,
        llm=llm,
        settings=_settings(),
        top_k=2,
    )
    assert _ids(result.final) == ["a", "b"]
    assert result.added == []


def test_seal_fallback_on_unparsable_ledger() -> None:
    initial = [_item("a", 0.9), _item("b", 0.5)]
    result = seal.run(
        "вопрос",
        initial,
        search=lambda query: [],
        score=_score_by_prefix,
        llm=FakeLLMClient(["не JSON"]),
        settings=_settings(),
        top_k=2,
    )
    assert result.status == "fallback"
    assert _ids(result.final) == ["a", "b"]


def test_seal_skips_items_already_present() -> None:
    initial = [_item("a", 0.9), _item("b", 0.1)]
    llm = FakeLLMClient([_ledger({1: "a"}, missing=["b"]), _ledger({1: "a"})])
    result = seal.run(
        "вопрос",
        initial,
        search=lambda query: [_item("b")],
        score=_score_by_prefix,
        llm=llm,
        settings=_settings(),
        top_k=2,
    )
    assert result.added == []
    assert _ids(result.final) == ["a", "b"]


# --- сводка -----------------------------------------------------------------


def test_selection_metrics_counts_gold_outside_pool() -> None:
    outcomes = [
        QueryOutcome(
            question_id="q1",
            question_type="t",
            retrieved=["a", "x"],
            relevant=["a", "x"],
            selected=["a"],
            seal_added=["x"],
            pool=["a", "b"],
            selection_status="seal_ok",
        ),
        QueryOutcome(
            question_id="q2",
            question_type="t",
            retrieved=["c"],
            relevant=["c"],
            selected=[],
            pool=["c"],
            selection_status="seal_fallback",
        ),
    ]
    values = selection_metrics(outcomes)
    assert values["set_size"] == 1.0
    assert values["set_recall"] == pytest.approx(0.25)  # 0.5 и 0
    assert values["set_full"] == 0.0
    assert values["seal_gold_outside_pool"] == pytest.approx(0.5)
    assert values["status_seal_fallback"] == pytest.approx(0.5)


def test_selection_metrics_empty_without_stage_two() -> None:
    outcome = QueryOutcome(question_id="q", question_type="t", retrieved=["a"], relevant=["a"])
    assert selection_metrics([outcome]) == {}


def test_parse_numbers_on_next_line() -> None:
    assert set_selection.parse_selection("### Final Selection:\n[2] [1]", 3) == [1, 0]
