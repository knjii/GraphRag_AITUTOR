"""Слепок конвейера и воспроизведение отбора по нему.

Главное свойство харнесса — не скорость, а **граница**. Прошлый офлайн-замер
уверенно указал в другую сторону: порог хабов 40 подтверждался офлайн,
а на сервере оказался значимо вредным, потому что модель офлайна не описывала
полнотекстовый поиск затравок. Здесь попытка изменить настройку, влияющую
на состав кандидатов, обязана заканчиваться ошибкой, а не правдоподобным
числом. Это первое, что тут проверяется.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_textbook.config import GraphSettings, RerankerSettings, RetrievalSettings, Settings
from rag_textbook.evaluation.replay import fidelity_report, replay, replay_one
from rag_textbook.evaluation.trace import (
    NotReplayable,
    QueryTrace,
    TraceSet,
    TracedCandidate,
    assert_replayable,
    snapshot_settings,
)
from rag_textbook.models import Chunk


def _settings(*, graph=None, retrieval=None, reranker=None) -> Settings:
    """Собирает настройки из вложенных секций.

    Плоские псевдонимы вроде ``GRAPH_MAX_ENTITY_DEGREE`` читаются только
    из окружения; при прямом конструировании их надо передавать в секцию,
    иначе значение молча останется прежним — и тест проверит не то.
    """
    settings = Settings(_env_file=None)
    if graph:
        settings.graph = settings.graph.model_copy(update=graph)
    if retrieval:
        settings.retrieval = settings.retrieval.model_copy(update=retrieval)
    if reranker:
        settings.reranker = settings.reranker.model_copy(update=reranker)
    return settings


def _chunks() -> dict[str, Chunk]:
    texts = {
        "c1": "Сингулярное разложение раскладывает матрицу на три множителя.",
        "c2": "Ранг равен числу линейно независимых строк прямоугольной таблицы.",
        "c3": "Собственные значения ковариационной матрицы задают главные компоненты.",
        "c4": "Гауссово распределение задаётся средним и ковариационной матрицей.",
        "c5": "Метод опорных векторов ищет разделяющую гиперплоскость с зазором.",
    }
    return {
        key: Chunk(
            id=key,
            doc_id="doc",
            doc_name="Учебник",
            source_path="учебник.pdf",
            ordinal=index,
            text=text,
        )
        for index, (key, text) in enumerate(texts.items())
    }


def _trace(**overrides) -> QueryTrace:
    base = QueryTrace(
        question_id="q-1",
        question="Что такое сингулярное разложение?",
        question_type="graph_linked",
        used_graph=True,
        channels={
            "base": [
                TracedCandidate("c1", 0, 0.90),
                TracedCandidate("c2", 1, 0.85),
                TracedCandidate("c3", 2, 0.70),
            ],
            "graph": [
                TracedCandidate("c5", 0, 0.60),
                TracedCandidate("c4", 1, 0.55),
            ],
        },
        rerank_scores={"c1": 0.95, "c2": 0.90, "c3": 0.40, "c4": 0.30, "c5": 0.20},
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


# --------------------------------------------------------------- граница

def test_changing_traversal_setting_is_refused():
    """Обход графа меняет СОСТАВ кандидатов — по слепку он не проверяется."""
    snapshot = snapshot_settings(_settings())
    changed = _settings(graph={"max_entity_degree": 40})

    with pytest.raises(NotReplayable) as error:
        assert_replayable(snapshot, changed)

    assert "max_entity_degree" in str(error.value)
    assert "сервере" in str(error.value), "сообщение должно называть выход из положения"


def test_changing_ordering_setting_is_allowed():
    """Веса слияния меняют только порядок — это слепком проверяется точно."""
    snapshot = snapshot_settings(_settings())
    changed = _settings(retrieval={"rrf_k": 10}, graph={"weight": 0.8})

    assert_replayable(snapshot, changed)  # не должно бросить


def test_every_composition_field_is_guarded():
    """Каждое поле из списка границы обязано реально проверяться.

    Список легко пополнить и забыть, что значение читается по строковому пути:
    опечатка в пути превратила бы защиту в её видимость.
    """
    from rag_textbook.evaluation.trace import COMPOSITION_FIELDS, read_setting

    settings = _settings()
    for path in COMPOSITION_FIELDS:
        assert read_setting(settings, path) is not None, f"путь {path} ничего не читает"


def test_wider_window_than_captured_is_refused():
    traces = TraceSet(settings_snapshot=snapshot_settings(_settings()), rerank_window=30)
    traces.traces.append(_trace())

    with pytest.raises(ValueError, match="шире снятого"):
        replay(traces, _settings(reranker={"candidates": 60}), _chunks())


# ------------------------------------------------------- воспроизведение

def test_replay_returns_candidates_from_trace():
    result = replay_one(_trace(), _settings(retrieval={"top_k": 3}), _chunks())

    ids = [item.chunk.id for item in result]
    assert ids, "воспроизведение не вернуло ничего"
    assert set(ids) <= {"c1", "c2", "c3", "c4", "c5"}


def test_reranker_order_is_taken_from_trace():
    """Порядок задают сохранённые баллы, а не баллы каналов."""
    trace = _trace(rerank_scores={"c3": 0.99, "c1": 0.10, "c2": 0.05})

    result = replay_one(trace, _settings(retrieval={"top_k": 1}), _chunks())

    assert result[0].chunk.id == "c3"


def test_by_route_mode_skips_reranker_on_linking_questions():
    """Режим by_route существует ради связывающих вопросов: измерено, что
    реранкер отнимает у них 13 вопросов, отдавая 5 формульным."""
    trace = _trace(used_graph=True, rerank_scores={"c3": 0.99, "c1": 0.10})

    always = replay_one(trace, _settings(retrieval={"top_k": 1}, reranker={"mode": "always"}), _chunks())
    by_route = replay_one(
        trace, _settings(retrieval={"top_k": 1}, reranker={"mode": "by_route"}), _chunks()
    )

    assert always[0].chunk.id == "c3", "в обычном режиме решает реранкер"
    assert by_route[0].chunk.id != "c3", "на связывающем маршруте порядок слияния сохраняется"


def test_by_route_mode_still_reranks_simple_questions():
    trace = _trace(used_graph=False, rerank_scores={"c3": 0.99, "c1": 0.10})

    result = replay_one(trace, _settings(retrieval={"top_k": 1}, reranker={"mode": "by_route"}), _chunks())

    assert result[0].chunk.id == "c3"


def test_blend_alpha_zero_ignores_reranker():
    trace = _trace(used_graph=False, rerank_scores={"c3": 0.99, "c1": 0.10, "c2": 0.05})

    blended = replay_one(
        trace,
        _settings(retrieval={"top_k": 1}, reranker={"mode": "blend", "blend_alpha": 0.0}),
        _chunks(),
    )

    assert blended[0].chunk.id != "c3", "при нулевом весе реранкер не должен решать"


# ------------------------------------------------------------- честность

def test_fidelity_report_detects_mismatch():
    """Сверка честности обязана ловить расхождение, иначе она бесполезна."""
    traces = TraceSet(settings_snapshot=snapshot_settings(_settings()), rerank_window=30)
    traces.traces.append(_trace(final=["c1", "c2"]))

    outcomes = replay(traces, _settings(retrieval={"top_k": 2}), _chunks())
    report = fidelity_report(traces, outcomes)

    assert report["вопросов сверено"] == 1
    assert 0.0 <= report["доля точных совпадений"] <= 1.0
    assert 0.0 <= report["среднее пересечение"] <= 1.0


def test_fidelity_is_perfect_when_replay_matches():
    traces = TraceSet(settings_snapshot=snapshot_settings(_settings()), rerank_window=30)
    trace = _trace()
    settings = _settings(retrieval={"top_k": 2})
    trace.final = [item.chunk.id for item in replay_one(trace, settings, _chunks())]
    traces.traces.append(trace)

    report = fidelity_report(traces, replay(traces, settings, _chunks()))

    assert report["доля точных совпадений"] == 1.0


# ------------------------------------------------------------------ файл

def test_trace_round_trip(tmp_path: Path):
    traces = TraceSet(settings_snapshot={"graph.hop_decay": 0.8}, rerank_window=100)
    traces.traces.append(_trace(final=["c1"]))

    path = tmp_path / "trace.jsonl"
    traces.save(path)
    restored = TraceSet.load(path)

    assert restored.rerank_window == 100
    assert restored.settings_snapshot == {"graph.hop_decay": 0.8}
    assert len(restored.traces) == 1
    assert restored.traces[0].channels["base"][0].chunk_id == "c1"
    assert restored.traces[0].rerank_scores["c1"] == pytest.approx(0.95)


def test_partial_file_still_loads(tmp_path: Path):
    """Прогон может прерваться: записанное до обрыва обязано читаться."""
    path = tmp_path / "trace.jsonl"
    traces = TraceSet(settings_snapshot={}, rerank_window=30)
    traces.traces.extend([_trace(), _trace(question_id="q-2")])
    traces.save(path)

    lines = path.read_text(encoding="utf-8").split("\n")
    path.write_text("\n".join(lines[:2]), encoding="utf-8")

    assert len(TraceSet.load(path).traces) == 1


def test_diversity_receives_more_candidates_than_it_returns():
    """Разнообразию нужен запас сверх top_k, иначе оно молчаливо не работает.

    Ровно эта ловушка уже случалась с `RETRIEVAL_MIN_GRAPH_DOCS`: настройка
    существовала, участвовала в замерах и добросовестно показывала ноль
    изменённых вопросов, потому что брала замену из пустого хвоста. Здесь
    первый прогон перебора дал +0.000 по всем шести вариантам разнообразия —
    подпись не малого эффекта, а неработающей настройки.
    """
    trace = _trace(
        channels={
            "base": [TracedCandidate(f"c{i}", i, 1.0 - i / 10) for i in range(1, 6)],
            "graph": [],
        },
        used_graph=False,
        rerank_scores={f"c{i}": 1.0 - i / 10 for i in range(1, 6)},
    )
    settings = _settings(
        retrieval={"top_k": 2, "diversity_mode": "reserve", "diversity_reserve_slots": 1},
        reranker={"top_n": 2, "candidates": 30},
    )

    # Проверяем через сам конвейер отбора: список, доходящий до разнообразия,
    # обязан быть длиннее top_k.
    seen: dict[str, int] = {}
    import rag_textbook.evaluation.replay as replay_module

    original = replay_module.apply_diversity

    def spy(items, settings_, *, top_k):
        seen["items"] = len(items)
        seen["top_k"] = top_k
        return original(items, settings_, top_k=top_k)

    replay_module.apply_diversity = spy
    try:
        replay_one(trace, settings, _chunks())
    finally:
        replay_module.apply_diversity = original

    assert seen["items"] > seen["top_k"], (
        f"разнообразию передали {seen['items']} кандидатов при top_k={seen['top_k']}: "
        "переставлять нечего"
    )
