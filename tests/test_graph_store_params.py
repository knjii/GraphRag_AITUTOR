"""Передача параметров в Neo4j.

Регрессия на отказ, который стоил целого вывода: параметр Cypher с именем
``query`` передавался в ``Session.run`` именованным аргументом и перекрывал
первый позиционный аргумент самого метода. Поиск стартовых сущностей падал
на каждом вопросе, графовый канал молча отдавал пустоту, и замер показывал
«граф не даёт прироста» — хотя граф в поиске просто не участвовал.
"""

from __future__ import annotations

import inspect

import pytest

from rag_textbook.config import GraphSettings
from rag_textbook.stores.graph_store import GraphStore


class _RecordingSession:
    """Заглушка сессии с сигнатурой настоящего драйвера.

    Сигнатура скопирована намеренно: столкновение имён воспроизводится только
    при ``**kwparameters``, и заглушка с ``**kwargs`` его бы не поймала.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, parameters=None, **kwparameters):  # noqa: ANN001, ANN003
        self.calls.append((query, dict(parameters or {}), dict(kwparameters)))
        return self

    def data(self):  # noqa: ANN201
        return []

    def single(self):  # noqa: ANN201
        return None

    def consume(self):  # noqa: ANN201
        return None


def _store() -> GraphStore:
    return GraphStore(GraphSettings(_env_file=None))  # type: ignore[arg-type]


def test_parameters_go_into_the_dictionary_not_keywords() -> None:
    session = _RecordingSession()
    _store()._run(session, "RETURN $limit", limit=5)

    _, parameters, keywords = session.calls[0]
    assert parameters == {"limit": 5}
    assert keywords == {}, "параметры не должны уходить именованными аргументами"


@pytest.mark.parametrize("name", ["query", "parameters"])
def test_reserved_names_survive(name: str) -> None:
    """Имена, совпадающие с аргументами метода драйвера, должны проходить."""
    session = _RecordingSession()
    _store()._run(session, f"RETURN ${name}", **{name: "значение"})

    cypher, parameters, _ = session.calls[0]
    assert cypher == f"RETURN ${name}"
    assert parameters == {name: "значение"}


def test_reserved_name_as_keyword_would_break() -> None:
    """Показывает сам дефект: тот же вызов напрямую падает.

    Тест существует, чтобы обоснование обёртки не пришлось принимать на веру.
    """
    session = _RecordingSession()
    with pytest.raises(TypeError):
        session.run("RETURN $query", query="значение")


def test_no_query_uses_keyword_parameters() -> None:
    """Ни один запрос в хранилище не должен обходить обёртку.

    Проверка по исходному коду: подключить настоящий Neo4j в тестах нельзя,
    а именно обход обёртки и вернёт дефект.
    """
    source = inspect.getsource(GraphStore)
    # Внутри самой обёртки вызов session.run законен, вне её — нет.
    wrapper = inspect.getsource(GraphStore._run)
    outside = source.replace(wrapper, "")
    assert "session.run(" not in outside.replace("self._run(session,", ""), (
        "запрос обходит обёртку _run и может столкнуться с именами аргументов драйвера"
    )


# ------------------------------------------------------------ очистка графа

class _CountingSession(_RecordingSession):
    """Возвращает число удалённых узлов, как настоящий драйвер."""

    def single(self):  # noqa: ANN201
        return {"nodes": 7}


def test_clear_reports_how_much_it_deleted() -> None:
    """Молчаливая очистка неотличима от очистки, которая ничего не нашла.

    Различать их обязательно: граф учебника снимается ради замера на чужом
    корпусе, и «удалено 0» означает, что снимать было нечего, — то есть
    предыдущий шаг не отработал.
    """
    store = _store()
    session = _CountingSession()
    store._session = lambda: _as_context(session)  # type: ignore[method-assign]

    assert store.clear() == {"nodes": 7}
    cypher, _, _ = session.calls[0]
    assert "DETACH DELETE" in cypher


class _as_context:
    def __init__(self, session) -> None:  # noqa: ANN001
        self.session = session

    def __enter__(self):  # noqa: ANN204
        return self.session

    def __exit__(self, *args) -> bool:  # noqa: ANN002
        return False
