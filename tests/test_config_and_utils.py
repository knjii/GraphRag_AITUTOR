"""Тесты конфигурации и утилит.

Регрессии на конкретные дефекты прежней версии: булев флаг включался пустой
строкой, значения читались один раз при импорте модуля, пароль хранился
обычной строкой и мог утечь в трейсы.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_textbook.config import ChunkingSettings, EvalSettings, GraphSettings, Settings
from rag_textbook.utils.cache import ArtifactCache
from rag_textbook.utils.retry import is_retryable, retry_sync
from rag_textbook.utils.text import (
    canonicalize_entity,
    content_terms,
    is_meaningful,
    near_duplicate,
)


def test_empty_env_value_falls_back_to_default(monkeypatch) -> None:
    """Пустая переменная означает «не задано», а не «включено».

    Прежний ``_env_bool`` возвращал ``True`` для пустой строки, поэтому строка
    ``NEO4J_ENABLED=`` в ``.env`` незаметно включала графовый слой.
    """
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    # Поле со значением по умолчанию False не должно включиться пустой строкой.
    monkeypatch.setenv("GRAPH_FAIL_ON_ERROR", "")
    assert GraphSettings().fail_on_error is False

    # Поле со значением по умолчанию True не должно выключиться.
    monkeypatch.setenv("GRAPH_ENABLED", "")
    assert GraphSettings().enabled is True

    # Явно заданное значение по-прежнему уважается.
    monkeypatch.setenv("GRAPH_ENABLED", "0")
    assert GraphSettings().enabled is False


def test_comma_separated_fields_are_parsed_from_environment(monkeypatch) -> None:
    """Списки задаются через запятую, а не JSON.

    Регрессия: поля составных типов pydantic-settings пытался разобрать как
    JSON ещё до наших валидаторов. Строка ``RELATES`` валидным JSON не является,
    поэтому попытка задать параметр через окружение роняла приложение с
    ``SettingsError`` — до валидатора, который умеет делить по запятой, дело
    не доходило. Дефект не проявлялся, пока никто не задавал эти переменные.
    """
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    monkeypatch.setenv("GRAPH_EXPANSION_REL_TYPES", "RELATES")
    assert GraphSettings().expansion_rel_types == ("RELATES",)

    # Несколько значений и приведение к верхнему регистру.
    monkeypatch.setenv("GRAPH_EXPANSION_REL_TYPES", "relates, mentions")
    assert GraphSettings().expansion_rel_types == ("RELATES", "MENTIONS")

    monkeypatch.setenv("CHUNKER_ENRICH_TYPES", "image,chart,table")
    assert ChunkingSettings().enrich_types == ("image", "chart", "table")

    # Числовой список приводится к int.
    monkeypatch.setenv("EVAL_K_VALUES", "1,5,20")
    assert EvalSettings().k_values == (1, 5, 20)


def test_settings_ignore_ambient_env_file(tmp_path, monkeypatch) -> None:
    """Конфигурация читает тот файл, который ей указан, и никакой другой.

    Регрессия: путь к ``.env`` был вшит константой, поэтому тесты подхватывали
    рабочий файл, если он оказывался рядом. Набор проходил на машине без
    развёрнутого окружения и падал на сервере с ним — то есть не проверял
    ничего конкретного.
    """
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    # Файл с рабочими значениями кладём ровно туда, откуда конфигурация читала бы
    # его по умолчанию — в текущий каталог.
    (tmp_path / ".env").write_text("RETRIEVAL_TOP_K=42\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings()
    assert settings.retrieval.top_k != 42, (
        "конфигурация подхватила посторонний .env из текущего каталога; "
        "в тестах путь задаётся переменной RAG_ENV_FILE"
    )
    # Переменная окружения при этом продолжает действовать.
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")
    assert Settings().retrieval.top_k == 7


def test_settings_reread_environment_on_each_instantiation(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "3")
    assert Settings().retrieval.top_k == 3
    monkeypatch.setenv("RETRIEVAL_TOP_K", "11")
    assert Settings().retrieval.top_k == 11, (
        "значения должны читаться при создании объекта, а не при импорте модуля"
    )


def test_password_is_masked_in_repr_and_dump(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "topsecret")
    settings = GraphSettings()
    assert "topsecret" not in repr(settings)
    assert "topsecret" not in str(settings.model_dump())
    assert settings.password.get_secret_value() == "topsecret"


def test_invalid_values_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValidationError):
        ChunkingSettings(chunk_size=500, chunk_overlap=500)


def test_multihop_over_cooccurrence_is_forbidden() -> None:
    # Именно этот режим делал прежний граф бесполезным: обход *1..2 по 280k рёбер
    # co-occurrence доставал почти весь граф.
    with pytest.raises(ValidationError):
        GraphSettings(expansion_rel_types=("RELATES", "CO_OCCURS"), expansion_hops=2)


def test_graph_disabled_without_password(monkeypatch) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    settings = Settings()
    assert settings.graph.enabled is False, "без пароля граф должен отключаться, а не падать"


def test_cache_roundtrip(tmp_path) -> None:
    cache = ArtifactCache(tmp_path / "c.sqlite3", "test")
    cache.set("k1", {"value": 1})
    cache.set_many({"k2": [1, 2], "k3": "text"})

    assert cache.get("k1") == {"value": 1}
    assert cache.get_many(["k1", "k2", "missing"]) == {"k1": {"value": 1}, "k2": [1, 2]}
    assert cache.get("missing") is None
    assert cache.stats()["entries"] == 3
    cache.close()


def test_cache_survives_reopen(tmp_path) -> None:
    path = tmp_path / "c.sqlite3"
    first = ArtifactCache(path, "ns")
    first.set("key", "value")
    first.close()

    second = ArtifactCache(path, "ns")
    assert second.get("key") == "value", "кэш обязан переживать перезапуск процесса"
    second.close()


def test_disabled_cache_is_transparent(tmp_path) -> None:
    cache = ArtifactCache(tmp_path / "c.sqlite3", "ns", enabled=False)
    cache.set("key", "value")
    assert cache.get("key") is None


def test_lemmatization_merges_word_forms() -> None:
    # Без pymorphy3 канонизация деградирует до приведения к нижнему регистру:
    # пайплайн продолжает работать, но словоформы не схлопываются.
    pytest.importorskip("pymorphy3", reason="морфология русского языка недоступна")
    first = canonicalize_entity("сингулярного разложения")
    second = canonicalize_entity("сингулярное разложение")
    assert first == second, "падежные формы должны схлопываться в одну каноническую"


def test_symbolic_entity_survives_canonicalization() -> None:
    assert canonicalize_entity("C++") != ""
    assert canonicalize_entity("SVD") == "svd"


def test_stopwords_are_filtered() -> None:
    terms = content_terms("это и который для матрица разложение")
    assert "матрица" in terms
    assert "который" not in terms
    assert not is_meaningful("для")


def test_near_duplicate_detects_overlapping_chunks() -> None:
    # Имитируем перекрытие чанков: общий текст плюс небольшой хвост.
    body = (
        "Сингулярное разложение матрицы применяется понижения размерности данных "
        "собственные значения ковариационная матрица дисперсия направление проекция"
    )
    left = body
    right = body + " дополнительный хвост"
    assert near_duplicate(left, right, threshold=0.85)
    assert not near_duplicate(left, "Совершенно другая тема про определённые интегралы", 0.85)


def test_retry_recovers_from_transient_error() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("HTTP 503 service unavailable")
        return "ok"

    assert retry_sync(flaky, attempts=5, base_delay=0.001) == "ok"
    assert attempts["count"] == 3


def test_retry_does_not_repeat_permanent_error() -> None:
    attempts = {"count": 0}

    def broken() -> str:
        attempts["count"] += 1
        raise ValueError("некорректный запрос")

    with pytest.raises(ValueError):
        retry_sync(broken, attempts=5, base_delay=0.001)
    assert attempts["count"] == 1, "неповторяемая ошибка не должна ретраиться"


def test_retryable_classification() -> None:
    assert is_retryable(RuntimeError("status code: 503"))
    assert is_retryable(RuntimeError("Read timed out"))
    assert not is_retryable(ValueError("bad schema"))
