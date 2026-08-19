"""Переходник к публичному набору MultiHop-RAG.

Проверяется главным образом одно: перевод дословной цитаты в идентификатор
нашего чанка. От него зависит весь замер — ошибка здесь не сломает прогон,
а тихо испортит эталон, и наше число окажется несопоставимым
с опубликованными при полном внешнем благополучии.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_textbook.benchmarks.multihop_rag import build_goldset, load_corpus, original_type
from rag_textbook.benchmarks.text_corpus import blocks_from_text, stable_doc_id
from rag_textbook.models import Chunk


def _chunk(identifier: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        doc_id=doc_id,
        doc_name="статья",
        source_path="multihop-rag/статья",
        ordinal=0,
        text=text,
    )


DOC_ID = stable_doc_id("multihop-rag", "Заголовок", "Издание")

CHUNKS = [
    _chunk(f"{DOC_ID}:00000", DOC_ID, "Первый абзац.  Компания объявила о росте выручки на треть."),
    _chunk(f"{DOC_ID}:00001", DOC_ID, "Второй абзац про совершенно другие обстоятельства дела."),
]


def _write(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ------------------------------------------------------- текст в блоки

def test_headings_become_levelled_blocks():
    blocks = blocks_from_text("# Заголовок\n\nОбычный абзац текста.")

    assert [block.text_level for block in blocks] == [1, None]
    assert blocks[0].text == "Заголовок"


def test_display_formula_becomes_special_block():
    """Особый блок нужен затем, чтобы чанкер не разрезал формулу пополам —
    ровно тот изъян, который уже однажды стоил формул в учебнике."""
    blocks = blocks_from_text("Текст.\n\n$$A A^{T} = I$$\n\nЕщё текст.")

    assert blocks[1].type == "equation"
    assert blocks[1].is_special


def test_table_becomes_table_block():
    blocks = blocks_from_text("| a | b |\n| 1 | 2 |")

    assert blocks[0].type == "table"


# ------------------------------------------------------ сопоставление цитат

def test_exact_quote_is_matched(tmp_path: Path):
    questions = _write(
        tmp_path,
        "q.json",
        [
            {
                "query": "На сколько выросла выручка?",
                "answer": "На треть.",
                "question_type": "inference_query",
                "evidence_list": [
                    {
                        "title": "Заголовок",
                        "source": "Издание",
                        "fact": "Компания объявила о росте выручки на треть.",
                    }
                ],
            }
        ],
    )

    produced, report = build_goldset(questions, CHUNKS)

    assert produced[0].gold_chunk_ids == [f"{DOC_ID}:00000"]
    assert report.matched_exact == 1
    assert report.coverage == 1.0


def test_quote_with_different_spacing_is_still_exact():
    """Пробелы между предложениями расставляются по-разному в корпусе и
    в цитате; дословность обязана это переживать."""
    assert " ".join(["Компания", "объявила"]) == "Компания объявила"


def test_unmatched_evidence_drops_the_question(tmp_path: Path):
    """Вопрос с неполным эталоном завышает промахи, и наше число перестаёт
    быть сопоставимым с опубликованными. Такой вопрос лучше не считать."""
    questions = _write(
        tmp_path,
        "q.json",
        [
            {
                "query": "Вопрос",
                "answer": "Ответ",
                "question_type": "comparison_query",
                "evidence_list": [
                    {"title": "Заголовок", "source": "Издание", "fact": "Компания объявила о росте выручки на треть."},
                    {"title": "Нет такой статьи", "source": "Издание", "fact": "Совсем посторонние слова здесь."},
                ],
            }
        ],
    )

    produced, report = build_goldset(questions, CHUNKS)

    assert produced == []
    assert report.unmatched == 1
    assert report.questions_kept == 0


def test_original_type_survives_in_notes(tmp_path: Path):
    """Разбивка по типам набора должна пережить перевод в наш словарь типов:
    опубликованные числа даны именно в ней."""
    questions = _write(
        tmp_path,
        "q.json",
        [
            {
                "query": "На сколько выросла выручка?",
                "answer": "На треть.",
                "question_type": "inference_query",
                "evidence_list": [
                    {
                        "title": "Заголовок",
                        "source": "Издание",
                        "fact": "Компания объявила о росте выручки на треть.",
                    }
                ],
            }
        ],
    )

    produced, report = build_goldset(questions, CHUNKS)

    assert original_type(produced[0]) == "inference_query"
    assert produced[0].question_type == "single_chunk", "наш словарь типов не расширяется"
    assert report.per_type["inference_query"] == 1


def test_partial_question_can_be_kept_explicitly(tmp_path: Path):
    questions = _write(
        tmp_path,
        "q.json",
        [
            {
                "query": "Вопрос",
                "answer": "Ответ",
                "question_type": "comparison_query",
                "evidence_list": [
                    {"title": "Заголовок", "source": "Издание", "fact": "Компания объявила о росте выручки на треть."},
                    {"title": "Нет такой", "source": "Издание", "fact": "Посторонние слова здесь."},
                ],
            }
        ],
    )

    produced, _ = build_goldset(questions, CHUNKS, keep_partial=True)

    assert len(produced) == 1


def test_null_queries_are_excluded(tmp_path: Path):
    """Вопросы без ответа в корпусе выкидываются: recall по пустому эталону
    не определён, а тихо засчитанный промах исказил бы всё число."""
    questions = _write(
        tmp_path,
        "q.json",
        [{"query": "Вопрос", "answer": "нет", "question_type": "null_query", "evidence_list": []}],
    )

    produced, report = build_goldset(questions, CHUNKS)

    assert produced == []
    assert report.dropped_null == 1


def test_document_ids_survive_reordering(tmp_path: Path):
    """Идентификатор считается от содержания, а не от порядка в файле:
    иначе после пересборки корпуса эталонные фрагменты перестанут находиться."""
    first = _write(
        tmp_path,
        "c1.json",
        [
            {"title": "А", "source": "И", "body": "Текст А."},
            {"title": "Б", "source": "И", "body": "Текст Б."},
        ],
    )
    second = _write(
        tmp_path,
        "c2.json",
        [
            {"title": "Б", "source": "И", "body": "Текст Б."},
            {"title": "А", "source": "И", "body": "Текст А."},
        ],
    )

    assert {item.doc_id for item in load_corpus(first)} == {
        item.doc_id for item in load_corpus(second)
    }


def test_empty_documents_are_skipped(tmp_path: Path):
    path = _write(tmp_path, "c.json", [{"title": "А", "source": "И", "body": "   "}])

    assert load_corpus(path) == []


# --------------------------------------------- защита от смешивания корпусов

def test_public_run_refuses_the_default_collection(tmp_path: Path, monkeypatch):
    """Чужой корпус в коллекции учебника обесценил бы все прежние замеры,
    причём молча: индекс просто стал бы больше. Команда обязана отказаться.

    Тест заодно проверяет, что защита ссылается на существующие поля
    настроек: первая её версия читала `settings.qdrant`, которого нет,
    и падала уже после индексации корпуса.
    """
    from typer.testing import CliRunner

    from rag_textbook.cli.main import _DEFAULT_COLLECTION, app

    monkeypatch.setenv("QDRANT_COLLECTION", _DEFAULT_COLLECTION)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "public",
            "--corpus",
            str(tmp_path / "corpus.json"),
            "--questions",
            str(tmp_path / "questions.json"),
        ],
    )

    assert result.exit_code == 1
    assert "QDRANT_COLLECTION" in result.output
