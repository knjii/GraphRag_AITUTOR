"""Общие правила якорей и ссылок учебника, без доступа к графу."""

from __future__ import annotations

import re
from typing import Any

# Номер формулы MinerU отдаёт разметкой \tag{2.32} внутри выражения —
# однозначный якорь: именно здесь формула определена, а не упомянута.
TAG = re.compile(r"\\tag\{\(?(\d+\.\d+[a-z]?)\)?\}")
# Ссылка на формулу в прозе: «подставив (2.32) в (2.47)».
EQ_REF = re.compile(r"\((\d+\.\d+[a-z]?)\)")

# Падежи перечислены явно: «[Пп]ример\w*» поймало бы «примерно 2.5 раза».
STATEMENT = (r"(?:[Оо]пределени(?:е|я|и|ю|ем)|[Тт]еорем(?:а|ы|е|у|ой)|"
             r"[Лл]емм(?:а|ы|е|у|ой)|[Сс]ледстви(?:е|я|и|ю|ем)|"
             r"[Пп]ример(?:а|е|у|ы|ов|ах)?|[Зз]амечани(?:е|я|и|ю|ем))")
# «Определение 2.10 (векторное подпространство).» — имя в скобках сразу
# после номера встречается только в месте определения, не в ссылке.
STATEMENT_ANCHOR = re.compile(STATEMENT + r"\s+(\d+\.\d+)\s*\(")
STATEMENT_REF = re.compile(STATEMENT + r"\s+(\d+\.\d+)")

SECTION_REF = re.compile(r"[Рр]азд(?:ел[аеу]?|\.)\s*(\d+(?:\.\d+)+)")
FIGURE_REF = re.compile(r"[Рр]ис(?:унок|унке|унка|\.)\s*(\d+\.\d+)")
# Подпись рисунка: номер, точка, прописная буква — в отличие от ссылки
# «на рис. 1.1 показано».
FIGURE_CAPTION = re.compile(r"[Рр]ис(?:унок|\.)\s*(\d+\.\d+)\.\s+[А-ЯЁ]")
# Номер раздела живёт в заголовках фрагмента: «1.1. ПОИСК ИНТУИТИВНО…».
SECTION_HEAD = re.compile(r"^\s*(\d+(?:\.\d+)+)\.")


def anchors_of(chunks: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """Где определён объект: вид → номер → фрагменты."""
    found: dict[str, dict[str, set[str]]] = {
        "equation": {}, "statement": {}, "section": {}, "figure": {},
    }
    sections: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        text = chunk["text"]
        for match in TAG.finditer(text):
            found["equation"].setdefault(match.group(1), set()).add(chunk["id"])
        for match in STATEMENT_ANCHOR.finditer(text):
            found["statement"].setdefault(match.group(1), set()).add(chunk["id"])
        # Раздел — это десятки фрагментов. Ссылка ведёт к его началу,
        # иначе одно упоминание «раздел 8.2» породило бы куст рёбер,
        # то есть ровно тот хаб, от которого мы лечили граф.
        for header in chunk.get("headers") or ():
            match = SECTION_HEAD.match(header)
            if not match:
                continue
            first = sections.get(match.group(1))
            if first is None or chunk["ordinal"] < first["ordinal"]:
                sections[match.group(1)] = chunk
        if chunk.get("has_figure"):
            for match in FIGURE_CAPTION.finditer(text):
                found["figure"].setdefault(match.group(1), set()).add(chunk["id"])
    for number, chunk in sections.items():
        found["section"][number] = {chunk["id"]}
    return found


def references_of(chunk: dict[str, Any]) -> list[tuple[str, str]]:
    """Ссылки фрагмента: (вид, номер). Разметка тегов вырезана."""
    text = TAG.sub(" ", chunk["text"])
    refs = [("equation", m.group(1)) for m in EQ_REF.finditer(text)]
    refs += [("statement", m.group(1)) for m in STATEMENT_REF.finditer(text)]
    refs += [("section", m.group(1)) for m in SECTION_REF.finditer(text)]
    refs += [("figure", m.group(1)) for m in FIGURE_REF.finditer(text)]
    return refs


