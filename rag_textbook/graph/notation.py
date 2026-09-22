"""Обозначения по правилам (гипотеза К3, docs/HYPOTHESES.md).

Задача 025 нашла, что извлечение идентификаторов без модели даёт
precision 83.7% и recall 94.8% (Semantification, §4.1.1). Поэтому
сначала правила, модель (поле ``notation`` в извлечении v4) — запасной
путь, если правила найдут меньше 80% обозначений на ручной выборке.

Шаблоны:

* «где $x$ — вектор признаков, $y$ — метка» — цепочка после «где»;
* «… обозначим через $X$» и «… будем обозначать $X$» — смысл стоит
  перед глаголом;
* «$X$ обозначает …».

Символ сам по себе не узел: $x$ в разных главах значит разное. Узел —
пара «символ := смысл», а его область — раздел, где он введён
(``scope_users``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Символ — формула в $…$ или короткое латинское либо греческое имя
# без разметки («Γ(·) — гамма-функция», «θ — тангенс угла»). Кириллица
# в простое имя не входит: иначе «обозначает приближение» дало бы
# символ «приближение».
_SYMBOL = (
    r"(?:\$(?P<sym>[^$]{1,40})\$"
    r"|(?P<plain>[A-Za-zΑ-Ωα-ω][A-Za-z0-9_'Α-Ωα-ω]{0,3}(?:\([^()\s]{1,6}\))?)(?![\w$]))"
)
_DASH = r"(?:\s*[—–]\s*|\s+это\s+)"
_MEANING_AFTER = r"(?P<meaning>[^,.;:$()]{3,80})"
_MEANING_BEFORE = r"(?P<meaning>[^,.;:$()]{3,80}?)"

# «где $x$ — …» открывает цепочку; следующие звенья «, $y$ — …».
_WHERE = re.compile(r"\bгде\s+" + _SYMBOL + _DASH + _MEANING_AFTER)
_CHAIN = re.compile(r"^\s*(?:,\s*(?:а\s+)?|\s+и\s+|\s+а\s+)" + _SYMBOL + _DASH + _MEANING_AFTER)
_DENOTE_BEFORE = re.compile(
    _MEANING_BEFORE
    + r"\s+(?:мы\s+)?(?:обозначим|будем\s+обозначать|обозначается|обозначают|обознач[её]н[аоы]?)"
    + r"\s+(?:(?:его|её|ее|их)\s+)?(?:(?:через|как)\s+)?"
    + _SYMBOL
)
_DENOTES = re.compile(_SYMBOL + r"\s+обозначает\s+" + _MEANING_AFTER)

# Смысл перед глаголом берётся с конца: из «Множество всех матриц размера
# m×n обозначим» нужен хвост, а не начало предложения.
_MAX_MEANING_WORDS = 6
_STOP_WORDS = frozenset(
    {
        "и", "а", "также", "далее", "здесь", "теперь", "это", "обычно", "часто",
        "мы", "как", "же", "будем", "его", "её", "их", "который", "которая",
        "которое", "которые", "которую", "символом", "буквой",
    }
)


@dataclass(frozen=True)
class Notation:
    symbol: str
    meaning: str
    start: int
    pattern: str


def _clean_meaning(raw: str, *, tail: bool) -> str:
    words = [word for word in raw.strip().split() if word]
    words = words[-_MAX_MEANING_WORDS:] if tail else words[:_MAX_MEANING_WORDS]
    while words and words[0].lower() in _STOP_WORDS:
        words.pop(0)
    while words and words[-1].lower() in _STOP_WORDS:
        words.pop()
    return " ".join(words).strip(" —–-")


def _symbol(match: re.Match[str]) -> str:
    raw = match["sym"] if match["sym"] is not None else match["plain"]
    return " ".join((raw or "").split())


def find_notations(text: str) -> list[Notation]:
    """Обозначения, введённые во фрагменте, в порядке появления."""
    found: list[Notation] = []

    def add(symbol: str, meaning: str, start: int, pattern: str) -> None:
        if symbol and len(meaning) >= 3 and not meaning.split()[0].isdigit():
            found.append(Notation(symbol, meaning, start, pattern))

    for match in _WHERE.finditer(text):
        add(_symbol(match), _clean_meaning(match["meaning"], tail=False), match.start(), "где")
        rest = text[match.end():]
        offset = match.end()
        while True:
            link = _CHAIN.match(rest)
            if not link:
                break
            add(_symbol(link), _clean_meaning(link["meaning"], tail=False), offset + link.start(), "где")
            offset += link.end()
            rest = rest[link.end():]
    for match in _DENOTE_BEFORE.finditer(text):
        add(_symbol(match), _clean_meaning(match["meaning"], tail=True), match.end("meaning"), "обозначим")
    for match in _DENOTES.finditer(text):
        add(_symbol(match), _clean_meaning(match["meaning"], tail=False), match.start(), "обозначает")

    unique: dict[tuple[str, str], Notation] = {}
    for item in sorted(found, key=lambda note: note.start):
        unique.setdefault((item.symbol, item.meaning.lower()), item)
    return list(unique.values())


def _tokens(latex: str) -> list[str]:
    return re.findall(r"\\[A-Za-z]+|[^\s{}]", latex)


def contains_symbol(text: str, symbol: str) -> bool:
    """Символ встречается внутри формулы фрагмента как целая лексема.

    Сравнение по лексемам LaTeX, а не по подстроке: иначе «a» нашлась бы
    в каждом ``\\gamma``, и область обозначения стала бы всей книгой.
    """
    wanted = _tokens(symbol)
    if not wanted:
        return False
    size = len(wanted)
    for formula in re.findall(r"\$\$?([^$]+)\$\$?", text):
        tokens = _tokens(formula)
        if any(tokens[index : index + size] == wanted for index in range(len(tokens) - size + 1)):
            return True
    return False
