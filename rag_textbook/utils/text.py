"""Работа с русским текстом: токенизация, лемматизация, канонизация терминов.

Зачем отдельный модуль: прежний граф хранил сущности как сырые токены, поэтому
запрос в косвенном падеже не совпадал с узлом графа, а fallback через
``CONTAINS`` тащил случайные подстроки. Лемматизация чинит обе проблемы разом
и используется одинаково при построении графа и при поиске.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence

from rag_textbook.utils.stopwords import STOPWORDS

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_\-]{1,63}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n{2,}")
_LATEX_INLINE_RE = re.compile(r"\$\$?[^$]+\$\$?")


# Термины предметной области, которые нельзя лемматизировать до неузнаваемости
# и нельзя выбрасывать как короткие.
DOMAIN_KEEP: frozenset[str] = frozenset(
    {"svd", "pca", "ols", "mse", "mle", "map", "knn", "svm", "relu", "gd", "sgd", "em", "kl"}
)


@functools.lru_cache(maxsize=1)
def _morph():  # pragma: no cover - зависит от наличия pymorphy3
    try:
        import pymorphy3

        return pymorphy3.MorphAnalyzer()
    except Exception:
        return None


@functools.lru_cache(maxsize=200_000)
def lemmatize_token(token: str) -> str:
    """Приводит токен к нормальной форме.

    Без ``pymorphy3`` возвращает исходный токен: деградация мягкая,
    качество ниже, но пайплайн не падает.
    """

    lowered = token.lower().replace("ё", "е")
    if lowered in DOMAIN_KEEP:
        return lowered
    if not any("а" <= ch <= "я" for ch in lowered):
        return lowered
    morph = _morph()
    if morph is None:
        return lowered
    try:
        parsed = morph.parse(lowered)
    except Exception:
        return lowered
    if not parsed:
        return lowered
    return str(parsed[0].normal_form).replace("ё", "е")


def tokenize(text: str) -> list[str]:
    """Токены без формул: LaTeX обрабатывается отдельно и не должен попадать в лексику."""
    cleaned = _LATEX_INLINE_RE.sub(" ", text or "")
    return [match.group(0) for match in _TOKEN_RE.finditer(cleaned)]


def is_meaningful(token: str, min_length: int = 3) -> bool:
    lowered = token.lower()
    if lowered in DOMAIN_KEEP:
        return True
    if len(lowered) < max(1, min_length):
        return False
    if lowered in STOPWORDS:
        return False
    if lowered.isdigit():
        return False
    return any(ch.isalpha() for ch in lowered)


def content_terms(
    text: str,
    *,
    min_length: int = 3,
    lemmatize: bool = True,
    limit: int | None = None,
) -> list[str]:
    """Содержательные термины текста в порядке первого появления."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in tokenize(text):
        token = lemmatize_token(raw) if lemmatize else raw.lower().replace("ё", "е")
        if not is_meaningful(token, min_length):
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if limit is not None and len(out) >= limit:
            break
    return out


def canonicalize_entity(name: str, *, lemmatize: bool = True, max_words: int = 4) -> str:
    """Каноническая форма имени сущности.

    Многословные термины («сингулярное разложение матрицы») лемматизируются
    пословно, поэтому «сингулярного разложения матриц» схлопнется в ту же форму.
    """

    tokens = tokenize(name)
    if not tokens:
        # Формулы и символьные обозначения тоже бывают сущностями: не теряем их.
        stripped = (name or "").strip().strip(".,;:!?()[]{}\"'`")
        return stripped.lower().replace("ё", "е")[:96]

    parts: list[str] = []
    for raw in tokens[: max_words * 2]:
        token = lemmatize_token(raw) if lemmatize else raw.lower().replace("ё", "е")
        if not is_meaningful(token, 2):
            continue
        parts.append(token)
        if len(parts) >= max_words:
            break
    if not parts:
        return ""
    return " ".join(parts)[:96]


def split_sentences(text: str, *, max_sentences: int = 512) -> list[str]:
    out: list[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(text or ""):
        sentence = " ".join(str(raw).split()).strip()
        if sentence:
            out.append(sentence)
        if len(out) >= max_sentences:
            break
    return out


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / float(len(a | b))


def near_duplicate(text_a: str, text_b: str, threshold: float = 0.92) -> bool:
    """Определяет почти-дубликаты.

    Нужно из-за перекрытия чанков: прежний ключ дедупликации сравнивал тексты
    точно, поэтому соседние чанки с общим хвостом считались разными документами.
    """

    if not text_a or not text_b:
        return False
    if text_a == text_b:
        return True
    terms_a = content_terms(text_a, lemmatize=False, limit=200)
    terms_b = content_terms(text_b, lemmatize=False, limit=200)
    return jaccard(terms_a, terms_b) >= threshold


def extract_latex_fragments(text: str, limit: int = 32) -> list[str]:
    return [match.group(0) for match in _LATEX_INLINE_RE.finditer(text or "")][:limit]


def truncate(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 1)].rstrip() + "…"


def format_pages(pages: Sequence[int]) -> str:
    if not pages:
        return ""
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}–{pages[-1]}"
