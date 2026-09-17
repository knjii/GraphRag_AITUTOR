"""Формульная часть награды: дошли ли формулы источника до ответа.

Версия 1 (``evaluation/answers.py``: ``latex_overlap``) сравнивает строки
после удаления пробелов, запятых и точек и засчитывает формулу, только если
её строка целиком входит в ответ. Для замера этого хватало, для награды —
нет: политика, которая пишет ``\\dfrac`` вместо ``\\frac`` или
``\\left(`` вместо ``(``, получала бы ноль за верную формулу, и обучение
выдавливало бы из неё не математику, а привычки оформления учебника.

Версия 2 сравнивает **канонические последовательности токенов**:

* снимается оформление, не меняющее смысла: ``\\left``/``\\right``,
  отступы ``\\,`` ``\\;`` ``\\quad``, ``\\displaystyle``, номера ``\\tag``;
* синонимы сводятся к одному имени: ``\\dfrac``/``\\tfrac`` → ``\\frac``,
  ``\\le`` → ``\\leq``, ``\\bmatrix`` → скобки, ``\\min`` → ``m i n``
  (так его записывает разбор), транспонирование ``^T``/``^\\top``/
  ``^{\\mathrm{T}}`` → одно;
* жирный шрифт снимается совсем: разбор и модели ставят его
  непоследовательно;
* группирующие фигурные скобки снимаются (кроме индексов и степеней).

Пары записей из реальных ответов — ``tests/test_rewards_canonical_real.py``.

**Чего версия 2 не делает.** Не проверяет алгебраическую эквивалентность:
``ab`` и ``ba``, ``x+y`` и ``y+x`` для неё разные формулы. Для переноса
формулы из учебника в ответ это правильно — задача политики донести формулу,
а не переписать её, — но это ограничение надо помнить, читая числа.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_EXTRA_DOLLARS_RE = re.compile(r"\${3,}")
_CYRILLIC_WORD_RE = re.compile(r"[а-яёА-ЯЁ]{3,}")

_TOKEN_RE = re.compile(
    r"\\[A-Za-z]+\*?"      # команда
    r"|\\."                # экранированный символ: \{ \, \;
    r"|[A-Za-z]"           # буква — отдельный токен: ab и a b одно и то же
    r"|\d+(?:\.\d+)?"      # число
    r"|\S"                 # всё остальное по символу
)

# Оформление, которое не меняет формулу.
_DROP = {
    r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg",
    r"\bigl", r"\bigr", r"\Bigl", r"\Bigr", r"\biggl", r"\biggr",
    r"\displaystyle", r"\textstyle", r"\scriptstyle",
    r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad",
    r"\nonumber", r"\notag", r"\limits", r"\nolimits",
}

# Обёртки, у которых важен только аргумент.
_UNWRAP = {
    r"\mathrm", r"\operatorname", r"\operatorname*", r"\text", r"\textrm", r"\mathit",
    # Жирный шрифт: разбор и модели ставят его непоследовательно
    # (``\pmb{A}`` в учебнике, ``A`` в ответе — та же матрица).
    r"\mathbf", r"\bf",
}

# Матричные окружения сводятся к скобкам вокруг содержимого.
_MATRIX_BRACKETS = {
    "bmatrix": ("[", "]"), "pmatrix": ("(", ")"),
    "vmatrix": ("|", "|"), "Vmatrix": (r"\|", r"\|"),
}

# Именованные функции раскладываются на буквы: разбор учебника даёт
# ``\operatorname* { m i n }``, модель пишет ``\min`` — это одна формула.
_NAMED_FUNCTIONS = {
    r"\min", r"\max", r"\log", r"\ln", r"\exp", r"\sin", r"\cos", r"\tan",
    r"\det", r"\dim", r"\ker", r"\arg", r"\lim", r"\sup", r"\inf", r"\tr",
    r"\rank", r"\sgn", r"\argmin", r"\argmax", r"\Pr", r"\deg", r"\gcd",
}

_SYNONYMS = {
    r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\cfrac": r"\frac",
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq",
    r"\leqslant": r"\leq", r"\geqslant": r"\geq",
    r"\to": r"\rightarrow", r"\gets": r"\leftarrow",
    r"\lbrace": r"\{", r"\rbrace": r"\}",
    r"\lvert": "|", r"\rvert": "|", r"\vert": "|",
    r"\lVert": r"\|", r"\rVert": r"\|", r"\Vert": r"\|",
    r"\bm": r"\mathbf", r"\boldsymbol": r"\mathbf", r"\pmb": r"\mathbf",
    r"\intercal": r"\top", r"\transpose": r"\top",
    r"\varepsilon": r"\epsilon", r"\varphi": r"\phi",
    r"\dots": r"\ldots", r"\cdots": r"\ldots",
    r"\bigcdot": r"\cdot",
    r"\colon": ":",
    r"\mid": "|",
}

# Формулы короче этого числа канонических токенов не оцениваются:
# «$x$» или «$n=1$» совпадают случайно и награждали бы за шум.
MIN_TOKENS = 5


def extract_math(text: str, limit: int | None = 64) -> list[str]:
    """Формулы из текста во всех видах разметки, без делимитеров.

    Фрагмент с двумя и больше русскими словами отбрасывается: это не формула,
    а проза, захваченная между одиночными долларами (см. ``utils/text.py``).
    ``limit`` защищает от ответа-петли; для контекста и эталона его надо
    снимать: в 6 эпизодах из 388 контекст несёт больше 64 формул, и
    законные формулы из его хвоста считались бы выдуманными.
    """
    # Фрагменты учебника несут выносные формулы как «$$$$ … $$$$»: разметка
    # разбора обёрнута ещё раз. Без сведения к «$$» каждая десятая эталонная
    # формула (185 из 1812) извлекалась с лишним долларом и не совпадала
    # с той же формулой в ответе.
    found: list[str] = []
    for body in _math_spans(text)[0]:
        found.append(body)
        if limit is not None and len(found) >= limit:
            break
    return found


def strip_math(text: str) -> str:
    """Текст без формул во всех видах разметки.

    ``utils.text.strip_latex`` знает только доллары, а модели пишут и
    ``\\( \\)``, и ``\\[ \\]`` — такие ответы мера латиницы считала бы
    английскими.
    """
    return _math_spans(text)[1]


_DISPLAY_RE = re.compile(r"\$\$")
_OTHER_MATH_RE = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)|\$([^$\n]+?)\$", re.DOTALL)
_TEXT_ARGUMENT_RE = re.compile(r"\\(?:text|mathrm|operatorname|textrm)\s*\{[^{}]*\}")


def _is_prose(body: str) -> bool:
    # Русские слова внутри \text{…} законны в формуле и не считаются.
    return len(_CYRILLIC_WORD_RE.findall(_TEXT_ARGUMENT_RE.sub(" ", body))) >= 2


def _math_spans(text: str) -> tuple[list[str], str]:
    """Формулы по порядку и текст без них.

    Выносные «$$» разбираются парами с проверкой: фрагмент учебника часто
    начинается с середины формулы (чанкер режет по границе окна), и первый
    «$$» в нём — закрывающий. Прямое сопоставление сдвигало все пары,
    между «открывающим» и «закрывающим» оказывалась проза, и формулы
    выбрасывались. Замер 2026-09-17: так терялись 789 из 1571
    нумерованной формулы учебника. Если между парой «$$» проза, первый
    считается закрывающим, и разбор сдвигается на один.
    """
    text = _EXTRA_DOLLARS_RE.sub("$$", text or "")
    positions = [match.start() for match in _DISPLAY_RE.finditer(text)]
    # Чётность выбирается по всему фрагменту: между соседними формулами
    # «$$ A $$ $$ B $$» или «$$ A $$ (2.6) $$ B $$» сдвинутая пара прозы
    # не содержит, и локальная проверка её не замечает.
    spans = min(
        (_pair_display(text, positions, first) for first in (0, 1)),
        key=lambda item: item[1],
    )[0]

    found: list[tuple[int, str]] = []
    rest: list[str] = []
    cursor = 0
    for span_start, span_end, body in spans:
        # Непарный «$$» (обрезанная формула) не должен сбивать поиск «$…$».
        segment = text[cursor:span_start].replace("$$", "  ")
        found.extend(_inline_math(segment, cursor))
        rest.append(_OTHER_MATH_RE.sub(" ", segment))
        rest.append(" ")
        found.append((span_start, body))
        cursor = span_end
    tail = text[cursor:].replace("$$", "  ")
    found.extend(_inline_math(tail, cursor))
    rest.append(_OTHER_MATH_RE.sub(" ", tail))
    found.sort(key=lambda item: item[0])
    return [body for _, body in found], "".join(rest)


_NOT_A_FORMULA_RE = re.compile(r"^[\s().,;:\d]*$")


def _pair_display(
    text: str, positions: Sequence[int], first: int
) -> tuple[list[tuple[int, int, str]], int]:
    """Пары «$$», начиная с ``first``; второе — число подозрительных пар."""
    spans: list[tuple[int, int, str]] = []
    suspicious = first
    index = first
    while index + 1 < len(positions):
        start, end = positions[index] + 2, positions[index + 1]
        body = text[start:end]
        if _is_prose(body) or _NOT_A_FORMULA_RE.match(body):
            suspicious += 1
            index += 1
            continue
        spans.append((positions[index], end + 2, body))
        index += 2
    return spans, suspicious


def _inline_math(segment: str, offset: int) -> list[tuple[int, str]]:
    result = []
    for match in _OTHER_MATH_RE.finditer(segment):
        body = next(group for group in match.groups() if group is not None)
        if not _is_prose(body):
            result.append((offset + match.start(), body))
    return result


def canonical_tokens(formula: str) -> tuple[str, ...]:
    """Каноническая последовательность токенов формулы."""
    text = re.sub(r"\\(?:tag|label)\*?\s*\{[^}]*\}", " ", formula or "")
    # Спецификация столбцов («{ l l l }») — вёрстка, а не формула: разбор
    # и модель расходятся в числе букв.
    text = re.sub(r"\\begin\s*\{(?:array|tabular)\}\s*\{[^{}]*\}", " ", text)
    text = re.sub(
        r"\\(begin|end)\s*\{([A-Za-z]+)\}",
        lambda m: " {} ".format(
            _MATRIX_BRACKETS[m.group(2)][m.group(1) == "end"] if m.group(2) in _MATRIX_BRACKETS else ""
        ),
        text,
    )
    text = re.sub(r"\\(?:begin|end)\s*\{[A-Za-z*]+\}", " ", text)
    raw = _TOKEN_RE.findall(text)

    tokens: list[str] = []
    index = 0
    while index < len(raw):
        token = _SYNONYMS.get(raw[index], raw[index])
        index += 1
        # Многоточие разбор пишет точками (которые снимаются), модель —
        # командой; «$» остаётся от сбитой разметки.
        if token in _DROP or token in {",", ".", ";", "&", "$", r"\ldots"} or token == r"\\":
            continue
        if token in _UNWRAP:
            # \mathrm{d}x → d x; скобки аргумента снимаются ниже как одиночные.
            continue
        if token in _NAMED_FUNCTIONS:
            tokens.extend(token[1:])
            continue
        tokens.append(token)

    # Висящий знак в конце — перенос строки при разборе, не часть формулы.
    while tokens and tokens[-1] in {"-", "+", "="}:
        tokens.pop()
    # Транспонирование — после снятия скобок: ``^{\mathrm{T}}`` превращается
    # в ``^ T`` только тогда.
    return tuple(_normalize_transpose(list(_strip_single_braces(_drop_grouping_braces(tokens)))))


def _drop_grouping_braces(tokens: list[str]) -> list[str]:
    """Оставляет фигурные скобки только у индексов и степеней.

    Разбор PDF и модели расставляют группирующие скобки по-разному:
    ``{ \\mathbb R } ^ n`` и ``\\mathbb{R}^n`` — одна формула. Скобки после
    ``^`` и ``_`` несут структуру (``x^{ab}`` не ``x^a b``), остальные —
    почти никогда. Цена упрощения: ``\\frac{ab}{c}`` и ``\\frac{a}{bc}``
    сливаются; для переноса формулы из учебника это редкий случай.
    """
    keep: set[int] = set()
    stack: list[tuple[int, bool]] = []
    for index, token in enumerate(tokens):
        if token == "{":
            significant_group = index > 0 and tokens[index - 1] in {"^", "_"}
            stack.append((index, significant_group))
        elif token == "}" and stack:
            start, significant_group = stack.pop()
            if significant_group:
                keep.update((start, index))
    return [
        token for index, token in enumerate(tokens)
        if token not in {"{", "}"} or index in keep
    ]


def _normalize_transpose(tokens: list[str]) -> list[str]:
    """``^T``, ``^{T}``, ``^\\top`` и ``^{\\top}`` — одно и то же."""
    result: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == "^":
            ahead = tokens[index + 1 : index + 4]
            if ahead[:1] in (["T"], [r"\top"]):
                result.extend(["^", r"\top"])
                index += 2
                continue
            if len(ahead) == 3 and ahead[0] == "{" and ahead[1] in ("T", r"\top") and ahead[2] == "}":
                result.extend(["^", r"\top"])
                index += 4
                continue
        result.append(tokens[index])
        index += 1
    return result


def _strip_single_braces(tokens: list[str]) -> tuple[str, ...]:
    """``{x}`` → ``x``: скобки вокруг одного токена смысла не несут."""
    changed = True
    while changed:
        changed = False
        result: list[str] = []
        index = 0
        while index < len(tokens):
            if (
                tokens[index] == "{"
                and index + 2 < len(tokens)
                and tokens[index + 2] == "}"
                and tokens[index + 1] not in {"{", "}"}
            ):
                result.append(tokens[index + 1])
                index += 3
                changed = True
                continue
            result.append(tokens[index])
            index += 1
        tokens = result
    return tuple(tokens)


def _contains(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """Непрерывное вхождение: формула внутри более длинного выражения."""
    size = len(needle)
    if size == 0 or size > len(haystack):
        return False
    first = needle[0]
    return any(
        haystack[i] == first and tuple(haystack[i : i + size]) == tuple(needle)
        for i in range(len(haystack) - size + 1)
    )


def similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Доля общей подпоследовательности: 2·LCS / (|a| + |b|)."""
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for a in left:
        current = [0]
        for j, b in enumerate(right, start=1):
            current.append(previous[j - 1] + 1 if a == b else max(previous[j], current[j - 1]))
        previous = current
    return 2 * previous[-1] / (len(left) + len(right))


def significant(formulas: Iterable[str]) -> list[tuple[str, ...]]:
    """Канонические формы, достаточно длинные для оценки, без повторов."""
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for formula in formulas:
        tokens = canonical_tokens(formula)
        if len(tokens) >= MIN_TOKENS and tokens not in seen:
            seen.add(tokens)
            result.append(tokens)
    return result


@dataclass(frozen=True)
class FormulaScore:
    """Итог сравнения формул ответа с источником.

    ``expected`` — формулы эталонных фрагментов;
    ``carried`` — сколько из них есть в ответе целиком (в том числе внутри
    более длинного выражения);
    ``partial`` — сумма лучших сходств по эталонным формулам, от 0 до ``expected``:
    частичный перенос для плотной награды;
    ``answer_formulas`` — формулы ответа;
    ``relevant`` — формулы ответа, совпадающие с эталонной;
    ``foreign`` — формулы ответа, которых нет в контексте вообще: выдуманные
    или принесённые из памяти модели.
    """

    expected: int
    carried: int
    partial: float
    answer_formulas: int
    relevant: int
    foreign: int

    @property
    def recall(self) -> float | None:
        return self.carried / self.expected if self.expected else None

    @property
    def partial_recall(self) -> float | None:
        return self.partial / self.expected if self.expected else None

    @property
    def foreign_share(self) -> float:
        return self.foreign / self.answer_formulas if self.answer_formulas else 0.0

    @property
    def precision(self) -> float | None:
        return self.relevant / self.answer_formulas if self.answer_formulas else None


# Порог «та же формула с опечаткой»: ниже — уже другая формула.
NEAR_MATCH = 0.8

# Доля символов формулы, которых нет в формулах контекста, выше которой
# формула считается принесённой извне.
FOREIGN_SYMBOLS = 0.3

# Структура, которая есть в любой формуле и ничего не говорит о её источнике.
_STRUCTURAL = {
    "{", "}", "^", "_", "(", ")", "[", "]", "=", "+", "-", "|", ":", "/", "<", ">",
    r"\ldots", r"\cdot", r"\vdots", r"\ddots",
}


def _unknown_share(tokens: Sequence[str], vocabulary: set[str]) -> float:
    meaningful = [
        token for token in tokens
        if token not in _STRUCTURAL and not token.replace(".", "").isdigit()
    ]
    if not meaningful:
        return 0.0
    return sum(1 for token in meaningful if token not in vocabulary) / len(meaningful)


def _matches(tokens: tuple[str, ...], pool: Sequence[tuple[str, ...]]) -> bool:
    return any(
        tokens == other or _contains(other, tokens) or _contains(tokens, other)
        or similarity(tokens, other) >= NEAR_MATCH
        for other in pool
    )


def _flip_equation(tokens: tuple[str, ...]) -> tuple[str, ...] | None:
    """``a = b`` → ``b = a``: та же формула (ручная сверка, ответ 11B).

    Только для одного знака равенства на верхнем уровне: в цепочке
    ``a = b = c`` перестановка частей уже меняет вывод.
    """
    depth = 0
    positions = []
    for index, token in enumerate(tokens):
        if token in {"{", "(", "["}:
            depth += 1
        elif token in {"}", ")", "]"}:
            depth -= 1
        elif token == "=" and depth == 0:
            positions.append(index)
    if len(positions) != 1:
        return None
    split = positions[0]
    left, right = tokens[:split], tokens[split + 1:]
    if not left or not right:
        return None
    return (*right, "=", *left)


def _split_dumps(
    answer_forms: Sequence[tuple[str, ...]], pool: Sequence[tuple[str, ...]]
) -> list[tuple[str, ...]]:
    """Формула ответа, склеенная из нескольких формул источника, — это они.

    Иначе приём «склеить все формулы контекста в одну» давал бы одну
    формулу ответа, которая содержит эталон, — полный перенос без штрафа
    за лишнее. Склейка раскладывается на вошедшие в неё формулы источника,
    и дальше каждая считается отдельно. Вывод ``a = b = c`` из двух
    эталонных формул раскладывается на две эталонные — штрафа не будет.
    """
    units: list[tuple[str, ...]] = []
    for item in answer_forms:
        inside = [other for other in pool if len(other) < len(item) and _contains(item, other)]
        # Формула источника, вложенная в другую вошедшую, отдельно не считается.
        inside = [
            other for other in inside
            if not any(len(bigger) > len(other) and _contains(bigger, other) for bigger in inside)
        ]
        units.extend(inside if len(inside) >= 2 else [item])
    return list(dict.fromkeys(units))


def score_formulas(reference_text: str, answer: str, context: str) -> FormulaScore:
    """Сравнивает формулы ответа с эталоном и с контекстом."""
    expected = significant(extract_math(reference_text, limit=None))
    raw_answer_forms = significant(extract_math(answer))
    context_forms = significant(extract_math(context, limit=None))

    carried = 0
    partial = 0.0
    for gold in expected:
        flipped = _flip_equation(gold)
        if any(
            gold == item or _contains(item, gold) or (flipped is not None and _contains(item, flipped))
            for item in raw_answer_forms
        ):
            carried += 1
            partial += 1.0
        elif raw_answer_forms:
            partial += max(similarity(gold, item) for item in raw_answer_forms)

    answer_forms = _split_dumps(raw_answer_forms, list(dict.fromkeys((*expected, *context_forms))))

    relevant = sum(1 for item in answer_forms if _matches(item, expected))
    # «Не из контекста» — не «нет такой строки»: модель законно пересказывает
    # формулы своими обозначениями. Выдуманной считается формула, заметная
    # доля символов которой не встречается ни в одной формуле контекста
    # и эталона. Первая версия проверки (по совпадению строк) помечала
    # 280–370 формул на модель, в основном пересказ.
    vocabulary = {token for form in (*context_forms, *expected) for token in form}
    foreign = sum(
        1 for item in answer_forms
        if not _matches(item, context_forms)
        and not _matches(item, expected)
        and _unknown_share(item, vocabulary) > FOREIGN_SYMBOLS
    )
    return FormulaScore(
        expected=len(expected),
        carried=carried,
        partial=round(partial, 6),
        answer_formulas=len(answer_forms),
        relevant=relevant,
        foreign=foreign,
    )
