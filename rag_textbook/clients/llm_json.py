r"""JSON из ответа модели, в котором LaTeX записан одной косой чертой.

Модель пишет формулы в JSON как в тексте: ``"\boldsymbol{\theta}"`` вместо
``"\\boldsymbol{\\theta}"``. ``json.loads`` принимает ``\b``, ``\t``, ``\f``,
``\r`` и ``\n`` за управляющие символы, и в эталон v2 (2026-09-23) попали
забой вместо ``\b`` и табуляция вместо ``\t``: 21 вопрос из 277. А ``\alpha``
или ``\sum`` — недопустимые экранирования, и такой ответ молча отбрасывался
как «не JSON» (генерация, абляция, судья).

``escape_latex_in_json`` удваивает косую черту перед командой LaTeX и перед
любым недопустимым экранированием, оставляя настоящие экранирования JSON
(``\n`` перед кириллицей, ``\"``, ``\\``, ``é``). На правильном JSON
она ничего не меняет. ``restore_latex`` чинит уже раскодированный текст.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Команды, которые начинаются с буквы допустимого экранирования JSON:
# только они неоднозначны. Остальные (\alpha, \sum) — всегда LaTeX.
_COMMANDS = """
    bar backslash begin beta big Big bigg Bigg bigl bigr bigcap bigcup bigoplus
    bigotimes bigwedge bigvee binom bmod bot boldsymbol bf bm boxed bullet because
    flat forall frac frak footnotesize
    nabla natural ne neg nearrow neq newline ni nleq ngeq nmid nolimits nonumber
    norm not notin nu nwarrow nexists
    rangle rbrace rbrack rceil rfloor rho right rightarrow rightharpoonup rm
    rVert rvert
    tag tan tanh tau tbinom text textbf textit textrm textsf texttt tfrac therefore
    theta thinspace tilde times to top triangle triangleq
"""
LATEX_COMMANDS = frozenset(_COMMANDS.split())

_ESCAPE = re.compile(r'\\(\\|["/]|u[0-9a-fA-F]{4}|[a-zA-Z]+|.)', re.DOTALL)
_UNICODE = re.compile(r"u[0-9a-fA-F]{4}")


def _fix(match: re.Match[str]) -> str:
    tail = match.group(1)
    if tail == "\\" or tail in ('"', "/") or _UNICODE.fullmatch(tail):
        return match.group(0)
    if tail[0] in "bfnrt" and tail not in LATEX_COMMANDS:
        # Настоящее экранирование: \n перед кириллицей или словом, а не командой.
        return match.group(0)
    return "\\\\" + tail


def escape_latex_in_json(raw: str) -> str:
    return _ESCAPE.sub(_fix, raw)


def loads_llm_json(raw: str) -> Any:
    """``json.loads`` с починкой LaTeX; при неудаче — исходная ошибка."""
    try:
        return json.loads(escape_latex_in_json(raw))
    except json.JSONDecodeError:
        return json.loads(raw)


_DECODED = {"\b": "b", "\f": "f", "\t": "t", "\r": "r", "\n": "n"}
# После json.loads от команды остаётся управляющий символ и хвост имени:
# \nabla — перевод строки и «abla». Ищем хвосты команд на свою букву.
_DECODED_COMMAND = {
    char: re.compile(re.escape(char) + "(" + "|".join(sorted(
        (name[1:] for name in LATEX_COMMANDS if name[0] == letter), key=len, reverse=True,
    )) + ")(?![a-zA-Z])")
    for char, letter in _DECODED.items()
}


def restore_latex(text: str) -> str:
    """Вернуть косую черту командам, съеденным ``json.loads`` раньше.

    Забой, перевод страницы и табуляция в вопросе смысла не имеют и всегда
    считаются бывшей командой; перевод строки и возврат каретки — только
    перед хвостом имени команды.
    """
    for char, letter in _DECODED.items():
        text = _DECODED_COMMAND[char].sub(lambda m, letter=letter: "\\" + letter + m.group(1), text)
    for char in "\b\f\t":
        text = text.replace(char, "\\" + _DECODED[char])
    return text
