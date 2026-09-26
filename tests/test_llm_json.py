import json

import pytest

from rag_textbook.clients.llm_json import escape_latex_in_json, loads_llm_json, restore_latex


def test_latex_with_single_backslash_survives():
    # Так модель вернула вопрос эталона v2: \b и \t стали забоем и табуляцией.
    raw = r'{"question": "по вектору средних $\boldsymbol{\theta}_k$ и $\frac{1}{N}\sum \nabla \rho$"}'
    assert "\b" in json.loads(r'"$\boldsymbol{\theta}$"')  # сам дефект
    question = loads_llm_json(raw)["question"]
    assert question == r"по вектору средних $\boldsymbol{\theta}_k$ и $\frac{1}{N}\sum \nabla \rho$"


def test_invalid_escapes_no_longer_drop_the_answer():
    with pytest.raises(json.JSONDecodeError):
        json.loads(r'{"answer": "$\alpha + \mu$"}')
    assert loads_llm_json(r'{"answer": "$\alpha + \mu$"}')["answer"] == r"$\alpha + \mu$"


@pytest.mark.parametrize("raw", [
    '{"a": "строка\\nВторая", "b": "кавычка \\" и косая \\\\ и \\u00e9"}',
    '{"a": "уже экранировано: \\\\boldsymbol{\\\\theta}"}',
    '{"a": "табуляция\\tи перевод\\n1. пункт"}',
])
def test_valid_json_is_untouched(raw):
    assert escape_latex_in_json(raw) == raw
    assert loads_llm_json(raw) == json.loads(raw)


def test_restore_latex_repairs_already_decoded_text():
    broken = json.loads(r'"$\boldsymbol{\theta}$, $\frac12$, $\nabla f$, $\rho$ и\nВторая строка"')
    assert restore_latex(broken) == "$\\boldsymbol{\\theta}$, $\\frac12$, $\\nabla f$, $\\rho$ и\nВторая строка"
    assert restore_latex("чистый текст\nбез формул") == "чистый текст\nбез формул"
