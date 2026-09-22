"""Составная награда для RL-дообучения генератора (гипотеза R6).

Устройство — «ворота плюс сумма»:

1. **Ворота формата.** Пустой ответ, размышление вместо ответа, ответ не
   по-русски или оборванный на пределе токенов получают фиксированный
   штраф, и остальное не считается. Без ворот политика может набирать
   формульную награду ответом из одних формул на английском.
2. **Сумма частей** для прошедших ворота:

   ``R = w_f·перенос формул + w_s·опора на контекст
         − w_x·доля формул не из контекста
         − w_p·формулы вне эталона сверх допуска
         − w_l·штраф за длину``

Защита от взлома — не одна мера, а несколько, каждая против своего приёма:

* «вывалить все формулы контекста» — штраф за формулы вне эталона сверх
  допуска и штраф за длину;
* «выдумать похожие формулы» — штраф за формулы не из контекста;
* «ответ из одних формул» — ворота языка и порог доли прозы;
* «склеить формулы в одну» — склейка раскладывается на формулы источника
  (``formula._split_dumps``) и штрафуется как перечисление;
* «переписать контекст целиком» — штраф за длину; опора на контекст при
  этом высокая, но формульная часть не растёт, если эталона там нет;
* «переписать предложения контекста не по вопросу» — опора засчитывается
  в меру того, насколько ответ покрывает термины вопроса;
* «повторять одно и то же с косметическими правками» — повторы ищутся
  по канонической форме формул и нормализованному тексту предложений;
* «оговорка "нет данных" внутри полного ответа» — отказом считается
  ответ без формул, в котором маркер стоит в первом предложении или
  который короткий; длинное вежливое продолжение отказ отказом
  не отменяет (задача 019: так снимался штраф −0.5);
* «повторить вопрос» — предложения, пересказывающие вопрос, в опору
  не засчитываются (задача 019: вопрос вместо ответа получал 0.3);
* «формула из памяти вместо отказа» — ожидаются только эталонные
  формулы, видимые в контексте (``formula.score_formulas``);
* «провалить ворота нарочно» — сумма не опускается ниже штрафа ворот:
  иначе очень плохой ответ (до −2.7) был хуже пустого (−1).

* «переписать подходящий фрагмент контекста» — штраф за слова, дословно
  взятые из контекста (восьмёрками слов), сверх допуска. Без него дамп
  фрагмента выигрывал у ответа с ручной оценкой 3 в 8 вопросах из 10
  (задача 020). Честные ответы ручной сверки берут дословно до 93 слов,
  лучшие дампы — от 114;
* «вопрос с формулой вместо ответа», «отказ плюс $z$» — повтором вопроса
  считается и вопросительное предложение с формулой, а отказ не
  отменяется формулой короче порога значимости (задача 020).

Предел: ответность по-прежнему не проверяется — пересказ фрагмента своими
словами штраф за копирование обходит. Доля дословно скопированных
предложений пишется в разбор (``diagnostics["copied"]``).

Атаки оформлены тестами в ``tests/test_rewards_attacks.py``,
``tests/test_rewards_review019.py`` и ``tests/test_rewards_review020.py``
(независимые ревью, задачи 019 и 020).
После любой правки награды — ``scripts/reward_recheck.py``: закрытый
эксплойт не должен ломать порядок на ответах с ручной оценкой.

Веса — начальные, их подбор входит в абляции спринта 5.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from rag_textbook.evaluation.answers import (
    is_refusal,
    looks_like_reasoning,
    sentence_support,
)
from rag_textbook.rewards.formula import (
    MIN_TOKENS,
    FormulaScore,
    canonical_tokens,
    extract_math,
    score_formulas,
    significant,
    strip_math,
)
from rag_textbook.utils.text import content_terms, split_sentences

_WORD_RE = re.compile(r"\w+")


def latin_share(answer: str) -> float:
    """Доля латиницы вне формул любой разметки (см. ``strip_math``)."""
    letters = [c for c in strip_math(answer) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "a" <= c.lower() <= "z") / len(letters)


def repetition_share(answer: str) -> float:
    """Доля повторных предложений и формул в ответе.

    Замер 2026-09-09: часть ответов Qwen3.5-9B — зацикливание, одна и та же
    формула десятки раз подряд. Старая метрика засчитывала такой ответ как
    «формула дошла». Ворота длины ловят только длинные петли.

    Сравниваются нормализованные предложения (регистр и знаки препинания
    сняты) и канонические формы формул: иначе петля обходила бы меру,
    меняя пробелы, скобки или точку на восклицательный знак.

    Короткие формулы не считаются: обозначения вроде ``$x_n$`` или ``$U$``
    законно встречаются в ответе много раз (ложные срабатывания на ответах
    моделей 2026-09-03 — 9–21 на модель).
    """
    units: list[object] = []
    for sentence in split_sentences(strip_math(answer)):
        words = " ".join(_WORD_RE.findall(sentence.lower().replace("ё", "е")))
        if len(words) > 20:
            units.append(words)
    for formula in extract_math(answer):
        tokens = canonical_tokens(formula)
        if len(tokens) >= MIN_TOKENS:
            units.append(tokens)
    # В коротком ответе одна законно повторённая формула — уже треть единиц.
    unit_share = 1 - len(set(units)) / len(units) if len(units) >= 6 else 0.0
    return max(unit_share, _loop_share(answer))


# Кусок от 12 знаков, повторённый подряд трижды и больше, — петля генерации,
# даже если он состоит из коротких формул.
_LOOP_RE = re.compile(r"(.{12,200}?)(?:\1){2,}", re.DOTALL)
_COMMAND_RE = re.compile(r"\\[A-Za-z]+")


def _loop_share(answer: str) -> float:
    text = " ".join((answer or "").split())
    if not text:
        return 0.0
    looped = sum(
        match.end() - match.start() - len(match.group(1))
        for match in _LOOP_RE.finditer(text + " ")
        # Повтор отступов вроде «\quad \quad» — оформление, а не петля.
        if sum(c.isalnum() for c in _COMMAND_RE.sub("", match.group(1))) >= 4
    )
    return looped / len(text)


@dataclass(frozen=True)
class RewardConfig:
    # Формульная часть: за перенос хотя бы одной эталонной формулы и за долю
    # перенесённых. Вопрос обычно спрашивает об одной формуле, а эталонный
    # фрагмент несёт их несколько: при одной доле точный ответ получал 0.25
    # (ручная сверка 2026-09-17, ответ №38).
    formula_weight: float = 0.7
    coverage_weight: float = 0.3
    partial_weight: float = 0.3
    # Опора на контекст — лексическая и шумная: на ручной сверке 2026-09-17
    # при весе 0.5 она переворачивала порядок внутри вопроса чаще формул.
    # Согласие пар внутри вопроса: 0.5 → 0.755, 0.3 → 0.816, 0 → 0.571
    # (вес выбран на той же выборке — оценка оптимистична).
    support_weight: float = 0.3
    foreign_weight: float = 0.5
    offtarget_weight: float = 0.4
    # Сколько формул вне эталона допустимо без штрафа: ответ может законно
    # привести вспомогательную формулу из соседнего фрагмента. Допуск —
    # не меньше числа эталонных формул: эталон связывающих вопросов шумный
    # (второй фрагмент часто оглавление), и 80% срабатываний штрафа
    # на ответах моделей приходилось на них.
    offtarget_allowance: int = 2
    # Сверх допуска штраф растёт до полного за столько лишних формул.
    # Прежде штраф нормировался на число формул ответа и не превышал 0.2 —
    # перечислить все формулы контекста было выгоднее, чем промахнуться.
    offtarget_saturation: int = 5
    # Штраф за ответ почти без прозы выключен: на ручной сверке он бил
    # по верным ответам на «как записывается формула» (7 из 32 лучших).
    no_prose_penalty: float = 0.0
    length_weight: float = 0.3
    # Длина, после которой начинается штраф. Медианы ответов в замерах
    # 2026-09: 462–699 знаков; 2200 знаков превышали 1–3% ответов.
    soft_max_chars: int = 1800
    hard_max_chars: int = 4000
    max_latin_share: float = 0.5
    # Ниже ворот латиница тоже штрафуется, плавно: медиана доли в ответах
    # моделей 0.0, 90-й процентиль 0.03 — термины вроде «softmax» не задеты.
    soft_latin_share: float = 0.15
    latin_weight: float = 0.5
    # Повторы: доля одинаковых предложений и формул, выше которой штраф.
    max_repetition: float = 0.2
    repetition_weight: float = 1.0
    # Ниже этой доли прозы ответ считается «формулами без объяснения».
    min_prose_share: float = 0.25
    gate_penalty: float = -1.0
    # Отказ при контексте, в котором есть эталонный фрагмент, — ошибка;
    # отказ при пустом эталоне в контексте — правильное поведение.
    refusal_penalty: float = -0.5
    refusal_reward: float = 0.5
    # Отказом считается только короткий ответ с маркером: оговорка
    # «про X нет данных» внутри полного ответа — не отказ.
    max_refusal_chars: int = 400
    # Опора на контекст засчитывается в меру покрытия терминов вопроса:
    # полное — от этой доли. Без этого переписанные предложения контекста
    # не по вопросу получали полную опору.
    question_coverage_full: float = 0.5
    # Дословное копирование: слова ответа, входящие в восьмёрку слов подряд
    # из контекста. Допуск — выше максимума честных ответов ручной сверки
    # (93 слова), штраф растёт до полного за copy_saturation слов сверх него.
    copy_ngram: int = 8
    copy_free_words: int = 100
    copy_saturation: int = 50
    copy_weight: float = 1.0


@dataclass
class RewardBreakdown:
    total: float
    gate: str = ""
    formula: FormulaScore | None = None
    support_judged: int = 0
    support_ok: int = 0
    parts: dict[str, float] = field(default_factory=dict)
    # Наблюдения, не входящие в сумму: видны в разборе генераций.
    diagnostics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.formula is not None:
            payload["formula"] = {
                **asdict(self.formula),
                "recall": self.formula.recall,
                "foreign_share": self.formula.foreign_share,
            }
        return payload


def question_coverage(question: str, answer: str) -> float | None:
    """Доля содержательных терминов вопроса, которые есть в ответе."""
    asked = set(content_terms(question))
    if not asked:
        return None
    return len(asked & set(content_terms(strip_math(answer)))) / len(asked)


def _prose_share(answer: str) -> float:
    letters = sum(1 for c in strip_math(answer) if c.isalpha())
    total = sum(1 for c in answer if not c.isspace())
    return letters / total if total else 0.0


def _normalized(sentence: str) -> str:
    return " ".join(_WORD_RE.findall(sentence.lower().replace("ё", "е")))


def looks_like_refusal(text: str, cfg: RewardConfig) -> bool:
    """Отказ — маркер без формул, в первом предложении или в коротком ответе.

    Прежде отказом был только ответ до 400 знаков, и вежливое продолжение
    длиной 456 знаков снимало штраф −0.5 за отказ при видимом эталоне.
    Ответ с формулами — не отказ: оговорка внутри полного ответа законна.
    """
    if not is_refusal(text):
        return False
    short = len(text) <= cfg.max_refusal_chars
    # В коротком ответе формула ниже порога значимости («$z$») отказ не
    # отменяет: «Недостаточно информации. $z$» получал 0 вместо −0.5
    # (задача 020). В длинном — отменяет: ручная сверка, ответ 14B (оценка 3)
    # объясняет по контексту, почему связи нет, с формулами $\theta$.
    if significant(extract_math(text)) or (not short and extract_math(text)):
        return False
    first = next(iter(split_sentences(text)), text)
    return short or is_refusal(first)


def without_question_echo(answer: str, question: str) -> str:
    """Ответ без предложений, повторяющих вопрос.

    Опора на контекст лексическая, а вопрос составлен из слов контекста:
    ответ «Как определяется скалярное произведение?» получал опору 0.3
    (задача 019). Повтором считается предложение без единого своего
    содержательного слова и без значимой формулы, а также вопросительное
    предложение из слов вопроса с любой формулой: «Как определяется …
    $$s=…$$?» получал полную награду (задача 020). Порог «почти все слова из вопроса»
    (0.8) выбрасывал верные короткие ответы: «Первое ненулевое значение
    в строке ступенчатой матрицы называется ведущим» — это вопрос плюс одно
    слово ответа, и ручная оценка 3 превращалась в награду 0.
    """
    asked = set(content_terms(question))
    if not asked:
        return answer
    kept = []
    for sentence in split_sentences(answer):
        terms = set(content_terms(strip_math(sentence)))
        if terms and terms <= asked and (
            sentence.rstrip().endswith("?") or not significant(extract_math(sentence))
        ):
            continue
        kept.append(sentence)
    return " ".join(kept)


def copied_share(answer: str, context: str) -> float:
    """Доля предложений ответа, дословно взятых из контекста (без формул)."""
    sentences = [_normalized(item) for item in split_sentences(strip_math(answer))]
    sentences = [item for item in sentences if len(item) > 20]
    if not sentences:
        return 0.0
    source = _normalized(strip_math(context))
    return sum(1 for item in sentences if item in source) / len(sentences)


def copied_words(answer: str, context: str, ngram: int = 8) -> int:
    """Слова ответа (без формул), входящие в ``ngram`` слов подряд из контекста.

    Не по предложениям: замена одного слова в предложении иначе снимала бы
    совпадение целиком.
    """
    source = _WORD_RE.findall(strip_math(context).lower().replace("ё", "е"))
    grams = {tuple(source[i : i + ngram]) for i in range(len(source) - ngram + 1)}
    words = _WORD_RE.findall(strip_math(answer).lower().replace("ё", "е"))
    covered = [False] * len(words)
    for start in range(len(words) - ngram + 1):
        if tuple(words[start : start + ngram]) in grams:
            covered[start : start + ngram] = [True] * ngram
    return sum(covered)


def _leaks_reasoning(text: str) -> bool:
    # Закрывающий тег без открывающего — хвост размышления, которое
    # шаблон чата срезал только наполовину.
    return looks_like_reasoning(text) or "think>" in text.lower()


def compute_reward(
    answer: str,
    *,
    context: str,
    reference: str,
    question: str = "",
    gold_in_context: bool = True,
    truncated: bool = False,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Награда за один ответ.

    ``context`` — текст, который видела модель; ``reference`` — текст
    эталонных фрагментов; ``gold_in_context`` — попал ли эталон в контекст
    (иначе правильное поведение — признать нехватку данных);
    ``question`` — текст вопроса (без него опора не взвешивается
    покрытием вопроса); ``truncated`` — генерация оборвана пределом токенов.
    """
    cfg = config or RewardConfig()
    text = (answer or "").strip()

    for failed, reason in (
        (not text, "пустой ответ"),
        (truncated, "оборван пределом токенов"),
        (_leaks_reasoning(text), "размышление вместо ответа"),
        (latin_share(text) > cfg.max_latin_share, "ответ не по-русски"),
        (len(text) > cfg.hard_max_chars, "слишком длинный ответ"),
    ):
        if failed:
            return RewardBreakdown(total=cfg.gate_penalty, gate=reason)

    if looks_like_refusal(text, cfg):
        value = cfg.refusal_reward if not gold_in_context else cfg.refusal_penalty
        return RewardBreakdown(total=value, gate="отказ", parts={"refusal": value})

    formula = score_formulas(reference, text, context)
    first = next(iter(split_sentences(text)), text)
    if is_refusal(first) and formula.answer_formulas and formula.foreign == formula.answer_formulas:
        # Отказ с одними выдуманными формулами — отказ плюс выдумка: иначе
        # формула после «недостаточно информации» поднимала отказ −0.5
        # до −0.44 (задача 020), а при невидимом эталоне оплачивалась бы +0.5.
        value = (cfg.refusal_reward if not gold_in_context else cfg.refusal_penalty)
        value -= cfg.foreign_weight
        return RewardBreakdown(
            total=value, gate="отказ", formula=formula, parts={"refusal": value}
        )
    judged, supported = sentence_support(without_question_echo(text, question), context)
    parts: dict[str, float] = {}

    if formula.expected:
        parts["formula"] = (
            cfg.formula_weight * bool(formula.carried)
            + cfg.coverage_weight * formula.carried / formula.expected
        )
        if not formula.carried:
            # Частичный перенос засчитывается, только пока ни одна формула
            # не перенесена целиком: иначе он поощрял бы искажать остальные.
            parts["partial"] = cfg.partial_weight * formula.partial / formula.expected
    if judged:
        coverage = question_coverage(question, text)
        weight = 1.0 if coverage is None else min(1.0, coverage / cfg.question_coverage_full)
        parts["support"] = cfg.support_weight * weight * supported / judged
    if formula.answer_formulas:
        parts["foreign"] = -cfg.foreign_weight * formula.foreign_share
        offtarget = formula.answer_formulas - formula.relevant - formula.foreign
        excess = max(0, offtarget - max(cfg.offtarget_allowance, formula.expected))
        # «Вне эталона» не определено, если эталона модель не видела: тогда
        # любая формула контекста — «лишняя», и честный ответ по контексту
        # штрафовался сильнее пустого (ручная сверка, вопрос 6: оценки 2
        # получали награду ниже оценки 0). Лучшим ответом здесь остаётся
        # отказ (+0.5), а выдумку ловит штраф за чужие формулы.
        # Задача 020 предлагала штрафовать и при частично видимом эталоне
        # (дамп 1.3 против 1.15 у честного частичного ответа). Проверено
        # и отвергнуто: на ручной сверке ответы по формулам контекста
        # в вопросах 6 и 9 (оценки 2–3) теряли до 0.3, согласие падало.
        if excess and gold_in_context:
            parts["offtarget"] = -cfg.offtarget_weight * min(1.0, excess / cfg.offtarget_saturation)
        if cfg.no_prose_penalty and _prose_share(text) < cfg.min_prose_share:
            parts["no_prose"] = -cfg.no_prose_penalty
    latin = latin_share(text)
    if latin > cfg.soft_latin_share:
        span = cfg.max_latin_share - cfg.soft_latin_share
        parts["latin"] = -cfg.latin_weight * min(1.0, (latin - cfg.soft_latin_share) / span)
    repeated = repetition_share(text)
    if repeated > cfg.max_repetition:
        parts["repetition"] = -cfg.repetition_weight * min(1.0, repeated / 0.5)
    words_copied = copied_words(text, context, cfg.copy_ngram)
    over = words_copied - cfg.copy_free_words
    if over > 0:
        parts["copied"] = -cfg.copy_weight * min(1.0, over / cfg.copy_saturation)
    if len(text) > cfg.soft_max_chars:
        overflow = (len(text) - cfg.soft_max_chars) / (cfg.hard_max_chars - cfg.soft_max_chars)
        parts["length"] = -cfg.length_weight * min(1.0, overflow)

    return RewardBreakdown(
        # Пройти ворота и ответить плохо не должно быть хуже, чем провалить
        # их нарочно: иначе пустой ответ выгоднее попытки (задача 019).
        total=round(max(cfg.gate_penalty, sum(parts.values())), 6),
        formula=formula,
        support_judged=judged,
        support_ok=supported,
        parts={key: round(value, 6) for key, value in parts.items()},
        # Сколько слов до порога штрафа — видно при чтении генераций.
        diagnostics={
            "copied": round(copied_share(text, context), 3),
            "copied_words": words_copied,
        },
    )


# --- Контрольные награды (по 2506.10947, «Spurious Rewards») ---------------
#
# Прирост от основной награды принимается, только если он заметно больше
# прироста от этих двух. На моделях Qwen даже случайная награда давала
# двузначный прирост на MATH-500.


def random_reward(answer: str, *, seed: int, key: str) -> float:
    """Случайная награда, воспроизводимая по (seed, key).

    Не связана с качеством ответа, но различает ответы: зерно включает
    текст, иначе внутри группы GRPO все ответы получили бы одно число
    и преимущество было бы нулевым. Одинаковые ответы получают одно число.
    """
    rng = random.Random(f"{seed}:{key}:{answer}")
    return rng.choice((0.0, 1.0))


def format_reward(
    answer: str, *, config: RewardConfig | None = None, truncated: bool = False
) -> float:
    """Только ворота формата, без содержания: 1 за «приличный» ответ."""
    cfg = config or RewardConfig()
    text = (answer or "").strip()
    if (
        not text
        or truncated
        or _leaks_reasoning(text)
        or latin_share(text) > cfg.max_latin_share
        or len(text) > cfg.hard_max_chars
    ):
        return 0.0
    return 1.0


def has_formulas(text: str) -> bool:
    return bool(extract_math(text))
