"""Стоп-слова для лексического канала и извлечения терминов.

Вынесены отдельным модулем, чтобы не загромождать логику работы с текстом.
Список русских слов включает служебные и связочные конструкции, которые в
технических текстах встречаются повсеместно и потому бесполезны как термины.
"""

from __future__ import annotations

_RUSSIAN = """
    а без более больше будет будто бы был была были было быть в вам вас вдруг ведь во вот
    впрочем все всегда всего всех всю вы где да даже данная данные данный два для до другой
    его ее ей ему если есть еще ж же за зачем здесь и из или им именно иногда их к каждая
    каждый кажется как какая какой когда конечно которая которого которое которой которые
    который которым которых кто куда ли лучше любая любой между меня мне много может можно мой
    моя мы на над надо наконец например нас не него нее ней некоторые некоторый нельзя нет ни
    нибудь никогда ним них ничего но ну о об один он она они опять от перед по под после потом
    потому почти при примерно про раз разве с сам свою себе себя сейчас сказал со совсем так
    также такой там тебя тем теперь то тогда того тоже только том тот три тут ты у уж уже
    хорошо хоть чего чем через что чтоб чтобы чуть эта эти этих это этого этой этом этот эту я
"""

_ENGLISH = """
    a all an and any are as at be been being both but by can chapter did do does each else
    equation example few figure for from here how if in into is it its just more most of on
    only onto or other own same section so some such table than that the then there these this
    those to too very was were what when where which who whom whose why will with without
"""

RUSSIAN_STOPWORDS: frozenset[str] = frozenset(_RUSSIAN.split())
ENGLISH_STOPWORDS: frozenset[str] = frozenset(_ENGLISH.split())
STOPWORDS: frozenset[str] = RUSSIAN_STOPWORDS | ENGLISH_STOPWORDS
