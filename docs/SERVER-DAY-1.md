# День аренды №1 (спринт 2): данные и проба обучения

Один день карты, всё пакетом. На сервер едут только проверенные шаги;
каждый блок заканчивается решением «дальше / стоп» по заранее
записанному критерию. Лишних опытов «раз уж подняли» не делаем.

## Что проверяем

| № | вопрос | критерий | если нет |
|---|---|---|---|
| Д1 | Согласна ли награда R6 с человеком **на генерациях 4B** | согласие пар внутри вопроса ≥ 0.75, нижняя граница > 0.5 | обучение не запускаем, чиним награду офлайн |
| Д2 | Есть ли у GRPO сигнал | доля групп без разброса награды ≤ 0.5 | поднять температуру / сменить набор вопросов |
| Д3 | Помещается ли GRPO 4B + LoRA на карту и сколько стоит шаг | проба 20 шагов без OOM; ≤ 60 с на шаг | Qwen3-4B + vLLM, S-GRPO или карта 48 ГБ |
| Д4 | Мешает ли удвоенная разметка `$$$$` переносу формул | перенос формул v2 с флагом не ниже, чем без него (сравнение по слепку) | флаг не включаем |
| Г4 | Помогают ли затравки графа «вопрос + верх выдачи» (`graph_seed_both`) | recall@16 у связывающих ≥ +0.010, интервал не покрывает ноль, одношаговые не хуже −0.005 | остаётся `query`; тему затравок закрыть до графа v2 |
| Д5 | Данные для обучения | ≥ 2000 принятых вопросов по библиотеке, 0 по MML | добрать генерацией во второй день |

Д1 и Д2 меряются одним прогоном (`scripts/sample_groups.py`) на тестовых
вопросах MML: это не обучение, утечки нет.

## До аренды (на ноутбуке)

- [ ] Владелец: слить `agents/dev` в `develop/graph_rag` (код награды,
      среды, чанкера, скрипты).
- [ ] Задача 014 (Codex) принята: закреплённые версии TRL / Unsloth / vLLM,
      правки `scripts/train_grpo.py`.
- [ ] Задача 015 (Codex) принята; манифест `evaluation/library/manifest.json`
      обновлён.
- [ ] Владелец: скачать книги —
      `python scripts/fetch_library.py --download --target C:\python\rag_textbook\documents\library`
      (книги со статусом `check` — по решению после 015).
- [ ] Эпизоды MML для Д1/Д4:
      `python scripts/rl_dataset.py --trace capture/session-0819/trace-always.jsonl --goldset evaluation/goldsets/goldset.json --prompt deploy/prompts/qa-v4.txt --out artifacts/rl/mml --test-docs 0690bb81b7e3c831`
- [ ] Загрузка: `.\deploy\upload.ps1 -ServerIp <ip> -WithCaches` + каталог
      `documents/library` + `artifacts/rl`.
- [ ] Карта: RTX A5000/3090 24 ГБ (Intelion, 34–37 ₽/ч). Если 014 покажет,
      что 4B GRPO не влезает в 24 ГБ, — сразу 48 ГБ (4090 48 ГБ, ~75 ₽/ч).

## Порядок на сервере

Время — оценка по прошлым замерам; фактическое записывается в журнал.

### 1. Окружение (≈30 мин)

```bash
bash deploy/bootstrap.sh && bash deploy/restore.sh     # сервис и индекс MML из кэшей
uv venv .venv-rl && uv pip install -p .venv-rl -r tasks/014-requirements-rl.txt
```

RL-стек — в отдельном окружении: сервис не должен поехать от новых
версий torch/transformers.

### 2. Д1 + Д2: группы генераций 4B (≈20 мин карты)

Поднять Qwen3.5-4B (BF16, sglang) с выключенным размышлением, как в сервисе.

```bash
python scripts/sample_groups.py --dataset artifacts/rl/mml-test.jsonl \
    --questions 20 --n 8 --out artifacts/rl/groups-4b.jsonl --sheet artifacts/rl/groups-4b
```

Лист `groups-4b-sheet.md` скачать; ручная оценка идёт на ноутбуке
параллельно с блоками 3–5 (≈1.5 ч исследователя). Решение по Д1:

```bash
python scripts/reward_agreement.py --key groups-4b-key.json --grades groups-4b-grades.json
```

Д2 — первая строка сводки `groups-4b.summary.json`.

### 3. Д4: удвоенная разметка (≈30 мин карты, та же модель)

Ответы 4B по тому же слепку, флаг выкл/вкл, без судьи:

```bash
bash deploy/answers.sh                                      # точка отсчёта, если нет свежей
CONTEXT_NORMALIZE_MATH_DELIMITERS=true uv run rag-textbook eval answers --no-judge --label math-norm
```

Сравнение — перенос формул v2 (`scripts/rl_score_saved.py` по обеим ячейкам).

### 3б. Г4: затравки графа (≈15 мин, попутно, пока поднят сервис MML)

Офлайн (задача 011, `scripts/ppr_fair.py`) затравки из понятий вопроса
плюс трёх верхних фрагментов дали +0.021 фрагмента «только из графа» на
вопрос при p = 0.145 — слабый сигнал, поэтому только попутно и с
критерием, записанным заранее. Точка отсчёта — рабочая конфигурация
(`RETRIEVAL_ROUTER_MODE=always`, `RETRIEVAL_TOP_K=16`).

```bash
uv run rag-textbook eval ab --experiment graph_seed_both
```

 (≈1.5–2 ч карты)

Сервер инференса остановить: MinerU и модель на одной карте не живут.

Библиотека индексируется **в отдельные коллекции по языку**, не в
коллекцию MML:

- общая коллекция сдвинула бы поиск по 388 вопросам MML (новые
  отвлекающие фрагменты) и все прежние точки отсчёта;
- лексический канал — серверный BM25 со стеммером одного языка
  (`QDRANT_SPARSE_LANGUAGE`), английские книги с русским стеммером
  искались бы хуже.

```bash
LIB="CHUNKER_RESPECT_FORMULAS=true GRAPH_ENABLED=false GRAPH_RETRIEVAL_ENABLED=false"
env $LIB PDF_DIR=documents/library/ru QDRANT_COLLECTION=library_ru QDRANT_SPARSE_LANGUAGE=russian     uv run rag-textbook ingest --stages parse,chunk,embed --monitor
env $LIB PDF_DIR=documents/library/en QDRANT_COLLECTION=library_en QDRANT_SPARSE_LANGUAGE=english     uv run rag-textbook ingest --stages parse,chunk,embed --monitor
```

~2800 страниц при 1.9 с/стр. (замер 2026-08-17) ≈ 90 мин. Граф по
библиотеке **не строится**: это спринт 4 и новый код (узлы-фрагменты,
PPR), а на старой схеме — часы карты впустую. Графовый поиск выключен:
граф MML вернул бы фрагменты, которых нет в коллекции библиотеки.
Проверка сразу после разбора: 10 страниц Гельфанда — формулы живые или
скан.

MML **не переиндексировать**: эталон ссылается на номера фрагментов
(`CHUNKER_RESPECT_FORMULAS` сдвигает границы).

### 5. Д5: вопросы для обучения (≈1–1.5 ч карты, модель 9B)

Только русский комплект. Промпт генерации (`evaluation/goldset.py`)
русский: по английским книгам получились бы русские вопросы к
английскому тексту, а лексическая опора награды между языками не
работает. Английский комплект индексируется для межкнижного среза графа
(спринт 4) и проверки переноса адаптера, в обучение спринта 3 не идёт.

```bash
L=ru; export $LIB QDRANT_COLLECTION=library_$L
uv run rag-textbook goldset build --single 2000 --multihop 500     --exclude-doc 0690bb81b7e3c831 --output artifacts/rl/train-$L.json --seed 20260928
uv run rag-textbook goldset audit --path artifacts/rl/train-$L.json --write artifacts/rl/train-$L-clean.json
uv run rag-textbook eval run --goldset artifacts/rl/train-$L-clean.json     --trace artifacts/rl/train-$L-trace.jsonl --label train-$L
python scripts/rl_dataset.py --trace artifacts/rl/train-$L-trace.jsonl     --goldset artifacts/rl/train-$L-clean.json --prompt deploy/prompts/qa-v4.txt     --out artifacts/rl/lib-$L --test-docs 0690bb81b7e3c831
```

`--exclude-doc` и `--test-docs` — страховка: в коллекции библиотеки
MML нет, оба фильтра должны отбросить ноль. Если не ноль — остановиться
и разобраться.

### 6. Д3: проба GRPO (≈30–60 мин карты)

Сервер инференса остановить.

```bash
.venv-rl/bin/python scripts/train_grpo.py --dataset artifacts/rl/lib-ru-train.jsonl \
    --probe --out runs/probe-4b --num-generations 4
```

Итог — `runs/probe-4b/probe.json` (секунды на шаг, пик памяти) и
`samples.jsonl` (читать до выводов). Если OOM — `--num-generations 2`,
затем `--max-seq-length 8192` (сколько эпизодов отброшено — в выводе).

### 7. Выгрузка и выключение

`artifacts/rl/`, `runs/probe-4b/`, отчёт индексации, кэши разбора и
векторов библиотеки → `deploy/backup.ps1`. Уведомление «можно выключать».

## Бюджет

≈ 5–6 ч карты при штатном ходе, ×1.5 на отладку: 8–9 ч × 34–75 ₽ ≈
300–700 ₽. Если Д1 провален — блоки 5–6 не выполняются, день
сокращается до ≈3 ч (разбор библиотеки нужен в любом случае).

## После дня

- Приёмка вопросов по M4 (выборка 50 вручную).
- Решение по Д3: бюджет спринтов 3, 5, 6 пересчитывается до спринта 3.
- Журнал: запись с фактическим временем по блокам.
