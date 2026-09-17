# Сверка стека GRPO — задача 014

## Ход работы

- Прочитаны `AGENTS.md`, задача и `scripts/train_grpo.py`. Изменяются только два разрешённых результата; код обучения не меняется. В рабочем дереве уже есть чужие изменения.
- Начата проверка PyPI и исходников релизов. Скачанные первоисточники сохраняются в `%TEMP%/research014`, карточки — в `%TEMP%/notes-014.md`.
- Задача исследовательская: установка пакетов и запуск GPU-обучения не предусмотрены. Результаты статической сверки не будут выдаваться за проверенную совместимость на GPU.

## Зафиксированные результаты проверки

- На 17.09.2026 PyPI публикует TRL 1.13.0, Unsloth 2026.9.6 и vLLM 0.29.0. Их совместная установка противоречит опубликованным зависимостям: Unsloth ограничивает TRL версией 0.24.0, Transformers — 5.5.0, Torch — ниже 2.13; vLLM 0.29.0 требует Transformers >=5.10.4 и Torch 2.13.0.
- В TRL 1.13.0 удалён max_prompt_length; строка 159 скрипта несовместима с актуальным GRPOConfig. В TRL 0.24.0 поле существует, default=512; при переданном токенизаторе направление обрезки зависит от него.
- Оригинальный шаблон Qwen3.5-4B принимает enable_thinking=False и добавляет закрытый блок think. Карточка прямо исключает официальную поддержку мягких переключателей /think и /nothink.
- Существующие CPU-тесты выполнены: 34 passed in 0.45s. Это не проверка GPU-стека.

Источники и окончательные выводы сведены ниже; исходные ответы сохранены в `%TEMP%/research014`, карточки — `%TEMP%/notes-014.md`.

## Итоговая таблица 1–7

Дата среза — **17 сентября 2026**. «Совпадает» означает совпадение с исходным контрактом, а не успешное обучение. Для HF вместо тега релиза указан неизменяемый commit модели. Актуальная документация Unsloth и notebook обозначены отдельно от релизного кода.

| № | Версия и проверка | Вывод и первоисточник |
|---|---|---|
| 1 | Последний TRL **1.13.0** (PyPI, опубликован 10.09.2026 UTC). `loss_type="dr_grpo"`, `scale_rewards=False`, `use_vllm`, `max_completion_length`, `num_generations`, `beta`, `temperature` существуют. | **Совпадает** для этих полей. `scale_rewards` аннотирован `str`, default `"group"`, но bool официально преобразуется: True→group, False→none; допустимо также batch. Defaults: loss_type=dapo, use_vllm=False, max_completion_length=512, num_generations=8, beta=0.0, temperature=1.0. [PyPI](https://pypi.org/pypi/trl/1.13.0/json), [поля и defaults](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_config.py#L479), [loss/reward scale](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_config.py#L784), [bool-преобразование](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_config.py#L1065). |
| 1 | `max_prompt_length` в TRL 1.13.0 | **Расходится**: поля нет ни в GRPOConfig, ни в его `_BaseConfig`; default и направление обрезки этим параметром **не существуют**. В строковом пути `_tokenize_prompts` вызывается `processing_class(text=prompts)` без `truncation/max_length`. Строка 159 скрипта приведёт к ошибке неизвестного аргумента конструктора. [GRPOConfig](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_config.py), [базовый класс](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/base_config.py), [токенизация](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_trainer.py#L1815). |
| 1 | Кратность batch в TRL 1.13.0 | **Совпадает** с текущими defaults скрипта. Проверяется generation_batch_size, а не только microbatch × процессы. По умолчанию `generation_batch_size = per_device_train_batch_size × world_size × gradient_accumulation_steps`; он должен делиться на num_generations, num_generations ≥2. При явных настройках steps_per_generation заменяет grad accumulation в этом произведении; generation_batch_size должен также делиться на глобальный microbatch. Одновременно задавать generation_batch_size и steps_per_generation нельзя. Для eval проверяется `per_device_eval_batch_size × world_size` относительно num_generations_eval. В скрипте 4×world_size×4 делится на 4. [Проверки конструктора](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_config.py#L1083). |
| 2 | Reward callable, TRL 1.13.0 | **Совпадает**. Передаются `prompts`, `completions`, `completion_ids`; остальные колонки датасета превращаются в списки keyword-аргументов. Исключаются колонки prompt/completion/completion_ids. Поэтому context, reference, gold_in_context, question, question_id доступны. Дополнительно передаются trainer_state, log_extra, log_metric, а при наличии окружений — environments. `**_` в скрипте принимает лишнее. Для строкового prompt completions — **list[str]**, декодированная с skip_special_tokens=True; для conversational prompt — списки сообщений. [Сбор kwargs и вызов](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_trainer.py#L1640), [вид completions](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_trainer.py#L2262). |
| 3 | Уже отформатированный строковый prompt, TRL 1.13.0 + оригинальный Qwen3.5-4B | **Совпадает** в части отсутствия второго chat template. Conversational и строковый пути разделены. Однако в новой строковой токенизации `add_special_tokens=False` явно не задан: общей гарантии для любого токенизатора нет. Для проверенного Qwen конфигурация содержит bos_token=null, add_bos_token=false, поэтому второго BOS здесь не ожидается. [TRL](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_trainer.py#L1815), [Qwen tokenizer_config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer_config.json#L284). Проверка не переносится автоматически на изменённый Unsloth токенизатор. |
| 4 | Qwen/Qwen3.5-4B, commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | **Совпадает**: при add_generation_prompt=True шаблон принимает enable_thinking=False и дописывает закрытый блок `<think>\n\n</think>\n\n`. Иначе открывает `<think>`. Альтернатива из карточки при работе через API: `extra_body={"chat_template_kwargs":{"enable_thinking":False}}`; для DashScope — `extra_body={"enable_thinking":False}`. Это способы передачи настройки, а не другой алгоритм. Мягкие `/think` и `/nothink` официально не поддерживаются. [Шаблон](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja#L147), [карточка, Non-Thinking Mode](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md#L954). |
| 5 | Unsloth **2026.9.6**, релиз **v0.1.810-beta**, 17.09.2026 UTC | **Совпадает** по сигнатуре: fast_inference и max_lora_rank есть; Qwen3.5 делегируется из FastLanguageModel в FastModel с этими аргументами. Пакетный номер не совпадает с именем git tag; соответствие проверено по `_version.py`. [Релиз](https://github.com/unslothai/unsloth/releases/tag/v0.1.810-beta), [версия](https://github.com/unslothai/unsloth/blob/v0.1.810-beta/unsloth/_version.py), [сигнатура](https://github.com/unslothai/unsloth/blob/v0.1.810-beta/unsloth/models/loader.py#L400), [делегирование](https://github.com/unslothai/unsloth/blob/v0.1.810-beta/unsloth/models/loader.py#L880). |
| 5 | GRPO Qwen3.5 и fast_inference | Поддержка GRPO представлена в официальном **Qwen3.5-4B Vision GRPO notebook**, но он использует **FastVisionModel, fast_inference=False, Transformers 5.2.0, TRL 0.22.2**, max_prompt_length=1024. Это не доказательство данного текстового стека с vLLM и длинным промптом. **Нельзя установить** первый релиз/дату внедрения GRPO Qwen3.5 и работоспособную комбинацию fast_inference=True по проверенным источникам. На дату среза пример уже есть; подменять дату появления датой последнего пакета нельзя. [Notebook, фиксированный commit](https://github.com/unslothai/notebooks/blob/bce1b9d12e566c9df217fa236904ae3da3415247/nb/Qwen3_5_(4B)_Vision_GRPO.ipynb). |
| 5 | Гибридные слои и требуемая vLLM | В релизном loader Qwen3.5 отнесён к семействам FLA/GDN; есть предупреждение о NaN при float16. Blanket-запрета Qwen3.5 в fast_inference_setup нет, но отсутствие запрета не доказывает корректную передачу LoRA/весов в vLLM. **Нельзя установить** подтверждённую версию vLLM для этой связки. Unsloth не задаёт единственный обязательный vLLM pin в основных зависимостях. **Расходится** утверждение CLI «нужен свежий Unsloth» как достаточное условие: найдены несовместимые ограничения пакетов (ниже). [FLA/GDN](https://github.com/unslothai/unsloth/blob/v0.1.810-beta/unsloth/models/loader.py#L125), [FLA families](https://github.com/unslothai/unsloth/blob/v0.1.810-beta/unsloth/models/loader.py#L258), [fast_inference_setup](https://github.com/unslothai/unsloth/blob/v0.1.810-beta/unsloth/models/_utils.py#L3665), [метаданные Unsloth](https://pypi.org/pypi/unsloth/2026.9.6/json). |
| 6 | vLLM **0.17.0**, 07.03.2026 UTC | **Совпадает** для поддержки архитектуры: первый подтверждённый релиз — 0.17.0, release notes объявляют Qwen3.5 family; есть Qwen3_5ForConditionalGeneration с IsHybrid и реализация GatedDeltaNet. В v0.16.0 файла по тому же пути нет. Это минимальная версия подтверждённой архитектурной поддержки, **не минимальная доказанная версия Unsloth GRPO+LoRA**. Текущий PyPI — 0.29.0. [Релиз 0.17.0](https://github.com/vllm-project/vllm/releases/tag/v0.17.0), [GDN](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/models/qwen3_5.py#L122), [архитектура](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/models/qwen3_5.py#L630), [актуальные метаданные](https://pypi.org/pypi/vllm/0.29.0/json). |
| 7 | GRPO+LoRA, 3–5B, prompt 8–12K, GPU 24–48GB | **Нельзя установить** требуемый объём памяти: в проверенных публикациях нет замера, одновременно удовлетворяющего всем условиям. Ниже приведены ближайшие опубликованные результаты и отличия. Собственных оценок и переноса цифр на Qwen3.5 нет. |

### Отдельно: контракт закреплённого TRL 0.24.0

Он выбран из-за ограничения Unsloth `trl<=0.24.0`, а не назван последним TRL.

- `max_prompt_length=512` по умолчанию, документация обещает обрезку слева. Но при переданном `processing_class=tokenizer` trainer оставляет объект вызывающей стороны; в обычной генерации передаёт `truncation=True`, `max_length=...`, `padding_side="left"`, `add_special_tokens=False`. **Padding не определяет truncation_side**. Направление обрезки определяется переданным токенизатором; явно задать `tokenizer.truncation_side="left"` полезно для соблюдения контракта. При автоматической загрузке trainer сам задаёт truncation_side="left". [Default](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py#L300), [загрузка](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py#L275), [обычная генерация](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py#L1272). В vLLM-пути передаётся truncate_prompt_tokens=max_prompt_length; это другой путь токенизации.
- Скрипт уже отбрасывает строки длиннее `12288−768=11520` токенов, посчитанных без дополнительных спецтокенов. При неизменном токенизаторе в обычном пути TRL 0.24.0 обрезка не должна срабатывать. Это проверка по коду, не наблюдение GPU-запуска.
- Reward получает те же колонки, prompts, completions, completion_ids, а из служебных kwargs — trainer_state; log_extra/log_metric относятся к проверенному новому API. Строка не становится разговором: `maybe_apply_chat_template` сначала проверяет формат. [Reward](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py#L1018), [prompt](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py#L1085), [проверка формата](https://github.com/huggingface/trl/blob/v0.24.0/trl/data_utils.py).
- `scale_rewards=False` поддерживается и здесь, хотя аннотация str; заменять на `"none"` необязательно. `vllm_mode` по умолчанию **server**, тогда как в TRL 1.13.0 — **colocate**. Само `use_vllm=True` в обычном TRL 0.24.0 не означает запуск локального colocated engine. [Config 0.24.0](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py#L406), [bool](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py#L644).

## Зависимости и предлагаемый файл версий

Метаданные PyPI проверены без установки. Использованы основные зависимости, без extras; статическая проверка ниже рассматривает Linux x86_64 / Python 3.11, а не разрешение всего дерева зависимостей.

| Пакет | Опубликованное ограничение, важное для задачи | Следствие |
|---|---|---|
| [Unsloth 2026.9.6](https://pypi.org/pypi/unsloth/2026.9.6/json) | trl>=0.18.2, !=0.19.0, <=0.24.0; transformers<=5.5.0; torch>=2.4,<2.13; datasets>=3.4.1,<4.4.0 с исключениями 4.0.*,4.1.0; peft>=0.18 | Актуальный TRL 1.13.0 не подходит по метаданным. |
| [TRL 1.13.0](https://pypi.org/pypi/trl/1.13.0/json) | datasets>=4.7.0; extra vllm>=0.19.1,<=0.28.0 | Конфликт datasets с Unsloth; latest vLLM 0.29.0 также вне extra. |
| [TRL 0.24.0](https://pypi.org/pypi/trl/0.24.0/json) | transformers>=4.56.1; datasets>=3.0.0; **extra vllm==0.10.2** | Базовые зависимости допускают выбранный кандидат без vLLM. Extra содержит версию до подтверждённой поддержки Qwen3.5. |
| [vLLM 0.17.0](https://pypi.org/pypi/vllm/0.17.0/json), [0.17.1](https://pypi.org/pypi/vllm/0.17.1/json), [0.18.0](https://pypi.org/pypi/vllm/0.18.0/json), [0.19.0](https://pypi.org/pypi/vllm/0.19.0/json) | transformers>=4.56.0,<5; torch==2.10.0 | Не допускают Transformers 5.2.0, использованный официальным Qwen3.5 GRPO notebook. Внутренняя реализация Qwen3.5 в vLLM не добавляет эту модель в Transformers 4.x для обучения. |
| [vLLM 0.19.1](https://pypi.org/pypi/vllm/0.19.1/json), [0.20.0](https://pypi.org/pypi/vllm/0.20.0/json), [0.21.0](https://pypi.org/pypi/vllm/0.21.0/json), [0.22.0](https://pypi.org/pypi/vllm/0.22.0/json) | Исключены transformers 5.0.*,5.1.*,5.2.*,5.3.*,5.4.*,5.5.0 | Обход переходом на эти проверенные версии не решает конфликт с Transformers 5.2.0. |
| [vLLM 0.24.0](https://pypi.org/pypi/vllm/0.24.0/json), [0.28.0](https://pypi.org/pypi/vllm/0.28.0/json) | transformers>=5.5.3 | Выше верхней границы Unsloth 5.5.0. |
| [vLLM 0.29.0](https://pypi.org/pypi/vllm/0.29.0/json) | transformers>=5.10.4; torch==2.13.0 | Два прямых конфликта с Unsloth. |

**Согласованный полный набор семи запрошенных библиотек не установлен.** Проверены перечисленные релизы, а не все исторические patch-релизы и не все возможные комбинации. Нельзя утверждать математическую невозможность любой связки; нельзя и предлагать заведомо конфликтующий набор как рабочий.

`tasks/014-requirements-rl.txt` содержит кандидат **без vLLM**: TRL 0.24.0, Unsloth 2026.9.6, Transformers 5.2.0, PEFT 0.18.1, datasets 4.3.0, Torch 2.10.0; дополнительно закреплён обязательный unsloth-zoo 2026.9.5. Для каждого пакета есть ссылка на неизменяемые метаданные. `vllm==0.17.0` приведён **комментарием как отклонённый кандидат**, с причиной. Следовательно, требование семи активных совместимых pins не выполнено; файл не надо представлять как готовую среду для `--vllm`.

Кандидат удовлетворяет 15 опубликованным зависимостям **между выбранными пакетами**. Это не полный lock, не проверка импорта, не ABI/CUDA-проверка и не свидетельство успешного Qwen3.5 GRPO. Выбор Transformers 5.2.0 дополнительно опирается на официальный notebook и релизный [AutoModel mapping](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/auto/modeling_auto.py#L693): qwen3_5→Qwen3_5ForCausalLM уже присутствует. Поэтому утверждение, что fallback AutoModelForCausalLM в строке 214 заведомо не знает эту архитектуру, было бы неверным.

## Опубликованная память: что именно найдено

| Источник | Опубликованные цифры | Почему это не искомый замер |
|---|---|---|
| [Unsloth Memory Efficient RL, Performance Experiments](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl#performance-experiments) | Qwen3-4B на **T4**, num_gen=2, grad_acc_steps=2: без standby при vllm_gpu_util=0.7 — **~15.1 GiB**, 28 steps / 39 min; со standby — **13 GiB**, преимущественно 10–11 GB, 29 steps / 40 min. При util=0.95 — **14.5 GiB**, 40 steps / 40 min. | GPU вне диапазона 24–48GB; не Qwen3.5; в этой таблице нет подтверждённой длины **промпта** 8–12K. Указание capacity 32K KV cache нельзя выдавать за длину обучающего промпта. |
| [Unsloth GRPO blog, memory breakdown](https://unsloth.ai/blog/grpo) | Llama3.1-8B, context 20K, 8 generations: **54.3GB** у Unsloth против **510.8GB** в сравнении авторов. | Размер модели, длина и память вне условий. В публикации есть расчётный breakdown, поэтому это не независимый замер искомой конфигурации. |
| [Unsloth R1 reasoning](https://unsloth.ai/blog/r1-reasoning) | Для Llama3.2-3B заявлена экономия **3GB** благодаря устранению дублирования весов. | Это экономия, не общий peak GRPO; нет нужной длины промпта и карты. |
| [Qwen3.5-4B Vision GRPO notebook](https://github.com/unslothai/notebooks/blob/bce1b9d12e566c9df217fa236904ae3da3415247/nb/Qwen3_5_(4B)_Vision_GRPO.ipynb) | loader max_seq_length=16384, но training max_prompt_length=1024 и max_completion_length=1024, fast_inference=False. | Значение loader max_seq_length нельзя считать измеренным длинным промптом. Сам пример не подтверждает нужный peak на 24–48GB. |

Проверены эти публикации, текущие руководства [RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) и [Qwen3.5](https://unsloth.ai/docs/models/qwen3.5). Полного опубликованного замера под условия задачи **не найдено в проверенных источниках**. Оценивать самостоятельно или обещать, что 24/48GB хватит, оснований нет. Исчерпывающего поиска всех GitHub issues и статей не проводилось.

## Что изменить в scripts/train_grpo.py — только предложения

Номера относятся к текущему файлу. **Код не изменялся.** Базовый предлагаемый путь — закреплённый TRL 0.24.0, без vLLM. Миграция на latest TRL — отдельный вариант, несовместимый с текущими метаданными Unsloth.

| Строка | Было | Стало / предлагаемое изменение | Основание |
|---|---|---|---|
| 118–121, 198 | GRPOConfig/GRPOTrainer импортируются до того, как load_model импортирует Unsloth. | При выборе Unsloth импортировать его до TRL/Transformers, затем импортировать trainer. Выбор backend сделать явным, сохранив отдельную ветку fallback. | Такой порядок использует официальный Qwen3.5 GRPO notebook; это требуется для применения патчей до использования trainer. |
| После загрузки tokenizer, перед строкой 128 | truncation_side явно не задан. | `tokenizer.truncation_side = "left"`; сохранить фильтр строк 128–147. | TRL 0.24.0 не переопределяет это свойство переданного токенизатора. Это фиксирует поведение при обрезке, а не заменяет отбрасывание слишком длинных примеров. |
| 159, **только для перехода на TRL 1.13.0** | `max_prompt_length=max_prompt` | Удалить аргумент из GRPOConfig, оставить собственный расчёт длины и фильтр. | В latest поле удалено. **Для предложенного TRL 0.24.0 строку оставить.** Одного удаления недостаточно, чтобы latest стал совместим с Unsloth. |
| 171, 202, 237 | `use_vllm=args.vllm`, `fast_inference=args.vllm`; help обещает «нужен свежий Unsloth». | Для предлагаемого стека — False в обоих местах; запрос `--vllm` завершать ранним понятным сообщением о неподтверждённом совместимом стеке. Help заменить на описание экспериментального, пока неподдержанного этим файлом версий режима. | Установлены конфликты зависимостей; обновления только Unsloth недостаточно. |
| 171, **если впоследствии будет подтверждён обычный TRL 0.24.0 + локальный vLLM** | vllm_mode не задан. | Явно `vllm_mode="colocate"`, если нужен локальный engine, либо явно server и параметры отдельного сервера. Для Unsloth-патчей проверить их собственный путь отдельно. | Default 0.24.0 — server; это условное изменение, не обход конфликта пакетов. |
| 209 | `except ImportError` охватывает импорт и всё создание/адаптацию модели. | Отделить отсутствие Unsloth при импорте от ошибки его загрузчика; ошибки зависимостей/создания модели не превращать молча в fallback. | Иначе проверка требуемого backend теряет смысл: сломанный стек может незаметно стать другим режимом обучения. Это вывод из локального кода. |

**Оставить:** reward signature с `**_`, строковый prompt, `enable_thinking=False`, `loss_type="dr_grpo"`, `scale_rewards=False`. Кратность defaults batch корректна; обязательного увеличения microbatch до num_generations нет, но менять batch ради предполагаемой экономии памяти без замеров здесь не предлагается.

## Проверки и незавершённое

Выполнена команда:

```text
C:\python\rag_textbook\.venv\Scripts\python.exe -m pytest tests/test_rl_env.py tests/test_rewards.py -q
```

Последние строки, как получены:

```text
..................................                                       [100%]
34 passed in 0.45s
```

Статическая проверка сохранённых метаданных (`%TEMP%/validate014.py`, никаких установок):

```text
PASS: 15 declared dependencies between 7 active pins (including unsloth-zoo); Linux x86_64, Python 3.11, no extras.
CONFIRMED CONFLICT: vllm 0.17.0 requires transformers<5,>=4.56.0; candidate transformers==5.2.0.
CONFIRMED CONFLICT: vllm 0.19.1 requires transformers!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*,!=5.4.*,!=5.5.0,>=4.56.0; candidate transformers==5.2.0.
CONFIRMED CONFLICT: vllm 0.29.0 requires transformers>=5.10.4; candidate transformers==5.2.0.
```

- Созданы только два разрешённых результата: этот отчёт и файл предлагаемых версий. Новые тесты в репозиторий не добавлялись: задачей разрешены только эти два файла; существующие тесты не менялись.
- Пакеты не установлены, окружение не изменено, GPU-проверка/обучение не выполнены. Код не исправлен по прямому условию задачи, коммит не сделан.
- **Не установлены:** первый релиз GRPO Qwen3.5 в Unsloth, подтверждённая работоспособность fast_inference=True для этой связки, совместимые семь активных pins с vLLM и искомый замер памяти. Это ограничения результата, а не выполненные проверки.
- Источники перечислены непосредственно у выводов. Сетевые обращения — только чтение публичных исходников, метаданных и документации через Python urllib; файлы проекта наружу не отправлялись.
- Финальная проверка формата: `PASS: requirements syntax, exact pins, report sections 1-7 and test result`. Семь активных строк включают unsloth-zoo; vLLM остаётся отключённым комментарием. Оба результата новые (untracked), коммит не выполнялся.
