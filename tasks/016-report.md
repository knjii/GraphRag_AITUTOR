# Отчёт по задаче 016

Проверка чтением кода текущего рабочего дерева; исходные незакоммиченные изменения сохранены. Код, данные и конфигурация не меняются. Чтение deploy/, docker/ и .env.example выполнено по прямому требованию задачи 016; реальные .env и секреты не читаются.

## Ход проверки

- Прочитаны AGENTS.md, tasks/016.md, deploy/day1.sh (1–328), docs/SERVER-DAY-1.md (1–116).
- Найдено: `set -uo pipefail` без `-e`; многие команды завершаются `done_mark` без проверки результата (например D4, G4, L1/L2, Z9). Уточняется по вызываемому коду.
- Проверка только статическая: сценарий аренды, Docker, установка пакетов и сетевые операции записи не запускаются.
- Подтверждено: Q2 автоматически аудирует один `*_chunks.json`, затем удаляет вопросы других документов как `missing_chunk` (`cli/main.py:632`, `evaluation/audit.py:126`, `cli/main.py:695`). Нужен явный объединённый корпус.
- Подтверждено: L2 включает enrichment; при `.env.example` vision URL наследует остановленный SGLang (8001), а модель vision — `qwen2.5vl:3b`. Ошибки описаний подавляются. Требуется явно выбрать политику enrichment.
- Импорты приложения RL не требуют установки qdrant-client/fastembed/neo4j: эти импорты отложены до обращения к хранилищам. CUDA/совместимость обучающего стека статически не подтверждены.
- fastembed: `english` есть в поддерживаемых языках BM25; исходник сохранён в `%TEMP%/016-bm25.py`, ссылка и найденные строки записаны в `%TEMP%/notes-016.md`.

## Итог

Сценарий пока не готов к запуску без исправлений. Имена и типы переданных опций CLI совпадают с реализацией; основные проблемы — продолжение после ошибок, неполный корпус аудита Q2 и модель зрения на L2. Оценка «ок» ниже означает соответствие коду, а не успешный прогон на GPU. Нумерация строк относится к текущему рабочему дереву.

## Команды E0–Z9

| Шаг | Команда / операция | Вывод | Основание |
|---|---|---|---|
| Общие | `cd`, PATH, UV_DEFAULT_INDEX, `mkdir -p`, журнал | ок | `deploy/day1.sh:21–33`: переход проверяется, каталоги RUN/OUT создаются; переменные shell не обязаны быть alias Settings. |
| Общие | `--list`, `--only D4`, `--from L1`, цикл PLAN, `done_mark` | риск | `deploy/day1.sh:89–112,323`: допустимые варианты работают; неизвестный идентификатор шага даёт пустой успешный прогон; метки не зависят от входных файлов/настроек. `--only` выполняет заново, `--from` пропускает уже отмеченное. |
| Общие | `set -uo pipefail`, `clean`, конвейеры | ошибка | `deploy/day1.sh:20,44,112`: pipefail вычисляет ненулевой статус, но без проверки статуса выполнение продолжается и создаётся `.done`. `grep -v` также возвращает 1 при полностью отфильтрованном выводе; просто добавить `-e` недостаточно без исправления фильтра. |
| Общие | `ensure_4b`, `model_4b_with_search`, `model_9b_alone` | риск | `deploy/day1.sh:48–59`: проверяются настройка модели и два HTTP health, не фактическая модель/Qdrant/Neo4j. Ошибка model-swap в функции 4B может быть скрыта последним успешным services up. `deploy/model-swap.sh:49–75` корректно останавливает/пересоздаёт SGLang и ждёт health. |
| E0 | `nvidia-smi`, `df`, `find ...pdf`, `sha256sum -c`, проверки файлов | ок / риск | `deploy/day1.sh:117–124`: суммы и три обязательных файла проверяются с остановкой; nvidia-smi и df лишь информируют. Не проверяются trace, prompt, requirements-rl-app и восстановленный корпус до дорогих шагов. |
| E0 | `reward_fingerprint.py --dataset ... --check ...` | ок | `scripts/reward_fingerprint.py:51–54`: оба Path, dataset обязателен, check — обязательная альтернатива write. Статус именно Python проверяется через PIPESTATUS (`day1.sh:127`). |
| E0 | фоновые `uv venv --python 3.11 .venv-rl`, две `uv pip install -p ... -r ...`, `touch rl-env.ok` | ок / риск | `day1.sh:129–135`: цепочка && не создаёт маркер после ошибки; разные окружения. Старый rl-env.ok не привязан к содержимому requirements, и R0 не проверяет все версии на равенство закреплённым. |
| E0 | установка CLI `hf`, volume inspect/create, `HF_HOME`, `hf download ... --exclude '*.gguf'` | риск | `day1.sh:67–71,140–147`: том совпадает с `rag-textbook_hf_cache` (`docker/docker-compose.vllm.yml:126`); фоновые загрузки не ожидаются и не проверяются. Ошибки/право записи в mountpoint превращаются в позднее скачивание весов. gguf не требуется сценарию. |
| D1 | `ensure_4b`; sample_groups 2×2; чтение smoke.summary.json | ок / риск | `scripts/sample_groups.py:147–155`: dataset/out обязательные Path, questions/n — int; `:137` пишет правильное имя summary. Python-ворота используют существующие поля и останавливают полностью проваленную разминку (`day1.sh:159–164`), но после ошибки генерации могут прочитать старую сводку. |
| D1 | sample_groups 20×8, `--sheet .../groups-4b`; Python-проверка доли | ок / риск | `sample_groups.py:107–108,137`: получаются groups-4b.jsonl, groups-4b.summary.json, groups-4b-sheet.md и groups-4b-key.json. `day1.sh:169–177`: отсутствие сигнала только печатается, после провала команды возможен done. |
| D1 | напечатанная команда reward_agreement | ок / риск | `scripts/reward_agreement.py:53–59`: key/grades — обязательные Path. Автоматически не запускается. grades — ручной JSON вида `{"1.1":3,"1.2":0}`; ключ тоже надо скачать и выполнить команду из каталога с обоими файлами. |
| D4 | два `eval answers --from-trace ... --no-judge --label ...` | ок / риск | `cli/main.py:1201–1225`: from-trace Path, label str, no-judge — булева опция. `:1280–1300` использует замороженный контекст без поиска, но грузит обычный goldset и локальные чанки (`:1233,1252`). `day1.sh:183–186` задаёт нормализацию только во второй ветке: при true в окружении обе ветки одинаковы. |
| D4 | Settings().paths.metrics_dir; `rl_score_saved.py` два файла, `--trace`, `--json` | ок / риск | `evaluation/answers.py:545` пишет answers_day1-4b-base.json / answers_day1-4b-mathnorm.json именно в metrics_dir; `scripts/rl_score_saved.py:53–55`: positional cells nargs+, trace/json Path. Пути совпадают. Ошибка любой команды не препятствует done (`day1.sh:193`). |
| G4 | `eval ab --experiment graph_seed_both`; сообщение критериев | ок / риск | `cli/main.py:1369–1394`: experiment str, значение зарегистрировано. Требуются Qdrant, embeddings/reranker и Neo4j при включённом графе; ensure4b обычно поднимает их. Пороги напечатаны, а не автоматически применены; статус eval не проверен (`day1.sh:199–201`). |
| L1 | `llm_off`; три `ingest --stages parse --monitor` для ru/ru-ocr/en | ок / риск | `cli/main.py:225–243,262`: stages str со списком через запятую, monitor bool. Значения языков/методов допустимы для MinerU 2.5.4 (источники ниже). `day1.sh:214–218` останавливает только SGLang: Infinity остаётся на GPU; обещание «только MinerU» неточно. Оценить достаточность VRAM без карты нельзя. |
| L1 | обработка результата ingest | ошибка | `indexing/pipeline.py:522–528` записывает ошибки разбора; CLI возвращает 1 при report.failed (`cli/main.py:286–287`), но day1 его игнорирует и отмечает L1 готовым. |
| L2 | `check_parsed_text.py` без опций | ок / риск | `scripts/check_parsed_text.py:65–105`: по умолчанию artifacts/parsed, проверяет ru-/en- blocks.json, пустой набор/плохой текст — exit 1; `day1.sh:223` его проверяет. Не сверяет полный список PDF: отсутствующую книгу среди успешных не заметит; PARSED_DIR из Settings не учитывает. |
| L2 | три `ingest --stages chunk,embed --monitor` | ошибка | `indexing/pipeline.py:565–568` вызывает enricher. `.env.example:36,113,115,151`: enrichment включён, vision model qwen2.5vl:3b, отдельный URL пуст. `config.py:370–385` наследует URL SGLang 8001, остановленный L1; `chunking/enrichment.py:105–116` подавляет ошибку. Описания картинок теряются, повторные попытки тратят время. Embeddings идут в оставшийся Infinity. При возобновлении после P1 Infinity тоже остановлен, а L2 его не поднимает. |
| Q1 | `search_off`; model-swap 9B .85 | ок / риск | `day1.sh:233–234`, `model-swap.sh:49–75`: останавливаются Infinity/Ollama, Qdrant/Neo4j остаются CPU; SGLang 9B занимает GPU. Ошибка swap не останавливает шаг. |
| Q1 | goldset build --single 2000 --multihop 500 --workers 16 --exclude-doc ID --output PATH --seed 20260928 | ок / риск | `cli/main.py:428–465`: int/int/int/list[str]/Path/int, все переданные опции существуют. `:489–503` читает готовые чанки Qdrant, не вызывает embed; `evaluation/goldset.py:294,452–455` при отсутствии графа выбирает пары эвристически. Нужны Qdrant + LLM, Neo4j при GRAPH_ENABLED=false не нужен (`context.py:219–224`). `test -s` на выходе не исключает старый файл и пустой JSON-набор (`day1.sh:241`). |
| Q2 | `ensure_4b`; `goldset audit --path ... --write ...` | ошибка | Опции Path существуют (`cli/main.py:643–650`), но нет `--chunks`. `_find_chunks_file` берёт **один первый** файл из capture, затем artifacts/parsed (`:632–639`), а не всю библиотеку. `evaluation/audit.py:126–132` помечает остальные ссылки missing_chunk, `cli/main.py:695–697` исключает такие вопросы. Возможен нулевой либо резко урезанный train-ru-clean.json. |
| Q2 | `eval run --goldset ... --trace ... --label day1-train-ru` | ок | `cli/main.py:924–958`: goldset/trace Path, label str; trace записывается по переданному пути. Нужны Qdrant и Infinity. При RETRIEVAL_ROUTER_MODE=always модель роутером **не вызывается** (`retrieval/router.py:116–124`); rewrite без истории также не вызывает её (`retrieval/pipeline.py:239–245`). Neo4j не нужен: оба GRAPH флага false. Поднятая 4B здесь лишняя, но не отсутствующая необходимая служба. |
| Q2 | `rl_dataset.py --trace ... --goldset ... --prompt ... --out .../lib-ru --test-docs ID`; wc-проверки | ок / риск | `scripts/rl_dataset.py:26–31`: четыре обязательных Path, test-docs nargs*; `:49–56` грузит все локальные чанки, пишет lib-ru-train.jsonl и lib-ru-test.jsonl. `day1.sh:259–263`: нулевой train запрещён, ненулевой MML-test и <2000 только предупреждают. После сбоя возможны старые файлы. |
| R0 | ожидание rl-env.ok, pgrep, tail; import unsloth → torch/trl/transformers/peft; assert CUDA | ок / риск | `day1.sh:269–281`: до 60×30 секунд, ошибка импортов/CUDA останавливает; Unsloth импортирован первым. Проверка статуса фонового процесса по тексту pgrep менее надёжна PID; не проверяется импорт datasets/train_grpo. Маркер может устареть. GPU ещё занят 4B/Infinity, но здесь модель обучения не загружается. |
| R0 | reward_fingerprint в .venv-rl | ок | `day1.sh:282–284`: тот же dataset/отпечаток, проверяется PIPESTATUS Python; отсутствие/изменение морфологии выявляется fingerprint (`reward_fingerprint.py:57–73`). |
| P1 | llm_off/search_off; HF_HOME; train_grpo --dataset ... --probe --out ... --num-generations 4/2 --grad-accum 4/2 | ок | `scripts/train_grpo.py:232–258`: dataset обязательный Path, out str, оба числа int, probe flag ограничивает steps до 20, backend по умолчанию unsloth, vllm не включён. `day1.sh:297–305`: сохраняет PIPESTATUS[0], повторяет только после OOM и завершает ошибкой после второй неудачи. При успешной остановке служб GPU свободен под обучение. |
| P1 | cat probe.json; сообщение о ручном просмотре samples | ок / риск | `train_grpo.py:129,195`: пишет samples.jsonl и probe.json в выбранный out, имена совпадают. cat не проверен, остановки Docker подавляют ошибки (`day1.sh:60–65`): если контейнер реально не остановился, возможен OOM. |
| Z9 | tar artifacts/rl artifacts/runs/day1 runs artifacts/metrics; du; напечатанные scp/backup | ошибка | `day1.sh:314–320`: tar скрывает stderr и статус не проверяется; при отсутствии runs/metrics или ошибке диска печатается «готово» и создаётся done. В tar попадают фиксированные artifacts/metrics, хотя D4 поддерживает METRICS_DIR. scp с placeholders и backup — только инструкции, не исполняются. |

## Команды инструкции SERVER-DAY-1

| Строка / команда | Вывод | Основание |
|---|---|---|
| 39–40: cd рабочего дерева, upload.ps1 -ServerIp -WithCaches -WithLibrary -DataRoot | ок / риск | Все параметры объявлены (`deploy/upload.ps1:17–35`). ServerIp обязателен; ключ, User=root, Port=22 и RemoteDir имеют defaults. DataRoot относится к данным, не к коду. `<ip>` надо заменить. Отсутствующие codePaths пропускаются, не считаются ошибкой (`:71–73`). |
| 51: bootstrap.sh && restore.sh | ок / риск | bootstrap выполняет uv sync и готовит .env из примера (`bootstrap.sh:184–215`); restore поднимает службы, снимает embedded/graphed и восстанавливает MML из кэша (`restore.sh:75,94–108`). Таким образом утверждение «MML не переиндексируется» относится к библиотечным шагам, не восстановлению пустого сервера. Без перенесённых кэшей восстановление не эквивалентно прежнему корпусу. |
| 52–53: day1.sh --list / полный запуск | ок / ошибка | Синтаксис соответствует парсеру; обещание остановки при ошибках (`docs:3–5`) не соответствует day1.sh:20 и unchecked командам выше. |
| 70–75: скачать лист, локальный reward_agreement | риск | Требуются лист, key и вручную созданный grades JSON, а не только sheet. Формат/CLI подтверждены `scripts/reward_agreement.py:10,53–59`. |
| 102–103: --only D4 / --from L1 | ок / риск | Синтаксис верен. Результаты/метки/службы от предыдущего запуска не сбрасываются автоматически (`day1.sh:103–112`), поэтому это не гарантия воспроизводимого повторения. |
| 105–116: архив/backup и последующие действия | риск | Архив полезен только после проверки tar; инструкция не является автоматической проверкой полноты выгрузки. Серверные команды в ходе аудита не выполнялись. |

## Переменные и версии внешних инструментов

| Переменная day1 | Alias / тип / значение | Вывод |
|---|---|---|
| PDF_DIR | `config.py:88`, Path | Каталоги documents/library/{ru,ru-ocr,en} корректно ограничивают обнаружение PDF (`indexing/pipeline.py:352–358`). |
| QDRANT_COLLECTION | `config.py`, VectorStoreSettings, str | library_ru для двух русских каталогов, library_en для английского; имена допустимы. |
| QDRANT_SPARSE_LANGUAGE | `config.py:406`, str | russian / english поддерживаются BM25 fastembed, см. источник ниже. |
| MINERU_METHOD / MINERU_LANG | `config.py:124–125`, str / Literal[auto,txt,ocr] | ru=auto/east_slavic, ru-ocr=ocr/east_slavic, en=auto/en. `parsing/pdf_parser.py:89–103` передаёт их как `-l` и `-m` в pipeline backend. Подтверждены CLI MinerU 2.5.4. |
| CHUNKER_RESPECT_FORMULAS | `config.py:164`, bool | true допустимо; не выключает enrichment. |
| GRAPH_ENABLED / GRAPH_RETRIEVAL_ENABLED | `config.py:423,514`, bool | false допустимо; context не создаёт GraphStore/GraphRetriever. services up всё равно требует Neo4j и его пароль, поскольку поднимает весь набор (`deploy/services.sh:44–52`). |
| QA_SYSTEM_PROMPT / PROMPT_VERSION | `config.py:757,794`, str | Текст файла / v4 допустимы. Отсутствие файла cat отдельно не проверяет. |
| LLM_CONTEXT_WINDOW | `config.py:323`, int ≥512 | 16384 допустимо; `.env.example:109` задаёт SGLANG_MAX_MODEL_LEN=16384. Если старая .env не содержит эту переменную, compose default=8192 (`docker/docker-compose.vllm.yml:103`) расходится с клиентом. |
| LLM_MAX_CONCURRENCY | `config.py:326`, int 1–64 | 16 допустимо. |
| CONTEXT_NORMALIZE_MATH_DELIMITERS | `config.py:791`, bool | true допустимо; baseline должен явно задавать false. |
| EVAL_TRACE_RERANK_ALL | `config.py:697`, bool | true допустимо, полезно для полного trace. |
| HF_HOME, REPO_DIR, PATH, UV_DEFAULT_INDEX, SGLANG_MODEL/GPU_FRACTION | Shell/UV/Hugging Face/Compose, не Settings | Отсутствие alias не ошибка. model-swap меняет SGLANG_MODEL, SGLANG_GPU_FRACTION, LLM_MODEL (`deploy/model-swap.sh:59–61`). |

Внешние источники прочитаны через Python urllib, сохранены в `%TEMP%`; карточки — `%TEMP%/notes-016.md`:

1. [MinerU 2.5.4, CLI client.py](https://github.com/opendatalab/MinerU/blob/mineru-2.5.4-released/mineru/cli/client.py#L66): строки 66–67 содержат `en`, `east_slavic`; строка 40 — auto/txt/ocr. Сохранён `016-mineru-release-cli.py`. Значения корректны для этой версии, произвольную будущую версию это не гарантирует.
2. [fastembed, BM25](https://github.com/qdrant/fastembed/blob/main/fastembed/sparse/bm25.py#L29): english есть в перечне языков, это также default параметра language (строка 98); russian — в том же перечне. Сохранён `016-bm25.py`.
3. [MinerU, современная нормализация OCR-языков](https://github.com/opendatalab/MinerU/blob/master/mineru/model/ocr/language.py#L48): `en` сохранён как внутренний alias и нормализуется в ch (`:131–134`). Это не подтверждение прежнего публичного CLI современной версии. Сохранён `016-mineru-language.py`.
4. [Дерево MinerU master](https://api.github.com/repos/opendatalab/MinerU/git/trees/master?recursive=1), [теги](https://api.github.com/repos/opendatalab/MinerU/tags?per_page=100): сохранены `016-mineru-tree.json`, `016-mineru-tags.json`. В прочитанном master нет прежнего mineru/cli/client.py, на который ссылается наш parser. `pyproject.toml:53` разрешает любой mineru[core]>=2.0, uv.lock в рабочем дереве отсутствует: свежая установка требует закрепления проверенной версии. Это риск воспроизводимости L1, а не утверждение, что на будущем сервере уже стоит несовместимая версия. При поиске были ответы 404 для неверных путей/tag v2.5.4; вывод о допустимости основан на успешно прочитанном release-tag выше.

## Изоляция коллекций и манифест

- Новая коллекция создаётся автоматически в `_embed_and_store` → `ensure_collection` (`indexing/pipeline.py:152`, `stores/vector_store.py:125–154`). Имя берётся из текущих settings, запись идёт в него (`vector_store.py:208`). Библиотечные шаги не удаляют и не перезаписывают коллекцию MML; graph выключен.
- Sparse создаётся, **если** QDRANT_SPARSE_ENABLED=true и fastembed успешно загрузил encoder (`vector_store.py:91–121,132–139`). В `.env.example:161–163` он включён. Исключение при загрузке даёт fallback без sparse; существующая коллекция вообще возвращается без сверки схемы (`:128–130`). Следовательно «всегда создаст с sparse» утверждать нельзя. Нужна проверка фактической схемы и наличия sparse после L2.
- Манифест общий: artifacts/manifests/indexing_manifest.json (`pipeline.py:113`). `indexing/manifest.py:62–82` читает и сохраняет весь набор, поэтому новые книги не стирают отметки MML. При другом PDF_DIR обнаруживаются только PDF нового каталога. Новые doc_id получают новые состояния.
- Ключ doc_id — SHA от **basename**, не каталога (`pipeline.py:93–100`); fingerprint файла проверяется отдельно (`manifest.py:85–110`). Коллекция, язык sparse, метод OCR, параметры чанкинга не входят в ключ стадии. При переносе уже embedded книги в другую коллекцию она может быть пропущена (`pipeline.py:589–591`); одинаковые basename разных PDF создают коллизии и общие пути `*_chunks.json`. Изменение PDF_DIR само по себе не сбрасывает стадии. Это условный риск, а не доказательство коллизии текущих книг. Для нового комплекта с уникальными именами и свежими стадиями схема работает.
- Восстановление в docs отдельно сбрасывает embedded/graphed, но day1 не делает такой сброс перед библиотекой. `--only L2` не является восстановлением новой/очищенной коллекции при старом манифесте. Форсировать только выбранные стадии либо делать манифест зависимым от коллекции; не применять `--force` к MML parse.

## Службы по шагам

| Шаг | GPU / необходимые обращения |
|---|---|
| E0 | Установка и скачивание — без загрузки моделей; состояние служб унаследовано после restore. |
| D1, D4 | SGLang 4B + Infinity, возможная загруженная vision-модель Ollama. Нужен только генератор и локальный контекст; поиск поднят для дальнейших шагов. |
| G4 | Infinity + Qdrant + Neo4j. При always и без истории eval retrieval не требует LLM роутера. |
| L1 | MinerU + оставшийся Infinity; SGLang остановлен. Qdrant/Neo4j CPU не мешают. Остаточная VRAM Ollama зависит от прежних запросов. |
| L2 | Infinity + Qdrant; дополнительно vision для enrichment. В примере vision URL ошибочно ведёт к остановленному SGLang; запущенный Ollama сам по себе это не исправляет. |
| Q1 | SGLang 9B + Qdrant CPU. Embeddings/reranker/Neo4j кодом build при GRAPH_ENABLED=false не вызываются. |
| Q2 | Infinity + Qdrant. 4B поднимается, но retrieval always без истории не требует LLM. |
| R0 | 4B/Infinity ещё подняты; импорты/CUDA проверяются без загрузки обучаемых весов. |
| P1 | SGLang, Infinity, Ollama останавливаются; GPU предназначен для Unsloth. Ошибки stop скрыты. |
| Z9 | Только упаковка файлов. |

Точные пики памяти и время не устанавливаются чтением кода. Фракции .75/.85 и одновременное размещение моделей надо измерить на целевой карте.

## Импорты .venv-rl

Проверена цепочка: train_grpo → rewards.composite → evaluation.answers / rewards.formula / utils.text; train_grpo → rl.env → config / evaluation.goldset / generation.answering → retrieval.pipeline → clients, stores, router, fusion, diversity, graph_retriever. Также проверены модели, logging_setup, cache и retry.

- Немедленные сторонние зависимости этой цепочки — pydantic, pydantic-settings, httpx; морфология использует pymorphy3 и русский словарь. Они перечислены в requirements-rl-app.txt. dotenv покрыт явно; rich/tenacity также перечислены, но просмотренная retry-реализация сама использует stdlib.
- qdrant-client, fastembed и neo4j импортируются при создании/использовании реальных клиентов (`stores/vector_store.py:81,104,126`, `stores/graph_store.py:57`), а не при импорте rl.env. Награда и обучение по готовому JSONL эти методы не вызывают. Отсутствие этих пакетов в RL app requirements не ошибка.
- train() сначала импортирует unsloth, затем torch, datasets, trl (`train_grpo.py:115–123`); альтернативный HF loader — peft/transformers (`:219–220`). Все прямые пакеты перечислены в requirements-rl.txt. По умолчанию используется unsloth; vllm не требуется и отклоняется CLI (`:253–256`).
- Импортная полнота по коду соблюдена. Две установки pip не являются lock-файлом; работоспособность CUDA, транзитивных пакетов и совместимость закреплённых версий не доказаны этим аудитом. Пакеты не устанавливались. R0 проверяет часть импортов и награду, P1 — реальное обучение.

## Что переносит upload.ps1

| Необходимое | Откуда → куда | Вывод |
|---|---|---|
| requirements-rl.txt, requirements-rl-app.txt, prompts/qa-v4.txt | Рабочий каталог/deploy → сервер/deploy | В codePaths есть deploy (`upload.ps1:66–76`), копируется рекурсивно. Промпт существует в проверенном рабочем дереве. |
| capture/session-0819/trace-always.jsonl | **Рабочий каталог**, не DataRoot → сервер/capture | capture тоже codePaths; файл реально присутствует в рабочем дереве. Нельзя считать, что DataRoot выбирает источник capture. |
| artifacts/rl/*.jsonl и reward-fingerprint.json | DataRoot/artifacts/rl → сервер/artifacts/rl | WithLibrary копирует весь каталог (`:125–135`), значит включает оба вида файлов. Проверяется наличие каталога, не каждого необходимого файла; E0 затем проверяет mml-test и fingerprint. |
| documents/library, SHA256SUMS | DataRoot/documents/library → сервер/documents/library | Копируется полностью, удалённая проверка SHA256SUMS есть (`:138`). |
| artifacts/parsed, cache, manifests, evaluation/goldsets | DataRoot → те же серверные каталоги | Нужен WithCaches, в docs он передан (`:80–104`). Отсутствующий каталог пропускается; полноту MML надо проверить до D4. |
| MML PDF | DataRoot + стандартный относительный путь Pdfs → documents/pdf_docs | `:142–151`; отсутствующий PDF только печатает предупреждение. |
| Код, скрипты, pyproject, .env.example, docs, tasks | Рабочий каталог → сервер | `:66–76`; реальные секретные .env здесь не проверялись. |

При данном рабочем дереве необходимые trace/prompt и файлы deploy включены. Полнота содержимого DataRoot/artifacts/rl не подтверждалась чтением всех эпизодов; статически покрытие путей корректно. Утверждение о переносе относится к успешному scp, а не к выполненной загрузке.

## Что исправить

Ниже предложения для следующего изменения, **в этом аудите они не внесены**. Сначала обязательные исправления логики; далее проверки воспроизводимости. Строки указаны до изменений.

1. **deploy/day1.sh:44,157–177,184–193,199–201,208–209,237–242,250–264,315–320 — не отмечать проваленный шаг выполненным.** Было: производящие результаты команды/конвейеры без проверки статуса, затем `done_mark`. Стало: каждый такой конвейер заканчивается `|| die "<шаг>: <операция> завершилась ошибкой"`; проверка выполняется до чтения сводок и создания метки. Фильтр заменить на `clean() { grep -v -E ' INFO | WARNING |pymorphy|neo4j\.notifications'; local rc=$?; [ "$rc" -le 1 ]; }`, чтобы отсутствие строк не считалось аварией, а ошибка grep не подавлялась. В `ingest_group` проверять конвейер внутри функции. Не заменять механически `set -uo pipefail` на `set -euo pipefail`: P1 намеренно анализирует неуспешный запуск через PIPESTATUS и повторяет после OOM; эту ветку надо сохранить. Основание: таблица команд выше и `cli/main.py:286–287` (реальная ошибка ingest выдаёт exit 1).
2. **deploy/day1.sh:48–65 и места вызова ensure_4b/model_9b_alone — проверять переключение служб.** Было: `model-swap ...; services.sh up`, игнорируемые коды docker stop. Стало: `model-swap ... || die "не удалось переключить модель"`, затем `services.sh up || die "службы не готовы"`; после каждой команды stop — `|| die "не удалось освободить GPU"`, stderr сохранить в журнале. Для прямого model_9b_alone также останавливать шаг при ошибке. Иначе сообщение об остановке и последующая дорогая команда не подтверждены состоянием Docker.
3. **deploy/day1.sh:250–251 — передать полный корпус аудиту Q2.** Было: `goldset audit --path "$OUT/train-ru.json" --write ...`, автоматический выбор одного файла. Стало: перед командой собрать `$RUN/library-chunks.json` из всех файлов текущего parsed_dir и передать `--chunks "$RUN/library-chunks.json"`. Готовый проектный загрузчик — `rag_textbook.rl.env.load_chunks(Settings().paths.parsed_dir)` (`rl/env.py:45–53`); сериализовать список `chunk.model_dump(mode="json")` из его values. Проверить успех сборки и наличие всех gold chunk IDs. Объединение может включать MML: аудит ищет по ID, а изоляция train обеспечивается Q1/rl_dataset. Причина: `cli/main.py:632–638,662` выбирает только один файл, `evaluation/audit.py:126–132` отбрасывает вопросы с отсутствующими фрагментами. После исправления повторить Q2 даже при старой метке done.
4. **deploy/day1.sh:38,214,223–227 — явно обеспечить службы L1/L2 и адрес vision.** Было: L1 останавливает только SGLang; L2 не поднимает службы и наследует пустой LLM_VISION_BASE_URL. Стало: после `llm_off` на L1 выполнить проверенный `search_off`; в начале L2 выполнить проверенный `bash deploy/services.sh up`; в LIB_ENV добавить `LLM_VISION_BASE_URL=http://127.0.0.1:11434/v1`. Так MinerU не конкурирует с Infinity, L2 поднимает Infinity/Ollama при отдельном повторном запуске, а vision-запросы идут в предусмотренный `.env.example:115` qwen2.5vl через Ollama. Убедиться в успешной загрузке vision-модели до чанкинга. Альтернатива, требующая решения о составе корпуса: явно `CHUNKER_ENRICH_ENABLED=false`; молчаливое выпадение enrichment текущего варианта недопустимо. Основание: `config.py:370–385`, `chunking/enrichment.py:105–113`, `deploy/services.sh:48–67`.
5. **deploy/day1.sh:184 — зафиксировать контрольную ветку D4.** Было: baseline наследует CONTEXT_NORMALIZE_MATH_DELIMITERS. Стало: добавить `CONTEXT_NORMALIZE_MATH_DELIMITERS=false` в env первой команды; во второй оставить true. Иначе при соответствующей настройке .env сравниваются одинаковые режимы.
6. **deploy/day1.sh:119–124,223,241 — проверять входы и полноту, а не только существование отдельных файлов.** Было: до затрат не проверяются TRACE_MML/PROMPT_V4/requirements-rl-app; check_parsed_text проверяет найденные blocks, `test -s train-ru.json` принимает `[]`. Стало: E0 проверяет чтение перечисленных файлов и доступность чанков всех doc_id MML goldset; перед L2 сверяет ожидаемые PDF с успешно разобранными документами; после Q1 читает JSON, проверяет непустой список и схему goldset. Проверки должны завершаться `die` до done. Основание: `check_parsed_text.py:65–86`, `day1.sh:241`, `cli/main.py:1252–1272`.
7. **deploy/day1.sh:315–320 — не объявлять неудачный архив готовым.** Было: `tar ... 2>/dev/null`, без проверки статуса. Стало: сохранить stderr и проверять `tar ... || die "архив не создан полностью"`; обязательные результаты проверять заранее, необязательные существующие каталоги собирать в массив аргументов. Если METRICS_DIR переопределён, включать фактический путь Settings, как уже делает D4, вместо одного жёсткого artifacts/metrics. Размер и команду scp печатать только после успешного tar.
8. **docs/SERVER-DAY-1.md:70–75 — описать все входы ручной проверки.** Было: скачать только groups-4b-sheet.md, затем запустить команду с key и grades. Стало: скачать также groups-4b-key.json, не смотреть ключ до выставления слепых оценок; создать groups-4b-grades.json со словарём ID→оценка, например `{"1.1":3,"1.2":0}`, выполнить приведённую команду из каталога с этими двумя JSON (или передать полные пути). Основание: `sample_groups.py:107–108`, `reward_agreement.py:53–59`.
9. **docs/SERVER-DAY-1.md:62–63 — согласовать описание GPU со сценарием.** Было: L1 «MinerU», L2 «Infinity». Стало после пункта 4: L1 «MinerU; Infinity/Ollama остановлены», L2 «Infinity + Ollama vision при включённом enrichment». Пока пункт 4 не внесён, фактически на L1 остаётся Infinity, а vision L2 обращается к остановленному SGLang.

Проверки воспроизводимости и условные риски:

10. **pyproject.toml:53 — закрепить проверенную ветку MinerU.** Было: `mineru[core]>=2.0` при отсутствии uv.lock в дереве. Стало: конкретная проверенная версия 2.x и воспроизводимый lock, либо адаптация parser к новой CLI с отдельной проверкой. Точный production pin нельзя выбрать только этим аудитом; просмотренная 2.5.4 подтверждает нужные `-l/-m`, но не тестировалась здесь. Не считать наличие внутреннего alias en в master доказательством совместимости `python -m mineru.cli.client`.
11. **deploy/day1.sh:129–147,269–275 — привязать подготовку окружения к входам.** Было: старый rl-env.ok и непроверяемые фоновые hf download. Стало: маркер содержит хеш обоих requirements и версию Python; при несовпадении подготовка повторяется. PID фоновых скачиваний сохраняются; перед первым использованием соответствующих весов проверяется `wait` и журнал ошибки. Это исключает принятие старого маркера и непредсказуемое позднее скачивание на оплаченной карте.
12. **deploy/day1.sh:89–99,103–112 — проверять выбор шага и устаревание результатов.** Было: произвольные ONLY/FROM и метка-дата. Стало: до цикла отвергать ID вне PLAN и отсутствие значения опции; сохранять вместе с меткой идентификаторы входов/настроек. При изменении корпуса/коллекции/параметров явно инвалидировать зависимые шаги. Уже существующий `--only` подходит для принудительного повтора одного исправленного шага.
13. **rag_textbook/indexing/manifest.py:85–110, stores/vector_store.py:128–139 (пути относительно rag_textbook) — устранить условные риски повторной индексации.** Было: состояние embedded не зависит от collection, существующая схема не проверяется, ошибка sparse допускает dense-only fallback. Стало: учитывать collection и конфигурацию embedding/sparse в состоянии стадии; при обязательном sparse проверять схему и encoder с явной ошибкой. До такого изменения в сценарии проверять уникальность basename библиотечных PDF и фактические sparse-конфигурации library_ru/library_en после L2; при восстановлении пустой коллекции сбрасывать только нужные стадии соответствующих документов. Нельзя объявлять коллекцию готовой на основании старого embedded. Основания приведены в разделе изоляции.

## Проверки и ограничения

Выполнен доступный локальный прогон существующих тестов, без установки пакетов и без pytest cache в проекте:

```text
C:\python\rag_textbook\.venv\Scripts\python.exe -m pytest tests/test_goldset_audit.py tests/test_check_parsed_text.py tests/test_rl_env.py tests/test_rewards.py -q -p no:cacheprovider
......................................................                   [100%]
54 passed in 0.79s
```

При запуске задан PYTHONDONTWRITEBYTECODE=1. Новые тесты не добавлялись: менять разрешено только этот отчёт. Эти 54 теста проверяют отдельные компоненты и не подтверждают корректность day1.sh как целого; найденные ошибки интеграции ими не закрываются.

Не выполнены: запуск day1/bootstrap/restore/upload, Docker и CUDA, установка .venv-rl, скачивание весов, обучение, реальные обращения к серверу. Совместимость обучающего стека, расход VRAM, длительность и полнота будущего DataRoot остаются непроверенными. Реальные .env, секреты и запрещённые каталоги не читались. Сеть использована только для чтения перечисленных исходников; код и данные проекта наружу не отправлялись.

Изменён исполнителем только tasks/016-report.md. Исходные изменения рабочего дерева сохранены; при финальной проверке дополнительно обнаружено изменение docs/engineering-log.md другим процессом, к нему исполнитель не обращался с записью. Коммит не создавался. Исправления из списка выше остаются работой следующей задачи.
