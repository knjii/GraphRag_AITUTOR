# Отчёт по задаче 006

Дата проверки: 2026-09-16. Отчёт дополнялся по ходу проверки; итоговые выводы — в конце.

Изменяется только этот отчёт. Исходный обзор прочитан целиком; материалы `%TEMP%/lit003` и `%TEMP%/research004` доступны и используются повторно. Чужие изменения в рабочем дереве не затрагиваются. Исполняемый код проекта и тесты не меняются.

## 1. Сверка источников и утверждений

Разделяются точное соответствие источнику, авторская интерпретация и собственные проектные измерения: статья не может подтвердить последние.

### 1.1. Все 44 arXiv ID раздела 8

Сверены заголовки и ID сохранённых первоисточников, без повторной загрузки. Краткие названия методов в обзоре допустимы, даже когда название статьи другое. Для RL ниже закреплена фактически прочитанная версия, отсутствующая в большинстве ссылок обзора.

| ID / проверенная версия | Полное название | Вердикт о соответствии упомянутой работе | Где проверено |
|---|---|---|---|
| [2404.16130v2](https://arxiv.org/html/2404.16130v2) | From Local to Global: A GraphRAG Approach to Query-Focused Summarization | верно | `lit003/graphrag.html`, заголовок и ID HTML |
| [2405.14831v3](https://arxiv.org/html/2405.14831v3) | HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models | верно | `lit003/hipporag.html`, заголовок и ID HTML |
| [2502.14802v2](https://arxiv.org/html/2502.14802v2) | From RAG to Memory: Non-Parametric Continual Learning for Large Language Models | верно | `lit003/hipporag2.html`, заголовок и ID HTML |
| [2502.01113v3](https://arxiv.org/html/2502.01113v3) | GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation | верно | `lit003/gfmrag.html`, заголовок и ID HTML |
| [2410.05779v3](https://arxiv.org/html/2410.05779v3) | LightRAG: Simple and Fast Retrieval-Augmented Generation | верно | `lit003/lightrag.html`, заголовок и ID HTML |
| [2502.14902v2](https://arxiv.org/html/2502.14902v2) | PathRAG: Pruning Graph-Based Retrieval Augmented Generation with Relational Paths | верно | `lit003/pathrag.html`, заголовок и ID HTML |
| [2504.11544v1](https://arxiv.org/html/2504.11544v1) | NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes | верно | `lit003/noderag.html`, заголовок и ID HTML |
| [2501.06713v3](https://arxiv.org/html/2501.06713v3) | MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation | верно | `lit003/minirag.html`, заголовок и ID HTML |
| [2409.13731v3](https://arxiv.org/html/2409.13731v3) | KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation | верно | `lit003/kag.html`, заголовок и ID HTML |
| [2307.07697v6](https://arxiv.org/html/2307.07697v6) | Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph | верно | `lit003/tog.html`, заголовок и ID HTML |
| [2407.10805v7](https://arxiv.org/html/2407.10805v7) | Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation | верно | `lit003/tog2.html`, заголовок и ID HTML |
| [2401.18059v1](https://arxiv.org/html/2401.18059v1) | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | верно | `lit003/raptor.html`, заголовок и ID HTML |
| [2410.08815v2](https://arxiv.org/html/2410.08815v2) | StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization | верно | `lit003/structrag.html`, заголовок и ID HTML |
| [2402.07630v3](https://arxiv.org/html/2402.07630v3) | G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding andQuestion Answering | верно | `lit003/gretriever.html`, заголовок и ID HTML |
| [2501.13958v3](https://arxiv.org/html/2501.13958v3) | A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models | верно | `lit003/survey.html`, заголовок и ID HTML |
| [2502.11371v3](https://arxiv.org/html/2502.11371v3) | RAG vs. GraphRAG: A Systematic Evaluation and Key Insights | верно | `lit003/compare.html`, заголовок и ID HTML |
| [2506.05690v3](https://arxiv.org/html/2506.05690v3) | When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation | верно | `lit003/bench.html`, заголовок и ID HTML |
| [2604.09666v1](https://arxiv.org/html/2604.09666v1) | Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search Systems | верно | `lit003/agent2026.html`, заголовок и ID HTML |
| [2606.25656v1](https://arxiv.org/html/2606.25656v1) | Is GraphRAG Needed?From Basic RAG to Graph-/Agentic Solutions with Context Optimization | верно | `lit003/needed2026.html`, заголовок и ID HTML |
| [2509.16780v3](https://arxiv.org/html/2509.16780v3) | Comparing RAG and GraphRAG for Page-Level Retrieval Question Answering on a Math Textbook | верно | `lit003/mathtext.html`, заголовок и ID HTML |
| [2505.13406v1](https://arxiv.org/html/2505.13406v1) | AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database | верно | `lit003/mathkg.html`, заголовок и ID HTML |
| [2510.23637v2](https://arxiv.org/html/2510.23637v2) | Combining Textual and Structural Information for Premise Selection in Lean | верно | `lit003/premise.html`, заголовок и ID HTML |
| [2508.04162v2](https://arxiv.org/html/2508.04162v2) | SSEmb: A Joint Structural and Semantic Embedding Framework for Mathematical Formula Retrieval | верно | `lit003/formula.html`, заголовок и ID HTML |
| [2505.15585v1](https://arxiv.org/html/2505.15585v1) | MIRB: Mathematical Information Retrieval Benchmark | верно | `lit003/mirb.html`, заголовок и ID HTML |
| [2402.03300v1](https://arxiv.org/html/2402.03300v1) | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | верно | `research004/math.html`, заголовок и ID HTML |
| [2501.12948v1](https://arxiv.org/html/2501.12948v1) | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | верно | `research004/r1.html`, заголовок и ID HTML |
| [2503.20783v1](https://arxiv.org/html/2503.20783v1) | Understanding R1-Zero-Like Training: A Critical Perspective | верно | `research004/dr.html`, заголовок и ID HTML |
| [2503.14476v1](https://arxiv.org/html/2503.14476v1) | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | верно | `research004/dapo.html`, заголовок и ID HTML |
| [2503.09516v1](https://arxiv.org/html/2503.09516v1) | Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | верно | `research004/search.html`, заголовок и ID HTML |
| [2503.05592v1](https://arxiv.org/html/2503.05592v1) | R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning | верно | `research004/rsearch.html`, заголовок и ID HTML |
| [2503.19470v1](https://arxiv.org/html/2503.19470v1) | ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning | верно | `research004/research.html`, заголовок и ID HTML |
| [2505.04588v1](https://arxiv.org/html/2505.04588v1) | ZeroSearch: Incentivize the Search Capabilityof LLMs without Searching | верно | `research004/zero.html`, заголовок и ID HTML |
| [2503.00223v3](https://arxiv.org/html/2503.00223v3) | DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning | верно | `research004/deep3.html`, заголовок и ID HTML |
| [2505.14146v1](https://arxiv.org/html/2505.14146v1) | s3: You Don’t Need That Much Data to Train a Search Agent via RL | верно | `research004/s3.html`, заголовок и ID HTML |
| [2507.21892v1](https://arxiv.org/html/2507.21892v1) | Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning | верно | `research004/graph.html`, заголовок и ID HTML |
| [2507.23581v1](https://arxiv.org/html/2507.23581v1) | GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning | верно | `research004/graphrag.html`, заголовок и ID HTML |
| [2511.11770v1](https://arxiv.org/html/2511.11770v1) | Learning to Refine: An Agentic RL Approach for Iterative SPARQL Query Construction | верно | `research004/kg.html`, заголовок и ID HTML |
| [2501.07861v1](https://arxiv.org/html/2501.07861v1) | ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding | верно | `research004/rearter.html`, заголовок и ID HTML |
| [2507.02962v1](https://arxiv.org/html/2507.02962v1) | RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism | верно | `research004/rag.html`, заголовок и ID HTML |
| [2402.04315v1](https://arxiv.org/html/2402.04315v1) | Training Language Models to Generate Text with Citations via Fine-grained Rewards | верно | `research004/citation.html`, заголовок и ID HTML |
| [2506.10947v1](https://arxiv.org/html/2506.10947v1) | Spurious Rewards: Rethinking Training Signals in RLVR | верно | `research004/spurious.html`, заголовок и ID HTML |
| [2601.11061v1](https://arxiv.org/html/2601.11061v1) | Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs | верно | `research004/paradox.html`, заголовок и ID HTML |
| [2504.13837v1](https://arxiv.org/html/2504.13837v1) | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? | верно | `research004/limit.html`, заголовок и ID HTML |
| [2504.20834v1](https://arxiv.org/html/2504.20834v1) | Reinforcement Learning for LLM Reasoning Under Memory Constraints | верно | `research004/memory.html`, заголовок и ID HTML |

**Версионная оговорка:** 2504.20834v1 действительно называется Reinforcement Learning for LLM Reasoning Under Memory Constraints; сохранённая страница abstract уже показывает другое название последующей v4 — Token-Efficient RL for LLM Reasoning. Для вывода о неудаче full-token GRPO необходимо явно цитировать v1. Аналогично DeepRetrieval v3 имеет другое название, чем v1.

**Уже выявленное искажение:** утверждение §2.4, что 2402.04315 проверяет только ссылку, неверно: §2.2 вводит Correctness R1 (покрытие ключевой информации через EM/NLI), Citation Recall R2 (семантическая выводимость предложения из источника), Citation Precision R3. Это прямой предшественник составной награды за перенос содержания, хотя специального LaTeX-компонента там нет.

### 1.2. Числа и содержательные утверждения §2.2–2.3

«Верно» ниже означает соответствие указанному эксперименту, а не независимое воспроизведение результата. Баллы применимости 1–5, диапазон корпуса 1–5 тыс., выбор 4–9B и RTX 3090 — условия/экспертные оценки проекта, в статьях их нет.

| Утверждение обзора | Вердикт | Где и уточнение |
|---|---|---|
| HippoRAG 2: NQ 78.0; MuSiQue 74.7; 2Wiki 90.4; HotpotQA 96.3 | верно, каждое число | [2502.14802v2, табл. 3](https://arxiv.org/html/2502.14802v2#S4.T3), строка HippoRAG 2, passage recall@5; среднее 78.2 |
| HippoRAG 2: лучший средний результат, «не проседает на простых» | верно с ограничением | Та же таблица: NQ 78.0, PopQA 51.7; на PopQA прежний HippoRAG 53.8. Нельзя понимать как превосходство на каждом простом датасете. §3: passage/phrase nodes, PPR и triple filtering соответствуют описанию |
| GFM-RAG: 8 млн параметров, 60 графов | верно | [2502.01113v3](https://arxiv.org/html/2502.01113v3), abstract, §4.1 Implementation Details: 8M, 60 KGs, 14M triples, 700k документов; обучение на 8×A100 80GB, не на 3090 |
| GFM-RAG R@5: 87.1 / 58.2 / 95.6 | верно, каждое число | Там же, табл. 1: HotpotQA / MuSiQue / 2Wiki, строка GFM-RAG |
| HippoRAG: 2Wiki R@5 89.5 | верно | Та же табл. 1, именно **HippoRAG (Contriever)**; вариант ColBERTv2 имеет 89.1 |
| LightRAG: HotpotQA 54.7, BM25 72.2 | верно, оба числа | Та же табл. 1, R@5. Это сравнительный эксперимент авторов GFM-RAG, не числа из исходной статьи LightRAG |
| RAPTOR: 71.2 против ColBERTv2 79.3 | верно, оба числа | Та же табл. 1, HotpotQA R@5. «Только под обзорные вопросы» — рекомендация проекта, не доказанное ограничение RAPTOR |
| MS GraphRAG: около 47 тыс. токенов | верно с атрибуцией | [2509.16780v3](https://arxiv.org/html/2509.16780v3), §3.2/§4: ~47K против ~3.7K у top-5. Это конкретная конфигурация учебникового эксперимента, не универсальная стоимость MS GraphRAG |
| PathRAG: борьба с избыточностью; NodeRAG: разнородные узлы; MiniRAG: SLM | верно | [PathRAG §1, §3](https://arxiv.org/html/2502.14902v2), [NodeRAG §3](https://arxiv.org/html/2504.11544v1), [MiniRAG abstract/§3](https://arxiv.org/html/2501.06713v3) |
| KAG: взаимный индекс и logical forms; ToG/ToG-2: агентный KG-поиск; StructRAG: выбор структуры | верно | [KAG §2–3](https://arxiv.org/html/2409.13731v3), [ToG](https://arxiv.org/html/2307.07697v6), [ToG-2](https://arxiv.org/html/2407.10805v7), [StructRAG §3](https://arxiv.org/html/2410.08815v2). Стоимость на русском 4B не измерена этими источниками |
| AutoMathKG: Definition/Theorem/Problem и связи | верно | [2505.13406v1](https://arxiv.org/html/2505.13406v1), описание schema; не эксперимент с RL-наградой генератора |
| Lean: +25% к ReProver | верно с уточнением | [2510.23637v2](https://arxiv.org/html/2510.23637v2), abstract и результаты: **over 25%**, относительное улучшение retrieval metrics; не +25 п.п. успешных доказательств |
| SSEmb: +5 п.п. ARQMath-3 | верно с уточнением | [2508.04162v2](https://arxiv.org/html/2508.04162v2), abstract, §4, табл. 1: **более** 5 п.п. P′@10 и nDCG′@10 относительно embedding-based методов. Не выигрыш над любой системой |
| MIRB: утверждения, QA, посылки, формулы | верно | [2505.15585v1](https://arxiv.org/html/2505.15585v1), taxonomy benchmark; не доказательство эффективности нашего графа |
| DeepSeekMath: GRPO, 7B, проверяемый ответ | верно для outcome-варианта | [2402.03300v1](https://arxiv.org/html/2402.03300v1), §4; статья обсуждает также process supervision, описание обзора неполно |
| DeepSeek-R1: RL и rule-based rewards | верно | [2501.12948v1](https://arxiv.org/html/2501.12948v1), §2, accuracy/format rewards; не эксперимент с RAG |
| Dr. GRPO исправляет смещение к длинным ответам | верно с уточнением | [2503.20783v1](https://arxiv.org/html/2503.20783v1), анализ length/difficulty bias. Это изменение нормировки objective; само по себе не отдельная награда R_len и не гарантия краткости |
| DAPO: 32B, decoupled clip, dynamic sampling, verl | верно | [2503.14476v1](https://arxiv.org/html/2503.14476v1), abstract/§3–4, Qwen2.5-32B |
| Search-R1: 3B/7B, +20%/+41% к RAG | верно как приблизительный относительный прирост в v5; источник внутренне несогласован | [v5 табл. 2](https://arxiv.org/html/2503.09516v5): лучший 3B .325 против RAG .270 → **20.37%**; лучший 7B .431 против .304 → **41.78%**. Abstract страницы abs и введение дают 41%/20%, но abstract HTML и §4.4 дают **24%/20%**: 24% получается для 7B против rejection sampling .348. В сохранённом v1 указаны 26%/21%. Следует цитировать v5, таблицу и конкретный baseline, а не смешивать версии |
| R1-Searcher / ReSearch: двухэтапный RL, 7B / 32B | неверно как общее описание обеих работ | [2503.05592v1](https://arxiv.org/html/2503.05592v1): двухэтапный R1-Searcher, Qwen2.5-7B и Llama-3.1-8B. [2503.19470v1](https://arxiv.org/html/2503.19470v1): ReSearch 7B **и** 32B, RL с нуля; двухэтапность R1-Searcher на него переносить нельзя |
| ZeroSearch: поисковик имитируется моделью | верно | [2505.04588v1](https://arxiv.org/html/2505.04588v1), simulation LLM и curriculum |
| DeepRetrieval: recall-награда, 3B, нужна v3 | верно | [2503.00223v3](https://arxiv.org/html/2503.00223v3), abstract/метод: retrieval metrics как reward; Qwen2.5-3B. Не только перезапись текста, но и SQL |
| s3: frozen generator, Gain Beyond RAG, 2.4 тыс., 70×, 5×A100 | верно с уточнениями | [2505.14146v1](https://arxiv.org/html/2505.14146v1), §4.1, табл. 1 и 4, App. A.2: 20 шагов × batch 120 = 2400; 170k/2400≈70.8, не сравнение со всеми конкурентами. 5×A100 **80GB**, policy 7B, PPO; не опубликованный 4B GRPO-реранкер |
| Graph-R1: 1.5B/3B/7B, 4×A100, hypergraph | верно | [2507.21892v1](https://arxiv.org/html/2507.21892v1), §3–4, implementation: 4×A100 **80GB** |
| GraphRAG-R1: RL с process constraints | верно | [2507.23581v1](https://arxiv.org/html/2507.23581v1), метод; уже занята широкая постановка RL поверх графа |
| SPARQL: 3B, GRPO без SFT, +17.5 п.п. | верно с границами | [2511.11770v1](https://arxiv.org/html/2511.11770v1), abstract, результаты: 49.7% против 32.2% **post-entity-linking**, curated executable single-answer subset LC-QuAD 2.0; не полный benchmark |
| ReARTeR: PRM/preferences; RAG-R1: параллельные запросы | верно | [2501.07861v1](https://arxiv.org/html/2501.07861v1), [2507.02962v1](https://arxiv.org/html/2507.02962v1), методы; ReARTeR не следует записывать в outcome-GRPO |
| Citation rewards: корректность, citation recall/precision, малая модель | верно | [2402.04315v1](https://arxiv.org/html/2402.04315v1), §2.2–3: LLaMA-2-7B. Позднейшее утверждение «не перенос содержания» неверно |
| Spurious Rewards: +21.4 / +29.1 п.п.; у Llama/OLMo не так | верно с границами | [2506.10947v1](https://arxiv.org/html/2506.10947v1), abstract/§3: Qwen2.5-Math-7B на MATH-500; случайная против ground-truth. Это не результат на RAG и не измерение Qwen3.5 |
| Spurious Rewards Paradox: активация memorization shortcuts | верно как вывод авторов | [2601.11061v1](https://arxiv.org/html/2601.11061v1), результаты механистических вмешательств. Не универсальный механизм для любого семейства/набора |
| «RL не расширяет pass@k базы» | неверно как безусловный закон | [2504.13837v1](https://arxiv.org/html/2504.13837v1): эмпирическое ограничение исследованных моделей/задач и больших k; улучшение pass@1 допускается. Нельзя вывести невозможность новых способностей вообще |
| LoRA full GRPO не улучшил 1.5B на 40GB | верно для эксперимента v1 | [2504.20834v1](https://arxiv.org/html/2504.20834v1), §IV, табл. I–II: Qwen2-1.5B, partitioned A100 40GB, **float32**, LoRA только в части слоёв. Авторы прямо запрещают обобщение до «GRPO хуже»; это не нижняя оценка памяти для современного BF16 4B |
| Unsloth: до 17B на 15GB | верно как заявление документации | [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), What is GRPO; не универсальная гарантия BF16 LoRA и не замер Qwen3.5-4B |
| Qwen3.5 не поддерживается vLLM; требуется fast_inference=False | неверно на дату проверки | См. §4: самостоятельный vLLM поддерживает; Unsloth PR #10891 merged 2026-09-14 снимает запрет для dense Qwen3.5, но ограниченность smoke-теста важна |
| Графовые RL-работы использовали 4–5×A100 80GB | не нашёл подтверждения как обобщению | 4× подтверждено для Graph-R1; 5× в обзоре взято из **s3, не графовой работы**. Нельзя делать общую характеристику graph-RL из этих двух случаев |
| На 3090 реалистичны 4B+LoRA+короткие ответы | не нашёл прямого подтверждения для Qwen3.5-4B | Это гипотеза осуществимости; нужен измеренный training run, см. §4 |

### 1.3. Остальные разделы: границы фактов и проектных обещаний

| Утверждение | Вердикт | Где / причина |
|---|---|---|
| §2.1: RAG сильнее на простых, граф — на многошаговых; полезны selection/integration | верно как наблюдение benchmark | [2502.11371v3](https://arxiv.org/html/2502.11371v3), результаты QA и табл. 20; не гарантия для каждого графа/набора |
| §2.1: учебник, 477 QA, пять эмбеддеров; BM25 обходит два; ~47K/~3.7K; межглавный выигрыш не проверен | верно | [2509.16780v3](https://arxiv.org/html/2509.16780v3), §2–4, табл. 1–2. Результат относится к page-level задаче |
| §2.1: GraphRAG-Bench часто фиксирует проигрыш; четыре уровня задач | верно | [2506.05690v3](https://arxiv.org/html/2506.05690v3), abstract и taxonomy: факт, complex reasoning, summary, creative generation |
| §2.1: агентный поиск частично заменяет граф | верно; «остаётся ключевым» сильнее источника | [2604.09666v1](https://arxiv.org/html/2604.09666v1), abstract/результаты: gap сужается, у графа преимущество в сложных multi-hop и стабильности при амортизации offline cost. Не доказана необходимость графа |
| §2.1: retrieval-generation gap, затухание внимания и preferences | верно как интерпретация авторов | [2606.25656v1](https://arxiv.org/html/2606.25656v1), обсуждение gap; не эксперимент с русскими формулами. «Наши 0.30 найдены независимо» — аналогия проекта, не число из статьи |
| §0/1: внутри одной книги графу нечего добавить; граф нужен только межкнижно | неверно как общий вывод из литературы | Учебниковая статья показывает task-method mismatch на page lookup и сама допускает cross-concept пользу **внутри книги**. Междокументность не необходимое условие графовой пользы |
| §1: +0.018 recall@16, значимо | верно по журналу проекта | `docs/engineering-log.md:800`: .841→.859, CI [.006;.034]. «Весь за 8-м местом» не пересчитывался по сырым прогонам |
| §1: 56 из 100 вопросов без эталона в пуле | неверна единица счёта | `docs/engineering-log.md:1493`: **97 вопросов** с промахами, **56 отсутствующих фрагментов**, 44 присутствующих. 56/100 относится к фрагментам, не к вопросам |
| §1: пул .87–.95 / выдача ~.85 | неверно как сопоставление одной выборки | `docs/engineering-log.md:989`: пул .87–.95 **многошаговых** и выдача **.754**; .85 — агрегат другого среза. Запись :1500 прямо ограничивает старую гипотезу потерь после поиска |
| §1: MultiHop-RAG 609 документов, +.079; 93.5% общей сущности, .21 вклада | верно как цитирование проекта | `docs/engineering-log.md:1591`, `docs/HYPOTHESES.md:477–482`; независимого повторения этих прогонов не делал |
| §1: формулы .305/.395/.389; prompt .090 против context .019 | верно по журналу | `docs/engineering-log.md:2176–2179`, :1980. **27B — Qwen3.8-27B**, а не тот же Qwen3.5; сравнение не является чистой абляцией размера одного семейства |
| §1: семь опровергнутых графовых гипотез | верно как статус проекта | `docs/HYPOTHESES.md`, раздел «Уже опровергнутое»; не доказательство бесполезности любой архитектуры графа |
| §0/1/4: 388 вопросов, одна книга, 574 фрагмента | верно после уточнения | Read-only подсчёт `evaluation/goldsets/goldset.json`: **388 вопросов, 1 уникальный gold_doc_id, 574 вхождения gold_chunk_ids, 361 уникальный gold_chunk_id**. Называть 574 уникальными фрагментами неверно |
| §1: 0.19 мс / 5.4 мс / .002 мс; 364/388=.938; группы 226/270 | не нашёл первичный отчёт этих измерений | `tasks/005-report.md` в этом checkout отсутствует. Арифметика 364/388=.938144 верна; тайминги и компоненты связности заново не воспроизводились. Нельзя выдавать за независимо подтверждённые |
| §0/4: «чистого разбиения нет» из компоненты 270 | неверно как логическое следствие | Большая компонента ограничивает размер/репрезентативность split, но сама по себе не запрещает отделить остальные компоненты. Межкнижное обучение — разумный дизайн, а не доказанный единственный возможный split |
| §3: нынешний обход один шаг с rarity weighting | верно | `rag_textbook/evaluation/graph_offline.py:231`: один цикл по neighbours исходных сущностей, idf boost, нормировка по числу упоминаний; PPR там нет |
| §4.2: существующая formula reward — точная сохранность математических объектов | неверно в сильном смысле | `rag_textbook/evaluation/answers.py:205–238`: строковая нормализация и substring, только формулы длиной ≥12. Удаляются **в том числе точки и запятые**; определения/условия теорем не проверяются. Проверяемая строка ≠ математическая эквивалентность |
| §4.2: sentence_support проверяет опору | верно только как лексический proxy | `rag_textbook/evaluation/answers.py:285`: максимум доли общих слов с предложением контекста, порог .5, короткие предложения пропускаются. Это не NLI и не проверка истины; детерминированность не исключает ложной награды |
| §4.2: R_precision/R_lang/R_format/R_len и millisecond environment | не нашёл подтверждения для всей будущей награды | R_precision прямо обозначен новым; измерения старых функций не доказывают latency полной среды. Dr. GRPO меняет objective, не создаёт автоматически R_len. Поиск офлайн дешёв, но генерация rollout и независимый judge остаются дорогими |
| §4.4/6: тот же LoRA на 9B | неверно как обещание прямой переносимости | Адаптерные матрицы привязаны к архитектуре, размерам слоёв и base checkpoint. Перенос навыка/повтор обучения на 9B возможен как отдельный эксперимент; загрузка того же 4B адаптера не обоснована |
| §4.5: .36−.305=.055, больше пяти ×.010 | верно арифметически; статистическая сила не доказана | .055/.010=5.5. «Шум .010» не превращается автоматически в стандартную ошибку парного теста; bootstrap и контроль качества всё равно нужны |
| §4.6: 4B вдвое быстрее 9B при 2.1–2.9 с против 3.0 с | неверно | Из этих чисел ускорение **1.03–1.43×**, не 2×. Память в журнале 2.8/5.6GB действительно 2× для конкретных inference-конфигураций, не для GRPO |
| §3–7: критерии .03/.01/.05/.36, ≤10% hacking, корреляция ≥.6, объёмы 2–3k, 60→16, сроки/аренда | проектные решения, не факты литературы | Критерии, пороги и расписание ещё не результаты. Даты недель 21.09–15.11 согласованы с календарём 2026. Построчное сложение GPU-дней даёт **8–10**, строка общего бюджета говорит **9–11**; запас 1 день не обозначен |
| §8: TinyZero скачан неверно; LazyGraphRAG не проверен | верно для состояния материалов | `research004/tiny.html` — статья о passive soft inclusions, не TinyZero. LazyGraphRAG — блог, без arXiv ID; в таблицу 44 ID не включён. План проекта на эти источники не опирается |

Ссылки на локальные файлы выше служат указателями для исследователя; проектные данные наружу не отправлялись. Статусы «не нашёл» не заменены домыслами.

## 2. Поиск опровержений новизны

### 2.1. Метод поиска и границы вывода

Цель — найти контрпримеры, а не подтвердить исходную заявку. Через Python `urllib` прочитаны arXiv Atom API, полные HTML статей и первичные GitHub issues/PR. Старые статьи повторно не скачивались; новые материалы сохранены в `%TEMP%/research006`. Google вернул страницу переадресации, DuckDuckGo — challenge, Bing RSS — нерелевантные результаты; их выдача **не** используется как доказательство. Рабочий поиск — [arXiv API](https://export.arxiv.org/api/query?search_query=all:REARANK&max_results=10).

Выполнены запросы по `Rank-R1`, `REARANK`, RAG+reinforcement+faithfulness, retrieval+reinforcement+grounding, context+utilization+reinforcement; затем уточнения по полям `abs:`/`ti:`: formula/LaTeX/mathematical/textbook, copy/extract/utilization, citation/attribution, graph+reranking/context selection, Russian+RAG/retrieval+reinforcement/GRPO, random/spurious/format reward. Первые попытки с ошибочным полем `ab:` исключены; повторены с `abs:`. Запросы и ответы доступны в файлах `*_api.html`, `*2.html`, `russian_all.html`, `latex.html`. Нулевой результат неправильного запроса не учитывался.

Правильный formula-запрос вернул 54 записи, Russian в abstract — 0, Russian+RAG+reinforcement по all — 0, graph-selection — 4, spurious/control — 5. Для широких выдач просмотрены первые 60 результатов по relevance (copy/extract: 282 всего; citations: 66 всего), для LaTeX — первые 50 из 90. Фильтр 2024–2026 применялся при отборе, не к API: старые нерелевантные записи в выдаче есть. Это **не исчерпывающий обзор** всех конференций, русскоязычных журналов и репозиториев; отрицательные выводы ограничены этим поиском.

### 2.2. Близкие работы и степень перекрытия

«Полностью» относится только к явно названной широкой идее, а не ко всему проекту. Все строки, кроме специально обозначенной MoR, проверены по полному тексту (постановка, reward/method, ограничения); MoR — по abstract. Новые статьи не проверялись экспериментальным воспроизведением.

| Работа / год | Проверенная постановка и награда | Перекрытие замысла и отличие |
|---|---|---|
| [Fine-grained Citation Rewards, 2402.04315v1, 2024](https://arxiv.org/html/2402.04315v1) | §2.2: RL по покрытию key information через EM/NLI, entailment предложений и precision цитат; LLaMA-2-7B | **Частично №1; полностью общая идея «содержание + опора»**. Уже обучает перенос ключевой информации. Нет нормализованного LaTeX, русских учебников и random-reward RAG-контроля |
| [s3, 2505.14146v1, 2025](https://arxiv.org/html/2505.14146v1) | Поисковая policy, frozen generator, Gain Beyond RAG | **Частично №3**, а декомпозиция policy/environment и baseline-subtracted quality уже заняты. Действия — поиск, не списковая перестановка графового пула |
| [Rank-R1, 2503.06034v1, 2025](https://arxiv.org/html/2503.06034v1) | GRPO обучает setwise выбор релевантного документа, reward = правильный label и формат | **Частично №3**. Это не listwise permutation и не graph-specific reward; подробности §3 отчёта |
| [REARANK, 2505.20046v1, 2025](https://arxiv.org/html/2505.20046v1) | GRPO, полная перестановка, относительное улучшение NDCG@10 + format | **Полностью общий приём спискового RL-реранжирования; частично №3**. Нет графовой атрибуции выигрыша и формул; важный обязательный baseline |
| [R1-Ranker, 2506.21638v3, 2025](https://arxiv.org/html/2506.21638v3) | PPO, DRanker: весь список и MRR; IRanker: последовательное исключение отрицательных кандидатов | **Частично №3**. Уже обучаемый отбор/порядок малой моделью. Нет учебного графа и награды за формулы |
| [DynamicRAG, 2505.07233v2, 2025](https://arxiv.org/html/2505.07233v2) | §3: одновременно порядок и количество документов; SFT, затем DPO по качеству ответа генератора | **Почти полностью общая постановка №2 компонента; частично graph-specific №3 новизны**. Не GRPO, не специальный контроль графового вклада. Нельзя без оговорки приравнять реализацию DPO к online policy-gradient RL |
| [EviOmni, 2507.15586v7, 2025](https://arxiv.org/html/2507.15586v7) | §3: reasoning+extraction, knowledge-token masking, GRPO по answer F1, length, format | **Частично №1 и №3**. Учить использовать/извлекать найденный контекст уже предлагали. Не проверка сохранности формул конечного ответа |
| [RioRAG, 2505.20825v2, 2025](https://arxiv.org/html/2505.20825v2) | §3: nugget-centric проверка информативности с cross-source verification, RL для long-form RAG | **Частично №1**. Перенос проверенных единиц содержания и компромисс полноты/верности — занято; математическая нормализация не заявлена |
| [GRACE, 2601.04525v1, 2026](https://arxiv.org/html/2601.04525v1) | §3: gated format/path/content rewards, выбор ответа/отказа, Rouge-L F1 **извлечённого evidence** и ответа | **Частично №1 и №4**. Очень близкий extract-reward, проверка недостаточного evidence. Не normalized-LaTeX reward; evidence corruption/abstention не равны случайной награде |
| [RLFKV, 2602.05723v1, 2026](https://arxiv.org/html/2602.05723v1) | §3: atomic knowledge units, LLM-проверка каждой по документам, faithful+informative rewards; штраф за потерю информативности относительно базы | **Частично №1 и №4; полностью общая идея сохранности объектов знания с anti-hacking**. Финансовые данные, проверяющий Qwen3-32B, не дешёвая детерминированная LaTeX-проверка |
| [CRAFT, 2602.01348v3, 2026](https://arxiv.org/html/2602.01348v3) | §3: GRPO на этапе **post-retrieval**, структурированные traces, format/answer/citation и judge-faithfulness; policy 0.5B–7B | **Частично №1 и №4; полностью общая идея RL утилизации фиксированного контекста**. Есть reward/template ablations, но это не доказательство наличия random-reward контроля. Нет специфики русских формул |
| [CTRL-RAG, 2603.04406v1, 2026](https://arxiv.org/html/2603.04406v1) | §4: contrastive likelihood с supporting evidence и без него + внешняя correctness reward | **Частично №1 и №4**. Обучает зависимость ответа от evidence напрямую; не сравнение LaTeX и не random-reward training |
| [Faithful Industrial RAG, 2602.22584v1, 2026](https://arxiv.org/html/2602.22584v1) | §2: graph+hybrid retrieval, GRPO генератора; validity URLs проверяется по evidence либо approved prefix+HTTP, плюс faithfulness/style/safety | **Частично №1 и №3**. Проверяемая сохранность объектов в GraphRAG-ответе уже есть для URL. Графовый канал здесь не равен RL-обученному graph-aware реранкеру; язык китайский, область реклама |
| [Graph-R1, 2507.21892v1, 2025](https://arxiv.org/html/2507.21892v1) | End-to-end RL агентного поиска по гиперграфу | **Частично №3**, полностью широкая идея «RL над графом». Не выделяет восстановление graph-only находок в фиксированном гибридном пуле |
| [GraphRAG-R1, 2507.23581v1, 2025](https://arxiv.org/html/2507.23581v1) | Декомпозиция/инструменты, process-constrained RL | **Частично №3 и №4**. Process constraints существуют, но не заменяют placebo-reward ablation и отдельную атрибуцию прироста графу |
| [MoR, 2502.20317v4, 2025](https://arxiv.org/abs/2502.20317v4) | По abstract: Planning–Reasoning–Organizing, структурный+текстовый поиск, **rerank кандидатов по structural trajectory** | **Частично №3, важный контрпример к «графовые находки никто не поднимает»**. Полный HTML вернул HTTP 406; обучение RL не подтверждено. Не объявляю полным RL-предшественником |
| [Empirical Reasoning-Search RL, 2505.15117v1, 2025](https://arxiv.org/html/2505.15117v1) | §4: outcome-only против outcome+format, intermediate retrieval reward; §5: слабый и неинформативный поиск | **Частично №4**. Исследовать побочные источники выигрыша RL-RAG уже принято. В этой работе не нашёл именно обучения на случайной награде; random search engine — другая абляция |
| [Spurious Rewards, 2506.10947v1, 2025](https://arxiv.org/html/2506.10947v1) и [Paradox, 2601.11061v1, 2026](https://arxiv.org/html/2601.11061v1) | Random/format/incorrect-label controls, затем механизм memorization | **Полностью общая методика placebo-контроля; частично №4 в RAG**. Это математическое RLVR, не новый контроль, изобретённый текущим проектом |
| [ARVRE, 2606.15591v1, 2026](https://arxiv.org/html/2606.15591v1) | §II: граф цепочек уравнений, agentic RAG для topic phrases, SARSA по solvability/novelty и пользовательскому/LLM feedback | **Частично №1/№3**. Уже RL, retrieval и математические объекты вместе, но учится выбор цепочек для **создания задач по физике**, не перенос формул в ответе по учебнику |
| [Table2LaTeX-RL, 2509.17589v1, 2025](https://arxiv.org/html/2509.17589v1) | §4: VSGRPO, TEDS-Structure и CW-SSIM от отрендеренного LaTeX | **Частично широкий LaTeX-aware RL; нет полного перекрытия №1**. Это image-to-table reconstruction, не QA/RAG и не награда по эталонным математическим фрагментам |
| [Verifier Audit, 2609.01354v1, 2026](https://arxiv.org/html/2609.01354v1) | Категориальная проверка сбоев математических верификаторов, в том числе LaTeX/string paths | **Частично №4**, полезный контрпример к отождествлению детерминированности и правильности награды. Не обучение русского RAG |
| [ProRank, 2506.03487v3, 2025](https://arxiv.org/html/2506.03487v3) | GRPO warmup с format/relevance reward, затем fine-grained discriminative scoring | **Частично №3**, практичный малый baseline; pointwise, не listwise и не graph-specific |

Результат поиска: **более 10 близких работ проверено; полного совпадения узкой комбинации русский учебник + нормализованные формулы в RAG-ответе + placebo RL-контроли не найдено**. Это оставляет проверяемую узкую гипотезу, но не право объявлять новыми составную faithfulness reward, post-retrieval RL, списковой RL-отбор или сам random-reward контроль.

## 3. Списковое RL-реранжирование

| Работа | Действие и обучение | Награда | Размер и проверенные результаты |
|---|---|---|---|
| [Rank-R1](https://arxiv.org/html/2503.06034v1) | **Setwise**, выбирает наиболее релевантный ID из небольшой группы; GRPO, затем setwise sorting. Не прямая перестановка всего списка | 1 только если верны формат и label, иначе 0 | Qwen2.5-3B/7B/14B-Instruct. Табл. 1, nDCG@10 DL19/DL20: 3B GRPO **.713/.668**, 7B **.727/.685**, 14B **.714/.691**. Для 7B SFT .738/.692: RL не везде лучше. ~72k запросов = 18% от ~400k SFT; 4×H100, 3–5 дней |
| [REARANK](https://arxiv.org/html/2505.20046v1) | **Listwise**, reasoning + permutation списка; GRPO; 179 размеченных queries с многократным семплированием candidate sets | §3.3, eq. 5–6: `0.8*(NDCG_new-NDCG_init)/(NDCG_best-NDCG_init) + 0.1*format1 + 0.1*format2` | Qwen2.5-7B. Табл. 1: DL19 **74.16**, DL20 **70.00**, BEIR avg **54.59** nDCG@10; RankGPT-4 соответственно 75.59/70.56/55.84. «Comparable» не значит превосходство на каждом наборе. 179 — число исходных queries, не число RL-rollouts |
| [R1-Ranker](https://arxiv.org/html/2506.21638v3) | DRanker выдаёт список; IRanker по шагам исключает худшего, обратный порядок даёт ranking; PPO | DRanker: MRR + penalty за неправильный состав; IRanker: step-wise exclusion reward | Qwen2.5-3B-Instruct, сравнение с 7B. Табл. 2: IRanker-3B на Passage-5/7/9: **60.98/53.22/49.96 MRR**. Это искусственные пулы 5/7/9 с одним positive, не наш пул 60 и не TREC NDCG |
| [DynamicRAG](https://arxiv.org/html/2505.07233v2) | Отбирает **число и порядок** документов; behavior cloning, затем sampling trajectories и **DPO** по best/worst | §3.2.3: EM, BERTScore, ROUGE, `1/(1+len)`, LLM-eval; в App. C веса по 0.2 | LLaMA2-7B/13B, LLaMA3-8B; табл. 1/§4: 8B NQ **48.4 EM**. App. B.3: подключение к GPT-4o даёт +2.3/+0.8/+0.7 п.п. NQ/HotpotQA/ASQA. 8×A100 80GB. Не заявляю идентичность frozen-generator GRPO: реализация и настройки отличаются |
| [ProRank](https://arxiv.org/html/2506.03487v3) | **Pointwise** альтернатива: GRPO обучает бинарной релевантности и формату, затем fine-grained scoring | Format+accuracy в GRPO; затем дискриминативная стадия | Qwen 0.5B/1.5B; §4: fine-grained версии улучшают средний показатель относительно bge-gemma на 1.93/2.51 в процентной шкале авторов; не выдавать за отдельный эффект только RL. Полезен как дешёвый baseline |

Практический вывод для компонента №2: ближайшие обязательные сравнения — REARANK-подобный прирост NDCG и DynamicRAG-подобное downstream качество. Применение этих приёмов к графовому пулу само по себе слабая методическая новизна. Чтобы выделить graph-specific вклад, нужны matched pools/одинаковый token budget, сравнение с лучшим обычным реранкером и отдельный срез документов, которые **добавлены только графом**. Это предложение протокола, не найденный результат.

## 4. Малые модели и память

### 4.1. Qwen3.5: обнаружено изменение от 14 сентября 2026

**Категорическое «нет vLLM / обязательно fast_inference=False» устарело.** [Официальный recipe vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) содержит запуск Qwen3.5, а [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html) — поддерживаемые архитектуры. Это подтверждение инференса, не памяти RL.

В Unsloth [PR #10891](https://github.com/unslothai/unsloth/pull/10891) **слит 2026-09-14 06:09:42 UTC**, merge commit `f8cd9149eeecf200c492e6b00754f391cade7d8b`. Проверены API metadata, diff и авторский протокол: `qwen3_5` добавлен в `VLLM_SUPPORTED_VLM`. До изменения dense Qwen3.5 попадал в VLM gate даже при text-only и отклонялся.

Автор проверил **Qwen3.5-2B**, BF16 (`load_in_4bit=False`), **B200**, перенос LoRA в vLLM и три шага GRPO с ненулевыми градиентами. Это поддержка данного пути на main, **не** подтверждение релизной сборки, 4B на 3090, длительной устойчивости или скорости. QLoRA и `qwen3_5_moe` явно не покрыты. Для transformers 5.17 требуется также unsloth-zoo #1213; на ≤5.16 указана другая совместимость. Старую инструкцию Unsloth в `research004/qwen35.txt` нельзя считать актуальной после этого merge.

Сохранённая рекомендация BF16 имеет дополнительное ограничение: прямой запрет BitsandBytes 4-bit в guide относится к **MoE**, поэтому переносить его автоматически на dense 4B нельзя. Указанные там 10GB для BF16 LoRA 4B — fine-tuning estimate, не GRPO с групповой генерацией и KV-cache.

### 4.2. Что действительно измерено

| Первоисточник / кто | Модель, железо и настройки | Реальное наблюдение | Что это подтверждает |
|---|---|---|---|
| [Unsloth Memory Efficient RL, Experiments](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl), команда Unsloth | **Qwen3-4B**, T4 16GB, standby=True, vLLM util .95, num_generations=2, в строке grad_acc_steps=2 | **40 шагов / 40 минут**, **14.5 GiB**; при util .9 — 32 шага/40 мин, 13.8 GiB | Первичный пример успешного GRPO малого 4B. Это **Qwen3, T4**, не Qwen3.5/3090. ~60 и ~75 с/шаг вычислены из опубликованных totals; длина реальных completions и полный workload не зафиксированы этой таблицей. Имя первой конфигурации содержит ga1, поле — 2: ещё одна неоднозначность |
| [Unsloth issue #3771](https://github.com/unslothai/unsloth/issues/3771), пользователь `l-besiege-l` | **Qwen3-4B-Instruct-2507**, **RTX 4090 24GB**, standby; сравнение 4-bit/FP8 | Пользователь сообщает OOM 4-bit даже при context 2000, FP8 работает при 5000 | Прямой пользовательский запуск на 24GB, но **нет измеренных step/s или tokens/s**, нет полного независимого бенчмарка. Ответ maintainer «FP8 3× faster» не является замером этой конфигурации |
| [Комментарий maintainer к #3771](https://github.com/unslothai/unsloth/issues/3771#issuecomment-5556643787), Daniel Han-Chen | 4-bit + standby, 24GB | Сообщает об ограничении gpu_memory_utilization около .78 для устранения OOM | Старый OOM нельзя трактовать как вечную невозможность. Это сообщение об исправлении, не воспроизведение нами и не гарантированный release matrix |
| [Unsloth PR #5108](https://github.com/unslothai/unsloth/pull/5108), автор benchmark PR | Qwen3-4B-Base BF16 LoRA rank 32, **B200**, 20 GRPO steps, group=2, batch=2 | vLLM: mean **6.76 с/шаг**, peak **157.1GB**; transformers paged: **76.01 с/шаг**, **98.2GB** | Полезное прямое сравнение engines. Совершенно не 24GB run; generation-only 7224 tok/s из соседней таблицы нельзя выдавать за training throughput |
| [Unsloth PR #10891](https://github.com/unslothai/unsloth/pull/10891) | Qwen3.5-**2B**, B200, BF16+LoRA, 3 шага | Загрузка, adapter propagation и градиенты проверены автором; скорости нет | Поддержка интеграции после merge. Не замер Qwen3.5-4B на 24GB |
| [2504.20834v1](https://arxiv.org/html/2504.20834v1), авторы статьи | Qwen2-1.5B, FP32, partial LoRA, 40GB A100 partition | Положительный результат S-GRPO/T-SPMO и отрицательный full-token GRPO | Нельзя заключать, что 4B обязательно требует >40GB или что любой LoRA-GRPO неэффективен |

Поиск измерений включал arXiv GRPO/LoRA+24GB/3090/4090 и GitHub issues с 4B/GRPO/speed/s/it; просмотрены исходные тела результатов и relevant comments. **Полного первичного замера «Qwen3.5-4B + GRPO + LoRA + 24GB + скорость» не найдено. Для Qwen3-4B на 24GB есть сообщение о работоспособности, но надёжной скорости также не найдено.** Нельзя заполнять этот пробел скоростью inference, T4, B200, другой моделью или предположением о 3090 по 4090. Для бюджета спринтов нужны собственные warmup/steady-state замеры: версии, dtype, target modules/rank, input/output lengths, group/batch/accumulation, peak allocated/reserved, с/шаг и tokens/s rollout. GPU-прогон в этой задаче не запускался.

## Проверки и ограничения

Изменён только `tasks/006-report.md`: сверка 44 ID, таблицы проверок, поиск предшественников, сравнение RL-реранкеров, первичные данные о поддержке и памяти. Сеть использовалась только для чтения через Python; пакеты не устанавливались, скачанный исходный код не запускался. Коммитов нет. Все использованные первичные источники перечислены ссылками рядом с выводами; поисковые выдачи используются только для обнаружения работ.

Полные версии статей, полученные заново: Rank-R1 v1, REARANK v1, R1-Ranker v3, DynamicRAG v2, ProRank v3, EviOmni v7, RioRAG v2, GRACE v1, RLFKV v1, CRAFT v3, CTRL-RAG v1, Industrial RAG v1, Empirical Study v1, ARVRE v1, Table2LaTeX v1, Verifier Audit v1, Search-R1 v5. Для MoR прочитан только abstract v4. Остальные источники — ранее сохранённые материалы, версии зафиксированы в §1.1.

**Не сделано:** независимое воспроизведение экспериментов статей, GPU-бенчмарк 24GB, полный аудит сырых проектных метрик из отсутствующего отчёта 005, исчерпывающий обзор русскоязычных публикаций вне arXiv. Уверенное заявление «новизна доказана отсутствием аналога» невозможно. Эти ограничения не скрыты за статусом «верно».

Автоматические проверки отчёта и воспроизводимых чисел выполняются отдельным временным pytest-файлом в `%TEMP%/research006`. Они проверяют трассируемость/арифметику и не заменяют чтение статей. Первый запуск обнаружил ошибку **в новом проверочном скрипте**: граница слова после arXiv ID не учитывала суффикс `vN`, поэтому считалось только 20 неверсионированных ID. Исправлен regex временного теста; чужие тесты не менялись. Последние строки первого запуска:

```text
=========================== short test summary info ===========================
FAILED ..\..\Users\Максим\AppData\Local\Temp\codex-rag\research006\test_review_evidence.py::test_all_bibliography_ids_are_traced
1 failed, 10 passed in 1.03s
```

Повторный запуск после исправления проверочного regex:

```powershell
C:\python\rag_textbook\.venv\Scripts\python.exe -m pytest "$env:TEMP\research006\test_review_evidence.py" -q -p no:cacheprovider
```

Последние строки, как есть:

```text
...........                                                              [100%]
11 passed in 0.85s
```

Проверены: покрытие всех 44 ID; шесть целых строк исходной HTML-таблицы GFM-RAG; проценты Search-R1 с правильными baseline; 388/574/361/1 по goldset; merged metadata и изменение allowlist PR #10891; структура и UTF-8 отчёта. Кэш pytest в проекте отключён, временный тест сохранён в `%TEMP%/research006/test_review_evidence.py`. Продуктовые тесты не запускались: продуктовый код не менялся.

## Итог по четырём пунктам новизны

| Пункт §2.4 | Вывод | Аргумент и допустимая узкая формулировка |
|---|---|---|
| 1. RL-награда за перенос математических объектов/формул | **Новизна частично подтверждена** | Общая «сохранность содержания + grounding» уже есть в [2402.04315](https://arxiv.org/html/2402.04315v1), [RLFKV](https://arxiv.org/html/2602.05723v1), [GRACE](https://arxiv.org/html/2601.04525v1), а LaTeX-aware RL — в [Table2LaTeX](https://arxiv.org/html/2509.17589v1). Прямого reward по нормализованным формулам **в конечном учебном RAG-ответе** не найдено. Именно это, вместе с независимой проверкой смысловой корректности, остаётся кандидатом |
| 2. RL для русскоязычного учебного math-RAG | **Новизна частично подтверждена: найденного опровержения нет, приоритет не доказан** | У близких работ другой язык/домен; targeted Russian searches не дали результата. Это новизна постановки/данных, не нового алгоритма. До систематического поиска вне arXiv писать «первый в мире» нельзя |
| 3. Обучение отбора поднимать графовые находки | **Новизна частично подтверждена только в узкой graph-specific абляции** | [REARANK](https://arxiv.org/html/2505.20046v1), [DynamicRAG](https://arxiv.org/html/2505.07233v2), [s3](https://arxiv.org/html/2505.14146v1) занимают RL-отбор/перестановку/downstream reward; [MoR](https://arxiv.org/abs/2502.20317v4) уже реранжирует по structural trajectory. Широкая новизна опровергнута. Может остаться обучение и причинная оценка восстановления **graph-only** evidence относительно сильного hybrid-reranker при равном бюджете |
| 4. Контроли ложной награды в RL-RAG | **Новизна частично подтверждена только как применение протокола; методическая новизна опровергнута** | Random/format controls — известный протокол [Spurious Rewards](https://arxiv.org/html/2506.10947v1). В RAG уже есть [reward ablations](https://arxiv.org/html/2505.15117v1), [anti-hacking informativeness](https://arxiv.org/html/2602.05723v1), [faithfulness ablations](https://arxiv.org/html/2602.01348v3). Идентичного комплекта random-reward + format-only + independent judge именно для русского formula-RAG не найдено. «Большинство не проверяет» не подтверждено количественным обзором |

Объявлять все четыре направления пустыми нельзя. Защищаемая заявка — узкий эксперимент по сохранности математических формул в русском учебном RAG с проверкой смысла, контролями и сравнением с близкими подходами. Успех самого эксперимента пока не установлен.
