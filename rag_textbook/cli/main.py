"""Единый CLI.

Прежде каждая операция была отдельным скриптом со своим разбором аргументов и
своей сборкой зависимостей, а часть инструментов вообще не версионировалась.
Здесь одна точка входа и общий контекст приложения.

    rag-textbook health                 проверить доступность зависимостей
    rag-textbook ingest                 проиндексировать корпус
    rag-textbook ask "вопрос"           задать вопрос
    rag-textbook goldset build          собрать эталонный набор
    rag-textbook eval run               измерить качество поиска
    rag-textbook eval ab                сравнить конфигурации (например, граф вкл/выкл)
    rag-textbook graph stats            статистика графа знаний
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag_textbook.config import Settings
from rag_textbook.context import build_context
from rag_textbook.evaluation.goldset import GoldsetBuilder, load_goldset, save_goldset
from rag_textbook.evaluation.runner import (
    run_ab_comparison,
    run_retrieval_evaluation,
    save_evaluation,
)
from rag_textbook.indexing.pipeline import IndexingPipeline
from rag_textbook.logging_setup import configure_logging

app = typer.Typer(help="RAG-ассистент по математической литературе", no_args_is_help=True)
eval_app = typer.Typer(help="Оценка качества", no_args_is_help=True)
goldset_app = typer.Typer(help="Эталонный набор вопросов", no_args_is_help=True)
graph_app = typer.Typer(help="Граф знаний", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
app.add_typer(goldset_app, name="goldset")
app.add_typer(graph_app, name="graph")

console = Console()


def _settings() -> Settings:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)
    return settings


@app.command()
def health() -> None:
    """Проверяет доступность хранилищ и моделей."""
    context = build_context(_settings())
    try:
        report = context.health()
    finally:
        context.close()

    table = Table(title="Состояние зависимостей")
    table.add_column("Компонент")
    table.add_column("Статус")
    table.add_column("Детали")
    for name, payload in report["components"].items():
        status = str(payload.get("status"))
        colour = {"ok": "green", "disabled": "yellow"}.get(status, "red")
        details = ", ".join(f"{key}={value}" for key, value in payload.items() if key != "status")
        table.add_row(name, f"[{colour}]{status}[/{colour}]", details[:80])
    console.print(table)
    if report["status"] == "error":
        raise typer.Exit(code=1)


@app.command()
def ingest(
    force: Annotated[bool, typer.Option("--force", help="Переиндексировать всё заново")] = False,
    source: Annotated[
        list[Path] | None, typer.Option("--source", help="Конкретные файлы вместо всего корпуса")
    ] = None,
    monitor: Annotated[
        bool, typer.Option("--monitor/--no-monitor", help="Собирать метрики ресурсов по стадиям")
    ] = True,
    monitor_interval: Annotated[float, typer.Option(help="Интервал замеров, с")] = 2.0,
    stages: Annotated[
        str,
        typer.Option(
            help=(
                "Какие стадии выполнить через запятую: parse,chunk,embed,graph. "
                "По умолчанию все. Нужно там, где карта одна: сервер инференса "
                "держит видеопамять постоянно и не помещается рядом с MinerU"
            )
        ),
    ] = "",
) -> None:
    """Индексирует корпус: разбор, чанкинг, векторы, граф."""
    from datetime import datetime

    from rag_textbook.observability.monitor import NullMonitor, ResourceMonitor

    settings = _settings()
    context = build_context(settings)
    run_dir = settings.paths.metrics_dir / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    resource_monitor = (
        ResourceMonitor(run_dir, interval_seconds=monitor_interval) if monitor else NullMonitor()
    )
    try:
        with resource_monitor:
            pipeline = IndexingPipeline(context, monitor=resource_monitor)
            report = pipeline.run(
                sources=source,
                force=force,
                stages=[item for item in stages.split(",") if item.strip()] or None,
            )
    finally:
        context.close()

    table = Table(title="Индексация")
    table.add_column("Документ")
    table.add_column("Статус")
    table.add_column("Чанки", justify="right")
    table.add_column("Стадии, с")
    for document in report.documents:
        stages = " ".join(f"{k}={v:.0f}" for k, v in document.stage_seconds.items())
        colour = {"ok": "green", "empty": "yellow"}.get(document.status, "red")
        table.add_row(
            document.doc_name[:40],
            f"[{colour}]{document.status}[/{colour}]",
            str(document.chunks),
            stages or "из кэша",
        )
    console.print(table)
    console.print(f"Всего чанков: [bold]{report.total_chunks}[/bold], ошибок: {report.failed}")
    if monitor:
        console.print(f"\nМетрики ресурсов: [dim]{run_dir}[/dim]")
        console.print(f"Разбор узких мест: [bold]rag-textbook bottlenecks {run_dir}[/bold]")
    if report.failed:
        raise typer.Exit(code=1)


@app.command("bottlenecks")
def bottlenecks(
    run_dir: Annotated[Path, typer.Argument(help="Каталог с замерами (monitor_*)")],
    as_json: Annotated[bool, typer.Option("--json", help="Печатать сырой отчёт")] = False,
) -> None:
    """Показывает, во что упирается каждая стадия конвейера.

    Отвечает на практический вопрос: что чинить первым и даст ли выигрыш
    конвейеризация стадий. Последнее зависит от того, упираются ли тяжёлые
    стадии в разные ресурсы — на одной карте «GPU плюс GPU» не ускорится.
    """
    import json as _json

    from rag_textbook.observability.analyze import analyze_run

    report = analyze_run(run_dir)
    if "error" in report:
        console.print(f"[red]{report['error']}[/red]")
        raise typer.Exit(code=1)

    if as_json:
        console.print_json(_json.dumps(report, ensure_ascii=False))
        return

    table = Table(title="Ресурсы по стадиям")
    table.add_column("Стадия")
    table.add_column("Время", justify="right")
    table.add_column("% прогона", justify="right")
    table.add_column("GPU, %", justify="right")
    table.add_column("простой GPU, %", justify="right")
    table.add_column("VRAM, МиБ", justify="right")
    table.add_column("CPU, %", justify="right")
    table.add_column("Узкое место")

    colours = {
        "gpu": "red",
        "cpu": "yellow",
        "disk": "yellow",
        "waiting": "bright_magenta",
        "mixed": "white",
        "unknown": "dim",
    }
    for stage in report["stages"]:
        colour = colours.get(stage["bottleneck"], "white")
        table.add_row(
            stage["stage"],
            f"{stage['duration_minutes']:.1f} мин",
            f"{stage['share_of_time_pct']:.0f}",
            _fmt(stage["gpu_util_mean_pct"]),
            _fmt(stage["gpu_idle_share_pct"]),
            _fmt(stage["gpu_mem_peak_mib"], 0),
            _fmt(stage["cpu_util_mean_pct"]),
            f"[{colour}]{stage['bottleneck']}[/{colour}]",
        )
    console.print(table)

    summary = report.get("summary") or {}
    if summary:
        console.rule("Вывод")
        for note in summary.get("notes", []):
            console.print(f"• {note}\n")
        pipelining = summary.get("pipelining_likely_useful")
        if pipelining is True:
            console.print(
                "[green]Тяжёлые стадии упираются в разные ресурсы — "
                "конвейеризация документов даст выигрыш.[/green]"
            )
        elif pipelining is False:
            console.print(
                "[yellow]Тяжёлые стадии упираются в один и тот же ресурс — "
                "конвейеризация выигрыша не даст, стадии поделят его по времени.[/yellow]"
            )

    for stage in report["stages"]:
        if stage["gpu_processes_peak_mib"]:
            console.print(f"\n[dim]Видеопамять по процессам на стадии «{stage['stage']}»:[/dim]")
            for name, memory in sorted(
                stage["gpu_processes_peak_mib"].items(), key=lambda x: -(x[1] or 0)
            ):
                console.print(f"  [dim]{name}: {memory:.0f} МиБ[/dim]")
            break


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Вопрос к материалам")],
    session: Annotated[str, typer.Option(help="Идентификатор сессии")] = "default",
    user: Annotated[str, typer.Option(help="Идентификатор пользователя")] = "cli",
    stateless: Annotated[bool, typer.Option("--stateless", help="Без истории")] = False,
    show_context: Annotated[bool, typer.Option("--show-context")] = False,
) -> None:
    """Задаёт вопрос и печатает ответ с цитатами."""
    context = build_context(_settings())
    try:
        history = (
            []
            if stateless
            else context.history.recent(user, session, context.settings.retrieval.max_history_turns)
        )
        answer = context.generator.answer(question, history)

        if not stateless:
            context.history.append(user, session, "user", question)
            context.history.append(user, session, "assistant", answer.answer)

        console.rule("Ответ")
        console.print(answer.answer)

        if answer.rewritten_question and answer.rewritten_question != question:
            console.print(f"\n[dim]Переформулированный вопрос: {answer.rewritten_question}[/dim]")

        if answer.citations:
            console.rule("Источники")
            for citation in answer.citations:
                marker = " [graph]" if citation.from_graph else ""
                console.print(f"  [{citation.index}] {citation.label}{marker}")

        if show_context:
            console.rule("Контекст")
            for index, item in enumerate(answer.contexts, start=1):
                channels = ",".join(item.channels)
                console.print(
                    f"[{index}] {item.chunk.citation_label()} "
                    f"(score={item.score:.4f}, каналы={channels})"
                )

        timings = " ".join(f"{k}={v:.0f}мс" for k, v in answer.timings_ms.items())
        console.print(f"\n[dim]{timings}[/dim]")
    finally:
        context.close()


@goldset_app.command("build")
def goldset_build(
    single: Annotated[int, typer.Option(help="Сколько одношаговых вопросов")] = 100,
    multihop: Annotated[int, typer.Option(help="Сколько многошаговых вопросов")] = 50,
    output: Annotated[Path | None, typer.Option(help="Куда сохранить")] = None,
) -> None:
    """Собирает эталонный набор из проиндексированных чанков."""
    settings = _settings()
    context = build_context(settings)
    try:
        chunks = list(context.vector_store.iter_chunks())
        if not chunks:
            console.print("[red]В хранилище нет чанков. Сначала выполните ingest.[/red]")
            raise typer.Exit(code=1)
        console.print(f"Доступно чанков: {len(chunks)}")
        # Граф передаётся, чтобы часть многошаговых пар отбиралась по связям,
        # а не по общим словам: на лексически похожих парах вклад графа
        # принципиально неизмерим.
        builder = GoldsetBuilder(context.llm, graph_store=context.graph_store)
        questions = builder.build(chunks, single_count=single, multihop_count=multihop)
        path = save_goldset(questions, output or settings.evaluation.goldset_path)
    finally:
        context.close()

    console.print(f"[green]Сохранено {len(questions)} вопросов в {path}[/green]")
    console.print(
        "[yellow]Набор сгенерирован моделью. Перед использованием как эталона "
        "просмотрите его вручную и проставьте verified=true.[/yellow]"
    )


@goldset_app.command("stats")
def goldset_stats(
    path: Annotated[Path | None, typer.Option(help="Путь к набору")] = None,
) -> None:
    """Показывает состав эталонного набора."""
    settings = _settings()
    questions = load_goldset(path or settings.evaluation.goldset_path)
    table = Table(title=f"Эталонный набор: {len(questions)} вопросов")
    table.add_column("Тип")
    table.add_column("Количество", justify="right")
    table.add_column("Проверено вручную", justify="right")
    by_type: dict[str, list] = {}
    for question in questions:
        by_type.setdefault(question.question_type, []).append(question)
    for qtype, items in sorted(by_type.items()):
        verified = sum(1 for item in items if item.verified)
        table.add_row(qtype, str(len(items)), str(verified))
    console.print(table)
    if len(questions) < 100:
        console.print(
            "[yellow]Меньше 100 вопросов: доверительный интервал шире типичного "
            "эффекта, различия между конфигурациями будут недостоверны.[/yellow]"
        )


@eval_app.command("run")
def eval_run(
    goldset: Annotated[Path | None, typer.Option(help="Путь к эталонному набору")] = None,
    label: Annotated[str, typer.Option(help="Метка прогона")] = "current",
) -> None:
    """Измеряет качество поиска на эталонном наборе."""
    settings = _settings()
    questions = load_goldset(goldset or settings.evaluation.goldset_path)
    context = build_context(settings)
    try:
        metrics, outcomes = run_retrieval_evaluation(
            context, questions, max_workers=settings.evaluation.max_concurrency
        )
        save_evaluation(metrics, outcomes, settings, label=label)
    finally:
        context.close()

    table = Table(title=f"Качество поиска ({label})")
    table.add_column("k", justify="right")
    for name in ("recall", "precision", "ndcg", "hit_rate"):
        table.add_column(name, justify="right")
    for k, values in sorted(metrics.per_k.items()):
        table.add_row(
            str(k),
            *[
                f"{values.get(name, 0.0):.3f}"
                for name in ("recall", "precision", "ndcg", "hit_rate")
            ],
        )
    console.print(table)
    console.print(f"MRR: [bold]{metrics.mrr:.3f}[/bold]")

    if metrics.by_type:
        type_table = Table(title="По типам вопросов")
        type_table.add_column("Тип")
        type_table.add_column("Вопросов", justify="right")
        type_table.add_column("recall", justify="right")
        type_table.add_column("ndcg", justify="right")
        for qtype, values in sorted(metrics.by_type.items()):
            type_table.add_row(
                qtype,
                str(int(values.get("questions", 0))),
                f"{values.get('recall', 0):.3f}",
                f"{values.get('ndcg', 0):.3f}",
            )
        console.print(type_table)

    routed = metrics.graph_usage.get("routed_to_graph", 0.0)
    share = metrics.graph_usage.get("avg_graph_share_in_context", 0.0)
    only = metrics.graph_usage.get("avg_graph_only_share", 0.0)
    console.print(
        f"Граф: маршрутизировано {routed:.1%} вопросов, "
        f"нашёл {share:.1%} контекста, "
        f"из них [bold]только он — {only:.1%}[/bold]"
    )
    if share > 0.05 and only <= 0.001:
        console.print(
            "[yellow]Графовый канал находит то же, что и векторный: "
            "исключительного вклада нет. Прирост качества от него невозможен "
            "при любых весах слияния.[/yellow]"
        )
    # Расхождение «канал включён, вопросы в него направлены, вклад нулевой»
    # означает поломку канала, а не отсутствие пользы от графа. Именно так
    # выглядел отказ обхода графа: запрос падал на каждом вопросе, метрики
    # считались по одному лишь векторному каналу, и вывод «граф не помогает»
    # получался про графовый канал, который не работал.
    if settings.graph.retrieval_enabled and routed > 0 and share <= 0.0:
        console.print(
            "[red]Внимание: графовый канал включён и получал вопросы, но не дал "
            "ни одного фрагмента. Это отказ канала, а не отсутствие эффекта — "
            "результаты сравнения с графом недействительны.[/red]"
        )


@eval_app.command("ab")
def eval_ab(
    goldset: Annotated[Path | None, typer.Option(help="Путь к эталонному набору")] = None,
    experiment: Annotated[
        str,
        typer.Option(
            help=(
                "Что сравниваем: graph | graph_seed | graph_seed_both | "
                "reranker | candidates | fusion"
            )
        ),
    ] = "graph",
) -> None:
    """Сравнивает две конфигурации на одном наборе вопросов."""
    settings = _settings()
    questions = load_goldset(goldset or settings.evaluation.goldset_path)

    experiments = {
        # Главный вопрос проекта: даёт ли графовый канал прирост.
        "graph": (
            {"graph.retrieval_enabled": False},
            {"graph.retrieval_enabled": True},
            ("без графа", "с графом"),
        ),
        # От чего отталкивается обход графа. Замер показал, что старт от
        # терминов вопроса вырождает канал в ослабленный лексический поиск:
        # он ищет по тому же сигналу, что и BM25. Старт от найденных
        # фрагментов — другой сигнал, и проверять надо именно его.
        "graph_seed": (
            {"graph.retrieval_enabled": True, "graph.seed_mode": "query"},
            {"graph.retrieval_enabled": True, "graph.seed_mode": "passages"},
            ("граф от вопроса", "граф от найденного"),
        ),
        "graph_seed_both": (
            {"graph.retrieval_enabled": True, "graph.seed_mode": "query"},
            {"graph.retrieval_enabled": True, "graph.seed_mode": "both"},
            ("граф от вопроса", "граф от обоих"),
        ),
        # Реранкер доказал крупный прирост, поэтому осмысленно проверить,
        # не упирается ли он в число кандидатов, которые ему подают.
        "candidates": (
            {"reranker.enabled": True, "reranker.candidates": 30},
            {"reranker.enabled": True, "reranker.candidates": 60},
            ("30 кандидатов", "60 кандидатов"),
        ),
        # Резерв мест в пуле кандидатов за находками одного лишь графа.
        # Целится в измеренный разрыв: канал находит 10-12 процентных пунктов
        # эталонного материала вне векторной выдачи, а до контекста доходит ноль.
        "graph_quota": (
            {"graph.retrieval_enabled": True, "retrieval.graph_candidate_quota": 0},
            {"graph.retrieval_enabled": True, "retrieval.graph_candidate_quota": 6},
            ("без резерва", "резерв 6 мест"),
        ),
        # Разложение связывающего вопроса на подвопросы. Целится в самую
        # крупную измеренную потерю: из 118 эталонных фрагментов многошаговых
        # вопросов в пул кандидатов попадают 103, а в выдачу — 68.
        "decompose": (
            {"retrieval.decompose_enabled": False},
            {"retrieval.decompose_enabled": True},
            ("без разложения", "с разложением"),
        ),
        # Схлопывание похожих фрагментов. На многошаговых вопросах эталонных
        # фрагментов два, и они по построению об одном и том же — есть риск,
        # что дедупликация выбрасывает второй как дубликат первого.
        "dedup": (
            {"retrieval.dedup_enabled": True},
            {"retrieval.dedup_enabled": False},
            ("со схлопыванием", "без схлопывания"),
        ),
        "reranker": (
            {"reranker.enabled": False},
            {"reranker.enabled": True},
            ("без реранкера", "с реранкером"),
        ),
        "fusion": (
            {"retrieval.fusion": "rrf"},
            {"retrieval.fusion": "dbsf"},
            ("rrf", "dbsf"),
        ),
    }
    if experiment not in experiments:
        console.print(f"[red]Неизвестный эксперимент: {experiment}[/red]")
        raise typer.Exit(code=1)

    baseline, candidate, labels = experiments[experiment]
    result = run_ab_comparison(
        questions, baseline, candidate, base_settings=settings, labels=labels
    )

    paired = result["paired"]
    table = Table(
        title=(
            f"A/B: {labels[0]} против {labels[1]} "
            f"(k={paired['k']}, вопросов {paired['questions']}, парное сравнение)"
        )
    )
    table.add_column("Метрика")
    table.add_column(labels[0], justify="right")
    table.add_column(labels[1], justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("95% интервал", justify="center")
    table.add_column("Лучше/хуже", justify="center")
    table.add_column("Значимо", justify="center")
    for name in ("recall", "precision", "ndcg", "hit_rate", "mrr"):
        row = paired["metrics"][name]
        delta = row["delta"]
        colour = "green" if delta > 0 else ("red" if delta < 0 else "white")
        table.add_row(
            name,
            f"{row['baseline']:.3f}",
            f"{row['candidate']:.3f}",
            f"[{colour}]{delta:+.3f}[/{colour}]",
            f"[{row['ci_low']:+.3f}; {row['ci_high']:+.3f}]",
            f"{row['improved']}/{row['worsened']}",
            "[bold]да[/bold]" if row["significant"] else "нет",
        )
    console.print(table)
    console.print(
        "Значимость — доверительный интервал среднего различия по парному "
        "бутстрэпу не покрывает ноль. «Лучше/хуже» — на скольких вопросах "
        "метрика выросла и упала."
    )
    if paired["questions"] < 100:
        console.print(
            "[yellow]Выборка меньше 100 вопросов: интервалы широкие, "
            "отсутствие значимости здесь не означает отсутствия эффекта.[/yellow]"
        )


@graph_app.command("stats")
def graph_stats() -> None:
    """Показывает статистику графа знаний."""
    context = build_context(_settings())
    try:
        if context.graph_store is None:
            console.print("[yellow]Граф отключён в конфигурации[/yellow]")
            raise typer.Exit(code=0)
        stats = context.graph_store.stats()
    finally:
        context.close()

    table = Table(title="Граф знаний")
    table.add_column("Метрика")
    table.add_column("Значение", justify="right")
    for name, value in stats.items():
        table.add_row(name, str(value))
    console.print(table)

    relates = stats.get("relates", 0)
    cooccurs = stats.get("cooccurs", 0)
    total = relates + cooccurs
    if total:
        share = 100.0 * relates / total
        console.print(f"Доля типизированных связей: [bold]{share:.1f}%[/bold]")
        if share < 20:
            console.print(
                "[yellow]Граф всё ещё состоит преимущественно из co-occurrence. "
                "Повысьте GRAPH_COOCCURRENCE_MIN_PMI или отключите канал.[/yellow]"
            )


@app.command("bench")
def bench(
    levels: Annotated[str, typer.Option(help="Уровни параллелизма через запятую")] = "1,2,4,8,16",
    requests: Annotated[int, typer.Option(help="Запросов на каждом уровне")] = 24,
    prompt_chars: Annotated[int, typer.Option(help="Длина промпта в символах")] = 3500,
    max_tokens: Annotated[int, typer.Option(help="Ограничение на длину ответа")] = 512,
    chunks: Annotated[int, typer.Option(help="Размер корпуса для пересчёта времени")] = 3533,
) -> None:
    """Меряет пропускную способность сервера инференса.

    Отвечает на два вопроса: какой параллелизм ставить в LLM_MAX_CONCURRENCY
    и оправдан ли переход на vLLM или SGLang вместо Ollama.
    """
    import json as _json
    from datetime import datetime

    from rag_textbook.evaluation.benchmark import estimate_indexing_time, run_throughput_sweep

    settings = _settings()
    context = build_context(settings)
    try:
        console.print(
            f"Движок: [bold]{settings.llm.base_url}[/bold], "
            f"модель: [bold]{settings.llm.model_for('extraction')}[/bold]\n"
        )
        result = run_throughput_sweep(
            context.llm,
            concurrency_levels=tuple(int(x) for x in levels.split(",") if x.strip()),
            requests_per_level=requests,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
        )
    finally:
        context.close()

    table = Table(title="Пропускная способность инференса")
    table.add_column("Параллелизм", justify="right")
    table.add_column("зап/с", justify="right")
    table.add_column("чанков/час", justify="right")
    table.add_column("p50, мс", justify="right")
    table.add_column("p95, мс", justify="right")
    table.add_column("ошибок", justify="right")
    for point in result["points"]:
        table.add_row(
            str(point["concurrency"]),
            f"{point['throughput_rps']:.2f}",
            f"{point['chunks_per_hour']:,}".replace(",", " "),
            f"{point['latency_p50_ms']:.0f}",
            f"{point['latency_p95_ms']:.0f}",
            str(point["failed"]),
        )
    console.print(table)

    estimate = estimate_indexing_time(chunks, result["best_throughput_rps"])
    console.print(
        f"\nЛучший параллелизм: [bold]{result['best_concurrency']}[/bold] "
        f"(ускорение против одного потока — [bold]{result['scaling_factor']}×[/bold])"
    )
    if estimate.get("hours") is not None:
        console.print(
            f"Экстракция графа для {chunks} чанков займёт [bold]{estimate['hours']} ч[/bold]"
        )

    # Коэффициент масштабирования — главный индикатор качества движка.
    if result["scaling_factor"] < 1.5:
        console.print(
            "\n[yellow]Параллелизм почти не помогает: движок обрабатывает запросы "
            "по очереди. Это основной довод в пользу vLLM или SGLang "
            "с непрерывным батчингом.[/yellow]"
        )
    elif result["scaling_factor"] < 3.0:
        console.print("\n[yellow]Умеренное масштабирование. Стоит сравнить с vLLM.[/yellow]")
    else:
        console.print("\n[green]Движок хорошо батчит запросы.[/green]")

    out_dir = settings.paths.metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bench_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "engine_base_url": settings.llm.base_url,
        "model": settings.llm.model_for("extraction"),
        **result,
        "indexing_estimate": estimate,
    }
    path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\nРезультат сохранён: {path}")


@app.command()
def config_dump() -> None:
    """Печатает эффективную конфигурацию (секреты скрыты)."""
    settings = _settings()
    console.print_json(
        json.dumps(settings.model_dump(mode="json"), ensure_ascii=False, default=str)
    )


def ingest_app() -> None:  # точка входа rag-ingest
    typer.run(ingest)


def query_app() -> None:  # точка входа rag-query
    typer.run(ask)


def eval_app_entry() -> None:  # точка входа rag-eval
    eval_app()


if __name__ == "__main__":
    app()
