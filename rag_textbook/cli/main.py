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
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from rag_textbook.benchmarks.multihop_rag import build_goldset, load_corpus, original_type
from rag_textbook.benchmarks.text_corpus import chunk_documents
from rag_textbook.chunking.layout_chunker import LayoutAwareChunker
from rag_textbook.config import Settings
from rag_textbook.context import build_context
from rag_textbook.evaluation import graph_offline as graph_offline_eval
from rag_textbook.evaluation.ablation import (
    ablate_question,
    run_ablation,
    summarize_ablation,
)
from rag_textbook.evaluation.answers import run_answer_evaluation, save_answer_evaluation
from rag_textbook.evaluation.audit import audit_questions, summarize_audit
from rag_textbook.evaluation.goldset import (
    GoldsetBuilder,
    load_goldset,
    merge_goldsets,
    save_goldset,
)
from rag_textbook.evaluation.metrics import compare_paired, evaluate_retrieval, paired_bootstrap
from rag_textbook.evaluation.replay import fidelity_report, replay
from rag_textbook.evaluation.runner import (
    load_outcomes,
    run_ab_comparison,
    run_retrieval_evaluation,
    save_evaluation,
)
from rag_textbook.evaluation.trace import NotReplayable, TraceSet, align_to_snapshot
from rag_textbook.evaluation.verdicts import VerdictSet, apply_verdicts, summarize
from rag_textbook.indexing.pipeline import IndexingPipeline
from rag_textbook.logging_setup import configure_logging
from rag_textbook.models import Chunk

app = typer.Typer(help="RAG-ассистент по математической литературе", no_args_is_help=True)
eval_app = typer.Typer(help="Оценка качества", no_args_is_help=True)
goldset_app = typer.Typer(help="Эталонный набор вопросов", no_args_is_help=True)
graph_app = typer.Typer(help="Граф знаний", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
app.add_typer(goldset_app, name="goldset")
app.add_typer(graph_app, name="graph")

console = Console()


# Реестр A/B-экспериментов вынесен на уровень модуля намеренно: тест проверяет,
# что каждый путь настройки существует. Опечатка в пути означала бы сравнение
# конфигурации с самой собой — молча и совершенно правдоподобно.
AB_EXPERIMENTS: dict[str, tuple[dict[str, Any], dict[str, Any], tuple[str, str]]] = {
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
    # Гипотеза о реранкере как узком месте многошаговых вопросов.
    # Измерено на полном корпусе: реранкер поднимает вопросы с формулами
    # (0.915 → 0.966) и роняет многошаговые (0.703 → 0.576). Кросс-энкодер
    # оценивает каждый фрагмент против вопроса поодиночке, а второй фрагмент
    # пары сам по себе на вопрос не отвечает — он нужен вместе с первым.
    # Обязательная квота мест в итоговом контексте защищает находки графа
    # от такой оценки, не отменяя реранкер для остальных.
    "min_graph_docs": (
        {"graph.retrieval_enabled": True, "retrieval.min_graph_docs": 0},
        {"graph.retrieval_enabled": True, "retrieval.min_graph_docs": 3},
        ("без обязательной квоты", "три места за графом"),
    ),
    # Резерв в пуле кандидатов принят равным 6, но граф держит нужный
    # фрагмент в первых тридцати в 81% случаев — шести мест мало.
    "graph_quota_wide": (
        {"graph.retrieval_enabled": True, "retrieval.graph_candidate_quota": 6},
        {"graph.retrieval_enabled": True, "retrieval.graph_candidate_quota": 16},
        ("резерв 6 мест", "резерв 16 мест"),
    ),
    # Маршрутизатор — эвристика по ключевым словам со значением по умолчанию
    # «графа не нужно». Измерено: в граф уходят лишь 66.7% вопросов типа
    # graph_linked, и те же 33% заодно теряют расширенную выдачу
    # (top_k_linking применяется только при маршруте в граф).
    "router": (
        {"retrieval.router_mode": "heuristic"},
        {"retrieval.router_mode": "llm"},
        ("маршрут по эвристике", "маршрут моделью"),
    ),
    # Затухание веса соседа при обходе. Подобрано офлайн на локальном кэше:
    # MRR второго шага 0.225 → 0.253. Здесь проверяется на продукте.
    "hop_decay": (
        {"graph.retrieval_enabled": True, "graph.hop_decay": 0.5},
        {"graph.retrieval_enabled": True, "graph.hop_decay": 0.8},
        ("затухание 0.5", "затухание 0.8"),
    ),
    # Вклад сущности, взвешенный её редкостью. Офлайн: доля попаданий
    # второго фрагмента в первую восьмёрку 0.404 → 0.449.
    "graph_idf": (
        {"graph.retrieval_enabled": True, "graph.passage_idf_enabled": False},
        {"graph.retrieval_enabled": True, "graph.passage_idf_enabled": True},
        ("без веса редкости", "с весом редкости"),
    ),
    "fusion": (
        {"retrieval.fusion": "rrf"},
        {"retrieval.fusion": "dbsf"},
        ("rrf", "dbsf"),
    ),
}


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
    append: Annotated[
        bool,
        typer.Option(
            "--append",
            help=(
                "Дописать к существующему набору, а не перезаписать его. "
                "Сохраняет сравнимость с прежними прогонами и отметки verified"
            ),
        ),
    ] = False,
    seed: Annotated[
        int,
        typer.Option(
            help=(
                "Зерно отбора фрагментов. При дозаписи задавайте другое, "
                "иначе отберутся те же фрагменты и вопросы повторятся"
            )
        ),
    ] = 20260814,
    verify: Annotated[
        bool,
        typer.Option(
            "--verify",
            help=(
                "Приёмка абляцией: вопрос попадает в набор, только если ответ "
                "не получается по одному фрагменту. Втрое дороже по вызовам "
                "модели и настолько же честнее"
            ),
        ),
    ] = False,
) -> None:
    """Собирает эталонный набор из проиндексированных чанков."""
    settings = _settings()
    target = output or settings.evaluation.goldset_path

    existing: list = []
    if append:
        try:
            existing = load_goldset(target)
        except FileNotFoundError:
            console.print(f"[yellow]Набора {target} нет, создаю новый[/yellow]")
        else:
            console.print(f"Существующий набор: {len(existing)} вопросов")

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
        builder = GoldsetBuilder(context.llm, seed=seed, graph_store=context.graph_store)
        verifier = None
        if verify:
            by_id = {chunk.id: chunk for chunk in chunks}

            def verifier(question):  # noqa: ANN001, ANN202
                return ablate_question(context.llm, question, by_id).verdict

        produced = builder.build(
            chunks, single_count=single, multihop_count=multihop, verifier=verifier
        )
        if append:
            questions, appended = merge_goldsets(existing, produced)
            console.print(
                f"Сгенерировано {len(produced)}, добавлено новых {appended}, "
                f"повторов отброшено {len(produced) - appended}"
            )
        if verify:
            rejected = {
                name.removeprefix("вердикт:"): count
                for name, count in builder.failures.items()
                if name.startswith("вердикт:")
            }
            console.print(f"Приёмка абляцией: {rejected}")
        else:
            questions = produced
        path = save_goldset(questions, target)
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


@goldset_app.command("verdicts")
def goldset_verdicts(
    path: Annotated[Path | None, typer.Option(help="Путь к набору")] = None,
    verdicts: Annotated[
        Path, typer.Option(help="Файл вердиктов ручной проверки")
    ] = Path("evaluation/goldsets/verdicts.json"),
    apply: Annotated[
        bool, typer.Option("--apply", help="Записать отметки в набор")
    ] = False,
) -> None:
    """Применяет вердикты ручной проверки к эталонному набору.

    Вердикты живут отдельным файлом и применяются по идентификатору вопроса,
    поэтому переживают расширение набора: проверка идёт по локальной копии,
    а набор растёт на сервере.

    Без ``--apply`` только показывает, что получится.
    """
    settings = _settings()
    target = path or settings.evaluation.goldset_path
    questions = load_goldset(target)
    verdict_set = VerdictSet.load(verdicts)
    if not len(verdict_set):
        console.print(f"[yellow]В {verdicts} нет вердиктов.[/yellow]")
        raise typer.Exit(code=1)

    updated, counts = apply_verdicts(questions, verdict_set)
    summary = summarize(questions, verdict_set)

    table = Table(title=f"Вердикты: {sum(counts.values())} из {len(questions)} вопросов")
    table.add_column("Тип вопроса")
    table.add_column("Вердикт")
    table.add_column("Количество", justify="right")
    for question_type, bucket in sorted(summary.items()):
        for verdict, count in sorted(bucket.items(), key=lambda pair: -pair[1]):
            table.add_row(question_type, verdict, str(count))
    console.print(table)

    linked = summary.get("graph_linked", {})
    checked = sum(linked.values())
    if checked:
        share = linked.get("single_hop_enough", 0) / checked
        console.print(
            f"Связывающих вопросов, которым хватает одного фрагмента: "
            f"[bold]{share:.0%}[/bold] из {checked} проверенных"
        )

    if not apply:
        console.print("[dim]Показан предпросмотр. Для записи добавьте --apply.[/dim]")
        return

    save_goldset(updated, target)
    verified = sum(1 for item in updated if item.verified)
    console.print(f"Записано в {target}: проверенными помечены {verified} вопросов.")


def _load_chunks_file(path: Path) -> dict[str, Chunk]:
    """Читает фрагменты из выгрузки. Нужен офлайн, где хранилища нет."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("chunks", [])
    return {chunk.id: chunk for chunk in (Chunk.model_validate(item) for item in items)}


def _find_chunks_file(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    for directory in (Path("capture"), Path("artifacts/parsed")):
        found = next(directory.glob("*_chunks.json"), None) if directory.exists() else None
        if found is not None:
            return found
    return None


@goldset_app.command("audit")
def goldset_audit(
    path: Annotated[Path | None, typer.Option(help="Путь к набору")] = None,
    chunks: Annotated[
        Path | None, typer.Option(help="Выгрузка фрагментов; по умолчанию ищется сама")
    ] = None,
    write: Annotated[
        Path | None,
        typer.Option(help="Сохранить очищенный набор (вопросы с изъянами исключаются)"),
    ] = None,
) -> None:
    """Проверяет набор арифметикой, без модели и без сервера.

    Ищет то, что можно найти наверняка: ссылки на номера формул, эталонные
    фрагменты-оглавления, пустые ответы, повторы. Проверка «нужны ли оба
    фрагмента» сюда не входит — она требует модели, см. ``goldset label``.
    """
    settings = _settings()
    target = path or settings.evaluation.goldset_path
    questions = load_goldset(target)
    chunks_path = _find_chunks_file(chunks)
    corpus = _load_chunks_file(chunks_path) if chunks_path else None
    if corpus is None:
        console.print(
            "[yellow]Выгрузка фрагментов не найдена: проверки по фрагментам пропущены.[/yellow]"
        )

    audits = audit_questions(questions, corpus)
    summary = summarize_audit(audits)

    table = Table(title=f"Аудит набора: {summary['всего']} вопросов")
    table.add_column("Изъян")
    table.add_column("Вопросов", justify="right")
    table.add_column("Доля", justify="right")
    for defect, count in summary["изъяны"].items():
        table.add_row(defect, str(count), f"{count / summary['всего']:.1%}")
    console.print(table)

    by_type = Table(title="По типам вопросов")
    by_type.add_column("Тип")
    by_type.add_column("Вопросов", justify="right")
    by_type.add_column("Годных", justify="right")
    for name, bucket in summary["по типам"].items():
        total = bucket["вопросов"]
        good = bucket.get("годен", 0)
        by_type.add_row(name, str(total), f"{good} ({good / total:.0%})")
    console.print(by_type)
    console.print(f"Годных всего: [bold]{summary['годных']}[/bold] ({summary['доля годных']:.1%})")

    if write is None:
        console.print("[dim]Для сохранения очищенного набора добавьте --write путь.[/dim]")
        return

    usable = {item.question_id for item in audits if item.usable}
    cleaned = [item for item in questions if item.id in usable]
    save_goldset(cleaned, write)
    console.print(
        f"[green]Сохранено {len(cleaned)} вопросов в {write}.[/green] "
        f"Исключено {len(questions) - len(cleaned)}."
    )
    console.print(
        "[yellow]Прогоны по очищенному набору несравнимы с прежними напрямую: "
        "изменился состав вопросов. Сравнивайте только между собой.[/yellow]"
    )


@goldset_app.command("label")
def goldset_label(
    path: Annotated[Path | None, typer.Option(help="Путь к набору")] = None,
    verdicts: Annotated[
        Path, typer.Option(help="Файл вердиктов")
    ] = Path("evaluation/goldsets/verdicts.json"),
    limit: Annotated[int, typer.Option(help="Ограничить число вопросов")] = 0,
    only_linked: Annotated[
        bool,
        typer.Option("--only-linked", help="Только связывающие вопросы: изъян сосредоточен там"),
    ] = False,
    workers: Annotated[int, typer.Option(help="Параллельных вопросов")] = 4,
) -> None:
    """Размечает набор абляцией: нужны ли вопросу оба эталонных фрагмента.

    Требует модели, поэтому работает на сервере. Ручные вердикты не
    затираются: они остаются как есть, а совпадение с ними печатается —
    это и есть мера доверия к машинной разметке.
    """
    settings = _settings()
    target = path or settings.evaluation.goldset_path
    questions = load_goldset(target)
    if only_linked:
        questions = [item for item in questions if len(item.gold_chunk_ids) > 1]
    if limit:
        questions = questions[:limit]

    existing = VerdictSet.load(verdicts)
    human = dict(existing.verdicts)

    context = build_context(settings)
    try:
        corpus = {chunk.id: chunk for chunk in context.vector_store.iter_chunks()}
        if not corpus:
            console.print("[red]В хранилище нет фрагментов. Сначала выполните ingest.[/red]")
            raise typer.Exit(code=1)
        results = run_ablation(context.llm, questions, corpus, max_workers=workers)
    finally:
        context.close()

    summary = summarize_ablation(results)
    table = Table(title=f"Абляция: {len(results)} вопросов")
    table.add_column("Тип")
    table.add_column("Вердикт")
    table.add_column("Количество", justify="right")
    for question_type, bucket in sorted(summary["по типам"].items()):
        for verdict, count in sorted(bucket.items(), key=lambda pair: -pair[1]):
            table.add_row(question_type, verdict, str(count))
    console.print(table)
    if "доля одношаговых среди связывающих" in summary:
        console.print(
            "Связывающих вопросов, которым хватает одного фрагмента: "
            f"[bold]{summary['доля одношаговых среди связывающих']:.0%}[/bold]"
        )

    # Совпадение с ручной проверкой — единственная доступная мера доверия
    # к машинной разметке. Без неё её числа нечем поверить.
    overlap = [item for item in results if item.question_id in human]
    if overlap:
        agreed = sum(1 for item in overlap if human[item.question_id].verdict == item.verdict)
        console.print(
            f"Совпадение с ручной проверкой: [bold]{agreed}/{len(overlap)}[/bold] "
            f"({agreed / len(overlap):.0%})"
        )
    else:
        console.print("[yellow]Пересечения с ручной проверкой нет: доверие не измерено.[/yellow]")

    for item in results:
        if item.question_id not in human:
            existing.add(item.to_verdict())
    existing.save(verdicts)
    console.print(
        f"[green]Записано в {verdicts}: всего вердиктов {len(existing)}, "
        f"из них проверенных вручную {len(human)}.[/green]"
    )


# Коллекция по умолчанию. Считается один раз при загрузке модуля: если
# считать её внутри команды, значение придёт из того же окружения, что
# и рабочее, и защита от смешивания корпусов молча перестанет срабатывать.
_DEFAULT_COLLECTION = "textbook_chunks"


@eval_app.command("public")
def eval_public(
    corpus: Annotated[Path, typer.Option(help="corpus.json набора MultiHop-RAG")],
    questions: Annotated[Path, typer.Option(help="MultiHopRAG.json набора")],
    label: Annotated[str, typer.Option(help="Метка прогона")] = "multihop-rag",
    limit: Annotated[int, typer.Option(help="Ограничить число вопросов")] = 0,
    index: Annotated[
        bool, typer.Option("--index/--no-index", help="Проиндексировать корпус перед замером")
    ] = True,
    with_graph: Annotated[
        bool, typer.Option("--graph/--no-graph", help="Строить граф по чужому корпусу")
    ] = True,
) -> None:
    """Прогоняет конвейер на публичном наборе MultiHop-RAG.

    Зачем. Собственный набор отвечает, стало ли лучше, чем вчера; на вопрос
    «как это выглядит рядом с другими системами» он не отвечает никак.
    MultiHop-RAG размечен дословными цитатами, поэтому по нему считается
    полнота поиска — ровно та величина, вокруг которой идёт весь проект, —
    и его числа опубликованы.

    Корпус чужой: новостной и английский. Выводы на наш учебник не
    переносятся, это внешняя точка сравнения, а не замена своему набору.

    Индексировать нужно **в отдельную коллекцию**: иначе чужие документы
    смешаются с учебником и испортят все прежние замеры. Задайте
    QDRANT_COLLECTION и NEO4J_DATABASE перед запуском.
    """
    settings = _settings()
    console.print(
        f"Коллекция: [bold]{settings.vector_store.collection}[/bold], "
        f"база графа: [bold]{settings.graph.database}[/bold]"
    )
    if settings.vector_store.collection == _DEFAULT_COLLECTION and index:
        console.print(
            "[red]Коллекция та же, что у учебника. Чужой корпус смешается "
            "с ним и испортит все прежние замеры. Задайте QDRANT_COLLECTION.[/red]"
        )
        raise typer.Exit(code=1)

    documents = load_corpus(corpus)
    console.print(f"Документов в корпусе: {len(documents)}")

    context = build_context(settings)
    try:
        chunker = LayoutAwareChunker(settings.chunking)
        chunks = chunk_documents(documents, chunker, source_label=label)
        console.print(f"Чанков: {len(chunks)}")

        if index:
            pipeline = IndexingPipeline(context)
            result = pipeline.index_chunks(chunks, source_label=label, with_graph=with_graph)
            console.print(f"Проиндексировано: {result}")

        gold, mapping = build_goldset(questions, chunks, limit=limit)
        console.print(json.dumps(mapping.as_dict(), ensure_ascii=False, indent=2))
        if mapping.coverage < 0.9:
            console.print(
                "[yellow]Свидетельств сопоставилось меньше 90%: часть эталона "
                "потеряна, и число нельзя ставить рядом с опубликованными "
                "без этой оговорки.[/yellow]"
            )
        if not gold:
            console.print("[red]Ни одного вопроса не сопоставилось с корпусом.[/red]")
            raise typer.Exit(code=1)

        metrics, outcomes = run_retrieval_evaluation(context, gold)
        path = save_evaluation(metrics, outcomes, settings, label=label)
    finally:
        context.close()

    table = Table(title=f"MultiHop-RAG: {len(gold)} вопросов")
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

    # Разбивка по типам самого набора: опубликованные числа даны в ней,
    # а наш словарь типов с ней не совпадает.
    by_original: dict[str, list[Any]] = {}
    outcome_by_id = {item.question_id: item for item in outcomes}
    for question in gold:
        outcome = outcome_by_id.get(question.id)
        if outcome is not None:
            by_original.setdefault(original_type(question), []).append(outcome)
    if by_original:
        types = Table(title="По типам вопросов набора")
        types.add_column("Тип")
        types.add_column("Вопросов", justify="right")
        types.add_column("recall@k", justify="right")
        for name, items in sorted(by_original.items()):
            share = evaluate_retrieval(items, [settings.retrieval.top_k])
            types.add_row(
                name,
                str(len(items)),
                f"{share.per_k[settings.retrieval.top_k].get('recall', 0.0):.3f}",
            )
        console.print(types)

    # Участие графа обязано печататься рядом с метриками, а не лежать в JSON.
    # Без этого сравнение «с графом против без графа» выглядит осмысленным,
    # даже когда графовый канал получил считанные проценты вопросов, —
    # ровно так первый прогон на MultiHop-RAG дал «графа не видно»,
    # хотя маршрутизатор направил в граф 6.6% вопросов из 2255.
    routed = metrics.graph_usage.get("routed_to_graph", 0.0)
    share = metrics.graph_usage.get("avg_graph_share_in_context", 0.0)
    only = metrics.graph_usage.get("avg_graph_only_share", 0.0)
    console.print(
        f"Граф: маршрутизировано {routed:.1%} вопросов, "
        f"нашёл {share:.1%} контекста, из них только он — {only:.1%}"
    )
    if settings.graph.retrieval_enabled and routed < 0.5:
        console.print(
            f"[red]В граф направлено лишь {routed:.1%} вопросов. Сравнение "
            "с графом и без него на таком прогоне ничего не покажет: канал "
            "почти не участвовал. Эвристический маршрутизатор настроен "
            "на русские приметы и на чужом языке молчит — задайте "
            "RETRIEVAL_ROUTER_MODE=always.[/red]"
        )

    console.print(f"Метрики сохранены: {path}")


@eval_app.command("run")
def eval_run(
    goldset: Annotated[Path | None, typer.Option(help="Путь к эталонному набору")] = None,
    label: Annotated[str, typer.Option(help="Метка прогона")] = "current",
    trace: Annotated[
        Path | None,
        typer.Option(help="Куда сохранить слепок поиска для офлайн-перебора"),
    ] = None,
) -> None:
    """Измеряет качество поиска на эталонном наборе.

    С ``--trace`` дополнительно снимает слепок: кандидаты каналов, баллы
    реранкера и итоговый отбор по каждому вопросу. По слепку весь порядок
    и отбор пересчитываются офлайн бесплатно, без аренды сервера.
    """
    settings = _settings()
    questions = load_goldset(goldset or settings.evaluation.goldset_path)
    trace_set = TraceSet() if trace else None
    if trace and not settings.evaluation.trace_rerank_all:
        console.print(
            "[yellow]EVAL_TRACE_RERANK_ALL выключен: баллы реранкера сохранятся "
            "только по выдаче, и гипотезу о ширине окна проверить не выйдет.[/yellow]"
        )
    context = build_context(settings)
    try:
        metrics, outcomes = run_retrieval_evaluation(
            context,
            questions,
            max_workers=settings.evaluation.max_concurrency,
            trace=trace_set,
        )
        save_evaluation(metrics, outcomes, settings, label=label)
    finally:
        context.close()

    if trace and trace_set is not None:
        trace_set.save(trace)
        console.print(
            f"Слепок сохранён: {trace} ({len(trace_set.traces)} вопросов, "
            f"окно {trace_set.rerank_window})"
        )

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


# Сетка офлайн-перебора. Держится в одном месте намеренно: перебор «по месту»
# не оставляет следа, и потом невозможно сказать, что именно проверялось.
REPLAY_GRID: dict[str, list[dict[str, dict]]] = {
    "П1-разнообразие": [
        {"retrieval": {"diversity_mode": "reserve", "diversity_reserve_slots": 1}},
        {"retrieval": {"diversity_mode": "reserve", "diversity_reserve_slots": 2}},
        {"retrieval": {"diversity_mode": "reserve", "diversity_reserve_slots": 3}},
        {"retrieval": {"diversity_mode": "mmr", "diversity_lambda": 0.5}},
        {"retrieval": {"diversity_mode": "mmr", "diversity_lambda": 0.7}},
        {"retrieval": {"diversity_mode": "mmr", "diversity_lambda": 0.9}},
    ],
    "П2-реранкер": [
        {"reranker": {"mode": "by_route"}},
        {"reranker": {"mode": "blend", "blend_alpha": 0.3}},
        {"reranker": {"mode": "blend", "blend_alpha": 0.5}},
        {"reranker": {"mode": "blend", "blend_alpha": 0.7}},
        {"reranker": {"mode": "off"}},
    ],
    "П3-окно": [
        {"reranker": {"candidates": 45}},
        {"reranker": {"candidates": 60}},
        {"reranker": {"candidates": 100}},
    ],
    # Проверка следствия из замера на MultiHop-RAG: там весь прирост графа
    # лежал за восьмым местом, то есть граф находил нужное, а отбор его
    # прятал. Если то же верно на учебнике, резерв мест под графовые
    # фрагменты обязан поднять recall на малых k.
    "П8-резерв-графа": [
        {"retrieval": {"min_graph_docs": 1}},
        {"retrieval": {"min_graph_docs": 2}},
        {"retrieval": {"min_graph_docs": 3}},
        {"retrieval": {"min_graph_docs": 4}},
        {"retrieval": {"min_graph_docs": 2, "graph_candidate_quota": 12}},
        {"retrieval": {"min_graph_docs": 3, "graph_candidate_quota": 16}},
    ],
    "П4-слияние": [
        {"retrieval": {"rrf_k": 20}},
        {"retrieval": {"rrf_k": 100}},
        {"graph": {"weight": 0.2}},
        {"graph": {"weight": 0.6}},
        {"retrieval": {"dedup_similarity": 0.85}},
    ],
}


def _apply_overrides(settings: Settings, overrides: dict[str, dict]) -> Settings:
    updated = settings.model_copy(deep=True)
    for section, values in overrides.items():
        current = getattr(updated, section)
        setattr(updated, section, current.model_copy(update=values))
    return updated


def _describe(overrides: dict[str, dict]) -> str:
    parts = [
        f"{key}={value}"
        for section in overrides.values()
        for key, value in section.items()
    ]
    return ", ".join(parts) or "как есть"


@eval_app.command("replay")
def eval_replay(
    trace: Annotated[Path, typer.Argument(help="Слепок, снятый на сервере")],
    goldset: Annotated[Path | None, typer.Option(help="Путь к эталонному набору")] = None,
    chunks: Annotated[
        Path | None, typer.Option(help="Файл фрагментов; по умолчанию ищется в artifacts/parsed")
    ] = None,
    group: Annotated[str, typer.Option(help="Какую группу гипотез перебрать")] = "",
) -> None:
    """Пересчитывает отбор по слепку и перебирает настройки офлайн.

    Первое, что делается, — сверка честности: воспроизведение с рабочими
    настройками обязано повторить серверную выдачу. Если не повторяет, слепок
    неполон, и все выводы по нему недействительны — перебор не запускается.

    Проверяются только настройки, меняющие порядок и отбор. Попытка тронуть
    состав кандидатов прерывается с объяснением: такие гипотезы проверяются
    прогоном на сервере.
    """
    settings = _settings()
    traces = TraceSet.load(trace)
    if not traces.traces:
        console.print("[red]Слепок пуст.[/red]")
        raise typer.Exit(code=1)

    chunks_path = chunks or next(Path("artifacts/parsed").glob("*_chunks.json"), None)
    if chunks_path is None:
        console.print("[red]Не найден файл фрагментов.[/red]")
        raise typer.Exit(code=1)
    corpus = {
        item.id: item
        for item in (
            Chunk.model_validate(raw)
            for raw in json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        )
    }

    questions = load_goldset(goldset or settings.evaluation.goldset_path)
    gold = {item.id: item.gold_chunk_ids for item in questions}
    types = {item.id: item.question_type for item in questions}
    for item in traces.traces:
        item.question_type = item.question_type or types.get(item.question_id, "")

    # Поля состава кандидатов берутся из слепка: он и есть истина о том,
    # как эти кандидаты получены. Локальный .env знать этого не может —
    # на сервере значения могут отличаться от значений по умолчанию.
    # Выравнивание идёт по обоим снимкам: состав кандидатов задаёт, ЧТО
    # в слепке, а порядок — от какой точки отсчёта считать улучшения.
    # Без второго точкой отсчёта стала бы локальная конфигурация, а не та,
    # что дала измеренные на сервере числа.
    settings, aligned = align_to_snapshot(
        settings, {**traces.settings_snapshot, **traces.ordering_snapshot}
    )
    if aligned:
        console.print("[dim]Настройки приведены к слепку:[/dim]")
        for line in aligned:
            console.print(f"[dim]  {line}[/dim]")

    baseline = replay(traces, settings, corpus, gold)
    report = fidelity_report(traces, baseline)
    console.print(
        f"Сверка честности: точных совпадений "
        f"[bold]{report['доля точных совпадений']:.3f}[/bold], "
        f"среднее пересечение {report['среднее пересечение']:.3f} "
        f"на {int(report['вопросов сверено'])} вопросах"
    )
    if report["вопросов сверено"] and report["среднее пересечение"] < 0.99:
        console.print(
            "[red]Воспроизведение расходится с серверной выдачей. Слепок неполон: "
            "выводы по нему были бы недействительны. Перебор не запускается.[/red]"
        )
        raise typer.Exit(code=1)

    base_metrics = evaluate_retrieval(baseline, settings.evaluation.k_values)
    top_k = settings.retrieval.top_k
    console.print(f"Точка отсчёта: recall@{top_k} = {base_metrics.per_k[top_k]['recall']:.3f}")

    groups = {group: REPLAY_GRID[group]} if group else REPLAY_GRID
    for name, variants in groups.items():
        table = Table(title=name)
        table.add_column("Настройка")
        table.add_column(f"recall@{top_k}", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("связывающие", justify="right")
        # Счёт «лучше/хуже» важен сам по себе: прирост из +12 и −9 вопросов
        # требует другого решения, чем прирост из +3 и −0.
        table.add_column("лучше/хуже", justify="right")
        table.add_column("значимо", justify="right")
        for overrides in variants:
            try:
                candidate = _apply_overrides(settings, overrides)
                outcomes = replay(traces, candidate, corpus, gold)
            except (NotReplayable, ValueError) as error:
                table.add_row(_describe(overrides), "—", "—", "—", "—", str(error)[:40])
                continue
            metrics = evaluate_retrieval(outcomes, settings.evaluation.k_values)
            delta = metrics.per_k[top_k]["recall"] - base_metrics.per_k[top_k]["recall"]
            comparison = compare_paired(baseline, outcomes, k=top_k)
            recall_block = comparison["metrics"]["recall"]
            linked = metrics.by_type.get("graph_linked", {}).get("recall", 0.0)
            table.add_row(
                _describe(overrides),
                f"{metrics.per_k[top_k]['recall']:.3f}",
                f"{delta:+.3f}",
                f"{linked:.3f}",
                f"{recall_block['improved']}/{recall_block['worsened']}",
                "да" if recall_block["significant"] else "нет",
            )
        console.print(table)

    console.print(
        "[dim]Офлайн годится, чтобы отбрасывать. Выжившее подтверждается "
        "прогоном на сервере — порядок обратный уже стоил одной ошибки.[/dim]"
    )


@eval_app.command("answers")
def eval_answers(
    goldset: Annotated[Path | None, typer.Option(help="Путь к эталонному набору")] = None,
    label: Annotated[str, typer.Option(help="Метка прогона")] = "answers",
    limit: Annotated[int, typer.Option(help="Сколько вопросов взять, 0 — все")] = 0,
    verified_only: Annotated[
        bool,
        typer.Option("--verified-only", help="Только вопросы, вычитанные вручную"),
    ] = False,
    judge: Annotated[
        bool, typer.Option("--judge/--no-judge", help="Оценивать ответы моделью-судьёй")
    ] = True,
) -> None:
    """Измеряет качество ОТВЕТОВ, а не только поиска.

    Считаются две объективные величины — сохранность формул и доля выдумки, —
    и две судейские: верность и обоснованность. Судьёй работает та же модель,
    что и отвечает, поэтому судейские оценки годятся для сравнения
    конфигураций между собой, но не как абсолютная оценка качества.

    ``--verified-only`` берёт лишь вопросы, вычитанные человеком: на них
    числа означают то, что написано, а не свойство разметки.
    """
    settings = _settings()
    questions = load_goldset(goldset or settings.evaluation.goldset_path)
    if verified_only:
        questions = [item for item in questions if item.verified]
        if not questions:
            console.print(
                "[red]Проверенных вопросов нет. Примените вердикты: "
                "goldset verdicts --apply[/red]"
            )
            raise typer.Exit(code=1)
    if limit > 0:
        questions = questions[:limit]

    # Фрагменты выбираются по документам эталонного набора, а не «первым
    # попавшимся файлом». Именно так замер однажды посчитался по чужому
    # корпусу: рядом лежали 609 файлов публичного набора, glob вернул
    # новостную статью, эталонные фрагменты не нашлись — и сохранность
    # формул молча не посчиталась ни для одного вопроса.
    wanted = {doc_id for question in questions for doc_id in question.gold_doc_ids}
    corpus: dict[str, Chunk] = {}
    for path in sorted(Path(settings.paths.parsed_dir).glob("*_chunks.json")):
        if wanted and path.stem.removesuffix("_chunks") not in wanted:
            continue
        corpus.update(
            {
                item.id: item
                for item in (
                    Chunk.model_validate(raw)
                    for raw in json.loads(path.read_text(encoding="utf-8"))
                )
            }
        )

    # Молчаливый пропуск здесь недопустим: метрика сохранности формул просто
    # исчезнет из сводки, а прогон будет выглядеть удачным.
    covered = sum(
        1
        for question in questions
        if any(chunk_id in corpus for chunk_id in question.gold_chunk_ids)
    )
    if not corpus or covered < len(questions) // 2:
        console.print(
            f"[red]Фрагменты эталонного набора не найдены "
            f"({covered} из {len(questions)} вопросов). Сохранность формул "
            f"не посчитается. Проверьте {settings.paths.parsed_dir}.[/red]"
        )
        raise typer.Exit(code=1)

    context = build_context(settings)
    try:
        summary, outcomes = run_answer_evaluation(
            context,
            questions,
            chunks=corpus,
            judge=judge,
            max_workers=max(1, settings.evaluation.max_concurrency // 2),
        )
        path = save_answer_evaluation(
            summary, outcomes, settings.paths.metrics_dir, label=label
        )
    finally:
        context.close()

    table = Table(title=f"Качество ответов ({label}, {len(questions)} вопросов)")
    table.add_column("Показатель")
    table.add_column("Всего", justify="right")
    types = sorted(summary["по типам"])
    for name in types:
        table.add_column(name, justify="right")

    rows = ["отказов", "выдумка", "формулы дошли", "верность", "обоснованность"]
    for row in rows:
        values = [summary["всего"].get(row)]
        values += [summary["по типам"][name].get(row) for name in types]
        if all(value is None for value in values):
            continue
        table.add_row(
            row,
            *[f"{value:.3f}" if isinstance(value, (int, float)) else "—" for value in values],
        )
    console.print(table)

    if judge:
        console.print(
            "[yellow]Судьёй работает та же модель, что и отвечает: она склонна "
            "одобрять собственные ответы. Верность и обоснованность годятся "
            "для сравнения конфигураций, но не как абсолютная оценка.[/yellow]"
        )
    console.print(f"Сохранено: {path}")


@eval_app.command("ab")
def eval_ab(
    goldset: Annotated[Path | None, typer.Option(help="Путь к эталонному набору")] = None,
    experiment: Annotated[
        str,
        typer.Option(
            help=(
                "Что сравниваем: graph | graph_seed | graph_seed_both | "
                "graph_quota | graph_quota_wide | min_graph_docs | router | "
                "hop_decay | graph_idf | reranker | candidates | dedup | "
                "decompose | fusion"
            )
        ),
    ] = "graph",
) -> None:
    """Сравнивает две конфигурации на одном наборе вопросов."""
    settings = _settings()
    questions = load_goldset(goldset or settings.evaluation.goldset_path)

    experiments = AB_EXPERIMENTS

    if experiment not in experiments:
        console.print(f"[red]Неизвестный эксперимент: {experiment}[/red]")
        raise typer.Exit(code=1)

    baseline, candidate, labels = experiments[experiment]
    result = run_ab_comparison(
        questions, baseline, candidate, base_settings=settings, labels=labels
    )

    _render_paired(result["paired"], labels)


def _render_paired(paired: dict[str, Any], labels: tuple[str, str]) -> None:
    """Печатает парное сравнение: средние, разбор по типам, предупреждения.

    Вынесено из ``eval ab``, потому что ровно то же нужно при сравнении
    двух сохранённых прогонов — там, где конфигурации нельзя переключить
    на лету и приходится снимать два индекса подряд.
    """
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

    # Среднее по набору скрывает размен между типами вопросов. Реранкер поднял
    # вопросы с формулами на 5 пунктов и уронил многошаговые на 13 — по среднему
    # это выглядело чистым выигрышем. Разбор по типам печатается всегда.
    by_type = paired.get("by_type") or {}
    if len(by_type) > 1:
        breakdown = Table(title="Тот же результат по типам вопросов (recall)")
        breakdown.add_column("Тип")
        breakdown.add_column("Вопросов", justify="right")
        breakdown.add_column(labels[0], justify="right")
        breakdown.add_column(labels[1], justify="right")
        breakdown.add_column("Δ", justify="right")
        breakdown.add_column("95% интервал", justify="center")
        breakdown.add_column("Лучше/хуже", justify="center")
        opposite = False
        signs = set()
        for name, payload in by_type.items():
            row = payload["metrics"]["recall"]
            delta = row["delta"]
            colour = "green" if delta > 0 else ("red" if delta < 0 else "white")
            if abs(delta) > 1e-9:
                signs.add(delta > 0)
            breakdown.add_row(
                name,
                str(payload["questions"]),
                f"{row['baseline']:.3f}",
                f"{row['candidate']:.3f}",
                f"[{colour}]{delta:+.3f}[/{colour}]",
                f"[{row['ci_low']:+.3f}; {row['ci_high']:+.3f}]",
                f"{row['improved']}/{row['worsened']}",
            )
        opposite = len(signs) > 1
        console.print(breakdown)
        if opposite:
            console.print(
                "[yellow]Изменение помогает одним типам вопросов и вредит другим. "
                "Решение по среднему здесь будет неверным: нужна либо настройка "
                "по типу вопроса, либо явный выбор, чем жертвуем.[/yellow]"
            )

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


@eval_app.command("compare")
def eval_compare(
    baseline: Annotated[Path, typer.Argument(help="Файл прогона, взятого за точку отсчёта")],
    candidate: Annotated[Path, typer.Argument(help="Файл прогона, который проверяем")],
) -> None:
    """Парно сравнивает два сохранённых прогона по одним и тем же вопросам.

    Нужно там, где ``eval ab`` не работает. Он переключает настройки на лету
    и потому умеет сравнивать только то, что влияет на запрос. Настройки,
    влияющие на индекс — например, порог отсечения хабов, — требуют пересборки
    графа: две конфигурации не могут существовать одновременно. Остаётся снять
    два прогона подряд и сравнить их по файлам.
    """
    settings = _settings()
    for path in (baseline, candidate):
        if not path.exists():
            console.print(f"[red]Файл не найден: {path}[/red]")
            raise typer.Exit(code=1)

    base_label, base_outcomes = load_outcomes(baseline)
    cand_label, cand_outcomes = load_outcomes(candidate)
    if not base_outcomes or not cand_outcomes:
        console.print("[red]В одном из файлов нет результатов по вопросам[/red]")
        raise typer.Exit(code=1)

    retrieval = settings.retrieval
    k = max(retrieval.top_k, retrieval.top_k_linking)
    paired = compare_paired(base_outcomes, cand_outcomes, k)

    if paired["questions"] == 0:
        console.print(
            "[red]У прогонов нет общих вопросов: они сняты на разных наборах, "
            "сравнивать их нельзя[/red]"
        )
        raise typer.Exit(code=1)
    missing = len(base_outcomes) - paired["questions"]
    if missing:
        console.print(
            f"[yellow]Общих вопросов {paired['questions']}, в точке отсчёта было "
            f"{len(base_outcomes)}: сравнение идёт только по пересечению[/yellow]"
        )

    _render_paired(paired, (base_label, cand_label))


@graph_app.command("drop")
def graph_drop(
    yes: Annotated[
        bool, typer.Option("--yes", help="Подтверждение: операция необратима на месте")
    ] = False,
) -> None:
    """Удаляет граф целиком.

    Neo4j Community держит одну базу, поэтому измерить граф на публичном
    наборе можно, только сняв граф учебника. Потеря восполнима: граф
    учебника пересобирается из кэша извлечения за минуты и без обращений
    к модели — командой ``ingest --stages graph`` после
    ``deploy/reset-stages.sh graphed``.
    """
    settings = _settings()
    context = build_context(settings)
    try:
        store = context.graph_store
        if store is None or not store.verify():
            console.print("[red]Граф недоступен.[/red]")
            raise typer.Exit(code=1)
        before = store.stats()
        console.print(f"Сейчас в графе: {before}")
        if not yes:
            console.print("[yellow]Показан предпросмотр. Для удаления добавьте --yes.[/yellow]")
            return
        result = store.clear()
        console.print(f"[green]Удалено узлов: {result['nodes']}[/green]")
    finally:
        context.close()


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


@graph_app.command("offline")
def graph_offline(
    degrees: Annotated[
        str, typer.Option(help="Пороги отсечения хабов через запятую")
    ] = "64,48,40,32,24",
    hops: Annotated[str, typer.Option(help="Затухание веса соседа через запятую")] = "0.0,0.5,0.8",
    model: Annotated[
        str, typer.Option(help="Модель, которой собран кэш (иначе берётся из настроек)")
    ] = "",
    reasoning_effort: Annotated[
        str, typer.Option(help="Режим размышления, которым собран кэш")
    ] = "",
    save: Annotated[bool, typer.Option(help="Сохранить результат в artifacts/metrics")] = True,
) -> None:
    """Перебирает настройки графового канала по локальному кэшу, без сервера.

    Мера — место второго фрагмента пары при известном первом. Разбор промахов
    показал, что все неудачи на многошаговых вопросах устроены одинаково:
    один фрагмент найден, второй нет. Здесь меряется ровно этот переход,
    поэтому перебор стоит секунды и не требует ни Neo4j, ни видеокарты.
    """
    settings = _settings()
    overrides = {
        "model": model or None,
        "reasoning_effort": reasoning_effort or None,
    }
    try:
        probe = graph_offline_eval.reconstruct(settings, max_entity_degree=0, **overrides)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    pairs = graph_offline_eval.linked_pairs(settings, probe)
    if not pairs:
        console.print(
            "[yellow]В эталонном наборе нет вопросов с двумя фрагментами — "
            "мерить переход не на чем[/yellow]"
        )
        raise typer.Exit(code=0)

    console.print(
        f"Фрагментов {probe.passages}, найдено в кэше {probe.cache_hits}, "
        f"пар для замера {len(pairs)} (по два измерения на пару)"
    )

    degree_values = [int(value) for value in degrees.split(",") if value.strip()]
    hop_values = [float(value) for value in hops.split(",") if value.strip()]

    baseline_ranks: list[int] | None = None
    rows: list[dict[str, Any]] = []
    table = Table(title="Графовый канал: поиск второго фрагмента пары")
    for column in ("порог", "затухание", "IDF", "сущностей", "рёбер", "MRR", "hit@8", "hit@30"):
        table.add_column(column, justify="right" if column != "IDF" else "center")

    for degree in degree_values:
        graph = graph_offline_eval.reconstruct(
            settings, max_entity_degree=degree, **overrides
        )
        for hop_decay in hop_values:
            for use_idf in (False, True):
                ranks = graph_offline_eval.second_hop_ranks(
                    graph, pairs, hop_decay=hop_decay, use_idf=use_idf
                )
                summary = graph_offline_eval.summarize(ranks)
                # Точка отсчёта — то, что стояло до замеров: порог 64,
                # затухание 0.5, без IDF. Всё остальное сравнивается с ней.
                if degree == 64 and hop_decay == 0.5 and not use_idf:
                    baseline_ranks = ranks
                rows.append({"degree": degree, "hop_decay": hop_decay, "idf": use_idf,
                             "entities": graph.entities, "edges": graph.edges, **summary})
                table.add_row(
                    str(degree), f"{hop_decay:.2f}", "да" if use_idf else "нет",
                    str(graph.entities), str(graph.edges),
                    f"{summary['mrr']:.3f}", f"{summary['hit@8']:.3f}",
                    f"{summary['hit@30']:.3f}",
                )
    console.print(table)

    best = max(rows, key=lambda row: row["mrr"])
    console.print(
        f"Лучшее: порог [bold]{best['degree']}[/bold], затухание "
        f"[bold]{best['hop_decay']}[/bold], IDF [bold]{'да' if best['idf'] else 'нет'}[/bold]"
    )
    if baseline_ranks is not None:
        best_ranks = graph_offline_eval.second_hop_ranks(
            graph_offline_eval.reconstruct(
                settings, max_entity_degree=best["degree"], **overrides
            ),
            pairs,
            hop_decay=best["hop_decay"],
            use_idf=best["idf"],
        )
        console.print(_offline_comparison(baseline_ranks, best_ranks))

    console.print(
        "[yellow]Это оценка одной подсистемы. Стартовые сущности по тексту вопроса, "
        "реранкер и слияние каналов сюда не входят: итоговый recall меряется "
        "прогоном eval на сервере.[/yellow]"
    )

    if save:
        path = settings.paths.metrics_dir / "graph_offline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"pairs": len(pairs), "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"Сохранено: {path}")


def _offline_comparison(baseline: Sequence[int], candidate: Sequence[int]) -> Table:
    """Парное сравнение двух настроек обхода на одних и тех же парах."""
    table = Table(title="Против нынешней настройки (порог 64, затухание 0.5, без IDF)")
    for column in ("метрика", "было", "стало", "Δ", "95% интервал", "лучше/хуже", "значимо"):
        table.add_column(column, justify="right" if column != "метрика" else "left")

    scorers = {
        "MRR": lambda rank: 1.0 / rank,
        "hit@8": lambda rank: float(rank <= 8),
        "hit@30": lambda rank: float(rank <= 30),
    }
    for name, score in scorers.items():
        base_values = [score(rank) for rank in baseline]
        cand_values = [score(rank) for rank in candidate]
        differences = [c - b for b, c in zip(base_values, cand_values, strict=True)]
        stats = paired_bootstrap(differences)
        better = sum(1 for value in differences if value > 0)
        worse = sum(1 for value in differences if value < 0)
        significant = stats["low"] > 0 or stats["high"] < 0
        table.add_row(
            name,
            f"{statistics.fmean(base_values):.3f}",
            f"{statistics.fmean(cand_values):.3f}",
            f"{stats['mean']:+.3f}",
            f"[{stats['low']:+.3f}; {stats['high']:+.3f}]",
            f"{better}/{worse}",
            "[green]да[/green]" if significant else "нет",
        )
    return table


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
