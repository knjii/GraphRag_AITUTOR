"""Конвейер индексации.

Основные отличия от прежнего ``ingest.py``:

* **стадии разделены и возобновляемы** — падение на записи графа не заставляет
  заново парсить PDF, а неизменённый документ вообще пропускается;
* **нет борьбы за видеопамять** — модели живут в своих сервисах (Infinity, Ollama),
  а MinerU работает отдельным процессом; исчезли пять механизмов выгрузки моделей
  и безусловные паузы, которые прежде занимали минуты на документ;
* **эмбеддинги считаются один раз** и переиспользуются графовой стадией
  через общий кэш, а не пересчитываются заново;
* **измеряется каждая стадия** — без этого невозможно понять, что именно
  занимает часы, и доказать ускорение.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook.chunking.enrichment import BlockEnricher
from rag_textbook.chunking.layout_chunker import LayoutAwareChunker
from rag_textbook.context import AppContext
from rag_textbook.graph.builder import GraphBuilder
from rag_textbook.indexing.manifest import IndexingManifest
from rag_textbook.indexing.resources import VramGuard, run_with_oom_backoff
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Block, Chunk, content_hash
from rag_textbook.observability.monitor import NullMonitor
from rag_textbook.parsing.pdf_parser import MineruPdfParser, PdfParseError, file_fingerprint

logger = get_logger("indexing.pipeline")

# Стадии обхода корпуса в порядке выполнения. Каждая занимает видеопамять
# своим потребителем, поэтому их можно разносить по времени.
ALL_STAGES: tuple[str, ...] = ("parse", "chunk", "embed", "graph")


@dataclass
class DocumentReport:
    doc_id: str
    doc_name: str
    source_path: str
    status: str = "ok"
    chunks: int = 0
    blocks: int = 0
    error: str = ""
    stage_seconds: dict[str, float] = field(default_factory=dict)
    enrichment: dict[str, int] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexingReport:
    started_at: str = ""
    finished_at: str = ""
    documents: list[DocumentReport] = field(default_factory=list)
    total_chunks: int = 0
    failed: int = 0
    skipped: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_chunks": self.total_chunks,
            "failed": self.failed,
            "skipped": self.skipped,
            "config": self.config,
            "documents": [
                {
                    "doc_id": report.doc_id,
                    "doc_name": report.doc_name,
                    "status": report.status,
                    "chunks": report.chunks,
                    "blocks": report.blocks,
                    "error": report.error,
                    "stage_seconds": {k: round(v, 2) for k, v in report.stage_seconds.items()},
                    "enrichment": report.enrichment,
                    "graph": report.graph,
                }
                for report in self.documents
            ],
        }


def document_id(source_path: Path) -> str:
    """Стабильный идентификатор документа.

    Считается от имени файла, а не от полного пути: иначе перенос корпуса
    на арендованный сервер поменял бы все идентификаторы и превратил
    обновление индекса в полную переиндексацию.
    """
    return content_hash(source_path.name)[:16]


class IndexingPipeline:
    def __init__(self, context: AppContext, monitor: Any | None = None) -> None:
        self.context = context
        self.settings = context.settings
        # Мониторинг размечает замеры ресурсов по стадиям: без этой разметки
        # «карта загружена на 45%» ничего не говорит о том, что оптимизировать.
        self.monitor = monitor if monitor is not None else NullMonitor()
        self.parser = MineruPdfParser(self.settings.parsing, self.settings.paths.parsed_dir)
        self.chunker = LayoutAwareChunker(self.settings.chunking)
        self.enricher = BlockEnricher(self.settings.chunking, context.llm, context.enrichment_cache)
        self.manifest = IndexingManifest(
            self.settings.paths.manifest_dir / "indexing_manifest.json"
        )
        # Страховка от нехватки видеопамяти: позволяет безопасно увеличивать
        # батчи, потому что при ошибке нагрузка снижается автоматически.
        self.guard = VramGuard(
            min_free_mib=self.settings.indexing.min_free_vram_mib,
            poll_seconds=self.settings.indexing.vram_poll_seconds,
            max_wait_seconds=self.settings.indexing.vram_wait_seconds,
            enabled=self.settings.indexing.vram_guard_enabled,
        )

    # ------------------------------------------------------------------ этапы

    def _chunks_path(self, doc_id: str) -> Path:
        return self.settings.paths.parsed_dir / f"{doc_id}_chunks.json"

    def _save_chunks(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        payload = [chunk.model_dump() for chunk in chunks]
        self._chunks_path(doc_id).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _load_chunks(self, doc_id: str) -> list[Chunk]:
        path = self._chunks_path(doc_id)
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [Chunk.model_validate(item) for item in payload]

    def _embed_and_store(self, chunks: Sequence[Chunk]) -> int:
        """Векторизация с защитой от нехватки памяти.

        Батч подбирается по свободной памяти, а при ошибке делится пополам
        и повторяется. Благодаря этому можно ставить крупные батчи ради
        утилизации, не рискуя потерять многочасовой прогон.
        """
        if not chunks:
            return 0
        self.context.vector_store.ensure_collection(self.settings.embedding.dimensions)

        batch = self.guard.batch_size(
            "embed",
            self.settings.indexing.embed_window,
            per_item_mib=self.settings.indexing.embed_per_item_mib,
        )

        def handle(part: Sequence[Chunk]) -> int:
            vectors = self.context.embeddings.embed_documents([chunk.text for chunk in part])
            return self.context.vector_store.upsert(list(part), vectors)

        written = run_with_oom_backoff(
            list(chunks), handle, guard=self.guard, stage="embed", batch_size=batch
        )
        return sum(written)

    def _build_graph(
        self, chunks: Sequence[Chunk], doc_id: str, doc_name: str, source_path: str
    ) -> dict[str, Any]:
        if not self.settings.graph.enabled or self.context.graph_store is None:
            return {"status": "disabled"}
        store = self.context.graph_store
        if not store.verify():
            return {"status": "unavailable"}
        store.ensure_schema()
        builder = GraphBuilder(
            self.settings.graph,
            self.context.entity_extractor(),
            store,
            max_workers=self.settings.llm.max_concurrency,
        )
        result = builder.build(
            chunks,
            doc_id=doc_id,
            doc_name=doc_name,
            source_path=source_path,
            model_name=self.settings.llm.model_for("extraction"),
            write=True,
        )
        return result.as_dict()

    # ------------------------------------------------------------- документ

    def index_document(self, source_path: Path, force: bool = False) -> DocumentReport:
        source_path = Path(source_path)
        doc_id = document_id(source_path)
        doc_name = source_path.stem
        report = DocumentReport(doc_id=doc_id, doc_name=doc_name, source_path=str(source_path))

        try:
            fingerprint = file_fingerprint(source_path)
        except OSError as exc:
            report.status = "failed"
            report.error = f"Не удалось прочитать файл: {exc}"
            return report

        state = self.manifest.get(doc_id, str(source_path), fingerprint)

        # Стадия 1: разбор
        blocks: list[Block] = []
        if force or not state.is_done("parsed"):
            stage_started = time.perf_counter()
            try:
                with self.monitor.stage("parse", doc_name):
                    blocks = self.parser.parse(source_path, force=force)
            except PdfParseError as exc:
                state.mark("parsed", "failed", str(exc))
                self.manifest.save()
                report.status = "failed"
                report.error = str(exc)
                return report
            report.stage_seconds["parse"] = time.perf_counter() - stage_started
            state.mark("parsed")
        else:
            blocks = self.parser.parse(source_path)  # из кэша, быстро
        report.blocks = len(blocks)

        # Стадия 2: обогащение и чанкинг
        if force or not state.is_done("chunked"):
            stage_started = time.perf_counter()
            images_dir = self.parser.images_dir_for(source_path)
            with self.monitor.stage("enrich", doc_name):
                report.enrichment = self.enricher.enrich(blocks, images_dir)
            with self.monitor.stage("chunk", doc_name):
                chunks = self.chunker.chunk(
                    blocks,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    source_path=str(source_path),
                )
            self._save_chunks(doc_id, chunks)
            report.stage_seconds["chunk"] = time.perf_counter() - stage_started
            state.mark("chunked")
            state.chunks = len(chunks)
        else:
            chunks = self._load_chunks(doc_id)
        report.chunks = len(chunks)

        if not chunks:
            report.status = "empty"
            self.manifest.save()
            return report

        # Стадия 3: эмбеддинги и векторное хранилище
        if force or not state.is_done("embedded"):
            stage_started = time.perf_counter()
            try:
                with self.monitor.stage("embed", doc_name):
                    self._embed_and_store(chunks)
            except Exception as exc:  # noqa: BLE001
                state.mark("embedded", "failed", str(exc))
                self.manifest.save()
                report.status = "failed"
                report.error = f"Векторизация: {exc}"
                return report
            report.stage_seconds["embed"] = time.perf_counter() - stage_started
            state.mark("embedded")

        # Стадия 4: граф
        if force or not state.is_done("graphed"):
            stage_started = time.perf_counter()
            try:
                with self.monitor.stage("graph", doc_name):
                    report.graph = self._build_graph(chunks, doc_id, doc_name, str(source_path))
                state.mark("graphed")
            except Exception as exc:  # noqa: BLE001
                # Граф важен, но его сбой не должен обесценивать уже проиндексированные
                # векторы: помечаем стадию как неуспешную и продолжаем.
                state.mark("graphed", "failed", str(exc))
                report.graph = {"status": "failed", "error": str(exc)[:300]}
                if self.settings.graph.fail_on_error:
                    self.manifest.save()
                    report.status = "failed"
                    report.error = f"Граф: {exc}"
                    return report
                logger.warning("Стадия графа не удалась для %s: %s", doc_name, exc)
            report.stage_seconds["graph"] = time.perf_counter() - stage_started

        self.manifest.save()
        return report

    # ---------------------------------------------------------------- корпус

    def discover_documents(self) -> list[Path]:
        pdf_dir = Path(self.settings.paths.pdf_dir)
        if not pdf_dir.is_dir():
            return []
        # Уникальность по имени файла: в корпусе встречаются копии одного учебника
        # в подкаталогах вида test/, test2/, test_prev/ — индексировать их повторно бессмысленно.
        seen: dict[str, Path] = {}
        for path in sorted(pdf_dir.rglob("*.pdf")):
            seen.setdefault(path.name, path)
        return list(seen.values())

    def run(
        self,
        sources: Sequence[Path] | None = None,
        force: bool = False,
        stages: Sequence[str] | None = None,
    ) -> IndexingReport:
        """Индексирует корпус.

        ``stages`` ограничивает набор выполняемых стадий. Это нужно там, где
        карта одна: сервер инференса держит свою долю видеопамяти постоянно,
        пока контейнер запущен, и вместе с MinerU они на 24 ГБ не помещаются.
        Разнести их по времени иначе нельзя — проход по стадиям выполняется
        одним процессом. Манифест делает разбиение безопасным: пропущенная
        стадия не помечается выполненной и доделывается следующим запуском.
        """
        selected = self._resolve_stages(stages)
        documents = list(sources) if sources else self.discover_documents()
        report = IndexingReport(
            started_at=datetime.now(UTC).isoformat(),
            config={
                "parsing_backend": self.settings.parsing.backend,
                "chunk_size": self.settings.chunking.chunk_size,
                "chunk_overlap": self.settings.chunking.chunk_overlap,
                "enrich_types": list(self.settings.chunking.enrich_types),
                "embedding_model": self.settings.embedding.model,
                "graph_enabled": self.settings.graph.enabled,
                "graph_extractor": self.settings.graph.extractor,
                "cooccurrence_min_pmi": self.settings.graph.cooccurrence_min_pmi,
            },
        )

        if not documents:
            logger.warning("Не найдено ни одного PDF в %s", self.settings.paths.pdf_dir)
            report.finished_at = datetime.now(UTC).isoformat()
            return report

        mode = self.settings.indexing.mode
        report.config["indexing_mode"] = mode
        report.config["stages"] = list(selected)
        logger.info(
            "Индексация: документов=%s, режим=%s, force=%s, стадии=%s",
            len(documents),
            mode,
            force,
            ",".join(selected),
        )

        if mode == "stage":
            self._run_stage_major(documents, report, force=force, stages=selected)
        else:
            if set(selected) != set(ALL_STAGES):
                raise ValueError(
                    "Выбор стадий поддерживается только в режиме INDEXING_MODE=stage: "
                    "обход по документам по устройству не разделяется на стадии"
                )
            for position, source in enumerate(documents, start=1):
                logger.info("[%s/%s] %s", position, len(documents), source.name)
                doc_report = self.index_document(source, force=force)
                report.documents.append(doc_report)
                report.total_chunks += doc_report.chunks
                if doc_report.status == "failed":
                    report.failed += 1
                elif not doc_report.stage_seconds:
                    report.skipped += 1

        report.config["oom_events"] = self.guard.oom_events
        report.finished_at = datetime.now(UTC).isoformat()
        self._save_report(report)
        logger.info(
            "Индексация завершена: чанков=%s, ошибок=%s, пропущено=%s",
            report.total_chunks,
            report.failed,
            report.skipped,
        )
        return report

    # -------------------------------------------------- обход по стадиям

    @staticmethod
    def _resolve_stages(stages: Sequence[str] | None) -> tuple[str, ...]:
        if not stages:
            return ALL_STAGES
        selected = tuple(
            dict.fromkeys(str(item).strip().lower() for item in stages if str(item).strip())
        )
        unknown = [item for item in selected if item not in ALL_STAGES]
        if unknown:
            raise ValueError(
                f"Неизвестные стадии: {', '.join(unknown)}. Доступны: {', '.join(ALL_STAGES)}"
            )
        return selected or ALL_STAGES

    def _run_stage_major(
        self,
        documents: Sequence[Path],
        report: IndexingReport,
        *,
        force: bool,
        stages: Sequence[str] = ALL_STAGES,
    ) -> None:
        """Проходит корпус по стадиям, а не по документам.

        Зачем так. Во-первых, в каждый момент на карте работает **один**
        потребитель: сначала MinerU, потом модель зрения, потом эмбеддер,
        потом модель извлечения. Это исключает конкуренцию за видеопамять,
        то есть убирает главный источник риска нехватки памяти.

        Во-вторых, батчи становятся крупными: эмбеддер получает тысячи чанков
        подряд вместо нескольких сотен на документ, а модель извлечения —
        непрерывный поток запросов. Именно на непрерывном потоке работает
        непрерывный батчинг сервера инференса.

        Возобновляемость сохраняется: каждая стадия сверяется с манифестом
        и пропускает уже сделанное.
        """
        reports: dict[str, DocumentReport] = {}
        states: dict[str, Any] = {}
        blocks_by_doc: dict[str, list[Block]] = {}
        chunks_by_doc: dict[str, list[Chunk]] = {}
        paths: dict[str, Path] = {}

        # Подготовка: отпечатки файлов и состояние из манифеста.
        for source in documents:
            source = Path(source)
            doc_id = document_id(source)
            doc_report = DocumentReport(
                doc_id=doc_id, doc_name=source.stem, source_path=str(source)
            )
            try:
                fingerprint = file_fingerprint(source)
            except OSError as exc:
                doc_report.status = "failed"
                doc_report.error = f"Не удалось прочитать файл: {exc}"
                reports[doc_id] = doc_report
                continue
            reports[doc_id] = doc_report
            states[doc_id] = self.manifest.get(doc_id, str(source), fingerprint)
            paths[doc_id] = source

        alive = [doc_id for doc_id, item in reports.items() if item.status != "failed"]

        # Стадия 1: разбор. Карту целиком занимает MinerU.
        if "parse" not in stages:
            logger.info("Стадия «разбор» пропущена по выбору стадий")
            alive = [doc_id for doc_id in alive if states[doc_id].is_done("parsed")]
        else:
            logger.info("Стадия «разбор»: документов %s", len(alive))
        for doc_id in list(alive) if "parse" in stages else []:
            source, state, doc_report = paths[doc_id], states[doc_id], reports[doc_id]
            started = time.perf_counter()
            try:
                with self.monitor.stage("parse", doc_report.doc_name):
                    self.guard.wait_for_headroom(
                        self.settings.indexing.parse_required_vram_mib, "parse"
                    )
                    blocks_by_doc[doc_id] = self.parser.parse(source, force=force)
                if force or not state.is_done("parsed"):
                    doc_report.stage_seconds["parse"] = time.perf_counter() - started
                state.mark("parsed")
            except PdfParseError as exc:
                state.mark("parsed", "failed", str(exc))
                doc_report.status = "failed"
                doc_report.error = str(exc)
                alive.remove(doc_id)
            doc_report.blocks = len(blocks_by_doc.get(doc_id, []))
        self.manifest.save()

        # Стадия 2: обогащение и чанкинг. Карту занимает модель зрения.
        logger.info("Стадия «обогащение и чанкинг»: документов %s", len(alive))
        for doc_id in list(alive):
            source, state, doc_report = paths[doc_id], states[doc_id], reports[doc_id]
            blocks = blocks_by_doc.get(doc_id) or []
            # Загрузка готовых чанков выполняется всегда, даже если стадия
            # не выбрана: последующим стадиям они нужны на входе.
            if not force and state.is_done("chunked"):
                chunks_by_doc[doc_id] = self._load_chunks(doc_id)
                doc_report.chunks = len(chunks_by_doc[doc_id])
                continue
            if "chunk" not in stages:
                logger.info("Стадия «чанкинг» пропущена по выбору стадий: %s", doc_report.doc_name)
                continue
            started = time.perf_counter()
            images_dir = self.parser.images_dir_for(source)
            with self.monitor.stage("enrich", doc_report.doc_name):
                doc_report.enrichment = self.enricher.enrich(blocks, images_dir)
            with self.monitor.stage("chunk", doc_report.doc_name):
                chunks = self.chunker.chunk(
                    blocks,
                    doc_id=doc_id,
                    doc_name=doc_report.doc_name,
                    source_path=str(source),
                )
            self._save_chunks(doc_id, chunks)
            chunks_by_doc[doc_id] = chunks
            doc_report.chunks = len(chunks)
            doc_report.stage_seconds["chunk"] = time.perf_counter() - started
            state.mark("chunked")
            state.chunks = len(chunks)
        self.manifest.save()

        # Блоки больше не нужны: освобождаем оперативную память до тяжёлых стадий.
        blocks_by_doc.clear()

        # Стадия 3: векторизация. Один непрерывный поток по всему корпусу.
        pending_embed = (
            [
                doc_id
                for doc_id in alive
                if (force or not states[doc_id].is_done("embedded")) and chunks_by_doc.get(doc_id)
            ]
            if "embed" in stages
            else []
        )
        if pending_embed:
            total = sum(len(chunks_by_doc[doc_id]) for doc_id in pending_embed)
            logger.info(
                "Стадия «векторизация»: чанков %s из %s документов", total, len(pending_embed)
            )
            started = time.perf_counter()
            try:
                with self.monitor.stage("embed", "корпус"):
                    for doc_id in pending_embed:
                        self._embed_and_store(chunks_by_doc[doc_id])
                        states[doc_id].mark("embedded")
                elapsed = time.perf_counter() - started
                for doc_id in pending_embed:
                    share = len(chunks_by_doc[doc_id]) / max(1, total)
                    reports[doc_id].stage_seconds["embed"] = elapsed * share
            except Exception as exc:  # noqa: BLE001
                for doc_id in pending_embed:
                    states[doc_id].mark("embedded", "failed", str(exc))
                    reports[doc_id].status = "failed"
                    reports[doc_id].error = f"Векторизация: {exc}"
                alive = [doc_id for doc_id in alive if doc_id not in pending_embed]
            self.manifest.save()

        # Стадия 4: граф. Карту занимает модель извлечения.
        pending_graph = (
            [
                doc_id
                for doc_id in alive
                if (force or not states[doc_id].is_done("graphed")) and chunks_by_doc.get(doc_id)
            ]
            if "graph" in stages
            else []
        )
        if pending_graph:
            logger.info("Стадия «граф»: документов %s", len(pending_graph))
            for doc_id in pending_graph:
                doc_report, state = reports[doc_id], states[doc_id]
                started = time.perf_counter()
                try:
                    with self.monitor.stage("graph", doc_report.doc_name):
                        doc_report.graph = self._build_graph(
                            chunks_by_doc[doc_id],
                            doc_id,
                            doc_report.doc_name,
                            str(paths[doc_id]),
                        )
                    state.mark("graphed")
                except Exception as exc:  # noqa: BLE001
                    state.mark("graphed", "failed", str(exc))
                    doc_report.graph = {"status": "failed", "error": str(exc)[:300]}
                    if self.settings.graph.fail_on_error:
                        doc_report.status = "failed"
                        doc_report.error = f"Граф: {exc}"
                    else:
                        logger.warning(
                            "Стадия графа не удалась для %s: %s", doc_report.doc_name, exc
                        )
                doc_report.stage_seconds["graph"] = time.perf_counter() - started
            self.manifest.save()

        for doc_report in reports.values():
            report.documents.append(doc_report)
            report.total_chunks += doc_report.chunks
            if doc_report.status == "failed":
                # Причина обязана быть видна в логе, а не только в JSON-отчёте:
                # иначе прогон выглядит как «failed» без объяснений и разбираться
                # приходится вручную.
                logger.error(
                    "Документ %s не проиндексирован: %s", doc_report.doc_name, doc_report.error
                )
                report.failed += 1
            elif not doc_report.stage_seconds:
                report.skipped += 1

    def _save_report(self, report: IndexingReport) -> Path:
        metrics_dir = Path(self.settings.paths.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = metrics_dir / f"indexing_{stamp}.json"
        path.write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Отчёт индексации сохранён: %s", path)
        return path
