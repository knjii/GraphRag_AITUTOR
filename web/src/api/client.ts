// Единая точка доступа к сервису. Две реализации с одним интерфейсом:
// заглушка (VITE_API=mock, по умолчанию в разработке) и HTTP к демо-сервису
// web/server/demo_api.py (VITE_API=http). Ответ приходит потоком событий:
// шаги поиска для сцены графа, затем текст ответа.

import { ANSWERS, FRAGMENTS, MODELS, PRESETS, SOURCES } from "./mock";
import type {
  Citation,
  DemoMode,
  Fragment,
  FragmentPage,
  IngestStage,
  RetrievalSettings,
  ServiceInfo,
  Source,
  TraceEvent,
  TraceFragment,
} from "./types";

export interface AskRequest {
  question: string;
  sourceIds: string[];
  presetId: string | null;
  settings: RetrievalSettings;
}

export interface Api {
  info(): Promise<ServiceInfo>;
  listSources(): Promise<Source[]>;
  upload(file: File, onUpdate: (source: Source) => void): Promise<Source>;
  remove(id: string): Promise<void>;
  fragments(sourceId: string, at: { around?: string; offset?: number }): Promise<FragmentPage>;
  ask(request: AskRequest, onEvent: (event: TraceEvent) => void, signal: AbortSignal): Promise<void>;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const PAGE = 40;

// ------------------------------------------------------------------ HTTP

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await errorText(response));
  return response.json() as Promise<T>;
}

async function errorText(response: Response) {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : `сервис ответил ${response.status}`;
  } catch {
    return `сервис ответил ${response.status}`;
  }
}

function createHttpApi(base = "/api"): Api {
  return {
    info: () => getJson(`${base}/info`),
    listSources: () => getJson(`${base}/sources`),
    async upload() {
      throw new Error("загрузка на этом стенде выключена");
    },
    async remove() {
      throw new Error("удаление на этом стенде выключено");
    },
    fragments(sourceId, { around, offset }) {
      const query = new URLSearchParams({ limit: String(PAGE) });
      if (around) query.set("around", around);
      if (offset !== undefined) query.set("offset", String(offset));
      return getJson(`${base}/sources/${encodeURIComponent(sourceId)}/fragments?${query}`);
    },
    async ask(request, onEvent, signal) {
      const response = await fetch(`${base}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal,
      });
      if (!response.ok || !response.body) throw new Error(await errorText(response));
      // Поток SSE: блоки «event: …\ndata: …» через пустую строку.
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        let cut: number;
        while ((cut = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, cut);
          buffer = buffer.slice(cut + 2);
          const type = /^event: (.+)$/m.exec(block)?.[1];
          const data = /^data: (.+)$/m.exec(block)?.[1];
          if (type && data) onEvent({ type, data: JSON.parse(data) } as TraceEvent);
        }
      }
    },
  };
}

// ------------------------------------------------------------------ заглушка

function createMockApi(mode: DemoMode): Api {
  let sources = SOURCES.map((s) => ({ ...s }));
  const fragments = [...FRAGMENTS];

  return {
    async info() {
      return {
        mode,
        models: MODELS,
        presets: PRESETS,
        generator: "llm",
        graph: true,
        uploads: true,
        selections: ["off", "setr", "seal"],
      };
    },
    async listSources() {
      await sleep(150);
      return sources.map((s) => ({ ...s }));
    },
    async upload(file, onUpdate) {
      const id = `up-${Date.now().toString(36)}`;
      const title = file.name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ");
      let source: Source = {
        id,
        title,
        authors: "Загружено вами",
        pages: 0,
        fragments: 0,
        formulas: 0,
        status: "processing",
        stage: "upload",
        progress: 0,
        enabled: true,
        addedAt: new Date().toISOString().slice(0, 10),
      };
      sources = [source, ...sources];
      const stages: IngestStage[] = ["upload", "parse", "chunk", "graph"];
      for (const stage of stages) {
        for (let step = 0; step <= 4; step++) {
          source = { ...source, stage, progress: step / 4 };
          onUpdate(source);
          await sleep(stage === "parse" ? 450 : 250);
        }
      }
      const pages = 40 + Math.round(file.size / 50_000);
      source = { ...source, status: "ready", stage: undefined, progress: undefined, pages, fragments: pages * 3, formulas: pages * 7 };
      sources = sources.map((s) => (s.id === id ? source : s));
      fragments.push({
        id: `${id}-p1-1`,
        sourceId: id,
        page: 1,
        section: "Начало документа",
        text: "Документ разобран. Здесь появятся фрагменты с формулами, когда подключится сервис.",
      });
      onUpdate(source);
      return source;
    },
    async remove(id) {
      sources = sources.filter((s) => s.id !== id);
    },
    async fragments(sourceId) {
      await sleep(120);
      const items = fragments.filter((f) => f.sourceId === sourceId).sort((a, b) => a.page - b.page);
      return { total: items.length, offset: 0, items };
    },
    async ask({ question, sourceIds, settings }, onEvent, signal) {
      const emit = async (event: TraceEvent, pause = 0) => {
        if (signal.aborted) throw new DOMException("остановлено", "AbortError");
        onEvent(event);
        if (pause) await sleep(pause);
      };
      const started = performance.now();
      const allowed = new Set(sourceIds);
      const scripted = ANSWERS.find(
        (a) => a.match.test(question) && a.cites.every((c) => allowed.has(sourceOf(c.fragmentId))),
      );

      await emit({ type: "start", data: { question, preset: null } });
      await emit({ type: "stage", data: { id: "route", state: "start" } }, 80);
      await emit({ type: "route", data: { useGraph: settings.router !== "off", reason: "заглушка" } });

      const pool = mockPool(scripted?.cites.map((c) => c.fragmentId) ?? [], allowed);
      const textItems = pool.filter((_, i) => i % 3 !== 2);
      await emit({ type: "stage", data: { id: "text", state: "start", hop: 0 } }, 150);
      await emit({ type: "candidates", data: { hop: 0, channel: "text", total: 30, items: textItems.map(asCandidate("dense")) } });

      if (scripted && settings.router !== "off") {
        const seeds = scripted.seeds.map((name, i) => node(name, 1 - i * 0.15));
        const neighbours = scripted.neighbours.map(([name], i) => node(name, 0.8 - i * 0.03));
        await emit({ type: "stage", data: { id: "graph", state: "start", hop: 0 } }, 120);
        await emit({ type: "seeds", data: { hop: 0, origin: "question", terms: [], total: 20, entities: seeds } });
        await emit({
          type: "expand",
          data: {
            hop: 0,
            total: 32,
            entities: neighbours,
            links: scripted.neighbours.map(([name, seed, label]) => ({ source: nodeId(scripted.seeds[seed]), target: nodeId(name), label })),
          },
        });
        const graphItems = pool.filter((_, i) => i % 3 !== 0).map((f, i) => ({
          ...f,
          entities: [scripted.seeds[i % scripted.seeds.length], scripted.neighbours[i % scripted.neighbours.length][0]],
        }));
        await emit({ type: "candidates", data: { hop: 0, channel: "graph", total: 30, items: graphItems.map(asCandidate("graph")) } }, 200);
      }

      const cited = scripted?.cites.map((c) => c.fragmentId) ?? [];
      const order = [...cited, ...pool.map((f) => f.id).filter((id) => !cited.includes(id))];
      if (settings.reranker) {
        await emit({ type: "stage", data: { id: "rerank", state: "start", hop: 0 } }, 200);
        await emit({ type: "rerank", data: { hop: 0, order } });
      }
      if (settings.selection === "seal" && scripted) {
        await emit({ type: "seal_hop", data: { hop: 1, query: `${scripted.seeds[0]}: чего не хватает для ответа` } }, 700);
      } else if (settings.selection === "setr") {
        await sleep(500);
      }
      const selected = order.slice(0, Math.min(settings.topK, order.length));
      await emit({
        type: "selected",
        data: { ids: selected, pool: pool.length, sealAdded: [], retrievalMs: performance.now() - started },
      });
      const retrievalMs = performance.now() - started;

      if (!scripted) {
        const text =
          sourceIds.length === 0
            ? "Все документы выключены из контекста. Включите хотя бы один источник слева, чтобы я мог ответить."
            : "В выбранных документах не нашлось фрагментов, которые отвечают на этот вопрос. " +
              "Попробуйте включить другие источники или переформулировать вопрос.";
        await stream(text, (t) => onEvent({ type: "token", data: { text: t } }));
        await emit({ type: "done", data: { contexts: [], citations: [], timings: { retrievalMs, generationMs: 400 }, multiHop: false } });
        return;
      }

      const citations: Citation[] = scripted.cites.map((c, i) => {
        const fragment = fragments.find((f) => f.id === c.fragmentId)!;
        return { n: i + 1, fragmentId: fragment.id, sourceId: fragment.sourceId, page: fragment.page, section: fragment.section, channel: c.channel };
      });
      await emit({ type: "stage", data: { id: "generation", state: "start" } });
      const genStarted = performance.now();
      await stream(scripted.content, (t) => {
        if (!signal.aborted) onEvent({ type: "token", data: { text: t } });
      });
      await emit({
        type: "done",
        data: {
          contexts: selected,
          citations,
          multiHop: scripted.multiHop,
          timings: { retrievalMs, generationMs: performance.now() - genStarted },
        },
      });
    },
  };

  function sourceOf(fragmentId: string) {
    return fragments.find((f) => f.id === fragmentId)?.sourceId ?? "";
  }

  // Пул кандидатов: настоящие фрагменты заглушки и безымянные соседи для объёма.
  function mockPool(first: string[], allowed: Set<string>): Fragment[] {
    const real = fragments.filter((f) => allowed.has(f.sourceId));
    const ordered = [...real.filter((f) => first.includes(f.id)), ...real.filter((f) => !first.includes(f.id))];
    const fill: Fragment[] = [];
    const docs = [...allowed];
    for (let i = 0; ordered.length + fill.length < 22 && docs.length; i++) {
      const sourceId = docs[i % docs.length];
      fill.push({ id: `${sourceId}-fill-${i}`, sourceId, page: 20 + i * 7, section: "", text: "Фрагмент из того же документа." });
    }
    return [...ordered, ...fill];
  }
}

const nodeId = (name: string) => `n-${name}`;
const node = (name: string, weight: number) => ({ id: nodeId(name), name, canonical: name, kind: "concept" as const, weight });

const asCandidate =
  (channel: "dense" | "graph") =>
  (f: Fragment & { entities?: string[] }, rank: number): Omit<TraceFragment, "found" | "hop"> & { rank: number } => ({
    id: f.id,
    sourceId: f.sourceId,
    page: f.page,
    section: f.section,
    preview: f.text.slice(0, 160),
    channel,
    entities: f.entities ?? [],
    rank,
  });

// Ответ приходит кусками, как при потоковой генерации; формулы не рвём.
async function stream(text: string, onToken: (text: string) => void) {
  const parts = text.match(/\$\$[\s\S]*?\$\$|\$[^$]*\$|\s+|[^\s$]+/g) ?? [text];
  let shown = "";
  for (const part of parts) {
    shown += part;
    onToken(shown);
    await sleep(part.trim() ? 28 : 0);
  }
}

const kind = (import.meta.env.VITE_API as string | undefined) ?? "mock";
const mode = (import.meta.env.VITE_DEMO_MODE as DemoMode | undefined) ?? "local";
export const api: Api = kind === "http" ? createHttpApi() : createMockApi(mode);
