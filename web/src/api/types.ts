// Типы обмена с бэкендом. Форма совпадает с тем, что отдаёт демо-сервис
// (web/server/demo_api.py), поэтому заглушка и сервер взаимозаменяемы.

export type SourceStatus = "ready" | "processing" | "error";
export type IngestStage = "upload" | "parse" | "chunk" | "graph";

export interface Source {
  id: string;
  title: string;
  authors: string;
  pages: number;
  fragments: number;
  formulas: number;
  status: SourceStatus;
  stage?: IngestStage;
  progress?: number; // 0..1 внутри текущей стадии
  error?: string;
  enabled: boolean; // участвует ли в контексте ассистента
  addedAt: string;
}

export interface Fragment {
  id: string;
  sourceId: string;
  page: number;
  section: string;
  text: string; // Markdown с формулами $…$ и $$…$$
}

export type Channel = "dense" | "sparse" | "graph";

export interface Citation {
  n: number;
  fragmentId: string;
  sourceId: string;
  page: number;
  section: string;
  channel: Channel;
}

export interface Timings {
  retrievalMs: number;
  generationMs: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  timings?: Timings;
  multiHop?: boolean;
  trace?: Trace;
  pending?: boolean;
  error?: string;
}

export type RouterMode = "auto" | "always" | "off";
export type Selection = "off" | "setr" | "seal";

export interface RetrievalSettings {
  topK: number;
  graphWeight: number;
  router: RouterMode;
  reranker: boolean;
  selection: Selection;
  model: string;
}

export interface Preset {
  id: string;
  name: string;
  hint: string;
  settings: RetrievalSettings;
}

// local — можно менять любой параметр; server — только готовые пресеты.
export type DemoMode = "local" | "server";

export interface ServiceInfo {
  mode: DemoMode;
  models: string[];
  presets: Preset[];
  // llm — отвечает модель; extractive — модели нет, показываем выписку.
  generator: "llm" | "extractive";
  graph: boolean;
  uploads: boolean;
  selections: Selection[];
}

export interface FragmentPage {
  total: number;
  offset: number;
  items: Fragment[];
}

// ---------------------------------------------------------------- трасса

export interface TraceNode {
  id: string;
  name: string;
  canonical: string;
  kind: "concept" | "notation";
  weight: number;
  role: "seed" | "neighbour";
  hop: number; // 0 — вопрос, дальше дозапросы SEAL
}

export interface TraceLink {
  source: string;
  target: string;
  label: string;
}

export interface TraceFragment {
  id: string;
  sourceId: string;
  page: number | null;
  section: string;
  preview: string;
  channel: Channel;
  entities: string[];
  found: ("text" | "graph")[];
  hop: number;
}

export type TracePhase = "route" | "text" | "graph" | "rerank" | "selection" | "generation" | "done" | "error";

export interface Trace {
  phase: TracePhase;
  useGraph: boolean | null;
  terms: string[];
  nodes: TraceNode[];
  links: TraceLink[];
  seedTotal: number;
  neighbourTotal: number;
  fragments: TraceFragment[]; // в порядке появления, не больше нескольких десятков
  textTotal: number;
  graphTotal: number;
  order: string[]; // порядок после переранжирования
  selected: string[];
  contexts: string[]; // порядок, в котором фрагменты увидела модель: [n] = contexts[n-1]
  sealQueries: string[];
  pool: number;
  // Источник, названный в вопросе: поиск шёл только по нему.
  scope?: { titles: string[]; reason: string };
}

export type TraceEvent =
  | { type: "start"; data: { question: string; preset: string | null } }
  | { type: "stage"; data: { id: TracePhase; state: "start"; hop?: number } }
  | { type: "route"; data: { useGraph: boolean; reason: string } }
  | {
      type: "seeds";
      data: { hop: number; origin: string; terms: string[]; total: number; entities: Omit<TraceNode, "role" | "hop">[] };
    }
  | { type: "expand"; data: { hop: number; total: number; entities: Omit<TraceNode, "role" | "hop">[]; links: TraceLink[] } }
  | {
      type: "candidates";
      data: {
        hop: number;
        channel: "text" | "graph";
        total: number;
        items: (Omit<TraceFragment, "found" | "hop"> & { rank: number })[];
      };
    }
  | { type: "rerank"; data: { hop: number; order: string[] } }
  | { type: "seal_hop"; data: { hop: number; query: string } }
  | { type: "scope"; data: { sourceIds: string[]; titles: string[]; reason: string } }
  | { type: "selected"; data: { ids: string[]; pool: number; sealAdded: string[]; retrievalMs: number } }
  | { type: "token"; data: { text: string } }
  | {
      type: "done";
      data: { text?: string; contexts: string[]; citations: Citation[]; timings: Timings; multiHop: boolean };
    }
  | { type: "error"; data: { message: string } };
