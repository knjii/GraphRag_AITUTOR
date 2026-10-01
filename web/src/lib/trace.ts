import type { Trace, TraceEvent, TraceFragment, TraceNode } from "../api/types";

// Сколько держать на сцене каждый шаг. Сервис проходит поиск за секунду-две,
// и без задержки зритель увидит только итог. Задержка ложится только на показ:
// события приходят и копятся сразу, ответ от неё не медленнее больше чем на
// пару секунд.
const DWELL: Partial<Record<TraceEvent["type"], number>> = {
  route: 250,
  seeds: 650,
  expand: 750,
  candidates: 550,
  rerank: 700,
  seal_hop: 500,
  selected: 750,
};

export function emptyTrace(): Trace {
  return {
    phase: "route",
    useGraph: null,
    terms: [],
    nodes: [],
    links: [],
    seedTotal: 0,
    neighbourTotal: 0,
    fragments: [],
    textTotal: 0,
    graphTotal: 0,
    order: [],
    selected: [],
    contexts: [],
    sealQueries: [],
    pool: 0,
  };
}

function addNodes(nodes: TraceNode[], incoming: Omit<TraceNode, "role" | "hop">[], role: TraceNode["role"], hop: number) {
  const known = new Set(nodes.map((n) => n.id));
  return [...nodes, ...incoming.filter((n) => !known.has(n.id) && n.name).map((n) => ({ ...n, role, hop }))];
}

export function applyTrace(trace: Trace, event: TraceEvent): Trace {
  switch (event.type) {
    case "stage":
      return { ...trace, phase: event.data.id };
    case "route":
      return { ...trace, useGraph: event.data.useGraph };
    case "seeds": {
      const { hop, terms, total, entities } = event.data;
      return {
        ...trace,
        phase: "graph",
        terms: hop === 0 && terms.length ? terms : trace.terms,
        seedTotal: hop === 0 && (event.data.origin === "question" || !trace.seedTotal) ? total : trace.seedTotal,
        nodes: addNodes(trace.nodes, entities, "seed", hop),
      };
    }
    case "expand": {
      const { hop, total, entities, links } = event.data;
      const seen = new Set(trace.links.map((l) => `${l.source}>${l.target}`));
      return {
        ...trace,
        neighbourTotal: hop === 0 ? total : trace.neighbourTotal,
        nodes: addNodes(trace.nodes, entities, "neighbour", hop),
        links: [...trace.links, ...links.filter((l) => !seen.has(`${l.source}>${l.target}`))],
      };
    }
    case "candidates": {
      const { hop, channel, total, items } = event.data;
      const byId = new Map(trace.fragments.map((f) => [f.id, f]));
      for (const item of items) {
        const previous = byId.get(item.id);
        if (previous) {
          const found = previous.found.includes(channel) ? previous.found : [...previous.found, channel];
          byId.set(item.id, {
            ...previous,
            found,
            entities: previous.entities.length ? previous.entities : item.entities,
          });
        } else {
          const fragment: TraceFragment = { ...item, found: [channel], hop };
          byId.set(item.id, fragment);
        }
      }
      return {
        ...trace,
        phase: channel,
        fragments: [...byId.values()],
        textTotal: channel === "text" && hop === 0 ? total : trace.textTotal,
        graphTotal: channel === "graph" && hop === 0 ? total : trace.graphTotal,
      };
    }
    case "rerank":
      return event.data.hop === 0 ? { ...trace, phase: "rerank", order: event.data.order } : trace;
    case "scope":
      return { ...trace, scope: { titles: event.data.titles, reason: event.data.reason } };
    case "seal_hop":
      return { ...trace, phase: "selection", sealQueries: [...trace.sealQueries, event.data.query] };
    case "selected":
      return { ...trace, phase: "selection", selected: event.data.ids, pool: event.data.pool };
    case "done":
      return { ...trace, phase: "done", contexts: event.data.contexts };
    case "error":
      return { ...trace, phase: "error" };
    default:
      return trace;
  }
}

// Очередь показа: события применяются по порядку, шаги трассы — с выдержкой.
// cancel() останавливает показ (кнопка «Остановить» или новый вопрос).
export function pacedQueue(apply: (event: TraceEvent) => void, reducedMotion: boolean) {
  const queue: TraceEvent[] = [];
  let running = false;
  let cancelled = false;
  let drainedResolve: (() => void) | null = null;

  async function pump() {
    running = true;
    while (queue.length && !cancelled) {
      const event = queue.shift()!;
      apply(event);
      const dwell = reducedMotion ? 0 : (DWELL[event.type] ?? 0);
      if (dwell) await new Promise((r) => setTimeout(r, dwell));
    }
    running = false;
    if (!queue.length && drainedResolve) drainedResolve();
  }

  return {
    push(event: TraceEvent) {
      if (cancelled) return;
      // Токены копятся быстро: в очереди нужен только последний.
      const last = queue[queue.length - 1];
      if (event.type === "token" && last?.type === "token") queue[queue.length - 1] = event;
      else queue.push(event);
      if (!running) void pump();
    },
    drained() {
      if (!running && !queue.length) return Promise.resolve();
      return new Promise<void>((resolve) => (drainedResolve = resolve));
    },
    cancel() {
      cancelled = true;
      queue.length = 0;
    },
  };
}
