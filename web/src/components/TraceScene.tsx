import { ChevronDown, Network } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Citation, Trace, TraceFragment, TraceNode } from "../api/types";
import { plural } from "../lib/plural";

// Сцена поиска: как вопрос превращается в понятия графа, куда уходит обход,
// что приносит каждый канал и что в итоге попадает в ответ.

interface Props {
  trace: Trace;
  citations: Citation[];
  answering: boolean; // ответ уже печатается — сцену можно свернуть
  onOpenFragment: (sourceId: string, fragmentId: string) => void;
  sourceTitle: (id: string) => string;
}

const STEPS = [
  { id: "seeds", label: "Понятия" },
  { id: "walk", label: "Обход графа" },
  { id: "pool", label: "Кандидаты" },
  { id: "rerank", label: "Переранжирование" },
  { id: "select", label: "Отбор" },
  { id: "answer", label: "Ответ" },
] as const;

type StepId = (typeof STEPS)[number]["id"];

function activeStep(trace: Trace): StepId {
  if (trace.phase === "generation" || trace.phase === "done") return "answer";
  if (trace.selected.length) return "select";
  if (trace.order.length || trace.phase === "rerank") return "rerank";
  if (trace.fragments.length && (trace.phase === "text" || trace.fragments.some((f) => f.found.includes("graph")))) return "pool";
  if (trace.nodes.some((n) => n.role === "neighbour")) return "walk";
  if (trace.nodes.length) return "seeds";
  return trace.phase === "text" ? "pool" : "seeds";
}

interface Layout {
  width: number;
  height: number;
  cx: number;
  cy: number;
  inner: number;
  outer: number;
  tiles: { x: number; y: number; cols: number; w: number; h: number; gap: number; max: number };
}

const WIDE: Layout = {
  width: 680,
  height: 270,
  cx: 200,
  cy: 134,
  inner: 50,
  outer: 100,
  tiles: { x: 412, y: 28, cols: 6, w: 37, h: 30, gap: 8, max: 30 },
};

const NARROW: Layout = {
  width: 360,
  height: 470,
  cx: 180,
  cy: 128,
  inner: 46,
  outer: 92,
  tiles: { x: 24, y: 286, cols: 6, w: 44, h: 28, gap: 8, max: 24 },
};

const shorten = (text: string, n: number) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);

function placeNodes(nodes: TraceNode[], links: Trace["links"], layout: Layout) {
  const seeds = nodes.filter((n) => n.role === "seed");
  const neighbours = nodes.filter((n) => n.role === "neighbour");
  const pos = new Map<string, { x: number; y: number; angle: number }>();
  const step = (2 * Math.PI) / Math.max(seeds.length, 1);
  seeds.forEach((n, i) => {
    const angle = -Math.PI / 2 + i * step;
    const r = seeds.length === 1 ? 0 : layout.inner;
    pos.set(n.id, { x: layout.cx + r * Math.cos(angle), y: layout.cy + r * Math.sin(angle), angle });
  });

  // Сосед встаёт напротив своей затравки; без связи с показанными — по кругу.
  const wanted = neighbours.map((n, i) => {
    const anchors = links
      .filter((l) => l.source === n.id || l.target === n.id)
      .map((l) => pos.get(l.source === n.id ? l.target : l.source))
      .filter((p): p is { x: number; y: number; angle: number } => !!p && seeds.length > 1);
    const angle = anchors.length
      ? Math.atan2(
          anchors.reduce((s, a) => s + Math.sin(a.angle), 0),
          anchors.reduce((s, a) => s + Math.cos(a.angle), 0),
        )
      : -Math.PI / 2 + (i + 0.5) * ((2 * Math.PI) / Math.max(neighbours.length, 1));
    return { n, angle };
  });
  // Разводим соседей, чтобы подписи не слипались.
  wanted.sort((a, b) => a.angle - b.angle);
  const gap = Math.min((2 * Math.PI) / Math.max(wanted.length, 1), 0.42);
  for (let pass = 0; pass < 4; pass++) {
    for (let i = 1; i < wanted.length; i++) {
      if (wanted[i].angle - wanted[i - 1].angle < gap) wanted[i].angle = wanted[i - 1].angle + gap;
    }
  }
  // Раздвинутые не должны заехать по кругу на первых — тогда ставим равномерно.
  if (wanted.length > 1 && wanted[wanted.length - 1].angle - wanted[0].angle > 2 * Math.PI - gap) {
    const even = (2 * Math.PI) / wanted.length;
    wanted.forEach((w, i) => (w.angle = wanted[0].angle + i * even));
  }
  wanted.forEach(({ n, angle }) => {
    pos.set(n.id, { x: layout.cx + layout.outer * Math.cos(angle), y: layout.cy + layout.outer * Math.sin(angle), angle });
  });
  return pos;
}

function tileOrder(trace: Trace): TraceFragment[] {
  if (!trace.order.length) return trace.fragments;
  const rank = new Map(trace.order.map((id, i) => [id, i]));
  return [...trace.fragments].sort((a, b) => (rank.get(a.id) ?? 1e6) - (rank.get(b.id) ?? 1e6));
}

export function TraceScene({ trace, citations, answering, onOpenFragment, sourceTitle }: Props) {
  const boxRef = useRef<HTMLDivElement>(null);
  const [narrow, setNarrow] = useState(false);
  const [open, setOpen] = useState(true);
  const collapsedOnce = useRef(false);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setNarrow(entry.contentRect.width < 520));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Когда начинается ответ, сцена уступает место тексту — один раз,
  // дальше пользователь сам решает, держать её открытой или нет.
  useEffect(() => {
    if (answering && !collapsedOnce.current) {
      collapsedOnce.current = true;
      setOpen(false);
    }
  }, [answering]);

  const layout = narrow ? NARROW : WIDE;
  const pos = useMemo(() => placeNodes(trace.nodes, trace.links, layout), [trace.nodes, trace.links, layout]);
  const tiles = tileOrder(trace).slice(0, layout.tiles.max);
  const selected = new Set(trace.selected);
  const decided = trace.selected.length > 0;
  const citationOf = new Map(citations.map((c) => [c.fragmentId, c.n]));
  const contextIndex = new Map(trace.contexts.map((id, i) => [id, i + 1]));
  const byCanonical = new Map(trace.nodes.flatMap((n) => [[n.canonical, n] as const, [n.name, n] as const]));
  const step = activeStep(trace);
  const stepIndex = STEPS.findIndex((s) => s.id === step);
  const graphSkipped = trace.useGraph === false;
  const seedCount = trace.nodes.filter((n) => n.role === "seed" && n.hop === 0).length;
  const graphFound = trace.fragments.filter((f) => f.found.includes("graph")).length;
  const graphOnlySelected = trace.fragments.filter((f) => selected.has(f.id) && f.found.length === 1 && f.found[0] === "graph").length;

  const tileXY = (i: number) => {
    const t = layout.tiles;
    return { x: t.x + (i % t.cols) * (t.w + t.gap), y: t.y + Math.floor(i / t.cols) * (t.h + t.gap) };
  };

  const summary = [
    graphSkipped ? "граф не понадобился" : `${trace.seedTotal || seedCount} ${plural(trace.seedTotal || seedCount, "понятие", "понятия", "понятий")} в вопросе`,
    !graphSkipped && trace.neighbourTotal ? `${trace.neighbourTotal} ${plural(trace.neighbourTotal, "сосед", "соседа", "соседей")} по графу` : "",
    trace.pool ? `${trace.pool} ${plural(trace.pool, "кандидат", "кандидата", "кандидатов")}` : "",
    decided ? `в ответ ${trace.selected.length}` : "",
  ]
    .filter(Boolean)
    .join(", ");

  const caption = (() => {
    switch (step) {
      case "seeds":
        return seedCount
          ? `Нашёл в вопросе ${trace.seedTotal} ${plural(trace.seedTotal, "понятие", "понятия", "понятий")} графа, на сцене ${seedCount} главных.`
          : "Разбираю вопрос.";
      case "walk": {
        const shownNeighbours = trace.nodes.filter((n) => n.role === "neighbour" && n.hop === 0).length;
        return `Обход связей добавил ${trace.neighbourTotal} ${plural(trace.neighbourTotal, "соседнее понятие", "соседних понятия", "соседних понятий")}. На схеме — ${shownNeighbours} из них: те, через которые граф дошёл до фрагментов.`;
      }
      case "pool":
        return `Кандидаты: ${trace.textTotal} по тексту${trace.graphTotal ? `, ${trace.graphTotal} через граф` : ""}. Бирюзовые нашёл только граф.`;
      case "rerank":
        return "Переранжирование выстраивает кандидатов по тому, насколько каждый отвечает на вопрос.";
      case "select":
        return `Отобрано ${trace.selected.length} из ${trace.pool || trace.fragments.length}${graphOnlySelected ? `, из них ${graphOnlySelected} ${plural(graphOnlySelected, "нашёл", "нашли", "нашли")} только граф` : ""}. Их и прочитает модель.`;
      default:
        return `Модель пишет ответ по ${trace.selected.length} ${plural(trace.selected.length, "фрагменту", "фрагментам", "фрагментам")}. Номер в ответе совпадает с номером плитки.`;
    }
  })();

  return (
    <div className="trace" ref={boxRef} data-open={open}>
      <button className="trace-toggle" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <Network size={15} />
        <span className="trace-toggle-text">{open ? "Как я ищу" : `Как искал: ${summary || "без графа"}`}</span>
        <ChevronDown size={16} className="trace-chevron" />
      </button>

      {trace.scope && (
        <p className="trace-scope">
          Источник назван в вопросе, ищу только в{" "}
          {trace.scope.titles.slice(0, 2).map((t) => `«${t}»`).join(" и ")}
          {trace.scope.titles.length > 2 ? ` и ещё ${trace.scope.titles.length - 2}` : ""}.
        </p>
      )}

      {open && (
        <>
          <ol className="trace-steps" aria-label="Шаги поиска">
            {STEPS.map((s, i) => {
              const skipped = graphSkipped && (s.id === "seeds" || s.id === "walk");
              const state = skipped ? "skipped" : i < stepIndex ? "done" : i === stepIndex ? "active" : "todo";
              return (
                <li key={s.id} data-state={state} aria-current={state === "active" ? "step" : undefined}>
                  {s.label}
                </li>
              );
            })}
          </ol>

          <svg
            className="trace-svg"
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            role="img"
            aria-label={`Сцена поиска. ${caption}`}
          >
            {/* связи вопроса с затравками и затравок с соседями */}
            <g className="trace-links">
              {trace.nodes
                .filter((n) => n.role === "seed")
                .map((n) => {
                  const p = pos.get(n.id)!;
                  return <line key={`q-${n.id}`} className="link-q" x1={layout.cx} y1={layout.cy} x2={p.x} y2={p.y} />;
                })}
              {trace.links.map((l) => {
                const a = pos.get(l.source);
                const b = pos.get(l.target);
                if (!a || !b) return null;
                return (
                  <line key={`${l.source}>${l.target}`} className="link-g" x1={a.x} y1={a.y} x2={b.x} y2={b.y}>
                    <title>{l.label}</title>
                  </line>
                );
              })}
            </g>

            {/* от понятий к фрагментам, которые нашёл граф */}
            <g className="trace-reach">
              {tiles.map((f, i) => {
                if (!f.found.includes("graph")) return null;
                const t = tileXY(i);
                return f.entities.slice(0, 2).map((name) => {
                  const n = byCanonical.get(name);
                  const p = n && pos.get(n.id);
                  if (!p) return null;
                  const tx = narrow ? t.x + layout.tiles.w / 2 : t.x;
                  const ty = narrow ? t.y : t.y + layout.tiles.h / 2;
                  const mid = narrow ? `${p.x} ${(p.y + ty) / 2}, ${tx} ${(p.y + ty) / 2}` : `${(p.x + tx) / 2} ${p.y}, ${(p.x + tx) / 2} ${ty}`;
                  return (
                    <path
                      key={`${f.id}-${name}`}
                      d={`M ${p.x} ${p.y} C ${mid}, ${tx} ${ty}`}
                      className="reach"
                      data-selected={decided && selected.has(f.id)}
                      data-faded={decided && !selected.has(f.id)}
                    />
                  );
                });
              })}
            </g>

            <g className="trace-question">
              <circle cx={layout.cx} cy={layout.cy} r={13} />
              <text x={layout.cx} y={layout.cy + 4}>?</text>
            </g>

            <g className="trace-nodes">
              {/* Затравки рисуем последними: их подписи важнее и ложатся поверх соседей. */}
              {[...trace.nodes].sort((a, b) => Number(a.role === "seed") - Number(b.role === "seed")).map((n) => {
                const p = pos.get(n.id);
                if (!p) return null;
                const right = Math.cos(p.angle) >= -0.05;
                const centre = n.role === "seed" && trace.nodes.filter((x) => x.role === "seed").length === 1;
                const lx = centre ? p.x : p.x + (right ? 10 : -10);
                const ly = centre ? p.y + 24 : p.y + 4 + (Math.abs(Math.sin(p.angle)) > 0.9 ? (Math.sin(p.angle) > 0 ? 10 : -8) : 0);
                return (
                  <g key={n.id} className="node" data-role={n.role} data-hop={n.hop > 0} data-kind={n.kind}>
                    <circle cx={p.x} cy={p.y} r={n.role === "seed" ? 7 : 4.5} />
                    <text x={lx} y={ly} textAnchor={centre ? "middle" : right ? "start" : "end"}>
                      {shorten(n.name, narrow ? (n.role === "seed" ? 13 : 11) : n.role === "seed" ? 20 : 16)}
                    </text>
                    <title>{n.canonical && n.canonical !== n.name ? `${n.name} — ${n.canonical}` : n.name}</title>
                  </g>
                );
              })}
            </g>

            <g className="trace-tiles">
              {tiles.length > 0 && (
                <text className="tiles-caption" x={layout.tiles.x} y={layout.tiles.y - 10}>
                  Фрагменты-кандидаты
                </text>
              )}
              {tiles.map((f, i) => {
                const { x, y } = tileXY(i);
                const isSelected = selected.has(f.id);
                const channel = f.found.length > 1 ? "both" : f.found[0];
                const n = citationOf.get(f.id) ?? contextIndex.get(f.id);
                const label = `${sourceTitle(f.sourceId)}${f.page ? `, с. ${f.page}` : ""}. ${
                  channel === "graph" ? "Нашёл только граф" : channel === "both" ? "Нашли текст и граф" : "Нашёл поиск по тексту"
                }${isSelected ? ", взят в ответ" : ""}.`;
                return (
                  <g
                    key={f.id}
                    className="tile"
                    data-channel={channel}
                    data-selected={decided && isSelected}
                    data-faded={decided && !isSelected}
                    style={{ transform: `translate(${x}px, ${y}px)` }}
                    role="button"
                    tabIndex={0}
                    aria-label={label}
                    onClick={() => onOpenFragment(f.sourceId, f.id)}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onOpenFragment(f.sourceId, f.id)}
                  >
                    <rect width={layout.tiles.w} height={layout.tiles.h} rx={6} />
                    {decided && isSelected && n !== undefined && (
                      <text x={layout.tiles.w / 2} y={layout.tiles.h / 2 + 4}>
                        {n}
                      </text>
                    )}
                    <title>{`${label}\n${f.preview}`}</title>
                  </g>
                );
              })}
            </g>
          </svg>

          <div className="trace-foot">
            <p className="trace-caption" aria-live="polite">
              {caption}
            </p>
            <ul className="trace-legend" aria-label="Обозначения">
              <li data-channel="text">по тексту</li>
              <li data-channel="graph">только через граф</li>
              <li data-channel="both">и так, и так</li>
            </ul>
          </div>
          {trace.sealQueries.length > 0 && (
            <ul className="trace-seal" aria-label="Дозапросы SEAL">
              {trace.sealQueries.map((q, i) => (
                <li key={i}>Дозапрос: {q}</li>
              ))}
            </ul>
          )}
          {graphFound === 0 && trace.phase === "done" && !graphSkipped && (
            <p className="trace-caption">Граф в этом вопросе ничего нового не добавил.</p>
          )}
        </>
      )}
    </div>
  );
}
