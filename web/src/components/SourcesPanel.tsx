import { Check, EllipsisVertical, FileText, Minus, Plus, Trash2, Eye } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { IngestStage, Source } from "../api/types";
import { plural } from "../lib/plural";

const STAGE_LABEL: Record<IngestStage, string> = {
  upload: "Загрузка",
  parse: "Распознаём текст и формулы",
  chunk: "Делим на фрагменты",
  graph: "Строим граф понятий",
};

interface Props {
  sources: Source[];
  activeId?: string;
  onToggle: (id: string, enabled: boolean) => void;
  onToggleAll: (enabled: boolean) => void;
  onOpen: (id: string) => void;
  // Без обработчиков стенд не даёт загружать и удалять: библиотека готова заранее.
  onRemove?: (id: string) => void;
  onAdd?: () => void;
}

export function SourcesPanel({ sources, activeId, onToggle, onToggleAll, onOpen, onRemove, onAdd }: Props) {
  const ready = sources.filter((s) => s.status === "ready");
  const enabled = ready.filter((s) => s.enabled).length;
  const allRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (allRef.current) allRef.current.indeterminate = enabled > 0 && enabled < ready.length;
  }, [enabled, ready.length]);

  return (
    <section className="panel sources" aria-labelledby="sources-title">
      <header className="panel-head">
        <h2 className="panel-title" id="sources-title">
          Источники
        </h2>
      </header>
      <div className="sources-tools">
        {onAdd && (
          <button className="btn add-source" onClick={onAdd}>
            <Plus size={18} /> Добавить документ
          </button>
        )}
        {ready.length > 0 && (
          <label className="select-all">
            <span className="check">
              <input
                ref={allRef}
                type="checkbox"
                checked={enabled === ready.length}
                onChange={(e) => onToggleAll(e.target.checked)}
              />
              <span className="check-box" aria-hidden>
                {enabled === ready.length ? <Check size={14} strokeWidth={3} /> : enabled > 0 ? <Minus size={14} strokeWidth={3} /> : null}
              </span>
            </span>
            Использовать все в ответах
          </label>
        )}
      </div>
      <div className="panel-body">
        {sources.length === 0 ? (
          <div className="empty-doc">
            <div>
              <FileText size={28} />
              <h3>Здесь будут ваши учебники</h3>
              <p>Добавьте PDF — ассистент будет отвечать по нему и показывать, откуда взят ответ.</p>
            </div>
          </div>
        ) : (
          <ul className="source-list">
            {sources.map((source) => (
              <SourceItem
                key={source.id}
                source={source}
                active={source.id === activeId}
                onToggle={onToggle}
                onOpen={onOpen}
                onRemove={onRemove}
              />
            ))}
          </ul>
        )}
      </div>
      <footer className="sources-foot">
        {ready.length === 0
          ? "Нет готовых документов"
          : enabled === 0
            ? "Ни один документ не участвует в ответах"
            : `В ответах ${enabled} из ${ready.length} ${plural(ready.length, "документа", "документов", "документов")}`}
      </footer>
    </section>
  );
}

interface ItemProps {
  source: Source;
  active: boolean;
  onToggle: (id: string, enabled: boolean) => void;
  onOpen: (id: string) => void;
  onRemove?: (id: string) => void;
}

function SourceItem({ source, active, onToggle, onOpen, onRemove }: ItemProps) {
  const [menu, setMenu] = useState(false);
  const ready = source.status === "ready";
  const stages: IngestStage[] = ["upload", "parse", "chunk", "graph"];
  const overall =
    source.stage !== undefined ? (stages.indexOf(source.stage) + (source.progress ?? 0)) / stages.length : 0;

  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(false);
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, [menu]);

  return (
    <li className="source" data-active={active} data-enabled={source.enabled} style={{ position: "relative" }}>
      <span className="check" title={source.enabled ? "Убрать из ответов" : "Использовать в ответах"}>
        <input
          type="checkbox"
          checked={ready && source.enabled}
          disabled={!ready}
          aria-label={`Использовать «${source.title}» в ответах`}
          onChange={(e) => onToggle(source.id, e.target.checked)}
        />
        <span className="check-box" aria-hidden>
          {ready && source.enabled && <Check size={14} strokeWidth={3} />}
        </span>
      </span>
      <div>
        <button className="source-open" onClick={() => ready && onOpen(source.id)} disabled={!ready}>
          <span className="source-title">{source.title}</span>
        </button>
        {ready && (
          <div className="source-meta">
            {source.pages} {plural(source.pages, "страница", "страницы", "страниц")}, {source.formulas.toLocaleString("ru")}{" "}
            {plural(source.formulas, "формула", "формулы", "формул")}
          </div>
        )}
        {source.status === "processing" && source.stage && (
          <div className="source-progress" role="status">
            <div className="progress-label">
              <span>{STAGE_LABEL[source.stage]}</span>
              <span>{Math.round(overall * 100)}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${overall * 100}%` }} />
            </div>
          </div>
        )}
        {source.status === "error" && <div className="source-meta msg-error">{source.error ?? "Не удалось разобрать файл"}</div>}
      </div>
      {onRemove && (
      <button
        className="icon-btn"
        aria-label={`Действия с «${source.title}»`}
        aria-expanded={menu}
        onClick={(e) => {
          e.stopPropagation();
          setMenu((m) => !m);
        }}
      >
        <EllipsisVertical size={18} />
      </button>
      )}
      {menu && onRemove && (
        <div className="menu" role="menu">
          {ready && (
            <button role="menuitem" onClick={() => onOpen(source.id)}>
              <Eye size={16} /> Открыть
            </button>
          )}
          <button role="menuitem" className="danger" onClick={() => onRemove(source.id)}>
            <Trash2 size={16} /> Удалить из блокнота
          </button>
        </div>
      )}
    </li>
  );
}
