import { BookOpen, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { Fragment, Source } from "../api/types";
import { RichText } from "../lib/RichText";
import { plural } from "../lib/plural";

interface Props {
  source?: Source;
  highlightId?: string;
  onClose: () => void;
}

// Книга целиком — тысячи фрагментов с формулами, поэтому грузим окно
// вокруг нужного места и подгружаем соседние по кнопкам.
export function ViewerPanel({ source, highlightId, onClose }: Props) {
  const [fragments, setFragments] = useState<Fragment[] | null>(null);
  const [range, setRange] = useState({ offset: 0, end: 0, total: 0 });
  const [error, setError] = useState("");
  const highlightRef = useRef<HTMLDivElement>(null);
  const sourceId = source?.id;

  useEffect(() => {
    if (!sourceId) return;
    if (highlightId && fragments?.some((f) => f.id === highlightId)) return;
    let alive = true;
    setFragments(null);
    setError("");
    api
      .fragments(sourceId, { around: highlightId })
      .then((page) => {
        if (!alive) return;
        setFragments(page.items);
        setRange({ offset: page.offset, end: page.offset + page.items.length, total: page.total });
      })
      .catch((err: Error) => alive && setError(err.message));
    return () => {
      alive = false;
    };
    // fragments не в зависимостях: окно перезагружается только при смене места
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceId, highlightId]);

  const more = async (direction: "before" | "after") => {
    if (!sourceId || !fragments) return;
    const offset = direction === "before" ? Math.max(0, range.offset - 40) : range.end;
    const page = await api.fragments(sourceId, { offset });
    if (direction === "before") {
      const items = page.items.filter((f) => !fragments.some((x) => x.id === f.id));
      setFragments([...items, ...fragments]);
      setRange((r) => ({ ...r, offset: page.offset, total: page.total }));
    } else {
      setFragments([...fragments, ...page.items]);
      setRange((r) => ({ ...r, end: page.offset + page.items.length, total: page.total }));
    }
  };

  useEffect(() => {
    highlightRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [fragments, highlightId]);

  if (!source) {
    return (
      <section className="panel viewer" aria-label="Документ">
        <div className="empty-doc">
          <div>
            <BookOpen size={28} />
            <h3>Документ откроется здесь</h3>
            <p>Нажмите на источник слева или на номер ссылки в ответе — покажу нужную страницу.</p>
          </div>
        </div>
      </section>
    );
  }

  let lastPage = -1;
  return (
    <section className="panel viewer" aria-labelledby="viewer-title">
      <header className="panel-head">
        <div style={{ flex: 1, minWidth: 0 }}>
          <h2 className="panel-title" id="viewer-title" title={source.title}>
            {source.title}
          </h2>
          <div className="doc-head-meta">
            {source.authors ? `${source.authors}, ` : ""}
            {source.pages} {plural(source.pages, "страница", "страницы", "страниц")}
          </div>
        </div>
        <button className="icon-btn" aria-label="Закрыть документ" onClick={onClose}>
          <X size={18} />
        </button>
      </header>
      <div className="panel-body">
        <div className="doc-body">
          {fragments === null && !error && <p className="doc-head-meta">Открываю…</p>}
          {error && <p className="msg-error">Документ не открылся: {error}.</p>}
          {fragments && range.offset > 0 && (
            <button className="btn btn-quiet doc-more" onClick={() => void more("before")}>
              Показать раньше
            </button>
          )}
          {fragments?.length === 0 && <p className="doc-head-meta">В документе пока нет разобранных фрагментов.</p>}
          {fragments?.map((f) => {
            const showPage = f.page !== lastPage;
            lastPage = f.page;
            const highlighted = f.id === highlightId;
            return (
              <div key={f.id}>
                {showPage && <div className="doc-page">Страница {f.page}</div>}
                <div className="doc-fragment" data-highlight={highlighted} ref={highlighted ? highlightRef : undefined}>
                  {f.section && <div className="doc-section">{f.section}</div>}
                  <RichText text={f.text} />
                </div>
              </div>
            );
          })}
          {fragments && range.end < range.total && (
            <button className="btn btn-quiet doc-more" onClick={() => void more("after")}>
              Показать дальше
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
