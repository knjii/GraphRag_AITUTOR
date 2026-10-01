import { ArrowUp, Clock, GitBranch, SlidersHorizontal, Square, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Citation, Message, Source } from "../api/types";
import { SUGGESTIONS } from "../api/mock";
import { RichText } from "../lib/RichText";
import { plural } from "../lib/plural";
import { TraceScene } from "./TraceScene";

interface Props {
  messages: Message[];
  sources: Source[];
  busy: boolean;
  presetName: string;
  onAsk: (question: string) => void;
  onStop: () => void;
  onClear: () => void;
  onCite: (citation: Citation) => void;
  onOpenFragment: (sourceId: string, fragmentId: string) => void;
  generator: "llm" | "extractive";
  onSettings: () => void;
  onShowSources: () => void;
}

const CHANNEL_LABEL = { dense: "по смыслу", sparse: "по словам", graph: "через граф понятий" } as const;

export function ChatPanel(props: Props) {
  const { messages, sources, busy, presetName, onAsk, onStop, onClear, onCite, onOpenFragment, generator, onSettings, onShowSources } =
    props;
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const enabled = sources.filter((s) => s.status === "ready" && s.enabled);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [draft]);

  const submit = (text = draft) => {
    const question = text.trim();
    if (!question || busy) return;
    onAsk(question);
    setDraft("");
  };

  return (
    <section className="panel chat" aria-labelledby="chat-title">
      <header className="panel-head">
        <h2 className="panel-title" id="chat-title">
          Чат
        </h2>
        <button className="chat-head-preset" onClick={onSettings} title="Изменить режим поиска">
          <SlidersHorizontal size={14} /> {presetName}
        </button>
        {messages.length > 0 && (
          <button className="icon-btn" aria-label="Очистить чат" title="Очистить чат" onClick={onClear} disabled={busy}>
            <Trash2 size={18} />
          </button>
        )}
      </header>

      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-column">
          {messages.length === 0 ? (
            <div className="empty-chat">
              <h2>Спросите что-нибудь по своим учебникам</h2>
              <p>
                {enabled.length === 0
                  ? "Сейчас ни один документ не участвует в ответах. Отметьте источники слева."
                  : `Ответ строится только по ${enabled.length} ${plural(enabled.length, "отмеченному документу", "отмеченным документам", "отмеченным документам")}, а каждое утверждение ведёт к странице, откуда оно взято.`}
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="suggestion" onClick={() => submit(s)} disabled={busy}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m) => (
              <MessageView key={m.id} message={m} sources={sources} onCite={onCite} onOpenFragment={onOpenFragment} />
            ))
          )}
        </div>
      </div>

      <div className="composer-wrap">
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <label htmlFor="question" className="visually-hidden">
            Вопрос
          </label>
          <textarea
            id="question"
            ref={inputRef}
            rows={1}
            value={draft}
            placeholder="Задайте вопрос по документам"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />
          {busy ? (
            <button type="button" className="btn btn-quiet send" aria-label="Остановить ответ" onClick={onStop}>
              <Square size={16} fill="currentColor" />
            </button>
          ) : (
            <button type="submit" className="btn btn-primary send" aria-label="Отправить" disabled={!draft.trim()}>
              <ArrowUp size={20} />
            </button>
          )}
        </form>
        <p className="composer-note">
          {enabled.length === 0 ? (
            <>
              Нет документов в контексте. <button onClick={onShowSources}>Выбрать источники</button>
            </>
          ) : (
            `Ищу в ${enabled.length} ${plural(enabled.length, "документе", "документах", "документах")}. ${
              generator === "extractive"
                ? "Модель ответа не подключена: вместо ответа покажу выписку из найденного."
                : "Проверяйте важное по ссылкам на страницы."
            }`
          )}
        </p>
      </div>
    </section>
  );
}

function MessageView({
  message,
  sources,
  onCite,
  onOpenFragment,
}: {
  message: Message;
  sources: Source[];
  onCite: (c: Citation) => void;
  onOpenFragment: (sourceId: string, fragmentId: string) => void;
}) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const byN = new Map(message.citations.map((c) => [c.n, c]));
  const titleOf = (id: string) => sources.find((s) => s.id === id)?.title ?? "Удалённый документ";

  return (
    <div className="msg msg-assistant">
      {message.trace && (
        <TraceScene
          trace={message.trace}
          citations={message.citations}
          answering={!!message.content}
          onOpenFragment={onOpenFragment}
          sourceTitle={titleOf}
        />
      )}
      {message.pending && !message.content ? (
        message.trace ? null : (
        <div className="searching" role="status">
          <span className="searching-bar" aria-hidden />
          Ищу в документах
        </div>
        )
      ) : (
        <div className="answer" aria-live={message.pending ? "polite" : undefined}>
          <RichText
            text={message.content}
            renderCitation={(n) => {
              const c = byN.get(n);
              if (!c) return null;
              return (
                <button
                  className="cite"
                  data-channel={c.channel}
                  title={`${titleOf(c.sourceId)}, с. ${c.page}. Найдено ${CHANNEL_LABEL[c.channel]}`}
                  onClick={() => onCite(c)}
                >
                  {n}
                </button>
              );
            }}
          />
        </div>
      )}
      {message.error && <p className="msg-error">{message.error}</p>}
      {!message.pending && message.citations.length > 0 && (
        <div className="answer-sources" aria-label="Источники ответа">
          {message.citations.map((c) => (
            <button key={c.n} className="source-card" onClick={() => onCite(c)}>
              <span className="cite" data-channel={c.channel}>
                {c.n}
              </span>
              <span style={{ minWidth: 0 }}>
                <span className="source-card-title" style={{ display: "block" }}>
                  {c.section}
                </span>
                <span className="source-card-meta">
                  {titleOf(c.sourceId)}, с. {c.page}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
      {!message.pending && message.timings && (
        <div className="answer-meta">
          {message.multiHop && (
            <span className="badge-hop" title="Ответ собран из нескольких мест, связанных через граф понятий">
              <GitBranch size={13} /> Связал несколько мест
            </span>
          )}
          <span title="Поиск фрагментов и генерация ответа">
            <Clock size={13} /> {fmt(message.timings.retrievalMs)} поиск, {fmt(message.timings.generationMs)} ответ
          </span>
        </div>
      )}
    </div>
  );
}

const fmt = (ms: number) => `${(ms / 1000).toLocaleString("ru", { maximumFractionDigits: 1, minimumFractionDigits: 1 })} с`;
