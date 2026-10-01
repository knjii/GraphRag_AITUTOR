import { BookOpen, FileStack, MessagesSquare, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api/client";
import type { Citation, Message, Preset, RetrievalSettings, ServiceInfo, Source, TraceEvent } from "./api/types";
import { ChatPanel } from "./components/ChatPanel";
import { Rail } from "./components/Rail";
import { SettingsSheet } from "./components/SettingsSheet";
import { SourcesPanel } from "./components/SourcesPanel";
import { UploadDialog } from "./components/UploadDialog";
import { ViewerPanel } from "./components/ViewerPanel";
import { applyTrace, emptyTrace, pacedQueue } from "./lib/trace";

type Tab = "sources" | "chat" | "viewer";
type Theme = "light" | "dark";

const uid = () => Math.random().toString(36).slice(2, 10);

function readTheme(): Theme {
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* хранилище недоступно — берём системную тему */
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function App() {
  const [info, setInfo] = useState<ServiceInfo | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [presetId, setPresetId] = useState<string | null>("balanced");
  const [settings, setSettings] = useState<RetrievalSettings | null>(null);
  const [open, setOpen] = useState<{ sourceId: string; fragmentId?: string } | null>(null);
  const [panel, setPanel] = useState<"settings" | "upload" | "help" | null>(null);
  const [tab, setTab] = useState<Tab>("chat");
  const [theme, setTheme] = useState<Theme>(readTheme);
  const [toast, setToast] = useState("");
  const [offline, setOffline] = useState(false);
  const runRef = useRef<{ abort: () => void } | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* не критично */
    }
  }, [theme]);

  useEffect(() => {
    // Сервис мог ещё подниматься: пробуем снова, пока не ответит.
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const [i, list] = await Promise.all([api.info(), api.listSources()]);
        if (!alive) return;
        setInfo(i);
        setSettings(i.presets[0].settings);
        setPresetId(i.presets[0].id);
        setSources(list);
        setOffline(false);
      } catch {
        if (!alive) return;
        setOffline(true);
        timer = setTimeout(load, 3000);
      }
    };
    void load();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 3200);
    return () => clearTimeout(t);
  }, [toast]);

  const presetName = useMemo(
    () => info?.presets.find((p) => p.id === presetId)?.name ?? "Свои параметры",
    [info, presetId],
  );

  const toggle = useCallback((id: string, enabled: boolean) => {
    // Выбор документов живёт в браузере и уходит с каждым вопросом.
    setSources((list) => list.map((s) => (s.id === id ? { ...s, enabled } : s)));
  }, []);

  const toggleAll = useCallback(
    (enabled: boolean) => {
      for (const s of sources) if (s.status === "ready" && s.enabled !== enabled) toggle(s.id, enabled);
    },
    [sources, toggle],
  );

  const remove = useCallback(
    (id: string) => {
      const source = sources.find((s) => s.id === id);
      setSources((list) => list.filter((s) => s.id !== id));
      if (open?.sourceId === id) setOpen(null);
      void api.remove(id);
      if (source) setToast(`«${source.title}» удалён из блокнота`);
    },
    [sources, open],
  );

  const upload = useCallback((files: File[]) => {
    setPanel(null);
    setTab("sources");
    for (const file of files) {
      void api
        .upload(file, (update) =>
          setSources((list) =>
            list.some((s) => s.id === update.id) ? list.map((s) => (s.id === update.id ? update : s)) : [update, ...list],
          ),
        )
        .then((done) => setToast(`«${done.title}» готов к вопросам`));
    }
  }, []);

  const ask = useCallback(
    async (question: string) => {
      if (!settings) return;
      runRef.current?.abort();
      const answerId = uid();
      const controller = new AbortController();
      setBusy(true);
      setTab("chat");
      setMessages((list) => [
        ...list,
        { id: uid(), role: "user", content: question, citations: [] },
        { id: answerId, role: "assistant", content: "", citations: [], pending: true, trace: emptyTrace() },
      ]);
      const patch = (p: (m: Message) => Partial<Message>) =>
        setMessages((list) => list.map((m) => (m.id === answerId ? { ...m, ...p(m) } : m)));

      const show = (event: TraceEvent) => {
        if (event.type === "token") patch(() => ({ content: event.data.text }));
        else if (event.type === "done")
          patch((m) => ({
            content: event.data.text ?? m.content,
            citations: event.data.citations,
            timings: event.data.timings,
            multiHop: event.data.multiHop,
            pending: false,
            trace: m.trace && applyTrace(m.trace, event),
          }));
        else if (event.type === "error") patch((m) => ({ pending: false, error: `Ответ не получен: ${event.data.message}.`, trace: m.trace && applyTrace(m.trace, event) }));
        else patch((m) => ({ trace: m.trace && applyTrace(m.trace, event) }));
      };
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const queue = pacedQueue(show, reduced);
      runRef.current = {
        abort: () => {
          controller.abort();
          queue.cancel();
        },
      };

      const sourceIds = sources.filter((s) => s.status === "ready" && s.enabled).map((s) => s.id);
      try {
        await api.ask({ question, sourceIds, presetId, settings }, queue.push, controller.signal);
        await queue.drained();
      } catch (err) {
        if (!controller.signal.aborted)
          patch(() => ({ pending: false, error: `Ответ не получен: ${err instanceof Error ? err.message : "сервис недоступен"}.` }));
      } finally {
        if (!controller.signal.aborted) {
          patch((m) => (m.pending ? { pending: false } : {}));
          runRef.current = null;
          setBusy(false);
        }
      }
    },
    [settings, sources, presetId],
  );

  const stop = useCallback(() => {
    runRef.current?.abort();
    runRef.current = null;
    setBusy(false);
    setMessages((list) =>
      list.map((m) => (m.pending ? { ...m, pending: false, content: m.content || "Ответ остановлен." } : m)),
    );
  }, []);

  const openFragment = useCallback((sourceId: string, fragmentId?: string) => {
    setOpen({ sourceId, fragmentId });
    setTab("viewer");
  }, []);

  const cite = useCallback((c: Citation) => openFragment(c.sourceId, c.fragmentId), [openFragment]);

  const choosePreset = (preset: Preset) => {
    setPresetId(preset.id);
    setSettings(preset.settings);
  };

  const changeSettings = (next: RetrievalSettings) => {
    setSettings(next);
    const match = info?.presets.find((p) => JSON.stringify(p.settings) === JSON.stringify(next));
    setPresetId(match?.id ?? null);
  };

  const openSource = sources.find((s) => s.id === open?.sourceId);

  return (
    <div className="app" data-viewer={open ? "open" : "closed"} data-tab={tab}>
      <Rail
        mode={info?.mode ?? "local"}
        theme={theme}
        settingsOpen={panel === "settings"}
        onSettings={() => setPanel("settings")}
        onTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        onHelp={() => setPanel("help")}
      />
      <SourcesPanel
        sources={sources}
        activeId={open?.sourceId}
        onToggle={toggle}
        onToggleAll={toggleAll}
        onOpen={(id) => {
          setOpen({ sourceId: id });
          setTab("viewer");
        }}
        onRemove={info?.uploads ? remove : undefined}
        onAdd={info?.uploads ? () => setPanel("upload") : undefined}
      />
      <ChatPanel
        messages={messages}
        sources={sources}
        busy={busy}
        presetName={presetName}
        onAsk={ask}
        onStop={stop}
        onClear={() => setMessages([])}
        onCite={cite}
        onOpenFragment={openFragment}
        generator={info?.generator ?? "llm"}
        onSettings={() => setPanel("settings")}
        onShowSources={() => setTab("sources")}
      />
      {(open || tab === "viewer") && (
        <ViewerPanel
          source={openSource}
          highlightId={open?.fragmentId}
          onClose={() => {
            setOpen(null);
            setTab("chat");
          }}
        />
      )}

      <nav className="tabbar" role="tablist" aria-label="Разделы">
        {(
          [
            ["sources", "Источники", FileStack],
            ["chat", "Чат", MessagesSquare],
            ["viewer", "Документ", BookOpen],
          ] as const
        ).map(([id, label, Icon]) => (
          <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>
            <Icon size={20} />
            {label}
          </button>
        ))}
      </nav>

      {panel === "upload" && <UploadDialog onClose={() => setPanel(null)} onFiles={upload} />}
      {panel === "settings" && info && settings && (
        <SettingsSheet
          info={info}
          presetId={presetId}
          settings={settings}
          onPreset={choosePreset}
          onChange={changeSettings}
          onClose={() => setPanel(null)}
        />
      )}
      {panel === "help" && <HelpDialog onClose={() => setPanel(null)} />}
      {offline && (
        <div className="toast" role="status">
          Сервис поиска не отвечает. Пробую подключиться снова…
        </div>
      )}
      {toast && !offline && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}

function HelpDialog({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="help-title">
        <button className="icon-btn dialog-close" aria-label="Закрыть" onClick={onClose}>
          <X size={18} />
        </button>
        <h2 id="help-title">Как отвечает ассистент</h2>
        <p className="dialog-lead">Ответ строится только по документам, отмеченным слева.</p>
        <ol className="stages" style={{ paddingLeft: 18, listStyle: "decimal" }}>
          <li>Ищет подходящие фрагменты по смыслу, по словам и по связям между понятиями.</li>
          <li>Отбирает несколько фрагментов, которые вместе отвечают на вопрос.</li>
          <li>Пишет ответ и ставит номер у каждого утверждения — по номеру откроется страница источника.</li>
        </ol>
      </div>
    </>
  );
}
