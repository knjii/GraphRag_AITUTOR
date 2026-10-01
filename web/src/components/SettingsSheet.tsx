import { Lock, X } from "lucide-react";
import { useEffect } from "react";
import type { Preset, RetrievalSettings, RouterMode, Selection, ServiceInfo } from "../api/types";

interface Props {
  info: ServiceInfo;
  presetId: string | null;
  settings: RetrievalSettings;
  onPreset: (preset: Preset) => void;
  onChange: (settings: RetrievalSettings) => void;
  onClose: () => void;
}

const ROUTER: { value: RouterMode; label: string }[] = [
  { value: "auto", label: "Сам решает" },
  { value: "always", label: "Всегда" },
  { value: "off", label: "Выключен" },
];

const SELECTION: { value: Selection; label: string }[] = [
  { value: "off", label: "По рангу" },
  { value: "setr", label: "SetR" },
  { value: "seal", label: "SEAL" },
];

export function SettingsSheet({ info, presetId, settings, onPreset, onChange, onClose }: Props) {
  const locked = info.mode === "server";
  const set = <K extends keyof RetrievalSettings>(key: K, value: RetrievalSettings[K]) =>
    onChange({ ...settings, [key]: value });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="sheet" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header className="panel-head">
          <h2 className="panel-title" id="settings-title">
            Настройки поиска
          </h2>
          <button className="icon-btn" aria-label="Закрыть" onClick={onClose}>
            <X size={18} />
          </button>
        </header>
        <div className="sheet-body">
          <div className="field-group" role="radiogroup" aria-labelledby="presets-title">
            <h3 id="presets-title">Режим</h3>
            {info.presets.map((p) => (
              <button key={p.id} className="preset" role="radio" aria-checked={p.id === presetId} onClick={() => onPreset(p)}>
                <span className="preset-dot" aria-hidden />
                <span>
                  <span className="preset-name">{p.name}</span>
                  <span className="preset-hint" style={{ display: "block" }}>
                    {p.hint}
                  </span>
                </span>
              </button>
            ))}
            {presetId === null && <p className="field-hint">Сейчас свои параметры — ни один режим не выбран.</p>}
          </div>

          <div className="field-group">
            <h3>Тонкая настройка</h3>
            {locked && (
              <div className="locked-note">
                <Lock size={16} style={{ flex: "none", marginTop: 2 }} />
                <span>На сервере доступны только готовые режимы. Параметры меняются при локальном запуске.</span>
              </div>
            )}
            <fieldset disabled={locked}>
              <legend className="visually-hidden">Параметры поиска</legend>
              <label className="field">
                <span className="field-label">
                  Фрагментов в ответе <output>{settings.topK}</output>
                </span>
                <input
                  type="range"
                  min={3}
                  max={16}
                  value={settings.topK}
                  onChange={(e) => set("topK", Number(e.target.value))}
                />
                <span className="field-hint">Больше — полнее контекст, но больше шума и дольше ответ.</span>
              </label>

              <label className="field">
                <span className="field-label">
                  Вес графа понятий <output>{settings.graphWeight.toFixed(2)}</output>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={settings.graphWeight}
                  onChange={(e) => set("graphWeight", Number(e.target.value))}
                />
                <span className="field-hint">Насколько доверять связям между понятиями рядом с поиском по тексту.</span>
              </label>

              <div className="field">
                <span className="field-label" id="router-label">
                  Обход графа
                </span>
                <Segmented labelledBy="router-label" options={ROUTER} value={settings.router} onChange={(v) => set("router", v)} />
              </div>

              <div className="field">
                <span className="field-label" id="selection-label">
                  Отбор фрагментов
                </span>
                <Segmented
                  labelledBy="selection-label"
                  options={SELECTION.filter((o) => info.selections.includes(o.value))}
                  value={settings.selection}
                  // SEAL оценивает пары реранкером: без него режим не запустится.
                  onChange={(v) => onChange({ ...settings, selection: v, reranker: v === "seal" ? true : settings.reranker })}
                />
                <span className="field-hint">
                  {info.selections.length > 1
                    ? "SetR отбирает набор целиком, SEAL дособирает недостающие звенья."
                    : "SetR и SEAL работают через модель ответа, а она в этом запуске не подключена."}
                </span>
              </div>

              <div className="field switch-row">
                <span>
                  <span className="field-label" id="reranker-label">
                    Переранжирование
                  </span>
                  <span className="field-hint">
                    {settings.selection === "seal"
                      ? "Для SEAL обязательно: им он оценивает пары фрагментов."
                      : "Второй проход, уточняющий порядок фрагментов."}
                  </span>
                </span>
                <button
                  className="switch"
                  role="switch"
                  aria-checked={settings.reranker}
                  aria-labelledby="reranker-label"
                  disabled={settings.selection === "seal"}
                  onClick={() => set("reranker", !settings.reranker)}
                />
              </div>

              <label className="field">
                <span className="field-label">Модель ответа</span>
                <select value={settings.model} onChange={(e) => set("model", e.target.value)}>
                  {info.models.map((m) => (
                    <option key={m}>{m}</option>
                  ))}
                </select>
              </label>
            </fieldset>
          </div>
        </div>
      </aside>
    </>
  );
}

function Segmented<T extends string>({
  options,
  value,
  onChange,
  labelledBy,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  labelledBy: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-labelledby={labelledBy}>
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
