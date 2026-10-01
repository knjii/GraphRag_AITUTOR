import { CircleHelp, Moon, SlidersHorizontal, Sun } from "lucide-react";
import type { DemoMode } from "../api/types";

interface Props {
  mode: DemoMode;
  theme: "light" | "dark";
  settingsOpen: boolean;
  onSettings: () => void;
  onTheme: () => void;
  onHelp: () => void;
}

export function Rail({ mode, theme, settingsOpen, onSettings, onTheme, onHelp }: Props) {
  return (
    <nav className="rail" aria-label="Приложение">
      <div className="rail-logo" title="Матчасть — ассистент по учебникам">
        Мч
      </div>
      <button
        className="rail-button"
        aria-pressed={settingsOpen}
        aria-label="Настройки поиска"
        title="Настройки поиска"
        onClick={onSettings}
      >
        <SlidersHorizontal size={20} />
      </button>
      <button
        className="rail-button"
        aria-label={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
        title={theme === "dark" ? "Светлая тема" : "Тёмная тема"}
        onClick={onTheme}
      >
        {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
      </button>
      <button className="rail-button" aria-label="Как это работает" title="Как это работает" onClick={onHelp}>
        <CircleHelp size={20} />
      </button>
      <div className="rail-spacer" />
      <div className="rail-mode" title={mode === "local" ? "Все параметры доступны" : "Доступны готовые режимы"}>
        {mode === "local" ? "Локально" : "Сервер"}
      </div>
    </nav>
  );
}
