import { FileUp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface Props {
  onClose: () => void;
  onFiles: (files: File[]) => void;
}

const MAX_MB = 200;

export function UploadDialog({ onClose, onFiles }: Props) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const accept = (list: FileList | null) => {
    const files = Array.from(list ?? []);
    const pdfs = files.filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
    const tooBig = pdfs.find((f) => f.size > MAX_MB * 1024 * 1024);
    if (pdfs.length === 0) return setError("Нужен файл в формате PDF.");
    if (tooBig) return setError(`«${tooBig.name}» больше ${MAX_MB} МБ. Разделите его на части.`);
    onFiles(pdfs);
  };

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="upload-title">
        <button className="icon-btn dialog-close" aria-label="Закрыть" onClick={onClose}>
          <X size={18} />
        </button>
        <h2 id="upload-title">Добавить документ</h2>
        <p className="dialog-lead">
          Учебник, конспект или задачник в PDF. Формулы распознаются и остаются формулами, а не картинками.
        </p>
        <div
          className="dropzone"
          data-over={over}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            accept(e.dataTransfer.files);
          }}
        >
          <FileUp size={30} />
          <strong>Перетащите PDF сюда</strong>
          <small>или</small>
          <button className="btn btn-primary" onClick={() => inputRef.current?.click()}>
            Выбрать файл
          </button>
          <small>До {MAX_MB} МБ, можно несколько файлов</small>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            hidden
            onChange={(e) => accept(e.target.files)}
          />
        </div>
        {error && (
          <p className="msg-error" role="alert" style={{ margin: "12px 0 0", fontSize: 14 }}>
            {error}
          </p>
        )}
        <ul className="stages" aria-label="Что произойдёт после загрузки">
          <li>
            <b>Распознавание</b> текст, таблицы и формулы в LaTeX
          </li>
          <li>
            <b>Фрагменты</b> делим так, чтобы не рвать формулы
          </li>
          <li>
            <b>Граф понятий</b> связи между определениями и теоремами
          </li>
        </ul>
      </div>
    </>
  );
}
