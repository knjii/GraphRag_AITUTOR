import katex from "katex";
import "katex/dist/katex.min.css";
import { Fragment as ReactFragment, type ReactNode } from "react";

// Минимальная разметка ответа: абзацы, формулы $…$ / $$…$$, **жирный**
// и ссылки на источники [n]. Полноценный Markdown здесь не нужен,
// а формулы обязаны рендериться всегда — продукт про математику.

function renderMath(tex: string, display: boolean) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
  return display ? (
    <div className="math-block" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span className="math-inline" dangerouslySetInnerHTML={{ __html: html }} />
  );
}

interface Props {
  text: string;
  renderCitation?: (n: number) => ReactNode;
}

// Модель иногда пишет формулы в скобках \( \) и \[ \] — приводим к долларам.
const normalize = (text: string) =>
  text
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex: string) => `$$${tex}$$`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex: string) => `$${tex}$`);

export function RichText({ text, renderCitation }: Props) {
  type Block = { math: string } | { para: string };
  const blocks: Block[] = normalize(text).split(/(\$\$[\s\S]*?\$\$)/g).flatMap((chunk) =>
    chunk.startsWith("$$") && chunk.endsWith("$$") && chunk.length > 4
      ? [{ math: chunk.slice(2, -2).trim() } as Block]
      : chunk
          .split(/\n{2,}/)
          .map((p) => p.trim())
          .filter(Boolean)
          .map((p): Block => ({ para: p })),
  );

  return (
    <>
      {blocks.map((block, i) =>
        "math" in block ? (
          <ReactFragment key={i}>{renderMath(block.math, true)}</ReactFragment>
        ) : (
          <p key={i}>{inline(block.para, renderCitation)}</p>
        ),
      )}
    </>
  );
}

function inline(text: string, renderCitation?: (n: number) => ReactNode): ReactNode[] {
  const out: ReactNode[] = [];
  const pattern = /\$([^$]+)\$|\*\*([^*]+)\*\*|\[(\d+)\]/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > last) out.push(text.slice(last, index));
    if (match[1] !== undefined) out.push(<ReactFragment key={index}>{renderMath(match[1], false)}</ReactFragment>);
    else if (match[2] !== undefined) out.push(<strong key={index}>{match[2]}</strong>);
    else if (match[3] !== undefined)
      out.push(<ReactFragment key={index}>{renderCitation ? renderCitation(Number(match[3])) : match[0]}</ReactFragment>);
    last = index + match[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
