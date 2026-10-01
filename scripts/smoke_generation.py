"""Прочитать несколько ответов модели, прежде чем тратить на неё замер.

Правило, купленное дорого. Ячейка Muse Glimmer 2026-09-03 считалась
полтора часа и оказалась пустой: 325 ответов из 388 — пустота, потому что
модель по умолчанию рассуждает, размышление съедало лимит токенов, а наша
защита честно возвращала пустоту вместо обрывка чужих мыслей. Всё это было
видно на первом же ответе.

Проверка идёт по замороженному контексту, как и сам замер, поэтому проверяет
ровно тот путь, которым пойдут 388 вопросов, а не какой-то отдельный.

    python scripts/smoke_generation.py                 3 вопроса
    python scripts/smoke_generation.py --count 5

Возвращает ненулевой код, если хоть один ответ негоден. Это важнее печати:
скрипт замера обязан на этом останавливаться, а не идти дальше.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.context import build_context  # noqa: E402
from rag_textbook.evaluation.answers import (  # noqa: E402
    latin_share,
    looks_like_reasoning,
)
from rag_textbook.models import Chunk, ScoredChunk  # noqa: E402


def load_chunks(settings: Settings) -> dict[str, Chunk]:
    """Фрагменты берутся из выгрузки разбора, как и в самом замере."""
    chunks: dict[str, Chunk] = {}
    for path in sorted(Path(settings.paths.parsed_dir).rglob("*_chunks.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["chunks"] if isinstance(payload, dict) else payload
        for item in items:
            chunks[item["id"]] = Chunk(**item)
    return chunks


def probe_reasoning(settings: Settings) -> int | None:
    """Длина рассуждения на простом вопросе, в знаках.

    Спрашивается напрямую у движка, а не через наш клиент: клиент отдаёт
    только текст ответа, а рассуждение llama.cpp кладёт в отдельное поле
    ``reasoning_content``. Ноль означает, что модель не рассуждает,
    и тогда медленный ответ объясняется размером, а не цепочкой мыслей.
    """
    import httpx

    try:
        response = httpx.post(
            f"{settings.llm.base_url_for('chat')}/chat/completions",
            json={
                "model": "probe",
                "messages": [
                    {"role": "user", "content": "Ответь одним словом: столица России?"}
                ],
                "max_tokens": 400,
                "temperature": 0.0,
            },
            timeout=300.0,
        )
        message = (response.json().get("choices") or [{}])[0].get("message", {})
    except Exception:  # noqa: BLE001
        return None
    return len(str(message.get("reasoning_content") or ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--trace", default="capture/session-0819/trace-always.jsonl")
    args = parser.parse_args()

    settings = Settings()
    chunks = load_chunks(settings)
    if not chunks:
        print("Нет разобранных фрагментов — проверять не на чем.", file=sys.stderr)
        return 1

    traces = []
    for line in Path(args.trace).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "trace-header":
            continue
        traces.append(record)
        if len(traces) >= args.count:
            break

    context = build_context(settings)
    bad = 0
    elapsed_all: list[float] = []
    try:
        served = ""
        describe = getattr(context.llm, "describe_model", None)
        if callable(describe):
            served = (describe() or {}).get("model", "")
        print(f"Отвечает: {served or settings.llm.model_for('chat')}\n")

        for record in traces:
            picked = [
                ScoredChunk(chunk=chunks[cid], score=1.0)
                for cid in record["final"]
                if cid in chunks
            ]
            _, text, elapsed = context.generator.answer_from_context(
                record["question"], picked
            )
            elapsed_all.append(elapsed / 1000)
            print("=" * 70)
            print("ВОПРОС:", record["question"])
            print(f"ОТВЕТ ({elapsed / 1000:.1f} с, {len(text)} знаков):")
            print(text[:900] if text else "<ПУСТО>")

            problems = []
            if not text.strip():
                problems.append("пустой ответ: скорее всего размышление съело лимит токенов")
            elif looks_like_reasoning(text):
                problems.append("размышление вместо ответа")
            elif latin_share(text) > 0.5:
                problems.append("ответ не по-русски")
            if problems:
                bad += 1
                print("НЕГОДЕН:", "; ".join(problems))
    finally:
        context.close()

    print("=" * 70)
    # Вердикт. Время само по себе рассуждение не выдаёт: крупная модель
    # медленная просто потому, что крупная. Qwen3.8-27B дала медиану 11.4 с
    # и была помечена как «похоже на рассуждение», хотя это мог быть размер.
    # Различает их только счётчик: рассуждение движок кладёт в отдельное поле
    # reasoning_content, и его длина отвечает на вопрос прямо.
    if elapsed_all:
        median = sorted(elapsed_all)[len(elapsed_all) // 2]
        print(f"Время ответа: медиана {median:.1f} с")
    thinking = probe_reasoning(settings)
    if thinking is None:
        print("Рассуждение: проверить не удалось, движок не ответил на прямой запрос")
    elif thinking > 0:
        print(
            f"РАССУЖДАЕТ: {thinking} знаков в reasoning_content на простом вопросе. "
            "Это платится временем карты на каждом ответе."
        )
        print(
            "    Гасить: --reasoning off для семейства Qwen, "
            "--reasoning-effort minimal там, где рассуждение встроено в шаблон."
        )
    else:
        print("Рассуждения нет — время ответа определяется размером модели.")
    if bad:
        print(f"НЕГОДНЫХ ОТВЕТОВ: {bad} из {len(traces)}. Замер запускать нельзя.")
        return 1
    print(f"Все {len(traces)} ответа годны. Можно мерить.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
