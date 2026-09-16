"""Сырой ответ движка, без нашей обработки.

Нужен, когда ответ приходит пустым и непонятно, кто виноват: модель,
шаблон чата движка или наш разбор. Печатает JSON как есть.

    python scripts/raw_probe.py
    python scripts/raw_probe.py --url http://127.0.0.1:8001/v1
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--text", default="Ответь одним словом по-русски: столица России?"
    )
    args = parser.parse_args()

    payload = {
        "model": "probe",
        "messages": [{"role": "user", "content": args.text}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    response = httpx.post(
        f"{args.url}/chat/completions", json=payload, timeout=600.0
    )
    print("код:", response.status_code)
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        print(response.text[:2000])
        return 1

    print(json.dumps(data, ensure_ascii=False, indent=2)[:4000])

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    usage = data.get("usage", {})
    print("\n--- разбор:")
    print("токенов сгенерировано:", usage.get("completion_tokens"))
    print("finish_reason:", choice.get("finish_reason"))
    for key in ("content", "reasoning_content", "reasoning"):
        value = message.get(key)
        if value:
            print(f"{key}: {len(str(value))} знаков, начало: {str(value)[:200]!r}")
        else:
            print(f"{key}: пусто")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
