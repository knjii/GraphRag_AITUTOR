"""Запуск демо-сервиса.

    python web/server/run.py                       # службы из .env, как у сервиса
    python web/server/run.py --offline graph.json.gz

``--repo`` указывает, чей код ``rag_textbook`` брать: демо работает поверх
рабочей ветки и сам пакет не меняет. ``--offline`` поднимает всё из файла
графа, без Qdrant, Neo4j и модели.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо-сервис «Матчасть»")
    parser.add_argument("--repo", default=str(HERE.parent.parent), help="корень репозитория с пакетом rag_textbook")
    parser.add_argument("--offline", default="", help="файл графа rag-graph: всё из него, без служб")
    parser.add_argument("--mode", choices=["local", "server"], default=os.environ.get("DEMO_MODE", "local"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["DEMO_MODE"] = args.mode
    if args.offline:
        os.environ["DEMO_OFFLINE_GRAPH"] = str(Path(args.offline).resolve())
    sys.path[:0] = [str(Path(args.repo).resolve()), str(HERE)]

    import uvicorn

    from demo_api import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
