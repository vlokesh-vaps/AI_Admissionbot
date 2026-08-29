"""Primary command-line entry point for the admission chatbot.

Usage:
    python main.py                 Start the API server.
    python main.py api             Start the API server.
    python main.py chat            Start the interactive CLI chat.
    python main.py build-kb        Build the Qdrant knowledge base.
"""

from __future__ import annotations

import argparse
from typing import Sequence


def _run_api() -> None:
    import uvicorn
    from src.config import settings

    uvicorn.run("src.api.app:app", host=settings.api_host, port=settings.api_port, reload=True)


def _run_chat() -> None:
    from scripts.cli_chat import main as chat_main

    chat_main()


def _run_build_kb(no_reset: bool, mi_id: str | None) -> None:
    from scripts.build_kb import build_knowledge_base
    from src.config import settings
    from src.utils.logging import configure_logging

    configure_logging(settings.log_dir, settings.log_level)
    indexed = build_knowledge_base(reset=not no_reset, mi_id=mi_id)
    print(f"Indexed {indexed} chunks into Qdrant collection '{settings.qdrant_collection}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Admission Bot command hub",
        epilog="Use 'python main.py <command> --help' for command details.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("api", help="Start the FastAPI server")
    subparsers.add_parser("chat", help="Start the interactive terminal chat")
    build_parser = subparsers.add_parser("build-kb", help="Build the Qdrant knowledge base")
    build_parser.add_argument("--no-reset", action="store_true", help="Keep existing Qdrant records")
    build_parser.add_argument("--mi-id", type=str, help="Build one tenant knowledge base")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = args.command or "api"
    if command == "api":
        _run_api()
    elif command == "chat":
        _run_chat()
    elif command == "build-kb":
        _run_build_kb(args.no_reset, args.mi_id)


if __name__ == "__main__":
    main()
