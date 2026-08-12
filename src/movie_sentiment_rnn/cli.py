"""Command-line utilities for configuration and dataset quality reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from movie_sentiment_rnn.config import ProjectConfig
from movie_sentiment_rnn.data import build_quality_report, load_reviews_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line argument parser."""
    parser = argparse.ArgumentParser(description="Movie Sentiment RNN utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quality = subparsers.add_parser("quality-report", help="Print a JSON data quality report")
    quality.add_argument("csv_path", type=Path, help="Path to the movie reviews CSV")
    quality.add_argument("--min-words", type=int, default=4)

    config = subparsers.add_parser("show-config", help="Print resolved project configuration")
    config.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the requested utility command and return its exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "quality-report":
        frame = load_reviews_csv(args.csv_path)
        report = build_quality_report(frame, min_words=args.min_words)
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "show-config":
        config = ProjectConfig.from_env(args.env_file)
        print(json.dumps(_serialise_config(config), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _serialise_config(config: ProjectConfig) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }
