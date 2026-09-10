"""Deciding whether a piece of text is a path to a report or the report itself.

The summarizer accepts either, and this is the only place that distinction is made.
Kept free of graph and LLM imports so it stays trivially testable.
"""

from __future__ import annotations

import shlex
from pathlib import Path

__all__ = ["UnsupportedSource", "resolve_source"]

MAX_BYTES = 200_000


class UnsupportedSource(Exception):
    """Raised when the user pointed at a real file we deliberately will not read."""


def _candidate_tokens(text: str) -> list[str]:
    """Tokens that could plausibly be a filesystem path.

    shlex handles quoted paths; it raises on unbalanced quotes in ordinary prose, so
    fall back to a plain split in that case.
    """
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    return [token.strip("\"'`,;:") for token in tokens]


def resolve_source(text: str | None) -> str | None:
    """Return the contents of a `.txt` file referenced in `text`, else None.

    None means "no path here, treat the text itself as the report". An existing file we
    will not read raises `UnsupportedSource` rather than returning None, so the user gets
    told why instead of having their file path summarized as if it were prose.
    """
    if not text or not text.strip():
        return None

    for token in _candidate_tokens(text):
        if "." not in token:
            continue
        path = Path(token).expanduser()
        if not path.is_file():
            continue
        if path.suffix.lower() != ".txt":
            raise UnsupportedSource(
                f"{path.name}: only .txt reports are supported for now."
            )
        if path.stat().st_size > MAX_BYTES:
            raise UnsupportedSource(
                f"{path.name} is larger than {MAX_BYTES // 1000}kB; trim it first."
            )
        content = path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            raise UnsupportedSource(f"{path.name} is empty.")
        return content

    return None
