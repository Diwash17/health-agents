#!/usr/bin/env python3
"""Entrypoint: `uv run main.py`."""

import sys
import warnings

# langgraph 1.0.5 emits a pending-deprecation warning from its own serializer import —
# nothing for us to act on, but it would print above the banner on every start.
# It has to be silenced in this order: langchain_core prepends its own "always" filter
# for this warning category at import time, and the first matching filter wins, so our
# filter must be registered *after* langchain_core is loaded and *before* langgraph is.
import langchain_core  # noqa: F401

warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from health_agents.cli import main  # noqa: E402  (must follow the filter above)

if __name__ == "__main__":
    sys.exit(main())
