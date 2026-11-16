"""One place to configure logging for the CLI and the dashboard.

What is logged: each pipeline step (interpretation source, constraints, counts, warnings,
timing), Gemini call outcomes, chatbot retrieval and verification results. What is never
logged: API keys, full model responses, or model reasoning. User text is truncated.

Set VIBEMATCH_LOG_FILE to also append logs to a file (for example logs/vibematch.log).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Idempotent: safe to call on every Streamlit rerun."""
    global _CONFIGURED
    logger = logging.getLogger("src")
    logger.setLevel(level)
    if _CONFIGURED:
        return
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)
    log_file = log_file or os.getenv("VIBEMATCH_LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    logger.propagate = False
    _CONFIGURED = True


_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|(?i:key=)[^\s&]+")


def redact(text: str) -> str:
    """Remove anything that looks like an API key, including the configured one."""
    text = _KEY_PATTERN.sub("[redacted]", str(text))
    configured = os.getenv("GEMINI_API_KEY", "").strip()
    if len(configured) >= 8:
        text = text.replace(configured, "[redacted]")
    return text


def short(text: str, limit: int = 120) -> str:
    """Redact and truncate external or user-provided text before it reaches a log line."""
    text = " ".join(redact(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
