from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(message)s", force=True)
    # httpx's INFO log is human-formatted and duplicates our metadata-only JSON event.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": "INFO",
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))
