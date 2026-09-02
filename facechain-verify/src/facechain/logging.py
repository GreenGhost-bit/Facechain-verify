"""Structured logging + timed spans.

A tiny dependency-free structured logger. Human-readable lines on a TTY, one JSON
object per line when ``FACECHAIN_LOG_JSON=1`` or output is redirected. Every
:func:`span` emits a ``*.start`` / ``*.end`` pair with a millisecond duration so
the run timeline is machine-readable in ``runs/<id>/telemetry.jsonl``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TextIO

_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class StructuredLogger:
    """Minimal structured logger with optional JSON-lines sink to a file."""

    name: str = "facechain"
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    level: int = _LEVELS["info"]
    json_mode: bool = field(default_factory=lambda: _truthy(os.environ.get("FACECHAIN_LOG_JSON")))
    context: dict[str, Any] = field(default_factory=dict)
    _sinks: list[TextIO] = field(default_factory=list)

    def bind(self, **kwargs: Any) -> StructuredLogger:
        child = StructuredLogger(
            name=self.name,
            stream=self.stream,
            level=self.level,
            json_mode=self.json_mode,
            context={**self.context, **kwargs},
        )
        child._sinks = self._sinks
        return child

    def add_sink(self, sink: TextIO) -> None:
        """Attach an extra JSON-lines sink (used for runs/<id>/telemetry.jsonl)."""
        self._sinks.append(sink)

    def remove_sink(self, sink: TextIO) -> None:
        with_suppress = [s for s in self._sinks if s is not sink]
        self._sinks[:] = with_suppress

    # -- emit ---------------------------------------------------------------
    def _emit(self, level: str, event: str, fields: Mapping[str, Any]) -> None:
        lvl = _LEVELS[level]
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "level": level,
            "logger": self.name,
            "event": event,
            **self.context,
            **fields,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=repr)
        for sink in self._sinks:
            try:
                sink.write(line + "\n")
                sink.flush()
            except (OSError, ValueError):  # pragma: no cover - sink closed
                pass
        if lvl < self.level:
            return
        if self.json_mode or not self.stream.isatty():
            self.stream.write(line + "\n")
        else:
            ctx = " ".join(
                f"{k}={v}" for k, v in record.items() if k not in {"ts", "level", "logger", "event"}
            )
            colour = {"debug": "37", "info": "36", "warning": "33", "error": "31"}[level]
            self.stream.write(f"\x1b[{colour}m{level:>7}\x1b[0m {event:<28} {ctx}\n")
        self.stream.flush()

    def debug(self, event: str, **fields: Any) -> None:
        self._emit("debug", event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("info", event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit("warning", event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit("error", event, fields)

    @contextmanager
    def span(self, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Time a block of work; emits ``<event>.start`` and ``<event>.end``."""
        extra: dict[str, Any] = {}
        start = time.perf_counter()
        self._emit("info", f"{event}.start", fields)
        try:
            yield extra
        except BaseException as exc:
            dur = round((time.perf_counter() - start) * 1000, 2)
            self._emit(
                "error",
                f"{event}.end",
                {**fields, **extra, "duration_ms": dur, "ok": False, "error": repr(exc)},
            )
            raise
        else:
            dur = round((time.perf_counter() - start) * 1000, 2)
            self._emit("info", f"{event}.end", {**fields, **extra, "duration_ms": dur, "ok": True})


def get_logger(name: str = "facechain") -> StructuredLogger:
    lvl = _LEVELS.get(os.environ.get("FACECHAIN_LOG_LEVEL", "info").lower(), _LEVELS["info"])
    return StructuredLogger(name=name, level=lvl)


LOG = get_logger()
