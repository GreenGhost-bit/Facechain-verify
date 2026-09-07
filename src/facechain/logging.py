"""Structured logging + timed spans.

A tiny dependency-free structured logger. Human-readable lines on a TTY, one JSON
object per line when ``FACECHAIN_LOG_JSON=1`` or output is redirected.

Every run attaches file sinks under ``runs/<id>/``:

* ``telemetry.jsonl`` — machine-readable JSON lines (timings, fields)
* ``run.log`` — plain text timeline you can ``tail -f``

:func:`span` emits a ``*.start`` / ``*.end`` pair with a millisecond duration.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _human_line(record: Mapping[str, Any]) -> str:
    ctx = " ".join(
        f"{k}={v}" for k, v in record.items() if k not in {"ts", "level", "logger", "event"}
    )
    return f"{record['ts']}  {str(record['level']).upper():7}  {record['event']:<32}  {ctx}".rstrip()


@dataclass
class RunLogBundle:
    """Paired JSON + plain-text log files for one pipeline / search run."""

    run_dir: Path
    jsonl: TextIO
    text: TextIO

    @property
    def telemetry_path(self) -> Path:
        return self.run_dir / "telemetry.jsonl"

    @property
    def run_log_path(self) -> Path:
        return self.run_dir / "run.log"

    def close(self) -> None:
        for fh in (self.jsonl, self.text):
            try:
                fh.close()
            except Exception:
                pass


def open_run_logs(run_dir: Path) -> RunLogBundle:
    """Create ``telemetry.jsonl`` + ``run.log`` under ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (run_dir / "telemetry.jsonl").open("w", encoding="utf-8")
    text = (run_dir / "run.log").open("w", encoding="utf-8")
    text.write(f"# facechain run log — {run_dir}\n")
    text.write(f"# started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
    text.flush()
    return RunLogBundle(run_dir=run_dir, jsonl=jsonl, text=text)


@dataclass
class StructuredLogger:
    """Minimal structured logger with optional JSON-lines + plain-text file sinks."""

    name: str = "facechain"
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    level: int = _LEVELS["info"]
    json_mode: bool = field(default_factory=lambda: _truthy(os.environ.get("FACECHAIN_LOG_JSON")))
    context: dict[str, Any] = field(default_factory=dict)
    _sinks: list[TextIO] = field(default_factory=list)
    _text_sinks: list[TextIO] = field(default_factory=list)
    _bundles: list[RunLogBundle] = field(default_factory=list)

    def bind(self, **kwargs: Any) -> StructuredLogger:
        child = StructuredLogger(
            name=self.name,
            stream=self.stream,
            level=self.level,
            json_mode=self.json_mode,
            context={**self.context, **kwargs},
        )
        child._sinks = self._sinks
        child._text_sinks = self._text_sinks
        child._bundles = self._bundles
        return child

    def add_sink(self, sink: TextIO) -> None:
        """Attach an extra JSON-lines sink (used for runs/<id>/telemetry.jsonl)."""
        self._sinks.append(sink)

    def add_text_sink(self, sink: TextIO) -> None:
        """Attach a plain-text sink (used for runs/<id>/run.log)."""
        self._text_sinks.append(sink)

    def attach_run_logs(self, bundle: RunLogBundle) -> None:
        """Wire both file sinks from a :class:`RunLogBundle`."""
        self._bundles.append(bundle)
        self.add_sink(bundle.jsonl)
        self.add_text_sink(bundle.text)

    def detach_run_logs(self, bundle: RunLogBundle) -> None:
        self.remove_sink(bundle.jsonl)
        self.remove_text_sink(bundle.text)
        self._bundles = [b for b in self._bundles if b is not bundle]

    def remove_sink(self, sink: TextIO) -> None:
        self._sinks[:] = [s for s in self._sinks if s is not sink]

    def remove_text_sink(self, sink: TextIO) -> None:
        self._text_sinks[:] = [s for s in self._text_sinks if s is not sink]

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
        human = _human_line(record)

        # File sinks always capture every level (full forensic trail).
        for sink in self._sinks:
            try:
                sink.write(line + "\n")
                sink.flush()
            except (OSError, ValueError):  # pragma: no cover
                pass
        for sink in self._text_sinks:
            try:
                sink.write(human + "\n")
                sink.flush()
            except (OSError, ValueError):  # pragma: no cover
                pass

        if lvl < self.level:
            return
        if self.json_mode or not self.stream.isatty():
            self.stream.write(line + "\n")
        else:
            colour = {"debug": "37", "info": "36", "warning": "33", "error": "31"}[level]
            ctx = " ".join(
                f"{k}={v}" for k, v in record.items() if k not in {"ts", "level", "logger", "event"}
            )
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
