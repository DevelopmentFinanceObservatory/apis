"""One JSON object per log line on stdout, in the shape Cloud Logging parses.

uvicorn applies this once per process through `--log-config log-config.json`
(see the Dockerfile CMD), whose root handler is `stdout_handler`. Each line
carries `severity`, `message`, `logger`, and `time`. A line logged while a
request is in flight also carries `logging.googleapis.com/trace`, built from
the Cloud Trace ID that `dfo_platform.app._trace_id_dispatch` binds, so Cloud
Logging groups the line under that request.

The trace resource names the Google Cloud project, read from the
`GOOGLE_CLOUD_PROJECT` environment variable. Without it the lines omit the
trace field and the handler logs one WARNING when uvicorn applies the config.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

from dfo_platform.trace import get_cloud_trace_id

PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
TRACE_FIELD = "logging.googleapis.com/trace"


class JsonFormatter(logging.Formatter):
    """Format a record as one JSON object, with a traceback folded into `message`.

    Cloud Error Reporting reads a stack trace from the `message` field, so the
    traceback goes there and has no field of its own.
    """

    def __init__(self, *, project: str | None) -> None:
        super().__init__()
        self._project = project

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"
        entry = {
            "severity": record.levelname,
            "message": message,
            "logger": record.name,
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
        }
        trace_id = get_cloud_trace_id()
        if self._project and trace_id:
            entry[TRACE_FIELD] = f"projects/{self._project}/traces/{trace_id}"
        return json.dumps(entry, default=str)


def stdout_handler() -> logging.Handler:
    """Build the stdout handler `log-config.json` installs on the root logger.

    Warns once, through the handler itself, when `GOOGLE_CLOUD_PROJECT` is
    unset, since no logger has a handler yet while `dictConfig` runs.
    """
    project = os.environ.get(PROJECT_ENV) or None
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(project=project))
    if project is None:
        handler.handle(
            logging.LogRecord(
                name=__name__,
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg="%s is unset, so log lines carry no %s field",
                args=(PROJECT_ENV, TRACE_FIELD),
                exc_info=None,
            )
        )
    return handler
