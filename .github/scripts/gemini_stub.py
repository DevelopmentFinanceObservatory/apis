#!/usr/bin/env python3
"""Answer the Gemini key check the apis service makes at startup, for ci.yml's `docker` job.

`search.engine.clients.llm.warm` sends `GET /v1beta/models/<model>` before the service
serves. The job points the container's `GOOGLE_GEMINI_BASE_URL` here, so the image boots
with a fake key and no route to Google. Every model read answers 200 with the model's name;
any other request answers 404. Each request is printed, so the job can confirm the check ran.
Standard library only.

Usage: gemini_stub.py PORT
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_MODELS_PREFIX = "/v1beta/models/"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith(_MODELS_PREFIX):
            status = 200
            body = {"name": f"models/{self.path.removeprefix(_MODELS_PREFIX)}"}
        else:
            status = 404
            body = {"error": {"code": 404, "status": "NOT_FOUND"}}
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"gemini-stub {format % args}", flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), _Handler).serve_forever()
