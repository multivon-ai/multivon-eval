"""HTTP harness for directory-mode ``multivon-eval view``.

Pure plumbing — all page rendering lives in :mod:`multivon_eval.dirview`.
This module only wires those render functions to URL routes and starts a
local server whose lifecycle mirrors the single-file ``cmd_view`` path
exactly: a reusable TCP server, SIGTERM→KeyboardInterrupt translation,
suppressed access logs, a delayed browser open, and clean port-bind error
handling.

Lazy + read-only by design: only file PATHS are touched at launch (via
``discover``); each request re-reads and re-parses just the files it
needs, so a huge directory never hangs startup and nothing in the user's
tree is ever written.
"""
from __future__ import annotations

import html as _html
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .dirview import (
    _page,
    discover,
    load_report,
    render_diff,
    render_index,
    render_open,
)


def serve_directory(base_dir: Path, *, recursive: bool, port: int, no_browser: bool) -> int:
    """Start the directory-mode view server. Mirrors cmd_view's harness."""
    import http.server
    import signal
    import socketserver
    import sys
    import threading
    import webbrowser

    def _term_handler(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term_handler)

    base_dir = base_dir.resolve()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress access logs
            pass

        def _send(self, status: int, html_body: str) -> None:
            payload = html_body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _entries(self):
            # Re-discover per request — stateless and read-only.
            return discover(base_dir, recursive)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            try:
                if path == "/":
                    reports, skipped = self._entries()
                    sort = (qs.get("sort") or ["when"])[0]
                    direction = (qs.get("dir") or ["desc"])[0]
                    self._send(200, render_index(
                        reports, skipped, sort=sort, direction=direction,
                        base_dir=base_dir,
                    ))
                    return
                if path.startswith("/r/"):
                    self._open(path)
                    return
                if path == "/diff":
                    self._diff(qs)
                    return
                self._send(404, _page("not found", "<h1>404</h1>"
                                      '<p><a href="/">← all reports</a></p>'))
            except Exception as ex:  # never leak a traceback to the browser
                self._send(500, _page("error", f"<h1>error</h1>"
                                      f"<p class='dim'>{_html.escape(str(ex))}</p>"
                                      "<p><a href='/'>← all reports</a></p>"))

        def _open(self, path: str) -> None:
            reports, _ = self._entries()
            by_idx = {e.idx: e for e in reports}
            try:
                idx = int(path[len("/r/"):])
            except ValueError:
                self._send(404, _page("not found", "<h1>404</h1>"))
                return
            entry = by_idx.get(idx)
            if entry is None:
                self._send(404, _page("not found", "<h1>404</h1>"
                                      '<p><a href="/">← all reports</a></p>'))
                return
            report = load_report(entry.path)
            self._send(200, render_open(report, entry))

        def _diff(self, qs: dict) -> None:
            reports, _ = self._entries()
            by_idx = {e.idx: e for e in reports}
            try:
                a = int((qs.get("a") or [""])[0])
                b = int((qs.get("b") or [""])[0])
            except ValueError:
                self._send(400, _page("bad request", "<h1>400</h1>"
                                      "<p>diff needs ?a=&b= report indices.</p>"
                                      '<p><a href="/">← all reports</a></p>'))
                return
            ea, eb = by_idx.get(a), by_idx.get(b)
            if ea is None or eb is None:
                self._send(404, _page("not found", "<h1>404</h1>"
                                      '<p><a href="/">← all reports</a></p>'))
                return
            ra, rb = load_report(ea.path), load_report(eb.path)
            self._send(200, render_diff(ra, rb, name_a=ea.stem, name_b=eb.stem))

    class _ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        httpd = _ReusableServer(("127.0.0.1", port), _Handler)
    except OSError as ex:
        target = f"127.0.0.1:{port}" if port else "127.0.0.1:auto"
        print(f"multivon-eval view: could not bind {target} — {ex}", file=sys.stderr)
        return 1

    with httpd:
        actual_port = httpd.server_address[1]
        url = f"http://127.0.0.1:{actual_port}/"
        print(f"  multivon-eval view  →  {url}")
        print(f"  Source: {base_dir}{' (recursive)' if recursive else ''}")
        print(f"  Press Ctrl-C to stop.\n")

        if no_browser:
            print("  --no-browser was set; not opening browser automatically.")
        else:
            threading.Timer(0.2, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopping server.")
    return 0


__all__ = ["serve_directory"]
