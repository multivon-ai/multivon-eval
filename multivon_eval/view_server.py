"""Read-only local serving for a saved report or report directory."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def cmd_view(args):
    """Open a saved JSON report as HTML, served locally.

    Generates the HTML once into a temp dir, starts a tiny stdlib
    http.server on the requested port, and opens the user's browser.
    Stays alive until Ctrl-C. The temp dir is cleaned up on every exit
    path (success, Ctrl-C, port collision).
    """
    import http.server
    import signal
    import socketserver
    import tempfile
    import threading
    import webbrowser

    # Translate SIGTERM into a KeyboardInterrupt so the with-block's
    # cleanup (TemporaryDirectory unlink, httpd shutdown) runs on
    # `docker stop`, `kill <pid>`, or pytest's proc.terminate() the
    # same way Ctrl-C does.
    def _term_handler(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term_handler)

    # Directory mode: `view --dir <d>` or a positional path that is a
    # directory routes to the directory-level server (index / open / diff).
    dir_arg = getattr(args, "dir", None)
    target = dir_arg or args.file
    target_path = Path(target) if target else None
    if dir_arg or (target_path is not None and target_path.is_dir()):
        from .dirview import serve_directory
        if not target_path.is_dir():
            print(f"Not a directory: {target_path}", file=sys.stderr)
            return 1
        return serve_directory(
            target_path,
            recursive=getattr(args, "recursive", False),
            port=args.port,
            no_browser=args.no_browser,
        )

    if not args.file:
        print("multivon-eval view: provide a report file or --dir <directory>",
              file=sys.stderr)
        return 1

    report_path = Path(args.file)
    if not report_path.exists():
        print(f"Report not found: {report_path}", file=sys.stderr)
        return 1

    with open(report_path) as f:
        data = json.load(f)

    from .result import EvalReport
    report = EvalReport.from_dict(data)
    html = report.to_html()

    # TemporaryDirectory removes the dir on context exit — including the
    # Ctrl-C path inside it via the with-block. No orphaned multivon-view-*
    # dirs left behind on bind failure, exception, or normal shutdown.
    with tempfile.TemporaryDirectory(prefix="multivon-view-") as tmp_str:
        tmp_dir = Path(tmp_str)
        (tmp_dir / "index.html").write_text(html, encoding="utf-8")

        port = args.port

        from .viewer_security import LocalViewerSecurity

        class _Handler(LocalViewerSecurity, http.server.SimpleHTTPRequestHandler):
            def __init__(self, *posargs, **kw):
                super().__init__(*posargs, directory=str(tmp_dir), **kw)

            def log_message(self, format, *fmtargs):
                # Suppress default access logs — user just wants the URL.
                pass

            def do_GET(self):
                if self.allow_viewer_request():
                    super().do_GET()

            def do_HEAD(self):
                if self.allow_viewer_request():
                    super().do_HEAD()

        class _ReusableServer(socketserver.TCPServer):
            # Quick-restart friendly: skip the kernel's TIME_WAIT timer if
            # the user Ctrl-C'd a moment ago and is now rerunning.
            allow_reuse_address = True

        try:
            httpd = _ReusableServer(("127.0.0.1", port), _Handler)
        except OSError as ex:
            # Print a clean error instead of leaking a traceback when
            # the explicit --port is taken.
            target = f"127.0.0.1:{port}" if port else "127.0.0.1:auto"
            print(f"multivon-eval view: could not bind {target} — {ex}", file=sys.stderr)
            return 1

        with httpd:
            actual_port = httpd.server_address[1]
            url = f"http://127.0.0.1:{actual_port}/"
            print(f"  multivon-eval view  →  {url}")
            print(f"  Source: {report_path}")
            print("  Press Ctrl-C to stop.\n")

            if args.no_browser:
                print("  --no-browser was set; not opening browser automatically.")
            else:
                # Delay browser open until AFTER the server is bound so the
                # first request can't race.
                threading.Timer(0.2, lambda: webbrowser.open(url)).start()

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n  Stopping server.")
    return 0

