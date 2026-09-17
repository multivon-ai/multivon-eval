"""Loopback report viewer request boundaries; no browser writes or remote fetches."""
from __future__ import annotations


class LocalViewerSecurity:
    """Mixin for the two stdlib report handlers; this is not user authentication."""

    def allow_viewer_request(self) -> bool:
        port = self.server.server_address[1]
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if port == 80:
            hosts |= {'127.0.0.1', 'localhost'}
        origins = {f'http://{host}' for host in hosts}
        if (self.headers.get('Host') not in hosts or
                self.headers.get('Origin') not in {None, *origins} or
                self.headers.get('Sec-Fetch-Site') == 'cross-site'):
            self.send_error(403, 'This viewer accepts same-origin loopback requests only')
            return False
        return True

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy',
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        super().end_headers()
