# OpenSculptBoy
# Apache License, Version 2.0
"""
Serve the viewer page on this computer, for the AI providers' passkeys (docs/viewer.md, "AI providers").

    python serve_viewer.py                    # anny_viewer.html next to this script
    python serve_viewer.py PAGE.html --open   # another page file, and open it in the browser

Passkeys need the page at http://localhost. This server:

- serves the page file alone, plus MediaPipe's files beside it (mediapipe/*.js, *.wasm and
  *.task, which the page checks by their pinned SHA-256), the data files of a page in parts
  (data/model-N.txt) and the statue starters' packages (starters/<statue>.zip, which the
  New screen and a first visit fetch and check by their pinned SHA-256), and answers every
  other path with 404, so no other file of the folder shares the page's origin;
- listens on the loopback addresses only, 127.0.0.1 and ::1 both, so no other program can take the
  page's address on the other one, and stops when either is taken;
- uses one fixed port (``serve.json``, which the page reads too), because the page's stored keys
  belong to its origin, port included;
- answers only requests that name localhost, 127.0.0.1 or [::1] at that port, and sends headers that
  keep the page out of frames and caches.

It needs Python 3.8 or later and nothing else.
"""

import argparse
import errno
import http.server
import json
import pathlib
import re
import socket
import sys
import threading
import webbrowser

HERE = pathlib.Path(__file__).resolve().parent


def port() -> int:
    """the page's port, from serve.json in the repository (src/ai/) or next to a built page"""
    for p in (HERE / "src" / "ai" / "serve.json", HERE / "serve.json"):
        if p.exists():
            return int(json.loads(p.read_text(encoding="utf-8"))["port"])
    raise SystemExit("serve.json is missing next to serve_viewer.py")


PORT = port()
DATA = re.compile(r"^data/model-\d+\.txt$")
MEDIAPIPE = {
    ".js": "text/javascript; charset=utf-8",
    ".wasm": "application/wasm",
    ".task": "application/octet-stream",
}
HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def files_of(page: pathlib.Path) -> dict:
    """the URL paths the server answers, and their files"""
    served = {"/" + page.name: (page, "text/html; charset=utf-8")}
    data = page.parent / "data"
    if data.is_dir():
        for f in sorted(data.iterdir()):
            rel = "data/" + f.name
            if f.is_file() and DATA.match(rel):
                served["/" + rel] = (f, "text/plain; charset=utf-8")
    # MediaPipe's files for From a picture (the page checks each by its pinned SHA-256)
    mp = page.parent / "mediapipe"
    if mp.is_dir():
        for f in sorted(mp.iterdir()):
            kind = MEDIAPIPE.get(f.suffix)
            if f.is_file() and kind and re.fullmatch(r"[\w.-]+", f.name):
                served["/mediapipe/" + f.name] = (f, kind)
    # the statue starters of the New screen (viewer/src/starters.ts): each a fitted card with its props in one
    # package, which the page checks by its pinned SHA-256
    st = page.parent / "starters"
    if st.is_dir():
        for f in sorted(st.iterdir()):
            if f.is_file() and re.fullmatch(r"[a-z0-9_]+\.zip", f.name):
                served["/starters/" + f.name] = (f, "application/zip")
    return served


def handler_for(page: pathlib.Path, port_: int):
    served = files_of(page)
    hosts = {f"localhost:{port_}", f"127.0.0.1:{port_}", f"[::1]:{port_}"}

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "OpenSculptBoy"
        sys_version = ""

        def log_message(self, *args):
            pass

        def _answer(self, body: bool):
            if self.headers.get("Host", "") not in hosts:
                return self._plain(
                    421, "This server answers only http://localhost:%d." % port_
                )
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/" + page.name)
                for k, v in HEADERS.items():
                    self.send_header(k, v)
                self.end_headers()
                return
            if path not in served:
                return self._plain(404, "Not found.")
            f, kind = served[path]
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            for k, v in HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(data)

        def _plain(self, code: int, text: str):
            data = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for k, v in HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def do_GET(self):
            self._answer(True)

        def do_HEAD(self):
            self._answer(False)

    return Handler


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    # Windows' SO_REUSEADDR lets another program bind the same port, so the server asks for
    # the port alone there (SO_EXCLUSIVEADDRUSE); elsewhere SO_REUSEADDR only reuses a closed port
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self):
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


class Server6(Server):
    address_family = socket.AF_INET6


def start(page: pathlib.Path, port_: int = PORT):
    """the servers on 127.0.0.1 and ::1 (::1 left out on a computer without IPv6); raises
    OSError when the port is taken"""
    handler = handler_for(page, port_)
    servers = [Server(("127.0.0.1", port_), handler)]
    try:
        servers.append(Server6(("::1", port_), handler))
    except OSError as e:
        if e.errno not in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, 10049, 10047):
            servers[0].server_close()
            raise
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    return servers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve_viewer.py", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "page",
        nargs="?",
        default=None,
        help="the page file (default: anny_viewer.html next to this script, or dist/anny_viewer.html)",
    )
    parser.add_argument(
        "--open", action="store_true", help="open the page in the browser"
    )
    args = parser.parse_args(argv)
    page = (
        pathlib.Path(args.page)
        if args.page
        else next(
            (
                p
                for p in (HERE / "anny_viewer.html", HERE / "dist" / "anny_viewer.html")
                if p.exists()
            ),
            HERE / "anny_viewer.html",
        )
    )
    page = page.resolve()
    if not page.is_file():
        print(f"No page at {page}.", file=sys.stderr)
        return 2
    try:
        servers = start(page)
    except OSError:
        print(
            f"Port {PORT} is taken by another program. The viewer's stored keys belong to"
            " this port, so close that program and start again.",
            file=sys.stderr,
        )
        return 1
    url = f"http://localhost:{PORT}/{page.name}"
    print(
        f"Serving {page.name} at {url} on this computer only. Press Ctrl+C to stop.",
        flush=True,
    )
    if args.open:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    for s in servers:
        s.shutdown()
        s.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
