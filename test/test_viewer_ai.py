# OpenSculptBoy
# Apache License, Version 2.0
"""
The AI providers of the viewer (viewer/src/ai/, the temporary panel viewer/src/ui/ai_ui.ts,
docs/viewer.md "AI providers") and the page's server (viewer/serve_viewer.py).

In CI: the key vault's cryptography and records with a fake passkey, the lock without a timer,
the keys of one visit, the page checks, each provider's requests and errors with a fake fetch,
and a scan of the page's sources for remote hosts (viewer/test/ai_vault.mjs, TestVaultNode); and
the server, which serves the page and its data files alone on 127.0.0.1 and ::1, answers only
requests for localhost at its port, and refuses a taken port (TestServe).

Local only, in the built page served by serve_viewer.py, in Chromium through Playwright with
Chromium's virtual authenticator (TestAiPage): a passkey with PRF, a stored key, check, lock,
unlock, a refused key, forget and forget all, with the providers' endpoints mocked; every request
of the page is routed, and the test fails on any request to a host other than the page and the
provider of the key it carries; IndexedDB, web storage and the URL never hold a key; the Content
Security Policy reports no violation; an authenticator without PRF keeps a key for one visit;
the page opened as a file shows the way to the server and opens no database; and From a picture
runs MediaPipe from its pinned files beside the page, with no request to another host (on a
file page it names the server).
"""

import contextlib
import errno
import json
import pathlib
import shutil
import socket
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request

from test.markers import local_only

REPO = pathlib.Path(__file__).resolve().parents[1]
PAGE = REPO / "viewer" / "dist" / "anny_viewer.html"
CHROMIUM = pathlib.Path("/opt/pw-browsers/chromium")
SCRIPT = REPO / "viewer" / "test" / "ai_vault.mjs"
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
KEY_A = "sk-ant-api03-" + "Test1234" * 10
KEY_O = "sk-proj-" + "Wrong987" * 8


def node_available():
    if shutil.which("node") is None:
        return False
    out = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout
    return int(out.strip().lstrip("v").split(".")[0]) >= 22


def playwright_available():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return PAGE.exists()


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestVaultNode(unittest.TestCase):
    def test_vault_and_providers(self):
        r = subprocess.run(
            ["node", str(SCRIPT)], capture_output=True, text=True, cwd=REPO
        )
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["ok"])
        self.assertGreaterEqual(len(out["checks"]), 30)


def serve_module():
    """viewer/serve_viewer.py as a module"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "serve_viewer", REPO / "viewer" / "serve_viewer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free_port() -> int:
    """a port free on both loopback addresses"""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        try:
            with socket.socket(socket.AF_INET6) as s6:
                s6.bind(("::1", port))
        except OSError as e:
            if e.errno not in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, 10049, 10047):
                continue
        return port
    raise RuntimeError("no free port")


@contextlib.contextmanager
def served(page: pathlib.Path):
    """the page served by serve_viewer.py on a free port, as the person serves it"""
    port = free_port()
    servers = serve_module().start(page, port)
    try:
        yield port
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()


def fetch(url: str, host: str | None = None):
    """status, headers and body of a GET, with another Host header when given"""
    req = urllib.request.Request(url, headers={"Host": host} if host else {})
    opener = urllib.request.build_opener(NoRedirect)
    try:
        r = opener.open(req, timeout=10)
        return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class TestServe(unittest.TestCase):
    """serve_viewer.py: the page and its data files alone, on both loopback addresses"""

    def test_serves_the_page_alone(self):
        mod = serve_module()
        port_file = REPO / "viewer" / "src" / "ai" / "serve.json"
        self.assertEqual(mod.PORT, json.loads(port_file.read_text())["port"])
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "page.html").write_text("<p>page</p>", encoding="utf-8")
            (d / "other.html").write_text("<p>other</p>", encoding="utf-8")
            (d / "secret.txt").write_text("secret", encoding="utf-8")
            (d / "data").mkdir()
            (d / "data" / "model-1.txt").write_text("AAAA", encoding="utf-8")
            (d / "data" / "notes.txt").write_text("notes", encoding="utf-8")
            (d / "mediapipe").mkdir()
            (d / "mediapipe" / "loader.wasm").write_bytes(b"\0asm")
            (d / "mediapipe" / "other.html").write_text("<p>no</p>", encoding="utf-8")
            with served(d / "page.html") as port:
                base = f"http://127.0.0.1:{port}"
                code, headers, body = fetch(base + "/page.html")
                self.assertEqual((code, body), (200, b"<p>page</p>"))
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertIn(
                    "frame-ancestors 'none'", headers["Content-Security-Policy"]
                )
                self.assertEqual(fetch(base + "/data/model-1.txt")[:1], (200,))
                code, headers, body = fetch(base + "/mediapipe/loader.wasm")
                self.assertEqual(
                    (code, headers["Content-Type"], body),
                    (200, "application/wasm", b"\0asm"),
                )
                self.assertEqual(fetch(base + "/mediapipe/other.html")[0], 404)
                for path in (
                    "/other.html",
                    "/secret.txt",
                    "/data/notes.txt",
                    "/data/",
                    "/..%2fsecret.txt",
                    "/page.html/../secret.txt",
                ):
                    self.assertEqual(fetch(base + path)[0], 404, path)
                code, headers, _ = fetch(base + "/")
                self.assertEqual((code, headers["Location"]), (302, "/page.html"))
                self.assertEqual(
                    fetch(base + "/page.html", host=f"evil.example:{port}")[0], 421
                )
                self.assertEqual(
                    fetch(base + "/page.html", host=f"localhost:{port}")[0], 200
                )
                # the IPv6 loopback answers too, where the computer has it
                try:
                    with socket.create_connection(("::1", port), timeout=5):
                        v6 = True
                except OSError:
                    v6 = False
                if v6:
                    self.assertEqual(fetch(f"http://[::1]:{port}/page.html")[0], 200)
                # a second server on the same port: refused
                with self.assertRaises(OSError):
                    mod.start(d / "page.html", port)
            # a port taken by another program: refused
            with socket.socket() as other:
                other.bind(("127.0.0.1", 0))
                other.listen()
                with self.assertRaises(OSError):
                    mod.start(d / "page.html", other.getsockname()[1])


def add_authenticator(tab, prf=True):
    """Chromium's virtual authenticator: a platform passkey with user verification"""
    cdp = tab.context.new_cdp_session(tab)
    cdp.send("WebAuthn.enable")
    options = {
        "protocol": "ctap2",
        "ctap2Version": "ctap2_1",
        "transport": "internal",
        "hasResidentKey": True,
        "hasUserVerification": True,
        "isUserVerified": True,
        "automaticPresenceSimulation": True,
        "hasPrf": prf,
    }
    aid = cdp.send("WebAuthn.addVirtualAuthenticator", {"options": options})
    return cdp, aid["authenticatorId"]


# every record of the vault's database, with byte arrays as latin-1 text, so that a test can
# look for a key in them
DUMP = """async () => {
  const names = (await indexedDB.databases()).map((d) => d.name);
  if (!names.includes('opensculptboy-ai')) return null;
  const db = await new Promise((ok, bad) => {
    const r = indexedDB.open('opensculptboy-ai');
    r.onsuccess = () => ok(r.result);
    r.onerror = () => bad(r.error);
  });
  const out = {};
  for (const t of db.objectStoreNames) {
    out[t] = await new Promise((ok) => {
      const g = db.transaction(t).objectStore(t).getAll();
      g.onsuccess = () => ok(g.result);
    });
  }
  db.close();
  const text = (v) => v instanceof Uint8Array ? String.fromCharCode(...v)
    : v && typeof v === 'object' ? Object.values(v).map(text).join('|') : String(v);
  const counts = Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.length]));
  return { counts, text: text(out) };
}"""
STORAGE = """() => JSON.stringify(Object.assign({}, localStorage))
  + JSON.stringify(Object.assign({}, sessionStorage)) + location.href + document.cookie"""


def route_all(ctx, port, anthropic, openai):
    """routes every request of a browser context: the page itself goes through, the two
    providers go to their mocks, and anything else is aborted and listed"""
    blocked = []
    page = f"http://localhost:{port}/" if port else "file://"

    def router(route):
        url = route.request.url
        if url.startswith(page):
            return route.continue_()
        if url.startswith("https://api.anthropic.com/"):
            return anthropic(route)
        if url.startswith("https://api.openai.com/"):
            return openai(route)
        blocked.append(url)
        route.abort()

    ctx.route("**/*", router)
    return blocked


def watch_csp(tab):
    """the page's Content Security Policy violations, in window.__csp"""
    tab.add_init_script(
        "window.__csp = []; document.addEventListener('securitypolicyviolation',"
        " (e) => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI))"
    )


def plain_png(path: pathlib.Path, w=320, h=240):
    """a picture of one colour, with no figure in it"""
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    raw = b"".join(b"\x00" + b"\x80\x90\xa0" * w for _ in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestAiPage(unittest.TestCase):
    def open_panel(self, tab):
        tab.wait_for_function("() => window.__ready === true", timeout=300000)
        # the page reopens the section it showed last, so a click on its tab would close it
        tab.wait_for_timeout(300)
        if not tab.locator("#sec-scene").is_visible():
            tab.click("#tab-scene")
        summary = tab.locator(".ai-panel summary")
        summary.scroll_into_view_if_needed()
        summary.click()

    def status(self, tab, text):
        """waits for the panel's status line to hold text, and returns the line"""
        tab.wait_for_function(
            "(t) => (document.querySelector('.ai-status') || {}).textContent.includes(t)",
            arg=text,
            timeout=120000,
        )
        return tab.locator(".ai-status").text_content()

    def test_passkey_vault(self):
        from playwright.sync_api import sync_playwright

        seen = []  # (url, headers) of every request that left the page

        def anthropic(route):
            r = route.request
            seen.append((r.url, r.headers))
            if r.method == "OPTIONS":
                return route.fulfill(
                    status=204,
                    headers={
                        "access-control-allow-origin": "*",
                        "access-control-allow-headers": "*",
                    },
                )
            ok = (
                r.headers.get("x-api-key") == KEY_A
                and r.headers.get("anthropic-dangerous-direct-browser-access") == "true"
            )
            body = (
                {"data": [{"id": "claude-haiku-5-5"}]}
                if ok
                else {"error": {"message": "invalid x-api-key"}}
            )
            route.fulfill(
                status=200 if ok else 401,
                json=body,
                headers={"access-control-allow-origin": "*"},
            )

        def openai(route):
            r = route.request
            seen.append((r.url, r.headers))
            body = {"error": {"message": "Incorrect API key provided: " + KEY_O}}
            route.fulfill(
                status=401, json=body, headers={"access-control-allow-origin": "*"}
            )

        with sync_playwright() as p, served(PAGE) as port:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            ctx = browser.new_context(viewport=dict(width=1200, height=800))
            blocked = route_all(ctx, port, anthropic, openai)
            tab = ctx.new_page()
            tab.set_default_timeout(180000)
            watch_csp(tab)
            errors = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.on("dialog", lambda d: d.accept())
            cdp, aid = add_authenticator(tab)
            tab.goto(
                f"http://localhost:{port}/{PAGE.name}?acc=1&blink=off", timeout=300000
            )
            self.open_panel(tab)
            panel = tab.locator(".ai-panel")
            # no record before setup (the panel may have opened the empty database)
            panel.locator("button", has_text="Set up passkey").wait_for()
            first = tab.evaluate(DUMP)
            self.assertIn(first and first["counts"], (None, {"keys": 0, "passkeys": 0}))
            # set up the passkey: the vault opens
            panel.locator("button", has_text="Set up passkey").click()
            self.status(tab, "Passkey ready")
            creds = cdp.send("WebAuthn.getCredentials", {"authenticatorId": aid})[
                "credentials"
            ]
            self.assertEqual(len(creds), 1)
            self.assertEqual(creds[0]["rpId"], "localhost")
            # a key: saved, emptied from the field, checked against the mocked model list
            panel.locator(".seg button", has_text="Anthropic").click()
            panel.locator("input.ai-key").fill(KEY_A)
            panel.locator("button", has_text="Save key").click()
            self.assertIn("The key works: 1 models", self.status(tab, "The key works"))
            self.assertEqual(panel.locator("input.ai-key").input_value(), "")
            self.assertEqual(panel.locator(".ai-row").count(), 1)
            dump = tab.evaluate(DUMP)
            self.assertEqual(dump["counts"], {"keys": 1, "passkeys": 1})
            for part in (KEY_A, KEY_A[13:33]):
                self.assertNotIn(part, dump["text"])
                self.assertNotIn(part, tab.evaluate(STORAGE))
            # lock: no check, no field; unlock with the passkey, and check again
            panel.locator("button", has_text="Lock").click()
            tab.wait_for_function("() => document.querySelector('.ai-add').hidden")
            self.assertEqual(
                panel.locator(".ai-row button", has_text="Check").count(), 0
            )
            panel.locator("button", has_text="Unlock").click()
            self.status(tab, "Unlocked")
            panel.locator(".ai-row button", has_text="Check").click()
            self.status(tab, "The key works")
            # a refused key: plain words, without the key
            panel.locator(".seg button", has_text="OpenAI").click()
            self.assertEqual(
                panel.locator(".seg button[aria-pressed=true]").text_content(), "OpenAI"
            )
            panel.locator("input.ai-key").fill(KEY_O)
            panel.locator("button", has_text="Save key").click()
            line = self.status(tab, "refused the key")
            self.assertNotIn(KEY_O[8:24], line)
            self.assertIn(
                "refused", panel.locator(".ai-row[data-provider=openai]").text_content()
            )
            # each key went to its own provider only, and no request left for another host
            self.assertEqual(blocked, [])
            self.assertTrue(any("api.anthropic.com" in u for u, _ in seen))
            for url, headers in seen:
                blob = url + json.dumps(headers)
                host = url.split("/")[2]
                self.assertIn(host, ("api.anthropic.com", "api.openai.com"))
                if KEY_A in blob:
                    self.assertEqual(host, "api.anthropic.com")
                if KEY_O in blob:
                    self.assertEqual(host, "api.openai.com")
            # forget one key, then everything
            panel.locator(
                ".ai-row[data-provider=openai] button", has_text="Forget"
            ).click()
            self.status(tab, "OpenAI key forgotten")
            self.assertEqual(tab.evaluate(DUMP)["counts"], {"keys": 1, "passkeys": 1})
            panel.locator("button", has_text="Forget all").click()
            self.status(tab, "Everything forgotten")
            self.assertEqual(tab.evaluate(DUMP)["counts"], {"keys": 0, "passkeys": 0})
            self.assertEqual(
                panel.locator("button", has_text="Set up passkey").count(), 1
            )
            self.assertFalse(errors, errors)
            self.assertEqual(tab.evaluate("window.__csp"), [])

            # an authenticator without PRF: the plain line, a key for this visit only, no record
            ctx2 = browser.new_context()
            blocked2 = route_all(ctx2, port, anthropic, openai)
            tab2 = ctx2.new_page()
            tab2.set_default_timeout(180000)
            watch_csp(tab2)
            add_authenticator(tab2, prf=False)
            url = f"http://localhost:{port}/{PAGE.name}?acc=1&blink=off"
            tab2.goto(url, timeout=300000)
            self.open_panel(tab2)
            panel2 = tab2.locator(".ai-panel")
            panel2.locator("button", has_text="Set up passkey").click()
            self.status(tab2, "used for this visit only")
            panel2.locator(".seg button", has_text="Anthropic").click()
            panel2.locator("input.ai-key").fill(KEY_A)
            panel2.locator("button", has_text="Use for this visit").click()
            self.status(tab2, "The key works")
            self.assertIn("this visit only", panel2.locator(".ai-row").text_content())
            dump = tab2.evaluate(DUMP)
            self.assertEqual(dump["counts"], {"keys": 0, "passkeys": 0})
            self.assertNotIn(KEY_A[13:33], tab2.evaluate(STORAGE))
            # a new visit: the key is gone, and the panel stays in the visit mode
            tab2.goto(url, timeout=300000)
            self.open_panel(tab2)
            tab2.locator(".ai-panel button", has_text="Try a passkey again").wait_for()
            self.assertEqual(panel2.locator(".ai-row").count(), 0)
            self.assertIn("used for this visit only", panel2.text_content())
            self.assertEqual(blocked2, [])
            self.assertEqual(tab2.evaluate("window.__csp"), [])

            # the page opened as a file: the way to http://localhost, and no database
            ctx3 = browser.new_context()
            blocked3 = route_all(ctx3, None, anthropic, openai)
            tab3 = ctx3.new_page()
            tab3.set_default_timeout(180000)
            tab3.goto(PAGE.as_uri() + "?acc=1&blink=off", timeout=300000)
            self.open_panel(tab3)
            text = tab3.locator(".ai-panel").text_content()
            self.assertIn("opened as a file", text)
            port_ = serve_module().PORT
            self.assertIn(f"http://localhost:{port_}/{PAGE.name}", text)
            self.assertIn("python serve_viewer.py --open", text)
            self.assertIsNone(tab3.evaluate(DUMP))
            # From a picture on a file page: the way to the server
            with tempfile.TemporaryDirectory() as d:
                png = pathlib.Path(d) / "plain.png"
                plain_png(png)
                if not tab3.locator("#sec-pose").is_visible():
                    tab3.click("#tab-pose")
                tab3.set_input_files("#sec-pose input[type=file]", str(png))
                tab3.wait_for_function(
                    "() => [...document.querySelectorAll('#sec-pose p.inline-msg')]"
                    ".some((p) => p.textContent.includes('serve_viewer.py'))",
                    timeout=120000,
                )
            self.assertEqual(blocked3, [])
            browser.close()

    def test_picture_from_the_page_alone(self):
        """From a picture runs MediaPipe from its pinned files beside the page, with no other host"""
        from playwright.sync_api import sync_playwright

        with tempfile.TemporaryDirectory() as d, sync_playwright() as p:
            png = pathlib.Path(d) / "plain.png"
            plain_png(png)
            with served(PAGE) as port:
                kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
                browser = p.chromium.launch(args=ARGS, **kw)
                ctx = browser.new_context(viewport=dict(width=1200, height=800))
                blocked = route_all(ctx, port, lambda r: r.abort(), lambda r: r.abort())
                tab = ctx.new_page()
                tab.set_default_timeout(180000)
                watch_csp(tab)
                errors = []
                tab.on("pageerror", lambda e: errors.append(str(e)))
                tab.goto(
                    f"http://localhost:{port}/{PAGE.name}?acc=1&blink=off",
                    timeout=300000,
                )
                tab.wait_for_function("() => window.__ready === true", timeout=300000)
                tab.wait_for_timeout(300)
                if not tab.locator("#sec-pose").is_visible():
                    tab.click("#tab-pose")
                tab.set_input_files("#sec-pose input[type=file]", str(png))
                tab.wait_for_function(
                    "() => [...document.querySelectorAll('#sec-pose p.inline-msg')]"
                    ".some((p) => /No figure|could not/.test(p.textContent))",
                    timeout=300000,
                )
                msg = tab.locator("#sec-pose p.inline-msg").first.text_content()
                self.assertIn("No figure was found", msg)
                self.assertEqual(blocked, [])
                self.assertEqual(tab.evaluate("window.__csp"), [])
                self.assertFalse(errors, errors)
                browser.close()


if __name__ == "__main__":
    unittest.main()
