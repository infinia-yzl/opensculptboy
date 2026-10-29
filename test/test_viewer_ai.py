# OpenSculptBoy
# Apache License, Version 2.0
"""
The AI providers of the viewer (viewer/src/ai/, the temporary panel viewer/src/ui/ai_ui.ts,
docs/viewer.md "AI providers"). In node: the key vault's cryptography and records with a fake
passkey, the idle lock, the page checks and each provider's requests and errors with a fake
fetch (viewer/test/ai_vault.mjs). In the built page, in Chromium through Playwright with
Chromium's virtual authenticator (local only): the page served at http://localhost registers a
passkey with PRF, stores a key, checks it, locks, unlocks, refuses a bad key, forgets a key and
forgets everything, with the providers' endpoints mocked; IndexedDB, localStorage and the URL
never hold a key; an authenticator without PRF stores nothing; and the page opened as a file
shows the way to http://localhost and opens no database.
"""

import contextlib
import functools
import http.server
import json
import pathlib
import shutil
import socketserver
import subprocess
import threading
import unittest

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


@contextlib.contextmanager
def served(folder: pathlib.Path):
    """the folder on a free port of the loopback address, quietly"""

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(folder))
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as srv:
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            yield srv.server_address[1]
        finally:
            srv.shutdown()


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


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestAiPage(unittest.TestCase):
    def open_panel(self, tab):
        tab.wait_for_function("() => window.__ready === true", timeout=300000)
        tab.click("#tab-scene")
        summary = tab.locator(".ai-panel summary")
        summary.scroll_into_view_if_needed()
        summary.click()

    def status(self, tab, text):
        """waits for the panel's status line to hold text, and returns the line"""
        tab.wait_for_function(
            "(t) => (document.querySelector('.ai-status') || {}).textContent.includes(t)",
            arg=text,
            timeout=30000,
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

        with sync_playwright() as p, served(PAGE.parent) as port:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            ctx = browser.new_context(viewport=dict(width=1200, height=800))
            ctx.route("https://api.anthropic.com/**", anthropic)
            ctx.route("https://api.openai.com/**", openai)
            tab = ctx.new_page()
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
            # the keys went to their own provider only, and nothing else left the page
            for url, headers in seen:
                blob = url + json.dumps(headers)
                if "api.anthropic.com" in url:
                    self.assertNotIn(KEY_O, blob)
                else:
                    self.assertIn("api.openai.com", url)
                    self.assertNotIn(KEY_A, blob)
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

            # an authenticator without PRF: a clear message, and no record
            ctx2 = browser.new_context()
            tab2 = ctx2.new_page()
            add_authenticator(tab2, prf=False)
            tab2.goto(
                f"http://localhost:{port}/{PAGE.name}?acc=1&blink=off", timeout=300000
            )
            self.open_panel(tab2)
            tab2.locator(".ai-panel button", has_text="Set up passkey").click()
            self.assertIn("no PRF support", self.status(tab2, "cannot lock the vault"))
            self.assertEqual(tab2.evaluate(DUMP)["counts"], {"keys": 0, "passkeys": 0})

            # the page opened as a file: the way to http://localhost, and no database
            tab3 = browser.new_context().new_page()
            tab3.goto(PAGE.as_uri() + "?acc=1&blink=off", timeout=300000)
            self.open_panel(tab3)
            text = tab3.locator(".ai-panel").text_content()
            self.assertIn("opened as a file", text)
            self.assertIn(f"http://localhost:8765/{PAGE.name}", text)
            self.assertIn("python -m http.server 8765 --bind 127.0.0.1", text)
            self.assertIsNone(tab3.evaluate(DUMP))
            browser.close()


if __name__ == "__main__":
    unittest.main()
