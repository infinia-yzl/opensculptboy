# OpenSculptBoy
# Apache License, Version 2.0
"""
Render the thumbnails of the New screen's starters and write viewer/src/starter_thumbs.ts.

    python viewer/starter_thumbnails.py [PAGE.html]

Each light starter (viewer/src/starters.ts) shows in its stone and its pose, from the front three-quarter, in the
character sheet's studio lights with the exposure that the sheet's meter sets, on a transparent ground, so the New
screen's card shows through. The page renders them (window.showStarter and the sheet's renderer, window.sheet.host)
in Chromium through Playwright; a GPU makes it quick. Run it after a change to the starters or to the stones, then
build the page again (cd viewer && node build.mjs --reuse-data).
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "src" / "starter_thumbs.ts"
# the thumbnail's size in pixels (twice the slot's size on a screen of density 1)
W, H = 240, 360
TURN = -32  # degrees: the front three-quarter, as the page's view shows it

RENDER = """async ([id, w, h, turn]) => {
  if (!window.showStarter(id)) throw new Error('no starter ' + id);
  const host = window.sheet.host;
  await new Promise((r) => setTimeout(r, 400));
  host.hold(true);
  try {
    host.lights('studio');
    host.turn(turn);
    const f = host.figure();
    const halfH = (f.top - f.ground) / 2 * 1.1 + 0.02;
    const view = { turn, cx: (f.lo[0] + f.hi[0]) / 2, cy: (f.top + f.ground) / 2, halfH, width: w, height: h };
    const tick = () => new Promise((r) => requestAnimationFrame(() => r()));
    await host.meter(view, 0, tick);
    const img = await host.render(view, 32, tick);
    const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
    c.getContext('2d').putImageData(img, 0, 0);
    return c.toDataURL('image/webp', 0.86);
  } finally { host.hold(false); }
}"""


def main():
    from playwright.sync_api import sync_playwright

    page = pathlib.Path(
        sys.argv[1] if len(sys.argv) > 1 else HERE / "dist" / "anny_viewer.html"
    ).resolve()
    ids = [
        line.split("'")[1]
        for line in (HERE / "src" / "starters.ts")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip().startswith("{ id: '")
    ]
    args = (
        ["--use-angle=d3d11", "--ignore-gpu-blocklist", "--enable-gpu"]
        if os.name == "nt"
        else ["--ignore-gpu-blocklist"]
    )
    thumbs = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=args)
        tab = browser.new_context(viewport=dict(width=1280, height=800)).new_page()
        tab.goto(page.as_uri() + "?base&blink=off&acc=2", timeout=300000)
        tab.wait_for_function("() => window.__ready === true", timeout=300000)
        for sid in ids:
            thumbs[sid] = tab.evaluate(RENDER, [sid, W, H, TURN])
            print(sid, len(thumbs[sid]) * 3 // 4, "bytes", flush=True)
        browser.close()
    lines = [f"  {sid}: {json.dumps(url)}," for sid, url in thumbs.items()]
    OUT.write_text(
        "// OpenSculptBoy\n// Apache License, Version 2.0\n//\n"
        "// The thumbnails of the New screen's starters: each light starter in its stone and its\n"
        "// pose, front three-quarter, in the character sheet's metered studio lights, on a\n"
        "// transparent ground (WebP data URLs). Written by viewer/starter_thumbnails.py; run it\n"
        "// again after a change to the starters or the stones.\n\n"
        "export const STARTER_THUMBS: Record<string, string> = {{\n"
        + "\n".join(lines)
        + "\n};\n",
        encoding="utf-8",
    )
    print("wrote", OUT, sum(len(u) for u in thumbs.values()) // 1024, "KB")


if __name__ == "__main__":
    main()
