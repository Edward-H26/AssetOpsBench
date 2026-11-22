"""The screenshots of the episode explorer for the docs page, and the figure of one episode at the text width of a paper.

    python -m CausalDiagnosis episode_viewer explorer-shots --url http://127.0.0.1:8765/e/<run>/<episode>/ --out <folder> [--figure --step N]

It drives a browser already on the machine (playwright with its Chromium, else the machine's Chrome) against an explorer
that the explorer's serve command started, and imports nothing of the tree, so any interpreter with playwright runs it. The shots are the
run list, the episode page at 1440 and 390 px wide and the Decision and Graph tabs; --figure adds figure.png and figure.pdf of
the figure page with the smallest text size in points at the text width. Every shot stops above the footer, which says when
the data was read, and the run list stops before its column Started and before a button of the top bar that this edge would
split, so no shot shows a clock time. When the server sends the hidden side, shots of it may not enter the tree:
the command refuses an --out inside the repository then, and it never switches the hidden side on."""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEXT_WIDTH_PT = 487.8  # the text width of a two-column paper page the figure is measured against
# the box of a shot in the coordinates of the screenshot: above the footer, and before the column of the run list that the
# second argument names and before any button of the top bar that this edge would split
CUT = """([full, column]) => {
  const foot = document.getElementById('foot').getBoundingClientRect().top + (full ? window.scrollY : 0);
  const th = column ? [...document.querySelectorAll('th')].find((x) => x.textContent.trim().toLowerCase() === column) : null;
  let width = th ? th.getBoundingClientRect().left : full ? document.documentElement.scrollWidth : window.innerWidth;
  for (let moved = Boolean(th); moved;) {
    moved = false;
    for (const b of [...document.querySelectorAll('header.top button')].map((x) => x.getBoundingClientRect())) {
      if (b.width && b.left < width && b.right > width) { width = b.left; moved = true; }
    }
  }
  const height = full ? document.documentElement.scrollHeight : window.innerHeight;
  return { x: 0, y: 0, width: Math.floor(width), height: Math.floor(Math.min(height, foot)) };
}"""
MEASURE = """() => {
  const fig = document.querySelector('.figure'); const W = fig.closest('main').getBoundingClientRect().width;
  const sizes = [];
  const walker = document.createTreeWalker(fig, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const t = walker.currentNode; if (!t.textContent.trim()) continue;
    const el = t.parentElement; const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden' || !el.getClientRects().length) continue;
    let px = parseFloat(st.fontSize);
    const svg = el.closest('svg');
    if (svg) { const vb = svg.viewBox.baseVal; const r = svg.getBoundingClientRect(); if (vb && vb.width) px = px * r.width / vb.width; }
    sizes.push(px);
  }
  return { width_px: W, smallest_px: Math.min(...sizes), texts: sizes.length };
}"""


def parse(argv):
    ap = argparse.ArgumentParser(description="Screenshots of the episode explorer and the figure of one episode.")
    ap.add_argument("--url", required=True, help="the page of one episode, http://127.0.0.1:N/e/<run>/<episode>/")
    ap.add_argument("--out", required=True, help="the folder for the images")
    ap.add_argument("--figure", action="store_true", help="also write figure.png and figure.pdf")
    ap.add_argument("--step", type=int, default=1, help="the step the figure marks")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("shots.py needs playwright and its Chromium in this interpreter; nothing was written")
    url = args.url if args.url.endswith("/") else args.url + "/"
    out = Path(args.out).resolve()
    served = json.loads(urllib.request.urlopen(url + "hidden-status.json").read()).get("served")
    if served and out.is_relative_to(ROOT):
        raise SystemExit(f"{url} sends the hidden side, so its images stay outside the repository; {out} lies inside it")
    out.mkdir(parents=True, exist_ok=True)
    run, episode = [p for p in url.split("/") if p][-2:]
    written, errors = [], []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:  # the browser this playwright bundles is not installed: the Chrome of the machine
            browser = p.chromium.launch(channel="chrome")

        def page(width, height=900, scale=2):
            pg = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            # the Chrome of the machine asks for a favicon, which the explorer serves none of; every other error counts
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and not m.location.get("url", "").endswith("/favicon.ico") else None)
            return pg

        def shot(pg, name, before_column=None, **kw):
            # a sticky header or record panel is drawn where the page was scrolled to, so a full page starts at the top
            if kw.get("full_page"):
                pg.evaluate("window.scrollTo(0, 0)"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(out / name), clip=pg.evaluate(CUT, [bool(kw.get("full_page")), before_column]), **kw)
            written.append(name)

        pg = page(1440)
        pg.goto(url + "#/"); pg.wait_for_selector("table"); pg.wait_for_timeout(300)
        shot(pg, "runs_1440.png", before_column="started")
        for width in (1440, 390):
            # at a phone width the record is a sheet fixed to the bottom of the screen, which a full-page shot would
            # draw in the middle of the page, so the phone shot is one screen
            pg = page(width, 844 if width == 390 else 900)
            pg.goto(url + f"#/ep/{run}/{episode}/decision/{args.step}"); pg.wait_for_selector(".ovb-step"); pg.wait_for_timeout(500)
            shot(pg, f"episode_{width}.png")
            if width == 1440:
                shot(pg, "episode_1440_full.png", full_page=True)
                for tab in ("graph", "readings"):
                    pg.click(f"#tab-{tab}"); pg.wait_for_timeout(500)
                    shot(pg, f"{tab}_1440.png", full_page=True)
        figure = None
        if args.figure:
            pg = page(1000, 700, 1.6)
            pg.goto(url + f"#/figure/{run}/{episode}/{args.step}"); pg.wait_for_selector(".figure"); pg.wait_for_timeout(600)
            figure = pg.evaluate(MEASURE)
            figure["smallest_pt"] = round(figure["smallest_px"] * TEXT_WIDTH_PT / figure["width_px"], 2)
            pg.locator("main").screenshot(path=str(out / "figure.png"))
            pg.emulate_media(media="screen")  # page.pdf prints with the print styles otherwise
            box = pg.locator("main").bounding_box()
            pg.pdf(path=str(out / "figure.pdf"), width=f"{int(box['width']) + 2}px", height=f"{int(box['height']) + 2}px",
                   print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            written += ["figure.png", "figure.pdf"]
        browser.close()
    if errors:
        raise SystemExit(f"the pages logged errors: {errors[:3]}")
    print(json.dumps({"out": str(out), "written": written, "figure": figure}, indent=1))


if __name__ == "__main__":
    main()
