#!/usr/bin/env python3
"""Render each src/*.html (with style.css inlined) to a 2x PNG cropped to the content height."""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HEAD = ('<!doctype html><html><head><meta charset="utf-8">'
        '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=block" rel="stylesheet">'
        '<style>{css}</style></head><body>{body}'
        '<script>document.fonts.ready.then(()=>{{document.body.setAttribute("data-h",'
        'Math.ceil(document.querySelector(".wrap").getBoundingClientRect().height))}})</script></body></html>')


def chrome(*args):
    return subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", *args],
                          capture_output=True, text=True, timeout=120)


def render(name: str, out_dir: Path):
    css = (HERE / "style.css").read_text()
    body = (HERE / "src" / f"{name}.html").read_text()
    page = HERE / "build" / f"{name}.html"
    page.parent.mkdir(exist_ok=True)
    page.write_text(HEAD.format(css=css, body=body))
    dom = chrome("--virtual-time-budget=20000", "--window-size=1600,2000", "--dump-dom", page.as_uri()).stdout
    h = int(re.search(r'data-h="(\d+)"', dom).group(1))
    png = out_dir / f"{name}.png"
    chrome("--virtual-time-budget=20000", f"--window-size=1600,{h}", "--force-device-scale-factor=2",
           f"--screenshot={png}", page.as_uri())
    print(f"{png}  {1600 * 2}x{h * 2}")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for n in sys.argv[2:]:
        render(n, out)
