#!/usr/bin/env python3
"""Regenerate the author cards shown in README.md (assets/authors/*.svg).

Each author is drawn as two self-contained SVG images so each half can carry its own link: the top
half (handle + photo) links to GitHub, the bottom half (name, ORCID mark, role) links to ORCID. Each
half comes in a desktop and a phone size and a light and a dark theme; README.md picks the right one
with <picture>.

Photos are fetched from GitHub and embedded as base64 PNGs clipped to a circle inside the SVG, so
nothing in the README depends on an external image service and no image library is needed.
Re-run whenever a photo changes:

    python3 scripts/dev/make_author_cards.py
"""
import base64, html, io, os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "assets", "authors")
ORCID_ICON = "https://orcid.org/sites/default/files/images/orcid_16x16.png"

PEOPLE = [  # github handle, display name, phone-size name lines, role, ORCID iD
    ("terry-b-cho", "Byeongyeon (Terry) Cho", ["Byeongyeon", "(Terry) Cho"], "Lead developer",         "0000-0003-1370-2629"),
    ("adkostic",    "Aleksandar D. Kostic",   ["Aleksandar D.", "Kostic"],   "Principal investigator", "0000-0002-0837-4360"),
]
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
THEMES = {"light": {"name": "#1f2328", "muted": "#656d76"},
          "dark":  {"name": "#e6edf3", "muted": "#8d96a0"}}
# size name -> card width, photo diameter, handle px, name px, role px, orcid mark px
SIZES = {"desktop": dict(w=180, photo=64, handle=9.5, name=12,  role=9.5, orcid=11),
         "phone":   dict(w=82,  photo=44, handle=7.5, name=9,   role=7.5, orcid=9)}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "iqtree3-gpu-author-cards"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(), r.headers.get_content_type()


def text_width(t, px):   # average advances of a system UI semibold; textLength pins the drawn width to this
    w = 0.0
    for ch in t:
        if ch in "iljt.() ":              w += px * 0.45
        elif ch.isupper() or ch in "mw":  w += px * 0.75
        else:                             w += px * 0.55
    return round(w)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}">{body}</svg>')


def main():
    os.makedirs(OUT, exist_ok=True)
    orcid_uri = "data:image/png;base64," + base64.b64encode(fetch(ORCID_ICON)[0]).decode()
    for gh, name, name_lines, role, oid in PEOPLE:
        raw, ctype = fetch(f"https://github.com/{gh}.png?size=256")
        img = f"data:{ctype};base64," + base64.b64encode(raw).decode()
        for sz, S in SIZES.items():
            W, P = S["w"], S["photo"]
            x, y = (W - P) / 2, 18
            for theme, c in THEMES.items():
                # top half: handle above the circular photo (clipPath does the rounding; no raster tooling needed)
                top_h = 14 + 4 + P + 2
                top = svg(W, top_h,
                    f'<defs><clipPath id="c"><circle cx="{x + P/2}" cy="{y + P/2}" r="{P/2}"/></clipPath></defs>'
                    f'<text x="{W/2}" y="12" text-anchor="middle" font-size="{S["handle"]}" fill="{c["muted"]}">@{gh}</text>'
                    f'<image href="{img}" x="{x}" y="{y}" width="{P}" height="{P}" clip-path="url(#c)" '
                    f'preserveAspectRatio="xMidYMid slice"/>')
                # bottom half: name with the ORCID mark right after it, then the role
                if sz == "desktop":
                    nw = text_width(name, S["name"]); x0 = (W - (nw + 4 + S["orcid"])) / 2
                    body = (f'<text x="{x0:.1f}" y="14" font-size="{S["name"]}" font-weight="600" fill="{c["name"]}" '
                            f'textLength="{nw}" lengthAdjust="spacing">{html.escape(name)}</text>'
                            f'<image href="{orcid_uri}" x="{x0+nw+4:.1f}" y="{14-S["orcid"]+1}" width="{S["orcid"]}" height="{S["orcid"]}"/>'
                            f'<text x="{W/2}" y="32" text-anchor="middle" font-size="{S["role"]}" fill="{c["muted"]}">{role}</text>')
                    bot_h = 38
                else:
                    l1, l2 = name_lines; nw = text_width(l2, S["name"]); x0 = (W - (nw + 3 + S["orcid"])) / 2
                    body = (f'<text x="{W/2}" y="11" text-anchor="middle" font-size="{S["name"]}" font-weight="600" fill="{c["name"]}">{html.escape(l1)}</text>'
                            f'<text x="{x0:.1f}" y="23" font-size="{S["name"]}" font-weight="600" fill="{c["name"]}" '
                            f'textLength="{nw}" lengthAdjust="spacing">{html.escape(l2)}</text>'
                            f'<image href="{orcid_uri}" x="{x0+nw+3:.1f}" y="{23-S["orcid"]+1}" width="{S["orcid"]}" height="{S["orcid"]}"/>')
                    words = role.split()
                    lines = [role] if text_width(role, S["role"]) <= W - 4 else [words[0], " ".join(words[1:])]
                    for i, ln in enumerate(lines):
                        body += f'<text x="{W/2}" y="{36+i*10}" text-anchor="middle" font-size="{S["role"]}" fill="{c["muted"]}">{ln}</text>'
                    bot_h = 36 + (len(lines) - 1) * 10 + 4
                bot = svg(W, bot_h, body)
                io.open(os.path.join(OUT, f"{gh}-{sz}-{theme}-top.svg"), "w", encoding="utf-8").write(top)
                io.open(os.path.join(OUT, f"{gh}-{sz}-{theme}-bottom.svg"), "w", encoding="utf-8").write(bot)
        print(f"  {gh}")
    print(f"wrote {4*len(PEOPLE)*2} card halves to {OUT}")


if __name__ == "__main__":
    main()
