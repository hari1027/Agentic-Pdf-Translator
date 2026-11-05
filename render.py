"""Rebuilds a clean translated PDF from the translated elements.

Every original page starts a new output page of the same size, so the
translated book keeps the same page-by-page structure. If a translation is
longer than the original, that page simply flows onto an extra page instead
of shrinking the text. Pictures appear at the same point in the reading
order, between the same paragraphs as in the original.
"""
import html
import io
import re

import pymupdf as fitz

MARGIN = 42
LATIN_RUN = re.compile(r"[A-Za-z0-9\u00C0-\u024F][A-Za-z0-9\u00C0-\u024F\s.,;:!?'\"()&/%+-]*")

BASE_CSS = """
body {font-family: target; font-size: 12pt; line-height: 1.45; color: #111;}
h1 {font-size: 20pt; line-height: 1.3; margin: 0 0 8pt 0;}
h2 {font-size: 16pt; line-height: 1.3; margin: 6pt 0 6pt 0;}
p {margin: 0 0 7pt 0;}
p.verse {margin: 0 0 9pt 0;}
p.caption {font-size: 9.5pt; color: #444; text-align: center; margin: 2pt 0 9pt 0;}
div.img {text-align: center; margin: 4pt 0 8pt 0;}
table {border-collapse: collapse; margin: 4pt 0 10pt 0; width: 100%;}
td {border: 0.6pt solid #777; padding: 3pt 5pt; font-size: 10.5pt; vertical-align: top;}
"""


def _rich(text: str, mixed: bool) -> str:
    out = html.escape(text.strip(), quote=False)
    if mixed:  # MuPDF ignores font fallback lists: wrap English/digits in the Latin font
        out = LATIN_RUN.sub(lambda m: f'<span style="font-family: latin">{m.group(0)}</span>', out)
    return out.replace("\n", "<br>")


def _readable(hex_color: str) -> bool:
    """Light text (designed to sit on a dark picture) would vanish on white paper."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return (0.299 * r + 0.587 * g + 0.114 * b) < 150


def _element_html(el: dict, n: int, page_no: int, archive: fitz.Archive,
                  content: fitz.Rect, mixed: bool, first_heading: bool) -> str:
    kind = el["kind"]
    if kind == "image":
        name = f"p{page_no}_img{n}.png"
        archive.add((el["png"], name))
        # same size as in the original, but never wider/taller than the page
        w = min(el["w"], content.width)
        h = el["h"] * w / el["w"]
        if h > content.height * 0.8:
            w = w * content.height * 0.8 / h
        h = el["h"] * w / el["w"]
        return (f'<div class="img"><img src="{name}" width="{w:.0f}" height="{h:.0f}" '
                f'style="width: {w:.0f}pt; height: {h:.0f}pt"/></div>')

    if kind == "table":
        rows = "".join("<tr>" + "".join(f"<td>{_rich(c, mixed)}</td>" for c in row) + "</tr>"
                       for row in el["rows"])
        return f"<table>{rows}</table>"

    text = _rich(el["text"], mixed)
    style = []
    if el.get("size") and kind not in ("heading",):
        style.append(f"font-size: {el['size']}pt")
    if el.get("color") and el["color"] not in ("#000000", "#111111") and _readable(el["color"]):
        style.append(f"color: {el['color']}")
    if el.get("bold"):
        style.append("font-weight: bold")
    if el.get("italic"):
        style.append("font-style: italic")
    if el.get("align") == "center":
        style.append("text-align: center")
    attr = f' style="{"; ".join(style)}"' if style else ""

    if kind == "heading":
        tag = "h1" if first_heading else "h2"
        size = f' style="font-size: {min(el["size"], 26)}pt"' if el.get("size") else ""
        return f"<{tag}{size}>{text}</{tag}>"
    cls = f' class="{kind}"' if kind in ("verse", "caption") else ""
    return f"<p{cls}{attr}>{text}</p>"


def build_pdf(pages: list[dict], font_css: str, font_dir: str, rtl: bool, mixed: bool) -> bytes:
    """pages = [{"size": (w, h), "elements": [...]}, ...] with translated text."""
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    css = font_css + BASE_CSS + ("body {direction: rtl; text-align: right;}" if rtl else "")
    first_heading = True

    for page_no, page in enumerate(pages):
        mediabox = fitz.Rect(0, 0, *page["size"])
        content = mediabox + (MARGIN, MARGIN, -MARGIN, -MARGIN)
        archive = fitz.Archive(font_dir)
        parts = []
        for n, el in enumerate(page["elements"]):
            parts.append(_element_html(el, n, page_no, archive, content, mixed, first_heading))
            if el["kind"] == "heading":
                first_heading = False
        if not parts:
            continue
        body = "".join(parts)
        story = fitz.Story(html=f'<body{" dir=rtl" if rtl else ""}>{body}</body>',
                           user_css=css, archive=archive)
        more, guard = 1, 0
        while more and guard < 30:  # guard: never loop forever on odd content
            device = writer.begin_page(mediabox)
            more, filled = story.place(content)
            story.draw(device)
            writer.end_page()
            guard += 1
            if more and fitz.Rect(filled).is_empty:
                break
    writer.close()

    doc = fitz.open("pdf", buf.getvalue())
    if len(doc) == 0:
        raise ValueError("Nothing could be read from this PDF.")
    doc.subset_fonts(fallback=True)  # embed only the glyphs used (small file)
    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return out
