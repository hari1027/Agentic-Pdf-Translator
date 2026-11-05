"""Turns every PDF page into an ordered list of elements.

Element = {"kind": heading|paragraph|verse|caption|table|image, ...}

Each page is read one of two ways:
- TEXT mode:   clean digital text -> read directly from the PDF (fast, free,
               no API call). Pictures are cropped with the text removed.
- VISION mode: scans, photos, handwriting, broken/garbled font encodings,
               multi-column or table layouts -> the page is rendered as an
               image and the ReaderAgent (vision LLM) transcribes it and
               marks where each picture is.
"""
import re
import statistics
import unicodedata

import pymupdf as fitz

from agents import ReaderAgent

MIN_IMAGE_PT = 28          # smaller pictures are decorations/bullets: skipped
VISION_DPI = 150           # what the vision model sees (sharp enough for handwriting)
IMAGE_DPI = 150            # resolution of pictures in the output PDF

LEGACY_FONTS = ("bamini", "kruti", "tscii", "vanavil", "shree", "devlys", "walkman",
                "chanakya", "amudham", "kalaham", "mylai", "senthamizh", "stmzh",
                "karthika", "nudi", "baraha", "aparna", "akruti", "sun-tommy", "kanchi")
INDIC = "\u0900-\u0DFF"


# ------------------------------------------------------------------ garbled check

def garbled_reason(text: str, font_names: list[str]) -> str | None:
    """Returns why the extracted text is unreliable, or None if it looks fine."""
    for name in font_names:
        if any(legacy in name.lower() for legacy in LEGACY_FONTS):
            return f"uses a legacy non-Unicode font ({name.split('+')[-1]})"

    chars = [c for c in text if not c.isspace()]
    if len(chars) < 20:
        return None
    n = len(chars)

    bad = sum(1 for c in chars if c == "\ufffd" or unicodedata.category(c) in ("Co", "Cn", "Cc"))
    if bad / n > 0.02:
        return "has unreadable characters (missing font encoding)"

    indic = len(re.findall(f"[{INDIC}]", text))
    if indic > 20:
        # Indian scripts glued to Latin letters/symbols inside one word = broken shaping
        mixed = len(re.findall(f"[{INDIC}][A-Za-z\\\\^¦§¤|~`]|[A-Za-z\\\\^¦§¤|~`][{INDIC}]", text))
        # vowel signs can never start a word
        orphan_signs = sum(1 for w in text.split() if unicodedata.category(w[0]) in ("Mn", "Mc"))
        if (mixed + orphan_signs) / max(indic, 1) > 0.02:
            return "has broken Indian-script encoding"

    words = re.findall(r"\S{3,}", text)
    ascii_letters = [c for c in chars if c.isascii() and c.isalpha()]
    if words and len(ascii_letters) > 0.7 * n:
        # legacy fonts show up as Latin gibberish: "vd;dpd;", ",d le; dh ckr gS]"
        odd = sum(1 for w in words
                  if re.search(r"[A-Za-z][;\[\]{}`~|\\^@][A-Za-z]", w) or re.match(r",[a-zA-Z]", w))
        if odd / len(words) > 0.08:
            return "looks like a legacy font (text decodes as gibberish)"
        alpha = [re.sub(r"[^A-Za-z]", "", w) for w in text.split()]
        # ignore acronyms (CFO, BMW) and short titles (Mr, Dr, St, Ltd)
        alpha = [w for w in alpha if len(w) >= 2 and not w.isupper()
                 and not re.fullmatch(r"[A-Z][a-z]{1,2}", w)]
        if len(alpha) >= 15:
            # real words in Latin-script languages almost always contain a vowel
            no_vowel = sum(1 for w in alpha if not re.search(r"[aeiouyAEIOUY]", w))
            if no_vowel / len(alpha) > 0.2:
                return "text decodes as gibberish"
    return None


# ------------------------------------------------------------------ routing

def choose_mode(page: fitz.Page, font_names: list[str]) -> tuple[str, str, bool]:
    """Returns (mode, reason, required).

    required=True : the text cannot be read any other way (scan, garbled font).
    required=False: vision just gives a better reading order (tables, columns,
                    charts); if vision fails the page is read directly instead.
    """
    text = page.get_text()
    letters = sum(c.isalnum() for c in text)
    area = abs(page.rect)
    images = [fitz.Rect(i["bbox"]) & page.rect for i in page.get_image_info()]
    image_cover = max((abs(r) / area for r in images), default=0)

    if letters < 15:
        if images or page.get_drawings():
            return "vision", "scanned / image-only page", True
        return "skip", "blank page", False

    reason = garbled_reason(text, font_names)
    if reason:
        return "vision", f"text {reason}", True

    # scanned page with an invisible OCR text layer on top of the scan
    try:
        trace = page.get_texttrace()
        invisible = sum(len(t["chars"]) for t in trace if t.get("type") == 3)
        total = sum(len(t["chars"]) for t in trace) or 1
        if invisible / total > 0.5 and image_cover > 0.5:
            return "vision", "scanned page (hidden OCR layer)", True
    except Exception:
        pass

    if _is_multi_column(page):
        return "vision", "multi-column layout", False
    if _has_real_table(page):
        return "vision", "contains a table", False
    if _has_chart(page):
        return "vision", "contains drawings/charts", False
    return "text", "digital text", False


def _has_real_table(page: fitz.Page) -> bool:
    """Boxes, borders and underlines are not tables: need a real grid of text."""
    try:
        for t in page.find_tables().tables:
            cells = [c for row in t.extract() for c in row if c and str(c).strip()]
            if t.row_count >= 2 and t.col_count >= 2 and len(cells) >= 4:
                return True
    except Exception:
        pass
    return False


def _has_chart(page: fitz.Page) -> bool:
    """A chart/diagram = many vector shapes in one large area (not just a frame)."""
    area, covered, count = abs(page.rect), fitz.Rect(), 0
    for d in page.get_drawings():
        r = d["rect"]
        if 2 < r.width and 2 < r.height and abs(r) < 0.6 * area:
            covered |= r
            count += 1
    if covered.is_empty:
        return False
    return count >= 25 and abs(covered & page.rect) / area > 0.12


def _is_multi_column(page: fitz.Page) -> bool:
    blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks") if b[6] == 0
              and len(b[4].strip()) > 40]
    narrow = [r for r in blocks if r.width < page.rect.width * 0.45]
    for i, a in enumerate(narrow):
        for b in narrow[i + 1:]:
            side_by_side = a.x1 <= b.x0 or b.x1 <= a.x0
            overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
            if side_by_side and overlap_y > 20:
                return True
    return False


# ------------------------------------------------------------------ TEXT mode

def read_text_page(page: fitz.Page, clean_page: fitz.Page) -> list[dict]:
    """clean_page = same page with all text removed, used to crop pictures."""
    data = page.get_text("dict", sort=True)
    text_blocks = [b for b in data["blocks"] if b["type"] == 0]
    sizes = [s["size"] for b in text_blocks for l in b["lines"] for s in l["spans"]
             if s["text"].strip()]
    body = statistics.median(sizes) if sizes else 11

    found = []  # (top, left, element): reading order = top-to-bottom, then left-to-right
    for b in data["blocks"]:
        rect = fitz.Rect(b["bbox"]) & page.rect
        if b["type"] == 1:
            if rect.width >= MIN_IMAGE_PT and rect.height >= MIN_IMAGE_PT:
                img = _image_element(clean_page, rect)
                if img:
                    found.append((round(rect.y0), rect.x0, img))
            continue
        el = _text_element(b, body, page.rect.width)
        if el:
            found.append((round(rect.y0), rect.x0, el))
    found.sort(key=lambda t: (t[0], t[1]))
    elements = [el for _, _, el in found]
    return _merge_split_paragraphs(elements)


def _text_element(block: dict, body: float, page_w: float) -> dict | None:
    lines, widths, spans = [], [], []
    for line in block["lines"]:
        if abs(line["dir"][1]) > 0.1 and abs(line["dir"][0]) > 0.1:
            continue  # diagonal decoration text
        t = "".join(s["text"] for s in line["spans"]).strip()
        if t:
            lines.append(t)
            widths.append(line["bbox"][2] - line["bbox"][0])
            spans += [s for s in line["spans"] if s["text"].strip()]
    if not lines or not any(ch.isalnum() for ch in "".join(lines)):
        return None
    main = max(spans, key=lambda s: len(s["text"]))
    x0, _, x1, _ = block["bbox"]
    block_w = x1 - x0
    full = sum(w >= 0.85 * block_w for w in widths[:-1])
    wrapped = len(lines) == 1 or full >= 0.6 * (len(lines) - 1)
    centred = abs((x0 + x1) / 2 - page_w / 2) < page_w * 0.04 and block_w < page_w * 0.7

    if wrapped:
        text = lines[0]
        for nxt in lines[1:]:  # re-join hyphenated words
            text = text[:-1] + nxt if text.endswith("-") and nxt[:1].islower() else f"{text} {nxt}"
    else:
        text = "\n".join(lines)

    size = main["size"]
    if size >= body * 1.25 and len(text) < 200:
        kind = "heading"
    elif size <= body * 0.85 and len(text) < 300:
        kind = "caption"
    elif not wrapped:
        kind = "verse"
    else:
        kind = "paragraph"
    return {"kind": kind, "text": text, "size": round(min(max(size, 7), 36), 1),
            "bold": bool(main["flags"] & 16), "italic": bool(main["flags"] & 2),
            "color": f"#{main['color']:06x}", "align": "center" if centred else "left"}


def _merge_split_paragraphs(elements: list[dict]) -> list[dict]:
    """PDFs often split one paragraph into several blocks; join obvious cases."""
    out = []
    for el in elements:
        prev = out[-1] if out else None
        if (prev and el["kind"] == prev["kind"] == "paragraph"
                and prev["size"] == el["size"] and prev["text"][-1:] not in ".!?:;\"'”’"
                and el["text"][:1].islower()):
            prev["text"] += " " + el["text"]
        else:
            out.append(el)
    return out


def _image_element(clean_page: fitz.Page, rect: fitz.Rect, check_blank: bool = False) -> dict | None:
    if check_blank:  # vision boxes only: skip an empty area the model misjudged
        thumb = clean_page.get_pixmap(clip=rect, dpi=24, colorspace=fitz.csGRAY)
        if thumb.samples and max(thumb.samples) - min(thumb.samples) < 14:
            return None
    pix = clean_page.get_pixmap(clip=rect, dpi=IMAGE_DPI)
    return {"kind": "image", "png": pix.tobytes("png"), "w": rect.width, "h": rect.height}


# ------------------------------------------------------------------ VISION mode

def read_vision_page(page: fitz.Page, reader: ReaderAgent) -> list[dict]:
    pix = page.get_pixmap(dpi=VISION_DPI)
    if max(pix.width, pix.height) > 2000:  # keep requests small
        pix = page.get_pixmap(dpi=int(VISION_DPI * 2000 / max(pix.width, pix.height)))
    raw = reader.run(pix.tobytes("jpeg", jpg_quality=85))

    W, H = page.rect.width, page.rect.height
    elements = []
    for e in raw:
        kind = str(e.get("type", "")).lower()
        if kind == "image":
            box = e.get("box_2d") or e.get("box") or []
            if len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = (max(0.0, min(1000.0, float(v))) for v in box)
            pad = 8
            rect = fitz.Rect((xmin - pad) / 1000 * W, (ymin - pad) / 1000 * H,
                             (xmax + pad) / 1000 * W, (ymax + pad) / 1000 * H) & page.rect
            if rect.width >= MIN_IMAGE_PT and rect.height >= MIN_IMAGE_PT:
                img = _image_element(page, rect, check_blank=True)
                if img:
                    elements.append(img)
        elif kind == "table":
            rows = [[str(c) for c in row] for row in e.get("rows", []) if isinstance(row, list)]
            if rows:
                elements.append({"kind": "table", "rows": rows})
        elif kind in ("heading", "paragraph", "verse", "caption"):
            text = str(e.get("text", "")).strip()
            if text:
                elements.append({"kind": kind, "text": text})
    return elements
