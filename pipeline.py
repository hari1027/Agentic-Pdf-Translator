"""Reads the PDF, runs the agents, and writes translated text back in place.

How images stay in the same place: we never rebuild the PDF. We open the
original, erase only the text (redaction with images and drawings kept), then
write the translated text into the exact same boxes. Pictures, backgrounds,
borders and page size are untouched.
"""
import html
from typing import Callable

import pymupdf as fitz

from agents import LLM, AnalyzerAgent, ReviewerAgent, TranslatorAgent
from fonts import ensure_fonts
from languages import LANGUAGES

MAX_CHARS_PER_CALL = 5000  # keeps each request well inside free-tier limits


# ------------------------------------------------------------------ extract

def extract_blocks(page: fitz.Page, page_no: int) -> list[dict]:
    blocks = []
    for i, b in enumerate(page.get_text("dict")["blocks"]):
        if b["type"] != 0:
            continue  # type 1 = image, leave alone
        lines, spans = [], []
        for line in b["lines"]:
            if abs(line["dir"][1]) > 0.1:
                continue  # skip rotated/vertical text
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                lines.append(text)
                spans.extend(s for s in line["spans"] if s["text"].strip())
        if not lines:
            continue
        main = max(spans, key=lambda s: len(s["text"]))  # dominant style
        blocks.append({
            "id": f"p{page_no}b{i}",
            "bbox": fitz.Rect(b["bbox"]),
            "text": "\n".join(lines),
            "size": round(main["size"], 1),
            "color": f"#{main['color']:06x}",
            "bold": bool(main["flags"] & 16),
            "italic": bool(main["flags"] & 2),
            "align": _guess_align(b, page.rect.width),
        })
    return blocks


def _guess_align(block: dict, page_width: float) -> str:
    lines = [l for l in block["lines"] if "".join(s["text"] for s in l["spans"]).strip()]
    lefts = [l["bbox"][0] for l in lines]
    centres = [(l["bbox"][0] + l["bbox"][2]) / 2 for l in lines]
    if len(lines) > 1:
        if max(lefts) - min(lefts) < 3:
            return "left"
        return "center" if max(centres) - min(centres) < 4 else "left"
    x0, _, x1, _ = block["bbox"]
    near_middle = abs((x0 + x1) / 2 - page_width / 2) < page_width * 0.03
    return "center" if near_middle and (x1 - x0) < page_width * 0.6 else "left"


def _batches(blocks: list[dict]):
    batch, size = [], 0
    for b in blocks:
        if batch and size + len(b["text"]) > MAX_CHARS_PER_CALL:
            yield batch
            batch, size = [], 0
        batch.append(b)
        size += len(b["text"])
    if batch:
        yield batch


def _worth_translating(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


# ------------------------------------------------------------------ write back

def write_page(page: fitz.Page, blocks: list[dict], translations: dict,
               css: str, archive: fitz.Archive, rtl: bool) -> None:
    todo = [b for b in blocks if translations.get(b["id"])]
    if not todo:
        return
    for b in todo:
        # fill=False: erase the text but keep whatever is behind it (pictures!)
        page.add_redact_annot(b["bbox"], fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_NONE)

    for b in todo:
        text = html.escape(translations[b["id"]].strip()).replace("\n", "<br>")
        style = (
            f"font-family: target, latin; font-size: {b['size']}pt; color: {b['color']};"
            f"font-weight: {'bold' if b['bold'] else 'normal'};"
            f"font-style: {'italic' if b['italic'] else 'normal'};"
            f"text-align: {'right' if rtl and b['align'] == 'left' else b['align']};"
            "line-height: 1.25;"
        )
        direction = ' dir="rtl"' if rtl else ""
        # translations are often longer: give the box a little extra room,
        # and let insert_htmlbox shrink the font if it still doesn't fit.
        r = b["bbox"]
        rect = fitz.Rect(r.x0, r.y0, r.x1, min(r.y1 + r.height * 0.15, page.rect.y1))
        if b["align"] == "center":  # centred titles/poems can widen symmetrically
            grow = min(r.width * 0.25, r.x0 - 36, page.rect.x1 - 36 - r.x1)
            if grow > 0:
                rect.x0, rect.x1 = r.x0 - grow, r.x1 + grow
        page.insert_htmlbox(rect, f'<div{direction} style="{style}">{text}</div>',
                            css=css, archive=archive, scale_low=0)


# ------------------------------------------------------------------ orchestrate

def translate_pdf(pdf_bytes: bytes, target: str, llm: LLM, review: bool = True,
                  progress: Callable[[float, str], None] = lambda p, m: None) -> tuple[bytes, dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rtl = LANGUAGES[target].get("rtl", False)

    progress(0.02, f"Downloading {target} font (first time only)...")
    font_dir, css = ensure_fonts(target)
    archive = fitz.Archive(str(font_dir))

    pages = [extract_blocks(page, n) for n, page in enumerate(doc)]
    all_text = "\n".join(b["text"] for blocks in pages for b in blocks)
    if not all_text.strip():
        raise ValueError(
            "No selectable text found. This looks like a scanned PDF (text is part "
            "of the image). Run it through OCR first, e.g. `ocrmypdf in.pdf out.pdf`."
        )

    progress(0.05, "Analyzer agent: understanding the document...")
    brief = AnalyzerAgent(llm).run(all_text, target)
    translator, reviewer = TranslatorAgent(llm), ReviewerAgent(llm)

    context = ""
    for n, (page, blocks) in enumerate(zip(doc, pages)):
        blocks = [b for b in blocks if _worth_translating(b["text"])]
        translations: dict = {}
        for batch in _batches(blocks):
            progress(0.08 + 0.9 * n / len(doc), f"Translator agent: page {n + 1}/{len(doc)}")
            result = translator.run(batch, target, brief, context)
            missing = [b for b in batch if not result.get(b["id"], "").strip()]
            if missing:  # ask once more for anything the model skipped
                result.update(translator.run(missing, target, brief, context))
            if review:
                progress(0.08 + 0.9 * (n + 0.5) / len(doc),
                         f"Reviewer agent: polishing page {n + 1}/{len(doc)}")
                result = reviewer.run(batch, result, target, brief)
            translations.update(result)
        write_page(page, blocks, translations, css, archive, rtl)
        context = "\n".join(translations.get(b["id"], "") for b in blocks)[-1500:]

    progress(0.99, "Packing the PDF...")
    doc.subset_fonts(fallback=True)  # embed only the glyphs used (small file)
    out = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    progress(1.0, "Done")
    return out, brief


def render_preview(pdf_bytes: bytes, page_no: int = 0, dpi: int = 70) -> bytes:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_no = min(page_no, len(doc) - 1)
        return doc[page_no].get_pixmap(dpi=dpi).tobytes("png")
