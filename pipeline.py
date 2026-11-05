"""Orchestrates the whole job: read -> analyze -> translate -> review -> rebuild."""
import time
from typing import Callable

import pymupdf as fitz

from agents import LLM, log, AnalyzerAgent, ReaderAgent, ReviewerAgent, TranslatorAgent
from extract import choose_mode, read_text_page, read_vision_page
from fonts import ensure_fonts
from languages import LANGUAGES, LATIN
from render import build_pdf

MAX_CHARS_PER_CALL = 5000  # one request translates this much text (across pages)


def open_pdf(pdf_bytes: bytes, password: str = "") -> fitz.Document:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise ValueError("This file could not be opened as a PDF (it may be damaged).")
    if doc.needs_pass and not doc.authenticate(password or ""):
        raise ValueError("This PDF is password protected. Enter the correct password.")
    return doc


def needs_password(pdf_bytes: bytes) -> bool:
    try:
        return fitz.open(stream=pdf_bytes, filetype="pdf").needs_pass
    except Exception:
        return False


def _units(pages: list[dict]) -> list[dict]:
    """Flattens all translatable text into units with stable ids."""
    units = []
    for p, page in enumerate(pages):
        for e, el in enumerate(page["elements"]):
            if el["kind"] == "table":
                for r, row in enumerate(el["rows"]):
                    for c, cell in enumerate(row):
                        if any(ch.isalpha() for ch in cell):
                            units.append({"id": f"p{p}e{e}r{r}c{c}", "kind": "cell", "text": cell})
            elif el["kind"] != "image" and any(ch.isalpha() for ch in el["text"]):
                units.append({"id": f"p{p}e{e}", "kind": el["kind"], "text": el["text"]})
    return units


def _batches(units: list[dict]):
    batch, size = [], 0
    for u in units:
        if batch and size + len(u["text"]) > MAX_CHARS_PER_CALL:
            yield batch
            batch, size = [], 0
        batch.append(u)
        size += len(u["text"])
    if batch:
        yield batch


def _apply(pages: list[dict], translations: dict) -> None:
    for p, page in enumerate(pages):
        for e, el in enumerate(page["elements"]):
            if el["kind"] == "table":
                el["rows"] = [[translations.get(f"p{p}e{e}r{r}c{c}", cell) for c, cell in enumerate(row)]
                              for r, row in enumerate(el["rows"])]
            elif el["kind"] != "image":
                el["text"] = translations.get(f"p{p}e{e}", el["text"])


def translate_pdf(pdf_bytes: bytes, target: str, llm: LLM, review: bool = True,
                  password: str = "",
                  progress: Callable[[float, str], None] = lambda p, m: None
                  ) -> tuple[bytes, dict, list[str]]:
    """Returns (translated pdf, analyzer brief, notes for the user)."""
    started, last = time.time(), {"p": 0.0}
    report = progress

    def progress(p: float, msg: str) -> None:  # remember where we are for wait messages
        last["p"] = p
        report(p, msg)

    llm.on_wait = lambda msg: report(last["p"], msg)
    doc = open_pdf(pdf_bytes, password)
    clean = open_pdf(pdf_bytes, password)  # copy used to crop pictures without text
    total = len(doc)
    if total == 0:
        raise ValueError("This PDF has no pages.")

    progress(0.01, f"Preparing {target} font (first time only)...")
    font_dir, font_css = ensure_fonts(target)

    # ---- 1. read every page (text extraction or AI vision)
    reader = ReaderAgent(llm)
    pages, notes = [], []
    vision_ok = True  # after one failure, don't keep retrying vision on every page
    for n, page in enumerate(doc):
        progress(0.02 + 0.33 * n / total, f"Reading page {n + 1}/{total}...")
        fonts = [f[3] for f in doc.get_page_fonts(n)]
        mode, reason, required = choose_mode(page, fonts)
        size = (page.rect.width, page.rect.height)
        log.info(f"page {n + 1}: {mode} mode ({reason})")
        t0 = time.time()
        if mode == "skip":
            continue
        elements = None
        if mode == "vision" and not vision_ok and not required:
            notes.append(f"Page {n + 1}: {reason} - read directly (AI vision unavailable)")
        elif mode == "vision":
            progress(0.02 + 0.33 * n / total,
                     f"Reader agent: reading page {n + 1}/{total} with AI vision ({reason})")
            try:
                elements = read_vision_page(page, reader)
                notes.append(f"Page {n + 1}: {reason} - read with AI vision")
            except Exception as e:
                if required:  # scans/garbled text cannot be read any other way
                    raise RuntimeError(f"page {n + 1} needs AI vision ({reason}), "
                                       f"but it failed: {e}") from e
                log.info(f"page {n + 1}: vision failed ({e}), reading text directly")
                vision_ok = False
                notes.append(f"Page {n + 1}: {reason} - AI vision was unavailable, "
                             "so the text was read directly (layout may be simpler)")
        if elements is None:
            cpage = clean[n]
            cpage.add_redact_annot(cpage.rect, fill=False)
            cpage.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                                   graphics=fitz.PDF_REDACT_LINE_ART_NONE)
            elements = read_text_page(page, cpage)
        pages.append({"size": size, "elements": elements})
        log.info(f"page {n + 1}: read in {time.time() - t0:.1f}s")

    units = _units(pages)
    if not units and not any(p["elements"] for p in pages):
        raise ValueError("No text or pictures could be found in this PDF.")

    # ---- 2. understand the document
    if units:
        progress(0.37, "Analyzer agent: understanding the document...")
        sample = "\n".join(u["text"] for u in units)
        brief = AnalyzerAgent(llm).run(sample, target)
    else:
        brief = {"genre": "pictures only", "tone": "", "audience": "", "summary": ""}
        notes.append("No text was found, so the pictures were copied unchanged.")

    # ---- 3. translate (+ review) in batches across the document
    translator, reviewer = TranslatorAgent(llm), ReviewerAgent(llm)
    batches = list(_batches(units))
    translations, context = {}, ""
    for i, batch in enumerate(batches):
        progress(0.4 + 0.5 * i / max(len(batches), 1),
                 f"Translator agent: part {i + 1}/{len(batches)}")
        result = translator.run(batch, target, brief, context)
        missing = [u for u in batch if not result.get(u["id"], "").strip()]
        if missing:  # ask once more for anything the model skipped
            result.update(translator.run(missing, target, brief, context))
        if review:
            progress(0.4 + 0.5 * (i + 0.5) / max(len(batches), 1),
                     f"Reviewer agent: polishing part {i + 1}/{len(batches)}")
            result = reviewer.run(batch, result, target, brief)
        translations.update(result)
        context = "\n".join(result.get(u["id"], "") for u in batch)[-1500:]

    # ---- 4. rebuild the PDF
    progress(0.93, "Building the translated PDF...")
    _apply(pages, translations)
    rtl = LANGUAGES[target].get("rtl", False)
    mixed = LANGUAGES[target]["font"] is not LATIN
    out = build_pdf(pages, font_css, str(font_dir), rtl, mixed)
    doc.close()
    clean.close()
    log.info(f"TOTAL: {total} page(s) translated to {target} in {time.time() - started:.1f}s")
    progress(1.0, "Done")
    return out, brief, notes
