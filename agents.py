"""The agentic part: four cooperating LLM agents.

1. ReaderAgent     - (vision) reads pages that have no usable text: scans,
                     photos, handwriting, broken font encodings, complex
                     layouts. Returns the page in reading order, including
                     where every picture sits.
2. AnalyzerAgent   - reads a sample of the whole document and works out what
                     it is (story / poem / slogans ...), tone, audience and a
                     glossary for names and recurring terms.
3. TranslatorAgent - translates in batches using that brief plus the previous
                     batch as context, so it reads naturally (not word-for-word).
4. ReviewerAgent   - (optional) native-editor pass that fixes literal or
                     awkward lines.

Both free providers expose an OpenAI-compatible endpoint, so one `openai`
client works for either of them.
"""
import base64
import json
import logging
import re
import time

from openai import (OpenAI, RateLimitError, APIStatusError, NotFoundError,
                    BadRequestError, APITimeoutError, APIConnectionError)

log = logging.getLogger("translator")
if not log.handlers:  # timings show up in the terminal running streamlit
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

MAX_WAIT = 75  # never sit on one rate limit longer than this; switch provider instead


class QuotaExhausted(RuntimeError):
    """Daily free quota is used up: retrying is pointless, use the backup provider."""

PROVIDERS = {
    "Google Gemini (recommended)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "secret": "GEMINI_API_KEY",
        # tried in order if the configured model is gone; newer names first
        "preferred": ["gemini-flash-latest", "gemini-3-flash", "gemini-2.5-flash",
                      "gemini-2.0-flash"],
        "avoid": ["lite", "image", "tts", "live", "embed", "audio", "thinking", "vision"],
        # each Gemini model has its OWN free quota: when one runs out, try the next
        "quota_backups": ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3-flash",
                          "gemini-2.5-flash-lite", "gemini-flash-lite-latest",
                          "gemini-2.0-flash", "gemini-2.0-flash-lite"],
        "vision_same_model": True,  # Gemini Flash reads images itself
        "fast_mode": {"reasoning_effort": "low"},  # less "thinking" = much faster
        "key_page": "https://aistudio.google.com/apikey",
    },
    "Groq (Llama)": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "secret": "GROQ_API_KEY",
        "preferred": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b",
                      "moonshotai/kimi-k2-instruct",
                      "meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b"],
        "avoid": ["whisper", "guard", "tts", "playai", "compound", "prompt", "8b", "instant"],
        "quota_backups": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b",
                          "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b", "openai/gpt-oss-20b"],
        "vision_same_model": False,
        "fast_mode": {},
        "vision_preferred": ["meta-llama/llama-4-maverick-17b-128e-instruct",
                             "meta-llama/llama-4-scout-17b-16e-instruct"],
        "key_page": "https://console.groq.com/keys",
    },
}


class LLM:
    """Thin wrapper: JSON output, rate-limit retry, automatic model discovery."""

    def __init__(self, provider: str, api_key: str, model: str):
        self.name = provider
        self.cfg = PROVIDERS[provider]
        # max_retries=0: we handle retries ourselves (the SDK's hidden retries add up)
        self.client = OpenAI(api_key=api_key, base_url=self.cfg["base_url"],
                             max_retries=0, timeout=120)
        self.extra = dict(self.cfg["fast_mode"])
        self.on_wait = lambda msg: None  # the app shows waits on the loading screen
        self.tried = {model}
        self.model = model
        self.vision_model = model if self.cfg["vision_same_model"] else None
        self._resolved = False

    def _model_ids(self) -> list[str]:
        return [m.id.removeprefix("models/") for m in self.client.models.list()]

    def _pick_model(self) -> str:
        """Ask the provider which models exist and choose the best chat model."""
        ids = self._model_ids()
        for name in self.cfg["preferred"]:
            if name in ids:
                return name
        usable = sorted((i for i in ids if not any(a in i.lower() for a in self.cfg["avoid"])),
                        reverse=True)
        flash = [i for i in usable if "flash" in i]  # Gemini: prefer the fast models
        if flash or usable:
            return (flash or usable)[0]
        raise RuntimeError(f"No usable chat model found. Available: {ids[:15]}")

    def _pick_vision_model(self) -> str:
        ids = self._model_ids()
        hints = ("vision", "llama-4", "scout", "maverick", "-vl", "vl-", "llava", "gemma-3",
                 "pixtral")
        named = [n for n in self.cfg.get("vision_preferred", []) if n in ids]
        named += [i for i in ids if any(h in i.lower() for h in hints) and i not in named]
        # names change often: also test the other chat models with a tiny image
        others = [i for i in ids if i not in named
                  and not any(a in i.lower() for a in self.cfg["avoid"])]
        for model in (named + others)[:6]:
            if self._accepts_images(model):
                log.info(f"{self.name}: using {model} for reading pages")
                return model
        raise RuntimeError("no vision model is available for reading scanned pages")

    def _accepts_images(self, model: str) -> bool:
        import pymupdf as fitz
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
        pix.set_rect(pix.irect, (255, 255, 255))
        url = "data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpeg")).decode()
        try:
            self.client.chat.completions.create(
                model=model, max_tokens=5,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Reply OK."},
                    {"type": "image_url", "image_url": {"url": url}}]}])
            return True
        except Exception:
            return False

    def _switch_model(self, why: str) -> bool:
        """Moves to another model of the same provider that has its own quota."""
        try:
            ids = self._model_ids()
        except Exception:
            return False
        for name in self.cfg["quota_backups"]:
            if name in ids and name not in self.tried:
                self.tried.add(name)
                log.info(f"{self.name}: {why} on {self.model}, switching to {name}")
                self.model = name
                if self.cfg["vision_same_model"]:
                    self.vision_model = name
                return True
        return False

    def json_chat(self, system: str, user: str, image_jpeg: bytes | None = None,
                  retries: int = 12) -> dict:
        if image_jpeg is not None:
            if self.vision_model is None:
                self.vision_model = self._pick_vision_model()
            b64 = base64.b64encode(image_jpeg).decode()
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        else:
            content = user
        delay, waited = 5, 0.0
        for attempt in range(retries):
            model = self.vision_model if image_jpeg is not None else self.model
            kind = "vision" if image_jpeg is not None else "text"
            start = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": content}],
                    **self.extra,
                )
                log.info(f"{self.name} | {model} | {kind} request: {time.time() - start:.1f}s")
                return _parse_json(resp.choices[0].message.content or "")
            except NotFoundError:
                if self._resolved:
                    raise
                # configured model was renamed/retired: switch to one that exists
                self.model, self._resolved = self._pick_model(), True
                log.info(f"{self.name}: model not found, switched to {self.model}")
                if self.cfg["vision_same_model"]:
                    self.vision_model = self.model
                elif image_jpeg is not None:
                    self.vision_model = self._pick_vision_model()
            except BadRequestError as e:
                if self.extra and "reasoning" in str(e).lower():
                    log.info(f"{self.name}: model does not accept fast mode, turning it off")
                    self.extra = {}  # this model rejects reasoning_effort: retry without
                    continue
                raise
            except RateLimitError as e:
                msg = str(e)
                daily = _is_daily_quota(msg)
                wait = _suggested_wait(msg) or delay
                if daily or waited + wait > MAX_WAIT:
                    why = "free quota used up" if daily else "rate limit kept failing"
                    if self._switch_model(why):
                        self.on_wait(f"Free quota used up for one model, switching to {self.model}...")
                        delay, waited = 5, 0.0
                        continue
                    raise QuotaExhausted(f"{why} on every available model. "
                                         f"Details: {_short(msg)}") from e
                log.info(f"{self.name}: rate limited, waiting {wait:.0f}s")
                self.on_wait(f"Free-tier limit reached, waiting {wait:.0f}s...")
                time.sleep(wait)
                waited += wait
                delay = min(delay * 2, 30)
            except (APITimeoutError, APIConnectionError):
                log.info(f"{self.name}: network problem/timeout, retrying")
                if attempt == retries - 1:
                    raise
                time.sleep(3)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < retries - 1:
                    log.info(f"{self.name}: server error {e.status_code}, retrying")
                    time.sleep(delay)
                    continue
                raise
            except ValueError:
                if attempt == retries - 1:
                    raise
        raise RuntimeError("Rate limit kept failing. Wait a minute and try again.")


def _is_daily_quota(msg: str) -> bool:
    m = msg.lower().replace(" ", "")
    if re.search(r"limit\W*0\b", m):  # "limit: 0" = this model has no free quota at all
        return True
    return any(k in m for k in ("perday", "per_day", "requestsperday", "daily", "rpd",
                                "tokensperday", "tpd"))


def _short(msg: str) -> str:
    """The provider's own explanation, without the long JSON around it."""
    m = re.search(r"['\"]message['\"]\s*:\s*['\"](.+?)['\"]\s*[,}]", msg, re.S)
    text = (m.group(1) if m else msg).replace("\\n", " ").strip()
    return text[:220] + ("..." if len(text) > 220 else "")


def _suggested_wait(msg: str) -> float | None:
    """Gemini says retryDelay: '37s'; Groq says 'try again in 7.5s'."""
    for pat in (r"retry[_ ]?delay\W*(\d+(?:\.\d+)?)s", r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s"):
        m = re.search(pat, msg, re.I)
        if m:
            nums = [g for g in m.groups() if g]
            secs = float(nums[-1]) + (float(nums[0]) * 60 if len(nums) == 2 else 0)
            return min(secs + 1, 60)
    return None


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Model did not return valid JSON")


def _translations(data: dict) -> dict:
    data = data.get("translations", data)
    return {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int, float))}


# ------------------------------------------------------------------ agents

class ReaderAgent:
    SYSTEM = ("You are a meticulous document transcriber with perfect OCR, including "
              "handwriting in any language and script. Answer only with JSON.")

    PROMPT = """Transcribe this page image exactly, in its ORIGINAL language (do not translate).
Return JSON: {"elements": [ ... ]} in natural reading order (follow columns correctly).
Element types:
- {"type": "heading", "text": "..."}
- {"type": "paragraph", "text": "..."}   join wrapped lines into one paragraph
- {"type": "verse", "text": "line 1\\nline 2"}  poems, songs, lists, addresses: keep line breaks
- {"type": "caption", "text": "..."}     small text under/next to a picture
- {"type": "table", "rows": [["cell", "cell"], ...]}
- {"type": "image", "box_2d": [ymin, xmin, ymax, xmax]}  photos, illustrations, drawings,
  charts, diagrams, logos, stamps, signatures. Coordinates are 0-1000 of the page.
  Box the whole picture generously. Text that is part of a picture stays in the picture.
Rules: transcribe handwriting as best you can and mark unreadable words as [?].
Skip page numbers and repeated running headers/footers. If the page is blank,
return {"elements": []}."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, page_jpeg: bytes) -> list[dict]:
        data = self.llm.json_chat(self.SYSTEM, self.PROMPT, image_jpeg=page_jpeg)
        elements = data.get("elements", [])
        return [e for e in elements if isinstance(e, dict) and e.get("type")]


class AnalyzerAgent:
    SYSTEM = ("You are a senior literary translator preparing a translation brief. "
              "Answer only with JSON.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, sample: str, target: str) -> dict:
        user = f"""Read this sample from a document that will be translated into {target}.

SAMPLE:
\"\"\"{sample[:6000]}\"\"\"

Return JSON with these keys:
- "genre": e.g. "children's story", "poem", "slogans", "essay", "letter", "mixed"
- "tone": a few words (e.g. "playful, warm")
- "audience": who it is written for
- "summary": 2-3 sentences about the content
- "style_guide": 3-6 concrete instructions for translating THIS text into {target}
  naturally (register, how to handle rhyme/rhythm, idioms, honorifics, etc.)
- "glossary": object mapping recurring names/terms to their {target} rendering
  (transliterate personal names, keep brand names)"""
        try:
            return self.llm.json_chat(self.SYSTEM, user)
        except Exception:
            return {"genre": "unknown", "tone": "neutral", "audience": "general",
                    "summary": "", "style_guide": [], "glossary": {}}


class TranslatorAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, units: list[dict], target: str, brief: dict, context: str) -> dict:
        system = f"""You are an expert literary translator into {target}.
Translate MEANING, emotion and intent - never word-for-word. A native {target}
reader must feel the text was originally written in {target}.

Document brief:
- Genre: {brief.get('genre')}
- Tone: {brief.get('tone')}
- Audience: {brief.get('audience')}
- Summary: {brief.get('summary')}
- Style guide: {json.dumps(brief.get('style_guide', []), ensure_ascii=False)}
- Glossary (use consistently): {json.dumps(brief.get('glossary', {}), ensure_ascii=False)}

Each unit has a "kind": heading, paragraph, verse, caption or cell.
Rules:
- Translate every unit, whatever its source language, into {target}.
- Idioms and proverbs: use an equivalent {target} expression, not a literal one.
- verse (poems/songs): keep the line breaks (\\n), keep the rhythm, rhyme where natural.
- heading/slogans: short and catchy in {target}.
- Units can be parts of one sentence continuing into the next unit; translate so
  that read in order they form correct {target} sentences.
- [?] marks an unreadable word: guess from context or leave it out naturally.
- Numbers, URLs, emails and codes stay unchanged. Text already in {target}
  stays as it is (fix obvious typos only).
- Return ONLY JSON: {{"translations": {{"<id>": "<translated text>", ...}}}}
  with exactly one entry for every input id."""
        payload = [{"id": u["id"], "kind": u["kind"], "text": u["text"]} for u in units]
        user = (f"Previous part (for continuity, do not translate):\n{context or '(start)'}\n\n"
                f"Units to translate:\n{json.dumps(payload, ensure_ascii=False)}")
        return _translations(self.llm.json_chat(system, user))


class ReviewerAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, units: list[dict], drafts: dict, target: str, brief: dict) -> dict:
        system = f"""You are a native {target} editor reviewing a translation of a
{brief.get('genre', 'document')} (tone: {brief.get('tone', 'neutral')}).
Fix anything that is literal, awkward, grammatically wrong, or loses the
original meaning or feeling. Keep verse line breaks. Keep lines that are
already good unchanged.
Return ONLY JSON: {{"translations": {{"<id>": "<final text>", ...}}}} for every id."""
        items = [{"id": u["id"], "kind": u["kind"], "source": u["text"],
                  "draft": drafts.get(u["id"], "")} for u in units]
        try:
            fixed = _translations(self.llm.json_chat(system, json.dumps(items, ensure_ascii=False)))
            return {**drafts, **{k: v for k, v in fixed.items() if v.strip()}}
        except Exception:
            return drafts  # the reviewer is best-effort; keep the draft
