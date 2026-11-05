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

MAX_WAIT = 40  # never sit on one model's rate limit longer than this; switch instead


class QuotaExhausted(RuntimeError):
    """Daily free quota is used up: retrying is pointless, use the backup provider."""

PROVIDERS = {
    "Google Gemini (recommended)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "auto",  # pick the best model that exists right now
        "secret": "GEMINI_API_KEY",
        "family": "gemini",
        "exclude": ["image", "tts", "live", "embed", "audio", "vision", "learnlm", "gemma",
                    "aqa", "imagen", "veo", "robotics", "computer-use", "native", "nano"],
        "vision_same_model": True,  # Gemini Flash reads images itself
        "fast_mode": {"reasoning_effort": "low"},  # less "thinking" = much faster
        "key_page": "https://aistudio.google.com/apikey",
    },
    "Groq (Llama)": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "auto",
        "secret": "GROQ_API_KEY",
        "family": "groq",
        # good multilingual models, best first (any of them may be retired later)
        "favourites": ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct-0905",
                       "moonshotai/kimi-k2-instruct", "llama-3.3-70b-versatile",
                       "qwen/qwen3-32b", "openai/gpt-oss-20b"],
        "exclude": ["whisper", "guard", "tts", "playai", "orpheus", "compound", "prompt",
                    "safeguard", "distil", "allam", "8b", "instant"],
        "vision_same_model": False,
        "vision_hints": ["vision", "llama-4", "scout", "maverick", "-vl", "vl-", "llava",
                         "pixtral", "gemma-3"],
        "fast_mode": {"reasoning_effort": "low"},  # dropped automatically if unsupported
        "key_page": "https://console.groq.com/keys",
    },
}


class LLM:
    """JSON chat with automatic model discovery, quota-aware model switching,
    rate-limit waits and JSON repair. Never relies on a fixed model name."""

    def __init__(self, provider: str, api_key: str, model: str = "auto"):
        self.name = provider
        self.cfg = PROVIDERS[provider]
        # max_retries=0: we handle retries ourselves (the SDK's hidden retries add up)
        self.client = OpenAI(api_key=api_key, base_url=self.cfg["base_url"],
                             max_retries=0, timeout=120)
        self.extra = dict(self.cfg["fast_mode"])
        self.on_wait = lambda msg: None  # the app shows waits on the loading screen
        self.model = None if (model or "auto").lower() == "auto" else model
        self.vision_model = None
        self.dead, self.vision_dead, self.ok = set(), set(), set()
        self.no_json_mode = set()  # models whose strict JSON mode keeps failing
        self._ids = None

    # ---------------------------------------------------------- model choice
    def _model_ids(self) -> list[str]:
        if self._ids is None:
            self._ids = [m.id.removeprefix("models/") for m in self.client.models.list()]
            log.info(f"{self.name}: {len(self._ids)} models available")
        return self._ids

    def _ranked(self) -> list[str]:
        ids = [i for i in self._model_ids()
               if not any(x in i.lower() for x in self.cfg["exclude"])]
        if self.cfg["family"] == "gemini":
            def key(i: str):
                low = i.lower()
                m = re.search(r"gemini-(\d+(?:\.\d+)?)", low)
                version = float(m.group(1)) if m else (99.0 if "latest" in low else 0.0)
                tier = (1 if "lite" in low else 0) if "flash" in low else (2 if "pro" in low else 3)
                unstable = any(t in low for t in ("preview", "exp", "-0", "-1"))
                return (tier, unstable, -version, low)
            return sorted((i for i in ids if "gemini" in i.lower()), key=key)
        fav = [f for f in self.cfg.get("favourites", []) if f in ids]
        return fav + sorted(i for i in ids if i not in fav)

    def _text_model(self) -> str:
        if self.model is None or self.model in self.dead:
            choices = [i for i in self._ranked() if i not in self.dead]
            if not choices:
                raise QuotaExhausted("no working model is left on this provider")
            self.model = choices[0]
            log.info(f"{self.name}: using {self.model}")
        return self.model

    def _vision_for(self) -> str:
        if self.cfg["vision_same_model"]:
            return self._text_model()
        if self.vision_model is None:
            ids = self._model_ids()
            hints = self.cfg.get("vision_hints", [])
            named = [i for i in ids if any(h in i.lower() for h in hints)]
            others = [i for i in self._ranked() if i not in named]
            for m in [i for i in named + others if i not in self.vision_dead][:6]:
                if self._accepts_images(m):
                    self.vision_model = m
                    log.info(f"{self.name}: using {m} for reading pages")
                    break
                self.vision_dead.add(m)
            else:
                raise RuntimeError("no vision model is available for reading scanned pages")
        return self.vision_model

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

    def _drop(self, model: str, vision: bool, why: str, detail: str = "") -> None:
        """Marks a model unusable for this job; the next call picks another one."""
        if vision and not self.cfg["vision_same_model"]:
            self.vision_dead.add(model)
            self.vision_model = None
        else:
            self.dead.add(model)
        log.info(f"{self.name}: {model} {why}, trying another model")
        self.on_wait(f"{model} {why}, switching model...")
        left = [i for i in self._ranked() if i not in self.dead]
        if not vision and not left:
            raise QuotaExhausted(f"{why} on every available model."
                                 + (f" Details: {detail}" if detail else ""))

    # ---------------------------------------------------------- the call
    def json_chat(self, system: str, user: str, image_jpeg: bytes | None = None) -> dict:
        vision = image_jpeg is not None
        if vision:
            b64 = base64.b64encode(image_jpeg).decode()
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        else:
            content = user
        use_json, bad_json, delay, waited, net = True, 0, 5, 0.0, 0
        for _ in range(40):
            model = self._vision_for() if vision else self._text_model()
            kind = "vision" if vision else "text"
            args = dict(model=model, temperature=0.2,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": content}], **self.extra)
            if use_json and model not in self.no_json_mode:
                args["response_format"] = {"type": "json_object"}
            start = time.time()
            try:
                resp = self.client.chat.completions.create(**args)
                self.ok.add(model)
                log.info(f"{self.name} | {model} | {kind} request: {time.time() - start:.1f}s")
                text = resp.choices[0].message.content or ""
                try:
                    return _parse_json(text)
                except ValueError:
                    bad_json += 1
                    log.info(f"{self.name}: answer was not valid JSON (attempt {bad_json})")
                    if bad_json >= 2:
                        raise  # the pipeline splits the work into smaller parts
                    continue
            except NotFoundError:
                self._drop(model, vision, "is not available (retired or renamed)")
                delay, waited = 5, 0.0
            except BadRequestError as e:
                m = str(e).lower()
                if self.extra and "reasoning" in m:
                    log.info(f"{self.name}: {model} does not accept fast mode, turning it off")
                    self.extra = {}
                elif "json" in m or "failed_generation" in m:
                    # usually a long answer got cut off: retry without strict JSON mode
                    bad_json += 1
                    if model not in self.no_json_mode:
                        log.info(f"{self.name}: JSON mode failed on {model}, using plain mode")
                        self.no_json_mode.add(model)
                    if bad_json >= 2:
                        raise ValueError("answer was cut off (too much text in one request)") from e
                elif any(k in m for k in ("decommission", "deprecat", "not supported",
                                          "does not support", "no longer", "image")):
                    self._drop(model, vision, "cannot be used")
                    delay, waited = 5, 0.0
                else:
                    raise
            except RateLimitError as e:
                msg = str(e)
                wait = _suggested_wait(msg) or delay
                fresh = model not in self.ok  # failed on its very first request
                if _is_daily_quota(msg) or fresh or waited + wait > MAX_WAIT:
                    why = "has no free quota left" if (_is_daily_quota(msg) or fresh) \
                        else "keeps hitting the rate limit"
                    others = [i for i in self._ranked() if i not in self.dead and i != model]
                    if others or vision:
                        self._drop(model, vision, why, _short(msg))
                        delay, waited = 5, 0.0
                        continue
                    if _is_daily_quota(msg) or fresh or waited + wait > 2 * MAX_WAIT:
                        raise QuotaExhausted(f"free quota used up on every available model. "
                                             f"Details: {_short(msg)}") from e
                log.info(f"{self.name}: rate limited, waiting {wait:.0f}s")
                self.on_wait(f"Free-tier limit reached, waiting {wait:.0f}s...")
                time.sleep(wait)
                waited += wait
                delay = min(delay * 2, 30)
            except (APITimeoutError, APIConnectionError):
                net += 1
                log.info(f"{self.name}: network problem/timeout, retrying")
                if net >= 3:
                    raise
                time.sleep(3)
            except APIStatusError as e:
                if e.status_code >= 500:
                    net += 1
                    log.info(f"{self.name}: server error {e.status_code}, retrying")
                    if net >= 3:
                        self._drop(model, vision, f"keeps failing (error {e.status_code})")
                        net = 0
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("Too many failed attempts. Wait a minute and try again.")


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
