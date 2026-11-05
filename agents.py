"""The agentic part: three cooperating LLM agents.

1. AnalyzerAgent   - reads a sample of the document first and works out what it
                     is (story / poem / slogans ...), its tone, audience, and a
                     glossary for names and recurring terms.
2. TranslatorAgent - translates page by page using that "style brief" plus the
                     previous page as context, so the result reads naturally and
                     stays consistent (not word-for-word).
3. ReviewerAgent   - (optional) reads the draft like a native editor and fixes
                     anything that sounds unnatural or loses meaning.

Both free providers expose an OpenAI-compatible endpoint, so one `openai`
client works for either of them.
"""
import json
import re
import time

from openai import OpenAI, RateLimitError, APIStatusError, NotFoundError

PROVIDERS = {
    "Google Gemini (recommended)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "secret": "GEMINI_API_KEY",
        # tried in order if the configured model is gone; newer names first
        "preferred": ["gemini-flash-latest", "gemini-3-flash", "gemini-2.5-flash",
                      "gemini-2.0-flash"],
        "avoid": ["lite", "image", "tts", "live", "embed", "audio", "thinking", "vision"],
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
        "key_page": "https://console.groq.com/keys",
    },
}


class LLM:
    """Thin wrapper with JSON parsing and retry on free-tier rate limits."""

    def __init__(self, provider: str, api_key: str, model: str):
        self.cfg = PROVIDERS[provider]
        self.client = OpenAI(api_key=api_key, base_url=self.cfg["base_url"])
        self.model = model
        self._resolved = False

    def _pick_model(self) -> str:
        """Ask the provider which models exist and choose the best chat model."""
        ids = [m.id.removeprefix("models/") for m in self.client.models.list()]
        for name in self.cfg["preferred"]:
            if name in ids:
                return name
        usable = sorted((i for i in ids if not any(a in i.lower() for a in self.cfg["avoid"])),
                        reverse=True)
        flash = [i for i in usable if "flash" in i]  # Gemini: prefer the fast models
        if flash or usable:
            return (flash or usable)[0]
        raise RuntimeError(f"No usable chat model found. Available: {ids[:15]}")

    def json_chat(self, system: str, user: str, retries: int = 6) -> dict:
        delay = 5
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return _parse_json(resp.choices[0].message.content or "")
            except NotFoundError:
                if self._resolved:
                    raise
                # configured model was renamed/retired: switch to one that exists
                self.model, self._resolved = self._pick_model(), True
            except RateLimitError:
                # free tiers allow only a few requests per minute: wait and retry
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < retries - 1:
                    time.sleep(delay)
                    continue
                raise
            except ValueError:
                if attempt == retries - 1:
                    raise
        raise RuntimeError("Rate limit kept failing. Wait a minute and try again.")


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Model did not return valid JSON")


# ------------------------------------------------------------------ agents

class AnalyzerAgent:
    SYSTEM = (
        "You are a senior literary translator preparing a translation brief. "
        "Answer only with JSON."
    )

    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, sample: str, target: str) -> dict:
        user = f"""Read this sample from a document that will be translated into {target}.

SAMPLE:
\"\"\"{sample[:6000]}\"\"\"

Return JSON with these keys:
- "genre": e.g. "children's story", "poem", "slogans", "essay", "mixed"
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

    def run(self, blocks: list[dict], target: str, brief: dict, context: str) -> dict:
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

Rules:
- Idioms and proverbs: use an equivalent {target} expression, not a literal one.
- Poems/songs: keep line breaks (\\n), keep the rhythm, rhyme where natural.
- Slogans: make them short and catchy in {target}.
- Prose: line breaks inside a block are just layout wrapping; write flowing text.
- Blocks may be fragments of a sentence that continues in the next block;
  translate so that read in order they form correct {target} sentences.
- Page numbers, URLs, emails and codes stay unchanged.
- Return ONLY JSON: {{"translations": {{"<id>": "<translated text>", ...}}}}
  with exactly one entry for every input id."""
        payload = [{"id": b["id"], "text": b["text"]} for b in blocks]
        user = (f"Previous page (for continuity, do not translate):\n{context or '(start)'}\n\n"
                f"Blocks to translate:\n{json.dumps(payload, ensure_ascii=False)}")
        data = self.llm.json_chat(system, user)
        return {str(k): str(v) for k, v in data.get("translations", data).items()}


class ReviewerAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def run(self, blocks: list[dict], drafts: dict, target: str, brief: dict) -> dict:
        system = f"""You are a native {target} editor reviewing a translation of a
{brief.get('genre', 'document')} (tone: {brief.get('tone', 'neutral')}).
Fix anything that is literal, awkward, grammatically wrong, or loses the
original meaning or feeling. Keep lines that are already good unchanged.
Return ONLY JSON: {{"translations": {{"<id>": "<final text>", ...}}}} for every id."""
        items = [{"id": b["id"], "source": b["text"], "draft": drafts.get(b["id"], "")}
                 for b in blocks]
        try:
            data = self.llm.json_chat(system, json.dumps(items, ensure_ascii=False))
            fixed = {str(k): str(v) for k, v in data.get("translations", data).items()}
            return {**drafts, **{k: v for k, v in fixed.items() if v.strip()}}
        except Exception:
            return drafts  # the reviewer is best-effort; keep the draft
