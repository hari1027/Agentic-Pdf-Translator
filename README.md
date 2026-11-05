# 🌐 Agentic PDF Translator (100% free)

Upload a PDF (story book, poem, slogans…), pick a language, click **Convert**,
and download a translated PDF where every picture stays exactly where it was.

## How it works

```
PDF (typed, scanned, photographed, handwritten, any font or language)
  │
  ▼  for every page, pick the best way to read it:
  ├─ TEXT mode   clean digital text → read directly (free, no API call)
  │              pictures are cropped with the old text removed
  └─ VISION mode scans, photos, handwriting, garbled/legacy font encodings
                 (Bamini, Kruti Dev…), multi-column pages, tables, charts
                 → Reader agent (Gemini vision) transcribes the page and
                   marks where every picture is
  │
  ▼
Analyzer agent    genre, tone, audience, style guide, glossary of names
  ▼
Translator agent  translates meaning, idioms, rhyme (batches across pages)
  ▼
Reviewer agent    native-editor pass (optional)
  ▼
Rebuild           a clean PDF, page by page, same page size; each picture sits
                  between the same paragraphs as in the original. Longer
                  translations flow onto an extra page instead of shrinking.
```

After converting, the app lists which pages were read with AI vision and why,
so users know which pages (especially handwriting) to double-check.

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, language, password box (only if needed), Convert with loading screen, Download |
| `agents.py` | Reader (vision), Analyzer, Translator, Reviewer agents + free LLM client |
| `extract.py` | Garbled-text detector, per-page routing, text & vision readers |
| `render.py` | Rebuilds the translated PDF with pictures in context |
| `pipeline.py` | Runs the whole job: read → analyze → translate → review → rebuild |
| `languages.py` / `fonts.py` | Dropdown languages and their free Noto fonts |

## Which model, and why

**Google Gemini (free tier) is the default.** It is one of the
strongest models for Indian languages (Tamil, Malayalam, Hindi…) and for
Japanese/Korean, it understands poetry and idioms, and the free tier needs no
credit card. **Groq** is included as a free backup: very fast,
good for European languages, weaker for Tamil/Malayalam.

Classic translation models (Google Translate API, NLLB, IndicTrans2) are word-
and-sentence level; they can't keep a story's tone or a poem's rhythm. An LLM
that first *understands* the document is what gives a meaningful result.

> No model names are hard-coded: the app asks each provider which models exist
> right now, picks the best, and switches automatically when one is retired or out
> of free quota (each Gemini model has its own quota).
>
> Free tiers have limits (requests per minute/day) that providers change over
> time. The app automatically waits and retries when limited. A ~20 page book
> is fine; for very large books, switch the Reviewer off to halve requests.
> Note: on Gemini's free tier Google may use prompts to improve its products,
> so don't upload confidential documents.

---

## Step 1 – Install uv

uv is a fast Python package & project manager. It also installs Python for you.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Close and reopen the terminal, then check:
```bash
uv --version
```
(Alternative if you already have Python: `pip install uv`.)

## Step 2 – Set up the project

```bash
cd pdf-translator
uv python install 3.12     # downloads Python if you don't have it
uv sync                    # creates .venv and installs everything from pyproject.toml
```

Starting from scratch instead of this folder? The equivalent is:
```bash
uv init pdf-translator && cd pdf-translator
uv add streamlit pymupdf openai fonttools
```

## Step 3 – Get a free API key (no card needed)

- Gemini: https://aistudio.google.com/apikey → *Create API key*
- Groq (optional): https://console.groq.com/keys

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste
the key. The app reads its keys only from this file (there is no settings
panel in the UI). If you add a Groq key too, the app switches to Groq
automatically when Gemini's free limit runs out.

## Step 4 – Run

```bash
uv run streamlit run app.py
```
Opens http://localhost:8501. Upload a PDF → choose language → **Convert** →
**Download**. The app opens in dark mode with no Deploy button or ⋮ menu
(set in `.streamlit/config.toml`). The first time you pick a language, its font (70 KB–16 MB for
CJK) is downloaded and cached in `fonts/`.

## Step 5 – Host it for free (Streamlit Community Cloud)

1. Create a free GitHub account and a new repository; push this folder:
   ```bash
   git init && git add . && git commit -m "PDF translator"
   git branch -M main
   git remote add origin https://github.com/<you>/pdf-translator.git
   git push -u origin main
   ```
   `.gitignore` already keeps `secrets.toml` and `.venv` out of GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. **Deploy.** You get a public `https://<name>.streamlit.app` URL.
   Cloud installs from `requirements.txt` (already included). If you add
   packages later with `uv add`, refresh it with
   `uv export --no-hashes --format requirements-txt > requirements.txt`.

Free apps sleep after a period of no use and wake up when someone opens them.

**Important for a public app:** everyone using it spends *your* free quota.
Share the link only with people you trust, or add a Groq key as a backup.

Alternative free host: Hugging Face Spaces (create a Space with the Docker →
Streamlit template, upload these files, add the key under *Settings → Secrets*).

---

## Limits & tips

- **Scanned and handwritten pages need the Gemini key.** They are read with
  Gemini's vision ability; Groq's vision models are a weaker fallback.
- Each vision page costs 1 request on top of translation. Free-tier daily
  limits apply, so for very large scanned books turn the Reviewer off
  (`REVIEWER = "false"` in secrets) to halve the translation requests.
- Messy handwriting can be misread; unreadable words are marked `[?]` for the
  translator, and the app tells you which pages were read this way.
- Text that is part of a picture (words painted into an illustration, a
  poster photo) stays inside the picture untranslated.
- The output is a clean re-typeset PDF: original decorative fonts, coloured
  backgrounds and exact positions are replaced by a readable layout, while
  every picture stays in its place in the flow and page order is kept.
- Add a language: add a line in `languages.py` with a Noto font name.
