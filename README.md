# 🌐 Agentic PDF Translator (100% free)

Upload a PDF (story book, poem, slogans…), pick a language, click **Convert**,
and download a translated PDF where every picture stays exactly where it was.

## How it works

```
PDF ──► PyMuPDF extracts text blocks + positions (images untouched)
            │
            ▼
   ┌─ Analyzer agent ─┐  reads the document: genre, tone, audience,
   │                  │  style guide, glossary of names
   └────────┬─────────┘
            ▼
   ┌─ Translator agent ┐  page by page, using the brief + previous page
   │                   │  → translates meaning, idioms, rhyme, not words
   └────────┬──────────┘
            ▼
   ┌─ Reviewer agent ──┐  native-editor pass that fixes awkward/literal lines
   └────────┬──────────┘  (optional toggle)
            ▼
PyMuPDF erases ONLY the original text and writes the translation into the
same boxes with the right Noto font  ──►  translated PDF (images in place)
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (upload, language dropdown, Convert, Download, preview) |
| `agents.py` | The 3 agents + free LLM client with rate-limit retry |
| `pipeline.py` | PDF read → agents → write back in place |
| `languages.py` | Dropdown languages and the font each needs |
| `fonts.py` | Downloads the free Noto font on first use |

## Which model, and why

**Google Gemini 2.5 Flash (free tier) is the default.** It is one of the
strongest models for Indian languages (Tamil, Malayalam, Hindi…) and for
Japanese/Korean, it understands poetry and idioms, and the free tier needs no
credit card. **Groq (Llama 3.3 70B)** is included as a free backup: very fast,
good for European languages, weaker for Tamil/Malayalam.

Classic translation models (Google Translate API, NLLB, IndicTrans2) are word-
and-sentence level; they can't keep a story's tone or a poem's rhythm. An LLM
that first *understands* the document is what gives a meaningful result.

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

- **Scanned PDFs** (text is part of a photo) have no selectable text. OCR them
  first, e.g. free `ocrmypdf input.pdf output.pdf`, then upload the result.
- Translations are usually longer than the source; text is auto-shrunk to fit
  its original box so the layout never breaks.
- Text drawn *inside* a picture (part of the image pixels) can't be translated.
- Original decorative fonts are replaced by clean Noto fonts for the target
  script (the original fonts don't contain Tamil/Japanese/etc. letters).
- Add a language: add a line in `languages.py` with a Noto font name.
