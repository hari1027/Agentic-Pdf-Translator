import base64
from pathlib import Path

import streamlit as st

from agents import LLM, PROVIDERS
from languages import LANGUAGES
from pipeline import needs_password, translate_pdf

LOGO = Path(__file__).parent / "assets" / "logo.jpg"
ICON = str(LOGO) if LOGO.exists() else "🌐"  # browser tab icon

st.set_page_config(page_title="PDF Translator Agent", page_icon=ICON,
                   layout="centered", initial_sidebar_state="collapsed")

# Hide Streamlit's header bar (Deploy button, ⋮ menu) and sidebar toggle.
st.markdown("""
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
#MainMenu, footer {display: none !important;}
.block-container {padding-top: 2.5rem;}

/* full-screen loading overlay: blocks every click while converting */
.busy-overlay {position: fixed; inset: 0; z-index: 999999; display: flex;
  align-items: center; justify-content: center;
  background: rgba(10, 12, 18, 0.85);
  -webkit-backdrop-filter: blur(3px); backdrop-filter: blur(3px);}
.busy-card {background: #1e2230; padding: 2rem 2.5rem; border-radius: 16px;
  text-align: center; width: min(400px, 90vw); box-shadow: 0 10px 40px rgba(0,0,0,.5);}
.busy-spinner {width: 48px; height: 48px; margin: 0 auto 1rem; border-radius: 50%;
  border: 4px solid rgba(255,255,255,.15); border-top-color: #4F8BF9;
  animation: busy-spin 1s linear infinite;}
@keyframes busy-spin {to {transform: rotate(360deg);}}
.busy-title {font-size: 1.15rem; font-weight: 600; color: #fff;}
.busy-msg {margin-top: .4rem; color: #b8bfd0; font-size: .92rem;}
.busy-bar {height: 6px; margin-top: 1rem; border-radius: 3px; overflow: hidden;
  background: rgba(255,255,255,.1);}
.busy-bar > div {height: 100%; background: #4F8BF9; transition: width .4s;}
.busy-note {margin-top: .8rem; color: #7d859a; font-size: .8rem;}

/* header: logo + title side by side on every screen size */
.app-header {display: flex; align-items: center; gap: .9rem; margin-bottom: .75rem;}
.app-header img {width: 56px; height: 56px; object-fit: contain; flex-shrink: 0;
  border-radius: 12px;}
.app-emoji {font-size: 2.6rem; line-height: 1;}
.app-title {font-size: 2.3rem; font-weight: 700; line-height: 1.15; margin: 0;}

/* phones */
@media (max-width: 640px) {
  .block-container {padding: 1.25rem 1rem 3rem 1rem !important;}
  .app-header {gap: .65rem;}
  .app-header img {width: 42px; height: 42px; border-radius: 9px;}
  .app-emoji {font-size: 2rem;}
  .app-title {font-size: 1.55rem;}
  .stMarkdown p {font-size: .95rem;}
  /* big, full-width tap targets */
  .stButton button, .stDownloadButton button {width: 100%; min-height: 3rem;
    font-size: 1rem;}
  /* 16px+ stops iPhone Safari from zooming into the field when tapped */
  input, textarea, [data-baseweb="select"] * {font-size: 16px !important;}
  .busy-card {padding: 1.5rem 1.25rem;}
}
</style>
""", unsafe_allow_html=True)


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml present
        return default


# Settings come from .streamlit/secrets.toml. Gemini first; Groq is the backup.
BACKENDS = [
    (name, key, secret(f"{cfg['secret'].split('_')[0]}_MODEL", cfg["default_model"]))
    for name, cfg in PROVIDERS.items()
    if (key := secret(cfg["secret"]))
]
REVIEW = str(secret("REVIEWER", "true")).lower() != "false"

OVERLAY = """<div class="busy-overlay"><div class="busy-card">
<div class="busy-spinner"></div>
<div class="busy-title">Translating your PDF…</div>
<div class="busy-msg">{msg}</div>
<div class="busy-bar"><div style="width:{pct}%"></div></div>
<div class="busy-note">This can take a few minutes. Keep this tab open and
your screen on (on a phone, don't switch apps or lock the screen).</div>
</div></div>"""

# ---------------------------------------------------------------- main
def header_html() -> str:
    if LOGO.exists():  # embed the logo so it sits next to the title, even on phones
        mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".svg": "svg+xml", ".webp": "webp"}.get(
            LOGO.suffix.lower(), "png")
        b64 = base64.b64encode(LOGO.read_bytes()).decode()
        icon = f'<img src="data:image/{mime};base64,{b64}" alt="logo">'
    else:
        icon = '<span class="app-emoji">🌐</span>'
    return f'<div class="app-header">{icon}<div class="app-title">Agentic PDF Translator</div></div>'


st.markdown(header_html(), unsafe_allow_html=True)

st.markdown(
    "Bring your storybooks, poems, slogans and documents to readers in their own "
    "language. A team of AI agents first **reads and understands** your PDF: typed, "
    "scanned or even handwritten. It then **translates the meaning, emotion and "
    "rhythm** rather than word for word, and a **native-style editor** polishes "
    "every line. Every picture stays right where it belongs in the story, so the "
    "translated PDF reads like it was written in your language."
)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
language = st.selectbox("Translate to", list(LANGUAGES), index=0)

password = ""
if uploaded is not None and needs_password(uploaded.getvalue()):
    password = st.text_input("This PDF is password protected. Enter its password:",
                             type="password")

convert = st.button("🔄 Convert", type="primary", disabled=uploaded is None)

if uploaded is not None and st.session_state.get("source_name") != uploaded.name:
    st.session_state.pop("result", None)  # new file -> clear old result
    st.session_state["source_name"] = uploaded.name

if convert:
    if not BACKENDS:
        st.error("No API key configured. Add GEMINI_API_KEY to .streamlit/secrets.toml "
                 "(or to the app's Secrets when hosted).")
        st.stop()
    overlay = st.empty()

    def show_progress(p: float, msg: str) -> None:
        overlay.markdown(OVERLAY.format(msg=msg, pct=round(min(p, 1.0) * 100)),
                         unsafe_allow_html=True)

    show_progress(0.0, "Starting...")
    errors, done = [], False
    for provider, key, model in BACKENDS:
        try:
            result, brief, notes = translate_pdf(
                uploaded.getvalue(), language, LLM(provider, key, model), REVIEW,
                password=password, progress=show_progress,
            )
            st.session_state["result"] = (result, language, notes)
            done = True
            break
        except ValueError as e:  # problem with the PDF itself: another provider won't help
            errors.append(("PDF", e))
            break
        except Exception as e:  # key/quota/network problem: try the backup provider
            errors.append((provider, e))
    overlay.empty()  # remove the loading screen
    if done:
        st.success("Translation finished!")
    for provider, e in errors:  # show every provider's error, not just the last
        msg = str(e)
        if not msg.startswith(provider):
            msg = f"{provider}: {msg}"
        (st.warning if done else st.error)(msg)

if "result" in st.session_state and uploaded is not None:
    result, lang, notes = st.session_state["result"]
    if notes:
        st.info("Some pages were read with AI vision, because their text could not be "
                "taken directly from the PDF. Please check these pages, especially "
                "handwriting:\n\n" + "\n".join(f"- {n}" for n in notes))
    base = uploaded.name.rsplit(".", 1)[0]
    st.download_button("⬇️ Download translated PDF", data=result,
                       file_name=f"{base}_{lang.split()[0].lower()}.pdf",
                       mime="application/pdf", type="primary")
