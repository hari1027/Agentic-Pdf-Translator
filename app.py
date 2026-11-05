import streamlit as st

from agents import LLM, PROVIDERS
from languages import LANGUAGES
from pipeline import render_preview, translate_pdf
from pathlib import Path

LOGO = Path(__file__).parent / "assets" / "logo.jpg"

st.set_page_config(page_title="Agentic PDF Translator", page_icon=str(LOGO),
                   layout="centered", initial_sidebar_state="collapsed")

# Hide Streamlit's header bar (Deploy button, ⋮ menu) and sidebar toggle.
st.markdown("""
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
#MainMenu, footer {display: none !important;}
.block-container {padding-top: 2.5rem;}
</style>
""", unsafe_allow_html=True)


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml present
        return default


# Settings now come from .streamlit/secrets.toml instead of a sidebar.
# Gemini is used first; Groq is the automatic backup if a key is set.
BACKENDS = [
    (name, key, secret(f"{cfg['secret'].split('_')[0]}_MODEL", cfg["default_model"]))
    for name, cfg in PROVIDERS.items()
    if (key := secret(cfg["secret"]))
]
REVIEW = str(secret("REVIEWER", "true")).lower() != "false"

# ---------------------------------------------------------------- main
st.title("📖 Agentic PDF Translator")
st.markdown(
    "Bring your storybooks, poems, slogans and documents to readers in their own "
    "language. A team of AI agents first **reads and understands** your PDF: its "
    "story, tone and audience. It then **translates the meaning, emotion and "
    "rhythm** rather than word for word, and a **native-style editor** polishes "
    "every line. Your pictures and page layout stay exactly where they were, so "
    "the translated PDF looks like the original, just in a new language."
)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
language = st.selectbox("Translate to", list(LANGUAGES), index=0)
convert = st.button("🔄 Convert", type="primary", disabled=uploaded is None)

if uploaded is not None and st.session_state.get("source_name") != uploaded.name:
    st.session_state.pop("result", None)  # new file -> clear old result
    st.session_state["source_name"] = uploaded.name

if convert:
    if not BACKENDS:
        st.error("No API key configured. Add GEMINI_API_KEY to .streamlit/secrets.toml "
                 "(or to the app's Secrets when hosted).")
        st.stop()
    bar = st.progress(0.0, text="Starting...")
    errors, done = [], False
    for provider, key, model in BACKENDS:
        try:
            result, brief = translate_pdf(
                uploaded.getvalue(), language, LLM(provider, key, model), REVIEW,
                progress=lambda p, msg: bar.progress(min(p, 1.0), text=msg),
            )
            st.session_state["result"] = (result, brief, language)
            done = True
            break
        except ValueError as e:  # e.g. scanned PDF: another provider won't help
            errors.append(("PDF", e))
            break
        except Exception as e:  # key/quota/network problem: try the backup provider
            errors.append((provider, e))
    if done:
        st.success("Translation finished!")
    else:
        bar.empty()
    for provider, e in errors:  # show every provider's error, not just the last
        (st.warning if done else st.error)(f"{provider}: {e}")

if "result" in st.session_state and uploaded is not None:
    result, brief, lang = st.session_state["result"]
    base = uploaded.name.rsplit(".", 1)[0]
    st.download_button("⬇️ Download translated PDF", data=result,
                       file_name=f"{base}_{lang.split()[0].lower()}.pdf",
                       mime="application/pdf", type="primary")

    with st.expander("What the AI understood about your document"):
        st.write(f"**Genre:** {brief.get('genre')}  \n**Tone:** {brief.get('tone')}  \n"
                 f"**Audience:** {brief.get('audience')}")
        st.write(brief.get("summary", ""))

    page = st.number_input("Preview page", min_value=1, value=1, step=1) - 1
    left, right = st.columns(2)
    left.image(render_preview(uploaded.getvalue(), page), caption="Original")
    right.image(render_preview(result, page), caption=lang)
