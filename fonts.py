"""Downloads the needed Noto font once and caches it in ./fonts."""
import urllib.request
from pathlib import Path

from languages import LANGUAGES, LATIN

FONT_DIR = Path(__file__).parent / "fonts"


def _download(font: dict) -> Path:
    FONT_DIR.mkdir(exist_ok=True)
    path = FONT_DIR / font["file"]
    if not path.exists() or path.stat().st_size < 10_000:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(font["url"], tmp)
        tmp.replace(path)
    return path


def ensure_fonts(language: str) -> tuple[Path, str]:
    """Returns (font folder, CSS with @font-face rules) for a language."""
    main = LANGUAGES[language]["font"]
    _download(main)
    _download(LATIN)  # fallback for digits / English names inside the text
    css = (
        f"@font-face {{font-family: target; src: url({main['file']});}}\n"
        f"@font-face {{font-family: latin; src: url({LATIN['file']});}}\n"
    )
    return FONT_DIR, css
