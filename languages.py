"""Supported target languages and the Noto font each one needs.

Every script needs a font that contains its glyphs, otherwise the PDF shows
empty boxes. Noto fonts are free (SIL Open Font License) and cover all of them.
"""

NOTO = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/{name}/hinted/ttf/{name}-Regular.ttf"
CJK = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/{folder}/{name}-Regular.otf"


def noto(name: str) -> dict:
    return {"file": f"{name}-Regular.ttf", "url": NOTO.format(name=name)}


def cjk(folder: str, name: str) -> dict:
    return {"file": f"{name}-Regular.otf", "url": CJK.format(folder=folder, name=name)}


LATIN = noto("NotoSans")  # also covers Cyrillic and Greek

# name shown in dropdown -> settings
LANGUAGES = {
    "Tamil":                {"font": noto("NotoSansTamil")},
    "Hindi":                {"font": noto("NotoSansDevanagari")},
    "Malayalam":            {"font": noto("NotoSansMalayalam")},
    "Telugu":               {"font": noto("NotoSansTelugu")},
    "Kannada":              {"font": noto("NotoSansKannada")},
    "Bengali":              {"font": noto("NotoSansBengali")},
    "Marathi":              {"font": noto("NotoSansDevanagari")},
    "Gujarati":             {"font": noto("NotoSansGujarati")},
    "Punjabi (Gurmukhi)":   {"font": noto("NotoSansGurmukhi")},
    "Urdu":                 {"font": noto("NotoNaskhArabic"), "rtl": True},
    "Arabic":               {"font": noto("NotoNaskhArabic"), "rtl": True},
    "English":              {"font": LATIN},
    "Spanish":              {"font": LATIN},
    "French":               {"font": LATIN},
    "German":               {"font": LATIN},
    "Portuguese":           {"font": LATIN},
    "Italian":              {"font": LATIN},
    "Russian":              {"font": LATIN},
    "Japanese":             {"font": cjk("Japanese", "NotoSansCJKjp")},
    "Korean":               {"font": cjk("Korean", "NotoSansCJKkr")},
    "Chinese (Simplified)": {"font": cjk("SimplifiedChinese", "NotoSansCJKsc")},
}
