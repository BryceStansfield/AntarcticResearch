"""Censor party-identifying information from working papers.

The naive censoring strategy: replace every mention of a target country's name with a
neutral placeholder ("CountryName"), so the authorship models can't read the submitting
party straight off the page. The censored text is embedded separately (see
embeddings/embed_all_documents.py) to build a censored evaluation suite.
"""
import re
import pathlib

from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations

# Working-paper filenames end in a language code (_e, _s, _f, _r); the suite is English.
ENGLISH_SUFFIX = "_e"
# Default countries to censor — the authorship targets.
COUNTRIES = ["Australia", "Australian", "the United Kingdom", "United Kingdom", "the UK", "British", "the USA", "the US", "United States", "American", "Norway", "Norwegian", "Chile", "Chilean"]
PLACEHOLDER = "CountryName"


def get_working_paper_paths() -> list[pathlib.Path]:
    """All unique English working-paper text files (those under a /wp/ directory)."""
    locations = map_all_wp_ip_file_locations()
    paths = {
        pathlib.Path(p)
        for p in locations.values()
        if "/wp/" in p and pathlib.Path(p).stem.endswith(ENGLISH_SUFFIX)
    }
    return sorted(paths)


def _censor_pattern(countries: list[str]) -> re.Pattern:
    """Whole-phrase, case-insensitive matcher for the supplied country names, tolerant of
    OCR whitespace / line breaks. Longest names first so multi-word names take precedence."""
    bodies = [
        r"\s+".join(re.escape(word) for word in country.split())
        for country in sorted(countries, key=len, reverse=True)
    ]
    return re.compile(r"\b(?:" + "|".join(bodies) + r")\b", re.IGNORECASE)


def censor_text(text: str, countries: list[str] = COUNTRIES) -> str:
    """Replace every mention of a supplied country name with ``PLACEHOLDER``."""
    if not countries:
        return text
    return _censor_pattern(countries).sub(PLACEHOLDER, text)


if __name__ == "__main__":
    # Sanity demo: censor the first working paper and show a before/after around a match.
    pattern = _censor_pattern(COUNTRIES)
    sample = get_working_paper_paths()[0]
    original = sample.read_text(encoding="utf-8", errors="ignore")
    print(f"{sample.stem}: censoring {len(pattern.findall(original))} country mention(s) -> {PLACEHOLDER!r}")
    match = pattern.search(original)
    if match:
        start, end = max(0, match.start() - 60), min(len(original), match.end() + 60)
        print("before:", " ".join(original[start:end].split()))
        print("after :", " ".join(censor_text(original[start:end]).split()))
