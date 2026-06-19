"""Build a censored suite of working papers for the authorship models.

The aim is to strip out information that reveals which party submitted a paper, so the
models can't simply read the answer off the page. Most working papers announce their
authorship with a "submitted by ..." line; as a first step this module just reports the
working papers that do NOT contain that phrase, since those will need a different
censoring strategy.
"""
import re
import pathlib

from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations

SUBMITTED_BY = "submitted by"
# Working-paper filenames end in a language code (e.g. _e, _s, _f, _r); the "submitted
# by" check is English-specific, so the suite is restricted to English papers.
ENGLISH_SUFFIX = "_e"

# Countries whose authorship we want to censor; scanned for as in-text mentions.
COUNTRIES = ["Australia", "United Kingdom", "United States", "Norway", "Chile"]
CONTEXT_WINDOW_WORDS = 10
MENTIONS_REPORT_PATH = pathlib.Path("data/working_paper_country_mentions.txt")


def get_working_paper_paths() -> list[pathlib.Path]:
    """All unique English working-paper text files (those under a /wp/ directory)."""
    locations = map_all_wp_ip_file_locations()
    paths = {
        pathlib.Path(p)
        for p in locations.values()
        if "/wp/" in p and pathlib.Path(p).stem.endswith(ENGLISH_SUFFIX)
    }
    return sorted(paths)


def working_papers_missing_phrase(phrase: str = SUBMITTED_BY) -> list[pathlib.Path]:
    """Working papers whose text does not contain ``phrase`` (matched case-insensitively)."""
    needle = phrase.lower()
    missing = []
    for path in get_working_paper_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if needle not in text.lower():
            missing.append(path)
    return missing


def _country_pattern(country: str) -> re.Pattern:
    """Case-insensitive whole-phrase matcher, tolerant of OCR whitespace / line breaks."""
    body = r"\s+".join(re.escape(word) for word in country.split())
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


def iter_mentions(text: str, pattern: re.Pattern, window: int = CONTEXT_WINDOW_WORDS):
    """Yield (left_words, matched_text, right_words) for each match, with ``window`` words
    of context on either side."""
    for match in pattern.finditer(text):
        left = " ".join(text[:match.start()].split()[-window:])
        right = " ".join(text[match.end():].split()[:window])
        yield left, match.group(), right


def collect_country_mentions() -> dict[str, list[dict]]:
    """For every English working paper, gather each country mention with its surrounding
    context and a heuristic flag for whether it reads like an authorship line."""
    patterns = {country: _country_pattern(country) for country in COUNTRIES}
    mentions: dict[str, list[dict]] = {country: [] for country in COUNTRIES}
    for path in get_working_paper_paths():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for country, pattern in patterns.items():
            for left, matched, right in iter_mentions(text, pattern):
                mentions[country].append({
                    "paper": path.stem,
                    "left": left,
                    "match": matched,
                    "right": right,
                    "authorship_like": SUBMITTED_BY in left.lower(),
                })
    return mentions


def main() -> None:
    paths = get_working_paper_paths()
    mentions = collect_country_mentions()

    lines: list[str] = []
    print(f"Country mentions across {len(paths)} English working papers")
    print(f"(authorship heuristic: '{SUBMITTED_BY}' within the {CONTEXT_WINDOW_WORDS} words before a mention)\n")
    for country in COUNTRIES:
        hits = mentions[country]
        n_auth = sum(1 for h in hits if h["authorship_like"])
        pct = (100.0 * n_auth / len(hits)) if hits else 0.0
        print(f"  {country}: {len(hits)} mentions — {n_auth} ({pct:.1f}%) just after '{SUBMITTED_BY}'")

        lines.append(f"===== {country}: {len(hits)} mentions, {n_auth} authorship-like ({pct:.1f}%) =====")
        for h in hits:
            tag = "AUTH" if h["authorship_like"] else "    "
            lines.append(f"{tag} [{h['paper']}] ...{h['left']} «{h['match']}» {h['right']}...")
        lines.append("")

    MENTIONS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote full contexts to {MENTIONS_REPORT_PATH}")


if __name__ == "__main__":
    main()
