"""Is the *direct* country-authorship signal linearly encoded in the Qwen embedding space?

"Direct" signal = explicit statements of authorship / country-linked linguistic flourishes — the
country information that is NOT mediated through topic. We probe it by injection: take working papers
(a deterministic sample, or --all-wps for the whole corpus) and, for each, prepend the line

    THE FOLLOWING DOCUMENT IS AUTHORED BY {country}

for every country in our target list, over the naive-censored body. Embedding the injected strings and
differencing against the un-prefixed censored embedding isolates the effect of the *explicit authorship
claim alone* (the document body is held fixed). We then decompose each shift and ask three questions:

    delta[i,c]   = emb(prefix_c + doc_i) - emb(doc_i)      # full effect of the added line
    generic[i]   = mean_c delta[i,c]                       # shared "an authorship line exists" part
    country[i,c] = delta[i,c] - generic[i]                 # the country-SPECIFIC part

  1. Magnitude — how far does the explicit line move the embedding (vs the ~0.004 natural raw->naive
     censorship shift), and how much of that is country-specific rather than the generic prefix effect?
  2. Stability — is each country's direction consistent across documents (a document-independent
     "authored by X" axis), measured by mean pairwise cosine of country[i,c] over docs?
  3. Separation — do different countries point in different directions (cross-country cosine of the
     mean per-country direction)?

Unlike the other adhoc analyses this one is not purely cache-only: the injected strings are novel, so
the FIRST run makes live embedding calls (needs OPENROUTER_API_KEY). They are persisted to
``document_embeddings.sqlite3`` (keyed by sha256 of the injected text, type ``DirectCountrySignalProbeV1``),
so reruns with the same documents/prefix are cache reads. It is deterministic given the sample seed.

This lives with the authorship classifier because its ``direct_country_directions_allwps.npz`` output is
consumed by ``country_signal_projection.CountrySignalProjector`` to orthogonalise embeddings against the
direct signal (``country_authorship_classifier --orthogonalize-country``).

Usage:
    python -m working_paper_authorship.direct_country_signal_probe                 # 10-doc sample
    python -m working_paper_authorship.direct_country_signal_probe --all-wps -w 80 # every WP, 80 workers

Outputs (to ``data/country_signal/``; ``_allwps`` suffix in --all-wps mode so the sample is not clobbered):
  * ``direct_country_signal_report[...].txt``   — the three-question summary above
  * ``direct_country_signal_shifts[...].csv``   — per (document, country) cosine shift magnitude
  * ``direct_country_directions[...].npz``      — mean unit direction per country (+ consistency), for
                                                  downstream work (the projective embedding censor)
"""
import argparse
import hashlib
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from embeddings.working_paper_censorship import censor_text, get_working_paper_paths
from embeddings.document_embeddings import get_or_generate_embedding
from working_paper_authorship import country_authorship_classifier as cc

OUTPUT_DIR = pathlib.Path("data/country_signal")
COUNTRIES = cc.COUNTRIES
N_DOCS = 10
SAMPLE_SEED = 0
MIN_BODY_WORDS = 80          # drop blank / failed-OCR scans: prepending a line to those swamps the signal
DEFAULT_WORKERS = 32
INJECTION_EMBED_TYPE = "DirectCountrySignalProbeV1"  # document_type for the persisted injected embeddings
PREFIX = "THE FOLLOWING DOCUMENT IS AUTHORED BY {country}"
NATURAL_CENSORSHIP_SHIFT = 0.0040  # reference: mean raw->naive whole-doc cosine shift (see paired-space probe)


def sample_documents() -> list[dict]:
    """A deterministic sample of non-degenerate target-country WPs, with their naive-censored body."""
    rng = np.random.default_rng(SAMPLE_SEED)
    records = cc.load_working_papers()
    records = [r for r in records if len(censor_text(r["text"]).split()) >= MIN_BODY_WORDS]
    chosen = rng.choice(len(records), size=min(N_DOCS, len(records)), replace=False)
    docs = []
    for i in chosen:
        r = records[i]
        docs.append({
            "stem": r["stem"],
            "censored": censor_text(r["text"]),
            "true_authors": [c for c, b in zip(COUNTRIES, r["label"]) if b],
        })
    return docs


def all_documents() -> list[dict]:
    """Every non-degenerate working paper in the corpus, with its naive-censored body.

    Robustness run: labels are not needed for the direction analysis, so ``true_authors`` is filled
    best-effort from the party lookup (target countries only) and left empty when unknown."""
    lookup = cc._build_parties_lookup()
    docs = []
    for path in get_working_paper_paths():
        censored = censor_text(path.read_text(encoding="utf-8", errors="ignore"))
        if len(censored.split()) < MIN_BODY_WORDS:
            continue
        parties = lookup.get(path.stem)
        authors = sorted(cc.parties_to_target_countries(parties)) if parties is not None and not isinstance(parties, float) else []
        docs.append({"stem": path.stem, "censored": censored, "true_authors": authors})
    return docs


def embed_all(texts: list[str], workers: int) -> list[np.ndarray]:
    """Embed each string, persisting to the cache (keyed by sha256) so reruns are cache reads.

    ``get_or_generate_embedding`` returns the cached vector when present (the un-prefixed baselines are
    already cached as the naive-full embeddings) and otherwise generates it live and stores it — so the
    first run persists the novel injected strings and later runs read them back. Retries transient
    API/DB errors a few times; runs `workers`-way concurrent."""
    def one(text: str) -> np.ndarray | None:
        uuid = hashlib.sha256(text.encode()).hexdigest()
        for attempt in range(5):
            try:
                return np.asarray(get_or_generate_embedding(uuid, INJECTION_EMBED_TYPE, text), dtype=np.float32)
            except Exception:
                if attempt == 4:
                    # Persistent failure (e.g. a sporadic "No embedding data received" from the
                    # provider): give up on this one string rather than abort the whole run — the
                    # caller drops the affected document. Its cache stays unpopulated, so a rerun retries it.
                    return None
                time.sleep(1.5 * (attempt + 1))

    results: list[np.ndarray | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, t): i for i, t in enumerate(texts)}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 1000 == 0 or done == len(texts):
                print(f"    embedded {done}/{len(texts)}")
    return results


def mean_pairwise_cosine(unit_rows: np.ndarray) -> float:
    """Mean cosine similarity over all unordered pairs of unit row-vectors, in O(n*d).

    Uses the identity  sum_{i<j} u_i . u_j = (||sum_i u_i||^2 - n) / 2,  so the mean over the n(n-1)/2
    pairs is (||sum||^2 - n) / (n(n-1)). Exact — no subsampling — and scales to thousands of docs."""
    n = unit_rows.shape[0]
    if n < 2:
        return float("nan")
    s = unit_rows.sum(axis=0)
    return float((s @ s - n) / (n * (n - 1)))


def run(all_wps: bool = False, workers: int = DEFAULT_WORKERS) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_allwps" if all_wps else ""
    print("Loading documents...")
    docs = all_documents() if all_wps else sample_documents()
    n = len(docs)
    print(f"  documents: {n}")

    # Build the embedding jobs: per doc a baseline (censored, no prefix) + one per country, in that order.
    texts = []
    for d in docs:
        texts.append(d["censored"])
        texts.extend(f"{PREFIX.format(country=c)}\n\n{d['censored']}" for c in COUNTRIES)
    print(f"Embedding {len(texts)} strings ({workers} workers; cached ones are instant)...")
    vecs = embed_all(texts, workers)

    # Reshape into (n, 1+len(COUNTRIES), dim): column 0 = baseline, columns 1.. = per-country injections.
    # Drop any document with an unembeddable string (a sporadic provider failure) rather than abort;
    # over thousands of docs a handful dropping does not affect the aggregate directions.
    per = 1 + len(COUNTRIES)
    dim = next(v.shape[0] for v in vecs if v is not None)
    kept_docs, blocks, dropped = [], [], 0
    for di, d in enumerate(docs):
        block = vecs[di * per:(di + 1) * per]
        if any(v is None for v in block):
            dropped += 1
            continue
        kept_docs.append(d)
        blocks.append(np.stack(block))            # (per, dim)
    if dropped:
        print(f"  dropped {dropped} document(s) with an unembeddable string; kept {len(kept_docs)}")
    docs = kept_docs
    n = len(docs)
    stacked = np.array(blocks, dtype=np.float32)  # (n, per, dim)
    base = stacked[:, 0, :]                       # (n, dim)
    inj = stacked[:, 1:, :]                        # (n, C, dim)

    # Per (doc, country) shift magnitude (cosine distance baseline -> injected).
    bn = base / np.linalg.norm(base, axis=1, keepdims=True)
    injn = inj / np.linalg.norm(inj, axis=2, keepdims=True)
    shift = 1.0 - np.einsum("nd,ncd->nc", bn, injn)   # (n, C)

    # Decompose each shift: generic "authorship line" effect (shared across countries) vs country-specific.
    delta = inj - base[:, None, :]                    # (n, C, dim)
    generic = delta.mean(axis=1, keepdims=True)       # (n, 1, dim)
    country = delta - generic                         # (n, C, dim) — the country-specific part
    frac_country = float(np.mean(
        np.linalg.norm(country, axis=2) / np.linalg.norm(delta, axis=2)
    ))

    # Per-country direction stability (exact mean pairwise cosine) + mean direction.
    consistency, mean_dir = {}, {}
    for ci, c in enumerate(COUNTRIES):
        vc = country[:, ci, :]                         # (n, dim)
        uc = vc / np.linalg.norm(vc, axis=1, keepdims=True)
        consistency[c] = mean_pairwise_cosine(uc)
        m = vc.mean(axis=0)
        mean_dir[c] = m / np.linalg.norm(m)

    cross = np.array([[float(mean_dir[a] @ mean_dir[b]) for b in COUNTRIES] for a in COUNTRIES])

    # ----- report -----
    scope = "every WP in the corpus" if all_wps else f"{n}-doc sample (seed {SAMPLE_SEED})"
    lines = [
        "DIRECT COUNTRY-AUTHORSHIP SIGNAL — EMBEDDING INJECTION PROBE",
        f"Prefix injected into naive-censored WPs: {PREFIX!r}",
        f"Documents: {n} — {scope}, >= {MIN_BODY_WORDS} body words.   Countries: {', '.join(COUNTRIES)}",
        "",
        f"1) MAGNITUDE  mean shift = {shift.mean():.4f} cosine dist "
        f"(~{shift.mean() / NATURAL_CENSORSHIP_SHIFT:.1f}x the ~{NATURAL_CENSORSHIP_SHIFT:.4f} natural raw->naive censorship shift)",
        f"   country-specific fraction of each shift = {frac_country:.3f} "
        f"(the rest is the generic 'an authorship line exists' effect shared across countries)",
        "",
        "2) STABILITY  per-country direction consistency across docs (mean pairwise cosine; 1.0 = identical direction):",
    ]
    for c in COUNTRIES:
        lines.append(f"     {c:16s} {consistency[c]:+.3f}")
    lines += [
        "",
        "3) SEPARATION  cross-country cosine between mean directions (low/negative = distinct):",
        "     " + " " * 12 + " ".join(f"{c[:6]:>7s}" for c in COUNTRIES),
    ]
    for i, a in enumerate(COUNTRIES):
        lines.append(f"     {a:12s}" + " ".join(f"{cross[i, j]:7.2f}" for j in range(len(COUNTRIES))))
    lines += [
        "",
        "Reading: a large, country-specific shift with high per-country consistency and distinct",
        "cross-country directions means the DIRECT authorship signal is linearly encoded along stable,",
        "mostly-separable axes — the object an embedding-space censor would project out. (UK/US tend to",
        "share an axis: the two large anglophone Western states.)",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    (OUTPUT_DIR / f"direct_country_signal_report{tag}.txt").write_text(report)

    shift_df = pd.DataFrame(shift, columns=[f"shift__{c}" for c in COUNTRIES])
    shift_df.insert(0, "stem", [d["stem"] for d in docs])
    shift_df.insert(1, "true_authors", ["; ".join(d["true_authors"]) for d in docs])
    shift_df.to_csv(OUTPUT_DIR / f"direct_country_signal_shifts{tag}.csv", index=False)

    np.savez(
        OUTPUT_DIR / f"direct_country_directions{tag}.npz",
        countries=np.array(COUNTRIES),
        directions=np.vstack([mean_dir[c] for c in COUNTRIES]),   # (n_countries, dim), unit vectors
        consistency=np.array([consistency[c] for c in COUNTRIES]),
        cross_cosine=cross,
        n_documents=n,
    )
    print(f"\nWrote report, per-doc shifts, and country directions to {OUTPUT_DIR}/ (tag {tag!r})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all-wps", action="store_true",
                        help="Run over every WP in the corpus (robustness), not the 10-doc sample.")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent embedding workers (default {DEFAULT_WORKERS}).")
    args = parser.parse_args()
    run(all_wps=args.all_wps, workers=args.workers)


if __name__ == "__main__":
    main()
