import multiprocessing
import random
import time
import traceback
import pandas
from embeddings.document_embeddings import *
from downloaders.map_all_wp_ip_locations import map_all_wp_ip_file_locations
from embeddings.working_paper_censorship import (
    get_working_paper_paths, censor_text, COUNTRIES,
)

CENSORED_WORKING_PAPER_TYPE = "CensoredWorkingPaperV1"
MAX_ATTEMPTS = 3


class EmbeddingFailed(Exception):
    """A document could not be embedded after ``MAX_ATTEMPTS`` tries.

    Carries nothing but a string, and that is the point. A worker's exception is pickled back to
    the parent, and pickle rebuilds an exception by calling ``cls(*exc.args)`` -- so any exception
    class with required keyword-only arguments cannot be reconstructed. ``openai.APIStatusError``
    (raised for every non-2xx response, including the 429s a 200-process pool provokes) has
    ``args == (message,)`` but demands ``response`` and ``body``, so unpickling raises TypeError
    *inside* the pool's result-handler thread. That thread then dies, no result is ever delivered,
    and ``starmap`` blocks forever -- the run hangs instead of reporting the API error. Re-raising
    the failure as this type keeps it transportable, so the parent sees the real cause."""


def _is_retryable(exc) -> bool:
    """Whether a further attempt could plausibly succeed.

    Rate limits and 5xxs are transient, and are the reason retrying exists here. The other 4xxs are
    verdicts on the request itself -- an empty input, a bad key, an unknown model -- and are
    identical every time, so retrying only spends the backoff before the run reports what is
    actually wrong. Anything with no status at all (a dropped connection, a read timeout) never
    reached the API and is treated as transient."""
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status == 429 or status >= 500


def _embed_with_retry(*args):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return get_or_generate_embedding(*args)
        except Exception as exc:
            if attempt == MAX_ATTEMPTS - 1 or not _is_retryable(exc):
                document_uuid, document_type = args[0], args[1]
                attempts = attempt + 1
                raise EmbeddingFailed(
                    f"{document_type} {document_uuid} failed after {attempts} "
                    f"attempt{'s' if attempts > 1 else ''}:\n"
                    + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                ) from None
            # Back off before retrying. Without this the three attempts are spent within a few
            # milliseconds of each other, which is useless against the rate limiting and transient
            # 5xxs that are the whole reason for retrying, and 200 workers all retrying instantly
            # is what provokes the limit in the first place.
            time.sleep(2 ** attempt + random.random())

POOL_PROCESSES = 200
WORKER_RAMP_SECONDS = 0.05  # 200 workers -> ~10s from first request to full concurrency


def _stagger_worker_start(counter):
    """Hold each worker back a little longer than the last before it takes its first task.

    `Pool` has no pacing knob -- `starmap`'s `chunksize` batches tasks per worker but does not slow
    their issue -- so without this all `POOL_PROCESSES` workers open a connection and fire an
    embedding request in the same instant. That opening burst is what draws the 429s, and a rate
    limit hit on the first request is the worst case: every worker is in lockstep, so they retry in
    lockstep too. Staggering starts spreads the burst and desynchronises the pool for the whole run.

    The counter is a shared `Value` rather than a per-worker random sleep so the ramp is even and
    its total duration is known: worker *n* begins at `n * WORKER_RAMP_SECONDS`. Sleeping here does
    not drop work -- tasks wait on the shared queue and are picked up as each worker becomes ready."""
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    # The ramp exists to spread the opening burst, and only the first POOL_PROCESSES workers are
    # part of one. Beyond that we are being run by `Pool._maintain_pool`, which replaces workers
    # that died and re-runs the initializer for each -- against a pool that is already at full
    # concurrency, so there is nothing to spread. The index is not reset for those, so honouring
    # it would make every successive replacement rejoin more slowly than the last, without bound.
    if index < POOL_PROCESSES:
        time.sleep(index * WORKER_RAMP_SECONDS)


def embed_document_set(to_embed):
    # Inherited by fork rather than pickled; `Value` cannot cross a spawn boundary through initargs.
    counter = multiprocessing.Value("i", 0)
    with multiprocessing.Pool(
        processes=POOL_PROCESSES,
        initializer=_stagger_worker_start,
        initargs=(counter,),
    ) as pool:
        pool.starmap(_embed_with_retry, to_embed)

def embed_all_measures():
    pd = pandas.read_csv("data/MeasureCorpusEnriched.csv")

    to_embed = []
    stale = 0

    for row in pd.itertuples():
        if pandas.isna(row.Content):
            continue
        
        doc_num = row.Document_Number
        doc_id = measure_id_to_uuid(doc_num)
        text_rep = get_representation_of_measure(row)

        # Measures are the one corpus whose uuid is synthetic rather than a hash of its own text,
        # so `has_embedding` alone cannot tell "already embedded" from "embedded, then the CSV
        # changed underneath it". Every other type self-corrects: edit the text and you get a new
        # uuid and a cache miss. Here the key is stable, so an edited Subject/Content would keep
        # returning the vector of the superseded text indefinitely. Checking the recorded content
        # hash restores that property; the stale row is dropped so the insert is not ignored.
        if is_stale(doc_id, text_rep):
            stale += 1
            delete_embedding(doc_id)

        if not has_embedding(doc_id):
            to_embed.append((doc_id, "measure", text_rep,))

    if stale:
        print(f"  {stale} measure(s) changed in MeasureCorpusEnriched.csv since they were "
              f"embedded; their cached vectors were dropped and will be regenerated.")
    embed_document_set(to_embed)

def embed_all_censored_working_papers(countries=COUNTRIES):
    to_embed = []
    print("Hashing censored working papers for embedding")
    for path in get_working_paper_paths():
        censored_text = censor_text(path.read_text(encoding="utf-8", errors="ignore"), countries)
        to_embed.extend(get_wp_ip_embedding_args(censored_text, CENSORED_WORKING_PAPER_TYPE))

    print("Embedding censored working papers")
    embed_document_set(to_embed)

def embed_all():
    print("Embedding Measures")
    embed_all_measures()
    print("Done Embedding Measures")

    ip_wp_file_paths = map_all_wp_ip_file_locations()
    ip_wp_to_embed = []
    print("Hashing ips and wps for embedding")
    skipped = 0
    for path in ip_wp_file_paths.values():
        if "/wp/" in path:
            t = "WorkingPaper"
        elif "/ip/" in path:
            t = "InformationPaper"
        else:
            # Attachments under /uploads/ are neither. Without this branch `t` kept its value from
            # the previous iteration, so every one of them was silently filed under whichever type
            # happened to come before it — contaminating that type's enumeration.
            skipped += 1
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            ip_wp_to_embed.extend(get_wp_ip_embedding_args(f.read(), t))
    if skipped:
        print(f"  skipped {skipped} files under neither /wp/ nor /ip/ (attachments)")

    print("Embedding ips and wps")
    embed_document_set(ip_wp_to_embed)

    print("Embedding censored working papers")
    embed_all_censored_working_papers()


from utils import line_buffer_stdout

if __name__ == "__main__":
    line_buffer_stdout()
    embed_all()