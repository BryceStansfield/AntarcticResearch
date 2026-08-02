"""Tests for the embedding pool's failure path.

`embed_document_set` fans out over 200 processes, so every worker exception has to survive a
pickle round-trip back to the parent. It does not: pickle rebuilds an exception by calling
`cls(*exc.args)`, and `openai.APIStatusError` -- raised for every non-2xx response, and a 200-way
pool provokes plenty of 429s -- takes required keyword-only `response` and `body`. Reconstruction
raises TypeError inside `multiprocessing.pool`'s result-handler thread, killing the only thread
that delivers results, so `starmap` blocks forever instead of reporting the API error.

These tests pin the containment: `_embed_with_retry` converts any failure into a string-only
`EmbeddingFailed`, which is what makes the parent raise rather than hang. Nothing here makes a
paid call; the embedding function is stubbed out.
"""
import multiprocessing
import pickle

import httpx
import openai
import pytest

import embeddings.embed_all_documents as ead

ARGS = ("uuid-abc", "WorkingPaper", "some document text")
_REAL_SLEEP = ead.time.sleep  # kept before any fixture stubs it out


def api_error():
    """A realistic OpenRouter rate-limit error, built the way the openai client builds it."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    return openai.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


def empty_input_error():
    """What OpenRouter returns for an empty string: a 400, identical on every retry."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    return openai.BadRequestError(
        "Error code: 400 - Too small: expected string to have >=1 characters",
        response=httpx.Response(400, request=request),
        body=None,
    )


def raiser(exc):
    def _raise(*_args):
        raise exc
    return _raise


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retries back off for seconds; the tests only care that the attempts happen."""
    monkeypatch.setattr(ead.time, "sleep", lambda _seconds: None)


def test_openai_error_does_not_survive_a_pickle_round_trip():
    """The upstream defect these tests exist for. If this ever starts passing, openai has given
    APIStatusError a reconstructable signature and the wrapper is only belt-and-braces."""
    payload = pickle.dumps(api_error())  # pickling is fine -- only args are written
    with pytest.raises(TypeError, match="response"):
        pickle.loads(payload)


def test_embedding_failed_survives_a_pickle_round_trip():
    original = ead.EmbeddingFailed("WorkingPaper uuid-abc failed")
    assert str(pickle.loads(pickle.dumps(original))) == str(original)


def test_retry_wraps_an_unpicklable_api_error(monkeypatch):
    monkeypatch.setattr(ead, "get_or_generate_embedding", raiser(api_error()))

    with pytest.raises(ead.EmbeddingFailed) as caught:
        ead._embed_with_retry(*ARGS)

    message = str(caught.value)
    assert "uuid-abc" in message and "WorkingPaper" in message  # which document failed
    assert "RateLimitError" in message and "rate limited" in message  # and why
    assert "some document text" not in message  # not the whole document


def test_retry_returns_the_embedding_once_an_attempt_succeeds(monkeypatch):
    attempts = []

    def flaky(*args):
        attempts.append(args)
        if len(attempts) < ead.MAX_ATTEMPTS:
            raise api_error()
        return [0.5, 0.25]

    monkeypatch.setattr(ead, "get_or_generate_embedding", flaky)

    assert ead._embed_with_retry(*ARGS) == [0.5, 0.25]
    assert attempts == [ARGS] * ead.MAX_ATTEMPTS


def test_pool_reports_the_failure_instead_of_hanging():
    """The end-to-end guarantee: an API error in a worker reaches the parent as an exception.

    Before the wrapper this call never returned -- the result handler died and starmap waited on a
    result that would never arrive -- so a regression here shows up as the suite hanging."""
    with multiprocessing.Pool(processes=1) as pool:
        with pytest.raises(ead.EmbeddingFailed, match="uuid-abc"):
            pool.starmap(_module_level_worker, [ARGS])


def _module_level_worker(*args):
    """Pool workers must be importable by name, so the stubbing happens inside the child."""
    ead.get_or_generate_embedding = raiser(api_error())
    ead.time.sleep = lambda _seconds: None
    return ead._embed_with_retry(*args)


# ------------------------------------------------------------------------- which errors retry

@pytest.mark.parametrize("status, retryable", [
    (429, True),    # rate limited -- the case retrying exists for
    (500, True),
    (503, True),
    (400, False),   # empty input; identical on every attempt
    (401, False),   # bad key
    (404, False),   # unknown model
])
def test_only_transient_statuses_are_retried(status, retryable):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    exc = openai.APIStatusError("boom", response=httpx.Response(status, request=request), body=None)
    assert ead._is_retryable(exc) is retryable


def test_errors_without_a_status_are_retried():
    """A dropped connection or a read timeout never reached the API, so it may well succeed next
    time. These carry no status_code at all, and must not be mistaken for a permanent verdict."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    assert ead._is_retryable(openai.APIConnectionError(request=request)) is True
    assert ead._is_retryable(openai.APITimeoutError(request=request)) is True


def test_a_400_fails_immediately_without_burning_retries(monkeypatch):
    """The empty-input 400 that took down a whole run. Retrying it three times only delays the
    report by the full backoff; the answer is the same every time."""
    attempts = []

    def always_400(*args):
        attempts.append(args)
        raise empty_input_error()

    monkeypatch.setattr(ead, "get_or_generate_embedding", always_400)

    with pytest.raises(ead.EmbeddingFailed) as caught:
        ead._embed_with_retry(*ARGS)

    assert len(attempts) == 1
    assert "failed after 1 attempt:" in str(caught.value)


# --------------------------------------------------------------------------------- worker ramp

def test_workers_start_on_an_even_ramp(monkeypatch):
    """Each worker waits one more ramp step than the last, so the pool reaches full concurrency
    gradually instead of firing POOL_PROCESSES requests in the same instant."""
    slept = []
    monkeypatch.setattr(ead.time, "sleep", slept.append)
    monkeypatch.setattr(ead, "POOL_PROCESSES", 4)
    counter = multiprocessing.Value("i", 0)

    for _ in range(4):
        ead._stagger_worker_start(counter)

    assert slept == [0, ead.WORKER_RAMP_SECONDS, 2 * ead.WORKER_RAMP_SECONDS, 3 * ead.WORKER_RAMP_SECONDS]
    assert counter.value == 4  # every worker claimed a distinct slot


def test_replacement_workers_rejoin_without_waiting(monkeypatch):
    """`Pool._maintain_pool` re-runs the initializer for every worker it replaces, against a pool
    already at full concurrency. Those get indices past POOL_PROCESSES that are never reset, so
    honouring the ramp would make each successive replacement slower to rejoin than the last."""
    slept = []
    monkeypatch.setattr(ead.time, "sleep", slept.append)
    monkeypatch.setattr(ead, "POOL_PROCESSES", 4)
    counter = multiprocessing.Value("i", 4)  # the initial four have already started

    for _ in range(3):  # three deaths, three replacements
        ead._stagger_worker_start(counter)

    assert slept == []  # no ramp, no unbounded growth
    assert counter.value == 7


_RECORD_PATH = None


def _record_embedding(document_uuid, _document_type, _text):
    """Stands in for the real embedder. Workers are separate processes, so what they did has to
    come back through the filesystem; one small O_APPEND write per task is atomic on Linux."""
    with open(_RECORD_PATH, "a") as handle:
        handle.write(document_uuid + "\n")
    return [0.5]


def test_ramp_does_not_drop_work(monkeypatch, tmp_path):
    """Staggering delays workers; it must not cost tasks. Tasks queued while a worker is still
    sleeping are picked up when it wakes, so every input is still embedded exactly once."""
    global _RECORD_PATH
    _RECORD_PATH = tmp_path / "embedded.txt"
    _RECORD_PATH.write_text("")

    # Set before the Pool forks, so the children inherit all four. The real sleep goes back for
    # this one test: the point is to exercise the stagger, not to skip it.
    monkeypatch.setattr(ead.time, "sleep", _REAL_SLEEP)
    monkeypatch.setattr(ead, "POOL_PROCESSES", 4)
    monkeypatch.setattr(ead, "WORKER_RAMP_SECONDS", 0.01)
    monkeypatch.setattr(ead, "get_or_generate_embedding", _record_embedding)

    ead.embed_document_set([(f"uuid-{i}", "WorkingPaper", f"text {i}") for i in range(40)])

    assert sorted(_RECORD_PATH.read_text().split()) == sorted(f"uuid-{i}" for i in range(40))
