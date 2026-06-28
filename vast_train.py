"""Launch a self-terminating Vast.ai GPU run to fine-tune the authorship classifiers.

You pick the GPU *offer* in the Vast web UI, then drive everything from here. The GPU
instance bootstraps itself (clones the repo, pulls the dataset from a cheap persistent
storage instance, trains, copies results back to storage) and then destroys itself — so the
run survives your connection dropping, and big files only ever move host-to-host over Vast's
backbone, never across your slow uplink.

Typical workflow
----------------
    # add "VAST_API_KEY": "..." to secrets.json (from the Vast console; injected into the
    # GPU instance so it can self-destruct). VAST_API_KEY env var also works as a fallback.

    # Find the GPU machine id in the web UI (the `m:NNNNN` value it shows). The GPU box picks the
    # cheapest single-H200 (NVL or not) offer on that machine; the storage box is auto-picked
    # globally (cheapest offer with enough disk and inet < storage_max_tb_cost $/TB up & down).
    # Then in one step it rents storage, seeds the dataset, and launches the self-driving GPU run;
    # if any setup step fails, every instance created so far is destroyed before aborting.
    python vast_train.py run <gpu_machine_id> [--method naive]

    # whenever convenient: pull results from storage to your laptop (resumable)
    python vast_train.py pull <storage_instance_id> ./vast_results

The individual steps (`storage-up`, `seed`, `train`) remain available if you'd rather seed a
persistent storage box once and reuse it across many GPU runs (avoids re-seeding over 4G).

Requirements: the Vast CLI (`pip install vastai`) on PATH (auth comes from secrets.json's
VAST_API_KEY, threaded into every call — no `vastai set api-key` needed), plus `ssh`, `rsync`,
and a local SSH key in ~/.ssh (its public half is attached to the storage instance per-run, so
no account-level Vast key registration is needed — handy given SSH keys are personal-context
while billing is org). Assumes a PUBLIC GitHub repo (LFS blobs skipped on clone).
"""
import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import secret_management

# --------------------------------------------------------------------------------- config
CONFIG = {
    # EDIT: your public repo + branch.
    "repo_url": "https://github.com/BryceStansfield/AntarcticResearch.git",
    "branch": "master",

    # Censorship method to fine-tune (overridable per `train` invocation with --method).
    "method": "raw",

    # Images. GPU image must ship CUDA + a CUDA build of torch (or let `uv sync` install one).
    "gpu_image": "vastai/pytorch:cuda-13.2.1-auto",
    "storage_image": "vastai/base-image:@vastai-automatic-tag",
    # Disk (GB). 8B checkpoints are ~16 GB each; size for repo + dataset + the checkpoints you keep.
    "gpu_disk_gb": 400,
    "storage_disk_gb": 400,
    # Storage box is auto-picked: cheapest offer with enough disk, inet up+down each under
    # this $/TB cap (so the results download/upload stays cheap), and at least this much inet
    # bandwidth up+down (a weak filter against boxes on bad networks / flaky inter-instance copy).
    "storage_max_tb_cost": 5.0,
    "storage_min_inet_mbps": 500,

    # Paths on the storage instance, and the matching local dataset dir.
    "remote_dataset_dir": "/workspace/data/finetuning",
    "remote_results_dir": "/workspace/results",
    "local_dataset_dir": "data/finetuning",

    # Watchdog: the GPU instance hard-destroys itself after this many hours no matter what.
    "max_runtime_hours": 12,
}

WORKDIR = "/workspace"

# Bash bootstrap run on GPU boot. Everything run-specific arrives via env vars (passed with
# --env), so this stays a constant string with no fragile Python interpolation. Quoted
# heredocs keep inline Python from being mangled by the shell.
ONSTART = r"""#!/bin/bash
set -uo pipefail
mkdir -p WORKDIR_PLACEHOLDER
exec > WORKDIR_PLACEHOLDER/run.log 2>&1
echo "=== onstart $(date -u) label=$RUN_LABEL ==="

# Inject our SSH key first thing, so the GPU box is reachable for debugging during setup.
SSH_KEY_INJECT_MARKER

pip install -q --no-input vastai uv 2>/dev/null || true
apt-get update -y >/dev/null 2>&1 && apt-get install -y git rsync curl >/dev/null 2>&1 || true
# The image ships an OLD vastai at /opt/instance-tools/bin (first on PATH) whose `show
# instances` hits a removed v0 endpoint (HTTP 410, non-JSON). Prefer the newer pip CLI.
export PATH="/usr/local/bin:$PATH"

SELF_ID=""
resolve_self_id() {
  # Vast injects this instance's id straight into the container env (CONTAINER_ID) — no API
  # call, CLI version, or boot-timing dependency. Fall back to the labelled var (C.<id>).
  [ -n "${SELF_ID:-}" ] && return
  SELF_ID="${CONTAINER_ID:-}"
  [ -n "${SELF_ID:-}" ] && return
  SELF_ID="$(printf '%s' "${VAST_CONTAINERLABEL:-}" | sed 's/^C\.//')"
}

status_msg() {  # this instance's status_msg; tolerant of a deprecation-warning prefix on --raw
  vastai show instances --raw --api-key "$VAST_API_KEY" 2>/dev/null | python3 -c "
import sys, re, json
s = sys.stdin.read()
m = re.search(r'\[.*\]', s, re.DOTALL)
try:
    d = json.loads(m.group(0)) if m else []
    print(next((i.get('status_msg', '') for i in d if str(i.get('id')) == '$SELF_ID'), ''))
except Exception:
    print('')
" 2>/dev/null || true
}

# Push results to storage and WAIT for the copy to actually finish before the caller destroys
# us. The copy is async; on the SENDER side status_msg cycles "Copying container data..." ->
# "<bytes> <pct> <rate>" -> "Done copying" (empirically the reliable completion signal — the
# receiver side just sticks on "Receiving copy..."). We poll for "Done copying", and retry the
# whole copy if it never reports done (the async copy sometimes silently no-ops). run.log is
# staged into the results dir so the single copy captures it too.
copy_results() {
  [ -z "${SELF_ID:-}" ] && { echo "no self id; cannot copy results"; return; }
  cp -f WORKDIR_PLACEHOLDER/run.log WORKDIR_PLACEHOLDER/repo/data/finetuning/run.log 2>/dev/null || true
  for attempt in 1 2 3; do
    echo "results copy attempt $attempt -> storage..."
    vastai copy "$SELF_ID:WORKDIR_PLACEHOLDER/repo/data/finetuning" \
                "$STORAGE_ID:$REMOTE_RESULTS_DIR/$RUN_LABEL" --api-key "$VAST_API_KEY" || true
    sleep 15  # let the fresh copy take over status_msg (don't match a stale 'Done copying')
    for _ in $(seq 1 90); do   # up to ~15 min per attempt
      msg="$(status_msg)"
      echo "  status_msg: $msg"
      case "$msg" in *"Done copying"*) echo "results copy complete"; return;; esac
      sleep 10
    done
    echo "  no 'Done copying' after attempt $attempt; retrying"
  done
  echo "WARNING: results copy never confirmed 'Done copying' — results may be incomplete on storage"
}

cleanup() {
  set +e   # best-effort: never let a failed step abort teardown
  echo "=== cleanup $(date -u) ==="
  resolve_self_id   # last-ditch attempt if the bootstrap failed before the id was cached
  copy_results
  if [ -n "${SELF_ID:-}" ]; then
    echo "destroying self $SELF_ID"
    # -y is essential: `destroy instance` prompts [y/N] and otherwise aborts (NOT destroyed).
    vastai destroy instance "$SELF_ID" -y --api-key "$VAST_API_KEY"
  else
    echo "WARNING: could not resolve self id; destroy manually (web UI / label $RUN_LABEL)"
  fi
}
# Arm teardown as early as possible — right after the tooling it needs exists, and before any
# of the failure-prone clone / copy / train work below.
trap cleanup EXIT

resolve_self_id
echo "self id: ${SELF_ID:-<unknown>}"

# Hard runtime cap: kill the script (-> trap -> teardown) if training hangs.
( sleep "$(( MAX_RUNTIME_HOURS * 3600 ))"; echo "watchdog: runtime cap hit"; kill -TERM $$ ) &

set -e
cd WORKDIR_PLACEHOLDER
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$BRANCH" "$REPO_URL" repo
cd repo

# Pull the prepared dataset from storage. The remote->remote copy is ASYNC ("initiated") and,
# fired right at boot, sometimes silently delivers nothing (the instance pairing isn't ready
# yet). So drive it from the wait-loop: (re)trigger the copy whenever no bytes have arrived for
# a few checks, and only proceed once the GPU's copy reaches EXPECTED_BYTES (the laptop-computed
# size of what was seeded; file content only, so the figures match across filesystems). This
# both blocks training until the data is fully present and self-heals a no-op copy.
# mkdir the leaf (not just data/) so the find below never fails on a missing dir -- with
# `set -o pipefail` + `set -e`, `cur=$(find data/finetuning ... | awk ...)` would abort the
# whole script. Copying into the parent `data` still merges into this empty dir (no nesting).
mkdir -p data/finetuning
echo "pulling dataset from storage ($EXPECTED_BYTES bytes expected)..."
cur=0; last=-1; stalls=3   # start "stalled" so the first iteration triggers the copy
for _ in $(seq 1 180); do  # up to ~30 min
  cur=$(find data/finetuning -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'); cur=${cur:-0}
  [ "$cur" -ge "$EXPECTED_BYTES" ] && { echo "dataset complete: ${cur}/${EXPECTED_BYTES} bytes"; break; }
  if [ "$cur" = "$last" ]; then stalls=$((stalls + 1)); else stalls=0; fi
  if [ "$stalls" -ge 3 ]; then
    resolve_self_id   # a fresh instance often isn't in `show instances` at boot; re-resolve now
    if [ -n "${SELF_ID:-}" ]; then
      echo "  no progress at ${cur} bytes; (re)triggering copy (self=$SELF_ID)..."
      # NB1: empty SELF_ID drops the "id:" prefix -> Vast errors "Destination instance must have
      # an open port" and 0 bytes land, so we only trigger once it's resolved.
      # NB2: copy the dir into the PARENT (repo/data), not repo/data/finetuning -- `vastai copy`
      # places the source dir *inside* the dest (like `cp -r`), so dest=.../data lands it as
      # data/finetuning/..., not the doubly-nested data/finetuning/finetuning/....
      vastai copy "$STORAGE_ID:$REMOTE_DATASET_DIR" "$SELF_ID:WORKDIR_PLACEHOLDER/repo/data" \
                  --api-key "$VAST_API_KEY" || true
    else
      echo "  self id still unresolved; will retry"
    fi
    stalls=0
  fi
  echo "  ...${cur}/${EXPECTED_BYTES} bytes"
  last=$cur
  sleep 10
done
[ "${cur:-0}" -ge "$EXPECTED_BYTES" ] || { echo "ERROR: dataset incomplete (${cur}/${EXPECTED_BYTES} bytes)"; exit 1; }

# Resolve the environment and train (fr_finetuned_model loops all granularities and writes
# data/finetuning/test_report_<method>.txt). WPAUTH_METHOD selects the censorship method.
uv sync
uv run python -m working_paper_authorship.fr_finetuned_model

echo "=== training done $(date -u) ==="
# trap EXIT handles result copy + self-destroy.
"""


# --------------------------------------------------------------------------------- helpers

def _api_key() -> str:
    """Vast API key from secrets.json (preferred) or the VAST_API_KEY env var as a fallback.
    Used for our CLI calls and injected into the GPU instance so it can self-destruct."""
    try:
        key = secret_management.get("VAST_API_KEY").strip()
    except (KeyError, FileNotFoundError):
        key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        sys.exit('Add "VAST_API_KEY" to secrets.json (or set the VAST_API_KEY env var).')
    return key


def _vastai(*args: str) -> list[str]:
    """A vastai CLI invocation with our secrets-sourced key appended, so secrets.json is the
    single source of auth (no separate `vastai set api-key` needed)."""
    return ["vastai", *args, "--api-key", _api_key()]


def _redacted(cmd: list[str]) -> list[str]:
    """Copy of cmd with the API key masked, for safe echoing (it appears both after
    --api-key and inside the --env string)."""
    out, mask_next = [], False
    for tok in cmd:
        if mask_next:
            out.append("****"); mask_next = False
        elif tok == "--api-key":
            out.append(tok); mask_next = True
        else:
            out.append(re.sub(r"(VAST_API_KEY=)\S+", r"\1****", tok))
    return out


def _extract_obj(out: str):
    """Parse a vastai command's stdout into a dict/list, tolerating both clean JSON and the
    Python dict-repr some commands print (e.g. create's `Started. {'new_contract': 123}`,
    with single quotes and True/False). Returns None if nothing parseable is found."""
    out = out.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        pass
    m = re.search(r"[\[{].*[\]}]", out, re.DOTALL)
    if m:
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(m.group(0))
            except (ValueError, SyntaxError):
                continue
    return None


def _run(cmd: list[str], capture: bool = True) -> str:
    """Run a command, echoing it (API key redacted). Returns stdout (when captured); exits on failure."""
    print("+ " + " ".join(_redacted(cmd)))
    # stdin closed so an unexpected prompt (e.g. an ssh password fallback) can't block forever.
    result = subprocess.run(cmd, capture_output=capture, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        if capture:
            sys.stderr.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        sys.exit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout if capture else ""


def _run_label() -> str:
    return "wpauth-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _create_instance(offer_id: str, image: str, disk_gb: int, label: str,
                     env_pairs: dict | None = None, onstart_path: str | None = None) -> str | None:
    """Create an instance from an offer; returns the new instance id if parseable."""
    # No --raw: for `create instance` that returns the raw Response object, not JSON. The
    # human output is `Started. {'success': True, 'new_contract': <id>}` (a Python dict-repr),
    # which _extract_obj parses; new_contract is the new instance id.
    parts = ["create", "instance", str(offer_id), "--image", image,
             "--disk", str(disk_gb), "--label", label, "--ssh", "--direct"]
    if env_pairs:
        parts += ["--env", " ".join(f"-e {k}={v}" for k, v in env_pairs.items())]
    if onstart_path:
        parts += ["--onstart", onstart_path]
    out = _run(_vastai(*parts))
    obj = _extract_obj(out)
    if isinstance(obj, dict):
        new_id = obj.get("new_contract") or obj.get("id")
        if new_id:
            return str(new_id)
    print(out)  # show what came back so a parse miss is diagnosable
    return None


def _ssh_conn(instance_id: str) -> tuple[str, str, str]:
    """(user, host, port) for an instance, from `vastai ssh-url`."""
    url = _run(_vastai("ssh-url", str(instance_id))).strip()
    m = re.search(r"ssh://(?P<user>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)", url)
    if not m:
        sys.exit(f"could not parse ssh url: {url!r}")
    return m.group("user"), m.group("host"), m.group("port")


def _wait_running(instance_id: str, timeout_s: int = 900) -> None:
    """Poll until the instance reports running and SSH answers."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = _run(_vastai("show", "instances", "--raw"))
        data = _extract_obj(out)
        inst = next((i for i in data if isinstance(i, dict) and str(i.get("id")) == str(instance_id)), None) if isinstance(data, list) else None
        status = inst.get("actual_status") if inst else None
        print(f"  status: {status}")
        if status == "running":
            user, host, port = _ssh_conn(instance_id)
            try:
                probe = subprocess.run(
                    ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no",
                     "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{user}@{host}", "true"],
                    capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=20,
                )
                if probe.returncode == 0:
                    return
            except subprocess.TimeoutExpired:
                pass
        time.sleep(15)
    sys.exit(f"instance {instance_id} did not become reachable within {timeout_s}s")


def _rsync(src: str, dst: str, port: str) -> None:
    """Resumable rsync (survives 4G drops: --partial keeps half-sent files, --append-verify
    resumes them, and rerunning the command picks up where it stopped)."""
    _run([
        "rsync", "-avP", "--partial", "--append-verify",
        "-e", f"ssh -p {port} -o StrictHostKeyChecking=no -o BatchMode=yes",
        src, dst,
    ], capture=False)


# ---------------------------------------------------------------------- offer selection

def _norm_machine_id(machine_id: str) -> str:
    """Accept the UI's ``m:37586`` form as well as a bare ``37586``."""
    return str(machine_id).removeprefix("m:")


def _search_offers(machine_id: str) -> list[dict]:
    """Rentable offers on one machine, via ``vastai search offers``."""
    out = _run(_vastai("search", "offers", f"machine_id={_norm_machine_id(machine_id)}", "--raw"))
    try:
        return [o for o in json.loads(out) if o.get("rentable")]
    except json.JSONDecodeError:
        return []


def _describe_offer(o: dict) -> str:
    return (f"offer {o.get('id')}: {o.get('num_gpus')}x {o.get('gpu_name')} "
            f"${o.get('dph_total', 0):.3f}/hr disk={o.get('disk_space', 0):.0f}GB")


def _pick_gpu_offer(machine_id: str) -> str:
    """Cheapest rentable single-H200 offer (NVL or not) on the machine with enough disk."""
    offers = [o for o in _search_offers(machine_id)
              if o.get("num_gpus") == 1
              and "H200" in (o.get("gpu_name") or "")
              and (o.get("disk_space") or 0) >= CONFIG["gpu_disk_gb"]]
    if not offers:
        sys.exit(f"no rentable 1x H200 offer with >= {CONFIG['gpu_disk_gb']} GB disk on machine {machine_id}")
    best = min(offers, key=lambda o: o.get("dph_total", float("inf")))
    print(f"  selected GPU {_describe_offer(best)}")
    return str(best["id"])


def _pick_cheapest_storage_offer() -> str:
    """Cheapest rentable offer *anywhere* with enough disk and inet up AND down each under
    CONFIG['storage_max_tb_cost'] $/TB — keeps the eventual results download/upload cheap.
    (`inet_*_cost` is $/GB in the offer JSON, so $/TB = that * 1000.)"""
    gb_cap = CONFIG["storage_max_tb_cost"] / 1000.0  # $/TB -> $/GB
    disk = CONFIG["storage_disk_gb"]
    out = _run(_vastai("search", "offers",
                       f"rentable=true disk_space>{disk - 1} dph_total<2", "--raw"))
    try:
        offers = json.loads(out)
    except json.JSONDecodeError:
        offers = []
    min_mbps = CONFIG["storage_min_inet_mbps"]
    matches = [
        o for o in offers
        if o.get("rentable")
        and (o.get("disk_space") or 0) >= disk
        and (o["inet_up_cost"] if o.get("inet_up_cost") is not None else 1e9) < gb_cap
        and (o["inet_down_cost"] if o.get("inet_down_cost") is not None else 1e9) < gb_cap
        and (o.get("inet_up") or 0) > min_mbps
        and (o.get("inet_down") or 0) > min_mbps
    ]
    if not matches:
        sys.exit(f"no rentable offer with >= {disk} GB disk, inet "
                 f"< ${CONFIG['storage_max_tb_cost']}/TB and > {min_mbps} Mbps up & down")
    best = min(matches, key=lambda o: o.get("dph_total", float("inf")))
    print(f"  selected storage {_describe_offer(best)} "
          f"inet ${(best.get('inet_up_cost') or 0) * 1000:.2f}/${(best.get('inet_down_cost') or 0) * 1000:.2f} per TB, "
          f"{best.get('inet_up', 0):.0f}/{best.get('inet_down', 0):.0f} Mbps up/down")
    return str(best["id"])


# --------------------------------------------------------------------------------- cores

def _destroy(instance_id: str) -> None:
    """Best-effort destroy (never raises or hangs). The -y is essential: `vastai destroy
    instance` prompts [y/N] and otherwise aborts (with stdin closed it reads empty -> N -> the
    instance is NOT destroyed)."""
    try:
        r = subprocess.run(_vastai("destroy", "instance", str(instance_id), "-y"),
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        ok = "destroying" in (r.stdout + r.stderr).lower()
        print(f"  {'destroyed' if ok else 'destroy may have failed for'} instance {instance_id}"
              + ("" if ok else f": {r.stdout.strip()} {r.stderr.strip()}"))
    except subprocess.TimeoutExpired:
        print(f"  WARNING: destroy of {instance_id} timed out — verify in the web UI")


def _default_pubkey() -> str:
    """Path to this machine's default SSH public key (ed25519 preferred, then rsa/ecdsa,
    then any *.pub)."""
    ssh = Path.home() / ".ssh"
    for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
        if (ssh / name).exists():
            return str(ssh / name)
    pubs = sorted(ssh.glob("*.pub"))
    if pubs:
        return str(pubs[0])
    sys.exit("no SSH public key in ~/.ssh (generate one with `ssh-keygen`)")


def _attach_ssh_key(instance_id: str) -> None:
    """Attach this machine's default public key to an instance so we can ssh/rsync into it.
    Done per-instance so it works regardless of the SSH-keys-are-personal-only / billing-is-org
    account-context split (no account-level key registration needed). Best-effort backup to the
    onstart key injection — attaching to an already-running instance can lag, so don't abort on it."""
    try:
        subprocess.run(_vastai("attach", "ssh", str(instance_id), _default_pubkey()),
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        pass


def _ssh_inject_block() -> str:
    """Bash that bakes this machine's public key into root's authorized_keys with
    sshd-compatible perms (home dir + .ssh root-owned, not group/other-writable — this image
    leaves /root too permissive). Shared by the storage onstart and the GPU onstart so *both*
    instances are ssh-reachable, which makes debugging the GPU run far easier."""
    q = shlex.quote(Path(_default_pubkey()).read_text().strip())
    return (
        "mkdir -p /root/.ssh\n"
        f"grep -qxF {q} /root/.ssh/authorized_keys 2>/dev/null || echo {q} >> /root/.ssh/authorized_keys\n"
        "chown root:root /root /root/.ssh /root/.ssh/authorized_keys\n"
        "chmod 755 /root\n"
        "chmod 700 /root/.ssh\n"
        "chmod 600 /root/.ssh/authorized_keys"
    )


def _storage_onstart_path() -> str:
    """Onstart for the storage box: inject the SSH key so SSH works as soon as sshd is up
    (no dependency on attach-key propagation, which lags on a running instance)."""
    script = "#!/bin/bash\n" + _ssh_inject_block() + "\n"
    path = Path("/tmp") / "wpauth-storage.onstart.sh"
    path.write_text(script)
    return str(path)


def _dataset_expected_bytes() -> int:
    """Total content bytes of all files under the local dataset dir. The GPU waits until its
    async-copied dataset reaches exactly this before training — a precise completion signal.
    Counts file sizes only (not directory inodes) so the laptop and GPU figures agree across
    filesystems."""
    root = Path(CONFIG["local_dataset_dir"])
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _seed_dataset(storage_id: str) -> None:
    print(f"Waiting for storage instance {storage_id}...")
    _wait_running(storage_id)
    user, host, port = _ssh_conn(storage_id)
    remote = CONFIG["remote_dataset_dir"]
    # Pre-create both the dataset dir and the results dir: the GPU's results push (vastai copy)
    # can't mkdir -p an intermediate path, so REMOTE_RESULTS_DIR must already exist on storage.
    _run(["ssh", "-p", port, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
          f"{user}@{host}", f"mkdir -p {remote} {CONFIG['remote_results_dir']}"])
    local = CONFIG["local_dataset_dir"].rstrip("/") + "/"
    _rsync(local, f"{user}@{host}:{remote}/", port)


def _launch_gpu(gpu_offer_id: str, storage_id: str, method: str) -> str | None:
    api_key = _api_key()
    label = _run_label()
    onstart_path = Path("/tmp") / f"{label}.onstart.sh"
    onstart = ONSTART.replace("WORKDIR_PLACEHOLDER", WORKDIR)
    onstart = onstart.replace("SSH_KEY_INJECT_MARKER", _ssh_inject_block())
    onstart_path.write_text(onstart)
    env_pairs = {
        "VAST_API_KEY": api_key,
        "RUN_LABEL": label,
        "STORAGE_ID": str(storage_id),
        "REPO_URL": CONFIG["repo_url"],
        "BRANCH": CONFIG["branch"],
        "WPAUTH_METHOD": method,
        "REMOTE_DATASET_DIR": CONFIG["remote_dataset_dir"],
        "REMOTE_RESULTS_DIR": CONFIG["remote_results_dir"],
        "MAX_RUNTIME_HOURS": str(CONFIG["max_runtime_hours"]),
        "EXPECTED_BYTES": str(_dataset_expected_bytes()),
    }
    iid = _create_instance(gpu_offer_id, CONFIG["gpu_image"], CONFIG["gpu_disk_gb"],
                           label, env_pairs, str(onstart_path))
    print(f"\nGPU run launched: label={label}, id={iid or '?'}, method={method}.")
    print("It is now self-driving (clone -> pull dataset -> train -> push results -> "
          "self-destroy). You can safely disconnect.")
    print(f"Monitor:  vastai logs {iid or '<id>'}    (or watch results/{label}/run.log on storage)")
    return iid


# ------------------------------------------------------------------------------ subcommands

def run(args) -> None:
    """Rent storage -> seed dataset -> launch the GPU run, as one step. If anything before
    the GPU is self-driving fails (renting, waiting, seeding), every instance created so far
    is destroyed and the command aborts, so a half-built run never sits there billing."""
    _api_key()
    method = args.method or CONFIG["method"]
    created: list[str] = []
    try:
        print("=== 1/3 renting storage ===")
        storage_offer = _pick_cheapest_storage_offer()
        storage_id = _create_instance(storage_offer, CONFIG["storage_image"],
                                      CONFIG["storage_disk_gb"], "wpauth-storage",
                                      onstart_path=_storage_onstart_path())
        if not storage_id:
            raise RuntimeError("could not determine storage instance id from create output")
        created.append(storage_id)
        _attach_ssh_key(storage_id)  # best-effort backup to the onstart key injection

        print("=== 2/3 seeding dataset ===")
        _seed_dataset(storage_id)

        print("=== 3/3 launching GPU run ===")
        gpu_offer = _pick_gpu_offer(args.gpu_machine_id)
        gpu_id = _launch_gpu(gpu_offer, storage_id, method)
        if gpu_id:
            created.append(gpu_id)
    except BaseException as exc:  # incl. SystemExit from _run() and KeyboardInterrupt
        print(f"\n!! workflow failed ({exc}); terminating {len(created)} instance(s)...")
        for iid in created:
            _destroy(iid)
        raise
    print(f"\nDone. Storage instance {storage_id} persists (holds results); the GPU run "
          f"self-destroys when finished.\nPull results later with: "
          f"python vast_train.py pull {storage_id} ./vast_results")


def storage_up(args) -> None:
    _api_key()
    label = "wpauth-storage"
    offer = _pick_cheapest_storage_offer()
    iid = _create_instance(offer, CONFIG["storage_image"], CONFIG["storage_disk_gb"], label,
                           onstart_path=_storage_onstart_path())
    if iid:
        _attach_ssh_key(iid)
    print(f"\nStorage instance requested (label={label}, id={iid or '?'}).")
    print("Wait for it to come up (`vastai show instances`), then run `seed <id>`.")


def seed(args) -> None:
    _api_key()
    _seed_dataset(args.storage_id)
    print("\nDataset seeded to storage. Re-run this any time to sync changes (resumable).")


def train(args) -> None:
    _api_key()
    gpu_offer = _pick_gpu_offer(args.gpu_machine_id)
    _launch_gpu(gpu_offer, args.storage_id, args.method or CONFIG["method"])
    print(f"Pull results with: python vast_train.py pull {args.storage_id} ./vast_results")


def pull(args) -> None:
    _api_key()
    _wait_running(args.storage_id)
    user, host, port = _ssh_conn(args.storage_id)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    _rsync(f"{user}@{host}:{CONFIG['remote_results_dir']}/", str(dest).rstrip("/") + "/", port)
    print(f"\nResults pulled to {dest}/ (resumable — rerun to fetch new runs).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="rent storage (auto-picked) + seed dataset + launch GPU run in one "
                                    "step (destroys everything created if any setup step fails)")
    p.add_argument("gpu_machine_id", help="machine id for the GPU box (cheapest 1x H200 offer is picked)")
    p.add_argument("--method", choices=["raw", "naive", "llm_censorship"], default=None)
    p.set_defaults(func=run)

    p = sub.add_parser("storage-up", help="create the storage instance (cheapest offer with enough disk "
                                          "and inet < storage_max_tb_cost $/TB, auto-picked)")
    p.set_defaults(func=storage_up)

    p = sub.add_parser("seed", help="rsync the local dataset to the storage instance (resumable)")
    p.add_argument("storage_id")
    p.set_defaults(func=seed)

    p = sub.add_parser("train", help="rent a 1x H200 on a machine id and fire off a self-terminating run")
    p.add_argument("gpu_machine_id")
    p.add_argument("storage_id")
    p.add_argument("--method", choices=["raw", "naive", "llm_censorship"], default=None)
    p.set_defaults(func=train)

    p = sub.add_parser("pull", help="rsync results from storage to a local dir (resumable)")
    p.add_argument("storage_id")
    p.add_argument("dest")
    p.set_defaults(func=pull)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
