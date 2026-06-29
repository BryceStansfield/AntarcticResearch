"""Launch a self-terminating Vast.ai GPU run to fine-tune the authorship classifiers.

You pick the GPU *machine* in the Vast web UI, then drive everything from here. The GPU
instance bootstraps itself (clones the repo, pulls the dataset from your Google Drive via
rclone, trains, pushes results back to Drive) and then destroys itself — so the run survives
your connection dropping, and the only thing that ever crosses your uplink is the (small)
dataset upload and whatever results you choose to pull later.

There is NO second "storage" instance any more: Google Drive is the persistent middle. The
GPU talks to Drive directly with rclone, which (unlike Vast's flaky inter-instance copy)
handles large files reliably, resumably, and with real progress in the log.

Typical workflow
----------------
    # 1. One-time: configure an rclone Google Drive remote on your laptop (see the project
    #    notes / rclone config guide). Name it to match CONFIG["rclone_remote"] (default
    #    "gdrive") and use the drive.file scope so the token shipped to the GPU can only touch
    #    files rclone itself creates.
    # 2. Add "VAST_API_KEY": "..." to secrets.json (VAST_API_KEY env var also works).

    # Find the GPU machine id in the web UI (the `m:NNNNN` value). One step: sync the dataset to
    # Drive, then launch the self-driving GPU run (cheapest 1x H200 offer on that machine).
    python vast_train.py run <gpu_machine_id> [--method naive]

    # whenever convenient: pull results from Drive to your laptop (incremental/resumable)
    python vast_train.py pull ./vast_results

The individual steps (`seed`, `train`) remain available if you'd rather upload the dataset to
Drive once and fire off GPU runs without re-syncing it each time.

Requirements: the Vast CLI (`pip install vastai`) and `rclone` on PATH; auth comes from
secrets.json's VAST_API_KEY (threaded into every Vast call) and from your local rclone config.
A local SSH key in ~/.ssh is injected into the GPU box for debugging. Assumes a PUBLIC GitHub
repo (LFS blobs skipped on clone).
"""
import argparse
import ast
import base64
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import secret_management

# --------------------------------------------------------------------------------- config
CONFIG = {
    # EDIT: your public repo + branch.
    "repo_url": "https://github.com/BryceStansfield/AntarcticResearch.git",
    "branch": "master",

    # Censorship method to fine-tune (overridable per invocation with --method).
    "method": "raw",

    # GPU image must ship CUDA + a CUDA build of torch (or let `uv sync` install one).
    "gpu_image": "vastai/pytorch:cuda-13.2.1-auto",
    # Disk (GB). 8B `best/` models are ~16 GB each; size for repo + dataset + the in-flight
    # checkpoints of one granularity (the training script prunes them after each granularity).
    "gpu_disk_gb": 400,

    # Local dataset dir (uploaded to Drive by `seed`/`run`, pulled by the GPU).
    "local_dataset_dir": "data/finetuning",

    # rclone / Google Drive. Configure the remote once on your laptop (`rclone config`, drive.file
    # scope recommended). Everything lives under <rclone_remote>:<drive_base>/:
    #   <base>/data/finetuning   <- the dataset      <base>/results/<run-label>/  <- results
    "rclone_remote": "gdrive",
    "drive_base": "wpauth",

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
apt-get update -y >/dev/null 2>&1 && apt-get install -y git curl unzip >/dev/null 2>&1 || true
# The image ships an OLD vastai at /opt/instance-tools/bin (first on PATH) whose `show
# instances` hits a removed v0 endpoint (HTTP 410, non-JSON). Prefer the newer pip CLI.
export PATH="/usr/local/bin:$PATH"

# Install rclone and materialise the single Drive remote shipped from the laptop (base64'd so it
# survives the --env transport). --config keeps it self-contained.
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | bash || apt-get install -y rclone || true
fi
mkdir -p /root/.config/rclone
printf '%s' "$RCLONE_CONF_B64" | base64 -d > /root/.config/rclone/rclone.conf
RC="rclone --config /root/.config/rclone/rclone.conf --retries 10 --low-level-retries 20 -v --stats=20s --stats-one-line"

SELF_ID=""
resolve_self_id() {
  # Vast injects this instance's id straight into the container env (CONTAINER_ID) — no API
  # call, CLI version, or boot-timing dependency. Fall back to the labelled var (C.<id>).
  [ -n "${SELF_ID:-}" ] && return
  SELF_ID="${CONTAINER_ID:-}"
  [ -n "${SELF_ID:-}" ] && return
  SELF_ID="$(printf '%s' "${VAST_CONTAINERLABEL:-}" | sed 's/^C\.//')"
}

# Push results to Drive. rclone is synchronous, resumable, and reliable with large files (the
# whole reason we dropped the Vast inter-instance copy), so this is just two copies. Logs go
# FIRST (so diagnostics land even if the model copy dies), then the trained models, then a final
# run.log re-push so the saved log captures the copy phase itself. We push ONLY each granularity's
# best/ dir (weights, no optimiser state) — the dataset is already on Drive and the intermediate
# checkpoints are pruned by the training script.
push_results() {
  REPO_DATA="WORKDIR_PLACEHOLDER/repo/data/finetuning"
  STAGE="WORKDIR_PLACEHOLDER/_results_stage"
  rm -rf "$STAGE"; mkdir -p "$STAGE/logs" "$STAGE/models"

  cp -f WORKDIR_PLACEHOLDER/run.log "$STAGE/logs/run.log" 2>/dev/null || true
  find "$REPO_DATA" -maxdepth 1 -name 'test_report_*.txt' -exec cp -f {} "$STAGE/logs/" \; 2>/dev/null || true
  echo "pushing logs -> $DRIVE_RESULTS/logs"
  $RC copy "$STAGE/logs" "$DRIVE_RESULTS/logs" --transfers 4 || true

  found=0
  while IFS= read -r best; do
    md="$(dirname "$(dirname "$best")")"          # .../{gran}/{method}  (best is in .../checkpoints/best)
    tag="$(basename "$(dirname "$md")")__$(basename "$md")"   # e.g. full__raw
    cp -al "$best" "$STAGE/models/$tag" 2>/dev/null || cp -r "$best" "$STAGE/models/$tag"
    found=1
  done < <(find "$REPO_DATA" -type d -name best)
  if [ "$found" = 1 ]; then
    echo "pushing models -> $DRIVE_RESULTS/models"
    $RC copy "$STAGE/models" "$DRIVE_RESULTS/models" --transfers 4 || true
  else
    echo "no best/ model dirs found to push (run likely died before saving any)"
  fi

  cp -f WORKDIR_PLACEHOLDER/run.log "$STAGE/logs/run.log" 2>/dev/null || true
  $RC copy "$STAGE/logs/run.log" "$DRIVE_RESULTS/logs" || true   # capture the copy phase in the log
}

cleanup() {
  set +e   # best-effort: never let a failed step abort teardown
  echo "=== cleanup $(date -u) ==="
  resolve_self_id   # last-ditch attempt if the bootstrap failed before the id was cached
  push_results
  if [ -n "${SELF_ID:-}" ]; then
    echo "destroying self $SELF_ID"
    # -y is essential: `destroy instance` prompts [y/N] and otherwise aborts (NOT destroyed).
    vastai destroy instance "$SELF_ID" -y --api-key "$VAST_API_KEY"
  else
    echo "WARNING: could not resolve self id; destroy manually (web UI / label $RUN_LABEL)"
  fi
}
# Arm teardown as early as possible — right after the tooling it needs exists, and before any
# of the failure-prone clone / pull / train work below.
trap cleanup EXIT

resolve_self_id
echo "self id: ${SELF_ID:-<unknown>}"

# Hard runtime cap: kill the script (-> trap -> teardown) if training hangs.
( sleep "$(( MAX_RUNTIME_HOURS * 3600 ))"; echo "watchdog: runtime cap hit"; kill -TERM $$ ) &

set -e
cd WORKDIR_PLACEHOLDER
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$BRANCH" "$REPO_URL" repo
cd repo

# Pull the prepared dataset from Drive. rclone copy is synchronous + resumable, so no byte-polling
# dance — it simply blocks until the dataset is present, then we sanity-check it's non-empty.
echo "pulling dataset from Drive: $DRIVE_DATASET"
mkdir -p data/finetuning
$RC copy "$DRIVE_DATASET" data/finetuning --transfers 8
n_files=$(find data/finetuning -type f | wc -l)
echo "dataset files present: $n_files"
[ "$n_files" -gt 0 ] || { echo "ERROR: no dataset files pulled from Drive ($DRIVE_DATASET)"; exit 1; }

# Resolve the environment and train (fr_finetuned_model loops all granularities and writes
# data/finetuning/test_report_<method>.txt). WPAUTH_METHOD selects the censorship method.
uv sync
uv run python -m working_paper_authorship.fr_finetuned_model

echo "=== training done $(date -u) ==="
# trap EXIT handles result push + self-destroy.
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
    """Copy of cmd with secrets masked, for safe echoing (the Vast key and the rclone config
    blob both ride along after --api-key and inside the --env string)."""
    out, mask_next = [], False
    for tok in cmd:
        if mask_next:
            out.append("****"); mask_next = False
        elif tok == "--api-key":
            out.append(tok); mask_next = True
        else:
            tok = re.sub(r"(VAST_API_KEY=)\S+", r"\1****", tok)
            tok = re.sub(r"(RCLONE_CONF_B64=)\S+", r"\1****", tok)
            out.append(tok)
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
    """Run a command, echoing it (secrets redacted). Returns stdout (when captured); exits on failure."""
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


# ---------------------------------------------------------------------- rclone / Google Drive

def _rclone_check() -> None:
    """Fail early with a friendly message if rclone isn't installed locally."""
    try:
        subprocess.run(["rclone", "version"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        sys.exit("rclone not found on PATH. Install it and run `rclone config` to set up the "
                 f"'{CONFIG['rclone_remote']}' Google Drive remote (see the setup guide).")


def _drive(*parts: str) -> str:
    """An rclone path under our base folder, e.g. _drive('results', label)."""
    return f"{CONFIG['rclone_remote']}:" + "/".join([CONFIG["drive_base"], *parts])


def _rclone_remote_conf(remote: str) -> str:
    """Just the rclone.conf [section] for one remote, so we ship only that remote's token to the
    GPU (not every remote you have configured)."""
    _rclone_check()
    out = subprocess.run(["rclone", "config", "file"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL).stdout
    m = re.search(r"(\S*rclone\.conf)", out)
    conf_path = Path(m.group(1)) if m else Path.home() / ".config" / "rclone" / "rclone.conf"
    if not conf_path.exists():
        sys.exit(f"rclone config not found at {conf_path}; run `rclone config` first.")
    text = conf_path.read_text()
    section = re.search(rf"(?ms)^\[{re.escape(remote)}\]\s*$\n(.*?)(?=^\[|\Z)", text)
    if not section:
        sys.exit(f"no [{remote}] remote in {conf_path}. Run `rclone config` to create it, or set "
                 f"CONFIG['rclone_remote'] to one you have.")
    return f"[{remote}]\n{section.group(1).strip()}\n"


def _seed_dataset_to_drive() -> None:
    """Upload the local dataset to Drive (incremental: rclone skips unchanged files)."""
    _rclone_check()
    src = CONFIG["local_dataset_dir"].rstrip("/")
    if not Path(src).is_dir():
        sys.exit(f"local dataset dir not found: {src} (run prepare_data_for_finetuning first)")
    dst = _drive("data", "finetuning")
    print(f"syncing {src} -> {dst}")
    _run(["rclone", "copy", src, dst, "-P", "--transfers", "8"], capture=False)


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


# --------------------------------------------------------------------------------- cores

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


def _ssh_inject_block() -> str:
    """Bash that bakes this machine's public key into root's authorized_keys with
    sshd-compatible perms (home dir + .ssh root-owned, not group/other-writable — this image
    leaves /root too permissive). Lets us ssh into the GPU box to watch/debug a live run."""
    q = shlex.quote(Path(_default_pubkey()).read_text().strip())
    return (
        "mkdir -p /root/.ssh\n"
        f"grep -qxF {q} /root/.ssh/authorized_keys 2>/dev/null || echo {q} >> /root/.ssh/authorized_keys\n"
        "chown root:root /root /root/.ssh /root/.ssh/authorized_keys\n"
        "chmod 755 /root\n"
        "chmod 700 /root/.ssh\n"
        "chmod 600 /root/.ssh/authorized_keys"
    )


def _launch_gpu(gpu_offer_id: str, method: str) -> str | None:
    api_key = _api_key()
    label = _run_label()
    onstart_path = Path("/tmp") / f"{label}.onstart.sh"
    onstart = ONSTART.replace("WORKDIR_PLACEHOLDER", WORKDIR)
    onstart = onstart.replace("SSH_KEY_INJECT_MARKER", _ssh_inject_block())
    onstart_path.write_text(onstart)
    conf_b64 = base64.b64encode(_rclone_remote_conf(CONFIG["rclone_remote"]).encode()).decode()
    env_pairs = {
        "VAST_API_KEY": api_key,
        "RUN_LABEL": label,
        "REPO_URL": CONFIG["repo_url"],
        "BRANCH": CONFIG["branch"],
        "WPAUTH_METHOD": method,
        "MAX_RUNTIME_HOURS": str(CONFIG["max_runtime_hours"]),
        "RCLONE_CONF_B64": conf_b64,
        "DRIVE_DATASET": _drive("data", "finetuning"),
        "DRIVE_RESULTS": _drive("results", label),
    }
    iid = _create_instance(gpu_offer_id, CONFIG["gpu_image"], CONFIG["gpu_disk_gb"],
                           label, env_pairs, str(onstart_path))
    print(f"\nGPU run launched: label={label}, id={iid or '?'}, method={method}.")
    print("It is now self-driving (clone -> pull dataset -> train -> push results -> "
          "self-destroy). You can safely disconnect.")
    print(f"Monitor:  vastai logs {iid or '<id>'}")
    print(f"Results will land at: {_drive('results', label)}")
    return iid


# ------------------------------------------------------------------------------ subcommands

def run(args) -> None:
    """Sync the dataset to Drive, then launch the self-driving GPU run. The GPU self-destroys
    when finished (or on failure / the runtime-cap watchdog), so there's nothing left billing."""
    _api_key()
    _rclone_check()
    method = args.method or CONFIG["method"]

    print("=== 1/2 syncing dataset to Drive ===")
    _seed_dataset_to_drive()

    print("=== 2/2 launching GPU run ===")
    gpu_offer = _pick_gpu_offer(args.gpu_machine_id)
    _launch_gpu(gpu_offer, method)
    print(f"\nDone. Pull results later with: python vast_train.py pull ./vast_results")


def seed(args) -> None:
    _seed_dataset_to_drive()
    print("\nDataset synced to Drive. Re-run any time to update (incremental).")


def train(args) -> None:
    """Launch a GPU run assuming the dataset is already on Drive (skips the re-sync)."""
    _api_key()
    _rclone_check()
    gpu_offer = _pick_gpu_offer(args.gpu_machine_id)
    _launch_gpu(gpu_offer, args.method or CONFIG["method"])
    print("Pull results with: python vast_train.py pull ./vast_results")


def pull(args) -> None:
    _rclone_check()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = _drive("results")
    print(f"pulling {src} -> {dest}/")
    _run(["rclone", "copy", src, str(dest), "-P", "--transfers", "8"], capture=False)
    print(f"\nResults pulled to {dest}/ (incremental — rerun to fetch new runs).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="sync dataset to Drive + launch the self-driving GPU run")
    p.add_argument("gpu_machine_id", help="machine id for the GPU box (cheapest 1x H200 offer is picked)")
    p.add_argument("--method", choices=["raw", "naive", "llm_censorship"], default=None)
    p.set_defaults(func=run)

    p = sub.add_parser("seed", help="upload the local dataset to Google Drive (incremental)")
    p.set_defaults(func=seed)

    p = sub.add_parser("train", help="launch a GPU run (dataset assumed already on Drive)")
    p.add_argument("gpu_machine_id")
    p.add_argument("--method", choices=["raw", "naive", "llm_censorship"], default=None)
    p.set_defaults(func=train)

    p = sub.add_parser("pull", help="download results from Drive to a local dir (incremental)")
    p.add_argument("dest")
    p.set_defaults(func=pull)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
