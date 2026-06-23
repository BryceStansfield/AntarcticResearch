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

    # Pick a storage (CPU) offer and a GPU offer in the web UI, then in one step rent storage,
    # seed the dataset, and launch the self-driving GPU run. If any setup step fails, every
    # instance created so far is destroyed before aborting.
    python vast_train.py run <storage_offer_id> <gpu_offer_id> [--method naive]

    # whenever convenient: pull results from storage to your laptop (resumable)
    python vast_train.py pull <storage_instance_id> ./vast_results

The individual steps (`storage-up`, `seed`, `train`) remain available if you'd rather seed a
persistent storage box once and reuse it across many GPU runs (avoids re-seeding over 4G).

Requirements: the Vast CLI (`pip install vastai`) authenticated (`vastai set api-key ...`),
plus `ssh` and `rsync` on PATH. Assumes a PUBLIC GitHub repo (LFS blobs skipped on clone).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import secret_management

# --------------------------------------------------------------------------------- config
CONFIG = {
    # EDIT: your public repo + branch.
    "repo_url": "https://github.com/your-org/AntarcticResearch.git",
    "branch": "master",

    # Censorship method to fine-tune (overridable per `train` invocation with --method).
    "method": "raw",

    # Images. GPU image must ship CUDA + a CUDA build of torch (or let `uv sync` install one).
    # VERIFY: pick an image that matches the host's driver; vast's pytorch images are a safe bet.
    "gpu_image": "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel",
    "storage_image": "vastai/base-image:latest",
    # Disk (GB). 8B checkpoints are ~16 GB each; size for repo + dataset + the checkpoints you keep.
    "gpu_disk_gb": 400,
    "storage_disk_gb": 400,

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

pip install -q --no-input vastai uv 2>/dev/null || true
apt-get update -y >/dev/null 2>&1 && apt-get install -y git rsync curl >/dev/null 2>&1 || true

# Discover our own instance id by the unique launch label. A function (not a one-shot) so
# cleanup can re-attempt it even if the bootstrap died before the id was first cached.
cat > /tmp/selfid.py <<'PYEOF'
import os, json, subprocess
out = subprocess.check_output(["vastai", "show", "instances", "--raw",
                               "--api-key", os.environ["VAST_API_KEY"]])
data = json.loads(out)
label = os.environ["RUN_LABEL"]
print(next((str(i["id"]) for i in data if i.get("label") == label), ""))
PYEOF
SELF_ID=""
resolve_self_id() {
  [ -n "${SELF_ID:-}" ] && return
  SELF_ID="$(python3 /tmp/selfid.py 2>/dev/null || true)"
}

copy_results() {
  [ -z "${SELF_ID:-}" ] && { echo "no self id; cannot copy results"; return; }
  vastai copy "$SELF_ID:WORKDIR_PLACEHOLDER/repo/data/finetuning" \
              "$STORAGE_ID:$REMOTE_RESULTS_DIR/$RUN_LABEL" --api-key "$VAST_API_KEY" || true
  vastai copy "$SELF_ID:WORKDIR_PLACEHOLDER/run.log" \
              "$STORAGE_ID:$REMOTE_RESULTS_DIR/$RUN_LABEL/run.log" --api-key "$VAST_API_KEY" || true
}

cleanup() {
  echo "=== cleanup $(date -u) ==="
  resolve_self_id   # last-ditch attempt if the bootstrap failed before the id was cached
  copy_results
  if [ -n "${SELF_ID:-}" ]; then
    echo "destroying self $SELF_ID"
    vastai destroy instance "$SELF_ID" --api-key "$VAST_API_KEY" || true
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

# Pull the prepared dataset from storage (host-to-host, fast).
mkdir -p data/finetuning
vastai copy "$STORAGE_ID:$REMOTE_DATASET_DIR" "$SELF_ID:WORKDIR_PLACEHOLDER/repo/data/finetuning" \
            --api-key "$VAST_API_KEY"

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


def _run(cmd: list[str], capture: bool = True) -> str:
    """Run a command, echoing it. Returns stdout (when captured); exits on failure."""
    print("+ " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=capture, text=True)
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
    cmd = ["vastai", "create", "instance", str(offer_id), "--image", image,
           "--disk", str(disk_gb), "--label", label, "--ssh", "--direct", "--raw"]
    if env_pairs:
        cmd += ["--env", " ".join(f"-e {k}={v}" for k, v in env_pairs.items())]
    if onstart_path:
        cmd += ["--onstart", onstart_path]
    out = _run(cmd)
    try:  # VERIFY: create's --raw payload key for the new id (new_contract / id).
        data = json.loads(out)
        return str(data.get("new_contract") or data.get("id") or "") or None
    except json.JSONDecodeError:
        print(out)
        return None


def _ssh_conn(instance_id: str) -> tuple[str, str, str]:
    """(user, host, port) for an instance, from `vastai ssh-url`."""
    url = _run(["vastai", "ssh-url", str(instance_id)]).strip()
    m = re.search(r"ssh://(?P<user>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)", url)
    if not m:
        sys.exit(f"could not parse ssh url: {url!r}")
    return m.group("user"), m.group("host"), m.group("port")


def _wait_running(instance_id: str, timeout_s: int = 900) -> None:
    """Poll until the instance reports running and SSH answers."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = _run(["vastai", "show", "instance", str(instance_id), "--raw"])
        try:
            status = json.loads(out).get("actual_status")
        except json.JSONDecodeError:
            status = None
        print(f"  status: {status}")
        if status == "running":
            user, host, port = _ssh_conn(instance_id)
            probe = subprocess.run(
                ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=10", f"{user}@{host}", "true"],
                capture_output=True, text=True,
            )
            if probe.returncode == 0:
                return
        time.sleep(15)
    sys.exit(f"instance {instance_id} did not become reachable within {timeout_s}s")


def _rsync(src: str, dst: str, port: str) -> None:
    """Resumable rsync (survives 4G drops: --partial keeps half-sent files, --append-verify
    resumes them, and rerunning the command picks up where it stopped)."""
    _run([
        "rsync", "-avP", "--partial", "--append-verify",
        "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
        src, dst,
    ], capture=False)


# --------------------------------------------------------------------------------- cores

def _destroy(instance_id: str) -> None:
    """Best-effort destroy (never raises — used in failure cleanup)."""
    subprocess.run(["vastai", "destroy", "instance", str(instance_id)], capture_output=True, text=True)
    print(f"  destroyed instance {instance_id}")


def _seed_dataset(storage_id: str) -> None:
    print(f"Waiting for storage instance {storage_id}...")
    _wait_running(storage_id)
    user, host, port = _ssh_conn(storage_id)
    remote = CONFIG["remote_dataset_dir"]
    _run(["ssh", "-p", port, "-o", "StrictHostKeyChecking=no",
          f"{user}@{host}", f"mkdir -p {remote}"])
    local = CONFIG["local_dataset_dir"].rstrip("/") + "/"
    _rsync(local, f"{user}@{host}:{remote}/", port)


def _launch_gpu(gpu_offer_id: str, storage_id: str, method: str) -> str | None:
    api_key = _api_key()
    label = _run_label()
    onstart_path = Path("/tmp") / f"{label}.onstart.sh"
    onstart_path.write_text(ONSTART.replace("WORKDIR_PLACEHOLDER", WORKDIR))
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
        storage_id = _create_instance(args.storage_offer_id, CONFIG["storage_image"],
                                      CONFIG["storage_disk_gb"], "wpauth-storage")
        if not storage_id:
            raise RuntimeError("could not determine storage instance id from create output")
        created.append(storage_id)

        print("=== 2/3 seeding dataset ===")
        _seed_dataset(storage_id)

        print("=== 3/3 launching GPU run ===")
        gpu_id = _launch_gpu(args.gpu_offer_id, storage_id, method)
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
    iid = _create_instance(args.offer_id, CONFIG["storage_image"], CONFIG["storage_disk_gb"], label)
    print(f"\nStorage instance requested (label={label}, id={iid or '?'}).")
    print("Wait for it to come up (`vastai show instances`), then run `seed <id>`.")


def seed(args) -> None:
    _api_key()
    _seed_dataset(args.storage_id)
    print("\nDataset seeded to storage. Re-run this any time to sync changes (resumable).")


def train(args) -> None:
    _api_key()
    _launch_gpu(args.gpu_offer_id, args.storage_id, args.method or CONFIG["method"])
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

    p = sub.add_parser("run", help="rent storage + seed dataset + launch GPU run in one step "
                                    "(destroys everything created if any setup step fails)")
    p.add_argument("storage_offer_id")
    p.add_argument("gpu_offer_id")
    p.add_argument("--method", choices=["raw", "naive", "llm_censorship"], default=None)
    p.set_defaults(func=run)

    p = sub.add_parser("storage-up", help="create the persistent storage instance from a CPU offer")
    p.add_argument("offer_id")
    p.set_defaults(func=storage_up)

    p = sub.add_parser("seed", help="rsync the local dataset to the storage instance (resumable)")
    p.add_argument("storage_id")
    p.set_defaults(func=seed)

    p = sub.add_parser("train", help="rent a GPU offer and fire off a self-terminating run")
    p.add_argument("gpu_offer_id")
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
