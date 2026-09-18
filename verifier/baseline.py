"""Pre-task baseline snapshot: the harness's trusted record of the world
before the solver touches it.

Written into <state-dir>/baseline/ by the mounting harness (never by the
solver). The obligation compiler reads it to anchor the minimal-change
invariant of completion claims in concrete, mechanically checkable facts:
"commit <sha> recorded as pre-task HEAD still resolves", "files the task
does not name still match their baseline sha256".

Contents:
  git.json       rev-parse HEAD / for-each-ref / status --porcelain
                 (absent keys mean the root is not a git worktree)
  manifest.json  relative path -> sha256 (files above the size cap record
                 "size:<bytes>" instead of a hash)

CLI (used inside task containers): python -m verifier.baseline ROOT OUT
"""
import hashlib
import json
import os
import subprocess
import sys

SKIP_DIRS = {".git", ".gv", "__pycache__", "node_modules", ".venv"}
HASH_CAP = 64 * 1024 * 1024   # hash files up to 64 MB; record size beyond


def snapshot(root: str, out_dir: str) -> dict:
    root = os.path.abspath(root)
    os.makedirs(out_dir, exist_ok=True)

    git = {}
    for name, cmd in (("head", ["git", "rev-parse", "HEAD"]),
                      ("refs", ["git", "for-each-ref"]),
                      ("status", ["git", "status", "--porcelain"])):
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True,
                                  text=True)
            if proc.returncode == 0:
                git[name] = proc.stdout.strip()
        except OSError:
            pass
    with open(os.path.join(out_dir, "git.json"), "w") as f:
        json.dump(git, f, indent=1)

    manifest = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            try:
                if os.path.islink(path):
                    manifest[rel] = "link:" + os.readlink(path)
                elif (size := os.path.getsize(path)) > HASH_CAP:
                    manifest[rel] = f"size:{size}"
                else:
                    with open(path, "rb") as fh:
                        manifest[rel] = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                continue
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"root": root, "files": manifest}, f, indent=1)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(environment_fingerprint(), f, indent=1)
    return {"git": git, "file_count": len(manifest)}


def environment_fingerprint() -> dict:
    """The environment as the task supplies it, independent of any task:
    the executables reachable on PATH and the packages each Python on
    PATH can import. A deliverable is accepted as delivered, so whatever
    the solver adds here for its own session is not part of the delivery;
    the delta against this record tells the seats what was added."""
    seen = set()
    executables = set()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        try:
            for name in os.listdir(d):
                path = os.path.join(d, name)
                if os.access(path, os.X_OK) and not os.path.isdir(path):
                    executables.add(name)
        except OSError:
            continue
    pythons = {}
    for interp in ("python", "python3"):
        if interp not in executables:
            continue
        try:
            proc = subprocess.run(
                [interp, "-c", "import json,sys;"
                 "from importlib import metadata;"
                 "print(json.dumps({'executable': sys.executable,"
                 "'version': sys.version.split()[0],"
                 "'packages': sorted({d.metadata['Name'].lower() "
                 "for d in metadata.distributions() if d.metadata['Name']})}))"],
                capture_output=True, text=True, timeout=60)
            if proc.returncode == 0:
                pythons[interp] = json.loads(proc.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return {"executables": sorted(executables), "pythons": pythons}


def environment_delta(baseline_dir: str) -> dict:
    """What the environment now has beyond (or short of) the baseline
    record: executables and Python packages added or removed since the
    task started. Empty when nothing changed or no record exists."""
    path = os.path.join(baseline_dir, "environment.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            before = json.load(f)
    except (OSError, ValueError):
        return {}
    now = environment_fingerprint()
    delta = {}
    b_exe, n_exe = set(before.get("executables", [])), set(now["executables"])
    if n_exe - b_exe:
        delta["executables_added"] = sorted(n_exe - b_exe)
    if b_exe - n_exe:
        delta["executables_removed"] = sorted(b_exe - n_exe)
    for interp, info in now.get("pythons", {}).items():
        was = (before.get("pythons") or {}).get(interp) or {}
        added = sorted(set(info.get("packages", []))
                       - set(was.get("packages", [])))
        if added:
            delta.setdefault("python_packages_added", {})[interp] = added
        if was.get("executable") and was["executable"] != info.get("executable"):
            delta.setdefault("python_interpreter_changed", {})[interp] = {
                "before": was["executable"], "now": info.get("executable")}
    return delta


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m verifier.baseline ROOT OUT_DIR")
    info = snapshot(sys.argv[1], sys.argv[2])
    print(f"baseline: {info['file_count']} files, "
          f"git={'yes' if info['git'] else 'no'}")
