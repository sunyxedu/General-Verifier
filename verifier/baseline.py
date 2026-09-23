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
  repositories.json
                 every repository found under the root (relative path ->
                 the same three fields), so a repository one directory
                 down is as visible as one at the root
  manifest.json  relative path -> sha256 (files above the size cap record
                 "size:<bytes>" instead of a hash)

CLI (used inside task containers): python -m verifier.baseline ROOT OUT
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

SKIP_DIRS = {".git", ".gv", "__pycache__", "node_modules", ".venv"}
HASH_CAP = 64 * 1024 * 1024   # hash files up to 64 MB; record size beyond


# git refuses to read a repository owned by another OS user; the baseline
# (and the verifier's instruments) must read the task's repositories
# whoever owns them
GIT_ANY_OWNER = ["-c", "safe.directory=*"]
MAX_REPOSITORIES = 32


def _git_state(path: str) -> dict:
    git = {}
    for name, args in (("head", ["rev-parse", "HEAD"]),
                       ("refs", ["for-each-ref"]),
                       ("status", ["status", "--porcelain"])):
        try:
            proc = subprocess.run(["git", *GIT_ANY_OWNER, *args], cwd=path,
                                  capture_output=True, text=True)
            if proc.returncode == 0:
                git[name] = proc.stdout.strip()
        except OSError:
            pass
    return git


def repositories(root: str) -> dict:
    """Every repository under root (its own worktree included), by path
    relative to root ('.' for root itself). A checkout that is not at the
    task root was invisible to the baseline, and every check of 'the
    repository' guessed the root and found none."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames or ".git" in filenames:
            rel = os.path.relpath(dirpath, root)
            state = _git_state(dirpath)
            if state:
                found[rel] = state
            if len(found) >= MAX_REPOSITORIES:
                break
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    return found


def snapshot(root: str, out_dir: str) -> dict:
    root = os.path.abspath(root)
    os.makedirs(out_dir, exist_ok=True)

    git = _git_state(root)
    with open(os.path.join(out_dir, "git.json"), "w") as f:
        json.dump(git, f, indent=1)
    repos = repositories(root)
    with open(os.path.join(out_dir, "repositories.json"), "w") as f:
        json.dump(repos, f, indent=1)

    manifest = tree_manifest(root)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"root": root, "files": manifest}, f, indent=1)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(environment_fingerprint(), f, indent=1)
    return {"git": git, "repositories": sorted(repos),
            "file_count": len(manifest)}


def tree_manifest(root: str) -> dict:
    """relative path -> sha256 (or size:/link: marker) for every file under
    root, skipping VCS internals and vendored trees."""
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
    return manifest


def manifest_delta(before: dict, after: dict, cap: int = 20) -> dict:
    """What changed between two tree manifests: added, removed, modified
    relative paths (each list bounded to cap entries, with counts)."""
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    modified = sorted(k for k in before if k in after and before[k] != after[k])
    out = {}
    for name, items in (("added", added), ("removed", removed),
                        ("modified", modified)):
        if items:
            out[name] = items[:cap]
            out[name + "_count"] = len(items)
    return out


def delivered_root(state_dir) -> str:
    """The delivered world's root as the pre-task baseline recorded it
    ("" when no baseline exists or the root is gone)."""
    if not state_dir:
        return ""
    path = os.path.join(state_dir, "baseline", "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            root = json.load(f).get("root") or ""
    except (OSError, ValueError):
        return ""
    return root if root and os.path.isdir(root) else ""


def environment_fingerprint() -> dict:
    """The environment as the task supplies it, independent of any task:
    the executables reachable on PATH and the packages each Python on
    PATH can import. A deliverable is accepted as delivered, so whatever
    the solver adds here for its own session is not part of the delivery;
    the delta against this record tells the seats what was added."""
    # the task's PATH, not ours: this runs under the verifier's own venv,
    # which must not appear in a record of what the task supplied
    try:
        from .llm.provider import without_verifier_venv
        task_path = without_verifier_venv(dict(os.environ)).get("PATH", "")
    except Exception:  # noqa: BLE001 - the record degrades, never fails
        task_path = os.environ.get("PATH", "")
    seen = set()
    executables = set()
    for d in task_path.split(os.pathsep):
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
    env = {**os.environ, "PATH": task_path}
    for interp in ("python", "python3"):
        if interp not in executables:
            continue
        try:
            proc = subprocess.run(
                [shutil.which(interp, path=task_path) or interp,
                 "-c", "import json,sys;"
                 "from importlib import metadata;"
                 "print(json.dumps({'executable': sys.executable,"
                 "'version': sys.version.split()[0],"
                 "'packages': sorted({d.metadata['Name'].lower() "
                 "for d in metadata.distributions() if d.metadata['Name']})}))"],
                capture_output=True, text=True, timeout=60, env=env)
            if proc.returncode == 0:
                pythons[interp] = json.loads(proc.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    # the invocation environment a check must reproduce: the task's PATH
    # verbatim and where its interpreters resolve on it
    interpreters = {name: shutil.which(name, path=task_path)
                    for name in ("python", "python3", "node", "bash", "sh")
                    if shutil.which(name, path=task_path)}
    return {"executables": sorted(executables), "pythons": pythons,
            "path": task_path, "interpreters": interpreters}


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
