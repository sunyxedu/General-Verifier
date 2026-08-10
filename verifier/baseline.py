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
    return {"git": git, "file_count": len(manifest)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m verifier.baseline ROOT OUT_DIR")
    info = snapshot(sys.argv[1], sys.argv[2])
    print(f"baseline: {info['file_count']} files, "
          f"git={'yes' if info['git'] else 'no'}")
