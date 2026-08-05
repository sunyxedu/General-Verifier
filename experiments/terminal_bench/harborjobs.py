"""Tolerant readers for Harbor job output directories.

Harbor writes jobs/<name>/trials/<task>__<id>/{result.json, config.json,
logs/agent/*.txt, artifacts/...}. Field layouts have drifted across
versions, so extraction is defensive: rewards are found by recursive key
search, never by a hardcoded schema path.
"""
import json
import os


def _find_key(obj, key):
    """First value for `key` anywhere in a nested dict/list, else None."""
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def trials(job_dir):
    """Yield dicts: task, trial_dir, reward (float|None), agent_log path.

    Layout (harbor 0.20): jobs/<name>/<task>__<id>/ holds result.json,
    config.json, agent/<agent>.txt, verifier/reward.txt, artifacts/.
    """
    if not os.path.isdir(job_dir):
        return
    for name in sorted(os.listdir(job_dir)):
        trial_dir = os.path.join(job_dir, name)
        if "__" not in name or not os.path.isdir(trial_dir) \
                or not os.path.exists(os.path.join(trial_dir, "result.json")):
            continue
        task = name.split("__")[0]
        reward = None
        try:
            with open(os.path.join(trial_dir, "result.json")) as f:
                reward = _find_key(json.load(f), "reward")
            if reward is not None:
                reward = float(reward)
        except (json.JSONDecodeError, TypeError, ValueError):
            reward = None
        if reward is None:  # fallback: verifier/reward.txt
            rpath = os.path.join(trial_dir, "verifier", "reward.txt")
            if os.path.exists(rpath):
                try:
                    reward = float(open(rpath).read().strip())
                except ValueError:
                    pass
        log = None
        agent_logs = os.path.join(trial_dir, "agent")
        if os.path.isdir(agent_logs):
            txts = [os.path.join(agent_logs, f)
                    for f in sorted(os.listdir(agent_logs))
                    if f.endswith(".txt")]
            log = txts[0] if txts else None
        yield {"task": task, "trial": name, "trial_dir": trial_dir,
               "reward": reward, "agent_log": log}


def instruction_of(trial_dir):
    """Best-effort task instruction from the trial's config.json."""
    path = os.path.join(trial_dir, "config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return _find_key(json.load(f), "instruction")
    except (json.JSONDecodeError, TypeError):
        return None


def log_tail(path, chars=6000):
    """Human-readable tail of a stream-json agent log."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # prefer assistant text blocks when the log is stream-json
    texts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") if isinstance(ev, dict) else None
        content = (msg or {}).get("content")
        if ev.get("type") == "assistant" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text") or "")
        elif ev.get("type") == "result" and ev.get("result"):
            texts.append(str(ev["result"]))
    joined = "\n".join(t for t in texts if t.strip()) or raw
    return joined[-chars:]
