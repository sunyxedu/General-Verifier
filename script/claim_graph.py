#!/usr/bin/env python3
"""Render the claim graph of ANY run, for inspecting what was extracted.

The sibling make_claim_graph.py is a hand-laid figure of one canonical
run — beautiful, and unusable on a run it was not built for. This one is
the working tool: point it at a state dir and it lays the graph out with
graphviz, showing each claim's FULL text, its obligations and their
statuses, and (for effect claims) which ledger event they pin. That is
what you need to judge whether an extraction is faithful — glossed text
hides exactly the vagueness worth catching.

    python3 script/claim_graph.py --state-dir path/to/.gv [-o out.svg] [--png]
                                  [--details]

Reads claims.sqlite (+ events.jsonl for labels) and writes SVG via `dot`.
Effect claims are collapsed into one ledger strip unless a claim depends
on them (--all-effects keeps every one).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap

STATUS_STYLE = {          # (border, fill)
    "verified": ("#2E8B57", "#E8F4ED"),
    "refuted": ("#C0483B", "#FBEDEB"),
    "partial": ("#B07407", "#FBF1DF"),
    "stale": ("#7C838F", "#F3F4F6"),
    "unverified": ("#7C838F", "#FFFFFF"),
}
KIND_SHAPE = {"final": "box3d", "assertion": "box", "effect": "note"}
WRAP = 52
MAX_LINES = 8


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def wrap(text: str, width: int = WRAP, max_lines: int = MAX_LINES) -> list:
    lines = textwrap.wrap(" ".join(str(text).split()), width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [lines[max_lines - 1][:width - 2] + "…"]
    return lines


def obligation_id(text: str, required_basis: str) -> str:
    """Mirror verifier.core.model.Obligation.id without importing the package."""
    value = json.dumps(
        {"text": text, "required": required_basis},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clip_middle(value: str, limit: int = 82) -> str:
    """Keep both the command's entry point and its usually-useful target."""
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    left = (limit - 3) * 2 // 3
    right = limit - 3 - left
    return f"{value[:left]} … {value[-right:]}"


def js_string_field(source: str, field: str) -> str | None:
    """Extract one JSON-style string literal from a recorded JS tool call."""
    match = re.search(
        rf'["\']?{re.escape(field)}["\']?\s*:\s*'
        r'("(?:\\.|[^"\\])*")',
        source,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def redact_summary(value: str) -> str:
    """Avoid putting obvious credentials into a graph someone may share."""
    value = re.sub(
        r'(?i)\b(api[_-]?key|password|secret|token)\s*=\s*[^\s;,]+',
        r'\1=<redacted>',
        value,
    )
    return re.sub(r'(?i)\bBearer\s+[^\s;,]+', 'Bearer <redacted>', value)


def tool_description(call: dict | None, result: dict | None) -> dict | None:
    """Produce a compact, provenance-preserving description of one tool pair."""
    if call is None:
        return None
    payload = call.get("payload", {})
    source = str(payload.get("input") or "")
    nested = re.search(r"\btools\.([A-Za-z0-9_]+)\s*\(", source)
    tool = nested.group(1) if nested else str(payload.get("tool") or "tool")

    if tool == "exec_command":
        description = js_string_field(source, "cmd") or source.splitlines()[0]
    elif tool == "view_image":
        path = js_string_field(source, "path")
        detail = js_string_field(source, "detail")
        description = "inspect " + (path or "image")
        if detail:
            description += f" ({detail})"
    else:
        description = source.splitlines()[0] if source else "recorded invocation"
    description = clip_middle(redact_summary(description))

    result_summary = None
    if result is not None:
        rp = result.get("payload", {})
        chars = rp.get("response_chars")
        truncated = rp.get("excerpt_truncated")
        parts = []
        if isinstance(chars, int):
            parts.append(f"{chars:,} chars")
        if truncated is True:
            parts.append("truncated")
        elif truncated is False:
            parts.append("complete")
        if parts:
            result_summary = "result: " + " · ".join(parts)
    return {"tool": tool, "call": description, "result": result_summary}


def load(state_dir: str) -> dict:
    db = sqlite3.connect(os.path.join(state_dir, "claims.sqlite"))
    claim_columns = {
        row[1] for row in db.execute("pragma table_info(claims)")
    }
    basis_expr = "basis" if "basis" in claim_columns else "NULL"
    claims = {
        cid: {"id": cid, "kind": k, "status": s, "reason": r, "text": t,
              "revises": rev, "basis": basis}
        for cid, k, s, r, t, rev, basis in db.execute(
            "select id, kind, status, reason, text, revises, "
            f"{basis_expr} from claims")
    }
    deps = list(db.execute("select child, parent from deps"))
    obligations = {}
    for claim, obl in db.execute("select claim, obligation from obligation_edges"):
        obligations.setdefault(claim, []).append(obl)
    evidence = {}
    if db.execute(
        "select 1 from sqlite_master where type='table' and name='evidence'"
    ).fetchone():
        for cid, raw in db.execute("select claim_id, payload from evidence"):
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            evidence.setdefault(cid, []).append(payload)
    db.close()

    labels, certificates, required_basis = {}, [], {}
    events_by_seq, tool_calls, tool_results = {}, {}, {}
    events_path = os.path.join(state_dir, "events.jsonl")
    if os.path.exists(events_path):
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                payload = event["payload"]
                seq = event.get("seq")
                if seq is not None:
                    events_by_seq[seq] = event
                tool_use_id = payload.get("tool_use_id")
                if event["type"] == "tool_call" and tool_use_id:
                    tool_calls[tool_use_id] = event
                elif event["type"] == "tool_result" and tool_use_id:
                    tool_results[tool_use_id] = event
                if event["type"] == "commit":
                    for cid, c in claims.items():
                        if c["text"] == payload["text"] \
                                and c["kind"] == payload["kind"]:
                            labels[cid] = payload["label"]
                elif event["type"] == "certificate":
                    certificates.append(payload)
                elif event["type"] == "obligations_compiled":
                    for obligation in payload.get("obligations", []):
                        text = obligation.get("text")
                        required = obligation.get("required_basis", "record")
                        if text:
                            required_basis[obligation_id(text, required)] = required
    # A discharge also records the floor, including for older event streams
    # that did not include it in obligations_compiled.
    for oid, payloads in evidence.items():
        for payload in payloads:
            required = payload.get("required_basis")
            if required:
                required_basis[oid] = required
    effect_details = {}
    for cid, claim in claims.items():
        if claim["kind"] != "effect":
            continue
        try:
            effect = json.loads(claim["text"])
        except (json.JSONDecodeError, TypeError):
            continue
        pinned = events_by_seq.get(effect.get("event_seq"))
        if pinned is None:
            continue
        tool_use_id = pinned.get("payload", {}).get("tool_use_id")
        call = (pinned if pinned.get("type") == "tool_call"
                else tool_calls.get(tool_use_id))
        result = (pinned if pinned.get("type") == "tool_result"
                  else tool_results.get(tool_use_id))
        description = tool_description(call, result)
        if description:
            effect_details[cid] = description
    return {"claims": claims, "deps": deps, "obligations": obligations,
            "labels": labels, "certificates": certificates,
            "evidence": evidence, "required_basis": required_basis,
            "effect_details": effect_details}


def claim_label(g, cid, details: bool) -> str:
    """HTML-like node label: header line, full text, obligation rows."""
    c = g["claims"][cid]
    border, _fill = STATUS_STYLE.get(c["status"], STATUS_STYLE["unverified"])
    name = g["labels"].get(cid, cid[:6])
    basis = (f' · basis {esc(c["basis"])}' if c.get("basis") else "")
    head = (f'<tr><td align="left"><b>{esc(name)}</b>'
            f'<font point-size="9" color="#8B92A1">  {esc(cid[:6])} · '
            f'{esc(c["kind"])}{basis}</font></td>'
            f'<td align="right"><font color="{border}"><b>'
            f'{esc(c["status"])}</b></font></td></tr>')
    body = "".join(
        f'<tr><td colspan="2" align="left">{esc(line)}</td></tr>'
        for line in wrap(c["text"]))
    rows = [head, '<tr><td colspan="2"><font point-size="2"> </font></td></tr>',
            body]
    if c["reason"]:
        rows.append(f'<tr><td colspan="2" align="left">'
                    f'<font color="#C0483B" point-size="9">why: '
                    f'{esc(" ".join(wrap(c["reason"], 60, 3)))}</font></td></tr>')
    kids = g["obligations"].get(cid, [])
    if kids:
        rows.append('<tr><td colspan="2"><font point-size="3"> </font></td>'
                    '</tr>')
        for oid in kids:
            o = g["claims"].get(oid)
            if o is None:
                continue
            ob, _f = STATUS_STYLE.get(o["status"], STATUS_STYLE["unverified"])
            text = " ".join(wrap(o["text"], 46, 3))
            required = g["required_basis"].get(oid)
            achieved = o.get("basis")
            floor = ""
            if details and (required or achieved):
                floor = (f'<br/><font point-size="8" color="#8B92A1">'
                         f'req {esc(required or "?")} · got '
                         f'{esc(achieved or "—")}</font>')
            rows.append(
                f'<tr><td align="left"><font point-size="9" color="{ob}">'
                f'● </font><font point-size="9">{esc(text)}</font></td>'
                f'<td align="right"><font point-size="8" color="{ob}">'
                f'{esc(o["status"])}</font>{floor}</td></tr>')
            latest = (g["evidence"].get(oid) or [None])[-1]
            if details and latest:
                outcome = latest.get("outcome") or "?"
                instrument = latest.get("instrument") or "unknown instrument"
                evidence_text = " ".join(
                    wrap(f'evidence: {outcome} · {instrument}', 62, 2))
                rows.append(
                    f'<tr><td colspan="2" align="left"><font point-size="8" '
                    f'color="#606978">   ↳ {esc(evidence_text)}</font>'
                    f'</td></tr>')
                witness = latest.get("witness")
                if witness:
                    witness_text = " ".join(
                        wrap(f'witness: {witness}', 62, 2))
                    rows.append(
                        f'<tr><td colspan="2" align="left">'
                        f'<font point-size="8" color="#606978">'
                        f'     {esc(witness_text)}</font></td></tr>')
            if o["reason"] and o["status"] == "refuted":
                rows.append(
                    f'<tr><td colspan="2" align="left"><font point-size="8" '
                    f'color="#C0483B">   ↳ {esc(" ".join(wrap(o["reason"], 54, 2)))}'
                    f'</font></td></tr>')
    return ('<<table border="0" cellborder="0" cellspacing="1" cellpadding="2">'
            + "".join(rows) + "</table>>")


def effect_label(g, cid, details: bool) -> str:
    c = g["claims"][cid]
    description = g["effect_details"].get(cid) if details else None
    try:
        payload = json.loads(c["text"])
        tool = (description or {}).get("tool") or payload.get("tool") or "—"
        head = f"#{payload.get('event_seq')} {tool}"
    except (json.JSONDecodeError, TypeError):
        head = cid[:6]
    rows = [f'<tr><td align="left"><b>{esc(head)}</b></td></tr>']
    if description:
        for line in wrap(description["call"], 38, 2):
            rows.append(
                f'<tr><td align="left"><font point-size="8" color="#606978">'
                f'{esc(line)}</font></td></tr>')
        if description["result"]:
            rows.append(
                f'<tr><td align="left"><font point-size="8" color="#8B92A1">'
                f'{esc(description["result"])}</font></td></tr>')
    rows.append(
        f'<tr><td align="left"><font point-size="8" color="#8B92A1">'
        f'{esc(cid[:6])}</font></td></tr>')
    return ('<<table border="0" cellborder="0" cellspacing="0" cellpadding="1">'
            + "".join(rows) + '</table>>')


def build_dot(g, title: str, all_effects: bool, details: bool) -> str:
    claims, deps = g["claims"], g["deps"]
    depended_on = {p for _c, p in deps}
    lines = [
        "digraph claims {",
        '  graph [rankdir=LR, splines=ortho, nodesep=0.35, ranksep=0.9, '
        'bgcolor="white", fontname="Helvetica"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=10, margin="0.12,0.08"];',
        '  edge [color="#9AA2B0", arrowsize=0.7];',
        f'  labelloc="t"; fontsize=13; fontname="Helvetica";',
        f'  label=<<b>{esc(title)}</b>>;',
    ]
    effects = [cid for cid, c in claims.items() if c["kind"] == "effect"]
    shown_effects = [cid for cid in effects
                     if all_effects or cid in depended_on]

    for cid, c in claims.items():
        if c["kind"] == "effect" or c["kind"] == "obligation":
            continue          # obligations ride inside their claim's card
        border, fill = STATUS_STYLE.get(c["status"],
                                        STATUS_STYLE["unverified"])
        width = "2.2" if c["kind"] == "final" else "1.3"
        lines.append(
            f'  "{cid}" [shape={KIND_SHAPE.get(c["kind"], "box")}, '
            f'color="{border}", fillcolor="{fill}", penwidth={width}, '
            f'label={claim_label(g, cid, details)}];')

    if shown_effects:
        lines.append('  subgraph cluster_ledger {')
        lines.append('    label="ledger (effect claims a claim leans on)"; '
                     'fontsize=10; color="#E1E5EC"; style=filled; '
                     'fillcolor="#F6F7F9";')
        for cid in sorted(shown_effects,
                          key=lambda c: claims[c]["text"]):
            border, fill = STATUS_STYLE.get(claims[cid]["status"],
                                            STATUS_STYLE["unverified"])
            lines.append(f'    "{cid}" [shape=note, color="{border}", '
                         f'fillcolor="{fill}", fontsize=9, '
                         f'label={effect_label(g, cid, details)}];')
        lines.append("  }")

    drawn = {cid for cid, c in claims.items()
             if c["kind"] not in ("effect", "obligation")} | set(shown_effects)
    for child, parent in deps:
        if child in drawn and parent in drawn:
            lines.append(f'  "{parent}" -> "{child}";')
    for cid, c in claims.items():
        if c.get("revises") and cid in drawn and c["revises"] in drawn:
            lines.append(f'  "{c["revises"]}" -> "{cid}" '
                         f'[style=dashed, color="#7A5CC9", '
                         f'label=<<font point-size="8">revises</font>>];')
    lines.append("}")
    hidden = len(effects) - len(shown_effects)
    if hidden:
        print(f"  ({hidden} effect claims nothing depends on are hidden; "
              f"--all-effects shows them)")
    return "\n".join(lines)


def summarize(g) -> str:
    counts = {}
    for c in g["claims"].values():
        counts[(c["kind"], c["status"])] = \
            counts.get((c["kind"], c["status"]), 0) + 1
    parts = []
    for kind in ("assertion", "final", "obligation", "effect"):
        by_status = {s: n for (k, s), n in counts.items() if k == kind}
        if by_status:
            inner = ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
            parts.append(f"{sum(by_status.values())} {kind} ({inner})")
    verdicts = [c.get("verdict") for c in g["certificates"] if c.get("verdict")]
    if verdicts:
        parts.append("certificate: " + " → ".join(verdicts))
    return " · ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--all-effects", action="store_true")
    ap.add_argument(
        "--details", action="store_true",
        help="show required/achieved basis and latest obligation evidence",
    )
    ap.add_argument("--png", action="store_true", help="also write a PNG")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.state_dir, "claims.sqlite")):
        sys.exit(f"no claims.sqlite in {args.state_dir}")
    if shutil.which("dot") is None:
        sys.exit("graphviz `dot` not found (brew install graphviz)")

    g = load(args.state_dir)
    summary = summarize(g)
    title = args.title or os.path.abspath(args.state_dir)
    dot = build_dot(g, f"{title}\n{summary}", args.all_effects, args.details)
    out = args.out or os.path.join(args.state_dir, "claim-graph.svg")
    proc = subprocess.run(["dot", "-Tsvg", "-o", out], input=dot, text=True,
                          capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"dot failed: {proc.stderr[:500]}")
    print(f"wrote {out} — {summary}")
    if args.png:
        png = os.path.splitext(out)[0] + ".png"
        subprocess.run(["dot", "-Tpng", "-Gdpi=140", "-o", png], input=dot,
                       text=True, check=True)
        print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
