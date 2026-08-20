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

Reads claims.sqlite (+ events.jsonl for labels) and writes SVG via `dot`.
Effect claims are collapsed into one ledger strip unless a claim depends
on them (--all-effects keeps every one).
"""
from __future__ import annotations

import argparse
import html
import json
import os
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


def load(state_dir: str) -> dict:
    db = sqlite3.connect(os.path.join(state_dir, "claims.sqlite"))
    claims = {
        cid: {"id": cid, "kind": k, "status": s, "reason": r, "text": t,
              "revises": rev}
        for cid, k, s, r, t, rev in db.execute(
            "select id, kind, status, reason, text, revises from claims")
    }
    deps = list(db.execute("select child, parent from deps"))
    obligations = {}
    for claim, obl in db.execute("select claim, obligation from obligation_edges"):
        obligations.setdefault(claim, []).append(obl)
    db.close()

    labels, certificates = {}, []
    events_path = os.path.join(state_dir, "events.jsonl")
    if os.path.exists(events_path):
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                payload = event["payload"]
                if event["type"] == "commit":
                    for cid, c in claims.items():
                        if c["text"] == payload["text"] \
                                and c["kind"] == payload["kind"]:
                            labels[cid] = payload["label"]
                elif event["type"] == "certificate":
                    certificates.append(payload)
    return {"claims": claims, "deps": deps, "obligations": obligations,
            "labels": labels, "certificates": certificates}


def claim_label(g, cid) -> str:
    """HTML-like node label: header line, full text, obligation rows."""
    c = g["claims"][cid]
    border, _fill = STATUS_STYLE.get(c["status"], STATUS_STYLE["unverified"])
    name = g["labels"].get(cid, cid[:6])
    head = (f'<tr><td align="left"><b>{esc(name)}</b>'
            f'<font point-size="9" color="#8B92A1">  {esc(cid[:6])} · '
            f'{esc(c["kind"])}</font></td>'
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
            rows.append(
                f'<tr><td align="left"><font point-size="9" color="{ob}">'
                f'● </font><font point-size="9">{esc(text)}</font></td>'
                f'<td align="right"><font point-size="8" color="{ob}">'
                f'{esc(o["status"])}</font></td></tr>')
            if o["reason"] and o["status"] == "refuted":
                rows.append(
                    f'<tr><td colspan="2" align="left"><font point-size="8" '
                    f'color="#C0483B">   ↳ {esc(" ".join(wrap(o["reason"], 54, 2)))}'
                    f'</font></td></tr>')
    return ('<<table border="0" cellborder="0" cellspacing="1" cellpadding="2">'
            + "".join(rows) + "</table>>")


def effect_label(g, cid) -> str:
    c = g["claims"][cid]
    try:
        payload = json.loads(c["text"])
        head = f"#{payload.get('event_seq')} {payload.get('tool') or '—'}"
    except (json.JSONDecodeError, TypeError):
        head = cid[:6]
    return (f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="1">'
            f'<tr><td><b>{esc(head)}</b></td></tr>'
            f'<tr><td><font point-size="8" color="#8B92A1">{esc(cid[:6])}'
            f'</font></td></tr></table>>')


def build_dot(g, title: str, all_effects: bool) -> str:
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
            f'label={claim_label(g, cid)}];')

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
                         f'label={effect_label(g, cid)}];')
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
    ap.add_argument("--png", action="store_true", help="also write a PNG")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.state_dir, "claims.sqlite")):
        sys.exit(f"no claims.sqlite in {args.state_dir}")
    if shutil.which("dot") is None:
        sys.exit("graphviz `dot` not found (brew install graphviz)")

    g = load(args.state_dir)
    summary = summarize(g)
    title = args.title or os.path.abspath(args.state_dir)
    dot = build_dot(g, f"{title}\n{summary}", args.all_effects)
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
