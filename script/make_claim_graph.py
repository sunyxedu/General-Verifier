#!/usr/bin/env python3
"""Generate docs/claim-graph.svg — the claim graph of one real run, every node on it.

Source of truth is the run's own state, not a sketch:

    experiments/terminal_bench/results/jobs/gv/fix-git__pWuGmXs/artifacts/gv-state/
        claims.sqlite   claims, deps, obligation edges, evidence
        events.jsonl    labels, certificates, prune, the binding game

Ids, statuses, edges and counts are read out of that store; only the prose is
abridged to fit (GLOSS below). The build asserts that every node in the store is
drawn exactly once, so the figure cannot silently drift from the data.

    python3 docs/make_claim_graph.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "experiments/terminal_bench/results/jobs/gv/fix-git__pWuGmXs"
STATE = RUN / "artifacts/gv-state"
OUT = Path(__file__).resolve().parent / "claim-graph.svg"

# palette (shared with make_architecture.py)

INK = "#14161C"
BODY = "#4A5160"
MUTED = "#8B92A1"
HAIR = "#DFE3EA"
PANEL = "#F6F7F9"
PANEL_EDGE = "#E1E5EC"
CANVAS = "#FFFFFF"
ARROW = "#5B6272"

GREEN, GREEN_FILL = "#2E8B57", "#E8F4ED"
AMBER, AMBER_FILL = "#B07407", "#FBF1DF"
RED, RED_FILL = "#C0483B", "#FBEDEB"
VIOLET, VIOLET_FILL = "#7A5CC9", "#F1EDFA"
B_LINE, B_FILL, B_ICON = "#4A82C4", "#EAF1FA", "#2E5F9E"
D_LINE = "#C3C9D4"
GREY, GREY_FILL = "#7C838F", "#F3F4F6"

SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "Menlo, Monaco, 'Courier New', monospace"

T_ZONE = 11.5
T_COMP = 9.5
T_BODY = 8.2
T_MICRO = 7.0
T_EYE = 7.4

W, H = 1220, 928

out: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def est(s: str, size: float, mono: bool = False, bold: bool = False) -> float:
    k = 0.60 if mono else (0.53 if bold else 0.50)
    return len(s) * size * k


def card(x, y, w, h, fill="#FFFFFF", stroke=D_LINE, sw=1.15, rx=9, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def text(x, y, s, size=8, fill=BODY, anchor="start", weight=None, mono=False,
         italic=False, ls=None, limit=None):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    wgt = f' font-weight="{weight}"' if weight else ""
    fam = f' font-family="{MONO}"' if mono else ""
    it = ' font-style="italic"' if italic else ""
    lsp = f' letter-spacing="{ls}"' if ls else ""
    if limit:
        wide = est(s, size, mono, weight in ("bold", 600, 700))
        if ls:
            wide += len(s) * ls
        if wide > limit:
            print(f"  ! overflow {wide:6.1f} > {limit}  {s[:60]!r}")
    out.append(
        f'<text x="{x}" y="{y}" font-size="{size}"{fam} fill="{fill}"{a}{wgt}{it}{lsp}>'
        f"{esc(s)}</text>"
    )


def lines(x, y, rows, size=T_MICRO, lead=10.4, fill=BODY, anchor="start", limit=None,
          italic=False, mono=False):
    for i, r in enumerate(rows):
        text(x, y + i * lead, r, size=size, fill=fill, anchor=anchor, limit=limit,
             italic=italic, mono=mono)


def path(d, color=ARROW, sw=1.2, dash=None, marker="ar"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mk = f' marker-end="url(#{marker})"' if marker else ""
    out.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round"{da}{mk}/>'
    )


def ortho(pts, r=7.0, color="#9AA2B0", sw=1.15, dash=None, marker="arm"):
    """Manhattan route with rounded corners — every dependency edge is drawn this
    way, so the long transitive edges read as a bus rather than as loose arcs."""
    d = [f"M {pts[0][0]} {pts[0][1]}"]
    for i in range(1, len(pts) - 1):
        (px, py), (cx, cy), (nx, ny) = pts[i - 1], pts[i], pts[i + 1]
        din = (cx - px, cy - py)
        dout = (nx - cx, ny - cy)
        li = max(abs(din[0]), abs(din[1])) or 1
        lo = max(abs(dout[0]), abs(dout[1])) or 1
        ri = min(r, li / 2)
        ro = min(r, lo / 2)
        a = (cx - din[0] / li * ri, cy - din[1] / li * ri)
        b = (cx + dout[0] / lo * ro, cy + dout[1] / lo * ro)
        d.append(f"L {a[0]:.1f} {a[1]:.1f} Q {cx} {cy} {b[0]:.1f} {b[1]:.1f}")
    d.append(f"L {pts[-1][0]} {pts[-1][1]}")
    path(" ".join(d), color=color, sw=sw, dash=dash, marker=marker)


def pill(x, y, w, h, label, stroke, fill, size=T_MICRO):
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
    )
    text(x + w / 2, y + h / 2 + 2.5, label, size=size, fill=stroke,
         anchor="middle", weight="bold")


# ---------------------------------------------------------------- read the run

db = sqlite3.connect(STATE / "claims.sqlite")
claims = {
    cid: {"id": cid, "kind": k, "status": s, "reason": r, "text": t}
    for cid, k, s, r, t in db.execute("select id, kind, status, reason, text from claims")
}
deps = [(c, p) for c, p in db.execute("select child, parent from deps")]
obl_edges = [(c, o) for c, o in db.execute("select claim, obligation from obligation_edges")]
revises = {c: r for c, r in db.execute("select id, revises from claims where revises is not null")}

events = [json.loads(line) for line in (STATE / "events.jsonl").read_text().splitlines()]
labels: dict[str, str] = {}       # claim id -> human label from the commit event
certificates: list[dict] = []
pruned: set[str] = set()
for ev in events:
    p = ev["payload"]
    if ev["type"] == "certificate":
        certificates.append({"seq": ev["seq"], **p})
    elif ev["type"] == "branch_pruned":
        pruned.update(p["dead_cone"])
for ev in events:                  # labels come from commit events, in order
    if ev["type"] == "commit":
        p = ev["payload"]
        for cid, c in claims.items():
            if c["text"] == p["text"] and c["kind"] == p["kind"]:
                labels[cid] = p["label"]

effects = []
for cid, c in claims.items():
    if c["kind"] == "effect":
        payload = json.loads(c["text"])
        effects.append((payload["event_seq"], payload["tool"], cid))
effects.sort()

S = lambda pre: next(c for c in claims if c.startswith(pre))          # noqa: E731
short = lambda cid: cid[:6]                                            # noqa: E731

# ---------------------------------------------------------------- abridged prose
# Ids/statuses/edges are verbatim; these are the shortest true paraphrase of each
# node's own text, because the real text runs to several hundred characters.

CLAIM_GLOSS = {
    "adbbaf": ["the lost work is commit c499730 “Move to Stanford”, touching",
               "about.md + default.html — dangling, reflog-only, not on master"],
    "2fde52": ["master and c499730 fork at c4e38a1 and both edit about.md, so",
               "the merge will conflict there; default.html merges clean"],
    "950adf": ["how to resolve: take c499730’s first line, keep the shared",
               "“Previously,…” paragraph, take default.html whole"],
    "64023d": ["merged: master HEAD is 46d6e6d, tree clean; about.md and",
               "default.html carry the Stanford text; diff vs d7d3e4b = 19+/17−"],
    "ed7ac4": ["the merge (46d6e6d) did need manual resolution in about.md,",
               "and it was resolved by taking c499730’s version"],
    "946ce2": ["about.md is c499730’s in full, the job-search line is gone from",
               "both sides, default.html came over whole"],
    "bcc2b7": ["merged: master HEAD is 46d6e6d, tree clean; about.md and",
               "default.html carry the Stanford text; diff vs d7d3e4b = 19+/17−"],
}

OBL_GLOSS = {
    "f6b9af": "c499730 exists, subject “Move to Stanford”",
    "3e8e52": "c499730 changes _includes/about.md",
    "07ff94": "c499730 changes _layouts/default.html",
    "37953b": "unreachable from every ref",
    "43ba80": "but present in the reflog",
    "5704f1": "master resolves to d7d3e4b",
    "784511": "c499730 is not an ancestor of master",
    "cae77f": "its changes really are absent from master",
    "3f4ecf": "c4e38a1 is the merge base of the two",
    "790751": "master’s d7d3e4b also edits about.md",
    "4406a5": "master rewrote line 1, dropped the job line",
    "c05556": "46d6e6d merges d7d3e4b with c499730",
    "c424ee": "master now resolves to 46d6e6d",
    "a1fcdf": "the 3-way merge does conflict in about.md",
    "2ac3a2": "about.md at 46d6e6d == c499730’s bytes",
    "7270c4": "default.html untouched on master’s side",
    "d3353f": "default.html merged clean, == c499730’s",
    "1f32dd": "line 1 is “I am a Postdoctoral Researcher…”",
    "3690d8": "the shared “Previously,…” paragraph stays",
    "117648": "“looking for a full time job!” is gone",
    "2e67bb": "both sides had removed that line",
    "52924c": "“Postdoc @ Stanford” is in default.html",
    "db3ca1": "font link in c499730’s two-line form",
    "5a4916": "git status --porcelain is empty",
    "45f6bb": "diff d7d3e4b..master is exactly 19+/17−",
}

TOOL_SHORT = {"mcp__gv__commit": "gv.commit", "mcp__gv__certificate": "gv.cert", None: "—"}

STATUS_COLOR = {
    "verified": (GREEN, GREEN_FILL),
    "refuted": (RED, RED_FILL),
    "stale": (GREY, GREY_FILL),
    "unverified": (GREY, "#FFFFFF"),
    "partial": (AMBER, AMBER_FILL),
}

drawn: set[str] = set()

# ---------------------------------------------------------------- figure

out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" role="img" aria-label="claim graph of one run" '
    f'font-family="{SANS}">'
)
out.append(
    "<defs>"
    + "".join(
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="{mw}" '
        f'markerHeight="{mw}" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 Z" fill="{col}"/></marker>'
        for mid, col, mw in (("ar", ARROW, 5.4), ("arm", "#9AA2B0", 4.8),
                             ("arv", VIOLET, 4.6), ("arr", RED, 4.8))
    )
    + "</defs>"
)
out.append(f'<rect width="{W}" height="{H}" fill="{CANVAS}"/>')

# ---- header
text(44, 34, "CLAIM GRAPH · ONE RUN, EVERY NODE", T_EYE, "#7C838F", weight="bold", ls=2.1)
text(44, 56, "terminal-bench / fix-git — the 53 nodes the store actually holds",
     T_ZONE, INK, weight="bold")
text(44, 74, "5 assertions · 2 finals · 25 obligations · 21 effects. The first certificate came back "
             "refuted; the solver revised three claims; the second verified.",
     T_BODY, BODY, limit=880)
lines(1176, 34, [
    "trial fix-git__pWuGmXs · task reward 1.0",
    f"{len(events)} events · {len(certificates)} certificates · 1 counterexample",
    "read from artifacts/gv-state/claims.sqlite",
    "trigger policy at the time: on-first-dependent (now: on-commit)",
], size=T_MICRO, lead=11.4, fill=MUTED, anchor="end")
out.append(f'<line x1="44" y1="92" x2="{W-44}" y2="92" stroke="{HAIR}" stroke-width="1"/>')

# ---- geometry
CW, CH = 248, 74
C1, C2, C3, C4 = 44, 336, 628, 920
LANE_X, LANE_W = 316, 880
L1_Y, L2_Y = 196, 470           # card tops
ROOT_Y = 290
CHIP_W, CHIP_H, CHIP_PITCH = 248, 18, 20

card(LANE_X, 116, LANE_W, 276, fill=PANEL, stroke=PANEL_EDGE, sw=1.1, rx=12)
text(LANE_X, 110, "ATTEMPT 1   ·   CERTIFICATE REFUTED", T_EYE, "#9AA2B0", weight="bold", ls=2.0)
card(LANE_X, 424, LANE_W, 292, fill=PANEL, stroke=PANEL_EDGE, sw=1.1, rx=12)
text(LANE_X, 418, "ATTEMPT 2   ·   THE REVISION THAT CERTIFIED", T_EYE, "#9AA2B0",
     weight="bold", ls=2.0)


def claim_card(x, y, cid, note=None, w=CW):
    """One assertion/final node: label, id, abridged text, status."""
    c = claims[cid]
    stroke, fill = STATUS_COLOR[c["status"]]
    is_final = c["kind"] == "final"
    card(x, y, w, CH, fill="#FFFFFF", stroke=stroke, sw=1.6 if is_final else 1.3, rx=8)
    if is_final:                     # finals get a heavier left rule — they are what a
        out.append(f'<rect x="{x}" y="{y+1}" width="4.5" height="{CH-2}" rx="2.2" '
                   f'fill="{stroke}"/>')  # certificate is issued over
    name = labels.get(cid, "?") + ("′" if cid in revises else "")
    text(x + 14, y + 17, name, T_COMP, INK, weight="bold", mono=True, limit=150)
    lab = c["status"]
    pw = est(lab, T_MICRO, bold=True) + 14
    pill(x + w - pw - 10, y + 7, pw, 13, lab, stroke, fill)
    text(x + 14, y + 30, f"{short(cid)}…  {c['kind']}", T_MICRO, MUTED, mono=True, limit=150)
    lines(x + 14, y + 45, CLAIM_GLOSS[short(cid)], size=T_MICRO, lead=10.6, fill=BODY,
          limit=w - 24)
    if note:
        text(x + 14, y + 68, note, T_MICRO, MUTED, italic=True, limit=w - 26)
    drawn.add(cid)


def chips(x, y, cid, w=CHIP_W):
    """The obligation nodes the kernel compiled for this claim, all of them."""
    kids = [o for c, o in obl_edges if c == cid]
    out.append(f'<line x1="{x+9}" y1="{y-6}" x2="{x+9}" y2="{y + (len(kids)-1)*CHIP_PITCH + CHIP_H/2}" '
               f'stroke="{HAIR}" stroke-width="1.1"/>')
    for i, oid in enumerate(kids):
        cy = y + i * CHIP_PITCH
        st = claims[oid]["status"]
        stroke, fill = STATUS_COLOR[st]
        out.append(f'<line x1="{x+9}" y1="{cy + CHIP_H/2}" x2="{x+17}" y2="{cy + CHIP_H/2}" '
                   f'stroke="{HAIR}" stroke-width="1.1"/>')
        card(x + 17, cy, w - 17, CHIP_H, fill=fill if st == "refuted" else "#FFFFFF",
             stroke=stroke, sw=1.15, rx=5)
        out.append(f'<circle cx="{x+28}" cy="{cy + CHIP_H/2}" r="3" fill="{stroke}"/>')
        text(x + 36, cy + 12.4, short(oid), T_MICRO, MUTED, mono=True)
        text(x + 76, cy + 12.4, OBL_GLOSS[short(oid)], T_MICRO, INK, limit=w - 88)
        drawn.add(oid)
    return y + len(kids) * CHIP_PITCH


A1, A2, A3, F1 = S("adbbaf"), S("2fde52"), S("950adf"), S("64023d")
B2, B3, F2 = S("ed7ac4"), S("946ce2"), S("bcc2b7")

# ---- root claim (shared parent of both attempts)
text(C1, 282 - 6, "ROOT", T_EYE, "#9AA2B0", weight="bold", ls=2.0)
claim_card(C1, ROOT_Y, A1, note="8 obligations · all held · verified at seq 55")
chips(C1, ROOT_Y + CH + 8, A1)

card(C1, 556, CW, 104, fill="#FFFFFF", stroke=D_LINE, sw=1.05, rx=8, dash="3.4 2.6")
text(C1 + 12, 573, "HOW THIS GRAPH GOT BUILT", T_MICRO, "#7C838F", weight="bold", ls=1.4)
for i, (seq, note) in enumerate([
    ("18", "solver commits found-changes"),
    ("22–55", "8 obligations compiled · all held"),
    ("71–72", "one obligation refuted · dead cone pruned"),
    ("92", "certificate #1 — refuted"),
    ("95–128", "3 claims revised · 14 fresh obligations"),
    ("167", "certificate #2 — verified"),
]):
    text(C1 + 12, 588 + i * 11.4, f"seq {seq}", T_MICRO, MUTED, mono=True)
    text(C1 + 62, 588 + i * 11.4, note, T_MICRO, BODY, limit=CW - 74)

# ---- attempt 1
claim_card(C2, L1_Y, A2, note="refuted at seq 71, later superseded")
chips(C2, L1_Y + CH + 8, A2)
claim_card(C3, L1_Y, A3, note="never compiled — in the pruned dead cone")
claim_card(C4, L1_Y, F1, note="certificate asked for; slice fails closed")

# the counterexample, verbatim from the store (witness, truncated there at 200 chars)
cx = claims[S("4406a5")]["reason"]
card(C2, 346, CW, 40, fill=RED_FILL, stroke=RED, sw=1.1, rx=6)
text(C2 + 10, 359, "COUNTEREXAMPLE  ·  obligation 4406a5", T_MICRO, RED, weight="bold")
lines(C2 + 10, 371, ["“…at c4e38a1, about.md line 3 is ‘I am looking for a full",
                     "time job!…’ — not line 1”   (witness, as stored)"],
      size=T_MICRO, lead=9.8, fill=RED, limit=CW - 20)
assert "c4e38a1" in cx and "line 3" in cx

card(C3, 278, CW, 44, fill="#FFFFFF", stroke=D_LINE, sw=1.05, rx=6, dash="3.4 2.6")
lines(C3 + 12, 294, ["dead cone: when 2fde52 was refuted the kernel pruned",
                     "everything downstream (seq 72), so no obligation was",
                     "ever compiled for this claim — nothing is proved cheap"],
      size=T_MICRO, lead=10.2, fill=MUTED, limit=CW - 24)

c1_ev = certificates[0]
card(C4, 278, CW, 44, fill=RED_FILL, stroke=RED, sw=1.2, rx=6)
text(C4 + 12, 294, "CERTIFICATE #1  ·  REFUTED", T_MICRO, RED, weight="bold")
lines(C4 + 12, 306, [f"digest {c1_ev['digest'][:16]}…",
                     "verdict = worst case over the slice; never rounds up"],
      size=T_MICRO, lead=10.2, fill=RED, limit=CW - 24)

# ---- attempt 2
claim_card(C2, L2_Y, B2, note="6 fresh obligations · verified at seq 124")
chips(C2, L2_Y + CH + 8, B2)
claim_card(C3, L2_Y, B3, note="6 fresh obligations · verified at seq 153")
chips(C3, L2_Y + CH + 8, B3)
claim_card(C4, L2_Y, F2, note="2 obligations · verified at seq 166")
chips(C4, L2_Y + CH + 8, F2)

c2_ev = certificates[1]
card(C4, 600, CW, 44, fill=GREEN_FILL, stroke=GREEN, sw=1.2, rx=6)
text(C4 + 12, 616, "CERTIFICATE #2  ·  VERIFIED", T_MICRO, GREEN, weight="bold")
lines(C4 + 12, 628, [f"digest {c2_ev['digest'][:16]}…",
                     "the whole backward slice of this final is settled"],
      size=T_MICRO, lead=10.2, fill=GREEN, limit=CW - 24)

# ---- dependency edges: all 12, exactly as the deps table has them.
# Channels: the root fans out through the white gutter at x=300/306/312 — outer
# channel = farther target — then runs along a horizontal bus into each column.
E = [
    # attempt 1 — the bus runs above the card row
    (A2, A1, [(292, 316), (312, 316), (312, 240), (C2, 240)]),
    (A3, A2, [(C2 + CW, 233), (C3, 233)]),
    (F1, A3, [(C3 + CW, 233), (C4, 233)]),
    (A3, A1, [(292, 306), (306, 306), (306, 150), (C3 + 72, 150), (C3 + 72, L1_Y)]),
    (F1, A1, [(292, 296), (300, 296), (300, 132), (C4 + 220, 132), (C4 + 220, L1_Y)]),
    (F1, A2, [(C2 + 150, L1_Y), (C2 + 150, 176), (C4 + 150, 176), (C4 + 150, L1_Y)]),
    # attempt 2 — the bus runs below the obligation stacks
    (B2, A1, [(292, 338), (312, 338), (312, 500), (C2, 500)]),
    (B3, B2, [(C2 + CW, 507), (C3, 507)]),
    (F2, B3, [(C3 + CW, 507), (C4, 507)]),
    (B3, A1, [(292, 348), (306, 348), (306, 690), (C3 - 28, 690), (C3 - 28, 520), (C3, 520)]),
    (F2, A1, [(292, 358), (300, 358), (300, 702), (C4 - 28, 702), (C4 - 28, 520), (C4, 520)]),
    (F2, B2, [(C2 + 150, L2_Y), (C2 + 150, 442), (C4 + 150, 442), (C4 + 150, L2_Y)]),
]
for child, parent, pts in E:
    assert (child, parent) in deps, (short(child), short(parent))
    ortho(pts)
assert len(E) == len(deps) == 12

# ---- revision lineage (dashed — lineage, not a dependency)
for old, new, gutter in ((A2, B2, C2 + CW + 22), (A3, B3, C3 + CW + 22), (F1, F2, C4 + CW + 18)):
    assert revises[new] == old, (short(new), short(old))
    path(f"M {gutter - 22} 214 L {gutter} 214 L {gutter} 488 L {gutter - 22} 488",
         color=VIOLET, sw=1.05, dash="3.6 2.8", marker="arv")
assert len(revises) == 3
text(C2 + CW + 28, 408, "revises", T_MICRO, VIOLET, italic=True)

# ---- effect ledger
LY = 744
card(44, LY, W - 88, 118, fill=PANEL, stroke=PANEL_EDGE, sw=1.1, rx=12)
text(60, LY + 22, "EFFECT LEDGER — 21 nodes, one per tool result the hooks saw",
     T_COMP, INK, weight="bold")
text(60, LY + 36, "each is {event_seq, tool, digest} of the event bytes · written by the "
                  "deterministic builder, no LLM · all unverified here: effect claims are "
                  "checked only once some claim leans on one, and none did",
     T_MICRO, BODY, limit=1080)
ex, ew, gap = 58, 48, 4.6
for i, (seq, tool, cid) in enumerate(effects):
    x = ex + i * (ew + gap)
    gv = (tool or "").startswith("mcp__gv__")
    stroke, fill = (B_LINE, B_FILL) if gv else (D_LINE, "#FFFFFF")
    card(x, LY + 50, ew, 32, fill=fill, stroke=stroke, sw=1.1, rx=5)
    text(x + ew / 2, LY + 64, f"#{seq}", T_MICRO, B_ICON if gv else INK, anchor="middle",
         weight="bold", mono=True)
    text(x + ew / 2, LY + 75, TOOL_SHORT.get(tool, tool), T_MICRO, MUTED if not gv else B_ICON,
         anchor="middle", limit=ew - 4)
    text(x + ew / 2, LY + 92, short(cid), T_MICRO, "#B7BDC8", anchor="middle", mono=True)
    drawn.add(cid)
text(60, LY + 110, "solver's own tools (Bash, Read) in grey · gv.* MCP calls in blue — the "
                   "verifier's own traffic is on the ledger too", T_MICRO, MUTED, limit=1080)

# ---------------------------------------------------------------- legend + footnote
LEG = [
    (GREEN, GREEN_FILL, "verified", None),
    (RED, RED_FILL, "refuted — counterexample on record", None),
    (GREY, GREY_FILL, "stale — superseded by a revision", None),
    (GREY, "#FFFFFF", "unverified — no dependent yet", None),
    (VIOLET, "#FFFFFF", "revises (lineage, not a dependency)", "3.6 2.8"),
]
gap_l = 26
widths = [13 + 8 + est(lbl, T_MICRO) for _, _, lbl, _ in LEG]
lx = (W - (sum(widths) + gap_l * (len(LEG) - 1))) / 2
for (stroke, fill, label, dash), wd in zip(LEG, widths):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<rect x="{lx:.1f}" y="{H-34}" width="13" height="13" rx="3.4" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.3"{d}/>'
    )
    text(lx + 20, H - 24, label, T_MICRO, "#6C7382")
    lx += wd + gap_l
text(W / 2, H - 8, "ids · statuses · edges · counts are verbatim from the run's store; claim and "
                   "obligation text is abridged to fit. Obligations exist only for claims the "
                   "kernel actually verified — the dead cone has none.",
     T_MICRO, MUTED, anchor="middle", italic=True, limit=1120)

out.append("</svg>")

missing = set(claims) - drawn
assert not missing, f"{len(missing)} nodes not drawn: {[short(m) for m in missing]}"
assert len(drawn) == len(claims) == 53, (len(drawn), len(claims))

OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB, {W}x{H}) — {len(drawn)} nodes, "
      f"{len(deps)} dep edges, {len(obl_edges)} obligation edges")
