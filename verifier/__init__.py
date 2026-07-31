"""Test-time verification sidecar — M0 skeleton.

Design reference: Test-time Verification technical report v0.3 (report/report.html).
Module map (component numbers from the report's figure 3):
  model.py        data contracts: Claim/Obligation/Evidence, content hashing (§14)
  events.py       unified event stream, append-only JSONL (component 5/10 log side)
  store.py        claim store: graph, status machine, trigger rule (10, 12, 13)
  instruments.py  python sandbox + ledger instrument, fail-closed (25, 26)
  roles.py        stateless LLM roles with schema validation (generic templates)
  builder.py      mechanical parser for offline replay (6, 4)
  kernel.py       M0 kernel: obligations -> discharge -> mechanical audit (14, 15, 20)
  certificate.py  backward slice freeze (13, 23, 33)
"""
