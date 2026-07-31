"""Memory (M4, report §13): three stores, purely advisory.

1. obligations — obligation patterns that were actually discharged, keyed by
   the claim they served. Retrieved into the compiler's {memory_hits}.
2. attacks — variants and witnesses that exposed real problems (false passes
   caught by the exam, refutations with counterexamples). Retrieved into the
   variant generator's {memory_hits}: yesterday's escape becomes today's exam
   question.
3. instruments — instruments confirmed working by a bound method. Summarized
   into the oracle enumerator's {registry} so probing starts warm.

Master principle applies with full force: memory feeds PROMPT SLOTS only.
A hit can make the system faster or more thorough; it cannot make a wrong
claim verify, because every path it influences still ends at the same
mechanical gates. Retrieval is deterministic (character-bigram Jaccard),
storage is append-only JSONL with content-hash dedup.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .model import content_hash

STORES = ("obligations", "attacks", "instruments")


def _bigrams(text: str) -> set:
    text = "".join(text.split())
    return {text[i:i + 2] for i in range(len(text) - 1)} or {text}


def _similarity(a: str, b: str) -> float:
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / len(x | y)


class Memory:
    def __init__(self, base_dir: str):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self._records: Dict[str, List[dict]] = {}
        self._seen: Dict[str, set] = {}
        for store in STORES:
            self._records[store] = []
            self._seen[store] = set()
            path = self._path(store)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            self._records[store].append(record)
                            self._seen[store].add(content_hash(record))

    def _path(self, store: str) -> str:
        return os.path.join(self.base_dir, f"{store}.jsonl")

    def _append(self, store: str, record: dict) -> bool:
        digest = content_hash(record)
        if digest in self._seen[store]:
            return False
        self._seen[store].add(digest)
        self._records[store].append(record)
        with open(self._path(store), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return True

    # ---- recording (mechanical; called by kernel/game) -----------------------

    def record_obligations(self, claim_text: str,
                           obligations: List[str], outcome: str) -> bool:
        return self._append("obligations", {
            "claim": claim_text, "obligations": obligations, "outcome": outcome,
        })

    def record_attack(self, claim_text: str, variant_text: str,
                      evidence: str, kind: str) -> bool:
        """kind: false_pass_caught | refutation"""
        return self._append("attacks", {
            "claim": claim_text, "variant": variant_text,
            "evidence": evidence, "kind": kind,
        })

    def record_instrument(self, procedure: str) -> bool:
        return self._append("instruments", {
            "procedure": procedure,
        })

    # ---- retrieval -> prompt-slot strings ------------------------------------

    def _top(self, store: str, query: str, k: int) -> List[dict]:
        scored = [
            (_similarity(query, r.get("claim", "")), i, r)
            for i, r in enumerate(self._records[store])
        ]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [r for score, _i, r in scored[:k] if score > 0]

    def hits_for_obligations(self, claim_text: str, k: int = 3) -> str:
        return json.dumps(self._top("obligations", claim_text, k),
                          ensure_ascii=False)

    def hits_for_attacks(self, claim_text: str, k: int = 3) -> str:
        return json.dumps(self._top("attacks", claim_text, k),
                          ensure_ascii=False)

    def registry_summary(self, k: int = 10) -> str:
        return json.dumps(self._records["instruments"][-k:],
                          ensure_ascii=False)
