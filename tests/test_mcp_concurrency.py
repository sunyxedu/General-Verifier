"""The MCP verbs run off the server's event loop: a commit whose checks take
minutes no longer freezes the server. A status query answers while a check
is running, two claims' checks proceed side by side, and concurrent verbs
still ingest the hook feed exactly once."""
from __future__ import annotations

import json
import threading
import time

import anyio

from verifier.core.store import TRIGGER_ON_COMMIT
from verifier.mcp_server import Service, build_server

CHECK_SECONDS = 1.0


class SlowKernel:
    """Stands in for the kernel: every committed claim takes CHECK_SECONDS
    to check, and the checks record when they ran."""
    def __init__(self, service):
        self.service = service
        self.spans = []
        self._lock = threading.Lock()
        service.store.on_trigger(self.verify)

    def verify(self, claim_id, force=False):
        start = time.time()
        time.sleep(CHECK_SECONDS)
        with self._lock:
            self.spans.append((start, time.time()))
        return self.service.store.status(claim_id)


def service_and_server(tmp_path):
    service = Service(str(tmp_path), live_world=False, trigger=TRIGGER_ON_COMMIT,
                      kernel_factory=SlowKernel)
    return service, build_server(service)


def text_of(result):
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


def test_a_status_query_answers_while_a_commit_is_being_checked(tmp_path):
    service, server = service_and_server(tmp_path)
    timings = {}

    async def main():
        async def slow_commit():
            t0 = time.time()
            await server.call_tool("commit", {"label": "A1", "kind": "assertion",
                                              "text": "the file has three lines",
                                              "deps": []})
            timings["commit"] = time.time() - t0

        async with anyio.create_task_group() as tg:
            tg.start_soon(slow_commit)
            await anyio.sleep(0.2)
            t0 = time.time()
            out = await server.call_tool("status", {})
            timings["status"] = time.time() - t0
            timings["status_body"] = text_of(out)

    anyio.run(main)
    assert timings["commit"] >= CHECK_SECONDS
    assert timings["status"] < 0.5, timings
    assert "A1" in timings["status_body"]


def test_two_claims_are_checked_side_by_side(tmp_path):
    service, server = service_and_server(tmp_path)

    async def main():
        async with anyio.create_task_group() as tg:
            for label in ("A1", "F"):
                tg.start_soon(server.call_tool, "commit", {
                    "label": label, "kind": "final" if label == "F" else "assertion",
                    "text": f"claim {label}", "deps": []})

    t0 = time.time()
    anyio.run(main)
    elapsed = time.time() - t0
    spans = service.kernel.spans
    assert len(spans) == 2
    (a0, a1), (b0, b1) = sorted(spans)
    assert b0 < a1, "the second check waited for the first"
    assert elapsed < 2 * CHECK_SECONDS - 0.2, elapsed


def test_concurrent_verbs_ingest_each_feed_line_once(tmp_path):
    service, _ = service_and_server(tmp_path)
    lines = 200
    with open(tmp_path / "hook_feed.jsonl", "w", encoding="utf-8") as feed:
        for i in range(lines):
            feed.write(json.dumps({"type": "tool_call", "payload": {
                "tool": "Bash", "tool_use_id": f"c{i}", "input": {}}}) + "\n")
    threads = [threading.Thread(target=service.drain_hooks) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ingested = [e for e in service.log.replay() if e["type"] == "tool_call"]
    assert len(ingested) == lines
    assert len({e["payload"]["tool_use_id"] for e in ingested}) == lines
