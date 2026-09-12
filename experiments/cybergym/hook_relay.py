"""Forward normal tool hook records to this attempt's MCP sidecar."""
import json
import os
import sys
import urllib.request


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("pre", "post"):
        return 2
    raw = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        return 3
    request = urllib.request.Request(os.environ["GV_SIDECAR_URL"] + "/hooks/" + sys.argv[1],
        data=raw, headers={"Authorization": "Bearer " + os.environ["GV_MCP_BEARER_TOKEN"],
                           "Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
