"""Stop precisely the superseded experiment; retain all containers and data."""
import argparse
import json
import subprocess
import time
from pathlib import Path

EXPERIMENT = "cybergym-luna-core-source-pilot5"
PREVIOUS = Path("/Users/hemlix/Documents/General Verifier/.scratch/cybergym-luna-core-cyber30")


def retire(output):
    output.mkdir(parents=True, exist_ok=False)
    # Validate the process identity before signaling the old auto-advance loop.
    state = json.loads((PREVIOUS / "experiment-supervisor.json").read_text())
    pid = state["pid"]
    process = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True)
    command = process.stdout.strip()
    if command:
        if "diagnostics/advance_after_pilot.py" not in command:
            raise RuntimeError("old supervisor PID has been reused; refusing to signal")
        subprocess.run(["kill", "-TERM", str(pid)], check=True)
    ids = subprocess.check_output(["docker", "ps", "-q", "--filter",
        "label=gv.experiment=" + EXPERIMENT], text=True).split()
    rows = []
    for container in ids:
        row = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
        if row["Config"]["Labels"].get("gv.experiment") != EXPERIMENT:
            raise RuntimeError("container identity changed")
        rows.append({"id": row["Id"], "name": row["Name"],
                     "image": row["Image"], "started_at": row["State"]["StartedAt"]})
        with (output / (container + ".log")).open("wb") as log:
            subprocess.run(["docker", "logs", container], stdout=log,
                           stderr=subprocess.STDOUT, check=False)
    (output / "retirement.json").write_text(json.dumps({
        "retired_unix": time.time(), "experiment": EXPERIMENT,
        "supervisor_pid": pid, "supervisor_command": command,
        "containers": rows, "reason": "superseded-by-shared-GV-cyber-role-suite",
        "results_accepted_for_new_profile": False, "data_deleted": False,
    }, indent=2))
    orchestrators = [row["id"] for row in rows if "orchestrator" in row["name"]]
    if orchestrators:
        subprocess.run(["docker", "stop", "--time", "10", *orchestrators], check=True)
    rest = [row["id"] for row in rows if row["id"] not in orchestrators]
    if rest:
        subprocess.run(["docker", "stop", "--time", "10", *rest], check=True)
    print("Retired superseded workers; preserved containers, volumes, and traces.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    retire(parser.parse_args().output)
