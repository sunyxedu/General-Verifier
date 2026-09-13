# Cyber evaluation settings

## Local experiment setup

Check out the pinned official runtime (`OFFICIAL_COMMIT` in `run.py`) at
`/private/tmp/cybergym-official` and install its server and task-generator
dependencies in `.venv`. Set `CYBERGYM_OFFICIAL_ROOT` for another location.
Import the exact agent, Squid, and nginx image IDs recorded in `run.py`;
create the shared internal network with `docker network create --internal
cybergym-internal`. The runner reads `~/.codex/auth.json` by default;
`CYBERGYM_CODEX_AUTH` selects another private auth file. Auth files and raw
experiment outputs must never be committed.

Concurrent verifier CLI helpers use a 64 MiB tmpfs at `/codex-home/tmp`.
Keep this mount: macOS bind-backed helper directories can disappear during
another CLI's janitor startup, causing unified execution ENOENT. Private auth
stays outside tmpfs and the original shell permission profile stays in place.
An opt-in zero-model native regression exercises the actual runner mount:
`GV_CYBERGYM_HELPER_NATIVE=1 python -B -m pytest -q tests/test_cybergym_helper_native.py`.

Before launch, run the regression suite and
`python -B -m experiments.cybergym.preflight`, then run the offline generic
toolchain preflight inside the pinned agent image. Commit the implementation
and use a fresh, private results root:

```sh
python -B -u -m experiments.cybergym.run \
  --root experiments/cybergym/results/ROUND --port 18692
```

The controller records private status and tracking snapshots throughout the
run. It advances from 5/5 to at least 26/30, then the pinned 1,507-task catalog.
If a gate fails, inspect infrastructure errors and trajectories before making
a general code fix and launching a fresh cohort. Historical results from a
different implementation or an invalid search policy do not satisfy a gate.

`cyber` replaces the six role profiles inside the existing GV kernel. It is
not a second verifier layered on top of standard GV.

## Submission interface and evidence

The supplied `bash ./submit.sh PATH_TO_POC` interface remains functional. Its
public script delegates registration and observation to the existing MCP
tools; it contains no private evaluator configuration and cannot finalize a
candidate. Lost replies with a known job handle recover that same job rather
than launching another evaluation.

The host evaluator executes the original official submission script in its
private, read-only snapshot. It records the actual child command, candidate
byte identity, completion, transport status, and raw public response. Script,
shell, and HTTP-client dependencies are pinned, and static configuration
checks run before the solver starts. A clean child environment and controlled
HTTP-client configuration disable inherited credentials, proxies, and redirects.

This record is evidence, not a verdict. The child exit code, target exit code,
and later hidden patched-build score remain distinct. The original GV
Compiler, Critic, Oracle, Discharger, and Judger still decide the claims and
certificates; neither the script nor the evaluator marks a task verified.

## Public execution tools

The host can register a pinned, data-only runtime inventory with
`--cyber-runtime-manifest /usr/local/lib/gv-sanitizer/manifest.json`.
Omitting it preserves the existing behavior. The inventory describes tools
and limitations; it is not an observation, verdict, or certificate.

Execution and audit seats can explicitly run
`gv-sanitizer-run --runtime-profile clang-static|gcc-shared -- COMMAND ...`.
These profiles apply a narrowly scoped leak-policy compatibility hook without
rewriting the target or compilation command, disabling ASan memory checks, or
granting `/proc` or network access. They are declared runtime conventions,
not automatic executable detection: a mismatched profile or a target's strong
LSan callback can still fail. Leak checking remains **unchecked**, and zero
exit must not be interpreted as proof of leak freedom.

The host validates root-owned files and their hashes. Inside the sandbox,
the tool validates an immutable, read-only `/usr` anchor instead of treating
remapped UIDs as host identities. Planning seats cannot read the tool files.
Tool diagnostics remain ordinary execution output, including when a child
prints similar text; they never authorize a candidate by themselves.

Execution/audit calls use a private Linux mount namespace for fixed `/tmp`
paths. The host binds owned per-call scratch before the CLI starts; it never
grants access to the shared temporary directory or adds `/proc` access to seat
commands. CLI final output and image uploads use separate private capture
storage. Discharger temporary files join its bounded evidence snapshot before
cleanup, with an explicit transport-only mapping manifest. A failed namespace
setup refuses execution. Run the zero-model native regression before launching
on a new host or engine:

```sh
GV_CYBERGYM_HELPER_NATIVE=1 python -B -m pytest -q tests/test_cyber_native_tmp.py tests/test_cybergym_helper_native.py
```

## Model configuration

The solver and every verification role use **GPT-5.6 Luna, max effort, fast**.
Their context policy is shared in `model_config.py`: 1,050,000 total tokens,
automatic compaction at 780,000 input tokens. Pinned Codex real-turn traces
report 828,400 usable tokens; the threshold leaves 48,400 below that budget.
Both fresh solver turns and
same-thread continuations have explicit CLI overrides; all verifier roles
have the same overrides. The frozen manifest records both settings. This
does not disable compaction or claim that input alone can occupy 1M tokens.
[Official model limits](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
are 922,000 input and 128,000 output tokens.
[Official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
documents the context-window and automatic-compaction settings.

## Network boundary

Online mode submits to a **local** CyberGym service; it is not an Internet
answer service. Solver containers have no direct Internet route. Their
proxy allowlist consists of `.openai.com`, `.chatgpt.com`, `.oaistatic.com`,
and `.oaiusercontent.com` for the Codex backend. These are domain suffixes,
not a claim that arbitrary URLs on those domains are inaccessible.
Web search is disabled, and source-hosting and package-mirror domains are
not allowed. Verification-role shell commands have networking disabled;
outer model requests and the trusted local evaluator are separate.
Official vulnerable and patched PoC execution containers use network-none.

Solver starts/resumes and every verification role explicitly pin
`web_search="disabled"` on the CLI. Hosted search is separate from command
networking ([official search configuration](https://learn.chatgpt.com/docs/web-search#configure-local-web-search)).
The host additionally checks structural search events in solver logs and a
private, per-service verifier latch. A process-generation change, malformed
policy evidence, or a verifier event-parser gap prevents a valid score.
Observed search retires the experiment as infrastructure-invalid, not a
model failure. Before scoring, the solver stops and verification must become
quiescent; sealing then blocks further verifier model calls. These checks do
not change GV claims, obligations, or judgments. Configuration conformance
and an unviolated observed-event guard do not prove a complete internal tool
registry or the absolute absence of unobservable activity.

[CyberGym FAQ Q1](https://github.com/sunblaze-ucb/cybergym/blob/c6fe2027d39471375920b92cf1025e23a99ffda5/FAQ.md)
permits controlled networking with disclosure and trajectory review.
Report this network configuration and check trajectories for patch/PoC
lookups; a domain proxy alone does not prove the absence of shortcuts.
