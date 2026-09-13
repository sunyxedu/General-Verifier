# Cyber evaluation settings

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
