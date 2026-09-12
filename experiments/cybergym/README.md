# Cyber evaluation settings

`cyber` replaces the six role profiles inside the existing GV kernel. It is
not a second verifier layered on top of standard GV.

The solver and every verification role use **GPT-5.6 Luna, max effort, fast**.
Their context policy is shared in `model_config.py`: 1,050,000 total tokens,
automatic compaction at 880,000 input tokens. Both fresh solver turns and
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

[CyberGym FAQ Q1](https://github.com/sunblaze-ucb/cybergym/blob/c6fe2027d39471375920b92cf1025e23a99ffda5/FAQ.md)
permits controlled networking with disclosure and trajectory review.
Report this network configuration and check trajectories for patch/PoC
lookups; a domain proxy alone does not prove the absence of shortcuts.
