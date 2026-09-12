# Cold adapter diagnostic — 14a4fa1

**Not a no-loss pass, and not 10x.** One ordinary Terminal-Bench negative
candidate was measured with GPT-5.6-Luna, medium effort, standard service.
The final verdict and rank match, but a final-obligation decision and a
supporting claim's status differ. No production rollout follows from this run.

## Measurement

Candidate: `extract-moves-from-video__nFHKtnx`. It has an empty final response
and empty captured world. Only sealed inputs were copied from the historical
scored directory; extraction was fresh. No previous certificate/evidence or
benchmark gold was replayed. This is not a positive-example or media-quality
test and cannot establish Best-of-N ranking preservation.

Both arms load the same complete, independently admitted compiled bundle.
Their Oracle, executor and Judger runs use separate fresh processes/states.
The source and bills passed final admission. This controls upstream compilation
drift; downstream Oracle/executor/Judger sampling still varies, so one pair is
not a reliable average treatment-effect estimate. This is **not the ordinary
full-E2E scoring schedule** or an acceptance substitute.

| Charged metric, including complete common setup | Control | Treatment | Treatment / control |
| --- | ---: | ---: | ---: |
| Stage-process wall seconds | 896.788 | 581.097 | 0.648 |
| Started model attempts | 40 | 31 | 0.775 |
| Uncached input tokens | 1,098,456 | 652,183 | 0.594 |
| Cached input tokens | 3,821,824 | 2,311,168 | 0.605 |
| Output tokens | 89,807 | 62,174 | 0.692 |

Wall improved **1.543x** (35.2% less time). Replay-only wall was 647.118 vs
331.427 seconds; quoting that roughly 1.95x ratio as total latency would omit
the common setup's 249.670 seconds and 10 calls. With this fixed setup, even
zero replay time would cap improvement at roughly 3.59x.

Physical work was 61 attempts, 1,604,906 uncached input tokens, 6,022,144
cached input tokens and 140,184 output tokens. Setup is counted once
physically and once in each analytically charged arm; failed/schema attempts
are retained. Whole coordinator makespan was 1,228.279 seconds. Provider
prefix caching was not reset; the cached-token charges above remain explicit.
No Luna monetary rate or actual currency bill has been verified here.

## Result difference

Both final verdicts are `refuted`, rank keys `[0, 0]`, with identical final
contracts. The requirement for at least one bash tool call is `refuted` in
control but `verified` in treatment. Artifact-level review found two separate
causes:

1. Both ledgers contain 147 calls with storage label `payload.tool="exec"`.
   The evaluation's `replay_lav.convert_feed` accepts source calls named
   `bash`, then encodes them as `exec` wrappers around `tools.exec_command`.
   Control's exact storage-label test incorrectly treats this conversion as
   evidence of no bash calls. Treatment parses the wrapper and finds a
   command, agreeing with that converter's source semantics, but its proof
   does not bind a trusted converter profile/digest. This is not a general
   proof that arbitrary `exec` wrappers are bash calls. Both methods here
   were model-generated custom procedures with fresh Judgers, not typed
   ledger adapters or accepted-method-cache hits.
2. The support claim says the delivered answer contains no transcribed move
   sequence. Both sealed `final.txt` files are empty. Control eventually
   checks that complete empty answer and returns `verified`; treatment binds
   the subject to the absent `/app/solution.txt` instead, ending `partial`.
   This is downstream referent drift, not a different compiled contract.

The general repair direction is trusted, version-bound ingestion metadata
for source tool identity/capability and structured command arguments, plus
explicit subject binding. No task-name or wrapper-string heuristic should
stand in for that provenance. These fixes have not been implemented by this
diagnostic, and the measured savings cannot all be credited to adapters.

The other evaluated final outcomes agree after excluding explanation wording.
Only the initial five final obligations were evaluated under the unchanged
production scheduling behavior; later obligations remain explicitly marked
unevaluated in both certificates. No hidden checks were added or dropped by
this experiment. Matching final labels therefore cannot establish full result
equivalence or quality preservation.

## Reproduction records

- Frozen source: `experiments/terminal_bench/results/gv-cold-source.xijnNV`
- Source fingerprint: `6197ccdd2348e053f9e6bb3d8b8991ff6090c70bfa1de4e2004633cb91cdc613`
- Source input seal: `dc2d502bcb420af10ad6219571c203bb1748e79b9a151f83cf67cf2cbe126ba6`
- Output: `experiments/terminal_bench/results/gv-cold-runs.Zwt2OT/video-negative`
- `request.json` fixes the source, runtime and IDs; `*-process.json` retains
  commands, process timing and result pins; `diagnostic-result.json` contains
  full charged/physical vectors and equivalence fields. All raw setup/replay
  ledgers, contracts and artifacts remain in that output directory.
- Execution session 27880 is terminal, exit 0: measurement complete,
  `result_equivalent=false`, `full_e2e_acceptance=false`. Do not restart it.
