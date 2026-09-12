# Complete responses with incomplete billing metadata

The Codex backend now separates a narrow billing-counter failure from a valid
role response. This does not change the Compiler, Critic, Oracle, executor or
Judger, and does not grant a verification verdict from CLI success.

The [official JSONL example](https://learn.chatgpt.com/docs/non-interactive-mode#make-output-machine-readable)
contains a thread, turn, item events, and a terminal completion with four usage
counters. It does not establish that missing or malformed counters are valid.
The recovery policy below is our conservative engineering boundary, not a
claim that OpenAI guarantees compatibility with arbitrary event dialects.

Recovery is permitted only when all conditions hold:

- The process exited 0 without a provider failure. Usage failed solely on a
  known counter field or `cached_input_tokens > input_tokens`; absent/non-object
  usage, ambiguous threads, partial streams and malformed JSON remain failures.
- Strict JSONL has one ordered thread/turn, unique item IDs, matched item
  lifecycles, no unfinished items, one final agent message as the last item,
  and exactly one completion with no trailing events.
- The original role-specific capability policy passes. Authority compilation
  still rejects tools and unknown events; its sole known startup notice is
  allowed only exactly as before and at most once in the recovery parser.
- The output file is readable and nonempty, exactly matches the terminal
  message after trimming outer whitespace, and passes the unchanged role
  schema. Valid negative responses stay negative.

Then the original response is returned without another model generation.
The attempt is recorded as `succeeded` (the role response succeeded), but
`usage_status=invalid`, `usage_complete=false`, and **all token counters remain
null**. Existing cost gates still report incomplete. A contract's downstream
cache hit cannot turn that source setup into a complete cost measurement.
Schema failures still use the existing retry budget; structural or capability
failures still reject the response.

An additive `usage_error_code` records only a bounded parser category, such as
`invalid_counter:reasoning_output_tokens`. Raw illegal values, prompts and
tool output are not added to the usage ledger. Oversized-number or deeply
nested JSON parser failures are also recorded as failed attempts instead of
escaping before accounting.

The frozen gcode attempt lasting 445.47 seconds cannot be retroactively
recovered: its raw stream was not retained, so the precise failure and the
eligibility conditions cannot be established. Its cost stays unknown. The
new fixture tests establish avoided retries for the defined counter-only
case, **not** a measured 10x end-to-end or dollar-cost improvement.
