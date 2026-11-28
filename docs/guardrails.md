# Guardrail specification (E9)

Code: `src/rag/guardrails.py`. Tests: `tests/test_guardrails.py`, `tests/test_guarded_pipeline.py`, `eval/guardrail_tests.jsonl` (run by `run_experiments.py --experiments E9`, output `results/e9_guardrails.csv`).

## Input checks (`check_input`)
| Category | Rule | Result |
|---|---|---|
| empty | blank question | block |
| oversized | more than 1000 characters | block |
| prompt_injection | 11 regular expressions: "ignore previous instructions", "reveal your system prompt", "you are now", "jailbreak", "developer mode", fake `system:` lines and similar | block |
| personal_data | Aadhaar (12 digits), PAN, card number (13 to 19 digits), email, Indian mobile number, IFSC | block |
| out_of_scope | none of 89 banking and finance words present. Skipped for a follow-up when the conversation has history. | block |

## Output checks (`check_output`)
| Category | Rule | Result |
|---|---|---|
| empty_answer | model returned nothing | block |
| correct_refusal | no relevant hits and the answer is `Not found in the documents.` | pass |
| should_refuse | no relevant hits but the model answered | block |
| missing_citation | no `[document, p.N]` | block |
| bad_citation | cites a document or page that was not retrieved | block |
| ungrounded | a number in the answer is not in the retrieved text, or fewer than half of the answer sentences have 60% of their content words in the retrieved text | block |
| ok | none of the above | pass |
A blocked answer is replaced by `Not found in the documents.`

## DoS limiter
`RateLimiter(max_requests=10, window_s=60)` per session id, sliding window. Blocked requests return category `rate_limited`.

## Limits
- The scope check is a keyword test. A question about a bank that uses none of the listed words is blocked, and an off-topic question that happens to use one (for example "bank" in another sense) is let through.
- Injection patterns are English only and fixed. A paraphrase can pass.
- The 21 cases in `eval/guardrail_tests.jsonl` were written by us after the rules, so a pass shows the rules do what we meant, not that they catch unseen attacks. Add cases from teammates who did not write the rules before reporting a detection rate.
- The grounding check is lexical. It does not understand paraphrase, so it can block a correct answer that reuses few of the source words.
