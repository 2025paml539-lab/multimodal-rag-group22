# E8 self-improving loop

Rules are chosen on the dev split only. The test split is scored twice, at the end.

| iteration | rule proposed | dev correctness |
|---|---|---|
| 0 (baseline) | - | 0.933 |
| 1 (rejected) | The model should focus on identifying the specific numerical answer rather than restating the question or providing a formulaic explanation. | 0.933 |
| 2 (rejected) | The model should extract the exact numerical value from the text without including extraneous details or rephrasing the question. | 0.933 |
| 3 (rejected) | The model should recognize that the correct answer is a specific number or value directly mentioned in the text, rather than a descriptive statement about a calculation. | 0.933 |

Test correctness without rules: 0.733
Test correctness with kept rules (0): 0.733