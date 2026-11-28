# E8 self-improving loop

Rules are chosen on the dev split only. The test split is scored twice, at the end.

| iteration | rule proposed | dev correctness |
|---|---|---|
| 0 (baseline) | - | 0.750 |
| 1 (rejected) | The model should focus on the specific question and avoid being distracted by irrelevant details or partial information in the text. | 0.750 |
| 2 (rejected) | The model should prioritize extracting the most relevant information directly related to the question, rather than interpreting or summarizing unrelated content. | 0.750 |
| 3 (rejected) | The model should identify and respond to the exact intent of the question without incorporating or interpreting unrelated content from the text. | 0.750 |

Test correctness without rules: 1.000
Test correctness with kept rules (0): 1.000