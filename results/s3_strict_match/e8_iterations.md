# E8 self-improving loop

Rules are chosen on the dev split only. The test split is scored twice, at the end.

| iteration | rule proposed | dev correctness |
|---|---|---|
| 0 (baseline) | - | 0.250 |
| 1 (rejected) | The model should focus on extracting the most specific and directly relevant numerical or categorical value from the context, rather than providing general or indirect information. | 0.250 |
| 2 (rejected) | The model should focus on extracting the most specific and directly relevant numerical or categorical value from the context, rather than providing general or indirect information. | 0.250 |
| 3 (rejected) | The model should focus on extracting the most specific and directly relevant numerical or categorical value from the context, rather than providing general or indirect information. | 0.250 |

Test correctness without rules: 0.500
Test correctness with kept rules (0): 0.500