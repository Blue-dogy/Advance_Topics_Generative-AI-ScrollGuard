# Action-definition ablation (track A, Qwen2.5-7B-Instruct)

Same 30 cases (15 `ask_context`, 8 `no_intervention`, 7 `gentle_prompt`), one run each, temperature 0.3.

| Variant | `--prompt-variant` | Description |
|---|---|---|
| aligned | `aligned` (default) | Action definitions identical to track B and the annotation guide. Used for the main runs. |
| distinct | `distinct` | Actions separated by judgment state, friction and whether it is a question, plus one sentence on when asking helps. |

The decision rule was fixed before the results: adopt `distinct` if it asks more without intervening more; differences of 1-2 cases are not interpreted.

Result (`summary.txt`): `distinct` asked on 2 of 15 `ask_context` cases (aligned 0) but intervened on 3 of 8 `no_intervention` cases (aligned 0). Not adopted. `before18` was not run, and each variant ran once, so run-to-run noise is not measured.

```bash
python ablation.py docs/results/ablation/ablation_2_aligned.csv docs/results/ablation/ablation_3_distinct.csv
```
