# Tool-order ablation (track B)

The same 30 cases as the action-definition ablation (15 `ask_context`, 8 `no_intervention`, 7 `gentle_prompt`), one run each. The main runs list the tools as `get_touch_log`, `compute_stats`, `get_user_context`. `--tool-order context_first` moves `get_user_context` to the top and leaves the descriptions unchanged.

Decision rule fixed beforehand: if the context check rate at least doubles (to about 20% or more), the list order is a main cause of the low context checks.

| 30 cases | Qwen default | Qwen context first | EXAONE default | EXAONE context first |
|---|---|---|---|---|
| context check rate | 0.13 | **1.00** | 0.00 | **0.03** |
| accuracy | 0.23 | 0.33 | 0.27 | 0.27 |
| detect_rate | 0.14 | 0.43 | 0.57 | 0.43 |
| false_intervention | 0.25 | 0.12 | 0.62 | 0.38 |
| ask_recall | 0.00 | 0.00 | 0.13 | 0.00 |

- **Qwen** called the tools in the listed order in every case, so the list order alone decided whether it read the context.
- **EXAONE** asked for the touch log first in 23 of 30 cases and for the context in 1, whatever the order. 8 of its cases were rerun after the parser fix in #26 (`--reparse`, then `--resume`).
- Each condition ran once at temperature 0.3, so differences of a few cases in accuracy and the other rates are not interpreted. The context check rate is the measure the decision rule was set on.

```bash
python ablation.py docs/results/v3/predictions_qwen_B_v3.csv docs/results/ablation_tool_order/ablation_B_qwen_context_first.csv
python ablation.py docs/results/v3/predictions_exaone_B_v3.csv docs/results/ablation_tool_order/ablation_B_exaone_context_first.csv
```
