# Tool-order ablation (track B)

The same 30 cases as the action-definition ablation (15 `ask_context`, 8 `no_intervention`, 7 `gentle_prompt`), one run each. The main runs list the tools as `get_touch_log`, `compute_stats`, `get_user_context`. `--tool-order context_first` moves `get_user_context` to the top and leaves the descriptions unchanged.

Decision rule fixed beforehand: if the context check rate at least doubles (to about 20% or more), the list order is a main cause of the low context checks.

| Qwen2.5-7B, 30 cases | default order (main run) | context first |
|---|---|---|
| context check rate | 0.13 | **1.00** |
| accuracy | 0.23 | 0.33 |
| detect_rate | 0.14 | 0.43 |
| false_intervention | 0.25 | 0.12 |
| ask_recall | 0.00 | 0.00 |

Qwen called the tools in the listed order in every case. EXAONE results are pending: 8 of its 30 cases are being rerun after a parser fix (#26). Its first replies already show it asked for the touch log first in 24 of 30 cases even with the context tool listed first.

```bash
python ablation.py docs/results/v3/predictions_qwen_B_v3.csv docs/results/ablation_tool_order/ablation_B_qwen_context_first.csv
```
