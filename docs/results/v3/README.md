# Main runs (v3)

Prompts and runners as of the merged PRs up to #22: field descriptions (#11, #12), reason-first output (#15), parser fixes (#16, #17, #20), and the same action definitions in both tracks (#18, default `aligned` prompt). English inputs, LM Studio, temperature 0.3.

| File | Model | Track | Runs |
|---|---|---|---|
| `predictions_qwen_A_v3.csv` / `.jsonl` | Qwen2.5-7B-Instruct | A | 3 |
| `predictions_qwen_B_v3.csv` / `.jsonl` | Qwen2.5-7B-Instruct | B | 1 |

EXAONE runs will be added here. Scores: `evaluate_qwen.txt`.

```bash
python evaluate.py --inputs benchmark_inputs_en.csv --meta benchmark_meta_en.csv --pred docs/results/v3/predictions_qwen_A_v3.csv docs/results/v3/predictions_qwen_B_v3.csv --confusion --sensitivity
```
