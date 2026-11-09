# Main runs (v3)

Prompts and runners as of the merged PRs up to #22: field descriptions (#11, #12), reason-first output (#15), parser fixes (#16, #17, #20), and the same action definitions in both tracks (#18, default `aligned` prompt). English inputs, LM Studio, temperature 0.3.

| File | Model | Track | Runs |
|---|---|---|---|
| `predictions_qwen_A_v3.csv` / `.jsonl` | Qwen2.5-7B-Instruct | A | 3 |
| `predictions_qwen_B_v3.csv` / `.jsonl` | Qwen2.5-7B-Instruct | B | 1 |
| `predictions_exaone_A_v3.csv` / `.jsonl` | EXAONE-3.5-7.8B-Instruct | A | 3 |
| `predictions_exaone_B_v3.csv` / `.jsonl` | EXAONE-3.5-7.8B-Instruct | B | 1 |

All four runs: `evaluate_all.txt` (81 cases) and `evaluate_all_agreed.txt` (`--agreed-only`, 65 cases). The EXAONE files were re-parsed with the parser from #25 and #26; two track A timeouts were rerun with `--resume`. Scores: `evaluate_qwen.txt` (all 81 cases) and `evaluate_qwen_agreed.txt` (`--agreed-only`, the 65 cases where both annotators chose the final label).

```bash
python evaluate.py --inputs benchmark_inputs_en.csv --meta benchmark_meta_en.csv --pred docs/results/v3/predictions_qwen_A_v3.csv docs/results/v3/predictions_qwen_B_v3.csv --confusion --sensitivity
```
