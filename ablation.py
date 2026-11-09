"""
ablation.py - 프롬프트 변형끼리 같은 사례에서의 판단을 비교한다 (run_agent.py --prompt-variant 결과)
  python ablation.py ablation_1_before.csv ablation_2_aligned.csv ablation_3_distinct.csv

정답 범주별로 보는 지표
  ask_recall        정답이 ask_context인 사례에서 ask_context를 고른 비율 (실험의 주 지표)
  question_burden   정답이 no_intervention인 사례에서 ask_context를 고른 비율 (과하게 묻는지)
  false_intervention 정답이 no_intervention인 사례에서 gentle_prompt를 고른 비율
  detect_rate       정답이 gentle_prompt인 사례에서 gentle_prompt를 고른 비율
사례가 30개 안팎이라 차이 1~2건은 해석하지 않는다. 같은 사례에서 답이 바뀐 목록을 함께 본다.
"""
import os
import sys
import pandas as pd

ACTIONS = ["no_intervention", "gentle_prompt", "ask_context"]


def summary(lab, p):
    d = lab.join(p, how="inner")
    y, a = d.expected_action, d.action

    def rate(mask, act):
        return (a[mask] == act).mean() if mask.any() else float("nan")

    return {
        "n": len(d),
        "accuracy": (a == y).mean(),
        "ask_recall": rate(y == "ask_context", "ask_context"),
        "question_burden": rate(y == "no_intervention", "ask_context"),
        "false_intervention": rate(y == "no_intervention", "gentle_prompt"),
        "detect_rate": rate(y == "gentle_prompt", "gentle_prompt"),
        "asked_total": int((a == "ask_context").sum()),
        "invalid": int((~a.isin(ACTIONS)).sum()),
    }


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    lab = pd.read_csv("benchmark_labels.csv").set_index("case_id")
    preds, ctx = {}, {}
    for f in files:
        p = pd.read_csv(f)
        p = p[p.run == p.run.min()].set_index("case_id")   # 1회 실행 비교. 여러 번 돌렸으면 첫 실행만
        name = os.path.splitext(os.path.basename(f))[0]
        preds[name] = p.action
        if "tools" in p:   # 트랙 B: 맥락 도구를 부른 사례
            ctx[name] = p.tools.fillna("").astype(str).str.contains("get_user_context")
    cases = sorted(set.intersection(*(set(p.index) for p in preds.values())))
    lab = lab.loc[cases]
    print(f"공통 사례 {len(cases)}개 (정답: " + ", ".join(f"{k} {v}" for k, v in lab.expected_action.value_counts().items()) + ")\n")

    res = pd.DataFrame({k: summary(lab, p.loc[cases].rename("action")) for k, p in preds.items()}).T
    res = res.astype({c: int for c in ("n", "asked_total", "invalid")})
    if ctx:
        res["context_check_rate"] = [ctx[k].loc[cases].mean() if k in ctx else float("nan") for k in res.index]
    print(res.to_string(float_format=lambda v: f"{v:.2f}"))

    table = pd.DataFrame({k: p.loc[cases] for k, p in preds.items()})
    table.insert(0, "label", lab.expected_action)
    changed = table[table.drop(columns="label").nunique(axis=1) > 1]
    short = {"no_intervention": "NO", "gentle_prompt": "GP", "ask_context": "AC"}
    print(f"\n변형에 따라 답이 바뀐 사례 {len(changed)}개 (NO = no_intervention, GP = gentle_prompt, AC = ask_context)")
    if len(changed):
        print(changed.apply(lambda s: s.map(lambda v: short.get(v, v))).to_string())


if __name__ == "__main__":
    main()
