"""
evaluate.py - 기준선 + 에이전트 예측을 같은 라벨로 채점
사용:  python evaluate.py                         # 기준선만
       python evaluate.py --pred predictions_claude.csv predictions_gpt.csv
예측 파일 형식: case_id, run, action, reason   (run 여러 개면 일관성도 계산)
"""
import argparse, os
import numpy as np, pandas as pd

ACTIONS = ["no_intervention", "gentle_prompt", "ask_context"]
# 비용 행렬 COST[정답][예측]  (설계 선택값 - 보고서에 근거를 적고, 민감도 분석 권장)
# v2: 질문(ask_context)도 사용자를 멈춰 세우는 '방해'다. 의도적 사용 중에 묻는 것은 개입과 거의 같은 비용(0.8)으로 본다.
#     v1(0.5/0.4)에서는 '항상 ask_context'가 비용 점수 1위(0.65)가 되는 허점이 있었다. 모든 오답의 최대 비용을 피하는 '찍기'가 이기면 안 된다.
COST = {
    "no_intervention": {"no_intervention": 0.0, "ask_context": 0.8, "gentle_prompt": 1.0},  # 불필요 개입 = 제품 철학 위반, 불필요 질문도 방해
    "gentle_prompt":   {"gentle_prompt": 0.0, "ask_context": 0.5, "no_intervention": 0.8},  # 놓침 > 질문(질문도 흐름은 끊어 줌)
    "ask_context":     {"ask_context": 0.0, "gentle_prompt": 0.6, "no_intervention": 0.6},
}
COST_SETS = {  # --sensitivity: 가중치를 바꿔도 순위가 유지되는지 확인
    "v2(default)": COST,
    "v1_old": {"no_intervention": {"no_intervention": 0.0, "ask_context": 0.5, "gentle_prompt": 1.0},
               "gentle_prompt": {"gentle_prompt": 0.0, "ask_context": 0.4, "no_intervention": 0.8},
               "ask_context": {"ask_context": 0.0, "gentle_prompt": 0.6, "no_intervention": 0.6}},
    "miss_heavy": {"no_intervention": {"no_intervention": 0.0, "ask_context": 0.6, "gentle_prompt": 0.8},
                   "gentle_prompt": {"gentle_prompt": 0.0, "ask_context": 0.5, "no_intervention": 1.0},
                   "ask_context": {"ask_context": 0.0, "gentle_prompt": 0.6, "no_intervention": 0.6}},
    "equal": {a: {b: (0.0 if a == b else 1.0) for b in ["no_intervention", "gentle_prompt", "ask_context"]}
              for a in ["no_intervention", "gentle_prompt", "ask_context"]},   # 모든 오답 1.0 = 정확도와 같음
}
INVALID_COST = 1.0

# ---------------- 기준선 ----------------
def base_none(d):   return pd.Series("no_intervention", index=d.index)
def base_gentle(d): return pd.Series("gentle_prompt", index=d.index)
def base_ask(d):    return pd.Series("ask_context", index=d.index)
def base_threshold(d, minutes=30):
    return pd.Series(np.where(d.usage_minutes >= minutes, "gentle_prompt", "no_intervention"), index=d.index)
def autopilot_score(r):
    s = 0
    s += pd.notna(r.swipe_interval_cv) and r.swipe_interval_cv < 0.35
    s += pd.notna(r.avg_dwell_sec) and r.avg_dwell_sec < 10
    h = int(str(r.time_of_day)[:2]); s += (h >= 23 or h < 5)
    s += r.entry_trigger == "notification"
    s += r.usage_sessions >= 8
    return s
def base_feature_rule(d):  # 텍스트를 못 읽는 '최선의 규칙 기반' 상한 비교용
    return pd.Series(["gentle_prompt" if autopilot_score(r) >= 3 else "no_intervention" for r in d.itertuples()], index=d.index)
def base_swipe_cv(d, th=0.6):  # 제스처 숫자 하나만 보는 지름길. 에이전트가 이걸 넘어야 '맥락을 읽는다'고 말할 수 있다
    return pd.Series(np.where(d.swipe_interval_cv < th, "gentle_prompt", "no_intervention"), index=d.index)  # NaN -> 개입 안 함
BASELINES = {"B0_always_no_intervention": base_none, "B0_always_gentle_prompt": base_gentle,
             "B0_always_ask_context": base_ask, "B1_time_threshold_30min": base_threshold,
             "B2_feature_rule_no_text": base_feature_rule, "B3_swipe_cv_lt_0.6": base_swipe_cv}

def base_text_words(d, y):
    """B4 (진단용): 문장 속 단어만 보는 나이브 베이즈. 각 사례는 그 사례를 뺀 나머지로 학습(leave-one-out).
    정답을 보고 학습하므로 공정한 경쟁 상대가 아니라 '문장 표현에 정답 단서가 얼마나 남았는가'를 재는 진단 도구다."""
    import re
    us = d.user_state.fillna("") if "user_state" in d else pd.Series("", index=d.index)
    toks = [re.findall(r"[a-z가-힣]+", f"{g} {s} {c}".lower()) for g, s, c in zip(d.user_stated_goal.fillna(""), us, d.recent_context.fillna(""))]
    vocab = sorted({w for t in toks for w in t}); ix = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(toks), len(vocab)))
    for r, t in enumerate(toks):
        for w in t:
            X[r, ix[w]] += 1
    yv = y.reindex(d.index).values; out = []
    for r in range(len(toks)):
        tr = np.arange(len(toks)) != r; best, arg = -np.inf, ACTIONS[0]
        for a in ACTIONS:
            m = tr & (yv == a)
            if not m.any():
                continue
            cnt = X[m].sum(0) + 1.0
            score = np.log(m.sum() / tr.sum()) + (X[r] * np.log(cnt / cnt.sum())).sum()
            if score > best:
                best, arg = score, a
        out.append(arg)
    return pd.Series(out, index=d.index)

# ---------------- 지표 ----------------
def macro_f1(y, p):
    f = []
    for a in ACTIONS:
        tp = ((y == a) & (p == a)).sum(); fp = ((y != a) & (p == a)).sum(); fn = ((y == a) & (p != a)).sum()
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def cost_score(y, p, cost=COST):
    return 1 - float(np.mean([cost[a].get(b, INVALID_COST) for a, b in zip(y, p)]))

def score_run(lab, pred):
    y = lab.expected_action; p = pred.reindex(lab.index).fillna("MISSING")
    valid = p.isin(ACTIONS)
    correct = (y == p)
    cost = np.array([COST[a].get(b, INVALID_COST) for a, b in zip(y, p)])
    adv = lab.case_type.isin(["adversarial", "boundary", "counterfactual"])
    cf = lab.case_type == "counterfactual"
    # 반례 쌍: 맥락은 같고 제스처만 다르므로 두 예측이 같아야 하고(불변성), 둘 다 맞아야 한다
    pairs = lab.loc[cf, "pair_id"].str[:-1] if "pair_id" in lab else pd.Series(dtype=str)
    pg = pd.DataFrame({"pair": pairs, "p": p[pairs.index], "ok": correct[pairs.index]}).groupby("pair")
    return dict(
        accuracy=correct.mean(), macro_f1=macro_f1(y, p),
        detect_rate=correct[y == "gentle_prompt"].mean(),                       # 개입 필요 사례 탐지율
        false_intervention=(p[y == "no_intervention"] == "gentle_prompt").mean(), # 의도적 사용에 대한 불필요한 개입률
        question_burden=(p[y == "no_intervention"] == "ask_context").mean(),     # 의도적 사용에 대한 불필요한 질문률
        ask_recall=correct[y == "ask_context"].mean(),                          # 모호 사례의 맥락 확인 비율
        hard_acc=correct[adv].mean(), easy_acc=correct[~adv].mean(),
        cost_score=1 - cost.mean(), invalid_rate=1 - valid.mean(),
        cf_acc=correct[cf].mean() if cf.any() else np.nan,                       # 제스처 반례 사례 정확도
        pair_both_correct=pg.ok.all().mean() if cf.any() else np.nan,           # 쌍 두 개를 모두 맞힌 비율
        gesture_flip=(pg.p.nunique() > 1).mean() if cf.any() else np.nan,       # 제스처만 바꿨는데 판단이 바뀐 쌍 비율(낮을수록 좋음)
    ), correct.astype(float).values, p

def evaluate(lab, runs, n_boot=2000, seed=0):
    res, corr, preds = zip(*[score_run(lab, r) for r in runs])
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    c = np.mean(corr, axis=0); rng = np.random.default_rng(seed)
    boots = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n_boot)]
    out["acc_ci95"] = f"[{np.percentile(boots, 2.5):.2f}, {np.percentile(boots, 97.5):.2f}]"
    if len(runs) > 1:  # 같은 사례에 대해 run들이 같은 행동을 내는 비율
        P = pd.concat(preds, axis=1); out["consistency"] = float((P.nunique(axis=1) == 1).mean())
    else:
        out["consistency"] = np.nan
    out["n_runs"] = len(runs)
    return out, preds[0]

def tool_metrics(lab, d):
    """터치 트랙(run_agent_tools.py) 예측 파일에만 있는 tools 열로 도구 사용 행동을 요약"""
    d = d[d.case_id.isin(lab.index)].copy()
    t = d.tools.fillna("").astype(str)
    d["ctx"] = t.str.contains("get_user_context"); d["touch"] = t.str.contains("get_touch_log")
    d["ok"] = d.action.astype(str).str.strip().values == lab.expected_action.reindex(d.case_id).values
    return dict(context_check_rate=d.ctx.mean(), touch_check_rate=d.touch.mean(),
                stats_tool_rate=t.str.contains("compute_stats").mean(), mean_tool_calls=d.n_steps.mean(),
                acc_with_context=d.ok[d.ctx].mean() if d.ctx.any() else np.nan,
                acc_without_context=d.ok[~d.ctx].mean() if (~d.ctx).any() else np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="benchmark_inputs.csv"); ap.add_argument("--labels", default="benchmark_labels.csv")
    ap.add_argument("--meta", default="benchmark_meta.csv"); ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--out", default="results_summary.csv"); ap.add_argument("--confusion", action="store_true")
    ap.add_argument("--sensitivity", action="store_true", help="비용 가중치 세트별 cost_score와 순위 출력")
    a = ap.parse_args()
    inp = pd.read_csv(a.inputs).set_index("case_id")
    meta = pd.read_csv(a.meta, dtype={"pair_id": str}).set_index("case_id")
    lab = pd.read_csv(a.labels).set_index("case_id").join(meta[[c for c in ["case_type", "pair_id"] if c in meta]])
    inp = inp.loc[lab.index]
    table = {}; first = {}; tools = {}
    for name, fn in BASELINES.items():
        table[name], first[name] = evaluate(lab, [fn(inp)])
    table["B4_text_words_LOO"], first["B4_text_words_LOO"] = evaluate(lab, [base_text_words(inp, lab.expected_action)])
    for path in a.pred:
        d = pd.read_csv(path); d["run"] = d["run"] if "run" in d else 0
        runs = [g.set_index("case_id")["action"].astype(str).str.strip() for _, g in d.groupby("run")]
        name = os.path.splitext(os.path.basename(path))[0]
        table[name], first[path] = evaluate(lab, runs)
        if "tools" in d:
            tools[name] = tool_metrics(lab, d)
    res = pd.DataFrame(table).T
    cols = ["accuracy", "acc_ci95", "macro_f1", "detect_rate", "false_intervention", "question_burden",
            "ask_recall", "hard_acc", "cf_acc", "pair_both_correct", "gesture_flip", "cost_score", "invalid_rate", "consistency", "n_runs"]
    res = res[cols]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(res.apply(lambda s: s.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v)).to_string())
    if tools:
        tt = pd.DataFrame(tools).T
        print("\n[터치 트랙 도구 사용] context_check_rate=맥락 도구를 부른 비율, acc_with/without_context=맥락 확인 여부별 정확도")
        print(tt.apply(lambda c: c.map(lambda v: f"{v:.2f}")).to_string())
        res = res.join(tt)
    res.to_csv(a.out, encoding="utf-8-sig")
    if a.sensitivity:
        y = lab.expected_action
        rows = {}
        for n, p in first.items():
            p = p.reindex(lab.index).fillna("MISSING")
            rows[n if (n in BASELINES or n.startswith("B4")) else os.path.splitext(os.path.basename(n))[0]] = {k: cost_score(y, p, c) for k, c in COST_SETS.items()}
        sens = pd.DataFrame(rows).T
        print("\n[비용 가중치 민감도] 각 세트의 cost_score (괄호 = 순위, 첫 번째 run 기준)")
        print(sens.apply(lambda s: s.map(lambda v: f"{v:.2f}") + " (" + s.rank(ascending=False, method="min").astype(int).astype(str) + ")").to_string())
    if a.confusion:
        for n, p in first.items():
            print(f"\n[{n}] 행=정답, 열=예측"); print(pd.crosstab(lab.expected_action, p).reindex(index=ACTIONS, columns=ACTIONS + ["MISSING"], fill_value=0).loc[:, lambda x: x.sum() > 0].to_string())

if __name__ == "__main__":
    main()
