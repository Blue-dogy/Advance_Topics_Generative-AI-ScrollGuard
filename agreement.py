"""라벨 신뢰도: 독립 라벨러(annotation_sheet.csv 채운 파일)와 정답/라벨러 간 Cohen's kappa
  python agreement.py annot_friend1.csv annot_friend2.csv"""
import sys, itertools, numpy as np, pandas as pd
def kappa(a, b):
    labs = sorted(set(a) | set(b)); po = np.mean(a == b)
    pe = sum((np.mean(a == l) * np.mean(b == l)) for l in labs)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
lab = pd.read_csv("benchmark_labels.csv").set_index("case_id").expected_action
ann = {p: pd.read_csv(p).set_index("case_id").annotator_action.astype(str).str.strip().reindex(lab.index) for p in sys.argv[1:]}
for p, s in ann.items(): print(f"{p} vs 설계자 라벨: 일치 {np.mean(s == lab):.2f}, kappa {kappa(s.values, lab.values):.2f}")
for (p, s), (q, t) in itertools.combinations(ann.items(), 2): print(f"{p} vs {q}: 일치 {np.mean(s == t):.2f}, kappa {kappa(s.values, t.values):.2f}")
print("해석 참고: kappa 0.6 이상이면 '상당한 일치', 0.4 미만이면 라벨 기준 재정의 필요")
