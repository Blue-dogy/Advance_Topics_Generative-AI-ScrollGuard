"""라벨 신뢰도: 독립 라벨러(annotation_sheet*.csv를 채운 파일)와 설계자 라벨, 라벨러 간 Cohen's kappa
  python agreement.py annot_a.csv annot_b.csv      (라벨러 1명이면 파일 1개만 넣어도 된다)
엑셀·구글 시트에서 다시 저장해 행 전체가 따옴표 하나로 묶인 파일도 자동으로 풀어서 읽는다."""
import io, sys, itertools, numpy as np, pandas as pd

ACTIONS = {"no_intervention", "gentle_prompt", "ask_context"}

def kappa(a, b):
    labs = sorted(set(a) | set(b)); po = np.mean(a == b)
    pe = sum((np.mean(a == l) * np.mean(b == l)) for l in labs)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "annotator_action" not in df.columns or df["annotator_action"].isna().all():   # 행 전체가 따옴표로 묶인 경우
        lines = []
        for ln in open(path, encoding="utf-8-sig").read().splitlines():
            if not ln.strip():
                continue
            if ln.startswith('"') and ln.endswith('"'):
                ln = ln[1:-1].replace('""', '"')
            lines.append(ln)
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    s = df.set_index("case_id").annotator_action.astype(str).str.strip()
    bad = s[~s.isin(ACTIONS)]
    if len(bad):
        print(f"[주의] {path}: 세 행동이 아닌 값 {len(bad)}개 (불일치로 계산): {dict(bad.head(5))}")
    return s

lab = pd.read_csv("benchmark_labels.csv").set_index("case_id").expected_action
ann = {p: load(p).reindex(lab.index).fillna("MISSING") for p in sys.argv[1:]}
for p, s in ann.items():
    print(f"{p} vs 설계자 라벨: 일치 {np.mean(s == lab):.2f}, kappa {kappa(s.values, lab.values):.2f}")
    print(pd.crosstab(lab.rename("설계자"), s.rename("라벨러")).to_string(), "\n")
for (p, s), (q, t) in itertools.combinations(ann.items(), 2):
    print(f"{p} vs {q}: 일치 {np.mean(s == t):.2f}, kappa {kappa(s.values, t.values):.2f}")
print("해석 참고: kappa 0.6 이상이면 '상당한 일치', 0.4 미만이면 라벨 기준 재정의 필요")
