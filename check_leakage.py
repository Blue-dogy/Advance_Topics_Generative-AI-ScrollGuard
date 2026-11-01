"""입력 파일에 정답을 직접 서술하는 표현이 있는지 점검 (v1의 문제 재발 방지)"""
import re, pandas as pd
BANNED = ["무의식","의도적","비의도","목적을 잊","멈추기 어렵","좀비","중독","습관적","알 수 없","모른다","불명","정보는 없","개입","unintended","intentional"]
df = pd.read_csv("benchmark_inputs.csv")
hits = [(r.case_id, w) for r in df.itertuples() for w in BANNED if w in f"{r.recent_context} {r.user_stated_goal} {r.user_state}"]
print("누출 의심 표현:", hits if hits else "없음")
