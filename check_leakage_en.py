"""영어 입력에 정답을 직접 서술하는 표현이 있는지 점검"""
import pandas as pd
BANNED = ["mindless","unintended","intentional","unintentional","lost track","forgot the original","can't stop","zombie","addict","compulsive","habitual","unknown","no information","intervene","intervention","autopilot","unclear"]
df = pd.read_csv("benchmark_inputs_en.csv").fillna("")
hits = [(r.case_id, w) for r in df.itertuples() for w in BANNED if w in f"{r.recent_context} {r.user_stated_goal} {r.user_state}".lower()]
print("Suspected leakage expressions:", hits if hits else "none")
