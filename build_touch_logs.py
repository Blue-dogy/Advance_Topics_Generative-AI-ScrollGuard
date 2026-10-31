"""
build_touch_logs.py - 터치 트랙용 스와이프 간격 시퀀스 생성 (benchmark_inputs.csv -> touch_logs.json)

각 사례의 swipe_interval_cv(변동계수)와 avg_dwell_sec(평균 간격)를 목표값으로
로그정규분포에서 스와이프 간격(초) 시퀀스를 뽑는다. 표본의 평균·CV가 목표의 ±10% 안에 들 때까지 다시 뽑는다.
  - 개수: 최근 사용 시간 / 평균 간격 (8~40개로 제한)
  - 제스처 값이 없는 사례(지도·메신저 등 스크롤이 없는 사용)는 빈 리스트
숫자 열은 한국어·영어 입력이 같으므로 파일 하나를 두 언어가 같이 쓴다.
  python build_touch_logs.py
"""
import json
import numpy as np
import pandas as pd

SEED = 42
df = pd.read_csv("benchmark_inputs.csv")
logs = {}
for r in df.itertuples():
    rng = np.random.default_rng([SEED, int(r.case_id[1:])])     # 사례별 고정 시드: 사례가 늘어도 기존 시퀀스 불변
    if pd.isna(r.swipe_interval_cv) or pd.isna(r.avg_dwell_sec):
        logs[r.case_id] = []
        continue
    mean, cv = float(r.avg_dwell_sec), float(r.swipe_interval_cv)
    n = int(np.clip(round(r.usage_minutes * 60 / mean), 8, 40))
    sigma = np.sqrt(np.log(1 + cv ** 2)); mu = np.log(mean) - sigma ** 2 / 2
    for _ in range(100000):
        x = rng.lognormal(mu, sigma, n)
        m, c = x.mean(), x.std() / x.mean()
        if abs(m - mean) <= 0.1 * mean and abs(c - cv) <= 0.1 * cv:
            break
    else:
        raise RuntimeError(f"{r.case_id}: 목표 평균/CV에 맞는 표본을 찾지 못함")
    logs[r.case_id] = [round(float(v), 1) for v in np.maximum(x, 0.3)]

with open("touch_logs.json", "w", encoding="utf-8") as f:
    json.dump(logs, f, ensure_ascii=False, indent=0)
print("생성 완료:", len(logs), "사례 ->", "touch_logs.json",
      f"(빈 로그 {sum(1 for v in logs.values() if not v)}개)")
