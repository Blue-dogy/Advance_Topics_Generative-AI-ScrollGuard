'''
prepare_benchmark_data.py — 데이터 특성 분석
주요 변수 5개를 선택하고 평균, 중앙값, 분포 등을 확인. 
데이터의 전반적인 특성을 파악하는 단계
'''

import pandas as pd

# CSV 파일 경로: 실제 저장 위치에 맞게 수정
df = pd.read_csv("micro_datasetA.csv")

# 데이터 크기
print("Shape:", df.shape)

# 변수명과 자료형
print("\nColumns:")
print(df.dtypes)

# 데이터 5개 미리 보기
print("\nFirst 5 rows:")
print(df.head())

# 결측치 개수
print("\nMissing values:")
print(df.isnull().sum())

print('===================================')

# 실제 파일에 존재하는 변수만 선택
candidate_cols = [
    "Usage_sum_duration",
    "Usage_num_int_all",
    "Usage_median_duration",
    "Usage_quadratic",
    "valence",
]

available_cols = [c for c in candidate_cols if c in df.columns]

print("Available columns:", available_cols)

# 분석에 사용할 열만 추출
subset = df[available_cols].copy()

print(subset.describe().round(2))
print("\nMissing values:")
print(subset.isnull().sum())

print('===================================')

required = [
    "Usage_sum_duration",
    "Usage_num_int_all",
    "Usage_median_duration",
    "Usage_quadratic",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# 필수 변수에 결측치가 있는 행 제외
clean = df.dropna(subset=required).copy()

# 사용 시간 분포의 하위·중간·상위 구간
clean["usage_group"] = pd.qcut(
    clean["Usage_sum_duration"],
    q=3,
    labels=["low", "medium", "high"],
    duplicates="drop"
)

# 각 구간에서 최대 3~4개씩 선택
samples = (
    clean.groupby("usage_group", observed=True)
    .sample(n=3, random_state=42)
    .reset_index(drop=True)
)

samples.to_csv("sample_cases.csv", index=False)
print(samples)
print("\nSaved: sample_cases.csv")