'''
sample_cases.py — 샘플 10개 추출
사용 시간이 짧은 그룹에서 4개, 중간 그룹에서 3개, 긴 그룹에서 3개를 뽑아
sample_cases.csv로 저장
'''

import pandas as pd

# 데이터 불러오기
df = pd.read_csv("micro_datasetA.csv")

# 사용 시간 기준으로 3개 그룹 생성
df["usage_group"] = pd.qcut(
    df["Usage_sum_duration"].rank(method="first"),
    q=3,
    labels=["low", "medium", "high"]
)

# 초 단위 사용 시간을 분 단위로 변환
df["Usage_sum_minutes"] = df["Usage_sum_duration"] / 60

# 각 그룹에서 샘플 추출
sample_sizes = {
    "low": 4,
    "medium": 3,
    "high": 3
}

samples = []

for group, n in sample_sizes.items():
    group_df = df[df["usage_group"] == group]
    samples.append(
        group_df.sample(n=min(n, len(group_df)), random_state=42)
    )

result = pd.concat(samples)

# 벤치마크 탐색에 유용한 열만 선택
columns = [
    "user_id",
    "es_questionnaire_id",
    "Usage_sum_minutes",
    "Usage_num_int_all",
    "Usage_median_duration",
    "Usage_quadratic",
    "Nonusage_sum_duration",
    "valence",
    "usage_group"
]

result[columns].to_csv("sample_cases.csv", index=False)

print("추출된 샘플 수:", len(result))
print(result[columns].to_string(index=False))
print("\nsample_cases.csv 저장 완료")
