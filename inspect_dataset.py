'''
inspect_dataset.py — 원본 데이터 확인
CSV가 정상적으로 읽혔고, 데이터가 5,777행 × 17열이라는 점과 
각 변수의 자료형, 결측값을 확인
'''

import pandas as pd
import os

print("현재 폴더:", os.getcwd())
print("폴더 내 파일:", os.listdir())

# CSV 파일 자동 탐색
csv_files = [
    f for f in os.listdir(".")
    if f.lower().endswith(".csv")
]

if not csv_files:
    print("\n오류: 현재 폴더에 CSV 파일이 없습니다.")
else:
    for file in csv_files:
        print(f"\n{'=' * 50}")
        print("파일:", file)

        df = pd.read_csv(file)

        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        print("\n결측치:")
        print(df.isnull().sum().to_string())
