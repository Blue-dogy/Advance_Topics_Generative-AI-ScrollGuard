# Advance_Topics_Generative-AI-ScrollGuard

ScrollGuard benchmark (v2): 45 smartphone-usage cases for testing whether an agent can tell intentional use from mindless scrolling and pick one of `no_intervention` / `gentle_prompt` / `ask_context`.

ScrollGuard 벤치마크(v2): 스마트폰 사용 상황 45개를 보고 에이전트가 의도적 사용과 무의식적 스크롤을 구분해 `no_intervention` / `gentle_prompt` / `ask_context` 중 하나를 고르는지 평가합니다.

## File dependency graph (English)

![ScrollGuard file dependency graph (English)](docs/file_graph_en.png)

## 파일 의존 관계 그래프 (한국어)

![ScrollGuard 파일 의존 관계 그래프 (한국어)](docs/file_graph_ko.png)

## Quick start / 실행 방법

```bash
pip install pandas numpy openai anthropic
python build_benchmark.py                 # 1. build the 45 cases / 사례 45개 생성
python check_leakage.py                   # 2. leakage check / 정답 누출 점검
python run_agent.py --model <model> --runs 3 --out predictions_agent.csv   # 3. run the agent / 에이전트 실행
python evaluate.py --pred predictions_agent.csv --confusion               # 4. score / 채점
python agreement.py annot_a.csv annot_b.csv                               # 6. label agreement / 라벨 일치도
```

See the technical report PDFs (English / 한국어) for the full design, metrics and baselines.
설계, 지표, 기준선에 대한 자세한 내용은 기술 보고서 PDF(영문/한글)를 참고하세요.
