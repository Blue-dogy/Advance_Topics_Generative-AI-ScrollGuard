"""
run_agent.py - 벤치마크 입력만 에이전트(LLM)에 넣고 예측을 저장 (정답 파일은 읽지 않음)
  python run_agent.py --model <모델명> --runs 3 --out predictions_claude.csv
  python run_agent.py --mock --out predictions_mock.csv     # 파이프라인 점검용(랜덤, 성능 의미 없음)
다른 LLM을 쓰려면 call_llm() 한 함수만 바꾸면 된다.
"""
import argparse, json, re, random
import pandas as pd

SYSTEM_PROMPT = """너는 ScrollGuard, 사용자의 스마트폰 사용 맥락을 보고 개입 여부를 정하는 '의도 인식' 에이전트다.
사용 시간만으로 판단하지 말고(잠금 앱처럼 맥락을 못 보는 방식은 피한다), 의도적 사용을 방해하지 않으면서
자동조종(무의식적 스크롤) 상태일 때만 부드럽게 개입한다.

반드시 아래 셋 중 정확히 하나를 고른다.
- no_intervention: 의도적이거나 정상적인 사용. 아무것도 하지 않는다.
- gentle_prompt: 자동조종 사용으로 판단. 강제 차단 없이 부드러운 알림/확인 메시지를 띄운다.
- ask_context: 신호가 충돌하거나 판단에 필요한 정보가 부족. 사용자에게 지금 하려는 일을 먼저 묻는다.
불필요한 개입은 신뢰를 해치므로 비용이 크지만, 명백한 자동조종을 놓치는 것도 비용이다.

출력은 JSON 한 줄만: {"action": "<위 셋 중 하나>", "reason": "<근거 1~2문장>"}"""

FIELDS = [("time_of_day", "현재 시각"), ("entry_trigger", "진입 계기"), ("app_type", "앱 유형"),
          ("user_stated_goal", "사용자가 선언한 목표"), ("recent_context", "최근 맥락"),
          ("usage_minutes", "최근 60분 사용 시간(분)"), ("usage_sessions", "세션 수"),
          ("median_session_sec", "중앙 세션 길이(초)"), ("usage_quadratic", "사용 집중도(0~1)"),
          ("swipe_interval_cv", "스와이프 간격 변동계수(낮을수록 일정)"), ("avg_dwell_sec", "콘텐츠당 평균 체류(초)")]

def build_user_message(row):
    lines = []
    for col, name in FIELDS:
        v = row[col]
        lines.append(f"- {name}: " + ("(없음)" if pd.isna(v) or v == "" else str(v)))
    return "다음 사용 상황에 대해 행동을 결정하라.\n" + "\n".join(lines)

# 로컬 모델 기준.
def call_llm(system, user, model, temperature=None):
    from openai import OpenAI   # pip install openai
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content

def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)); return str(d.get("action", "")).strip(), str(d.get("reason", "")).strip()
    except Exception:
        return "PARSE_ERROR", text[:200]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="benchmark_inputs.csv"); ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=""); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=None); ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.inputs, keep_default_na=True)
    df["user_stated_goal"] = df["user_stated_goal"].fillna("")
    out = []
    for run in range(a.runs):
        for r in df.to_dict("records"):
            if a.mock:
                act, why = random.Random(f"{run}{r['case_id']}").choice(["no_intervention", "gentle_prompt", "ask_context"]), "mock"
            else:
                act, why = parse(call_llm(SYSTEM_PROMPT, build_user_message(r), a.model, a.temperature))
            out.append(dict(case_id=r["case_id"], run=run, action=act, reason=why))
        print(f"run {run} 완료")
    pd.DataFrame(out).to_csv(a.out, index=False, encoding="utf-8-sig"); print("저장:", a.out)

if __name__ == "__main__":
    main()
