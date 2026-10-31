"""
run_agent.py (v2.1) - 벤치마크 입력만 LLM에 넣고 예측을 '사례마다 즉시' 저장
  로컬(LM Studio): python run_agent.py --model exaone-3.5-7.8b-instruct --runs 1 --out predictions_test.csv
  포트가 다르면:   --base-url http://localhost:1235/v1
  Anthropic API:   --provider anthropic --model <모델명>
  파이프라인 점검: --mock
  중간에 멈췄다면 같은 명령에 --resume 을 붙이면 이어서 진행
"""
import argparse, csv, json, os, random, re, time
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


SYSTEM_PROMPT_EN = """You are ScrollGuard, an intent-aware agent that decides whether to intervene based on the context of a user's smartphone use.
Do not judge by usage time alone (avoid the context-blind approach of lock apps). Do not disturb intentional use,
and intervene gently only when the user appears to be on autopilot (mindless scrolling).

You must choose exactly one of the following three actions.
- no_intervention: Intentional or normal use. Do nothing.
- gentle_prompt: Judged as autopilot use. Show a gentle notification or check-in message without forcing a block.
- ask_context: Signals conflict or information needed to decide is missing. Ask the user what they are trying to do first.
Unnecessary intervention damages trust and is costly, but missing clear autopilot use is also costly.

Output only one line of JSON: {"action": "<one of the three above>", "reason": "<1-2 sentences of reasoning>"}"""

FIELDS_EN = [("time_of_day", "Current time"), ("entry_trigger", "Entry trigger"), ("app_type", "App type"),
             ("user_stated_goal", "User-stated goal"), ("recent_context", "Recent context"),
             ("usage_minutes", "Usage in the last 60 minutes (min)"), ("usage_sessions", "Number of sessions"),
             ("median_session_sec", "Median session length (sec)"), ("usage_quadratic", "Usage concentration (0-1)"),
             ("swipe_interval_cv", "Swipe interval coefficient of variation (lower = more uniform)"),
             ("avg_dwell_sec", "Average dwell per item (sec)")]

def build_user_message(row, lang="ko"):
    en = lang == "en"
    lines = []
    for col, name in (FIELDS_EN if en else FIELDS):
        v = row[col]
        lines.append(f"- {name}: " + (("(none)" if en else "(없음)") if pd.isna(v) or v == "" else str(v)))
    return ("Decide the action for the following usage situation.\n" if en else "다음 사용 상황에 대해 행동을 결정하라.\n") + "\n".join(lines)

def call_llm(system, user, a):
    if a.provider == "anthropic":
        import anthropic  # ANTHROPIC_API_KEY 필요
        msg = anthropic.Anthropic(timeout=a.timeout).messages.create(
            model=a.model, max_tokens=300, system=system, messages=[{"role": "user", "content": user}])
        return msg.content[0].text
    from openai import OpenAI  # pip install openai
    client = OpenAI(base_url=a.base_url, api_key="lm-studio", timeout=a.timeout, max_retries=0)
    r = client.chat.completions.create(
        model=a.model, max_tokens=150, temperature=a.temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content

def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)); return str(d.get("action", "")).strip(), str(d.get("reason", "")).strip()
    except Exception:
        return "PARSE_ERROR", text[:200].replace("\n", " ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="benchmark_inputs.csv"); ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=""); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--provider", choices=["local", "anthropic"], default="local")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--timeout", type=float, default=90, help="사례 1건 응답 대기 한도(초)")
    ap.add_argument("--temperature", type=float, default=0.3, help="0이면 3회 실행 결과가 같아져 일관성 측정이 무의미")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko", help="프롬프트/필드 언어 (입력 CSV 언어와 맞출 것)")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.inputs); df["user_stated_goal"] = df["user_stated_goal"].fillna("")
    done = set()
    if a.resume and os.path.exists(a.out):
        old = pd.read_csv(a.out)
        bad = old.action.isin(["API_ERROR"])           # 절전·타임아웃으로 실패한 건은 다시 실행
        old = old[~bad]; old.to_csv(a.out, index=False, encoding="utf-8-sig")
        done = set(zip(old.case_id, old.run.astype(int)))
        print(f"이어서 진행: 완료 {len(done)}건, 실패 {int(bad.sum())}건은 다시 실행")
    else:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["case_id", "run", "action", "reason"])

    total, n, fails, t_start = len(df) * a.runs, len(done), 0, time.time()
    for run in range(a.runs):
        for r in df.to_dict("records"):
            if (r["case_id"], run) in done:
                continue
            t0 = time.time()
            if a.mock:
                act = random.Random(f"{run}{r['case_id']}").choice(["no_intervention", "gentle_prompt", "ask_context"]); why = "mock"
            else:
                try:
                    act, why = parse(call_llm(SYSTEM_PROMPT_EN if a.lang == "en" else SYSTEM_PROMPT, build_user_message(r, a.lang), a)); fails = 0
                except Exception as e:  # 타임아웃/서버 오류: 기록하고 계속, 연속 3회면 중단
                    act, why, fails = "API_ERROR", f"{type(e).__name__}: {str(e)[:150]}", fails + 1
            with open(a.out, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([r["case_id"], run, act, why])
            n += 1
            eta = (time.time() - t_start) / max(n - len(done), 1) * (total - n) / 60
            print(f"[{n}/{total}] run{run} {r['case_id']} -> {act} ({time.time()-t0:.1f}s, 남은 약 {eta:.1f}분)", flush=True)
            if fails >= 3:
                print("\n연속 3회 실패: LM Studio에서 모델이 로드/응답 중인지 확인하세요. 지금까지 결과는", a.out, "에 저장됨. --resume으로 이어서 가능")
                return
    print("저장:", a.out)

if __name__ == "__main__":
    main()
