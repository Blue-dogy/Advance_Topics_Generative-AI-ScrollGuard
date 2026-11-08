"""
run_agent.py (v2.1) - 벤치마크 입력만 LLM에 넣고 예측을 '사례마다 즉시' 저장
  로컬(LM Studio): python run_agent.py --model exaone-3.5-7.8b-instruct --runs 1 --out predictions_test.csv
  포트가 다르면:   --base-url http://localhost:1235/v1
  Anthropic API:   --provider anthropic --model <모델명>
  파이프라인 점검: --mock
  중간에 멈췄다면 같은 명령에 --resume 을 붙이면 이어서 진행
  시험 실행: --limit 10 또는 --cases S001,S039 (본 실행 전에 판단 근거를 먼저 확인)
  원문 응답은 --out과 같은 이름의 .jsonl에 저장된다. 파서를 고친 뒤에는 --reparse로 다시 실행 없이 재해석
"""
import argparse, csv, json, os, random, re, time
import pandas as pd

SYSTEM_PROMPT = """너는 ScrollGuard, 사용자의 스마트폰 사용 맥락을 보고 개입 여부를 정하는 '의도 인식' 에이전트다.
사용 시간만으로 판단하지 말고(잠금 앱처럼 맥락을 못 보는 방식은 피한다), 의도적 사용을 방해하지 않으면서
자동조종(무의식적 스크롤) 상태일 때만 부드럽게 개입한다.

반드시 아래 셋 중 정확히 하나를 고른다.
- no_intervention: 의도적이거나 정상적인 사용. 아무것도 하지 않는다.
- gentle_prompt: 자동조종 사용으로 판단. 강제 차단 없이 화면을 흑백으로 서서히 바꾸고(Grayscale Fade) 작은 팝업이나 음성(TTS)으로 지금 하려는 일을 확인한다. 사용자는 언제든 계속할 수 있다.
- ask_context: 신호가 충돌하거나 판단에 필요한 정보가 부족. 사용자에게 지금 하려는 일을 먼저 묻는다.
불필요한 개입은 신뢰를 해치므로 비용이 크지만, 명백한 자동조종을 놓치는 것도 비용이다.

출력은 JSON 한 줄만. 근거를 먼저 쓰고, 그 근거에서 나오는 행동을 마지막에 쓴다: {"reason": "<근거 1~2문장>", "action": "<위 셋 중 하나>"}"""

FIELDS = [("time_of_day", "현재 시각"), ("entry_trigger", "진입 계기"), ("app_type", "앱 유형"),
          ("user_stated_goal", "사용자가 선언한 목표"), ("user_state", "사용자 상태"), ("recent_context", "최근 맥락(관찰 기록)"),
          ("usage_minutes", "최근 60분 사용 시간(분)"), ("usage_sessions", "최근 60분 세션 수"),
          ("median_session_sec", "최근 60분 중앙 세션 길이(초)"), ("usage_quadratic", "세션 몰림 정도(0~1): 1시간 사용이 긴 세션 하나에 몰린 정도, 1 = 세션 1개, 주의 집중도가 아님"),
          ("swipe_interval_cv", "스와이프 간격 변동계수 CV (0 = 완전히 일정; 약 0.3 이하 = 매우 일정; 약 1 이상 = 불규칙)"), ("avg_dwell_sec", "콘텐츠(게시물·영상) 하나에 머문 평균 시간(초)"),
          ("self_reported_mood", "기분 자기보고(1~6, 높을수록 좋음)"), ("typical_usage_minutes", "이 사용자의 평소 1시간 사용량 중앙값(분)")]


SYSTEM_PROMPT_EN = """You are ScrollGuard, an intent-aware agent that decides whether to intervene based on the context of a user's smartphone use.
Do not judge by usage time alone (avoid the context-blind approach of lock apps). Do not disturb intentional use,
and intervene gently only when the user appears to be on autopilot (mindless scrolling).

You must choose exactly one of the following three actions.
- no_intervention: Intentional or normal use. Do nothing.
- gentle_prompt: Judged as autopilot use. Without forcing a block, fade the screen to grayscale and show a small pop-up or TTS check-in about what the user wants to do. The user can always continue.
- ask_context: Signals conflict or information needed to decide is missing. Ask the user what they are trying to do first.
Unnecessary intervention damages trust and is costly, but missing clear autopilot use is also costly.

Output only one line of JSON. Write the reason first, then the action that follows from it: {"reason": "<1-2 sentences of reasoning>", "action": "<one of the three above>"}"""

FIELDS_EN = [("time_of_day", "Current time"), ("entry_trigger", "Entry trigger"), ("app_type", "App type"),
             ("user_stated_goal", "User-stated goal"), ("user_state", "User state"), ("recent_context", "Recent context (observation log)"),
             ("usage_minutes", "Usage in the last 60 minutes (min)"), ("usage_sessions", "Number of sessions in the last 60 minutes"),
             ("median_session_sec", "Median session length in the last 60 minutes (sec)"), ("usage_quadratic", "Session concentration (0-1): share of the hour's use packed into one long session, 1 = a single session; not a measure of attention"),
             ("swipe_interval_cv", "Swipe interval variability, CV (0 = perfectly regular; about 0.3 or lower = very regular; about 1 or higher = irregular)"),
             ("avg_dwell_sec", "Average time on each item (post or video) before swiping (sec)"),
             ("self_reported_mood", "Self-reported mood (1-6, higher = better)"), ("typical_usage_minutes", "This user's typical usage per hour, median (min)")]

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
        model=a.model, max_tokens=300, temperature=a.temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content

def _objects(text):
    """문장 안의 { ... } 덩어리를 괄호 짝으로 하나씩 꺼낸다"""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]

def parse(text):
    """코드 블록, 앞뒤 설명, JSON 여러 개, 끝의 쉼표 같은 흔한 형식 실수를 견디며 첫 번째 action JSON을 찾는다"""
    t = (text or "").replace("```json", "").replace("```", "")
    for cand in _objects(t):
        for c in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                d = json.loads(c)
            except Exception:
                continue
            if isinstance(d, dict) and "action" in d:
                return str(d.get("action", "")).strip(), str(d.get("reason", "")).strip()
    return "PARSE_ERROR", (text or "")[:200].replace("\n", " ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="benchmark_inputs.csv"); ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=""); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--provider", choices=["local", "anthropic"], default="local")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--timeout", type=float, default=120, help="사례 1건 응답 대기 한도(초)")
    ap.add_argument("--temperature", type=float, default=0.3, help="0이면 3회 실행 결과가 같아져 일관성 측정이 무의미")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko", help="프롬프트/필드 언어 (입력 CSV 언어와 맞출 것)")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="시험 실행: 앞에서 N개 사례만")
    ap.add_argument("--cases", default="", help="시험 실행: 쉼표로 구분한 case_id만 (예: S001,S039)")
    ap.add_argument("--reparse", action="store_true", help="모델을 다시 부르지 않고 저장된 원문(.jsonl)을 현재 파서로 다시 해석해 --out을 새로 쓴다")
    a = ap.parse_args()
    raw_path = os.path.splitext(a.out)[0] + ".jsonl"   # 모델 원문 응답: 파서를 고쳐도 다시 실행하지 않고 --reparse로 재채점

    if a.reparse:
        rows = [json.loads(x) for x in open(raw_path, encoding="utf-8")]
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["case_id", "run", "action", "reason"])
            for r in rows:
                act, why = (r["action"], r["reason"]) if r.get("raw") is None else parse(r["raw"])
                w.writerow([r["case_id"], r["run"], act, why])
        print(f"재해석 완료: {len(rows)}건 -> {a.out}")
        return

    df = pd.read_csv(a.inputs); df["user_stated_goal"] = df["user_stated_goal"].fillna("")
    if a.cases:
        df = df[df.case_id.isin([c.strip() for c in a.cases.split(",")])]
    if a.limit:
        df = df.head(a.limit)
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
        open(raw_path, "w").close()

    total, n, fails, t_start = len(df) * a.runs, len(done), 0, time.time()
    for run in range(a.runs):
        for r in df.to_dict("records"):
            if (r["case_id"], run) in done:
                continue
            t0 = time.time(); raw = None
            if a.mock:
                act = random.Random(f"{run}{r['case_id']}").choice(["no_intervention", "gentle_prompt", "ask_context"]); why = "mock"
                raw = json.dumps({"reason": why, "action": act})
            else:
                try:
                    raw = call_llm(SYSTEM_PROMPT_EN if a.lang == "en" else SYSTEM_PROMPT, build_user_message(r, a.lang), a)
                    act, why = parse(raw); fails = 0
                except Exception as e:  # 타임아웃/서버 오류: 기록하고 계속, 연속 3회면 중단
                    act, why, fails = "API_ERROR", f"{type(e).__name__}: {str(e)[:150]}", fails + 1
            with open(a.out, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([r["case_id"], run, act, why])
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"case_id": r["case_id"], "run": run, "action": act, "reason": why, "raw": raw}, ensure_ascii=False) + "\n")
            n += 1
            eta = (time.time() - t_start) / max(n - len(done), 1) * (total - n) / 60
            print(f"[{n}/{total}] run{run} {r['case_id']} -> {act} ({time.time()-t0:.1f}s, 남은 약 {eta:.1f}분)", flush=True)
            if fails >= 3:
                print("\n연속 3회 실패: LM Studio에서 모델이 로드/응답 중인지 확인하세요. 지금까지 결과는", a.out, "에 저장됨. --resume으로 이어서 가능")
                return
    print("저장:", a.out)

if __name__ == "__main__":
    main()
