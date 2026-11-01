"""
run_agent_tools.py - 터치 트랙: 도구를 써서 스스로 정보를 모으는 에이전트 실행 (방법 B)

run_agent.py와 달리 처음에는 시각·진입 계기·앱 유형·사용량 숫자만 보여 준다.
터치 기록, 균일성 통계, 사용자 맥락은 에이전트가 도구를 호출해야 얻는다.
  get_touch_log()          -> 최근 스와이프 간격(초) 리스트 (touch_logs.json)
  compute_stats(values)    -> n, mean, median, std, cv, min, max
  get_user_context()       -> 선언 목표, 최근 맥락
모든 모델에 같은 조건을 주려고 네이티브 tool calling 대신 'JSON 한 줄' 프로토콜을 쓴다.

  python build_touch_logs.py     (touch_logs.json이 없거나 사례가 바뀌었을 때)
  python run_agent_tools.py --lang en --inputs benchmark_inputs_en.csv --model exaone-3.5-7.8b-instruct --base-url http://localhost:1235/v1 --runs 3 --out predictions_tools_exaone_en.csv
  python run_agent_tools.py --lang en --inputs benchmark_inputs_en.csv --provider anthropic --model claude-sonnet-5-5 --runs 3 --out predictions_tools_claude_en.csv
  파이프라인 점검: --mock / 중간에 멈췄다면 같은 명령에 --resume
출력: 예측 CSV(case_id, run, action, reason, tools, n_steps) + 같은 이름의 .jsonl(대화 전체 기록)
"""
import argparse, csv, json, os, random, re, time
import numpy as np
import pandas as pd

ACTIONS = ["no_intervention", "gentle_prompt", "ask_context"]
TOOLS = ["get_touch_log", "compute_stats", "get_user_context"]

SYSTEM_EN = """You are ScrollGuard, an intent-aware agent that decides whether to intervene in a user's smartphone use.
Do not judge by usage time alone. Do not disturb intentional use, and intervene gently only when the user appears to be on autopilot (mindless scrolling).

At first you only see basic usage data. You can call tools to gather more evidence before deciding:
- get_touch_log: args {} -> the user's recent swipes: intervals (sec), speed (px/s) and direction (up = forward, down = back); empty if the app has no scrolling
- compute_stats: args {"values": [numbers]} -> n, mean, median, std, cv (coefficient of variation), min, max
- get_user_context: args {} -> the user's stated goal, current state (who, where, what kind of day), self-reported mood and recent in-app behavior

Final actions (choose exactly one):
- no_intervention: Intentional or normal use. Do nothing.
- gentle_prompt: Judged as autopilot use. Without forcing a block, fade the screen to grayscale and show a small pop-up or TTS check-in. The user can always continue.
- ask_context: Signals conflict or information needed to decide is missing. Ask the user what they are trying to do first.
Unnecessary intervention damages trust and is costly, but missing clear autopilot use is also costly.

Each reply must be exactly one line of JSON and nothing else, either a tool call:
{"tool": "<tool name>", "args": {...}}
or the final decision:
{"action": "<one of the three actions>", "reason": "<1-2 sentences>"}
You may call at most MAX_TOOLS tools."""

SYSTEM_KO = """너는 ScrollGuard, 사용자의 스마트폰 사용에 개입할지 정하는 '의도 인식' 에이전트다.
사용 시간만으로 판단하지 말고, 의도적 사용을 방해하지 않으면서 자동조종(무의식적 스크롤) 상태일 때만 부드럽게 개입한다.

처음에는 기본 사용 정보만 보인다. 결정 전에 도구를 호출해 근거를 더 모을 수 있다.
- get_touch_log: args {} -> 최근 스와이프의 간격(초), 속도(px/s), 방향(up = 다음, down = 뒤로). 스크롤이 없는 앱이면 빈 기록
- compute_stats: args {"values": [숫자들]} -> n, mean, median, std, cv(변동계수), min, max
- get_user_context: args {} -> 사용자가 선언한 목표, 현재 상태(누구인지·어디서·어떤 날인지), 기분 자기보고, 최근 앱 내 행동

최종 행동(정확히 하나):
- no_intervention: 의도적이거나 정상적인 사용. 아무것도 하지 않는다.
- gentle_prompt: 자동조종 사용으로 판단. 강제 차단 없이 화면을 흑백으로 서서히 바꾸고 작은 팝업이나 음성(TTS)으로 확인한다. 사용자는 언제든 계속할 수 있다.
- ask_context: 신호가 충돌하거나 판단에 필요한 정보가 부족. 지금 하려는 일을 먼저 묻는다.
불필요한 개입은 신뢰를 해치므로 비용이 크지만, 명백한 자동조종을 놓치는 것도 비용이다.

매 응답은 다른 말 없이 JSON 한 줄이어야 한다. 도구 호출:
{"tool": "<도구 이름>", "args": {...}}
또는 최종 결정:
{"action": "<세 행동 중 하나>", "reason": "<근거 1~2문장>"}
도구는 최대 MAX_TOOLS번까지 호출할 수 있다."""

OBS = {"en": [("time_of_day", "Current time"), ("entry_trigger", "Entry trigger"), ("app_type", "App type"),
              ("usage_minutes", "Usage in the last 60 minutes (min)"), ("usage_sessions", "Number of sessions"),
              ("median_session_sec", "Median session length (sec)"), ("usage_quadratic", "Usage concentration (0-1)"),
              ("typical_usage_minutes", "This user's typical usage per hour, median (min)")],
       "ko": [("time_of_day", "현재 시각"), ("entry_trigger", "진입 계기"), ("app_type", "앱 유형"),
              ("usage_minutes", "최근 60분 사용 시간(분)"), ("usage_sessions", "세션 수"),
              ("median_session_sec", "중앙 세션 길이(초)"), ("usage_quadratic", "사용 집중도(0~1)"),
              ("typical_usage_minutes", "이 사용자의 평소 1시간 사용량 중앙값(분)")]}

def first_message(row, lang):
    lines = [f"- {name}: {row[col]}" for col, name in OBS[lang]]
    head = "New usage situation. Gather evidence with tools if needed, then decide.\n" if lang == "en" \
        else "새 사용 상황이다. 필요하면 도구로 근거를 모은 뒤 결정하라.\n"
    return head + "\n".join(lines)

def run_tool(name, args, row, touch, lang):
    if name == "get_touch_log":
        v = touch.get(row["case_id"], [])
        if isinstance(v, list):          # 이전 형식(간격 리스트만)도 읽을 수 있게
            v = {"swipe_intervals_sec": v}
        if v.get("swipe_intervals_sec"):
            return v
        return {"swipe_intervals_sec": [], "note": "no swipe events recorded" if lang == "en" else "스와이프 기록 없음"}
    if name == "compute_stats":
        try:
            x = np.array([float(t) for t in (args or {}).get("values", [])])
        except Exception:
            return {"error": "values must be a list of numbers"}
        if len(x) == 0:
            return {"error": "empty values"}
        return {"n": int(len(x)), "mean": round(float(x.mean()), 2), "median": round(float(np.median(x)), 2),
                "std": round(float(x.std()), 2), "cv": round(float(x.std() / x.mean()), 3) if x.mean() else None,
                "min": float(x.min()), "max": float(x.max())}
    if name == "get_user_context":
        none = "(none)" if lang == "en" else "(없음)"
        return {"user_stated_goal": row["user_stated_goal"] or none, "user_state": row["user_state"] or none,
                "self_reported_mood_1to6": None if pd.isna(row["self_reported_mood"]) else row["self_reported_mood"],
                "recent_context": row["recent_context"] or none}
    return {"error": f"unknown tool '{name}'. Available: {', '.join(TOOLS)}"}

def call_llm(system, messages, a):
    if a.provider == "anthropic":
        import anthropic  # ANTHROPIC_API_KEY 필요
        msg = anthropic.Anthropic(timeout=a.timeout).messages.create(
            model=a.model, max_tokens=600, system=system, messages=messages)
        return msg.content[0].text
    from openai import OpenAI  # pip install openai
    client = OpenAI(base_url=a.base_url, api_key="lm-studio", timeout=a.timeout, max_retries=0)
    r = client.chat.completions.create(model=a.model, max_tokens=600, temperature=a.temperature,
                                       messages=[{"role": "system", "content": system}] + messages)
    return r.choices[0].message.content

def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def mock_reply(step, row, rnd):
    plan = rnd.sample([["get_touch_log", "compute_stats"], ["get_user_context"], ["get_touch_log", "compute_stats", "get_user_context"], []], 1)[0]
    if step < len(plan):
        t = plan[step]
        return json.dumps({"tool": t, "args": {"values": [1, 2, 3]} if t == "compute_stats" else {}})
    return json.dumps({"action": rnd.choice(ACTIONS), "reason": "mock"})

def run_case(row, a, touch, rnd):
    system = (SYSTEM_EN if a.lang == "en" else SYSTEM_KO).replace("MAX_TOOLS", str(a.max_tools))
    messages = [{"role": "user", "content": first_message(row, a.lang)}]
    tools_used, format_errors = [], 0
    for step in range(a.max_tools + 3):           # 도구 호출 + 형식 오류 안내 1회 + 최종 결정
        text = mock_reply(len(tools_used), row, random.Random(f"{rnd}{row['case_id']}")) if a.mock else call_llm(system, messages, a)
        messages.append({"role": "assistant", "content": text})
        d = parse(text)
        if isinstance(d, dict) and "action" in d:
            act = str(d.get("action", "")).strip()
            return act, str(d.get("reason", "")).strip(), tools_used, messages
        if isinstance(d, dict) and "tool" in d and len(tools_used) < a.max_tools:
            name = str(d.get("tool", "")).strip(); tools_used.append(name)
            result = run_tool(name, d.get("args", {}), row, touch, a.lang)
            messages.append({"role": "user", "content": "Tool result: " + json.dumps(result, ensure_ascii=False)})
            continue
        format_errors += 1
        if format_errors > 1:
            return "PARSE_ERROR", (text or "")[:200].replace("\n", " "), tools_used, messages
        remind = 'Reply with exactly one line of JSON: {"tool": ..., "args": {...}} or {"action": ..., "reason": ...}.'
        if len(tools_used) >= a.max_tools:
            remind = 'Tool limit reached. Reply now with the final decision: {"action": ..., "reason": ...}.'
        messages.append({"role": "user", "content": remind})
    return "MAX_STEPS", "no final decision", tools_used, messages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="benchmark_inputs.csv"); ap.add_argument("--out", default="predictions_tools.csv")
    ap.add_argument("--touch", default="touch_logs.json")
    ap.add_argument("--model", default=""); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--provider", choices=["local", "anthropic"], default="local")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--timeout", type=float, default=120, help="LLM 호출 1번의 응답 대기 한도(초)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tools", type=int, default=4, help="사례당 도구 호출 최대 횟수")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko", help="프롬프트 언어 (입력 CSV 언어와 맞출 것)")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.inputs)
    for c in ["user_stated_goal", "user_state", "recent_context"]:
        df[c] = df[c].fillna("")
    touch = json.load(open(a.touch, encoding="utf-8"))
    missing = set(df.case_id) - set(touch)
    if missing:
        raise SystemExit(f"{a.touch}에 없는 사례 {len(missing)}개: python build_touch_logs.py 를 먼저 실행하세요")
    trace = os.path.splitext(a.out)[0] + ".jsonl"
    cols = ["case_id", "run", "action", "reason", "tools", "n_steps"]
    done = set()
    if a.resume and os.path.exists(a.out):
        old = pd.read_csv(a.out)
        bad = old.action.isin(["API_ERROR"])
        old = old[~bad]; old.to_csv(a.out, index=False, encoding="utf-8-sig")
        done = set(zip(old.case_id, old.run.astype(int)))
        print(f"이어서 진행: 완료 {len(done)}건, 실패 {int(bad.sum())}건은 다시 실행")
    else:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(cols)
        open(trace, "w").close()

    total, n, fails, t_start = len(df) * a.runs, len(done), 0, time.time()
    for run in range(a.runs):
        for r in df.to_dict("records"):
            if (r["case_id"], run) in done:
                continue
            t0 = time.time()
            try:
                act, why, used, msgs = run_case(r, a, touch, run); fails = 0
            except Exception as e:  # 타임아웃/서버 오류: 기록하고 계속, 연속 3회면 중단
                act, why, used, msgs, fails = "API_ERROR", f"{type(e).__name__}: {str(e)[:150]}", [], [], fails + 1
            with open(a.out, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([r["case_id"], run, act, why, ">".join(used), len(used)])
            with open(trace, "a", encoding="utf-8") as f:
                f.write(json.dumps({"case_id": r["case_id"], "run": run, "action": act, "messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
            eta = (time.time() - t_start) / max(n - len(done), 1) * (total - n) / 60
            print(f"[{n}/{total}] run{run} {r['case_id']} -> {act} tools={'>'.join(used) or '-'} ({time.time()-t0:.1f}s, 남은 약 {eta:.1f}분)", flush=True)
            if fails >= 3:
                print("\n연속 3회 실패: 모델 서버가 응답 중인지 확인하세요. 지금까지 결과는", a.out, "에 저장됨. --resume으로 이어서 가능")
                return
    print("저장:", a.out, "/ 대화 기록:", trace)

if __name__ == "__main__":
    main()
