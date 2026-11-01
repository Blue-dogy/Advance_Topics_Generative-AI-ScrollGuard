"""
check_consistency.py - 맥락 문장과 숫자 열이 서로 모순되지 않는지 점검 (영어 입력 기준)
  python check_consistency.py [benchmark_inputs_en.csv]

점검 항목
  1. 진입 계기: entry_trigger와 문장의 Entry 칸이 맞는지 (notification / search / self_open)
  2. 실행 횟수: "opened the app/messenger/browser/mail app N times"의 N이 세션 수와 같은지
  3. 사용 시간: In app 칸에 적힌 'N min'이 최근 사용 시간(usage_minutes)을 넘지 않는지
  4. 항목 수 x 평균 체류: 스크롤·재생한 항목 수에 평균 체류(초)를 곱한 값이 사용 시간을 넘지 않는지
문제가 없으면 'OK'만 출력하고, 있으면 사례별로 이유를 출력한다(종료 코드 1).
"""
import re, sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_inputs_en.csv"
df = pd.read_csv(path)
ITEM = r"(\d+) (?:short videos|videos|posts|products|articles|feed videos|product pages)"
problems = []
for r in df.itertuples():
    ctx = str(r.recent_context)
    slots = dict(p.split(": ", 1) for p in ctx.split(" | ") if ": " in p)
    entry, inapp = slots.get("Entry", ""), slots.get("In app", "")
    if set(slots) != {"Entry", "In app", "App switches", "Interactions", "Schedule/settings"}:
        problems.append((r.case_id, f"슬롯 구성 이상: {list(slots)}"))
    # 1. 진입 계기
    has_noti, has_search = "notification" in entry, entry.startswith("searched")
    want = {"notification": (True, False), "search": (False, True), "self_open": (False, False)}[r.entry_trigger]
    if (has_noti, has_search) != want:
        problems.append((r.case_id, f"entry_trigger={r.entry_trigger}인데 Entry 칸: '{entry}'"))
    # 2. 실행 횟수
    for n in re.findall(r"opened the (?:app|messenger|browser|mail app) (\d+) times", inapp):
        if int(n) != int(r.usage_sessions):
            problems.append((r.case_id, f"실행 {n}회 vs 세션 수 {r.usage_sessions}"))
    # 3. 사용 시간 ('runtime N min'과 '(N min)'은 콘텐츠 전체 길이라 제외, 'now at N min' 같은 진행 지점은 포함)
    for n in re.findall(r"(?<!\()(?<!runtime )(\d+) min(?:utes)?\b", inapp):
        if int(n) > r.usage_minutes * 1.05:
            problems.append((r.case_id, f"앱 안 {n}분 > 사용 시간 {r.usage_minutes}분"))
    for n in re.findall(r"\((\d+) min\)", inapp):
        if "continuously" in inapp and int(n) > r.usage_minutes * 1.05:
            problems.append((r.case_id, f"{n}분짜리를 끝까지 재생했다고 읽히지만 사용 시간 {r.usage_minutes}분"))
    # 4. 항목 수 x 평균 체류
    if pd.notna(r.avg_dwell_sec):
        for n in re.findall(ITEM, inapp):
            need = int(n) * float(r.avg_dwell_sec) / 60
            if need > r.usage_minutes * 1.1:
                problems.append((r.case_id, f"항목 {n}개 x 체류 {r.avg_dwell_sec}s = {need:.0f}분 > 사용 시간 {r.usage_minutes}분"))

if problems:
    for cid, why in problems:
        print(f"{cid}: {why}")
    print(f"\n문제 {len(problems)}건")
    sys.exit(1)
print("OK: 맥락 문장과 숫자 열 사이에 모순이 발견되지 않음 (", len(df), "사례 )")
