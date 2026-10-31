"""
make_english.py - 한국어 벤치마크를 영어 버전으로 변환 (build_benchmark.py의 CASES 순서와 1:1 대응)
입력: benchmark_inputs.csv, benchmark_meta.csv, build_benchmark.py (같은 폴더)
출력: benchmark_inputs_en.csv, benchmark_meta_en.csv, annotation_sheet_en.csv
라벨(benchmark_labels.csv)은 언어와 무관하므로 그대로 재사용한다.
"""
import re
import pandas as pd

# (user_stated_goal, recent_context, label_rationale) - build_benchmark.py의 CASES와 같은 순서
EN = [
 # ---- no_intervention (18) ----
 ("Watching lectures to prepare for the operating systems midterm", "Watching a playlist in order, switched to a notes app 3 times, search query 'OS paging lecture'", "Long, but behavior matches the stated goal"),
 ("", "Turning pages in an e-book app at a steady 1 to 3 minute interval, no notifications received", "Reading app, self-directed use"),
 ("Checking a soybean paste stew recipe", "Entered 2 search queries, watched 3 recipe videos, closed the app after the last video", "Search, goal achieved, exit"),
 ("", "Opened a map app for directions, then closed it", "Short utility use"),
 ("", "Checked a messenger notification, sent 1 reply, closed the app", "Short utility use"),
 ("", "User-set break timer of 20 minutes, {m} minutes elapsed, watching videos from subscribed channels", "Within the user's self-set time budget"),
 ("Organizing a trip photo album and sharing it with friends", "Repeatedly uploading photos, writing captions and replying to comments; feed scrolling is brief and intermittent", "Social media and long duration, but productive activity"),
 ("Watching a conference presentation video", "Watching a single video continuously with 4 pauses, switching to a document app to take notes", "Purposeful despite high usage concentration"),
 ("Checking the group chat before the team project deadline", "Alternating between a messenger and a shared document app, document edits recorded", "Many sessions, but carrying out a task"),
 ("Comparing prices of a laptop added to the cart yesterday", "Compared specs of 3 items in the cart, completed payment for 1, closed the app", "Clear task completed"),
 ("Watching reference videos for an assignment due tomorrow, then going to bed", "Repeating search, watch, note-taking; all watched video titles match the assignment topic; calendar: assignment due tomorrow 09:00", "Late night and long duration, but purposeful"),
 ("", "Playing a podcast while recording a run, screen mostly off", "No screen consumption"),
 ("Searching restaurant reviews for a weekend gathering and then making a reservation", "Searched area names 3 times, read reviews, switched to a reservation app and completed the booking", "Using social media as a search tool"),
 ("Declared directly: it is a day off, so planning to watch one movie", "Watching 1 movie continuously, user-set do-not-disturb mode is on", "Respect the user's explicit declaration"),
 ("Checking email on the commute", "In transit (location status: moving), repeatedly opening the email app, checking briefly and closing it", "Utility use while in transit"),
 ("Studying Spanish expressions with short videos", "Paused on each video, saved 8 times, switched to a quiz app 2 times, search query 'everyday Spanish expressions'", "Short-form video, but learning behavior"),
 ("", "Played a playlist in a music app, then closed the app", "Short utility use"),
 ("Checking course registration reviews for next semester", "Entered the search query 'recommended electives', read 5 posts, switched to a timetable app", "Clear information seeking"),
 # ---- gentle_prompt (17) ----
 ("", "Entered via a notification, then watched only the recommended feed continuously, no search input, no switching to other apps", "Multiple autopilot signals"),
 ("", "Past the user's bedtime goal of 24:00, continuously scrolling the same feed, no interaction with posts", "Exceeded bedtime goal plus passive consumption"),
 ("", "Opens the same app every time the screen turns on, stays under 1 minute each time, only repeats refresh actions", "Short total time, but compulsive rechecking"),
 ("Watching just one video about a lunch menu", "After the first video ended, watched 5 auto-recommended videos in a row; topics shifted to games, entertainment and sports", "Drifted away from the stated goal"),
 ("Searching for assignment materials", "After 2 searches, watched a short-video feed continuously with no further searches or app switching", "Drifted into the feed after searching"),
 ("", "Calendar: class 13:00 to 14:15, continuously scrolling a feed during class time, location: lecture building", "Passive consumption that conflicts with the schedule"),
 ("", "Repeatedly closing the app and relaunching it immediately, repeatedly refreshing the top of the same feed", "Many sessions, late night, passive"),
 ("", "Browsing only the recommendations tab continuously, no search, wishlist, cart or purchase actions", "No goal-directed actions"),
 ("", "Continuously reading a news comment list and refreshing repeatedly, no article body opened", "Passive repetitive consumption"),
 ("", "15 minutes ago the user typed 'I should stop watching now', and kept scrolling the same feed afterwards", "Behavior conflicts with the user's own stated wish"),
 ("", "Entered right after dismissing the wake-up alarm and kept scrolling the feed, first event today at 09:30 (calendar)", "Passive consumption just before a scheduled event"),
 ("", "Repeatedly unlocking the phone while using a work app, each time checking the social media home screen and closing it immediately, no notifications received", "Repeated rechecking without notifications"),
 ("A 10-minute break during studying", "Break timer set to 10 minutes, kept watching a short-video feed after the timer ended, current usage {m} minutes", "Exceeded the self-set time budget"),
 ("", "Entered via a friend's DM notification, checked the DM, then moved to the explore tab and scrolled the feed continuously", "Drifted into the feed after the task was finished"),
 ("Watch short videos for just 5 minutes", "User typed 'just 5 minutes' at the start, {m} minutes elapsed so far, no switching to other apps", "Exceeded the self-set time"),
 ("", "Entered via a notification, no pauses, likes or saves, no switching to other apps", "Multiple autopilot signals"),
 ("", "Repeatedly refreshing a post list and jumping to the top, post bodies opened only briefly, no comments or searches", "Passive repetitive checking"),
 # ---- ask_context (10) ----
 ("", "After 1 search, moved on to recommended videos; watched videos span 3 topics: cooking, games and news", "Mixed goal-directed and drifting signals"),
 ("", "Wrote 1 post and then kept scrolling the feed, the account carries a business-account tag", "Cannot distinguish work from leisure"),
 ("", "Repeatedly launching a browser, search query history is access-restricted", "Key information not accessible"),
 ("Reference for tomorrow's presentation", "Past the bedtime goal of 24:00, only 1 of 5 watched videos matches the presentation topic", "Partially matches the goal, plus exceeded budget"),
 ("Watching short videos for learning", "Only some of the watched videos are on a study topic, no saves or pauses recorded", "Behavior partially matches the stated goal"),
 ("", "Lunchtime (no calendar events), no break timer set", "Cannot distinguish a normal break from a habit"),
 ("", "Watching long videos continuously, this is the first observation of night-time viewing for this user", "No baseline history"),
 ("", "Opened several product detail pages, wishlisted 2 times, no purchase", "Cannot distinguish comparison shopping from impulse browsing"),
 ("", "Mix of reading posts and writing comments, 3 comments written", "Cannot distinguish interaction from consumption"),
 ("", "Reading news and repeatedly moving on to related articles, before the start of a work event on the calendar", "Cannot distinguish information gathering from habit"),
]

# build_benchmark.py에서 CASES 리터럴만 꺼내 한국어 원문을 얻는다 (재입력으로 인한 오타 방지)
src = open("build_benchmark.py", encoding="utf-8").read()
s = src.index("CASES = ["); e = src.index("\n]\n", s) + 3
ns = dict(nan=float("nan"), NO="no_intervention", GP="gentle_prompt", AC="ask_context",
          IN="intentional", UN="unintended", UC="unclear")
exec(src[s:e], ns); CASES = ns["CASES"]
assert len(CASES) == len(EN) == 45, (len(CASES), len(EN))

def digits(t): return sorted(re.findall(r"\d+", t.replace("{m}", "")))
for i, (c, en) in enumerate(zip(CASES, EN)):                       # 순서·내용 정합성 점검
    goal_ko, ctx_ko = c[6], c[7]
    assert (goal_ko == "") == (en[0] == ""), f"goal 불일치 case {i}"
    assert digits(goal_ko) == digits(en[0]), f"goal 숫자 불일치 case {i}"
    assert digits(ctx_ko) == digits(en[1]), f"ctx 숫자 불일치 case {i}"
    assert ("{m}" in ctx_ko) == ("{m}" in en[1]), f"{{m}} 불일치 case {i}"

def rx(t): return re.compile("^" + re.escape(t).replace(re.escape("{m}"), r"(\d+)") + "$")
pats = [(rx(c[7]), c[6]) for c in CASES]

inp = pd.read_csv("benchmark_inputs.csv"); meta = pd.read_csv("benchmark_meta.csv").set_index("case_id")
goals, ctxs, whys = [], [], []
for r in inp.itertuples():
    goal_ko = "" if pd.isna(r.user_stated_goal) else r.user_stated_goal
    hits = [(i, p.match(r.recent_context)) for i, (p, g) in enumerate(pats) if g == goal_ko and p.match(r.recent_context)]
    assert len(hits) == 1, f"{r.case_id}: 매칭 {len(hits)}건"
    i, m = hits[0]
    goals.append(EN[i][0])
    ctxs.append(EN[i][1].format(m=m.group(1)) if m.groups() else EN[i][1])
    whys.append(EN[i][2])

en = inp.copy(); en["user_stated_goal"] = goals; en["recent_context"] = ctxs
en.to_csv("benchmark_inputs_en.csv", index=False, encoding="utf-8-sig")
me = meta.copy(); me["label_rationale"] = whys; me.reset_index().to_csv("benchmark_meta_en.csv", index=False, encoding="utf-8-sig")
ann = en.copy(); ann["annotator_action"] = ""; ann.to_csv("annotation_sheet_en.csv", index=False, encoding="utf-8-sig")
print("변환 완료:", len(en), "사례 -> benchmark_inputs_en.csv, benchmark_meta_en.csv, annotation_sheet_en.csv")
print(en[["case_id", "user_stated_goal", "recent_context"]].head(3).to_string(index=False))
