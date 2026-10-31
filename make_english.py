"""
make_english.py - 한국어 벤치마크를 영어 버전으로 변환 (build_benchmark.py의 CASES, CF_PAIRS, SC_CASES 순서와 1:1 대응)
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

# 제스처 반례 쌍 (build_benchmark.py의 CF_PAIRS와 같은 순서). 쌍의 a/b는 맥락이 같으므로 번역 1개를 공유한다.
EN_CF = [
 ("Binge-reading a completed webtoon over the weekend", "Reading a subscribed completed webtoon in order from episode 1, now on episode 23, not moving to other titles", "Stated goal and behavior match (regardless of gesture)"),
 ("Deleting blurry photos from 1000 travel photos", "Going through photos one by one, deleted 312 times, favorited 41 times, created 2 albums", "Carrying out an organizing task (regardless of gesture)"),
 ("Reading a long investigative article to the end", "Read the full body of 1 article to the end, highlighted 3 times, shared the link to a notes app at the end", "Purposeful reading (regardless of gesture)"),
 ("", "Past the user's bedtime goal of 23:00, continuously browsing the feed, no interaction with posts, no search or saves, no switching to other apps", "Exceeded bedtime goal plus no goal-directed actions (regardless of gesture)"),
 ("", "Entered via a notification, then watched only recommended videos, no search, saves or likes, calendar: assignment work time that started 30 minutes ago", "Drift after a notification plus schedule conflict (regardless of gesture)"),
 ("", "Browsing the recommended product list, repeatedly opening a product page and going straight back, no search, wishlist or cart, 15 minutes ago the user typed 'I should go to sleep now'", "Behavior contradicts the user's own stated wish plus no goal-directed actions (regardless of gesture)"),
]

# 지름길 반례 (build_benchmark.py의 SC_CASES와 같은 순서)
EN_SC = [
 ("", "Viewing a group account feed with photos from a friend's birthday party, wrote a comment on each post 12 times, saved photos 9 times, finally forwarded the photos via messenger and closed the app", "No declared goal, but productive interaction"),
 ("", "Entered the search query 'bicycle chain replacement', watched only 4 related short videos, switched to a shopping app during viewing and bought parts, then closed the app", "Search, goal achieved, exit"),
 ("", "Entered via a club account notification, wrote and edited an event announcement post 3 times, wrote 15 replies to applicants' comments, added the event to the calendar app", "Carrying out a task on social media"),
 ("", "Flipping cards in a vocabulary app and checking answers, completed 180 of today's target of 200 cards, re-studying only the missed cards", "Uniform flipping, but a learning task"),
 ("", "After a video call with family living abroad, viewing posts in the family group account, liking and commenting on each post, uploaded 3 photos", "Late night and social media, but productive interaction"),
 ("", "Studying how other short videos are edited in order to edit a video they filmed, switched to a video editing app 6 times while watching, edit project save recorded", "Using short-form video as reference material"),
 ("", "Entered via a delivery arrival notification, checked the shipment and processed 3 return requests, attached photos to each, 1 customer service chat", "Many sessions, but carrying out a task"),
 ("Checking recent posts from the account of a company they are applying to for an internship", "Viewing the company account's posts newest first, saved 4 hiring-related posts, switched to a document app and noted the content in an application", "Using social media as an information-gathering tool"),
 ("", "Videos keep playing via autoplay, almost no screen interaction, past the bedtime goal of 24:00", "Exceeded bedtime goal plus passive viewing"),
 ("Checking one notice", "After checking the notice, moved to the popular posts list and kept reading posts, no comments or searches", "Drifted from the stated goal plus no goal-directed actions"),
 ("Checking today's weather", "After checking the weather, moved to celebrity news and kept reading articles in a row, calendar: class at 09:00", "Drifted from the stated goal plus schedule conflict"),
 ("", "Opened the messenger with no new messages and closed it right away, repeated 24 times, calendar: study time", "Repeated rechecking plus schedule conflict"),
 ("", "Repeatedly refreshing the same community homepage, no posts opened, past the bedtime goal of 24:00", "Repeated refreshing plus exceeded bedtime goal"),
 ("Listening to 1 lecture", "After the lecture ended, watched 4 recommended videos in a row, topics shifted to variety shows and games, no switch to a notes app", "Drifted from the stated goal"),
 ("Ordering 2 household items", "After paying for the 2 items, moved to the recommended products list and kept browsing, nothing added to the cart, past the bedtime goal of 23:00", "Drift after task completion plus exceeded bedtime goal"),
 ("", "Entered via a breaking-news notification, then kept moving to unrelated articles, opened only the comment list of each article, calendar: team meeting 10:30–12:00", "Drift after a notification plus schedule conflict"),
 ("", "Saved 2 posts while continuously browsing the feed, then continued browsing right after, no bedtime goal set", "Saving actions mixed with continuous browsing"),
 ("Referring to cooking short videos", "Saved 3 cooking short videos, then kept watching the feed, topics mixed between cooking and travel", "Partly matches the goal plus drifting signals"),
 ("", "Watched 3 episodes of a documentary series in a row, no videos outside the series, calendar: assignment due at 14:00", "Consistent viewing vs an imminent deadline"),
 ("", "Repeatedly opening the messenger, chat history is blocked by the user's settings", "Key information inaccessible"),
 ("Checking the election results", "Reading vote-count breaking news and analysis articles in a row, refreshed the same article 6 times, bedtime goal of 24:00 approaching", "Goal matches vs repeated refreshing and approaching bedtime"),
 ("", "Repeatedly viewing posts for the same item on a second-hand market board, 1 chat with a seller, no completed transaction", "Hard to tell comparison shopping from repeated viewing"),
 ("", "Flipping through and viewing old photos, no deletion, editing or sharing, lingered on some photos for a long time", "Hard to tell reminiscing from passive consumption"),
 ("Choosing a winter padded jacket", "After searching for padded jackets, opened many product pages, 1 wishlist add, then moved to shoes and bags", "Partly matches the goal plus drifting signals"),
]

# build_benchmark.py에서 CASES 리터럴만 꺼내 한국어 원문을 얻는다 (재입력으로 인한 오타 방지)
src = open("build_benchmark.py", encoding="utf-8").read()
s = src.index("CASES = ["); e = src.index("\n]\n", s) + 3
ns = dict(nan=float("nan"), NO="no_intervention", GP="gentle_prompt", AC="ask_context",
          IN="intentional", UN="unintended", UC="unclear")
exec(src[s:e], ns); CASES = ns["CASES"]
assert len(CASES) == len(EN) == 45, (len(CASES), len(EN))
s2 = src.index("CF_PAIRS = ["); e2 = src.index("\n]\n", s2) + 3
exec(src[s2:e2], ns); CF_PAIRS = ns["CF_PAIRS"]
assert len(CF_PAIRS) == len(EN_CF), (len(CF_PAIRS), len(EN_CF))
# 반례 쌍도 CASES와 같은 형태(인덱스 6=목표, 7=맥락)로 맞춰 아래 점검·매칭에 함께 넣는다
CASES = CASES + [(None,) * 6 + (c[5], c[6]) for c in CF_PAIRS]
EN = EN + EN_CF
s3 = src.index("SC_CASES = ["); e3 = src.index("\n]\n", s3) + 3
exec(src[s3:e3], ns); SC_CASES = ns["SC_CASES"]
assert len(SC_CASES) == len(EN_SC), (len(SC_CASES), len(EN_SC))
CASES = CASES + list(SC_CASES); EN = EN + EN_SC

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
