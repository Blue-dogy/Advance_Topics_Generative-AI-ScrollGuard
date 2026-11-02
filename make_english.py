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
 ("Watching lectures to prepare for the operating systems midterm", "Entry: searched 'OS paging lecture' and opened a playlist | In app: watched 4 lectures in playlist order at 2x speed, rewound a section 7 times | App switches: notes app 3 times (2-4 min each) | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: operating systems midterm on 11/05", "Long, but behavior matches the stated goal"),
 ("", "Entry: opened the e-book app from the home screen | In app: resumed reading at page 312, turned pages every 1-3 minutes, underlined 2 passages | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: 18:00 gym", "Reading app, self-directed use"),
 ("Checking a soybean paste stew recipe", "Entry: searched 'soybean paste stew recipe' | In app: 2 searches, watched 3 recipe videos, paused 4 times at the ingredient list, closed the app after the last video | App switches: notes app 1 time (wrote an ingredient list) | Interactions: searches 2, saves 0, likes 0, comments 0 | Schedule/settings: 14:00 class", "Search, goal achieved, exit"),
 ("", "Entry: opened the map app from the home screen | In app: searched the destination 'engineering building', started walking directions, closed the app | App switches: — | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: 09:00 class", "Short utility use"),
 ("", "Entry: tapped a friend's message notification | In app: opened 1 chat, sent 1 reply, closed the app | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: 14:00 class", "Short utility use"),
 ("", "Entry: opened directly from the home screen | In app: watched 2 new videos from subscribed channels, liked 1 | App switches: — | Interactions: searches 0, saves 0, likes 1, comments 0 | Schedule/settings: user-set break timer of 20 min, {m} min elapsed, bedtime goal 24:00", "Within the user's self-set time budget"),
 ("Organizing a trip photo album and sharing it with friends", "Entry: opened directly | In app: uploaded 14 photos, wrote captions 5 times, 9 replies to comments, feed scrolling between uploads under 1 min each | App switches: gallery app 4 times (choosing photos) | Interactions: searches 0, saves 0, likes 0, comments 9 | Schedule/settings: 19:00 dinner", "Social media and long duration, but productive activity"),
 ("Watching a conference presentation video", "Entry: searched 'CHI 2026 talk' | In app: playing 1 video continuously (runtime 52 min, now at {m} min), paused 4 times, turned on subtitles | App switches: document app 2 times (taking notes) | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: lab meeting today 15:00", "Purposeful despite high usage concentration"),
 ("Checking the group chat before the team project deadline", "Entry: opened directly from the home screen | In app: opened the app {s} times, read 23 messages in the team project group chat, sent 6 messages, shared 2 files | App switches: alternated with a shared document app, 4 document edits recorded | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: team project due today 12:00", "Many sessions, but carrying out a task"),
 ("Comparing prices of a laptop added to the cart yesterday", "Entry: tapped a price-drop notification | In app: viewed a spec comparison of 3 items in the cart, read 6 reviews, paid for 1 and closed the app | App switches: card company app 1 time (payment verification) | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: class tomorrow 10:00", "Clear task completed"),
 ("Watching reference videos for an assignment due tomorrow, then going to bed", "Entry: searched 'machine learning assignment regularization example' | In app: 4 searches, watched 5 videos (all titles about regularization), paused 9 times | App switches: notes app 5 times (code notes) | Interactions: searches 4, saves 0, likes 0, comments 0 | Schedule/settings: assignment due tomorrow 09:00, bedtime goal 01:00", "Late night and long duration, but purposeful"),
 ("", "Entry: started together with a workout in a running app | In app: played 1 podcast episode, screen off 92% of the time | App switches: running app recording in progress (5.2 km) | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: 19:30 dinner", "No screen consumption"),
 ("Searching restaurant reviews for a weekend gathering and then making a reservation", "Entry: searched 'Seongsu brunch' | In app: 3 searches by area name, read 11 review posts, saved 2 posts | App switches: reservation app 1 time, booked Saturday 11:30 for 4 people | Interactions: searches 3, saves 2, likes 0, comments 0 | Schedule/settings: gathering scheduled for Saturday 11:30", "Using social media as a search tool"),
 ("Declared directly: it is a day off, so planning to watch one movie", "Entry: opened directly | In app: playing 1 movie (runtime 128 min, now at {m} min), paused 1 time | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: user turned on focus mode, meeting tomorrow 10:00", "Respect the user's explicit declaration"),
 ("Checking email on the commute", "Entry: opened directly | In app: opened the mail app {s} times, about 10 seconds each, checked new mail in the inbox and replied to 1 | App switches: subway map app 2 times | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: location status moving, work starts 09:00", "Utility use while in transit"),
 ("Studying Spanish expressions with short videos", "Entry: searched 'everyday Spanish expressions' | In app: watched 17 short videos, paused on each, saved 8 | App switches: quiz app 2 times (entering words) | Interactions: searches 1, saves 8, likes 0, comments 0 | Schedule/settings: Spanish class 13:00", "Short-form video, but learning behavior"),
 ("", "Entry: opened directly | In app: played a saved playlist, then closed the app | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: 13:00 class", "Short utility use"),
 ("Checking course registration reviews for next semester", "Entry: opened directly | In app: searched 'recommended electives', read 5 posts (1-2 min each), clipped 1 post | App switches: timetable app 1 time (added 2 courses) | Interactions: searches 1, saves 1, likes 0, comments 0 | Schedule/settings: course registration opens tomorrow 10:00", "Clear information seeking"),
 # ---- gentle_prompt (17) ----
 ("", "Entry: tapped a recommended-video notification | In app: played 412 short videos in a row from the recommended feed | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: class tomorrow 09:00", "Multiple autopilot signals"),
 ("", "Entry: opened directly | In app: scrolled 690 posts in the home feed | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: user bedtime goal 24:00 (70 min past), class tomorrow 09:00", "Exceeded bedtime goal plus passive consumption"),
 ("", "Entry: opened directly each time the screen turned on | In app: opened the app {s} times, about 40 seconds each, refreshed the top of the feed and closed it every time | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00, alarm tomorrow 08:00", "Short total time, but compulsive rechecking"),
 ("Watching just one video about a lunch menu", "Entry: opened directly | In app: watched 1 meal-plan video, then 5 more via autoplay, categories in order: cooking → games → entertainment → sports | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: team meeting 16:00", "Drifted away from the stated goal"),
 ("Searching for assignment materials", "Entry: searched 'database normalization' | In app: 2 searches, watched 1 search-result video, then moved to the short-video feed and played 290 videos in a row | App switches: — | Interactions: searches 2, saves 0, likes 0, comments 0 | Schedule/settings: database assignment due tomorrow 23:59", "Drifted into the feed after searching"),
 ("", "Entry: tapped a friend's post notification | In app: checked the post, then scrolled 370 posts in the home feed, likes 2 | App switches: — | Interactions: searches 0, saves 0, likes 2, comments 0 | Schedule/settings: class 13:00–14:15, location: lecture building", "Passive consumption that conflicts with the schedule"),
 ("", "Entry: opened directly | In app: opened the app {s} times, relaunching right after closing, refreshing the top of the same feed each time | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 01:00, alarm tomorrow 08:00", "Many sessions, late night, passive"),
 ("", "Entry: opened directly | In app: scrolled 350 products in the recommendations tab, opened product pages 2 times (about 10 seconds each) | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: online course study time 21:00–23:00, bedtime goal 24:00", "No goal-directed actions"),
 ("", "Entry: tapped a breaking-news notification | In app: opened the comment list of 1 article, refreshed the comment list 14 times, 5 seconds on the article body | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: dinner 19:00", "Passive repetitive consumption"),
 ("", "Entry: opened directly | In app: scrolled 590 posts in the home feed, likes 1 | App switches: — | Interactions: searches 0, saves 0, likes 1, comments 0 | Schedule/settings: 15 minutes ago the user typed 'I should stop watching now' in ScrollGuard", "Behavior conflicts with the user's own stated wish"),
 ("", "Entry: opened directly right after dismissing the wake-up alarm | In app: scrolled 95 posts in the home feed, watched 12 stories | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: first event today 09:15 class, commute 25 min", "Passive consumption just before a scheduled event"),
 ("", "Entry: unlocked the phone while using a work app and opened directly | In app: opened the app {s} times, checked the home screen and closed within 20 seconds each time, new notifications 0 | App switches: alternating with a document app | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: focus work time 13:00–17:00", "Repeated rechecking without notifications"),
 ("A 10-minute break during studying", "Entry: opened directly | In app: played 370 videos in a row in the short-video feed | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: user set a 10 min break timer, the timer has ended, {m} min of use so far", "Exceeded the self-set time budget"),
 ("", "Entry: tapped a friend's DM notification | In app: read the DM and replied 1 time, then moved to the explore tab and scrolled 50 posts | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 22:00 (30 min past)", "Drifted into the feed after the task was finished"),
 ("Watch short videos for just 5 minutes", "Entry: opened directly | In app: at the start the user typed 'just 5 minutes', played 520 short videos in a row, {m} min of use so far | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: part-time job 18:00", "Exceeded the self-set time"),
 ("", "Entry: tapped a recommended-video notification | In app: played 780 short videos in a row, pauses 0 | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 01:00, class tomorrow 09:00", "Multiple autopilot signals"),
 ("", "Entry: opened directly | In app: opened the app {s} times, refreshed the post list 41 times, repeatedly jumped to the top, 9 seconds average on post bodies | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: meeting 13:30", "Passive repetitive checking"),
 # ---- ask_context (10) ----
 ("", "Entry: opened directly | In app: searched 1 time ('how to make pasta'), then moved to recommended videos, 8 videos watched by category: cooking 3, games 3, news 2 | App switches: — | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: meeting tomorrow 10:00", "Mixed goal-directed and drifting signals"),
 ("", "Entry: opened directly | In app: wrote 1 post, then scrolled 130 posts in the feed, 2 comments on other accounts' posts | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 2 | Schedule/settings: the account has a business-account tag, 15:00–18:00 'content work'", "Cannot distinguish work from leisure"),
 ("", "Entry: opened directly | In app: opened the browser {s} times, about 25 seconds each, visited pages and search history are access-restricted by the user's settings | App switches: messenger 3 times | Interactions: search, save, like and comment logs are access-restricted | Schedule/settings: meeting 15:00", "Key information not accessible"),
 ("Reference for tomorrow's presentation", "Entry: searched 'presentation opening examples' | In app: watched 5 videos, only 1 title matches the presentation topic | App switches: slide app 1 time (2 min) | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00 (10 min past), presentation tomorrow 10:00", "Partially matches the goal, plus exceeded budget"),
 ("Watching short videos for learning", "Entry: opened directly | In app: played 300 short videos, 70 of them with study-topic titles, pauses 0 | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: study group 19:00", "Behavior partially matches the stated goal"),
 ("", "Entry: tapped a friend's post notification | In app: checked the post, then scrolled 40 posts in the home feed, likes 3 | App switches: — | Interactions: searches 0, saves 0, likes 3, comments 0 | Schedule/settings: lunch time, meeting 15:00", "Cannot distinguish a normal break from a habit"),
 ("", "Entry: opened directly | In app: opened the app {s} times, in 1 of them watched 2 long videos (about 25 min each) in a row | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: first observed night viewing for this user, bedtime goal 02:00", "No baseline history"),
 ("", "Entry: opened directly | In app: opened 12 product pages (category: sneakers), wishlisted 2, payments 0 | App switches: price comparison app 1 time | Interactions: searches 0, saves 2, likes 0, comments 0 | Schedule/settings: class 11:00", "Cannot distinguish comparison shopping from impulse browsing"),
 ("", "Entry: opened directly | In app: read 26 posts, wrote 3 comments, refreshed the popular posts list 5 times | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 3 | Schedule/settings: class tomorrow 09:00", "Cannot distinguish interaction from consumption"),
 ("", "Entry: opened directly | In app: read 1 economy article closely (4 min), then followed related-article links 9 times, about 40 seconds per article | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: work starts 09:30", "Cannot distinguish information gathering from habit"),
]

# 제스처 반례 쌍 (build_benchmark.py의 CF_PAIRS와 같은 순서). 쌍의 a/b는 맥락이 같으므로 번역 1개를 공유한다.
EN_CF = [
 ("Binge-reading a completed webtoon over the weekend", "Entry: opened directly | In app: reading a subscribed completed webtoon in order from episode 1, now on episode 23 | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: weekend, class Monday 09:00", "Stated goal and behavior match (regardless of gesture)"),
 ("Deleting blurry photos from 1000 travel photos", "Entry: opened directly | In app: going through travel album photos one by one, deleted 58, favorited 12, created 2 new albums | App switches: — | Interactions: searches 0, saves 12, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00", "Carrying out an organizing task (regardless of gesture)"),
 ("Reading a long investigative article to the end", "Entry: searched 'platform labor investigative report' | In app: read the full body of 1 article to the end, highlighted 3 times | App switches: notes app 1 time (shared the link) | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: work starts 09:30", "Purposeful reading (regardless of gesture)"),
 ("", "Entry: opened directly | In app: kept browsing posts in the home feed | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: user bedtime goal 23:00 (100 min past), class tomorrow 09:00", "Exceeded bedtime goal plus no goal-directed actions (regardless of gesture)"),
 ("", "Entry: tapped a recommended-video notification | In app: kept playing recommended videos only | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: assignment work time 14:30–17:00", "Drift after a notification plus schedule conflict (regardless of gesture)"),
 ("", "Entry: opened directly | In app: browsing the recommended product list, repeatedly opening a product page and going straight back | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: 15 minutes ago the user typed 'I should go to sleep now' in ScrollGuard", "Behavior contradicts the user's own stated wish plus no goal-directed actions (regardless of gesture)"),
]

# 지름길 반례 (build_benchmark.py의 SC_CASES와 같은 순서)
EN_SC = [
 ("", "Entry: opened directly | In app: viewed a group account's posts with photos from a friend's birthday party, wrote 12 comments, saved 9 photos | App switches: messenger 1 time (forwarded 9 photos), then closed the app | Interactions: searches 0, saves 9, likes 0, comments 12 | Schedule/settings: bedtime goal 24:00", "No declared goal, but productive interaction"),
 ("", "Entry: searched 'bicycle chain replacement' | In app: watched 4 related short videos, paused 6 times | App switches: shopping app 1 time (bought a chain tool), then closed the app | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: cycling club 19:00", "Search, goal achieved, exit"),
 ("", "Entry: tapped a club account notification | In app: wrote and edited an event announcement post 3 times, 15 replies to applicants' comments | App switches: calendar app 1 time (added the event) | Interactions: searches 0, saves 0, likes 0, comments 15 | Schedule/settings: club event on 11/20", "Carrying out a task on social media"),
 ("", "Entry: opened directly | In app: flipping vocabulary cards and checking answers, 180 of today's target of 200 cards done, re-studied 23 missed cards | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: TOEIC class 09:00", "Uniform flipping, but a learning task"),
 ("", "Entry: opened directly | In app: viewed the family group account's posts, likes 14, comments 6, uploaded 3 photos | App switches: video call app used for the previous 30 minutes | Interactions: searches 0, saves 0, likes 14, comments 6 | Schedule/settings: bedtime goal 01:00", "Late night and social media, but productive interaction"),
 ("", "Entry: opened directly | In app: watched short videos on editing techniques, saved 5 | App switches: video editing app 6 times, saved the editing project 3 times | Interactions: searches 0, saves 5, likes 0, comments 0 | Schedule/settings: video assignment due 18:00", "Using short-form video as reference material"),
 ("", "Entry: tapped a delivery arrival notification | In app: opened the app {s} times, checked shipping 2 times, processed 3 return requests (a photo attached to each) | App switches: customer service chat 1 time, camera 3 times | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: meeting 13:00", "Many sessions, but carrying out a task"),
 ("Checking recent posts from the account of a company they are applying to for an internship", "Entry: searched the company account name | In app: viewed the company account's posts newest first, saved 4 hiring-related posts | App switches: document app 2 times (notes for the application) | Interactions: searches 1, saves 4, likes 0, comments 0 | Schedule/settings: internship application due 11/10", "Using social media as an information-gathering tool"),
 ("", "Entry: opened directly | In app: 11 videos played one after another via autoplay, 2 screen touches | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00 (130 min past), class tomorrow 09:00", "Exceeded bedtime goal plus passive viewing"),
 ("Checking one notice", "Entry: tapped a department board notice notification | In app: checked 1 notice (30 seconds), then moved to the popular posts list, opened the app {s} times, read 20 posts | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00", "Drifted from the stated goal plus no goal-directed actions"),
 ("Checking today's weather", "Entry: opened directly | In app: checked the weather tab (15 seconds), then read 22 celebrity news articles in a row | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: class 09:00", "Drifted from the stated goal plus schedule conflict"),
 ("", "Entry: opened directly | In app: opened the messenger {s} times, new messages 0 each time, closed right away | App switches: alternating with a study timer app | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: study time 13:00–17:00", "Repeated rechecking plus schedule conflict"),
 ("", "Entry: opened directly | In app: opened the browser {s} times, refreshed the same community homepage, posts opened 0 | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00 (90 min past)", "Repeated refreshing plus exceeded bedtime goal"),
 ("Listening to 1 lecture", "Entry: searched 'linear algebra eigenvalue lecture' | In app: watched 1 lecture, then 4 recommended videos in a row, categories in order: lecture → variety → games → variety → games | App switches: — | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: linear algebra quiz tomorrow 10:00", "Drifted from the stated goal"),
 ("Ordering 2 household items", "Entry: searched 'dish soap' | In app: paid for 2 household items, then scrolled 60 products in the recommended list | App switches: — | Interactions: searches 1, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 23:00 (10 min past)", "Drift after task completion plus exceeded bedtime goal"),
 ("", "Entry: tapped a breaking-news notification | In app: checked the breaking news, then moved through 30 articles, categories: entertainment 12, sports 10, society 8, opened the comment list of each article | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: team meeting 10:30–12:00", "Drift after a notification plus schedule conflict"),
 ("", "Entry: opened directly | In app: saved 2 posts while scrolling 230 posts in the feed, resumed scrolling right after saving | App switches: — | Interactions: searches 0, saves 2, likes 0, comments 0 | Schedule/settings: bedtime goal 01:00", "Saving actions mixed with continuous browsing"),
 ("Referring to cooking short videos", "Entry: searched 'one-pan pasta' | In app: saved 3 cooking short videos, then played 210 feed videos, categories: cooking 110, travel 100 | App switches: — | Interactions: searches 1, saves 3, likes 0, comments 0 | Schedule/settings: dinner prep 19:30", "Partly matches the goal plus drifting signals"),
 ("", "Entry: opened directly | In app: watched 3 episodes of a documentary series in a row, all watched videos from the same series | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: assignment due 14:00", "Consistent viewing vs an imminent deadline"),
 ("", "Entry: tapped a message notification | In app: opened the messenger {s} times, chat content and viewing history are blocked by the user's settings | App switches: — | Interactions: search, save, like and comment logs are access-restricted | Schedule/settings: dinner 18:00", "Key information inaccessible"),
 ("Checking the election results", "Entry: tapped a vote-count breaking-news notification | In app: read 18 vote-count and analysis articles, refreshed the same vote-count article 6 times | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00 (10 min left)", "Goal matches vs repeated refreshing and approaching bedtime"),
 ("", "Entry: opened directly | In app: repeatedly viewing posts for the same item (camera) on a second-hand market board, 1 chat with a seller, completed transactions 0 | App switches: price comparison app 2 times | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: dinner 19:00", "Hard to tell comparison shopping from repeated viewing"),
 ("", "Entry: opened directly | In app: flipping through photos from 3 years ago, deleted 0, edited 0, shared 0, stayed on some photos for 2+ minutes | App switches: — | Interactions: searches 0, saves 0, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00", "Hard to tell reminiscing from passive consumption"),
 ("Choosing a winter padded jacket", "Entry: searched 'long padded jacket' | In app: opened 23 padded jacket product pages, wishlisted 1, then opened 17 shoe and bag product pages | App switches: — | Interactions: searches 1, saves 1, likes 0, comments 0 | Schedule/settings: bedtime goal 24:00", "Partly matches the goal plus drifting signals"),
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

def digits(t): return sorted(re.findall(r"\d+", t.replace("{m}", "").replace("{s}", "")))
for i, (c, en) in enumerate(zip(CASES, EN)):                       # 순서·내용 정합성 점검
    goal_ko, ctx_ko = c[6], c[7]
    assert (goal_ko == "") == (en[0] == ""), f"goal 불일치 case {i}"
    assert digits(goal_ko) == digits(en[0]), f"goal 숫자 불일치 case {i}"
    assert digits(ctx_ko) == digits(en[1]), f"ctx 숫자 불일치 case {i}"
    for ph in ("{m}", "{s}"):
        assert (ph in ctx_ko) == (ph in en[1]), f"{ph} 불일치 case {i}"

def rx(t): return re.compile("^" + re.escape(t).replace(re.escape("{m}"), r"(?P<m>\d+)").replace(re.escape("{s}"), r"(?P<s>\d+)") + "$")
pats = [rx(c[7]) for c in CASES]
import user_states as US
for i in range(len(CASES)):                                       # 사용자 상태·목표 번역의 숫자 일치 점검
    assert digits(US.PERSONA[i][0]) == digits(US.PERSONA[i][1]), f"persona 숫자 불일치 case {i}"
    assert digits(US.goal(i, "ko")) == digits(US.goal(i, "en")), f"goal 숫자 불일치 case {i}"

inp = pd.read_csv("benchmark_inputs.csv"); meta = pd.read_csv("benchmark_meta.csv").set_index("case_id")
goals, ctxs, whys, states = [], [], [], []
for r in inp.itertuples():
    hits = [(i, p.match(r.recent_context)) for i, p in enumerate(pats) if p.match(r.recent_context)]
    if len(hits) > 1:   # 반례 쌍은 맥락이 같으므로 목표 문장으로 구분 (쌍은 목표도 같아 어느 쪽이든 같은 번역)
        hits = [h for h in hits if US.goal(h[0], "ko") == r.user_stated_goal][:1]
    assert len(hits) == 1, f"{r.case_id}: 매칭 {len(hits)}건"
    i, m = hits[0]
    goals.append(US.goal(i, "en")); states.append(US.PERSONA[i][1])
    ctxs.append(EN[i][1].format(**m.groupdict()))
    whys.append(EN[i][2])

en = inp.copy(); en["user_stated_goal"] = goals; en["recent_context"] = ctxs; en["user_state"] = states
en.to_csv("benchmark_inputs_en.csv", index=False, encoding="utf-8-sig")
me = meta.copy(); me["label_rationale"] = whys; me.reset_index().to_csv("benchmark_meta_en.csv", index=False, encoding="utf-8-sig")
ann = en.copy(); ann["annotator_action"] = ""; ann.to_csv("annotation_sheet_en.csv", index=False, encoding="utf-8-sig")
print("변환 완료:", len(en), "사례 -> benchmark_inputs_en.csv, benchmark_meta_en.csv, annotation_sheet_en.csv")
print(en[["case_id", "user_stated_goal", "recent_context"]].head(3).to_string(index=False))
