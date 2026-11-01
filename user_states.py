"""
user_states.py - 사례별 사용자 상태와 선언 목표 (한국어, 영어)
build_benchmark.py와 make_english.py가 함께 읽는다. 인덱스는 CASES(0~44), CF_PAIRS(45~50), SC_CASES(51~74) 순서다.
  PERSONA[i]  = (한국어, 영어): 누구인지, 어떤 날인지, 지금 어디서 무엇을 하는 중인지. 정답을 직접 말하는 표현은 쓰지 않는다.
  GOALS[i]    = (한국어, 영어): 사용자가 앱을 열 때 ScrollGuard 목표 질문에 직접 입력한 문장
  GOALS에 없는 사례는 목표 칸이 비어 있는 이유를 BLANK_REASONS에서 i % 3으로 정한다(정답과 무관하게 배정).
"""

BLANK_REASONS = [
    ("(목표 미입력: 앱을 열 때 ScrollGuard 목표 질문에서 '건너뛰기'를 누름)",
     "(goal field blank: tapped 'skip' on ScrollGuard's goal question when opening the app)"),
    ("(목표 미입력: 사용자가 설정에서 목표 질문 기능을 꺼 둠)",
     "(goal field blank: the user turned off the goal question in settings)"),
    ("(목표 미입력: 목표 질문 창을 닫고 바로 사용을 시작함)",
     "(goal field blank: closed the goal question window and started using the app right away)"),
]

GOALS = {
    0: ("운영체제 중간고사(11/05) 대비로 페이징 강의 재생목록 끝까지 듣기", "Finish the paging lecture playlist to prepare for the operating systems midterm (11/05)"),
    2: ("점심으로 된장찌개를 끓이려고 레시피 확인하기, 재료는 냉장고에 있는 걸로", "Check a soybean paste stew recipe for lunch, using what is in the fridge"),
    6: ("지난주 여행 사진 앨범을 정리해서 친구들 단체 계정에 올리기", "Organize last week's trip photo album and post it to the friends' group account"),
    7: ("오후 연구실 미팅 전에 학회 발표 영상 1개 보고 메모하기", "Watch 1 conference talk and take notes before the afternoon lab meeting"),
    8: ("조별과제 마감 전에 단톡방 확인하고 내가 맡은 부분 문서 마무리하기", "Check the group chat before the team project deadline and finish my part of the document"),
    9: ("어제 장바구니에 담아 둔 노트북 3개 가격 비교하고 하나 결제하기", "Compare prices of the 3 laptops I put in the cart yesterday and buy one"),
    10: ("내일 9시 마감인 과제 참고 영상 보고 코드 정리한 뒤 1시 전에 자기", "Watch reference videos for the assignment due at 9 tomorrow, organize my code, and sleep before 1"),
    12: ("주말 모임 장소로 갈 브런치 맛집 후기 찾아보고 예약까지 하기", "Find reviews of brunch places for the weekend gathering and make a reservation"),
    13: ("오늘은 쉬는 날이니까 보고 싶던 영화 한 편을 처음부터 끝까지 볼 거야", "It's my day off, so I'm going to watch a movie I have wanted to see from start to finish"),
    14: ("출근길 지하철에서 밤사이 온 업무 메일만 확인하기", "Check only the work emails that came in overnight, on the subway to work"),
    15: ("쇼츠로 스페인어 일상 표현 공부하고 모르는 표현은 저장하기", "Study everyday Spanish expressions with short videos and save the unfamiliar ones"),
    17: ("다음 학기 교양 수강신청 후기 확인하고 시간표 짜기", "Check reviews of electives for next semester's registration and plan my timetable"),
    21: ("점심 식단 영상 하나만 보고 바로 메뉴 정하기", "Watch just one meal-plan video for lunch and decide the menu right away"),
    22: ("데이터베이스 과제에 쓸 정규화 설명 자료 찾기", "Find material explaining normalization for the database assignment"),
    30: ("공부하다가 10분만 쉬는 시간, 끝나면 다시 공부", "A 10-minute break from studying, then back to studying"),
    32: ("딱 5분만 쇼츠 보고 아르바이트 갈 준비하기", "Watch short videos for just 5 minutes, then get ready for my part-time job"),
    38: ("내일 발표 오프닝을 어떻게 할지 참고 영상 찾기", "Find reference videos on how to open tomorrow's presentation"),
    39: ("학습용 쇼츠 보면서 코딩 개념 복습하기", "Review coding concepts by watching study short videos"),
    45: ("주말에 완결된 웹툰을 처음부터 끝까지 정주행하기", "Binge-read a completed webtoon from start to finish over the weekend"),
    46: ("여행 사진 1000장 중 흔들린 사진 지우고 앨범 나누기", "Delete blurry photos from 1000 travel photos and sort them into albums"),
    47: ("출근 전에 플랫폼 노동 탐사보도 기사 끝까지 읽기", "Read the investigative article on platform labor to the end before work"),
    58: ("인턴 지원하는 회사 계정의 최근 게시물 보고 지원서에 반영하기", "Look at recent posts from the company I am applying to for an internship and reflect them in my application"),
    60: ("학과 공지 하나만 확인하기", "Check just one department notice"),
    61: ("오늘 날씨만 확인하고 수업 가기", "Check today's weather and head to class"),
    64: ("선형대수 고유값 강의 1개 듣고 퀴즈 준비하기", "Listen to 1 linear algebra eigenvalue lecture and prepare for the quiz"),
    65: ("주방 세제랑 휴지, 생필품 2개 주문하기", "Order 2 household items: dish soap and tissues"),
    68: ("저녁에 만들 원팬 파스타 요리 쇼츠 참고하기", "Look up one-pan pasta short videos for tonight's dinner"),
    71: ("선거 개표 결과 확인하기", "Check the election vote-count results"),
    74: ("이번 겨울에 입을 롱패딩 하나 고르기", "Pick a long padded jacket to wear this winter"),
}

PERSONA = [
    # ---- CASES no_intervention 0-17 ----
    ("컴퓨터공학과 3학년, 중간고사 1주 전 평일 밤. 저녁을 먹고 책상에 앉은 상태.", "3rd-year computer science student, a weekday night 1 week before midterms. Sitting at the desk after dinner."),
    ("직장인, 연차를 쓴 평일 오후. 카페에 앉아 있음.", "Office worker on a day of annual leave, weekday afternoon. Sitting in a cafe."),
    ("자취하는 대학생, 공강인 평일 점심. 집 부엌에 있음.", "University student living alone, weekday lunch with a free period. In the kitchen at home."),
    ("대학교 1학년, 평일 아침 등굣길. 처음 가 보는 건물로 이동 중.", "1st-year university student, on the way to school on a weekday morning. Heading to a building for the first time."),
    ("대학생, 평일 점심시간. 학생식당에서 식사 중.", "University student, weekday lunch time. Eating at the student cafeteria."),
    ("대학원생, 평일 밤. 논문 작업을 마치고 침대에 누워 쉬는 중.", "Graduate student, weekday night. Lying in bed resting after working on a paper."),
    ("대학생, 여행에서 돌아온 다음 날인 일요일 오후. 집에서 쉬는 중.", "University student, Sunday afternoon, the day after returning from a trip. Resting at home."),
    ("HCI 연구실 석사과정, 평일 오전. 연구실 자리에 앉아 있음.", "Master's student in an HCI lab, weekday morning. At the lab desk."),
    ("대학생, 조별과제 마감일 아침. 기숙사 방에서 노트북을 켠 상태.", "University student, the morning of a team project deadline. Laptop open in the dorm room."),
    ("새 학기를 앞둔 대학생, 평일 저녁. 노트북 교체를 2주째 고민 중.", "University student before the new semester, weekday evening. Has been considering a new laptop for 2 weeks."),
    ("컴퓨터공학과 4학년, 과제 마감 전날 밤. 책상에서 노트북으로 코딩 중.", "4th-year computer science student, the night before a deadline. Coding on a laptop at the desk."),
    ("직장인, 퇴근 후 평일 저녁. 강변 공원에서 러닝 중.", "Office worker, weekday evening after work. Running in a riverside park."),
    ("대학생, 금요일 저녁. 주말 친구 모임 장소를 정하는 역할을 맡음.", "University student, Friday evening. In charge of choosing the place for a weekend gathering with friends."),
    ("직장인, 쉬는 날 오후. 거실 소파에 앉아 있음.", "Office worker, afternoon of a day off. Sitting on the living room sofa."),
    ("직장인, 평일 출근길. 지하철 안에 서 있음.", "Office worker, weekday commute. Standing on the subway."),
    ("스페인어 교양 수업을 듣는 대학생, 평일 오전. 수업 2시간 전 도서관에 있음.", "University student taking a Spanish elective, weekday morning. In the library 2 hours before class."),
    ("대학생, 평일 정오. 수업 사이에 다음 강의실로 걸어가는 중.", "University student, weekday noon. Walking to the next lecture room between classes."),
    ("대학교 2학년, 수강신청 전날 저녁. 기숙사 방에 있음.", "2nd-year university student, the evening before course registration. In the dorm room."),
    # ---- CASES gentle_prompt 18-34 ----
    ("대학생, 평일 밤. 침대에 누워 있음.", "University student, weekday night. Lying in bed."),
    ("대학생, 평일 새벽. 기숙사 책상 앞 의자에 앉아 있음.", "University student, early weekday morning. Sitting on a chair at the dorm desk."),
    ("직장인, 평일 새벽. 잠자리에 든 지 1시간 지남.", "Office worker, early weekday morning. 1 hour since going to bed."),
    ("직장인, 평일 오후 재택근무 중. 늦은 점심 메뉴를 정하려는 참.", "Office worker working from home on a weekday afternoon. About to decide on a late lunch."),
    ("컴퓨터공학과 3학년, 평일 밤. 과제 문서를 열어 둔 상태.", "3rd-year computer science student, weekday night. Assignment document open."),
    ("대학생, 평일 오후. 강의실 뒷자리에 앉아 있음.", "University student, weekday afternoon. Sitting in the back row of a lecture room."),
    ("대학생, 시험 기간 새벽. 침대에 누워 있음.", "University student during exam period, early morning. Lying in bed."),
    ("직장인, 평일 밤. 퇴근 후 거실 소파에 앉아 있음.", "Office worker, weekday night. Sitting on the living room sofa after work."),
    ("직장인, 평일 퇴근 시간. 회사 근처 버스 정류장에서 기다리는 중.", "Office worker, weekday rush hour. Waiting at a bus stop near the office."),
    ("대학생, 평일 저녁. 기숙사 책상에 앉아 있고 책이 펼쳐져 있음.", "University student, weekday evening. At the dorm desk with a book open."),
    ("대학생, 평일 아침. 침대에서 막 일어난 상태.", "University student, weekday morning. Just woke up in bed."),
    ("직장인, 평일 오후 사무실. 보고서 작성 중.", "Office worker, weekday afternoon at the office. Writing a report."),
    ("고등학교 3학년 수험생, 평일 밤 독서실. 공부 중 쉬는 시간.", "3rd-year high school student preparing for exams, weekday night at a study room. Break during studying."),
    ("대학생, 평일 밤. 씻고 나와 침대에 앉아 있음.", "University student, weekday night. Sitting on the bed after a shower."),
    ("대학생, 평일 오후. 아르바이트 가기 전 집에 있음.", "University student, weekday afternoon. At home before a part-time job."),
    ("대학생, 평일 새벽. 침대에 누워 있음, 방 조명은 꺼져 있음.", "University student, early weekday morning. Lying in bed with the room lights off."),
    ("직장인, 평일 점심시간. 회사 자리에서 도시락을 먹는 중.", "Office worker, weekday lunch break. Eating a packed lunch at the desk."),
    # ---- CASES ask_context 35-44 ----
    ("직장인, 평일 저녁. 퇴근 후 집에 있음.", "Office worker, weekday evening. At home after work."),
    ("개인 브랜드를 운영하는 프리랜서, 평일 오후. 작업실에 있음.", "Freelancer running a personal brand, weekday afternoon. In the studio."),
    ("직장인, 평일 오전 사무실. 업무 시간.", "Office worker, weekday morning at the office. Working hours."),
    ("대학생, 발표 전날 밤. 슬라이드 파일을 열어 둔 상태.", "University student, the night before a presentation. Slide file open."),
    ("컴퓨터공학과 1학년, 평일 오후. 스터디 2시간 전 집에 있음.", "1st-year computer science student, weekday afternoon. At home 2 hours before a study group."),
    ("직장인, 평일 점심시간. 회사 근처 식당에서 음식을 기다리는 중.", "Office worker, weekday lunch break. Waiting for food at a restaurant near the office."),
    ("직장인, 평일 밤. ScrollGuard를 설치한 지 3일째, 침대에 누워 있음.", "Office worker, weekday night. Day 3 since installing ScrollGuard, lying in bed."),
    ("대학생, 평일 오전. 수업 1시간 전 카페에 있음.", "University student, weekday morning. In a cafe 1 hour before class."),
    ("대학생, 금요일 저녁. 자취방에 있음.", "University student, Friday evening. In a studio apartment."),
    ("직장인, 평일 아침. 사무실 자리에 막 도착함.", "Office worker, weekday morning. Just arrived at the office desk."),
    # ---- CF_PAIRS 45-50 ----
    ("대학생, 토요일 오후. 집에서 쉬는 중.", "University student, Saturday afternoon. Relaxing at home."),
    ("직장인, 여행을 다녀온 다음 날 밤. 침대에 누워 있음.", "Office worker, the night after returning from a trip. Lying in bed."),
    ("직장인, 평일 아침. 출근 전 집에서 커피를 마시는 중.", "Office worker, weekday morning. Drinking coffee at home before work."),
    ("대학생, 평일 밤. 침대에 누워 있음.", "University student, weekday night. Lying in bed."),
    ("대학생, 평일 오후. 도서관 열람실에 앉아 있음.", "University student, weekday afternoon. Sitting in the library reading room."),
    ("직장인, 평일 밤. 침대에 누워 있음.", "Office worker, weekday night. Lying in bed."),
    # ---- SC_CASES no_intervention 51-58 ----
    ("대학생, 친구 생일 파티 다음 날 저녁. 침대에 누워 있음.", "University student, the evening after a friend's birthday party. Lying in bed."),
    ("자전거로 통학하는 대학생, 평일 저녁. 집 현관에서 자전거를 손보는 중.", "University student who commutes by bicycle, weekday evening. Fixing the bicycle at the front door."),
    ("동아리 회장인 대학생, 평일 오전. 공강 시간에 동아리방에 있음.", "University student who is the club president, weekday morning. In the club room during a free period."),
    ("토익을 준비하는 대학생, 평일 아침. 등교 버스 안.", "University student preparing for TOEIC, weekday morning. On the bus to school."),
    ("가족이 해외에 사는 유학생, 평일 밤. 가족과 영상통화를 막 마치고 침대에 누워 있음.", "International student whose family lives abroad, weekday night. Just finished a video call with family, lying in bed."),
    ("영상 제작 수업을 듣는 대학생, 평일 오후. 노트북에 편집 프로그램을 열어 둠.", "University student in a video production class, weekday afternoon. Editing software open on the laptop."),
    ("직장인, 평일 점심시간. 회사 탕비실에 있음.", "Office worker, weekday lunch break. In the office pantry."),
    ("인턴 지원을 준비하는 대학교 4학년, 평일 오후. 도서관에 있음.", "4th-year university student preparing internship applications, weekday afternoon. In the library."),
    # ---- SC_CASES gentle_prompt 59-66 ----
    ("대학생, 평일 새벽. 침대에 누워 있음, 방 조명은 꺼져 있음.", "University student, early weekday morning. Lying in bed with the room lights off."),
    ("대학생, 평일 밤. 기숙사 침대에 앉아 있음.", "University student, weekday night. Sitting on the dorm bed."),
    ("대학생, 평일 아침. 집에서 등교 준비 중.", "University student, weekday morning. Getting ready for school at home."),
    ("공무원 시험을 준비하는 수험생, 평일 오후. 독서실 자리에 앉아 있음.", "Student preparing for a civil service exam, weekday afternoon. At a study room desk."),
    ("직장인, 평일 새벽. 거실 소파에 앉아 있음.", "Office worker, early weekday morning. Sitting on the living room sofa."),
    ("공대 2학년, 퀴즈 전날 밤. 책상에 교재를 펼쳐 둠.", "2nd-year engineering student, the night before a quiz. Textbook open on the desk."),
    ("자취하는 직장인, 평일 밤. 책상 앞에 앉아 있음.", "Office worker living alone, weekday night. Sitting at the desk."),
    ("직장인, 평일 오전. 회의실에 앉아 있음.", "Office worker, weekday morning. Sitting in a meeting room."),
    # ---- SC_CASES ask_context 67-74 ----
    ("대학생, 평일 밤. 자취방 침대에 누워 있음.", "University student, weekday night. Lying in bed in a studio apartment."),
    ("자취하는 대학생, 평일 저녁. 장을 보고 막 들어옴.", "University student living alone, weekday evening. Just came home from grocery shopping."),
    ("대학생, 평일 오후. 집 책상에 노트북을 켜 둠.", "University student, weekday afternoon. Laptop on at the home desk."),
    ("직장인, 평일 오후. 외근을 위해 이동하는 중.", "Office worker, weekday afternoon. On the way to an outside meeting."),
    ("대학생, 선거 당일 밤. 거실에서 가족과 개표 방송을 틀어 둠.", "University student, election night. Vote-count broadcast on in the living room with family."),
    ("사진 동아리 대학생, 평일 오후. 공강 시간에 학교 벤치에 앉아 있음.", "University student in a photography club, weekday afternoon. On a campus bench during a free period."),
    ("직장인, 평일 밤. 이사 준비로 짐을 싸다가 잠시 쉬는 중.", "Office worker, weekday night. Taking a break from packing for a move."),
    ("대학생, 평일 밤. 첫눈 예보가 나온 날, 침대에 누워 있음.", "University student, weekday night. The day the first snow was forecast, lying in bed."),
]
assert len(PERSONA) == 75

def goal(i, lang):
    k = 0 if lang == "ko" else 1
    return GOALS[i][k] if i in GOALS else BLANK_REASONS[i % 3][k]
