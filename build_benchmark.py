"""
build_benchmark.py  (v3.1)  -  ScrollGuard 벤치마크 81개 생성, 맥락은 관찰 로그 형식 + 사용자 상태 (기존 45개 + 제스처 반례 쌍 12개 + 지름길 반례 24개)
 1) 시나리오 문장에서 '정답을 말해 주는 서술'을 제거하고 관측 가능한 단서만 남김
 2) 숫자 4개(분/세션/중앙세션/집중도)는 micro_datasetA.csv의 실제 행에서 추출
 3) 반례(adversarial)·경계(boundary) 사례를 의도적으로 포함
 4) 입력 / 정답 / 메타(출처·근거) / 독립 라벨링 시트를 한 번에 생성
"""
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
data = pd.read_csv("micro_datasetA.csv").dropna(subset=["Usage_sum_duration", "Usage_num_int_all",
                                                         "Usage_median_duration", "Usage_quadratic"])

TIERS = {
    "trivial":   "Usage_sum_duration<300 and 1<=Usage_num_int_all<=3",
    "mid":       "600<=Usage_sum_duration<1200 and Usage_num_int_all<=6",
    "mid_budget":"600<=Usage_sum_duration<=900 and Usage_num_int_all<=5",
    "mid_long":  "1200<=Usage_sum_duration<1800 and Usage_num_int_all<=8",
    "long":      "1800<=Usage_sum_duration and Usage_num_int_all<=9",
    "long_conc": "1800<=Usage_sum_duration and Usage_quadratic>=0.9",
    "xlong":     "2700<=Usage_sum_duration and Usage_num_int_all<=12",
    "frag":      "Usage_num_int_all>=8 and 300<=Usage_sum_duration<=900",
    "many_long": "Usage_num_int_all>=10 and Usage_sum_duration>=1200",
}
NO, GP, AC = "no_intervention", "gentle_prompt", "ask_context"
IN, UN, UC = "intentional", "unintended", "unclear"
nan = np.nan

# (tier, 시각, 진입계기, 앱유형, 스와이프간격CV, 평균체류초, 사용자선언목표, 최근맥락, intent, action, 유형, 라벨근거)
CASES = [
 # ---------------- no_intervention (18) ----------------
 ("long","21:10","search","long_video",1.10,260,"운영체제 중간고사 대비 강의 시청","진입: 검색어 'OS 페이징 강의' 입력 후 재생목록 열기 / 앱 안: 재생목록 순서대로 강의 4편 시청, 2배속, 구간 되감기 7회 / 앱 전환: 노트 앱 3회(각 2~4분) / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 11/05 운영체제 중간고사",IN,NO,"adversarial","길지만 선언 목표·행동 일치"),
 ("long","14:20","self_open","reader",2.00,150,"","진입: 홈 화면에서 전자책 앱 직접 실행 / 앱 안: 이어 읽기로 312쪽부터 시작, 페이지 넘김 간격 1~3분, 밑줄 2회 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 18:00 헬스장",IN,NO,"standard","독서 앱, 자율적 사용"),
 ("mid","12:30","search","long_video",0.90,180,"된장찌개 레시피 확인","진입: 검색어 '된장찌개 레시피' 입력 / 앱 안: 검색 2회, 레시피 영상 3개 시청, 재료 목록 부분에서 일시정지 4회, 마지막 영상 후 앱 종료 / 앱 전환: 메모 앱 1회(재료 목록 작성) / 상호작용: 검색 2, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 14:00 수업",IN,NO,"standard","검색→목적 달성→종료"),
 ("trivial","08:10","self_open","map",nan,nan,"","진입: 홈 화면에서 지도 앱 직접 실행 / 앱 안: 목적지 '공학관' 검색, 도보 길찾기 시작 후 앱 종료 / 앱 전환: — / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 09:00 수업",IN,NO,"standard","짧은 유틸리티 사용"),
 ("trivial","13:00","notification","messenger",nan,nan,"","진입: 친구 메시지 알림 탭 / 앱 안: 대화방 1개 열람, 답장 1회 전송 후 앱 종료 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 14:00 수업",IN,NO,"standard","짧은 유틸리티 사용"),
 ("mid_budget","22:00","self_open","long_video",1.00,150,"","진입: 홈 화면에서 직접 실행 / 앱 안: 구독 채널 새 영상 2개 시청, 좋아요 1회 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 1, 댓글 0 / 일정·설정: 사용자가 설정한 휴식 타이머 20분 중 {m}분 경과, 취침 목표 24:00",IN,NO,"boundary","자기 설정 예산 내"),
 ("long","16:00","self_open","sns_feed",0.90,40,"여행 사진 앨범 정리 후 친구들과 공유","진입: 직접 실행 / 앱 안: 사진 14장 업로드, 캡션 작성 5회, 댓글 답글 9개, 업로드 사이 피드 스크롤은 각 1분 미만 / 앱 전환: 갤러리 앱 4회(사진 선택) / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 9 / 일정·설정: 19:00 저녁 약속",IN,NO,"adversarial","SNS+긴 시간이지만 생산 활동"),
 ("long_conc","10:00","search","long_video",1.30,600,"학회 발표 영상 시청","진입: 검색어 'CHI 2026 발표' 입력 / 앱 안: 영상 1개를 이어서 재생 중(러닝타임 52분, 현재 {m}분 지점), 일시정지 4회, 자막 켬 / 앱 전환: 문서 앱 2회(메모 작성) / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 오늘 15:00 연구실 미팅",IN,NO,"adversarial","사용 집중도 높아도 목적적"),
 ("frag","09:30","self_open","messenger",nan,nan,"조별과제 마감 전 단톡 확인","진입: 홈 화면에서 직접 실행 / 앱 안: 앱 {s}회 실행, 조별과제 단체방 메시지 23개 열람, 메시지 6개 전송, 파일 2개 공유 / 앱 전환: 공유 문서 앱과 번갈아 사용, 문서 편집 기록 4건 / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 오늘 12:00 조별과제 마감",IN,NO,"adversarial","세션 많지만 과업 수행"),
 ("mid","19:00","notification","shopping",0.80,50,"어제 담아둔 노트북 가격 비교","진입: 가격 하락 알림 탭 / 앱 안: 장바구니 상품 3개 사양 비교표 열람, 리뷰 6개 열람, 1개 결제 후 앱 종료 / 앱 전환: 카드사 앱 1회(결제 인증) / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 내일 10:00 수업",IN,NO,"standard","명확한 과업 완료"),
 ("long","23:30","search","long_video",1.00,220,"내일 제출할 과제 참고 영상 시청 후 취침","진입: 검색어 '머신러닝 과제 정규화 예제' 입력 / 앱 안: 검색 4회, 영상 5개 시청(제목 모두 정규화 관련), 일시정지 9회 / 앱 전환: 노트 앱 5회(코드 메모) / 상호작용: 검색 4, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 내일 09:00 과제 마감, 취침 목표 01:00",IN,NO,"adversarial","심야·장시간이지만 목적적"),
 ("mid","17:30","self_open","podcast",nan,nan,"","진입: 러닝 앱에서 운동 시작과 함께 실행 / 앱 안: 팟캐스트 에피소드 1개 재생, 화면 꺼짐 상태 비율 92% / 앱 전환: 러닝 앱 기록 진행 중(5.2km) / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 19:30 저녁",IN,NO,"standard","화면 소비 없음"),
 ("mid","20:10","search","sns_feed",0.60,35,"주말 모임 맛집 후기 검색 후 예약","진입: 검색어 '성수 브런치' 입력 / 앱 안: 지역명 검색 3회, 후기 게시물 11개 열람, 게시물 2개 저장 / 앱 전환: 예약 앱 1회, 토요일 11:30 4명 예약 완료 / 상호작용: 검색 3, 저장 2, 좋아요 0, 댓글 0 / 일정·설정: 토요일 11:30 모임 등록",IN,NO,"standard","SNS를 검색 도구로 사용"),
 ("long","15:00","self_open","long_video",1.20,300,"오늘은 쉬는 날이라 영화 한 편을 보겠다고 직접 선언","진입: 직접 실행 / 앱 안: 영화 1편 재생 중(러닝타임 128분, 현재 {m}분 지점), 일시정지 1회 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 사용자가 집중 모드 켬, 내일 10:00 회의",IN,NO,"boundary","사용자 자율 선언 존중"),
 ("frag","08:30","self_open","email",nan,nan,"출근길 메일 확인","진입: 직접 실행 / 앱 안: 메일 앱 {s}회 실행, 회당 10초 안팎, 받은편지함 새 메일 확인 후 1건 답장 / 앱 전환: 지하철 노선 앱 2회 / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 위치 상태 이동 중, 09:00 출근",IN,NO,"standard","이동 중 유틸리티"),
 ("mid","11:00","search","short_video",0.45,35,"쇼츠로 스페인어 표현 공부","진입: 검색어 '스페인어 일상 표현' 입력 / 앱 안: 쇼츠 17개 시청, 영상마다 일시정지, 저장 8회 / 앱 전환: 퀴즈 앱 2회(단어 입력) / 상호작용: 검색 1, 저장 8, 좋아요 0, 댓글 0 / 일정·설정: 13:00 스페인어 수업",IN,NO,"adversarial","숏폼이지만 학습 행동"),
 ("trivial","12:00","self_open","music",nan,nan,"","진입: 직접 실행 / 앱 안: 저장된 플레이리스트 재생 후 앱 종료 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 13:00 수업",IN,NO,"standard","짧은 유틸리티 사용"),
 ("mid","18:40","self_open","community",0.70,60,"다음 학기 수강신청 후기 확인","진입: 직접 실행 / 앱 안: 검색어 '교양 추천' 입력, 글 5개 열람(각 1~2분), 글 1개 스크랩 / 앱 전환: 시간표 앱 1회(과목 2개 추가) / 상호작용: 검색 1, 저장 1, 좋아요 0, 댓글 0 / 일정·설정: 내일 10:00 수강신청 시작",IN,NO,"standard","명확한 정보 탐색"),
 # ---------------- gentle_prompt (17) ----------------
 ("long","23:50","notification","short_video",0.12,6,"","진입: 추천 영상 알림 탭 / 앱 안: 추천 피드에서 쇼츠 412개 연속 재생 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 내일 09:00 수업",UN,GP,"standard","다중 자동조종 신호"),
 ("long","01:10","self_open","sns_feed",0.18,5,"","진입: 직접 실행 / 앱 안: 홈 피드 게시물 690개 스크롤 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 사용자 취침 목표 24:00(70분 경과), 내일 09:00 수업",UN,GP,"standard","취침 목표 초과+수동 소비"),
 ("frag","02:00","self_open","sns_feed",0.20,8,"","진입: 화면을 켤 때마다 직접 실행 / 앱 안: 앱 {s}회 실행, 회당 40초 안팎, 매번 피드 상단 새로고침 후 종료 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00, 내일 08:00 알람",UN,GP,"adversarial","총 시간은 짧아도 강박적 재확인"),
 ("mid_long","15:00","self_open","long_video",0.50,90,"점심 식단 영상 하나만 보기","진입: 직접 실행 / 앱 안: 식단 영상 1개 시청 후 자동 재생으로 5개 이어 시청, 카테고리 순서: 요리 → 게임 → 연예 → 스포츠 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 16:00 팀 회의",UN,GP,"standard","선언 목표에서 이탈"),
 ("long","21:30","search","short_video",0.15,7,"과제 자료 검색","진입: 검색어 '데이터베이스 정규화' 입력 / 앱 안: 검색 2회, 검색 결과 영상 1개 시청 후 쇼츠 피드로 이동해 영상 290개 연속 재생 / 앱 전환: — / 상호작용: 검색 2, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 내일 23:59 데이터베이스 과제 마감",UN,GP,"standard","검색 후 피드로 이탈"),
 ("long","13:20","notification","sns_feed",0.20,6,"","진입: 친구 게시물 알림 탭 / 앱 안: 알림 게시물 확인 후 홈 피드 게시물 370개 스크롤, 좋아요 2 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 2, 댓글 0 / 일정·설정: 13:00–14:15 수업, 위치: 강의동",UN,GP,"boundary","일정과 충돌하는 수동 소비"),
 ("many_long","03:40","self_open","sns_feed",0.20,5,"","진입: 직접 실행 / 앱 안: 앱 {s}회 실행, 종료 직후 재실행 반복, 매번 같은 피드 상단 새로고침 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 01:00, 내일 08:00 알람",UN,GP,"adversarial","세션 다수+심야+수동"),
 ("long","22:00","self_open","shopping",0.25,8,"","진입: 직접 실행 / 앱 안: 추천 탭 상품 350개 스크롤, 상세 페이지 열람 2회(각 10초 안팎) / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 21:00–23:00 온라인 강의 수강 시간, 취침 목표 24:00",UN,GP,"standard","목적 행동 부재"),
 ("mid","18:00","notification","news",0.30,10,"","진입: 속보 알림 탭 / 앱 안: 기사 1개의 댓글 목록 열람, 댓글 목록 새로고침 14회, 기사 본문 체류 5초 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 19:00 저녁 약속",UN,GP,"standard","수동적 반복 소비"),
 ("long","20:40","self_open","sns_feed",0.20,6,"","진입: 직접 실행 / 앱 안: 홈 피드 게시물 590개 스크롤, 좋아요 1 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 1, 댓글 0 / 일정·설정: 15분 전 ScrollGuard 기록에 사용자가 '이제 그만 봐야겠다'고 입력",UN,GP,"standard","사용자 본인 의사와 행동 불일치"),
 ("mid","09:00","self_open","sns_feed",0.25,7,"","진입: 기상 알람 해제 직후 직접 실행 / 앱 안: 홈 피드 게시물 95개 스크롤, 스토리 12개 시청 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 오늘 첫 일정 09:15 수업, 통학 25분",UN,GP,"standard","일정 직전 수동 소비"),
 ("many_long","14:00","self_open","sns_feed",0.30,5,"","진입: 작업 앱 사용 중 잠금 해제 후 직접 실행 / 앱 안: 앱 {s}회 실행, 매번 홈 화면 확인 후 20초 안에 종료, 새 알림 0건 / 앱 전환: 문서 작업 앱과 번갈아 사용 / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 13:00–17:00 집중 작업 시간",UN,GP,"adversarial","알림 없는 반복 재확인"),
 ("long","20:00","self_open","short_video",0.20,8,"공부 중 쉬는 시간 10분","진입: 직접 실행 / 앱 안: 쇼츠 피드 영상 370개 연속 재생 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 사용자가 휴식 타이머 10분 설정, 타이머 종료 후 현재까지 사용 {m}분",UN,GP,"standard","자기 설정 예산 초과"),
 ("mid","22:30","notification","sns_feed",0.30,12,"","진입: 친구 DM 알림 탭 / 앱 안: DM 확인 후 답장 1회, 이어서 탐색 탭으로 이동해 게시물 50개 스크롤 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 22:00(30분 경과)",UN,GP,"standard","과업 종료 후 피드 이탈"),
 ("long","16:30","self_open","short_video",0.12,5,"5분만 쇼츠 보기","진입: 직접 실행 / 앱 안: 시작할 때 사용자가 '5분만'이라고 입력, 쇼츠 영상 520개 연속 재생, 현재 사용 {m}분 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 18:00 아르바이트",UN,GP,"standard","자기 설정 시간 초과"),
 ("xlong","02:20","notification","short_video",0.10,4,"","진입: 추천 영상 알림 탭 / 앱 안: 쇼츠 영상 780개 연속 재생, 일시정지 0 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 01:00, 내일 09:00 수업",UN,GP,"standard","다중 자동조종 신호"),
 ("many_long","12:40","self_open","community",0.25,9,"","진입: 직접 실행 / 앱 안: 앱 {s}회 실행, 게시글 목록 새로고침 41회, 맨 위로 이동 반복, 글 본문 체류 평균 9초 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 13:30 회의",UN,GP,"standard","수동적 반복 확인"),
 # ---------------- ask_context (10) ----------------
 ("long","20:00","self_open","long_video",0.80,90,"","진입: 직접 실행 / 앱 안: 검색 1회('파스타 만들기') 후 추천 영상으로 이동, 시청 영상 8개 카테고리: 요리 3, 게임 3, 뉴스 2 / 앱 전환: — / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 내일 10:00 회의",UC,AC,"boundary","목적·이탈 신호 혼재"),
 ("long","15:30","self_open","sns_feed",0.50,20,"","진입: 직접 실행 / 앱 안: 게시물 작성 1회, 이후 피드 게시물 130개 스크롤, 다른 계정 게시물에 댓글 2개 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 2 / 일정·설정: 계정에 비즈니스 계정 태그, 15:00–18:00 '콘텐츠 작업'",UC,AC,"boundary","업무/여가 구분 불가"),
 ("frag","11:00","self_open","browser",nan,nan,"","진입: 직접 실행 / 앱 안: 브라우저 {s}회 실행, 회당 25초 안팎, 방문 페이지와 검색어 기록은 사용자 설정으로 접근 제한 / 앱 전환: 메신저 3회 / 상호작용: 검색·저장·좋아요·댓글 기록 접근 제한 / 일정·설정: 15:00 회의",UC,AC,"boundary","핵심 정보 접근 불가"),
 ("mid_long","00:10","search","long_video",0.40,60,"내일 발표 자료 참고","진입: 검색어 '발표 오프닝 예시' 입력 / 앱 안: 영상 5개 시청, 그중 1개만 제목이 발표 주제와 일치 / 앱 전환: 슬라이드 앱 1회(2분) / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00(10분 경과), 내일 10:00 발표",UC,AC,"boundary","목표 일부 일치+예산 초과"),
 ("long","17:00","self_open","short_video",0.30,12,"학습용 쇼츠 보기","진입: 직접 실행 / 앱 안: 쇼츠 영상 300개 재생, 그중 제목이 학습 주제인 영상 70개, 일시정지 0 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 19:00 스터디",UC,AC,"boundary","선언 목표와 행동 부분 일치"),
 ("mid","12:40","notification","sns_feed",0.35,15,"","진입: 친구 게시물 알림 탭 / 앱 안: 게시물 확인 후 홈 피드 게시물 40개 스크롤, 좋아요 3 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 3, 댓글 0 / 일정·설정: 점심시간, 15:00 회의",UC,AC,"standard","정상 휴식 vs 습관 구분 불가"),
 ("long","22:30","self_open","long_video",1.00,200,"","진입: 직접 실행 / 앱 안: 앱 {s}회 실행, 그중 1회에 긴 영상 2편(각 25분 안팎) 연속 시청 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 이 사용자의 야간 시청 기록은 이번이 첫 관측, 취침 목표 02:00",UC,AC,"standard","기준 이력 없음"),
 ("mid","10:00","self_open","shopping",0.50,45,"","진입: 직접 실행 / 앱 안: 상품 상세 페이지 12개 열람(카테고리: 운동화), 찜 2회, 결제 0 / 앱 전환: 가격 비교 앱 1회 / 상호작용: 검색 0, 저장 2, 좋아요 0, 댓글 0 / 일정·설정: 11:00 수업",UC,AC,"standard","비교 탐색 vs 충동 구분 불가"),
 ("long","19:00","self_open","community",0.60,30,"","진입: 직접 실행 / 앱 안: 글 26개 열람, 댓글 3개 작성, 인기글 목록 새로고침 5회 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 3 / 일정·설정: 내일 09:00 수업",UC,AC,"standard","소통 vs 소비 구분 불가"),
 ("mid","09:00","self_open","news",0.50,40,"","진입: 직접 실행 / 앱 안: 경제 기사 1개 정독(4분) 후 연관 기사 링크로 이동 9회, 기사당 체류 40초 안팎 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 09:30 업무 시작",UC,AC,"standard","정보 수집 vs 습관 구분 불가"),
]

used = set()
rows = []
for i, (tier, t, trig, app, cv, dwell, goal, ctx, intent, action, ctype, why) in enumerate(CASES):
    pool = data.query(TIERS[tier])
    pool = pool[~pool.index.isin(used)]
    idx = rng.choice(pool.index.values)
    used.add(idx)
    r = data.loc[idx]
    minutes = round(r.Usage_sum_duration / 60, 1)
    rows.append(dict(
        time_of_day=t, entry_trigger=trig, app_type=app, user_stated_goal=goal,
        spec=i,
        recent_context=ctx.format(m=int(round(minutes)), s=int(r.Usage_num_int_all)),
        usage_minutes=minutes, usage_sessions=int(r.Usage_num_int_all),
        median_session_sec=round(r.Usage_median_duration, 1), usage_quadratic=round(r.Usage_quadratic, 3),
        swipe_interval_cv=cv, avg_dwell_sec=dwell,
        user_intent=intent, intervention_needed={NO: "no", GP: "yes", AC: "uncertain"}[action],
        expected_action=action, case_type=ctype, label_rationale=why,
        src_user_id=int(r.user_id), src_questionnaire_id=int(r.es_questionnaire_id),
    ))

df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)  # 순서로 라벨 유추 방지
df.insert(0, "case_id", [f"S{i+1:03d}" for i in range(len(df))])

# ---------------- 제스처 반례 쌍 (counterfactual, S046~) ----------------
# 같은 맥락·같은 실제 사용량 행에 제스처(스와이프 간격 CV, 평균 체류)만 바꾼 쌍.
# 정답은 맥락으로 정해지므로 두 사례의 정답은 같다. 쌍 중 하나는 반드시 'CV 임계값' 지름길과 반대 방향이다.
# 기존 45개의 추출·셔플·ID는 바꾸지 않는다(별도 시드, 기존 행 제외, 뒤에 이어 붙임).
# (pair_id, tier, 시각, 진입계기, 앱유형, 사용자선언목표, 최근맥락, intent, action, 라벨근거, (CV_a, 체류_a), (CV_b, 체류_b))
CF_PAIRS = [
 ("P1","long","14:00","self_open","comic","주말에 완결 웹툰 정주행","진입: 직접 실행 / 앱 안: 구독 중인 완결 웹툰을 1화부터 순서대로 열람, 현재 23화 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 주말, 월요일 09:00 수업",IN,NO,"선언 목표와 행동 일치(제스처와 무관)",(0.15,6),(0.95,40)),
 ("P2","long","21:00","self_open","gallery","여행 사진 1000장 중 흔들린 사진 삭제","진입: 직접 실행 / 앱 안: 여행 앨범 사진을 한 장씩 넘기며 삭제 58회, 즐겨찾기 12회, 새 앨범 2개 생성 / 앱 전환: — / 상호작용: 검색 0, 저장 12, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00",IN,NO,"정리 과업 수행 중(제스처와 무관)",(0.10,2),(0.90,25)),
 ("P3","mid","08:00","search","news","긴 탐사보도 기사 끝까지 읽기","진입: 검색어 '플랫폼 노동 탐사보도' 입력 / 앱 안: 기사 1개 본문을 끝까지 열람, 하이라이트 3회 / 앱 전환: 메모 앱 1회(링크 공유) / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 09:30 출근",IN,NO,"목적적 읽기(제스처와 무관)",(0.20,9),(1.00,45)),
 ("P4","long","00:40","self_open","sns_feed","","진입: 직접 실행 / 앱 안: 홈 피드 게시물 탐색 계속 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 사용자 취침 목표 23:00(100분 경과), 내일 09:00 수업",UN,GP,"취침 목표 초과+목적 행동 부재(제스처와 무관)",(0.15,5),(1.10,25)),
 ("P5","mid_long","15:00","notification","short_video","","진입: 추천 영상 알림 탭 / 앱 안: 추천 영상만 이어서 재생 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 14:30–17:00 과제 작업 시간",UN,GP,"알림 후 이탈+일정 충돌(제스처와 무관)",(0.12,5),(0.95,30)),
 ("P6","mid","23:40","self_open","shopping","","진입: 직접 실행 / 앱 안: 추천 상품 목록 탐색, 상세 페이지 열람 후 바로 뒤로 가기 반복 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 15분 전 ScrollGuard 기록에 사용자가 '이제 자야지'라고 입력",UN,GP,"본인 의사와 행동 불일치+목적 행동 부재(제스처와 무관)",(0.20,6),(1.00,35)),
]
rng_cf = np.random.default_rng(SEED + 1)
cf_rows = []
for j, (pid, tier, t, trig, app, goal, ctx, intent, action, why, ga, gb) in enumerate(CF_PAIRS):
    pool = data.query(TIERS[tier]); pool = pool[~pool.index.isin(used)]
    idx = rng_cf.choice(pool.index.values); used.add(idx); r = data.loc[idx]
    minutes = round(r.Usage_sum_duration / 60, 1)
    for variant, (cv, dwell) in (("a", ga), ("b", gb)):
        cf_rows.append(dict(
            time_of_day=t, entry_trigger=trig, app_type=app, user_stated_goal=goal,
            spec=len(CASES) + j,
            recent_context=ctx.format(m=int(round(minutes)), s=int(r.Usage_num_int_all)),
            usage_minutes=minutes, usage_sessions=int(r.Usage_num_int_all),
            median_session_sec=round(r.Usage_median_duration, 1), usage_quadratic=round(r.Usage_quadratic, 3),
            swipe_interval_cv=cv, avg_dwell_sec=dwell,
            user_intent=intent, intervention_needed={NO: "no", GP: "yes", AC: "uncertain"}[action],
            expected_action=action, case_type="counterfactual", label_rationale=why,
            src_user_id=int(r.user_id), src_questionnaire_id=int(r.es_questionnaire_id), pair_id=f"{pid}{variant}",
        ))
cf = pd.DataFrame(cf_rows).sample(frac=1, random_state=SEED).reset_index(drop=True)  # 쌍이 연달아 나오지 않게
cf.insert(0, "case_id", [f"S{len(df)+i+1:03d}" for i in range(len(cf))])
df["pair_id"] = ""
df = pd.concat([df, cf], ignore_index=True)

# ---------------- 지름길 반례 (S058~) ----------------
# 맥락을 읽지 않고 숫자·범주 하나로 맞히는 지름길을 깨는 사례. 각 사례는 아래 지름길 중 여럿과 반대 방향이다.
#   체류<35초·CV<0.6 -> 개입 / 선언 목표 있음 -> 개입 안 함 / sns_feed·short_video -> 개입
#   제스처 값 없음 -> 개입 안 함 / 심야 -> 개입 / 문장이 길면 개입
# 형식은 CASES와 같다. 기존 57개는 바꾸지 않는다(별도 시드, 뒤에 이어 붙임).
SC_CASES = [
 # ---- no_intervention: 제스처·앱·시각은 '개입' 쪽, 맥락은 목적 행동 ----
 ("long","19:30","self_open","sns_feed",0.25,8,"","진입: 직접 실행 / 앱 안: 친구 생일 파티 사진이 올라온 단체 계정 게시물 열람, 댓글 작성 12회, 사진 저장 9회 / 앱 전환: 메신저 1회(사진 9장 전달) 후 앱 종료 / 상호작용: 검색 0, 저장 9, 좋아요 0, 댓글 12 / 일정·설정: 취침 목표 24:00",IN,NO,"shortcut","목표 선언 없지만 생산적 상호작용"),
 ("mid","18:00","search","short_video",0.30,10,"","진입: 검색어 '자전거 체인 교체' 입력 / 앱 안: 관련 쇼츠 4개 시청, 일시정지 6회 / 앱 전환: 쇼핑 앱 1회(체인 공구 구매) 후 앱 종료 / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 19:00 자전거 동호회",IN,NO,"shortcut","검색→목적 달성→종료"),
 ("mid_long","10:30","notification","sns_feed",0.40,12,"","진입: 동아리 계정 알림 탭 / 앱 안: 행사 안내 게시물 작성·수정 3회, 신청자 댓글에 답글 15개 / 앱 전환: 일정 앱 1회(행사 일정 등록) / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 15 / 일정·설정: 11/20 동아리 행사",IN,NO,"shortcut","SNS에서 과업 수행"),
 ("mid","07:40","self_open","education",0.15,3,"","진입: 직접 실행 / 앱 안: 단어 카드를 넘기며 정답 확인, 오늘 목표 200장 중 180장 완료, 틀린 카드 23장 재학습 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 09:00 토익 수업",IN,NO,"shortcut","균일한 넘김이지만 학습 과업"),
 ("long","23:30","self_open","sns_feed",0.20,6,"","진입: 직접 실행 / 앱 안: 가족 단체 계정 게시물 열람, 좋아요 14, 댓글 6, 사진 3장 업로드 / 앱 전환: 직전 30분 영상통화 앱 사용 / 상호작용: 검색 0, 저장 0, 좋아요 14, 댓글 6 / 일정·설정: 취침 목표 01:00",IN,NO,"shortcut","심야·SNS지만 생산적 상호작용"),
 ("long","13:00","self_open","short_video",0.35,15,"","진입: 직접 실행 / 앱 안: 편집 기법 쇼츠 시청, 저장 5회 / 앱 전환: 영상 편집 앱 6회, 편집 프로젝트 저장 3회 / 상호작용: 검색 0, 저장 5, 좋아요 0, 댓글 0 / 일정·설정: 18:00 영상 과제 마감",IN,NO,"shortcut","숏폼을 참고 자료로 사용"),
 ("frag","12:10","notification","shopping",0.30,9,"","진입: 택배 도착 알림 탭 / 앱 안: 앱 {s}회 실행, 배송 조회 2회, 반품 신청 3건 처리(건마다 사진 첨부) / 앱 전환: 고객센터 채팅 1회, 카메라 3회 / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 13:00 회의",IN,NO,"shortcut","세션 많지만 과업 수행"),
 ("mid","16:00","search","sns_feed",0.30,14,"인턴 지원 회사 계정의 최근 게시물 확인","진입: 회사 계정 이름 검색 / 앱 안: 회사 계정 게시물을 최신순으로 열람, 채용 관련 게시물 4개 저장 / 앱 전환: 문서 앱 2회(지원서 메모) / 상호작용: 검색 1, 저장 4, 좋아요 0, 댓글 0 / 일정·설정: 11/10 인턴 지원 마감",IN,NO,"shortcut","SNS를 정보 수집 도구로 사용"),
 # ---- gentle_prompt: 제스처는 '불규칙·긴 체류', 목표 선언 있음·SNS 아님·값 없음 등 ----
 ("xlong","02:10","self_open","long_video",0.90,240,"","진입: 직접 실행 / 앱 안: 자동 재생으로 영상 11개 이어 재생, 화면 터치 2회 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00(130분 경과), 내일 09:00 수업",UN,GP,"shortcut","취침 목표 초과+수동 시청"),
 ("mid_long","22:40","notification","community",0.80,50,"공지 하나 확인","진입: 학과 게시판 공지 알림 탭 / 앱 안: 공지 1개 확인(30초) 후 인기글 목록으로 이동, 앱 {s}회 실행, 글 20개 열람 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00",UN,GP,"shortcut","선언 목표에서 이탈+목적 행동 부재"),
 ("mid","08:50","self_open","news",0.70,45,"오늘 날씨 확인","진입: 직접 실행 / 앱 안: 날씨 탭 확인(15초) 후 연예 기사 22개 연속 열람 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 09:00 수업",UN,GP,"shortcut","선언 목표에서 이탈+일정 충돌"),
 ("many_long","14:30","self_open","messenger",nan,nan,"","진입: 직접 실행 / 앱 안: 메신저 {s}회 실행, 실행 때마다 새 메시지 0개, 바로 종료 / 앱 전환: 공부 타이머 앱과 번갈아 실행 / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 13:00–17:00 공부 시간",UN,GP,"shortcut","반복 재확인+일정 충돌"),
 ("frag","01:30","self_open","browser",nan,nan,"","진입: 직접 실행 / 앱 안: 브라우저 {s}회 실행, 같은 커뮤니티 홈페이지 새로고침, 글 열람 0 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00(90분 경과)",UN,GP,"shortcut","반복 새로고침+취침 목표 초과"),
 ("long","21:00","search","long_video",0.85,120,"강의 1개 듣기","진입: 검색어 '선형대수 고유값 강의' 입력 / 앱 안: 강의 1개 시청 후 추천 영상 4개 연속 시청, 카테고리 순서: 강의 → 예능 → 게임 → 예능 → 게임 / 앱 전환: — / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 내일 10:00 선형대수 퀴즈",UN,GP,"shortcut","선언 목표에서 이탈"),
 ("long","23:10","search","shopping",0.75,40,"생필품 2개 주문","진입: 검색어 '주방 세제' 입력 / 앱 안: 생필품 2개 결제, 이후 추천 상품 목록에서 상품 60개 스크롤 / 앱 전환: — / 상호작용: 검색 1, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 23:00(10분 경과)",UN,GP,"shortcut","과업 종료 후 이탈+취침 목표 초과"),
 ("mid_long","11:00","notification","news",0.65,38,"","진입: 속보 알림 탭 / 앱 안: 속보 기사 확인 후 기사 30개 이동, 카테고리: 연예 12, 스포츠 10, 사회 8, 기사마다 댓글 목록 열람 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 10:30–12:00 팀 회의",UN,GP,"shortcut","알림 후 이탈+일정 충돌"),
 # ---- ask_context 보강: 목적 신호와 이탈 신호가 섞이거나 핵심 정보 접근 불가 ----
 ("long","21:00","self_open","sns_feed",0.25,6,"","진입: 직접 실행 / 앱 안: 피드 게시물 230개 스크롤 중 게시물 2개 저장, 저장 직후 다시 스크롤 / 앱 전환: — / 상호작용: 검색 0, 저장 2, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 01:00",UC,AC,"boundary","저장 행동과 연속 탐색 혼재"),
 ("long","18:30","search","short_video",0.20,7,"요리 쇼츠 참고","진입: 검색어 '원팬 파스타' 입력 / 앱 안: 요리 쇼츠 3개 저장, 이어서 피드 영상 210개 재생, 카테고리: 요리 110, 여행 100 / 앱 전환: — / 상호작용: 검색 1, 저장 3, 좋아요 0, 댓글 0 / 일정·설정: 19:30 저녁 준비",UC,AC,"boundary","목표 일부 일치+이탈 신호"),
 ("xlong","13:30","self_open","long_video",1.00,300,"","진입: 직접 실행 / 앱 안: 다큐멘터리 시리즈 3편 연속 시청, 시청 영상 모두 같은 시리즈 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 14:00 과제 마감",UC,AC,"boundary","일관된 시청 vs 일정 임박"),
 ("frag","16:00","notification","messenger",nan,nan,"","진입: 메시지 알림 탭 / 앱 안: 메신저 {s}회 실행, 대화방 내용과 열람 기록은 사용자 설정으로 접근 차단 / 앱 전환: — / 상호작용: 검색·저장·좋아요·댓글 기록 접근 제한 / 일정·설정: 18:00 저녁 약속",UC,AC,"boundary","핵심 정보 접근 불가"),
 ("long","23:50","notification","news",0.30,12,"선거 결과 확인","진입: 개표 속보 알림 탭 / 앱 안: 개표 속보와 해설 기사 18개 열람, 같은 개표 현황 기사 새로고침 6회 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00(10분 남음)",UC,AC,"boundary","목표 일치 vs 반복 새로고침·취침 임박"),
 ("mid_long","15:00","self_open","community",0.70,40,"","진입: 직접 실행 / 앱 안: 중고거래 게시판에서 같은 품목(카메라) 글 반복 열람, 판매자에게 채팅 1회, 거래 완료 기록 0건 / 앱 전환: 가격 비교 앱 2회 / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 19:00 저녁 약속",UC,AC,"boundary","비교 탐색 vs 반복 열람 구분 어려움"),
 ("long","23:00","self_open","gallery",0.40,20,"","진입: 직접 실행 / 앱 안: 3년 전 사진을 넘기며 열람, 삭제 0, 편집 0, 공유 0, 일부 사진에서 2분 이상 체류 / 앱 전환: — / 상호작용: 검색 0, 저장 0, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00",UC,AC,"boundary","회상 vs 수동 소비 구분 어려움"),
 ("long","22:00","search","shopping",0.25,8,"겨울 패딩 고르기","진입: 검색어 '롱패딩' 입력 / 앱 안: 패딩 상세 페이지 23개 열람, 찜 1회, 이후 신발·가방 상품 상세 페이지 17개 열람 / 앱 전환: — / 상호작용: 검색 1, 저장 1, 좋아요 0, 댓글 0 / 일정·설정: 취침 목표 24:00",UC,AC,"boundary","목표 일부 일치+이탈 신호"),
]
rng_sc = np.random.default_rng(SEED + 2)
sc_rows = []
for j, (tier, t, trig, app, cv, dwell, goal, ctx, intent, action, ctype, why) in enumerate(SC_CASES):
    pool = data.query(TIERS[tier]); pool = pool[~pool.index.isin(used)]
    idx = rng_sc.choice(pool.index.values); used.add(idx); r = data.loc[idx]
    minutes = round(r.Usage_sum_duration / 60, 1)
    sc_rows.append(dict(
        time_of_day=t, entry_trigger=trig, app_type=app, user_stated_goal=goal,
        spec=len(CASES) + len(CF_PAIRS) + j,
        recent_context=ctx.format(m=int(round(minutes)), s=int(r.Usage_num_int_all)),
        usage_minutes=minutes, usage_sessions=int(r.Usage_num_int_all),
        median_session_sec=round(r.Usage_median_duration, 1), usage_quadratic=round(r.Usage_quadratic, 3),
        swipe_interval_cv=cv, avg_dwell_sec=dwell,
        user_intent=intent, intervention_needed={NO: "no", GP: "yes", AC: "uncertain"}[action],
        expected_action=action, case_type=ctype, label_rationale=why,
        src_user_id=int(r.user_id), src_questionnaire_id=int(r.es_questionnaire_id), pair_id="",
    ))
sc = pd.DataFrame(sc_rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
sc.insert(0, "case_id", [f"S{len(df)+i+1:03d}" for i in range(len(sc))])
df = pd.concat([df, sc], ignore_index=True)

# ---------------- 사용자 상태 (v3.1) ----------------
# user_stated_goal: 사용자가 ScrollGuard 목표 질문에 입력한 문장(비어 있으면 그 이유), user_state: 누구인지·어떤 날인지·지금 어디서 무엇을 하는지
# self_reported_mood: 원본 행의 실제 기분 자기보고(valence, 1~6), typical_usage_minutes: 같은 사용자의 다른 측정 기록에서 구한 1시간 사용량 중앙값(실제 값)
import user_states as US
df["user_stated_goal"] = [US.goal(k, "ko") for k in df.spec]
df["user_state"] = [US.PERSONA[k][0] for k in df.spec]
by_user = data.groupby("user_id")
mood, typical = [], []
for u, q in zip(df.src_user_id, df.src_questionnaire_id):
    g = by_user.get_group(u)
    mood.append(g.loc[g.es_questionnaire_id == q, "valence"].iloc[0])
    other = g[g.es_questionnaire_id != q]
    typical.append(round(other.Usage_sum_duration.median() / 60, 1))
df["self_reported_mood"] = mood
df["typical_usage_minutes"] = typical

input_cols = ["case_id", "time_of_day", "entry_trigger", "app_type", "user_stated_goal", "user_state", "recent_context",
              "usage_minutes", "usage_sessions", "median_session_sec", "usage_quadratic",
              "swipe_interval_cv", "avg_dwell_sec", "self_reported_mood", "typical_usage_minutes"]
label_cols = ["case_id", "user_intent", "intervention_needed", "expected_action"]
meta_cols = ["case_id", "case_type", "label_rationale", "src_user_id", "src_questionnaire_id", "pair_id"]
enc = "utf-8-sig"
df[input_cols].to_csv("benchmark_inputs.csv", index=False, encoding=enc)
df[label_cols].to_csv("benchmark_labels.csv", index=False, encoding=enc)
df[meta_cols].to_csv("benchmark_meta.csv", index=False, encoding=enc)
ann = df[input_cols].copy(); ann["annotator_action"] = ""   # 친구 독립 라벨링용
ann.to_csv("annotation_sheet.csv", index=False, encoding=enc)

print("생성 완료:", len(df), "사례")
print(df.expected_action.value_counts().to_string())
print(df.case_type.value_counts().to_string())
