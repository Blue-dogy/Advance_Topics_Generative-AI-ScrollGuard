"""
build_benchmark.py  (v2.2)  -  ScrollGuard 벤치마크 81개 생성 (기존 45개 + 제스처 반례 쌍 12개 + 지름길 반례 24개)
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
 ("long","21:10","search","long_video",1.10,260,"운영체제 중간고사 대비 강의 시청","재생목록 순서대로 시청, 노트 앱으로 전환 3회, 검색어 'OS 페이징 강의'",IN,NO,"adversarial","길지만 선언 목표·행동 일치"),
 ("long","14:20","self_open","reader",2.00,150,"","전자책 앱에서 페이지 넘김 간격이 1~3분으로 일정, 알림 수신 없음",IN,NO,"standard","독서 앱, 자율적 사용"),
 ("mid","12:30","search","long_video",0.90,180,"된장찌개 레시피 확인","검색어 입력 2회 후 레시피 영상 3개 시청, 마지막 영상 후 앱 종료",IN,NO,"standard","검색→목적 달성→종료"),
 ("trivial","08:10","self_open","map",nan,nan,"","지도 앱 길찾기 실행 후 종료",IN,NO,"standard","짧은 유틸리티 사용"),
 ("trivial","13:00","notification","messenger",nan,nan,"","메신저 알림 확인 후 답장 1회, 앱 종료",IN,NO,"standard","짧은 유틸리티 사용"),
 ("mid_budget","22:00","self_open","long_video",1.00,150,"","사용자가 사전 설정한 휴식 타이머 20분, 현재 {m}분 경과, 구독 채널 영상 시청",IN,NO,"boundary","자기 설정 예산 내"),
 ("long","16:00","self_open","sns_feed",0.90,40,"여행 사진 앨범 정리 후 친구들과 공유","사진 업로드·캡션 작성·댓글 답글 작성 반복, 피드 스크롤은 짧게 간헐적",IN,NO,"adversarial","SNS+긴 시간이지만 생산 활동"),
 ("long_conc","10:00","search","long_video",1.30,600,"학회 발표 영상 시청","단일 영상 연속 시청 중 일시정지 4회, 문서 앱으로 전환해 메모",IN,NO,"adversarial","사용 집중도 높아도 목적적"),
 ("frag","09:30","self_open","messenger",nan,nan,"조별과제 마감 전 단톡 확인","메신저와 공유 문서 앱을 번갈아 실행, 문서 편집 기록 있음",IN,NO,"adversarial","세션 많지만 과업 수행"),
 ("mid","19:00","notification","shopping",0.80,50,"어제 담아둔 노트북 가격 비교","장바구니 상품 3개 사양 비교 후 1개 결제 완료, 앱 종료",IN,NO,"standard","명확한 과업 완료"),
 ("long","23:30","search","long_video",1.00,220,"내일 제출할 과제 참고 영상 시청 후 취침","검색→시청→노트 앱 기록 반복, 시청 영상 제목이 모두 과제 주제와 일치, 캘린더: 내일 09:00 과제 마감",IN,NO,"adversarial","심야·장시간이지만 목적적"),
 ("mid","17:30","self_open","podcast",nan,nan,"","러닝 활동 기록 중 팟캐스트 재생, 화면 대부분 꺼짐",IN,NO,"standard","화면 소비 없음"),
 ("mid","20:10","search","sns_feed",0.60,35,"주말 모임 맛집 후기 검색 후 예약","지역명 검색 3회, 후기 열람 후 예약 앱으로 전환해 예약 완료",IN,NO,"standard","SNS를 검색 도구로 사용"),
 ("long","15:00","self_open","long_video",1.20,300,"오늘은 쉬는 날이라 영화 한 편을 보겠다고 직접 선언","영화 1편 이어서 시청 중, 사용자 설정 방해 금지 모드 켜짐",IN,NO,"boundary","사용자 자율 선언 존중"),
 ("frag","08:30","self_open","email",nan,nan,"출근길 메일 확인","이동 중(위치 상태: 이동), 메일 앱 실행 후 짧게 확인하고 종료 반복",IN,NO,"standard","이동 중 유틸리티"),
 ("mid","11:00","search","short_video",0.45,35,"쇼츠로 스페인어 표현 공부","영상마다 일시정지 후 저장 8회, 퀴즈 앱으로 전환 2회, 검색어 '스페인어 일상 표현'",IN,NO,"adversarial","숏폼이지만 학습 행동"),
 ("trivial","12:00","self_open","music",nan,nan,"","음악 앱에서 플레이리스트 재생 후 앱 종료",IN,NO,"standard","짧은 유틸리티 사용"),
 ("mid","18:40","self_open","community",0.70,60,"다음 학기 수강신청 후기 확인","검색어 '교양 추천' 입력, 글 5개 열람 후 시간표 앱으로 전환",IN,NO,"standard","명확한 정보 탐색"),
 # ---------------- gentle_prompt (17) ----------------
 ("long","23:50","notification","short_video",0.12,6,"","알림으로 진입 후 추천 피드만 연속 시청, 검색 입력 없음, 다른 앱 전환 없음",UN,GP,"standard","다중 자동조종 신호"),
 ("long","01:10","self_open","sns_feed",0.18,5,"","사용자 취침 목표 시각 24:00 경과, 동일 피드 연속 스크롤, 게시물 상호작용 없음",UN,GP,"standard","취침 목표 초과+수동 소비"),
 ("frag","02:00","self_open","sns_feed",0.20,8,"","화면 켤 때마다 같은 앱을 반복 실행, 매번 1분 미만 체류 후 종료, 갱신 동작만 반복",UN,GP,"adversarial","총 시간은 짧아도 강박적 재확인"),
 ("mid_long","15:00","self_open","long_video",0.50,90,"점심 식단 영상 하나만 보기","첫 영상 종료 후 자동 추천 영상 5개를 이어 시청, 영상 주제가 게임·연예·스포츠로 변화",UN,GP,"standard","선언 목표에서 이탈"),
 ("long","21:30","search","short_video",0.15,7,"과제 자료 검색","검색 2회 후 검색·앱 전환 없이 쇼츠 피드 연속 시청",UN,GP,"standard","검색 후 피드로 이탈"),
 ("long","13:20","notification","sns_feed",0.20,6,"","캘린더: 13:00–14:15 수업, 수업 시간대에 피드 연속 스크롤, 위치: 강의동",UN,GP,"boundary","일정과 충돌하는 수동 소비"),
 ("many_long","03:40","self_open","sns_feed",0.20,5,"","앱 종료 후 즉시 재실행 반복, 같은 피드 상단 갱신 반복",UN,GP,"adversarial","세션 다수+심야+수동"),
 ("long","22:00","self_open","shopping",0.25,8,"","추천 탭만 연속 탐색, 검색·찜·장바구니·구매 행동 없음",UN,GP,"standard","목적 행동 부재"),
 ("mid","18:00","notification","news",0.30,10,"","뉴스 댓글 목록 연속 열람과 새로고침 반복, 기사 본문 열람 없음",UN,GP,"standard","수동적 반복 소비"),
 ("long","20:40","self_open","sns_feed",0.20,6,"","15분 전 사용자가 '이제 그만 봐야겠다'고 입력, 이후에도 동일 피드 연속 스크롤",UN,GP,"standard","사용자 본인 의사와 행동 불일치"),
 ("mid","09:00","self_open","sns_feed",0.25,7,"","기상 알람 해제 직후 진입해 피드 스크롤 지속, 오늘 첫 일정 09:30(캘린더)",UN,GP,"standard","일정 직전 수동 소비"),
 ("many_long","14:00","self_open","sns_feed",0.30,5,"","작업 앱 사용 중 잠금 해제 반복, 매번 SNS 홈 확인 후 즉시 종료, 알림 수신 없음",UN,GP,"adversarial","알림 없는 반복 재확인"),
 ("long","20:00","self_open","short_video",0.20,8,"공부 중 쉬는 시간 10분","휴식 타이머 10분 설정, 타이머 종료 후에도 쇼츠 피드 연속 시청, 현재 사용 {m}분",UN,GP,"standard","자기 설정 예산 초과"),
 ("mid","22:30","notification","sns_feed",0.30,12,"","친구 DM 알림으로 진입, DM 확인 후 탐색 탭으로 이동해 피드 연속 스크롤",UN,GP,"standard","과업 종료 후 피드 이탈"),
 ("long","16:30","self_open","short_video",0.12,5,"5분만 쇼츠 보기","사용자가 시작 시 '5분만'이라고 입력, 현재 사용 {m}분 경과, 다른 앱 전환 없음",UN,GP,"standard","자기 설정 시간 초과"),
 ("xlong","02:20","notification","short_video",0.10,4,"","알림으로 진입, 일시정지·좋아요·저장 없음, 다른 앱 전환 없음",UN,GP,"standard","다중 자동조종 신호"),
 ("many_long","12:40","self_open","community",0.25,9,"","게시글 목록 새로고침과 상단 이동 반복, 글 본문 열람은 짧고 댓글·검색 없음",UN,GP,"standard","수동적 반복 확인"),
 # ---------------- ask_context (10) ----------------
 ("long","20:00","self_open","long_video",0.80,90,"","검색 입력 1회 후 추천 영상으로 이동, 시청 영상 주제가 요리·게임·뉴스 3종",UC,AC,"boundary","목적·이탈 신호 혼재"),
 ("long","15:30","self_open","sns_feed",0.50,20,"","게시물 작성 1회 후 피드 스크롤 지속, 계정에 비즈니스 계정 태그가 있음",UC,AC,"boundary","업무/여가 구분 불가"),
 ("frag","11:00","self_open","browser",nan,nan,"","브라우저 반복 실행, 검색어 기록은 접근 제한 상태",UC,AC,"boundary","핵심 정보 접근 불가"),
 ("mid_long","00:10","search","long_video",0.40,60,"내일 발표 자료 참고","취침 목표 24:00 경과, 시청 영상 5개 중 1개만 발표 주제와 일치",UC,AC,"boundary","목표 일부 일치+예산 초과"),
 ("long","17:00","self_open","short_video",0.30,12,"학습용 쇼츠 보기","시청 영상 중 일부만 학습 주제, 저장·일시정지 기록 없음",UC,AC,"boundary","선언 목표와 행동 부분 일치"),
 ("mid","12:40","notification","sns_feed",0.35,15,"","점심 시간대(캘린더 일정 없음), 휴식 타이머 설정 없음",UC,AC,"standard","정상 휴식 vs 습관 구분 불가"),
 ("long","22:30","self_open","long_video",1.00,200,"","긴 영상 연속 시청, 이 사용자의 야간 시청 이력은 이번이 첫 관측",UC,AC,"standard","기준 이력 없음"),
 ("mid","10:00","self_open","shopping",0.50,45,"","상품 상세 페이지 여러 개 열람, 찜 2회, 구매 없음",UC,AC,"standard","비교 탐색 vs 충동 구분 불가"),
 ("long","19:00","self_open","community",0.60,30,"","글 읽기와 댓글 작성 혼재, 작성 댓글 3개",UC,AC,"standard","소통 vs 소비 구분 불가"),
 ("mid","09:00","self_open","news",0.50,40,"","뉴스 읽기 후 연관 기사 이동 반복, 캘린더에 업무 일정 시작 전",UC,AC,"standard","정보 수집 vs 습관 구분 불가"),
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
        recent_context=ctx.format(m=int(round(minutes))),
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
 ("P1","long","14:00","self_open","comic","주말에 완결 웹툰 정주행","구독 중인 완결 웹툰을 1화부터 순서대로 열람, 현재 23화, 다른 작품으로 이동 없음",IN,NO,"선언 목표와 행동 일치(제스처와 무관)",(0.15,6),(0.95,40)),
 ("P2","long","21:00","self_open","gallery","여행 사진 1000장 중 흔들린 사진 삭제","사진을 한 장씩 넘기며 삭제 312회, 즐겨찾기 41회, 앨범 2개 생성",IN,NO,"정리 과업 수행 중(제스처와 무관)",(0.10,2),(0.90,25)),
 ("P3","mid","08:00","search","news","긴 탐사보도 기사 끝까지 읽기","기사 1개 본문을 끝까지 열람, 하이라이트 3회, 마지막에 메모 앱으로 링크 공유",IN,NO,"목적적 읽기(제스처와 무관)",(0.20,9),(1.00,45)),
 ("P4","long","00:40","self_open","sns_feed","","사용자 취침 목표 23:00 경과, 피드 연속 탐색, 게시물 상호작용·검색·저장 없음, 다른 앱 전환 없음",UN,GP,"취침 목표 초과+목적 행동 부재(제스처와 무관)",(0.15,5),(1.10,25)),
 ("P5","mid_long","15:00","notification","short_video","","알림으로 진입한 뒤 추천 영상만 시청, 검색·저장·좋아요 없음, 캘린더: 30분 전 시작한 과제 작업 시간",UN,GP,"알림 후 이탈+일정 충돌(제스처와 무관)",(0.12,5),(0.95,30)),
 ("P6","mid","23:40","self_open","shopping","","추천 상품 목록 탐색, 상세 페이지 열람 후 바로 뒤로 가기 반복, 검색·찜·장바구니 없음, 15분 전 사용자가 '이제 자야지'라고 입력",UN,GP,"본인 의사와 행동 불일치+목적 행동 부재(제스처와 무관)",(0.20,6),(1.00,35)),
]
rng_cf = np.random.default_rng(SEED + 1)
cf_rows = []
for pid, tier, t, trig, app, goal, ctx, intent, action, why, ga, gb in CF_PAIRS:
    pool = data.query(TIERS[tier]); pool = pool[~pool.index.isin(used)]
    idx = rng_cf.choice(pool.index.values); used.add(idx); r = data.loc[idx]
    minutes = round(r.Usage_sum_duration / 60, 1)
    for variant, (cv, dwell) in (("a", ga), ("b", gb)):
        cf_rows.append(dict(
            time_of_day=t, entry_trigger=trig, app_type=app, user_stated_goal=goal,
            recent_context=ctx.format(m=int(round(minutes))),
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
 ("long","19:30","self_open","sns_feed",0.25,8,"","친구 생일 파티 사진이 올라온 단체 계정 피드 열람, 게시물마다 댓글 작성 12회, 사진 저장 9회, 마지막에 메신저로 사진 전달 후 앱 종료",IN,NO,"shortcut","목표 선언 없지만 생산적 상호작용"),
 ("mid","18:00","search","short_video",0.30,10,"","검색어 '자전거 체인 교체' 입력 후 관련 쇼츠 4개만 시청, 시청 중 쇼핑 앱으로 전환해 부품 구매, 이후 앱 종료",IN,NO,"shortcut","검색→목적 달성→종료"),
 ("mid_long","10:30","notification","sns_feed",0.40,12,"","동아리 계정 알림으로 진입, 행사 안내 게시물 작성·수정 3회, 신청자 댓글에 답글 15개 작성, 일정 앱에 행사 일정 등록",IN,NO,"shortcut","SNS에서 과업 수행"),
 ("mid","07:40","self_open","education",0.15,3,"","단어장 앱에서 카드를 넘기며 정답 확인, 오늘 목표 카드 200장 중 180장 완료, 틀린 카드만 다시 학습",IN,NO,"shortcut","균일한 넘김이지만 학습 과업"),
 ("long","23:30","self_open","sns_feed",0.20,6,"","해외에 사는 가족과 영상통화 후 가족 단체 계정 게시물 열람, 게시물마다 좋아요와 댓글 작성, 사진 3장 업로드",IN,NO,"shortcut","심야·SNS지만 생산적 상호작용"),
 ("long","13:00","self_open","short_video",0.35,15,"","직접 찍은 영상을 편집하려고 다른 쇼츠의 편집 방식 참고, 시청 중 영상 편집 앱으로 전환 6회, 편집 프로젝트 저장 기록 있음",IN,NO,"shortcut","숏폼을 참고 자료로 사용"),
 ("frag","12:10","notification","shopping",0.30,9,"","택배 도착 알림으로 진입, 배송 조회 후 반품 신청 3건 처리, 각 건마다 사진 첨부, 고객센터 채팅 1회",IN,NO,"shortcut","세션 많지만 과업 수행"),
 ("mid","16:00","search","sns_feed",0.30,14,"인턴 지원 회사 계정의 최근 게시물 확인","회사 계정 게시물을 최신순으로 열람, 채용 관련 게시물 4개 저장, 문서 앱으로 전환해 지원서에 내용 메모",IN,NO,"shortcut","SNS를 정보 수집 도구로 사용"),
 # ---- gentle_prompt: 제스처는 '불규칙·긴 체류', 목표 선언 있음·SNS 아님·값 없음 등 ----
 ("xlong","02:10","self_open","long_video",0.90,240,"","자동 재생으로 영상이 이어짐, 화면 조작 거의 없음, 취침 목표 24:00 경과",UN,GP,"shortcut","취침 목표 초과+수동 시청"),
 ("mid_long","22:40","notification","community",0.80,50,"공지 하나 확인","공지 확인 후 인기글 목록으로 이동해 글 계속 열람, 댓글·검색 없음",UN,GP,"shortcut","선언 목표에서 이탈+목적 행동 부재"),
 ("mid","08:50","self_open","news",0.70,45,"오늘 날씨 확인","날씨 확인 후 연예 기사로 이동해 연속 열람, 캘린더: 09:00 수업",UN,GP,"shortcut","선언 목표에서 이탈+일정 충돌"),
 ("many_long","14:30","self_open","messenger",nan,nan,"","새 메시지 없이 메신저 실행 후 바로 종료 반복 24회, 캘린더: 공부 시간",UN,GP,"shortcut","반복 재확인+일정 충돌"),
 ("frag","01:30","self_open","browser",nan,nan,"","같은 커뮤니티 홈페이지 새로고침 반복, 글 열람 없음, 취침 목표 24:00 경과",UN,GP,"shortcut","반복 새로고침+취침 목표 초과"),
 ("long","21:00","search","long_video",0.85,120,"강의 1개 듣기","강의 종료 후 추천 영상 4개 연속 시청, 주제가 예능·게임으로 변화, 노트 앱 전환 없음",UN,GP,"shortcut","선언 목표에서 이탈"),
 ("long","23:10","search","shopping",0.75,40,"생필품 2개 주문","2개 결제 후 추천 상품 목록으로 이동해 계속 열람, 장바구니 추가 없음, 취침 목표 23:00 경과",UN,GP,"shortcut","과업 종료 후 이탈+취침 목표 초과"),
 ("mid_long","11:00","notification","news",0.65,38,"","속보 알림으로 진입 후 관련 없는 기사로 이동 반복, 기사마다 댓글 목록만 열람, 캘린더: 10:30–12:00 팀 회의",UN,GP,"shortcut","알림 후 이탈+일정 충돌"),
 # ---- ask_context 보강: 목적 신호와 이탈 신호가 섞이거나 핵심 정보 접근 불가 ----
 ("long","21:00","self_open","sns_feed",0.25,6,"","피드 연속 탐색 중 게시물 2개 저장, 직후 다시 연속 탐색, 취침 목표 설정 없음",UC,AC,"boundary","저장 행동과 연속 탐색 혼재"),
 ("long","18:30","search","short_video",0.20,7,"요리 쇼츠 참고","요리 쇼츠 3개 저장 후 피드 연속 시청, 시청 주제가 요리와 여행으로 섞임",UC,AC,"boundary","목표 일부 일치+이탈 신호"),
 ("xlong","13:30","self_open","long_video",1.00,300,"","다큐멘터리 시리즈 3편 연속 시청, 시리즈 외 영상 없음, 캘린더: 14:00 과제 마감",UC,AC,"boundary","일관된 시청 vs 일정 임박"),
 ("frag","16:00","notification","messenger",nan,nan,"","메신저 실행 반복, 대화방 기록은 사용자 설정으로 접근 차단",UC,AC,"boundary","핵심 정보 접근 불가"),
 ("long","23:50","notification","news",0.30,12,"선거 결과 확인","개표 속보와 해설 기사 연속 열람, 같은 기사 새로고침 6회, 취침 목표 24:00 임박",UC,AC,"boundary","목표 일치 vs 반복 새로고침·취침 임박"),
 ("mid_long","15:00","self_open","community",0.70,40,"","중고거래 게시판에서 같은 품목 글 반복 열람, 판매자에게 채팅 1회, 거래 완료 기록 없음",UC,AC,"boundary","비교 탐색 vs 반복 열람 구분 어려움"),
 ("long","23:00","self_open","gallery",0.40,20,"","오래된 사진을 넘기며 열람, 삭제·편집·공유 없음, 일부 사진에서 오래 머묾",UC,AC,"boundary","회상 vs 수동 소비 구분 어려움"),
 ("long","22:00","search","shopping",0.25,8,"겨울 패딩 고르기","패딩 검색 후 상세 페이지 다수 열람, 찜 1회, 이후 신발·가방 상품으로 이동",UC,AC,"boundary","목표 일부 일치+이탈 신호"),
]
rng_sc = np.random.default_rng(SEED + 2)
sc_rows = []
for tier, t, trig, app, cv, dwell, goal, ctx, intent, action, ctype, why in SC_CASES:
    pool = data.query(TIERS[tier]); pool = pool[~pool.index.isin(used)]
    idx = rng_sc.choice(pool.index.values); used.add(idx); r = data.loc[idx]
    minutes = round(r.Usage_sum_duration / 60, 1)
    sc_rows.append(dict(
        time_of_day=t, entry_trigger=trig, app_type=app, user_stated_goal=goal,
        recent_context=ctx.format(m=int(round(minutes))),
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

input_cols = ["case_id", "time_of_day", "entry_trigger", "app_type", "user_stated_goal", "recent_context",
              "usage_minutes", "usage_sessions", "median_session_sec", "usage_quadratic",
              "swipe_interval_cv", "avg_dwell_sec"]
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
