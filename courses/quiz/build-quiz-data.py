# -*- coding: utf-8 -*-
"""퀴즈 HTML에서 문항을 추출해 수업내용 페이지가 읽을 데이터 파일을 만든다."""
import io, json, re, collections, sys

def extract(path):
    s = io.open(path, encoding='utf-8').read()
    m = re.search(r'const QUESTIONS = (\[.*?\]);', s, re.S)
    return json.loads(m.group(1))

# 주차별 핵심 개념 후보 — 실제 문항에서 다뤄진 것만 살아남는다.
TERMS = {
 'ml-w01': [
   '판별 모델','생성 모델','생성형 AI','확률 분포','샘플링','GAN','확산','Diffusion','오토인코더',
   'text-to-image','image-to-image','AI 휴먼','블렌디드','플립러닝','PBL','출석','중간시험','기말시험','프로젝트',
   '지도학습','비지도학습','강화학습','딥러닝','머신러닝','인공지능','신경망','CNN','객체 검출','시멘틱 분할',
   '훈련','검증','테스트','과적합','일반화',
 ],
 'ml-w02': [
   '벡터','스칼라','n벡터','전치','스칼라 곱','벡터 덧셈','방향','크기',
   '훈련 데이터','특징벡터','레이블','관측','이진분류',
   '노름','내적','사잇각','코사인','초평면','결정 경계',
   '행렬','행렬 곱','교환 법칙','역행렬','단위 행렬','텐서','선형 결합','선형 종속','선형 독립','선형 변환',
   '도함수','미분','편미분','기울기','Gradient','임계점','극소점','경사','연쇄 법칙',
   '확률 변수','확률 분포','확률 질량 함수','확률 밀도 함수','기대값','분산','결합 확률','주변 확률','조건부 확률',
   '베이즈','가우시안','정규 분포','이산','연속',
   'Numpy','배열','ndim','shape','reshape','arange','인덱싱','슬라이싱','축','axis','브로드캐스팅','concatenate','stack','transpose',
 ],
 'nlp-w01': [
   'NLU','NLG','인공어','텍스트 분석','학제간','규칙 기반','통계 기반','신경망','사전학습',
   'ELIZA','튜링','Transformer','n-gram','word2vec','BERT','GPT','어텐션',
   '중의성','구조적 중의성','규칙의 예외','유연성과 확장성','교착어','형태소',
   '기계 번역','질의응답','기계 독해','텍스트 분류','감성 분석','문서 요약','추출 요약','생성 요약',
   '코퍼스','단어 임베딩','파이프라인','파싱','편향','환각','few-shot','벡터',
 ],
 'nlp-w02': [
   '표본공간','확률변수','확률질량함수','확률밀도함수','이산','연속',
   '결합확률','주변확률','조건부확률','연쇄법칙','마르코프 가정','bigram','trigram',
   '베이즈 정리','사전확률','사후확률','나이브 베이즈','기댓값','분산',
   '이항분포','다항분포','정규분포','Zipf','희소성',
   'MLE','MAP','우도','로그우도','스무딩','라플라스','Kneser-Ney','Good-Turing','가중치 감쇠',
   '정보량','엔트로피','조건부 엔트로피','결합 엔트로피','상호정보량','PMI',
   '교차 엔트로피','KL','Perplexity','Shannon',
   '원-핫','분산 표현','임베딩','노름','내적','코사인 유사도','TF-IDF','SVD','잠재 의미 분석','소프트맥스','선형변환',
   '음절','형태소','어절','실질 형태소','형식 형태소','자립 형태소','의존 형태소','이형태','교착어',
   '품사','체언','수식언','관계언','독립언','용언','관형사','부사','조사','어간','어미','선어말 어미',
   '품사 태깅','세종 태그셋','미등록어','서브워드',
   '구구조','문맥자유문법','생성 문법','구조적 중의성','의존구조','지배소','의존소','그래프 기반','전이 기반',
   '의미론','의미역','WSD','분포 가설','화용론','어휘 중의성',
 ],
}

def build(key, quiz_path, out_path, meta):
    qs = extract(quiz_path)
    # 개념어 → 해당 개념을 다루는 문항 id 목록
    term_map = {}
    for t in TERMS[key]:
        ids = [q['id'] for q in qs
               if t.lower() in (q['q'] + ' ' + ' '.join(q['opts']) + ' ' + q['exp']).lower()]
        if len(ids) >= 2:                      # 2문항 이상 다뤄진 개념만 하이라이트
            term_map[t] = ids
    # 짧은 개념어가 긴 개념어에 포함되면 긴 쪽을 우선(하이라이트 중복 방지)
    terms_sorted = sorted(term_map, key=len, reverse=True)
    sec_count = collections.Counter(q['sec'] for q in qs)

    payload = {
        'meta': meta,
        'secCount': dict(sorted(sec_count.items())),
        'terms': [{'t': t, 'ids': term_map[t]} for t in terms_sorted],
        'items': qs,
    }
    js = ('/* 자동 생성 파일 — %s 에서 추출. 직접 수정하지 마세요. */\n'
          'window.QUIZ_DATA = %s;\n') % (quiz_path.split('/')[-1],
                                         json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    io.open(out_path, 'w', encoding='utf-8').write(js)
    print('%-14s 문항 %d · 개념 %d · 절 %s' % (key, len(qs), len(term_map), dict(sec_count)))
    print('   ', ', '.join('%s(%d)' % (t, len(term_map[t])) for t in sorted(term_map, key=lambda x: -len(term_map[x]))[:14]))

build('nlp-w01', 'materials/자연어처리/week01-quiz100.html', 'courses/quiz/nlp-w01.js',
      {'title': '1주차 · 자연어처리 개요', 'quiz': '../materials/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC/week01-quiz100.html'})
build('nlp-w02', 'materials/자연어처리/week02-quiz100.html', 'courses/quiz/nlp-w02.js',
      {'title': '2주차 · 수학과 언어학의 기본 원리', 'quiz': '../materials/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC/week02-quiz100.html'})
build('ml-w01', 'materials/기계학습특론/week01-quiz100.html', 'courses/quiz/ml-w01.js',
      {'title': '1주차 · 강의소개와 생성형 AI(이미지)', 'quiz': '../materials/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5%ED%8A%B9%EB%A1%A0/week01-quiz100.html'})
build('ml-w02', 'materials/기계학습특론/week02-quiz100.html', 'courses/quiz/ml-w02.js',
      {'title': '2주차 · 인공지능 기초 수학', 'quiz': '../materials/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5%ED%8A%B9%EB%A1%A0/week02-quiz100.html'})
