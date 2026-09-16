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
 'ml-w03': [
   '회귀','분류','세그멘테이션','예측','집값',
   '독립 변수','종속 변수','입력 변수','출력 변수','특징 벡터','데이터셋','지도학습','스칼라',
   '상관관계','인과관계','해적','학습 과정','훈련 데이터',
   '단순 선형 회귀','파라미터','매개변수','가중치','절편','기울기',
   '손실함수','손실 함수','비용 함수','Cost function','MSE','MAE','평균 제곱 오차','평균 절대 오차','argmin','등고선','Minimum','최솟값',
   '경사하강법','Gradient','학습률','Learning rate','초기화','초기값','편미분','접선','발산','수렴','종료조건',
   '다중 선형 회귀','매핑 함수','x_i0','내적','바이어스','w_j',
 ],
 'nlp-w04': [
   '분석 계층','오류 전파','Garbage In','규칙 기반',
   '정형 데이터','비정형 데이터','구조적 노이즈','인코딩','mojibake','PDF','HTML','BeautifulSoup','파이프라인',
   '정제','정규표현식','치환','<URL>','정규화','NFC','NFD','NFKC','case folding','대소문자','반복 문자',
   '토큰화','축약형','교착어','어절','불용어','Stemming','Lemmatization','Porter','어간 추출','표제어 추출',
   '띄어쓰기','n-gram','음절','sequence labeling','Bi-LSTM','CRF','Kiwi','PyKoSpacing',
   '삽입','생략','대체','전치','편집 거리','Levenshtein','Damerau','동적 계획법','non-word','real-word',
   '잡음 채널','오타 모델','언어 모델','사전 확률','acress','서브워드','BPE','WordPiece','SentencePiece','OOV','<UNK>',
   '형태소','자립 형태소','의존 형태소','실질 형태소','형식 형태소','원형 복원','불규칙','후보 선택','감기는',
   'KoNLPy','Okt','Komoran','Hannanum','Kkma','Mecab','사용자 사전',
   '품사','체언','용언','수식언','독립언','관계언','조사','태그셋','세종','Penn Treebank','UPOS','NNG','JKS','JKO',
   '품사 태깅','중의성','Brill','제약 문법','HMM','방출 확률','전이 확률','비터비','Viterbi','BERT',
   '구문 분석','구구조','의존 문법','구성소','지배소','의존소','CFG','문맥 자유 문법','비단말','단말','시작 기호',
   '구문 중의성','부착 중의성','등위 접속','카탈란','하향식','상향식','차트 파싱','CYK','Earley',
   'PCFG','트리뱅크','inside-outside','Lexicalized','F1','전이 기반','그래프 기반','SHIFT','LEFT-ARC','빔 서치','동적 오라클',
   '단일 지배소','비순환','연결성','사영성','head-final','MST','Chu-Liu','Biaffine',
   '의미 분석','Colorless','어휘적 중의성','구조적 중의성','화용적 중의성','다의어','동음이의어','상호참조',
   'WSD','sense inventory','WordNet','synset','KorLex','Lesk','gloss','지도학습','Yarowsky','의미 유도',
   '정적 임베딩','문맥 임베딩','Word2Vec','GloVe','ELMo','GlossBERT','분포 가설','Firth',
   '의미역','행동주','피동주','경험자','수혜자','도구','출처','도달점','수의적','필수','theta-criterion','의미역 기준',
   'SRL','술어','논항','PropBank','FrameNet','ARG0','BIO',
   '논리식','AMR','분산 표현','Sentence-BERT','RAG','의미 검색',
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
 'nlp-w03': [
   '전처리','파이프라인','원시 텍스트','정제','수치 벡터','어휘 사전','피처 추출',
   '토큰화','토큰','타입','유니그램','바이그램','트라이그램','n-그램','희소성','스무딩',
   'word_tokenize','WordPunct','RegexpTokenizer','TweetTokenizer','sent_tokenize','punkt','약어',
   'OOV','[UNK]','서브워드','BPE','WordPiece','SentencePiece','병합','어휘 크기',
   '어절','형태소','교착어','KoNLPy','Okt','Kiwi','Mecab','Komoran','띄어쓰기','중의성',
   '품사 태깅','POS','Penn Treebank','태그셋','NNP','VBG','pos_tag','퍼셉트론','HMM','CRF','세종','조사','어미','선어말',
   '불용어','stopwords','감성 분석','not','도메인','기능어',
   '정규화','대소문자','case folding','유니코드','NFC','NFD','정규표현식','re.sub','HTML 태그','URL','이모지','replace',
   '철자 수정','autocorrect','편집거리','SymSpell','고유명사',
   '어간 추출','stemming','Porter','Lancaster','Snowball','접미사','battl','over-stemming',
   '표제어 추출','lemmatization','WordNet','표제어','lemmatize',
   '개체명','NER','BIO','B-LOC','PERSON','ORG','GPE','spaCy','ne_chunk','비식별화',
   'WSD','Lesk','gloss','문맥','중의성 해결',
   '문장 경계','SBD','종결어미',
   '순서','설계','통합 파이프라인','preprocess',
   '사전학습','BERT','GPT','문맥 임베딩','TF-IDF','BoW',
   '데이터 수집','데이터 전처리','모델 개발','모델 평가','모델 배포','Garbage In','정확도','정밀도','재현율','F1','drift','크롤링','Word2Vec','GloVe',
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
build('nlp-w03', 'materials/자연어처리/week03-quiz100.html', 'courses/quiz/nlp-w03.js',
      {'title': '3주차 · 토큰화, 품사 태깅, 불용어 제거, 텍스트 정규화', 'quiz': '../materials/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC/week03-quiz100.html'})
build('nlp-w04', 'materials/자연어처리/week04-quiz100.html', 'courses/quiz/nlp-w04.js',
      {'title': '4주차 · 텍스트의 전처리, 어휘, 구문 그리고 의미 분석', 'quiz': '../materials/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC/week04-quiz100.html'})
build('ml-w01', 'materials/기계학습특론/week01-quiz100.html', 'courses/quiz/ml-w01.js',
      {'title': '1주차 · 강의소개와 생성형 AI(이미지)', 'quiz': '../materials/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5%ED%8A%B9%EB%A1%A0/week01-quiz100.html'})
build('ml-w02', 'materials/기계학습특론/week02-quiz100.html', 'courses/quiz/ml-w02.js',
      {'title': '2주차 · 인공지능 기초 수학', 'quiz': '../materials/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5%ED%8A%B9%EB%A1%A0/week02-quiz100.html'})
build('ml-w03', 'materials/기계학습특론/week03-quiz100.html', 'courses/quiz/ml-w03.js',
      {'title': '3주차 · 선형 회귀', 'quiz': '../materials/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5%ED%8A%B9%EB%A1%A0/week03-quiz100.html'})
