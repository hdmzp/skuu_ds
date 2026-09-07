# materials — 강의자료 보관 규칙

```
materials/
  <과목명>/                     ← 폴더명은 한글 과목명 그대로
    <자료 파일>.pdf             ← 파일명은 URL 안전한 영문/숫자
```

## 규칙

1. **폴더명 = 과목명(한글)**
   예) `materials/기계학습특론/`, `materials/컴퓨터비전/`

2. **파일명은 영문**으로 둔다
   경로에 한글이 섞이면 브라우저·서버·PDF.js가 주소를 인코딩하는 방식이
   달라 링크가 깨질 수 있어, 저장소 파일명은 `week01-course-intro.pdf`처럼
   ASCII로 유지한다.

3. **화면에 보이는 이름과 내려받는 파일명은 한글로 지정한다**
   `courses/pdf-viewer.html`의 `DOCS` 목록에서 자료마다
   `title`(화면 표시)과 `download`(저장될 파일명)를 한글로 적어 준다.
   덕분에 저장소는 영문 경로를 쓰면서도, 사용자가 받는 파일은
   `기계학습특론_1주차_강의소개.pdf`처럼 한글 이름으로 저장된다.

## 자료를 추가할 때

1. `materials/<과목명>/` 아래에 PDF를 넣는다. (폴더가 없으면 새로 만든다)
2. `courses/pdf-viewer.html`의 `DOCS`에 항목을 추가한다.

```js
'ml-w02-slides': {
  file: '../materials/기계학습특론/week02-slides.pdf',
  title: '기계학습특론 2주차 — 인공지능 기초 수학',   // 화면 제목
  download: '기계학습특론_2주차_인공지능기초수학.pdf', // 저장될 파일명
  sub: '2주차 · 24페이지',
  back: 'machine-learning-week02.html'
}
```

3. 해당 주차 페이지의 "강의자료" 섹션에 카드를 추가한다.
   제목은 `pdf-viewer.html?doc=<키>` 로, 다운로드 버튼은 PDF 경로로 연결한다.

## 100문제(HTML) 자료

주차별 100문제는 단일 HTML 파일이므로 PDF와 같은 폴더에 둔다.

```
materials/자연어처리/week01-quiz100.html
materials/기계학습특론/week01-quiz100.html
```

주차 페이지에서는 `target="_blank"`로 바로 열고, 다운로드 버튼에는
`download="자연어처리_1주차_자연어처리개요_100문제.html"`처럼 한글 파일명을 지정한다.

## 100문제를 수업내용 페이지에 연동하기

주차별 `weekNN-quiz100.html` 안의 `const QUESTIONS = [...]` 를 추출해
수업내용 페이지에서 **개념 하이라이트 → 문제 팝업**으로 연결한다.

```
courses/quiz/
  build-quiz-data.py   ← 추출 스크립트 (저장소 루트에서 실행)
  inline-quiz.js       ← 공용 동작: 하이라이트 · 절별 버튼 · 팝업
  nlp-w01.js           ← 자동 생성 (window.QUIZ_DATA)
  nlp-w02.js
```

### 새 주차를 연동할 때

1. `courses/quiz/build-quiz-data.py` 의 `TERMS` 에 해당 주차 핵심 개념어 목록을 추가하고,
   맨 아래 `build(...)` 호출을 한 줄 추가한다.
2. 저장소 루트에서 `python3 courses/quiz/build-quiz-data.py` 실행 → `courses/quiz/<키>.js` 생성.
3. 수업내용 페이지 `</body>` 직전에 두 줄을 넣는다.

```html
<script src="quiz/nlp-w02.js"></script>
<script src="quiz/inline-quiz.js"></script>
```

### 동작 규칙

- **절 매칭** — 페이지의 `.sec-badge` 텍스트(`2.1`, `2.6` …)와 문항의 `sec` 값이 같으면
  그 절 제목 옆에 `문제 N` 버튼이 붙는다. badge 문자열을 바꾸면 연동이 끊기니 주의.
- **개념 하이라이트** — 2문항 이상에서 다뤄진 개념 중 출제 수 상위 30개만,
  한 개념당 페이지에서 최대 2번까지 표시한다. 윗첨자 숫자 = 그 개념이 출제된 문항 수.
- 하이라이트는 `.note-box / .tip / .good / .warn / .card2 p / table td` 안에만 들어가며,
  링크·버튼 안에는 넣지 않는다.
