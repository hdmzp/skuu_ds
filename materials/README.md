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
