# skuu_ds — 대학원 수업 정리 사이트

석사 과정 수업(기계학습특론 · 자연어처리 · 컴퓨터비전 등)의 강의자료와 필기를
과목·주차별 정적 HTML 페이지로 정리해 GitHub Pages로 서비스하는 저장소.

라이브: https://hdmzp.github.io/skuu_ds/

## 배포 — 작업이 끝나면 여기까지가 한 세트

배포 브랜치는 **`claude/masters-course-summary-site-4vp2sc`** 하나다.
`.github/workflows/deploy-pages.yml`이 이 브랜치 푸시에만 반응해 저장소 루트를
통째로 Pages에 올린다.

사용자가 따로 막지 않는 한, 페이지를 고쳤으면 **커밋 → 작업 브랜치 푸시 →
배포 브랜치 반영 → 배포 성공 확인 → 링크 안내**까지 한 번에 진행한다.
"배포해 줘"라는 말을 매번 기다리지 않는다.

```bash
git push -u origin <작업브랜치>
git push origin HEAD:refs/heads/claude/masters-course-summary-site-4vp2sc
```

- 배포 브랜치가 작업 브랜치의 조상이면 그대로 fast-forward 된다. 아니면
  **먼저 배포 브랜치를 작업 브랜치로 병합**해 충돌을 해소하고 푸시한다.
  배포 브랜치에 강제 푸시하지 않는다.
- 배포 확인은 GitHub MCP(`mcp__github__actions_list` / `actions_get`)로
  `deploy-pages.yml`의 최신 run이 `conclusion: success`인지 본다.
  샌드박스에서 `hdmzp.github.io`는 egress 정책상 직접 열리지 않으므로
  run 결과로 확인하고 페이지 주소를 안내한다.

### `main`으로 옮기려면 (사용자 설정 한 번 필요)

사용자는 배포를 `main`으로 일원화하기를 원한다. 다만 `github-pages` **환경의
배포 허용 브랜치 목록에 `main`이 없어서**, `main`에 푸시하면 잡이 실행되기도
전에 거부된다(로그 없이 2초 만에 실패 — run #74, #76이 그 사례).

사용자가 **Settings → Environments → github-pages → Deployment branches**에
`main`을 추가하고 나면, 워크플로의 `branches:`를 `[main]`으로 바꾸고
그때부터 `main`으로 배포한다. 그 전에는 위의 배포 브랜치를 쓴다.

## 구조

| 경로 | 내용 |
|---|---|
| `index.html` | 홈 — 과목 카드와 주차별 바로가기 |
| `courses/<과목>-week<NN>.html` | 주차별 정리 페이지 (본문 대부분이 여기) |
| `courses/<과목>.html` | 과목별 강의계획서 |
| `courses/quiz/<과목>-w<NN>.js` | 절별 인라인 퀴즈 데이터 + `inline-quiz.js` |
| `courses/pdf-viewer.html` | 앱 실행 없이 PDF를 화면에서 여는 뷰어 |
| `materials/<과목명(한글)>/` | 원본 PDF · 노트북 · 100문제 HTML · `figs/` 캡처 |
| `vendor/katex/` | 수식 렌더링 (외부 CDN 대신 로컬 번들) |

`materials/` 파일명은 `week03-slides.pdf`, `week04-quiz100.html`,
`week03-lab-solution.ipynb`처럼 `week<NN>-<용도>` 규칙을 따른다.

## 페이지 작성 규칙

각 주차 페이지는 **단일 HTML 파일**이다 — CSS는 `<style>`에, JS는 맨 아래
`<script>`에 인라인으로 둔다. 새 페이지는 같은 과목의 기존 주차 파일을
복사해 시작하는 것이 가장 빠르다.

주요 클래스 (과목 간 공통):

- 구조 — `.sec` / `.sec-head` / `.sec-badge` / `.sub-title`
- 박스 — `.note-box` `.tip`(노랑) `.warn`(빨강) `.good`(초록)
- 심화 — `.deep`(🔎 한 걸음 더, 보라) `.pit`(⚠ 여기서 갈린다, 주황)
- 필기 구분 — `.live`(🎙 실시간 강의 강조) `.mynote`(📝 내 복습 필기)
- 카드·흐름 — `.cards`/`.card2`, `.flow`/`.fl`, `.tbl-wrap`+`table`
- 수식 — `.tex`(인라인) `.tex-d`(별행). 안에는 **TeX 원문**을 그대로 쓰고
  KaTeX가 렌더한다. `<`는 `&lt;`, `&`는 `&amp;`로 이스케이프할 것
  (`textContent`를 읽으므로 엔티티는 자동으로 풀린다)
- 코드 — `pre.code`, 주석은 `.cm`, 출력은 `.out`
- **형광펜 + 설명 말풍선** — `<span class="term" tabindex="0" data-tip="설명">용어</span>`.
  페이지 하단 스크립트가 `#tipbox`를 만들어 hover·focus·click에 띄운다.
  새 페이지에 쓰려면 `.term`/`#tipbox` CSS와 해당 스크립트를 함께 옮긴다

탭바(`.tab-bar`)의 앵커와 섹션 `id`는 1:1로 맞춘다. 섹션을 추가하면
탭바에도 링크를 넣는다.

## 내용 원칙

- **기출 표시가 아니라 개념으로 쓴다.** 100문제에서 갈리는 지점은
  "몇 번 문제 출제"로 적지 않고, `.pit` 상자에 비교표와 설명으로 녹인다.
- 사용자는 파이썬·수식에 아직 익숙하지 않다고 밝혔다. 코드는 **줄 단위 풀이**와
  **출력값 해석**까지 붙이고, 수식은 유도 과정을 단계로 보여 준다.
- 교안에 없는 내용을 보탤 때는 `.mynote`/`.deep`처럼 **출처가 구분되는 상자**에
  넣어 강의 내용과 섞이지 않게 한다.
- 다른 주차·과목에 같은 개념이 있으면 `<a class="ext" href="...">`로 연결한다.

## 검증

파일을 크게 고쳤으면 커밋 전에 최소한 이것만은 확인한다.

```bash
# 태그 짝 맞음 · KaTeX 블록에 이스케이프 안 된 < & 없음
python3 - <<'EOF'
import io,re
from html.parser import HTMLParser
src=io.open('courses/<파일>.html',encoding='utf-8').read()
VOID={'br','img','hr','meta','link','input','source'}
class P(HTMLParser):
    def __init__(self): super().__init__(); self.stack=[]; self.errs=[]
    def handle_starttag(s,t,a):
        if t not in VOID: s.stack.append(t)
    def handle_endtag(s,t):
        if t in VOID: return
        if not s.stack or s.stack[-1]!=t: s.errs.append((t,s.getpos()))
        else: s.stack.pop()
p=P(); p.feed(src); print('errs',p.errs[:5],'unclosed',p.stack[:5])
for m in re.finditer(r'<div class="tex-d">(.*?)</div>',src,re.S):
    if re.search(r'&(?![a-zA-Z]+;|#\d+;)',m.group(1)) or '<' in m.group(1):
        print('TEX', src[:m.start()].count('\n')+1)
EOF
```

컨테이너에 `numpy` 등 과학 패키지는 없다. 노트북 출력값을 인용할 때는
직접 실행하지 말고 **`.ipynb`에 저장된 출력**을 근거로 삼는다.
