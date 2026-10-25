/* 개념 카드 제목에 적힌 교안 쪽수 "(54~61p)" "(4 · 5p)"를 읽어,
   같은 절의 '교안 슬라이드 모아보기'에서 그 쪽 슬라이드를 찾아 작은 썸네일로 붙인다.
   썸네일을 누르면 페이지의 슬라이드 확대 보기(window.__lbOpenFig)로 연다. */
(function () {
  var css =
    '.sref{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:-2px 0 9px}' +
    '.sref-l{font-size:10.5px;font-weight:700;color:var(--muted,#6b7280);letter-spacing:.02em;margin-right:2px}' +
    '.sref-t{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--border,#e2ddd6);background:#fff;' +
      'border-radius:7px;padding:2px 7px 2px 2px;cursor:zoom-in;font:inherit;line-height:1;transition:border-color .15s,box-shadow .15s}' +
    '.sref-t:hover,.sref-t:focus-visible{border-color:var(--accent,#1b3f8b);box-shadow:0 2px 8px rgba(18,34,80,.12);outline:none}' +
    '.sref-t img{display:block;width:44px;height:33px;object-fit:cover;border-radius:4px;background:#f1f1f1}' +
    '.sref-t b{font-size:11px;font-weight:700;color:var(--accent,#1b3f8b)}' +
    '.sref-more{font-size:11px;color:var(--muted,#6b7280)}';
  var st = document.createElement('style');
  st.textContent = css;
  document.head.appendChild(st);

  /* "4 · 5p", "54~61p", "19 ~ 24p" → 쪽 번호 목록 */
  function pagesIn(text) {
    var out = [], re = /(\d+(?:\s*(?:[·,~–]|-)\s*\d+)*)\s*p(?![a-z])/gi, m;
    while ((m = re.exec(text))) {
      m[1].split(/\s*[·,]\s*/).forEach(function (part) {
        var r = part.split(/\s*(?:~|–|-)\s*/);
        var a = parseInt(r[0], 10), b = parseInt(r[r.length - 1], 10);
        if (isNaN(a) || isNaN(b) || b < a || b - a > 40) return;
        for (var p = a; p <= b; p++) out.push(p);
      });
    }
    return out;
  }
  function pageOf(fig) {
    var b = fig.querySelector('figcaption b');
    if (!b) return null;
    var all = b.textContent.match(/\d+(?=\s*p)/g);
    return all ? parseInt(all[all.length - 1], 10) : null;
  }

  var MAX = 8;
  document.querySelectorAll('.sec').forEach(function (sec) {
    var figs = Array.prototype.slice.call(sec.querySelectorAll('.deck .sld'));
    if (!figs.length) return;
    var byPage = {}, byKey = {};
    figs.forEach(function (f) {
      var p = pageOf(f), cb = f.querySelector('figcaption b');
      if (p !== null && !byPage[p]) byPage[p] = f;
      if (cb) {
        var k = cb.textContent.replace(/\s+/g, ' ').trim();
        if (!byKey[k]) byKey[k] = f;
      }
    });
    sec.querySelectorAll('.pt-h').forEach(function (h) {
      var seen = {}, hits = [];
      var keys = h.getAttribute('data-sref');
      if (keys) {
        /* 제목에 쪽수가 없는 페이지 — data-sref="1부 3p, 2부 4p"처럼 슬라이드 캡션의 굵은 글씨로 직접 지정 */
        keys.split(',').forEach(function (k) {
          k = k.replace(/\s+/g, ' ').trim();
          if (byKey[k] && !seen[k]) { seen[k] = 1; hits.push({ p: k, f: byKey[k] }); }
        });
      } else {
        pagesIn(h.textContent).forEach(function (p) {
          if (byPage[p] && !seen[p]) { seen[p] = 1; hits.push({ p: p + 'p', f: byPage[p] }); }
        });
      }
      if (!hits.length) return;
      var row = document.createElement('div');
      row.className = 'sref';
      var lab = document.createElement('span');
      lab.className = 'sref-l';
      lab.textContent = '📑 교안';
      row.appendChild(lab);
      hits.slice(0, MAX).forEach(function (hit) {
        var src = hit.f.querySelector('img');
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'sref-t';
        btn.setAttribute('aria-label', '교안 ' + hit.p + ' 슬라이드 크게 보기');
        var im = document.createElement('img');
        im.loading = 'lazy';
        im.alt = '';
        im.src = src.getAttribute('src');
        var b = document.createElement('b');
        b.textContent = hit.p;
        btn.appendChild(im);
        btn.appendChild(b);
        btn.addEventListener('click', function (e) {
          e.preventDefault();
          if (window.__lbOpenFig) window.__lbOpenFig(hit.f);
          else hit.f.click();
        });
        row.appendChild(btn);
      });
      if (hits.length > MAX) {
        var more = document.createElement('span');
        more.className = 'sref-more';
        more.textContent = '+' + (hits.length - MAX) + '장 (팝업에서 → 로 넘기기)';
        row.appendChild(more);
      }
      h.parentNode.insertBefore(row, h.nextSibling);
    });
  });
})();
