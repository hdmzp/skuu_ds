/*
 * inline-quiz.js — 수업내용 페이지에 문제 풀이를 얹는다.
 *
 *  1) 절(1.1, 2.3 …) 제목 옆에 "문제 N" 버튼을 붙여, 그 절의 문항만 팝업으로 연다.
 *  2) 본문의 핵심 개념어를 하이라이트하고, 누르면 그 개념을 다루는 문항만 팝업으로 연다.
 *  3) 팝업에서 보기를 고르면 정답 여부와 해설이 바로 표시된다.
 *
 *  페이지에서는 quiz/<주차>.js (window.QUIZ_DATA) 를 먼저 불러온 뒤 이 파일을 불러온다.
 */
(function () {
  'use strict';

  var D = window.QUIZ_DATA;
  if (!D || !D.items || !D.items.length) { return; }

  var byId = {};
  D.items.forEach(function (q) { byId[q.id] = q; });

  /* ---------------- 스타일 ---------------- */
  var css = document.createElement('style');
  css.textContent = [
    /* 절 제목 옆 버튼 */
    '.q-secbtn{margin-left:auto;flex-shrink:0;display:inline-flex;align-items:center;gap:5px;',
    'background:#fef2f2;border:1px solid #f3c7c7;color:#b91c1c;border-radius:999px;',
    'font-family:inherit;font-size:11px;font-weight:700;padding:5px 11px;cursor:pointer;line-height:1;transition:.15s}',
    '.q-secbtn:hover{background:#b91c1c;color:#fff;border-color:#b91c1c}',
    '.q-secbtn b{font-size:11.5px}',
    /* 본문 하이라이트 */
    '.q-hl{background:linear-gradient(transparent 58%,#ffe9a8 58%);border:0;padding:0 1px;',
    'font-weight:700;color:inherit;cursor:pointer;border-radius:2px;font-family:inherit;font-size:inherit;line-height:inherit}',
    '.q-hl:hover{background:#ffdf7e}',
    '.q-hl sup{font-size:9px;font-weight:700;color:#b91c1c;margin-left:1px;vertical-align:super}',
    /* 안내 배너 */
    '.q-banner{display:flex;gap:9px;align-items:flex-start;background:#fffbeb;border:1px solid #f59e0b;',
    'border-radius:10px;padding:11px 14px;font-size:12.5px;color:#78350f;line-height:1.7;margin:0 0 18px}',
    '.q-banner .q-tag{flex-shrink:0;background:#b45309;color:#fff;font-size:10.5px;font-weight:700;',
    'padding:3px 9px;border-radius:999px;white-space:nowrap;margin-top:1px}',
    /* 팝업 */
    '.q-back{position:fixed;inset:0;background:rgba(12,20,45,.55);z-index:400;display:none;',
    'align-items:center;justify-content:center;padding:16px}',
    '.q-back.on{display:flex}',
    '.q-modal{background:#f5f3ef;width:100%;max-width:680px;max-height:88vh;border-radius:14px;',
    'display:flex;flex-direction:column;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.4)}',
    '.q-head{background:#1a1a2e;color:#fff;padding:13px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0}',
    '.q-head h3{font-size:14px;font-weight:700;flex:1;min-width:0;line-height:1.4}',
    '.q-head h3 small{display:block;font-size:11px;font-weight:400;color:rgba(255,255,255,.62);margin-top:2px}',
    '.q-x{background:rgba(255,255,255,.12);border:0;color:#fff;width:30px;height:30px;border-radius:50%;',
    'font-size:17px;cursor:pointer;flex-shrink:0;line-height:1;font-family:inherit}',
    '.q-x:hover{background:rgba(255,255,255,.25)}',
    '.q-body{overflow-y:auto;-webkit-overflow-scrolling:touch;padding:14px 16px 22px;display:flex;flex-direction:column;gap:12px}',
    /* 문항 카드 */
    '.q-card{background:#fff;border:1px solid #e2ddd6;border-radius:10px;padding:14px 15px}',
    '.q-meta{display:flex;gap:6px;align-items:center;margin-bottom:8px;flex-wrap:wrap}',
    '.q-chip{font-size:10px;font-weight:700;padding:2px 8px;border-radius:999px;background:#eef2fb;color:#1b3f8b}',
    '.q-chip.d1{background:#e8f5ec;color:#166534}.q-chip.d2{background:#fff4e5;color:#92400e}',
    '.q-chip.d3{background:#fdecec;color:#b91c1c}',
    '.q-no{font-size:10.5px;font-weight:700;color:#6b7280;margin-left:auto}',
    '.q-text{font-size:13.5px;font-weight:700;line-height:1.65;color:#1a1a2e;margin-bottom:10px}',
    '.q-opts{display:flex;flex-direction:column;gap:6px}',
    '.q-opt{display:grid;grid-template-columns:22px 1fr;gap:9px;align-items:start;text-align:left;',
    'background:#fbfaf8;border:1px solid #e2ddd6;border-radius:8px;padding:9px 11px;',
    'font-family:inherit;font-size:12.5px;line-height:1.6;color:#374151;cursor:pointer;transition:.12s}',
    '.q-opt:hover{background:#eef2fb;border-color:#b9c8e8}',
    '.q-opt i{font-style:normal;font-weight:700;color:#6b7280;font-size:11.5px;line-height:1.7}',
    '.q-opt.ok{background:#f0fdf4;border-color:#22c55e;color:#14532d}',
    '.q-opt.ok i{color:#166534}',
    '.q-opt.no{background:#fef2f2;border-color:#ef4444;color:#7f1d1d}',
    '.q-opt.no i{color:#b91c1c}',
    '.q-opt[disabled]{cursor:default}',
    '.q-exp{margin-top:10px;background:#f8f7f4;border-left:3px solid #1b3f8b;border-radius:0 8px 8px 0;',
    'padding:10px 13px;font-size:12.5px;line-height:1.75;color:#2d3748;display:none}',
    '.q-exp.on{display:block}',
    '.q-exp b{color:#1b3f8b}',
    '.q-foot{flex-shrink:0;border-top:1px solid #e2ddd6;background:#fff;padding:10px 16px;',
    'display:flex;align-items:center;gap:10px;font-size:12px;color:#6b7280}',
    '.q-score{font-weight:700;color:#1a1a2e}',
    '.q-all{margin-left:auto;font-size:11.5px;font-weight:700;color:#1b3f8b;text-decoration:none;',
    'border:1px solid #b9c8e8;border-radius:8px;padding:6px 11px;background:#eef2fb}',
    '.q-all:hover{background:#1b3f8b;color:#fff}',
    '@media(max-width:520px){.q-modal{max-height:92vh}.q-back{padding:8px}}',
    'body.q-lock{overflow:hidden}'
  ].join('');
  document.head.appendChild(css);

  /* ---------------- 팝업 뼈대 ---------------- */
  var back = document.createElement('div');
  back.className = 'q-back';
  back.innerHTML =
    '<div class="q-modal" role="dialog" aria-modal="true" aria-labelledby="q-title">' +
      '<div class="q-head"><h3 id="q-title">문제<small></small></h3>' +
      '<button class="q-x" type="button" aria-label="닫기">&times;</button></div>' +
      '<div class="q-body"></div>' +
      '<div class="q-foot"><span class="q-score">푼 문제 0 / 0</span>' +
      '<a class="q-all" href="#" target="_blank" rel="noopener">100문제 전체 풀기 ↗</a></div>' +
    '</div>';
  document.body.appendChild(back);

  var elTitle = back.querySelector('#q-title');
  var elSub   = back.querySelector('#q-title small');
  var elBody  = back.querySelector('.q-body');
  var elScore = back.querySelector('.q-score');
  var elAll   = back.querySelector('.q-all');
  if (D.meta && D.meta.quiz) { elAll.href = D.meta.quiz; } else { elAll.style.display = 'none'; }

  function close() {
    back.classList.remove('on');
    document.body.classList.remove('q-lock');
  }
  back.querySelector('.q-x').addEventListener('click', close);
  back.addEventListener('click', function (e) { if (e.target === back) { close(); } });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && back.classList.contains('on')) { close(); }
  });

  var DIFF = { '기초': 'd1', '중급': 'd2', '심화': 'd3' };

  function open(title, sub, list) {
    elTitle.childNodes[0].nodeValue = title;
    elSub.textContent = sub;
    elBody.innerHTML = '';

    var done = 0, ok = 0;
    function refresh() {
      elScore.textContent = '푼 문제 ' + done + ' / ' + list.length +
        (done ? '  ·  정답 ' + ok + '개' : '');
    }
    refresh();

    list.forEach(function (q, idx) {
      var card = document.createElement('div');
      card.className = 'q-card';

      var meta = '<div class="q-meta"><span class="q-chip">' + q.sec + '</span>' +
        '<span class="q-chip ' + (DIFF[q.diff] || '') + '">' + q.diff + '</span>' +
        '<span class="q-no">' + (idx + 1) + ' / ' + list.length + ' · 원본 ' + q.id + '번</span></div>';

      var opts = q.opts.map(function (o, i) {
        return '<button class="q-opt" type="button" data-i="' + i + '">' +
               '<i>' + '①②③④⑤'.charAt(i) + '</i><span></span></button>';
      }).join('');

      card.innerHTML = meta +
        '<p class="q-text"></p>' +
        '<div class="q-opts">' + opts + '</div>' +
        '<div class="q-exp"><b>해설</b> — <span></span></div>';

      // 텍스트는 textContent로 넣어 원문을 그대로 안전하게 표시한다.
      card.querySelector('.q-text').textContent = q.q;
      card.querySelectorAll('.q-opt span').forEach(function (sp, i) { sp.textContent = q.opts[i]; });
      card.querySelector('.q-exp span').textContent = q.exp;

      var answered = false;
      card.querySelectorAll('.q-opt').forEach(function (btn) {
        btn.addEventListener('click', function () {
          if (answered) { return; }
          answered = true;
          var pick = +btn.getAttribute('data-i');
          card.querySelectorAll('.q-opt').forEach(function (b) {
            var i = +b.getAttribute('data-i');
            b.disabled = true;
            if (i === q.a) { b.classList.add('ok'); }
            else if (i === pick) { b.classList.add('no'); }
          });
          card.querySelector('.q-exp').classList.add('on');
          done++; if (pick === q.a) { ok++; }
          refresh();
        });
      });

      elBody.appendChild(card);
    });

    elBody.scrollTop = 0;
    back.classList.add('on');
    document.body.classList.add('q-lock');
  }

  /* ---------------- 1. 절마다 버튼 붙이기 ---------------- */
  var secTitles = {};
  document.querySelectorAll('section.sec').forEach(function (sec) {
    var badge = sec.querySelector('.sec-badge');
    var head = sec.querySelector('.sec-head');
    if (!badge || !head) { return; }
    var key = badge.textContent.trim();
    var list = D.items.filter(function (q) { return q.sec === key; });
    if (!list.length) { return; }

    var titleEl = sec.querySelector('.sec-title');
    var name = titleEl ? titleEl.childNodes[0].nodeValue.trim() : key;
    secTitles[key] = name;

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'q-secbtn';
    btn.innerHTML = '문제 <b>' + list.length + '</b>';
    btn.title = key + ' 관련 문제 ' + list.length + '개 풀어보기';
    btn.addEventListener('click', function () {
      open(key + ' ' + name, '이 절에서 출제된 ' + list.length + '문항', list);
    });
    head.appendChild(btn);
  });

  /* ---------------- 2. 본문 개념어 하이라이트 ---------------- */
  // 출제 문항이 많은 개념 위주로 최대 30개만 하이라이트한다.
  var terms = (D.terms || [])
    .filter(function (t) { return t.ids && t.ids.length >= 2; })
    .sort(function (a, b) { return b.ids.length - a.ids.length; })
    .slice(0, 30)
    // 긴 개념어를 먼저 매칭해야 짧은 개념어에 잘려나가지 않는다.
    .sort(function (a, b) { return b.t.length - a.t.length; });

  if (terms.length) {
    // 하이라이트 대상은 본문 텍스트 블록으로 한정한다.
    var scopes = document.querySelectorAll(
      'section.sec .note-box, section.sec .tip, section.sec .good, section.sec .warn, ' +
      'section.sec .card2 p, section.sec table td'
    );
    var termRe = new RegExp(
      terms.map(function (t) {
        return t.t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      }).join('|'), 'g');
    var termById = {};
    terms.forEach(function (t) { termById[t.t.toLowerCase()] = t; });

    var used = {};

    // 개념 하나를 하이라이트 버튼으로 만든다.
    // (while 루프 안에서 직접 만들면 var 스코프 탓에 모든 버튼이 마지막 개념을 가리키게 된다)
    function makeMark(word, info) {
      var mark = document.createElement('button');
      mark.type = 'button';
      mark.className = 'q-hl';
      mark.textContent = word;
      mark.title = '"' + info.t + '" 관련 문제 ' + info.ids.length + '개 보기';
      var sup = document.createElement('sup');
      sup.textContent = info.ids.length;
      mark.appendChild(sup);
      mark.addEventListener('click', function () {
        var list = info.ids.map(function (id) { return byId[id]; }).filter(Boolean);
        open(info.t, '이 개념을 다루는 ' + list.length + '문항', list);
      });
      return mark;
    }

    scopes.forEach(function (scope) {
      var walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT, {
        acceptNode: function (node) {
          if (!node.nodeValue.trim()) { return NodeFilter.FILTER_REJECT; }
          // 이미 하이라이트된 곳, 링크, 제목 안에는 다시 넣지 않는다.
          var p = node.parentNode;
          while (p && p !== scope) {
            var tag = p.nodeName.toLowerCase();
            if (tag === 'button' || tag === 'a' || tag === 'sup' || p.classList.contains('q-hl')) {
              return NodeFilter.FILTER_REJECT;
            }
            p = p.parentNode;
          }
          return NodeFilter.FILTER_ACCEPT;
        }
      });
      var nodes = [];
      var n;
      while ((n = walker.nextNode())) { nodes.push(n); }

      nodes.forEach(function (node) {
        var text = node.nodeValue;
        termRe.lastIndex = 0;
        if (!termRe.test(text)) { return; }
        termRe.lastIndex = 0;

        var frag = document.createDocumentFragment();
        var last = 0, m;
        while ((m = termRe.exec(text)) !== null) {
          var word = m[0];
          var info = termById[word.toLowerCase()];
          if (!info) { continue; }
          // 같은 개념은 페이지에서 처음 2번까지만 표시해 과도한 강조를 피한다.
          used[info.t] = (used[info.t] || 0) + 1;
          if (used[info.t] > 2) { continue; }

          if (m.index > last) { frag.appendChild(document.createTextNode(text.slice(last, m.index))); }
          frag.appendChild(makeMark(word, info));
          last = m.index + word.length;
        }
        if (!frag.childNodes.length) { return; }
        if (last < text.length) { frag.appendChild(document.createTextNode(text.slice(last))); }
        node.parentNode.replaceChild(frag, node);
      });
    });
  }

  /* ---------------- 3. 사용법 안내 배너 ---------------- */
  var firstSec = document.querySelector('section.sec');
  if (firstSec) {
    var banner = document.createElement('div');
    banner.className = 'q-banner';
    banner.innerHTML = '<span class="q-tag">문제 연동</span>' +
      '<div><b>노란 형광펜으로 표시된 개념</b>은 100문제 세트에서 실제로 출제된 개념입니다. ' +
      '개념을 누르면 <b>그 개념을 다루는 문항</b>이, 절 제목 옆 <b>문제 N</b> 버튼을 누르면 ' +
      '<b>그 절의 문항 전체</b>가 팝업으로 열리고 정답과 해설을 바로 확인할 수 있습니다. ' +
      '<span style="color:#92400e">숫자는 그 개념이 출제된 문항 수</span>입니다.</div>';
    firstSec.parentNode.insertBefore(banner, firstSec);
  }
})();
