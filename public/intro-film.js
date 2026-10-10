// Introduction film: a 64-second animated walkthrough drawn on a canvas, from data gathering to trading.
// Every frame is a pure function of the time, so play, pause, seek and chapters all share one renderer.
(() => {
  'use strict';
  const root = document.getElementById('introFilm');
  if (!root) return;
  const el = id => document.getElementById(id);
  const canvas = el('filmCanvas'), ctx = canvas.getContext('2d'), stage = canvas.parentElement;
  const C = {bg:'#0d1738', panel:'#131f47', line:'#26335f', text:'#e8ecf8', muted:'#9aa8cc', train:'#3b4a86', direct:'#c4861c', transfer:'#7480e8',
    price:'#a5b4fc', green:'#4fe0b0', amber:'#f5b54a', pink:'#ff8fab', cyan:'#7dd3fc'};
  const SANS = '-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Arial,sans-serif', MONO = 'ui-monospace,SFMono-Regular,Consolas,Menlo,monospace';
  let W = 960, H = 540, narrow = false, fade = 1;

  const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
  const ease = p => p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
  const prog = (t, a, b) => ease(clamp((t - a) / (b - a)));
  const F = px => Math.round(Math.max(10.5, px * W / 960));
  const alpha = a => { ctx.globalAlpha = clamp(a) * fade; };
  function text(value, x, y, {size = 14, color = C.text, align = 'left', weight = 400, mono = false, a = 1} = {}) {
    alpha(a); ctx.font = `${weight} ${F(size)}px ${mono ? MONO : SANS}`; ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
    ctx.fillText(value, x, y);
  }
  function box(x, y, w, h, r = 6) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2)); else ctx.rect(x, y, w, h);
  }
  function fillBox(x, y, w, h, color, a = 1, r = 6) { alpha(a); ctx.fillStyle = color; box(x, y, w, h, r); ctx.fill(); }
  function strokeBox(x, y, w, h, color, a = 1, r = 6, dash = []) { alpha(a); ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash(dash); box(x, y, w, h, r); ctx.stroke(); ctx.setLineDash([]); }
  function line(points, color, {width = 2, dash = [], a = 1, reveal = 1} = {}) {
    if (reveal <= 0) return;
    ctx.save(); alpha(a);
    if (reveal < 1) { ctx.beginPath(); ctx.rect(0, 0, points[0][0] + (points.at(-1)[0] - points[0][0]) * reveal, H); ctx.clip(); }
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.setLineDash(dash); ctx.beginPath();
    points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); ctx.restore();
  }
  function pill(label, x, y, color, a = 1, size = 13) {
    ctx.font = `500 ${F(size)}px ${SANS}`; const w = ctx.measureText(label).width + F(22), h = F(size) + F(14);
    fillBox(x, y - h / 2, w, h, C.panel, a, h / 2); strokeBox(x, y - h / 2, w, h, color, a * .8, h / 2);
    text(label, x + w / 2, y + F(size) * .36, {size, align: 'center', weight: 500, a});
    return w;
  }

  // ---- Scene 1: gather ----
  const sources = [['ENTSO-E', 'load, wind and solar forecasts', C.cyan], ['Energinet', 'prices, consumption, exchange', C.price],
    ['Weather', 'temperature, humidity, wind', C.amber], ['Gas market', 'daily gas price', C.pink]];
  function gather(t) {
    const cx = W * .05, cw = W * (narrow ? .36 : .29), ch = H * .115, top = H * .2, gap = H * .158;
    const tx = W * .6, tw = W * .35, ty = H * .2, th = H * .52, head = F(13) + 16;
    const shown = prog(t, .2, 1);
    fillBox(tx, ty, tw, th, C.panel, shown, 8); strokeBox(tx, ty, tw, th, C.line, shown, 8);
    text(narrow ? 'aligned table' : 'one table per resolution', tx + 12, ty + F(11) + 9, {size: 11, color: C.muted, mono: true, a: shown});
    const rows = 8, rowH = (th - head - 8) / rows, filled = clamp((t - 2) / 5.2) * rows, cellW = (tw - 24 - 18) / 4;
    for (let r = 0; r < rows; r++) for (let c = 0; c < 4; c++)
      fillBox(tx + 12 + c * (cellW + 6), ty + head + r * rowH, cellW, rowH - 5, sources[c][2], .3 * clamp(filled - r), 3);
    sources.forEach(([name, detail, color], i) => {
      const a = prog(t, .3 + i * .6, .9 + i * .6), y = top + i * gap;
      fillBox(cx, y, cw, ch, C.panel, a, 8); strokeBox(cx, y, cw, ch, color, a * .7, 8);
      fillBox(cx + 10, y + ch / 2 - 4, 8, 8, color, a, 4);
      text(name, cx + 26, y + ch / 2 + (narrow ? F(13) * .36 : -1), {size: 14, weight: 600, a});
      if (!narrow) text(detail, cx + 26, y + ch / 2 + F(11) + 3, {size: 11, color: C.muted, a});
      const x0 = cx + cw, y0 = y + ch / 2, x1 = tx, y1 = ty + th * (.2 + .2 * i), mid = (x0 + x1) / 2;
      alpha(a * .4); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.bezierCurveTo(mid, y0, mid, y1, x1, y1); ctx.stroke();
      if (a >= 1) for (let k = 0; k < 3; k++) {
        const p = (t * .45 + k / 3 + i * .13) % 1, q = 1 - p;
        const px = q * q * q * x0 + 3 * q * q * p * mid + 3 * q * p * p * mid + p * p * p * x1;
        const py = q * q * q * y0 + 3 * q * q * p * y0 + 3 * q * p * p * y1 + p * p * p * y1;
        alpha(.9); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(px, py, Math.max(2, W / 380), 0, Math.PI * 2); ctx.fill();
      }
    });
    const hourly = Math.round(47066 * prog(t, 2.4, 5.6)), quarters = Math.round(14108 * prog(t, 4.8, 7.4)), base = ty + th;
    text(`${hourly.toLocaleString('en-US')} hourly rows`, tx, base + F(15) + 10, {size: 15, weight: 600, a: prog(t, 2.2, 3)});
    if (!narrow) text('Feb 2020 → Sep 2025', tx + tw, base + F(15) + 10, {size: 12, color: C.muted, align: 'right', a: prog(t, 2.2, 3)});
    text(`${quarters.toLocaleString('en-US')} quarter-hours`, tx, base + F(15) * 2 + 20, {size: 15, weight: 600, a: prog(t, 4.6, 5.4)});
    if (!narrow) text('Oct 2025 → Mar 2026', tx + tw, base + F(15) * 2 + 20, {size: 12, color: C.muted, align: 'right', a: prog(t, 4.6, 5.4)});
  }

  // ---- Scene 2: clean ----
  function clean(t) {
    const lx = W * (narrow ? .27 : .21), rx = W * .94, names = ['load', 'wind', 'weather', 'gas'], colors = [C.cyan, C.price, C.amber, C.pink];
    const offsets = [.36, -.42, .22, -.3], ticks = 24, step = (rx - lx) / (ticks - 1), snap = prog(t, 1, 2.3), top = H * .215, rowGap = H * .072;
    for (let k = 0; k < ticks; k += 4) { alpha(.5 * snap); ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(lx + k * step, top - 10); ctx.lineTo(lx + k * step, top + rowGap * 3 + 10); ctx.stroke(); }
    names.forEach((name, i) => {
      const y = top + i * rowGap, appear = prog(t, .1 + i * .15, .6 + i * .15);
      text(name, W * .05, y + F(12) * .36, {size: 12, color: C.muted, mono: true, a: appear});
      for (let k = 0; k < ticks; k++) {
        const missing = i === 1 && k >= 9 && k <= 14, a = missing ? prog(t, 3.5 + (k - 9) * .12, 3.9 + (k - 9) * .12) : appear;
        fillBox(lx + k * step + offsets[i] * step * (1 - snap) - 1.5, y - H * .022, 3, H * .044, missing ? C.amber : colors[i], a * (missing ? 1 : .85), 1.5);
      }
    });
    const gapShown = prog(t, 2.4, 2.9) * (1 - prog(t, 5, 5.6));
    strokeBox(lx + 8.5 * step, top + rowGap - H * .032, 6 * step, H * .064, C.pink, gapShown, 4, [4, 3]);
    text(t < 3.9 ? '72-hour gap' : 'filled and flagged', lx + 11.5 * step, top + rowGap - H * .045, {size: 11, color: C.text, align: 'center', mono: true, a: gapShown});
    text('one UTC clock · 15-minute steps', rx, top - H * .045, {size: 12, color: C.muted, align: 'right', a: snap * (1 - gapShown)});

    const my = H * .575, rowH = H * .088, cols = ['at auction', '1 day back', '2 days back', '7 days back'], gap = 8, cw = (rx - lx - gap * 3) / 4;
    const rowsM = [['wind forecast', [1, 1, 1, 1]], ['temperature', [0, 1, 1, 1]], ['gas price', [0, 1, 1, 1]]];
    cols.forEach((col, c) => text(narrow ? col.replace(' back', '') : col, lx + c * (cw + gap) + cw / 2, my - 8, {size: 11, color: C.muted, align: 'center', mono: true, a: prog(t, 5 + c * .25, 5.5 + c * .25)}));
    rowsM.forEach(([name, known], r) => {
      const y = my + r * rowH;
      text(narrow ? name.replace(' forecast', ' fc').replace('temperature', 'temp').replace(' price', '') : name, W * .05, y + rowH / 2 + 2, {size: 12, color: C.muted, mono: true, a: prog(t, 5, 5.6)});
      known.forEach((ok, c) => {
        const a = prog(t, 5.2 + c * .3 + r * .1, 5.8 + c * .3 + r * .1), x = lx + c * (cw + gap);
        if (ok) { fillBox(x, y, cw, rowH - 8, C.price, a * .22, 4); strokeBox(x, y, cw, rowH - 8, C.price, a * .6, 4); }
        else { strokeBox(x, y, cw, rowH - 8, C.pink, a * .8, 4, [4, 3]); text('not known yet', x + cw / 2, y + (rowH - 8) / 2 + 4, {size: 10.5, color: C.muted, align: 'center', a: narrow ? 0 : a}); }
      });
    });
    text(narrow ? '76 features · only what was known' : '76 features · nothing from after the auction enters a row', W * .05, H * .91, {size: 14, weight: 600, a: prog(t, 7, 7.8)});
  }

  // ---- Scene 3: model ----
  const baseline = [.25, .3, .42, .52, .6, .52, .66, .74, .6, .48, .52, .62];
  const forecast = baseline.flatMap((b, h) => {
    const slope = Math.sign((baseline[h + 1] ?? b) - (baseline[h - 1] ?? b)) || 1;
    return [-.12, -.045, .04, .115].map((r, q) => b + slope * r + .018 * Math.sin(h * 7.3 + q * 2.1));
  });
  function model(t) {
    const x0 = W * .07, x1 = W * .93, y0 = H * .36, y1 = H * .76, n = forecast.length;
    const x = q => x0 + (x1 - x0) * q / n, y = v => y1 - v * (y1 - y0);
    for (let g = 0; g < 3; g++) { alpha(.6); ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0 + (y1 - y0) * g / 2); ctx.lineTo(x1, y0 + (y1 - y0) * g / 2); ctx.stroke(); }
    const legend = [[.5, C.text, [6, 5], narrow ? 'Hourly baseline · 47,066 hours' : 'Hourly baseline · an ensemble trained on 47,066 hours of history'],
      [3.4, C.price, null, narrow ? 'Residual · the 15-min correction' : 'Residual · a second ensemble learns the 15-minute correction'],
      [6.2, C.price, [], narrow ? '15-minute forecast' : '15-minute forecast · baseline plus residual']];
    legend.forEach(([start, color, dash, label], i) => {
      const a = prog(t, start, start + .7), ly = H * .2 + i * (F(13) + 9);
      if (dash) line([[x0, ly - 4], [x0 + 26, ly - 4]], color, {width: 2, dash, a}); else fillBox(x0 + 8, ly - 11, 10, 13, color, a * .35, 2);
      text(label, x0 + 36, ly, {size: 13, a});
    });
    forecast.forEach((value, q) => {
      const a = prog(t, 3.5 + q * .045, 4 + q * .045), b = baseline[Math.floor(q / 4)], w = (x1 - x0) / n * .56;
      fillBox(x(q + .5) - w / 2, y(b), w, (y(value) - y(b)) * a, C.price, .3, 0);
    });
    line(baseline.flatMap((b, h) => [[x(h * 4), y(b)], [x(h * 4 + 4), y(b)]]), C.text, {width: 1.6, dash: [6, 5], reveal: clamp((t - .7) / 2.6)});
    line(forecast.map((value, q) => [x(q + .5), y(value)]), C.price, {width: 2.4, reveal: clamp((t - 6.3) / 2.6)});
    const eqY = H * .875, parts = [['15-minute forecast', C.price, 6.3], ['=', null, 6.3], ['hourly baseline', C.muted, .7], ['+', null, 3.5], ['learned residual', C.price, 3.5]];
    let px = x0;
    const space = W / 960 * (narrow ? 5 : 12);
    parts.forEach(([label, color, start]) => {
      const a = prog(t, start, start + .7);
      if (color) px += pill(label, px, eqY, color, a, narrow ? 10.5 : 13) + space;
      else { text(label, px + F(15) * .3, eqY + F(15) * .36, {size: 15, color: C.muted, align: 'center', a}); px += F(15) * .6 + space; }
    });
    const stack = 'XGBoost · LightGBM · CatBoost → ridge stack';
    ctx.font = `400 ${F(12)}px ${MONO}`;
    if (px + ctx.measureText(stack).width + 12 < x1) text(stack, x1, eqY + F(12) * .36, {size: 12, color: C.muted, align: 'right', mono: true, a: prog(t, 8.2, 9)});
  }

  // ---- Scene 4: train (every model family was trained two ways: directly, and with transfer learning) ----
  const families = [['Tree ensembles', 'gradient-boosted trees'], ['SARIMAX', 'seasonal statistics'],
    ['LSTM', 'recurrent network'], ['CNN-LSTM', 'conv + recurrent']];
  function node(x, y, w, h, title, sub, color, a) {
    fillBox(x, y, w, h, C.panel, a, 8); strokeBox(x, y, w, h, color, a * .9, 8);
    const hasSub = sub && !narrow;
    text(title, x + w / 2, y + h / 2 + (hasSub ? -2 : F(12.5) * .36), {size: 12.5, weight: 600, align: 'center', a});
    if (hasSub) text(sub, x + w / 2, y + h / 2 + F(10.5) + 3, {size: 10.5, color: C.muted, align: 'center', a});
  }
  function arrow(x0, x1, y, color, t, start, label) {
    const reveal = prog(t, start, start + .5);
    if (reveal <= 0) return;
    line([[x0, y], [x0 + (x1 - x0) * reveal, y]], color, {width: 1.8, a: .9});
    if (reveal >= 1) {
      alpha(.95); ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x1 - 7, y - 4); ctx.lineTo(x1 - 7, y + 4); ctx.fill();
      if (x1 - x0 > 40) { ctx.beginPath(); ctx.arc(x0 + (x1 - 10 - x0) * ((t * .6) % 1), y, Math.max(2, W / 380), 0, Math.PI * 2); ctx.fill(); }
    }
    if (label && !narrow) text(label, (x0 + x1) / 2, y - 8, {size: 10.5, color: C.muted, align: 'center', mono: true, a: reveal});
  }
  function train(t) {
    const left = W * .06, right = W * .94, bh = H * (narrow ? .1 : .115), yA = H * .235, yB = H * (narrow ? .4 : .41);
    const bw = W * (narrow ? .24 : .22), last = W * (narrow ? .26 : .27), mid = (left + bw + right - last) / 2 - bw / 2;
    text(narrow ? 'DIRECT' : 'DIRECT · FROM SCRATCH', left, yA - 8, {size: 10.5, color: C.muted, mono: true, a: prog(t, .2, .7)});
    node(left, yA, bw, bh, narrow ? '15-min data' : '15-minute data', '14,108 quarter-hours', C.direct, prog(t, .2, .8));
    arrow(left + bw + 6, right - last - 6, yA + bh / 2, C.direct, t, .8);
    node(right - last, yA, last, bh, narrow ? '15-min model' : '15-minute model', '15-minute data only', C.direct, prog(t, 1.2, 1.8));
    text('TRANSFER LEARNING', left, yB - 8, {size: 10.5, color: C.muted, mono: true, a: prog(t, 2, 2.5)});
    node(left, yB, bw, bh, 'Hourly data', '47,066 hours', C.transfer, prog(t, 2, 2.6));
    arrow(left + bw + 6, mid - 6, yB + bh / 2, C.transfer, t, 2.6);
    node(mid, yB, bw, bh, narrow ? 'Hourly model' : 'Hourly source model', 'trained first', C.transfer, prog(t, 3, 3.6));
    arrow(mid + bw + 6, right - last - 6, yB + bh / 2, C.transfer, t, 3.6, 'transfer');
    node(right - last, yB, last, bh, narrow ? '15-min model' : '15-minute model', 'builds on the hourly model', C.transfer, prog(t, 4, 4.6));

    const cols = narrow ? 2 : 4, gap = W * .015, cw = (right - left - gap * (cols - 1)) / cols, cy = H * (narrow ? .56 : .62), ch = H * (narrow ? .135 : .2);
    const place = i => [left + (i % cols) * (cw + gap), cy + Math.floor(i / cols) * (ch + gap)];
    families.forEach(([name, detail], i) => {
      const start = 4.8 + i * .45, a = prog(t, start, start + .5), [x, y] = place(i);
      fillBox(x, y, cw, ch, C.panel, a, 8); strokeBox(x, y, cw, ch, C.line, a, 8);
      text(name, x + 12, y + F(13) + 9, {size: 13, weight: 600, a});
      ctx.font = `400 ${F(10.5)}px ${SANS}`;
      if (!narrow && ctx.measureText(detail).width < cw - 20) text(detail, x + 12, y + F(13) + F(10.5) + 13, {size: 10.5, color: C.muted, a});
      [['direct', C.direct, .3], ['+ transfer', C.transfer, .6]].forEach(([label, color, delay], k) => {
        const on = prog(t, start + delay, start + delay + .4), bx = x + 12 + k * cw * .42, by = y + ch - 12;
        fillBox(bx, by - 8, 8, 8, color, on, 4); text(label, bx + 13, by, {size: 10.5, a: on});
      });
    });
    // Bracket over the two neural networks (the last two cards; on narrow screens they form the second row).
    const [nx, ny] = place(2), bracket = prog(t, 6.6, 7.2), lineY = narrow ? ny + ch + 7 : ny - 8, width = cw * 2 + gap;
    line([[nx, lineY + (narrow ? -4 : 4)], [nx, lineY], [nx + width, lineY], [nx + width, lineY + (narrow ? -4 : 4)]], C.muted, {width: 1, a: bracket * .8});
    text('neural networks', nx + width / 2, narrow ? lineY + F(10.5) + 4 : lineY - 6, {size: 10.5, color: C.muted, align: 'center', mono: true, a: bracket});
    text(narrow ? 'All four were trained both ways.' : 'All four families were trained both ways, including the neural networks.',
      left, H * (narrow ? .965 : .935), {size: 13.5, weight: 600, a: prog(t, 7.3, 8)});
  }

  // ---- Scene 5: validate ----
  const errors = [['Transfer, stacked', 17.85, 'best only in hindsight'], ['Hourly baseline', 18.48, 'selected, all 8 blocks'], ['Transfer, averaged', 19.81, ''], ['Direct 15-minute', 21.30, '']];
  function validate(t) {
    // Wide: the walk-forward diagram and the error bars sit side by side. Narrow: one after the other, full size.
    const first = narrow ? 1 - prog(t, 5, 5.5) : 1, second = narrow ? prog(t, 5.5, 6.1) : 1;
    const dx0 = W * .06, dx1 = W * (narrow ? .94 : .56), dy = H * .3, rowH = H * (narrow ? .058 : .052), w = dx1 - dx0, days = 35 + 14 * 8;
    fillBox(dx0, dy - H * .075, 12, 10, C.train, first * prog(t, .2, .8), 2);
    text(narrow ? 'train: earlier data' : 'train: everything earlier', dx0 + 18, dy - H * .075 + 9, {size: 11.5, color: C.muted, a: first * prog(t, .2, .8)});
    const lx = dx0 + F(11.5) * (narrow ? 11.5 : 14.5);
    fillBox(lx, dy - H * .075, 12, 10, C.green, first * prog(t, .4, 1), 2); text('test: next 14 days', lx + 18, dy - H * .075 + 9, {size: 11.5, color: C.muted, a: first * prog(t, .4, 1)});
    for (let i = 0; i < 8; i++) {
      const a = prog(t, .6 + i * .42, 1 + i * .42), y = dy + i * rowH, trainW = (35 + 14 * i) / days * w, testW = 14 / days * w;
      fillBox(dx0, y, trainW * a - 2, rowH - 5, C.train, a * first, 3); fillBox(dx0 + trainW, y, (testW - 2) * a, rowH - 5, C.green, a * first, 3);
    }
    text(narrow ? '8 blocks · 10,744 unseen quarter-hours' : '8 blocks · 10,744 quarter-hours no model had seen', dx0, dy + 8 * rowH + F(13) + 4, {size: 13, weight: 600, a: first * prog(t, 4.3, 5)});

    const bx0 = narrow ? W * .06 : W * .63, bx1 = W * .94, by = H * .3, bw = bx1 - bx0, slot = H * .125;
    text('Out-of-sample error · MAE, EUR/MWh', bx0, by - H * .075 + 9, {size: 11.5, color: C.muted, a: second * prog(t, 5.2, 5.8)});
    errors.forEach(([name, value, note], i) => {
      const a = second * prog(t, 5.6 + i * .3, 6.4 + i * .3), y = by - H * .02 + i * slot, selected = i === 1, barH = Math.max(5, H * .016), ty = y + F(12);
      text(name, bx0, ty, {size: 12, weight: selected ? 600 : 400, a});
      text(value.toFixed(2), bx1, ty, {size: 12, weight: 600, align: 'right', mono: true, a});
      if (note) text(note, bx0, ty + barH + F(10.5) + 9, {size: 10.5, color: C.muted, a: second * prog(t, 7.6, 8.3)});
      fillBox(bx0, ty + 6, bw, barH, C.line, a * .7, barH / 2); fillBox(bx0, ty + 6, bw * value / 24 * a, barH, C.price, a * (selected ? 1 : .5), barH / 2);
    });
    text(narrow ? 'Used only if it beat the baseline before.' : 'A model is used only if it beat the baseline on earlier data. The hourly baseline was chosen every time.',
      dx0, H * .92, {size: 13.5, weight: 600, a: prog(t, 8.2, 9)});
  }

  // ---- Scene 6: trade (a schematic path: held intervals win and lose, costs are paid on every change) ----
  const signal = Array.from({length: 61}, (_, i) => .78 * Math.sin(i * .21 + .4) + .2 * Math.sin(i * .63));
  const position = signal.map(value => value > .5 ? 1 : value < -.5 ? -1 : 0);
  const pnl = position.reduce((path, pos, i) => { path.push(path[i] + Math.abs(pos) * (.03 + .075 * Math.sin(i * 1.7)) - Math.abs(pos - (position[i - 1] || 0)) * .03); return path; }, [0]).slice(1);
  function trade(t) {
    const x0 = W * (narrow ? .25 : .15), x1 = W * .93, n = signal.length, x = i => x0 + (x1 - x0) * i / (n - 1), shown = clamp((t - .6) / 5.6), upto = shown * (n - 1);
    const sTop = H * .2, sBot = H * .46, sy = v => (sTop + sBot) / 2 - v * (sBot - sTop) / 2, colW = (x1 - x0) / (n - 1);
    position.forEach((pos, i) => { if (pos && i <= upto) fillBox(x(i) - colW / 2, sTop, colW + .5, sBot - sTop, pos > 0 ? C.green : C.pink, .16, 0); });
    [.5, -.5].forEach(level => line([[x0, sy(level)], [x1, sy(level)]], C.muted, {width: 1, dash: [4, 4], a: .7 * prog(t, .2, .8)}));
    line(signal.map((value, i) => [x(i), sy(value)]), C.amber, {width: 2.2, reveal: shown});
    text('signal', W * .05, sy(0) + 4, {size: 12, color: C.muted, mono: true, a: prog(t, .2, .8)});
    if (!narrow) { text('long above', x1, sy(.5) - 6, {size: 10.5, color: C.muted, align: 'right', a: prog(t, .4, 1)}); text('short below', x1, sy(-.5) + F(10.5) + 4, {size: 10.5, color: C.muted, align: 'right', a: prog(t, .4, 1)}); }
    const pMid = H * .56, pH = H * .045;
    text('position', W * .05, pMid + 4, {size: 12, color: C.muted, mono: true, a: prog(t, .8, 1.4)});
    line([[x0, pMid], [x1, pMid]], C.line, {width: 1, a: prog(t, .8, 1.4)});
    position.forEach((pos, i) => { if (pos && i <= upto) fillBox(x(i) - colW * .36, pos > 0 ? pMid - pH : pMid, colW * .72, pH, pos > 0 ? C.green : C.pink, .85, 1.5); });
    const lTop = H * .66, lBot = H * .84, hi = Math.max(...pnl), lo = Math.min(0, ...pnl), ly = v => lBot - (v - lo) / (hi - lo) * (lBot - lTop);
    text('P&L', W * .05, (lTop + lBot) / 2 + 4, {size: 12, color: C.muted, mono: true, a: prog(t, 1.2, 1.8)});
    line([[x0, ly(0)], [x1, ly(0)]], C.line, {width: 1, dash: [3, 4], a: prog(t, 1.2, 1.8)});
    line(pnl.map((value, i) => [x(i), ly(value)]), C.price, {width: 2.4, reveal: shown});
    if (!narrow) text('after costs', W * .05, (lTop + lBot) / 2 + F(10.5) + 8, {size: 10.5, color: C.muted, a: prog(t, 1.2, 1.8)});
    let px = narrow ? W * .05 : x0;
    [['walk-forward tuning', 'walk-forward', 6.6], ['significance tests', 'p-values', 7.1], ['portfolio of rules', 'portfolio', 7.6]].forEach(([label, short, start]) => {
      px += pill(narrow ? short : label, px, H * .925, C.line, prog(t, start, start + .6), narrow ? 10.5 : 12.5) + W / 960 * 10;
    });
  }

  function outro(t) {
    text('Now test your own rule.', W / 2, H * .4, {size: narrow ? 24 : 34, weight: 300, align: 'center', a: prog(t, .2, 1)});
    text('Every backtest on this site runs on those unseen forecasts.', W / 2, H * .4 + F(15) + 16, {size: narrow ? 12 : 15, color: C.muted, align: 'center', a: prog(t, .6, 1.4)});
  }

  const SCENES = [
    {title: 'Gather', dur: 9, draw: gather, caption: 'Gather. More than five years of hourly DK1 market data and five months at 15-minute resolution: load, wind and solar forecasts from ENTSO-E, prices and system data from Energinet, plus weather and gas prices.'},
    {title: 'Clean', dur: 9, draw: clean, caption: 'Clean. Every series goes onto one UTC clock and coverage is checked; a 72-hour hole in the wind and solar forecasts is filled and flagged. Each row then gets only values known before the auction, lagged by one, two and seven days: 76 features.'},
    {title: 'Model', dur: 11, draw: model, caption: 'Model. An ensemble trained on the long hourly history gives every quarter-hour a baseline. A second ensemble learns the 15-minute correction on top of it: transfer learning plus residual learning. A direct 15-minute model is kept as a benchmark.'},
    {title: 'Train', dur: 10, draw: train, caption: 'Train. Four model families were each trained two ways: directly on the five months of 15-minute data, and with transfer learning, where a model trained first on the long hourly history passes what it learned to the 15-minute model. That covers the tree ensembles and SARIMAX as well as the neural networks, the LSTM and the CNN-LSTM.'},
    {title: 'Validate', dur: 11, draw: validate, caption: 'Validate. Walk-forward: retrain on everything earlier, forecast the next 14 days, eight times. A model is only used if it beat the baseline on earlier data, and the hourly baseline was chosen in all eight blocks. Stacking scored best, but only in hindsight.'},
    {title: 'Trade', dur: 10, draw: trade, caption: 'Trade. A rule turns the forecast into long, short or flat. Positions are sized, costed and settled, then checked with walk-forward tuning, significance tests and a portfolio view.'},
    {title: '', dur: 4, draw: outro, caption: 'Now test your own rule in the Strategy Lab below.'},
  ];
  let cursor = 0;
  for (const scene of SCENES) { scene.start = cursor; cursor += scene.dur; }
  const TOTAL = cursor, chapters = SCENES.filter(scene => scene.title);
  const sceneAt = t => SCENES.findLast(scene => t >= scene.start) || SCENES[0];

  // ---- Player ----
  let time = 0, playing = false, lastFrame = 0, frame = 0, userPaused = false, shownScene = null, poster = false;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const clock = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  function render() {
    const scene = sceneAt(Math.min(time, TOTAL - .001)), local = time - scene.start;
    ctx.setTransform(canvas.width / W, 0, 0, canvas.height / H, 0, 0); ctx.globalAlpha = 1;
    const sky = ctx.createLinearGradient(0, 0, W, H); sky.addColorStop(0, '#101c44'); sky.addColorStop(1, C.bg); ctx.fillStyle = sky; ctx.fillRect(0, 0, W, H);
    fade = clamp(local / .45) * (scene === SCENES.at(-1) ? 1 : clamp((scene.dur - local) / .35));
    scene.draw(local); ctx.globalAlpha = 1;
    el('filmSeek').value = time; el('filmSeek').style.setProperty('--played', `${time / TOTAL * 100}%`);
    el('filmTime').textContent = `${clock(time)} / ${clock(TOTAL)}`;
    if (scene !== shownScene) {
      shownScene = scene; const index = chapters.indexOf(scene);
      el('filmChip').textContent = index >= 0 ? `0${index + 1} / 0${chapters.length} · ${scene.title}` : 'Next · Strategy Lab';
      el('filmCaption').textContent = scene.caption;
      root.querySelectorAll('[data-chapter]').forEach(button => button.setAttribute('aria-current', String(Number(button.dataset.chapter) === index)));
    }
    el('filmEnd').hidden = time < TOTAL - 3.2;
  }
  function setPlaying(value) {
    playing = value; root.classList.toggle('is-playing', playing);
    el('filmPlay').setAttribute('aria-label', playing ? 'Pause the introduction' : 'Play the introduction'); el('filmPlay').textContent = playing ? '❚❚' : '▶';
    el('filmBigPlay').hidden = playing || (time > 0 && !poster);
    cancelAnimationFrame(frame);
    if (playing) { lastFrame = performance.now(); frame = requestAnimationFrame(tick); }
  }
  function tick(now) {
    time = Math.min(TOTAL, time + Math.min(.1, (now - lastFrame) / 1000)); lastFrame = now; render();
    if (time >= TOTAL) setPlaying(false); else frame = requestAnimationFrame(tick);
  }
  function seek(t, play = playing) { time = clamp(t, 0, TOTAL); render(); setPlaying(play && time < TOTAL); }
  function toggle() {
    if (playing) { userPaused = true; setPlaying(false); return; }
    userPaused = false; if (time >= TOTAL || poster) time = 0;
    poster = false; setPlaying(true);
  }
  function resize() {
    const rect = stage.getBoundingClientRect(); if (!rect.width) return;
    const scale = window.devicePixelRatio || 1; W = rect.width; H = rect.height; narrow = W <= 560;
    canvas.width = Math.round(W * scale); canvas.height = Math.round(H * scale); render();
  }

  el('filmChapters').replaceChildren(...chapters.map((scene, i) => {
    const button = document.createElement('button'); button.type = 'button'; button.dataset.chapter = String(i);
    const number = document.createElement('span'); number.textContent = `0${i + 1}`; const name = document.createElement('b'); name.textContent = scene.title;
    button.append(number, name); button.addEventListener('click', () => { userPaused = false; seek(scene.start, true); });
    return button;
  }));
  el('filmTranscript').replaceChildren(...chapters.map(scene => { const item = document.createElement('li'); item.textContent = scene.caption; return item; }));
  el('filmSeek').max = String(TOTAL);
  el('filmSeek').addEventListener('input', event => { userPaused = true; seek(Number(event.target.value), false); });
  el('filmPlay').addEventListener('click', toggle); el('filmBigPlay').addEventListener('click', toggle); canvas.addEventListener('click', toggle);
  el('filmReplay').addEventListener('click', () => { userPaused = false; seek(0, true); });
  new ResizeObserver(resize).observe(stage);
  // Plays while it is on screen, unless the visitor paused it or prefers reduced motion (then it waits on a finished frame).
  if (reduced) { time = SCENES[2].start + 9.5; userPaused = true; poster = true; }
  new IntersectionObserver(entries => {
    const visible = entries.some(entry => entry.isIntersecting);
    if (visible && !playing && !userPaused && time < TOTAL) setPlaying(true); else if (!visible && playing) setPlaying(false);
  }, {threshold: .55}).observe(stage);
  document.addEventListener('visibilitychange', () => { if (document.hidden && playing) setPlaying(false); });
  resize(); setPlaying(false);
})();
