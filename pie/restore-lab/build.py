import json
D = json.load(open('data.json'))
html = r'''<title>Restoration Lab</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one column of voice strips; each strip = EQ curve, a version switcher that keeps the playhead, its clips. */
:root {
  --paper: #eef0ec; --ink: #1c1f23; --muted: #5b6168; --rule: #d2d6cf; --well: #e3e6df; --sel: #d8ddd2;
  --conv: #3c5a7a; --fix: #a8361f; --grid: #c9cec5;
  --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", sans-serif;
  --body: "Source Serif 4", Georgia, serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #15171a; --ink: #e4e6e1; --muted: #9aa0a7; --rule: #2e3237; --well: #1f2226; --sel: #2b3035;
  --conv: #8fb0d3; --fix: #f07a62; --grid: #33383d; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #15171a; --ink: #e4e6e1; --muted: #9aa0a7; --rule: #2e3237; --well: #1f2226; --sel: #2b3035;
  --conv: #8fb0d3; --fix: #f07a62; --grid: #33383d; color-scheme: dark }
body { background: var(--paper); color: var(--ink); font: 17px/1.6 var(--body); }
.wrap { max-width: 860px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 72px; display: grid; gap: 44px; }
h1, h2, h3 { font-family: var(--display); margin: 0; line-height: 1.12; text-wrap: balance; }
h1 { font-size: clamp(32px, 6vw, 50px); letter-spacing: -0.02em; }
h2 { font-size: 22px; }
p { margin: 0; max-width: 66ch; }
.eyebrow { font: 500 12px var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.hint { font-size: 15px; color: var(--muted); }
section { display: grid; gap: 14px; }
.voice { display: grid; gap: 12px; border-top: 1px solid var(--rule); padding-top: 18px; }
.vhead { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: baseline; justify-content: space-between; }
.scores { font: 13px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.scores b { color: var(--ink); font-weight: 500; }
.curve { width: 100%; height: auto; max-width: 100%; display: block; }
.curve text { fill: var(--muted); font: 10px var(--mono); }
.player { display: grid; gap: 10px; background: var(--well); padding: 12px; }
.player .words { font-style: italic; overflow-wrap: anywhere; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
button { font: 500 14px var(--display); border: 1px solid var(--rule); background: var(--paper); color: var(--ink); padding: 6px 12px; cursor: pointer; }
button:focus-visible { outline: 2px solid var(--fix); outline-offset: 2px; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.track { flex: 1 1 140px; min-width: 100px; height: 6px; background: var(--rule); cursor: pointer; position: relative; }
.track i { position: absolute; inset: 0 auto 0 0; background: var(--fix); width: 0; }
.time { font: 12px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.list { display: grid; max-height: 240px; overflow-y: auto; border: 1px solid var(--rule); }
.row { display: grid; grid-template-columns: 1fr auto; gap: 2px 12px; padding: 7px 10px; border: 0; border-bottom: 1px solid var(--rule); background: transparent; text-align: left; font: 15px var(--body); }
.row:hover { background: var(--well); }
.row[aria-selected="true"] { background: var(--sel); }
.row .d { font: 12px var(--mono); color: var(--muted); }
table.t { border-collapse: collapse; font-size: 14px; min-width: 420px; }
table.t th, table.t td { text-align: left; padding: 5px 12px 5px 0; border-bottom: 1px solid var(--rule); }
table.t th { font: 500 12px var(--mono); color: var(--muted); letter-spacing: 0.04em; }
table.t td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header style="display:grid;gap:12px">
    <div class="eyebrow">corpus v2 · restoration · step 1 of the pilot</div>
    <h1>Restoration Lab</h1>
    <p>One repair at a time on the converted corpus. Step 1 is mastering EQ: each voice gets its own tonal correction toward the long-term spectrum of clean professional recordings of its gender, as boosts only (it fills holes and never thins a voice), smoothed and capped at +4 dB. All versions are matched in loudness, so you hear tone, not level.</p>
  </header>
  <section>
    <h2>Step 1 · mastering EQ</h2>
    <p class="hint">For each voice: the curve it gets (dB per third octave), then 12 corpus clips. Switch between the conversion as it is, half the correction and the full correction while it plays; you stay in place. Number keys switch the player you last used. The scores are naturalness (UTMOS, higher is better) and the recognizer's misses (lower is better), as a check that nothing breaks: EQ is for how it sounds.</p>
    <div id="voices" style="display:grid;gap:30px"></div>
  </section>
  <section>
    <h2>Step 2 · the female s</h2>
    <p class="hint">On top of the full EQ from step 1. In the female conversions s slides toward ʃ: its energy centres around 5.6-6.2 kHz, where a female s sits near 7-8 kHz. Inside each s only, the correction cuts 2.5-5 kHz and lifts 6.5-10 kHz, in three strengths. The table gives the s's the recognizer misses, where its energy centres, and naturalness.</p>
    <div id="step2" style="display:grid;gap:30px"></div>
  </section>
  <section>
    <h2>Step 3 · stop bursts</h2>
    <p class="hint">On top of steps 1 and 2. The converter weakens the release bursts of p t k by 3-8 dB. Two repairs: a fixed lift of 2-8 kHz in the 18 ms after each release, and a per-burst fill of half of that burst's own shortfall against the source (its loudness relative to the next vowel). Neither helps the recognizer except for Valeria, and naturalness does not move; this is for your ear: does it sound crisper, or harsher?</p>
    <div id="step3" style="display:grid;gap:30px"></div>
  </section>
</div>

<script>
const D = __DATA__;
const VERS = [['conv', 'Conversion', 'conv'], ['tone50', 'Half EQ', 'fix'], ['tone', 'Full EQ', 'fix']];
const fmt = s => isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : '0:00';
const players = []; let lastPlayer = null;
function stopAll(ex) { for (const p of players) if (p !== ex) p.stop(); }
function Player(host, versions, head) {
  const P = { side: 0, playing: false };
  host.innerHTML = `<div class="head">${head}</div><div class="controls"><button class="play" type="button">Play</button>
    <span class="seg" role="group" aria-label="Version">${versions.map((v, i) => `<button type="button" aria-pressed="${i === 0}">${v.label}</button>`).join('')}</span>
    <div class="track" role="slider" tabindex="0" aria-label="Position"><i></i></div><span class="time">0:00</span></div>`;
  const playBtn = host.querySelector('.play'), bar = host.querySelector('.track i'), time = host.querySelector('.time'), track = host.querySelector('.track');
  const btns = [...host.querySelectorAll('.seg button')];
  let auds = [];
  const cur = () => auds[P.side];
  const pos = () => cur() ? cur().currentTime : 0;
  const dur = () => Math.max(...auds.map(a => a.duration || 0));
  const setPos = t => auds.forEach(a => { try { a.currentTime = Math.max(0, Math.min(t, (a.duration || t) - 0.01)); } catch (e) {} });
  function draw() { const d = dur(), t = pos(); bar.style.width = d ? `${Math.min(100, t / d * 100)}%` : '0'; time.textContent = `${fmt(t)} / ${fmt(d)}`; }
  P.stop = () => { auds.forEach(a => a.pause()); P.playing = false; playBtn.textContent = 'Play'; };
  P.play = () => { stopAll(P); lastPlayer = P; setPos(pos()); cur().play().catch(() => {}); P.playing = true; playBtn.textContent = 'Pause'; };
  P.pick = i => { if (!auds[i]) return; const t = pos(), was = P.playing; cur().pause(); P.side = i; setPos(t); btns.forEach((b, k) => b.setAttribute('aria-pressed', k === i)); if (was) cur().play().catch(() => {}); lastPlayer = P; draw(); };
  P.load = (vs, h) => { P.stop(); auds.forEach(a => { a.pause(); a.src = ''; });
    auds = vs.map(v => { const a = new Audio(); a.preload = 'metadata'; a.src = v.src; a.addEventListener('timeupdate', draw); a.addEventListener('loadedmetadata', draw); a.addEventListener('ended', () => { if (a === cur()) { P.stop(); setPos(0); draw(); } }); return a; });
    if (h !== undefined) host.querySelector('.head').innerHTML = h; draw(); };
  playBtn.onclick = () => P.playing ? P.stop() : P.play();
  btns.forEach((b, i) => b.onclick = () => P.pick(i));
  track.onclick = e => { const r = track.getBoundingClientRect(); setPos((e.clientX - r.left) / r.width * dur()); draw(); };
  track.onkeydown = e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { setPos(pos() + (e.key === 'ArrowRight' ? 1 : -1)); draw(); e.preventDefault(); } };
  P.load(versions); players.push(P); return P;
}
function curveSVG(eq) {
  const W = 600, H = 120, L = 34, R = 10, T = 10, B = 22, fmin = 100, fmax = 16000, gmax = 4;
  const x = f => L + (Math.log(f / fmin) / Math.log(fmax / fmin)) * (W - L - R), y = g => T + (1 - g / gmax) * (H - T - B);
  const ticks = [100, 250, 500, 1000, 2000, 4000, 8000, 16000];
  let s = `<svg class="curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="EQ curve">`;
  for (const g of [0, 2, 4]) s += `<line x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}" stroke="var(--grid)"/><text x="${L - 6}" y="${y(g) + 3}" text-anchor="end">+${g}</text>`;
  for (const f of ticks) s += `<text x="${x(f)}" y="${H - 6}" text-anchor="middle">${f >= 1000 ? f / 1000 + 'k' : f}</text>`;
  const pts = eq.filter(b => b.f <= fmax).map(b => `${x(b.f).toFixed(1)},${y(b.g).toFixed(1)}`).join(' ');
  s += `<polyline points="${pts}" fill="none" stroke="var(--fix)" stroke-width="2"/></svg>`;
  return s;
}
const host = document.getElementById('voices');
for (const v of D.voices) {
  const el = document.createElement('div'); el.className = 'voice';
  const sc = k => `<b>${v.mos[k].toFixed(2)}</b> / ${v.miss[k]}%`;
  el.innerHTML = `<div class="vhead"><h3>${v.name}</h3><span class="scores">naturalness / misses · conversion ${sc('conv')} · half ${sc('tone50')} · full ${sc('tone')}</span></div>${curveSVG(v.eq)}<div class="player"></div><div class="list" role="listbox" aria-label="${v.name} clips"></div>`;
  host.append(el);
  let sel = v.clips[0].id;
  const vers = id => VERS.map(([k, label]) => ({ label, src: `a/${v.tag}/${id}-${k}.mp3` }));
  const head = c => `<div class="words">${c.words}</div>`;
  const pl = Player(el.querySelector('.player'), vers(sel), head(v.clips[0]));
  const list = el.querySelector('.list');
  const render = () => { list.innerHTML = ''; for (const c of v.clips) { const b = document.createElement('button'); b.type = 'button'; b.className = 'row'; b.setAttribute('role', 'option'); b.setAttribute('aria-selected', c.id === sel);
    b.innerHTML = `<span>${c.words}</span><span class="d">${c.dur.toFixed(1)} s</span>`; b.onclick = () => { sel = c.id; pl.load(vers(c.id), head(c)); render(); pl.play(); }; list.append(b); } };
  render();
}
const host2 = document.getElementById('step2');
const V2 = [['tone', 'Step 1 only'], ['s1', 'Light'], ['s2', 'Medium'], ['s3', 'Strong']];
for (const v of D.step2) {
  const el = document.createElement('div'); el.className = 'voice';
  const rows = v.rows.map((r, i) => `<tr><td>${V2[i][1]}${i ? ' (' + v.cut_lift[i - 1] + ' dB)' : ''}</td><td class="n">${r.s[0]} / ${r.s[1]}</td><td class="n">${(r.cent / 1000).toFixed(1)} kHz</td><td class="n">${r.mos.toFixed(2)}</td></tr>`).join('');
  el.innerHTML = `<div class="vhead"><h3>${v.name}</h3></div><div style="overflow-x:auto"><table class="t"><thead><tr><th>Version</th><th>s missed</th><th>s centre</th><th>Naturalness</th></tr></thead><tbody>${rows}</tbody></table></div><div class="player"></div><div class="list" role="listbox" aria-label="${v.name} clips"></div>`;
  host2.append(el);
  let sel = v.clips[0].id;
  const vers = id => V2.map(([k, label]) => ({ label, src: `a/${v.tag}/${id}-${k}.mp3` }));
  const head = c => `<div class="words">${c.words}</div>`;
  const pl = Player(el.querySelector('.player'), vers(sel), head(v.clips[0]));
  const list = el.querySelector('.list');
  const render = () => { list.innerHTML = ''; for (const c of v.clips) { const b = document.createElement('button'); b.type = 'button'; b.className = 'row'; b.setAttribute('role', 'option'); b.setAttribute('aria-selected', c.id === sel);
    b.innerHTML = `<span>${c.words}</span><span class="d">${c.dur.toFixed(1)} s</span>`; b.onclick = () => { sel = c.id; pl.load(vers(c.id), head(c)); render(); pl.play(); }; list.append(b); } };
  render();
}
const host3 = document.getElementById('step3');
const V3 = [['c2', 'Steps 1-2'], ['b_eq', 'Fixed lift'], ['b_ct50', 'Half fill']];
for (const v of D.step3) {
  const el = document.createElement('div'); el.className = 'voice';
  const rows = v.rows.map((r, i) => `<tr><td>${V3[i][1]}</td><td class="n">${r.miss}%</td><td class="n">${r.burst} dB</td><td class="n">${r.mos.toFixed(2)}</td></tr>`).join('');
  el.innerHTML = `<div class="vhead"><h3>${v.name}</h3></div><div style="overflow-x:auto"><table class="t"><thead><tr><th>Version</th><th>Misses</th><th>Burst re vowel</th><th>Naturalness</th></tr></thead><tbody>${rows}</tbody></table></div><div class="player"></div><div class="list" role="listbox" aria-label="${v.name} clips"></div>`;
  host3.append(el);
  let sel = v.clips[0].id;
  const vers = id => V3.map(([k, label]) => ({ label, src: `a/${v.tag}/${id}-${k}.mp3` }));
  const head = c => `<div class="words">${c.words}</div>`;
  const pl = Player(el.querySelector('.player'), vers(sel), head(v.clips[0]));
  const list = el.querySelector('.list');
  const render = () => { list.innerHTML = ''; for (const c of v.clips) { const b = document.createElement('button'); b.type = 'button'; b.className = 'row'; b.setAttribute('role', 'option'); b.setAttribute('aria-selected', c.id === sel);
    b.innerHTML = `<span>${c.words}</span><span class="d">${c.dur.toFixed(1)} s</span>`; b.onclick = () => { sel = c.id; pl.load(vers(c.id), head(c)); render(); pl.play(); }; list.append(b); } };
  render();
}
addEventListener('keydown', e => { if (e.target.closest('input,textarea')) return; const k = '1234'.indexOf(e.key); if (k >= 0 && lastPlayer) lastPlayer.pick(k); });
</script>
'''
open('index.html', 'w').write(html.replace('__DATA__', json.dumps(D, ensure_ascii=False)))
