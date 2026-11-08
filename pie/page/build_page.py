import json
pilot = json.load(open('pilot.json'))
html = r'''<title>Chimera Speaks PIE</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one listening column. Every comparison is a strip whose version switch keeps the playhead. */
:root {
  --paper: #f1f2ee; --ink: #1d2024; --muted: #5d636b; --rule: #d4d7d0; --well: #e6e8e2;
  --vtl: #3c5a7a; --chim: #b0301c; --sel: #dfe3da;
  --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #15171a; --ink: #e5e6e2; --muted: #9aa0a7; --rule: #2e3237; --well: #1f2226; --sel: #2a2e33;
  --vtl: #8fb0d3; --chim: #f07a62; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #15171a; --ink: #e5e6e2; --muted: #9aa0a7; --rule: #2e3237; --well: #1f2226; --sel: #2a2e33;
  --vtl: #8fb0d3; --chim: #f07a62; color-scheme: dark }
body { background: var(--paper); color: var(--ink); font: 17px/1.6 var(--body); }
.wrap { max-width: 820px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 44px 72px; display: grid; gap: 48px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.12; }
h1 { font-size: clamp(34px, 6vw, 54px); font-weight: 700; letter-spacing: -0.02em; }
h2 { font-size: 23px; font-weight: 700; }
h3 { font-size: 16px; font-weight: 700; }
p { margin: 0; max-width: 66ch; }
section { display: grid; gap: 16px; }
.lede { display: grid; gap: 14px; }
.lede p { color: var(--muted); font-size: 18px; }
.eyebrow { font: 500 12px var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.hint { font-size: 15px; color: var(--muted); }
.pie { font-style: italic; }
.tablebox { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 9px 12px 9px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font: 500 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
td.n { font-family: var(--mono); font-size: 15px; white-space: nowrap; }
td.best { color: var(--chim); font-weight: 600; }
.player { background: var(--well); border-radius: 10px; padding: 14px 16px; display: grid; gap: 10px; }
.player .head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 16px; align-items: baseline; }
.player .words { font-style: italic; font-size: 17px; }
.player .ipa { font-family: var(--mono); font-size: 13px; color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
button { font: 500 13.5px var(--mono); border-radius: 999px; border: 1.5px solid var(--rule); background: var(--paper); color: var(--ink); padding: 6px 13px; cursor: pointer; }
button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
button.play { min-width: 76px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1.5px solid var(--rule); border-radius: 18px; overflow: hidden; }
.seg button { border: 0; border-radius: 0; background: transparent; padding: 6px 11px; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--paper); }
.seg button.v[aria-pressed="true"] { background: var(--vtl); }
.seg button.c[aria-pressed="true"] { background: var(--chim); }
.seg button small { opacity: .75; font-size: 11.5px; }
.track { flex: 1 1 140px; min-width: 0; height: 8px; border-radius: 4px; background: var(--rule); position: relative; cursor: pointer; }
.track i { position: absolute; inset: 0 auto 0 0; border-radius: 4px; background: var(--ink); width: 0; }
.time { font: 12.5px var(--mono); color: var(--muted); min-width: 84px; text-align: right; }
.list { display: grid; gap: 2px; max-height: 420px; overflow-y: auto; border: 1px solid var(--rule); border-radius: 10px; }
.row { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; padding: 8px 12px; cursor: pointer; border: 0; background: transparent; text-align: left; border-radius: 0; font: inherit; color: inherit; }
.row:hover { background: var(--well); }
.row[aria-selected="true"] { background: var(--sel); }
.row .w { font-style: italic; min-width: 0; overflow-wrap: anywhere; }
.row .e { font: 12.5px var(--mono); color: var(--muted); white-space: nowrap; }
.row .e b { color: var(--chim); font-weight: 500; }
.row .meta { grid-column: 1 / -1; font: 12px var(--mono); color: var(--muted); }
.filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; }
.card { background: var(--well); border-radius: 10px; padding: 12px 14px; display: grid; gap: 6px; align-content: start; }
.card .ex { font-style: italic; font-size: 18px; }
.card p { font-size: 14.5px; color: var(--muted); }
.card .btns { display: flex; gap: 6px; }
ul { margin: 0; padding-left: 1.2em; display: grid; gap: 8px; max-width: 66ch; }
.foot { font-size: 14px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 16px; }
.journal { display: grid; gap: 12px; margin: 0; padding: 0; list-style: none; max-width: 66ch; }
.journal li { display: grid; gap: 2px; border-left: 2px solid var(--rule); padding-left: 12px; }
.journal time { font: 12px var(--mono); color: var(--muted); }
.journal li:first-child { border-left-color: var(--chim); }
.budget { font: 13px var(--mono); color: var(--muted); }
.lab-what td:first-child { white-space: nowrap; }
.lab-what td.w { font-size: 14px; color: var(--muted); min-width: 16ch; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header class="lede">
    <div class="eyebrow">wékʷos · voice lab · 7 Nov 2026</div>
    <h1>Chimera speaks PIE</h1>
    <p>We forked Chimera, an open physical model of the human airway, and taught its engine Proto-Indo-European: labiovelars, murmured stops, trills, the laryngeals and the pitch accent. Then we put its renders through the same speech-to-speech conversion as our corpus. Every comparison below uses the same words on the same timing; switch versions while it plays and you stay in place.</p>
  </header>

  <section id="lab">
    <div class="eyebrow">overnight lab notebook · newest first</div>
    <h2>Conversion tests</h2>
    <p class="hint">Each test renders the same 12 dev sentences (24 χ, 54 s), sends them through the voice changer with the corpus settings, and scores before and after. Lower is better. The voice changer has no seed, so the same render converted twice differs a little; compare "pilot 2" with "pilot 2 again" to see how much.</p>
    <p class="budget" id="lab-budget"></p>
    <div class="tablebox"><table class="lab-what" id="lab-table"></table></div>
    <div class="player" id="lab-player"></div>
    <span class="seg" id="lab-kind"><button aria-pressed="true" data-k="1">Converted</button><button aria-pressed="false" data-k="0">Raw</button></span>
    <div class="list" id="lab-list" role="listbox" aria-label="Test sentences"></div>
    <h2>Journal</h2>
    <ol class="journal" id="lab-journal"></ol>
  </section>


  <section>
    <h2>The dawn hymn</h2>
    <p class="hint">The whole hymn on its corpus timing, each voice through the same voice changer with the corpus settings. Switch while it plays to stay in place; the site's current voice (the Kokoro model trained on the VocalTractLab corpus) has its own timing, so it won't line up.</p>
    <div id="hymn" style="display:grid;gap:12px"></div>
    <div class="tablebox"><table>
      <thead><tr><th>Dawn hymn, sounds missed</th><th>VTL raw</th><th>VTL conv.</th><th>Chimera eve raw</th><th>eve conv.</th><th>Chimera night raw</th><th>night conv.</th><th>u fix raw</th><th>u fix conv.</th><th>Site voice (Kokoro)</th></tr></thead>
      <tbody>
        <tr><td>Male</td><td class="n">40%</td><td class="n">52%</td><td class="n">37%</td><td class="n">41%</td><td class="n">33%</td><td class="n best">39%</td><td class="n">33%</td><td class="n">42%</td><td class="n">54%</td></tr>
        <tr><td>Female</td><td class="n">57%</td><td class="n">64%</td><td class="n">43%</td><td class="n">46%</td><td class="n">43%</td><td class="n best">44%</td><td class="n">43%</td><td class="n">48%</td><td class="n">59%</td></tr>
      </tbody>
    </table></div>
    <p class="hint">Scored the same way for all five: the whole hymn transcribed by the phone recognizer and compared with its 189 target sounds (edit distance, with the same near-neighbour equivalences as below), so the site voice's own timing doesn't matter. The site voice was trained on the converted VocalTractLab corpus; a voice trained on converted Chimera would start from the Chimera-converted columns. "Eve" is the engine of the evening of 7 November, "night" the engine after the overnight work (χ, trill, nasals, labiovelars, dark l), "u fix" the same with u no longer nearly closed (listen to the ú of dʰúgh₂ter). The two night conversions differ mostly in sounds the u fix does not touch (e, w, t): that is the voice changer varying from one run to the next, which on one 26-second passage is several points. Converting the hymn cost 1,282 credits, the night version 646.</p>
  </section>

  <section>
    <h2>The numbers</h2>
    <p>Share of target sounds a universal phone recognizer misses, each phone checked inside its own time span (PhoneticXeus, the best current model on the 2026 benchmarks; lower is better). Near neighbours that don't change the PIE sound count as right: e as ɛ, a velar x for χ, kʷ heard as k + w, long or syllabic variants.</p>
    <div class="tablebox"><table>
      <thead><tr><th>The 24 converted sentences</th><th>Raw</th><th>After conversion</th></tr></thead>
      <tbody>
        <tr><td>VocalTractLab (the current corpus)</td><td class="n">55.3%</td><td class="n">56.5%</td></tr>
        <tr><td>Chimera at the first pilot</td><td class="n">35.3%</td><td class="n">38.9%</td></tr>
        <tr><td>Chimera, second pilot (evening)</td><td class="n">32.4%</td><td class="n">35.6%</td></tr>
        <tr><td>Chimera tonight, before the u fix</td><td class="n best">27.4%</td><td class="n best">33.8%</td></tr>
        <tr><td>Chimera final (the full-run engine)</td><td class="n">27.9%</td><td class="n">35.0%</td></tr>
      </tbody>
    </table></div>
    <p class="hint">On 162 held-out corpus sentences (not converted): VocalTractLab 54.3%, the converted corpus 53.2%, Chimera at the second pilot 30.2%, final 27.0%. Today's Chimera beats VTL after conversion on all 24 sentences and the pilot-time Chimera on 12. Naturalness of the converted audio (UTMOS) is the same for all three, about 3.2–3.3.</p>
    <p class="hint">What the recognizer cannot hear in any of them: the murmured stops (bʰ dʰ gʰ), the palatals (ḱ ǵ) and ʁʷ. Those are checked by acoustics instead: murmur is 5–6.5 dB breathier than the plain stop and fully voiced, ḱ has palatal transitions. One regression shows only after conversion: today's χ survives the voice changer worse than the pilot's (63% missed after conversion against 33%, though both are about 30% raw).</p>
  </section>

  <section>
    <h2>The conversion test</h2>
    <p class="hint">24 held-out corpus sentences, 12 per voice, each sent through the voice changer with the corpus settings: VocalTractLab's render, Chimera's at the first pilot, and Chimera's now (1,110 credits a batch). Pick a sentence, press play, then switch; keys 1 to 9 and 0 switch too. The figures on each line are the recognizer's misses after conversion: VTL → Chimera at the first pilot → the second → tonight (before the u fix) → final (the engine for the full run). Between the last two the raw renders are nearly identical sentence by sentence, yet the converted figures swing by several sounds either way: that is how much the voice changer itself varies.</p>
    <div class="player" id="pilot-player"></div>
    <div class="filters" role="group" aria-label="Voice">
      <span class="eyebrow">Voice</span>
      <span class="seg" id="voice-filter"><button aria-pressed="true" data-v="all">Both</button><button aria-pressed="false" data-v="male">Male</button><button aria-pressed="false" data-v="female">Female</button></span>
    </div>
    <div class="list" id="pilot-list" role="listbox" aria-label="Sentences"></div>
  </section>

  <section>
    <h2>Passages</h2>
    <p class="hint">The opening of the dawn hymn and our hard-test passage on their VocalTractLab timing: VocalTractLab, Chimera as first tried, and Chimera now (PIE pitch accent, voiced closures, revised vowels and glides, voicing on time after voiceless sounds). None of these are converted.</p>
    <div id="passages" style="display:grid;gap:12px"></div>
  </section>

  <section>
    <h2>Sound by sound</h2>
    <p class="hint">Chimera now, from handbook spellings. Each pair puts the PIE sound next to its plain counterpart.</p>
    <div class="grid" id="sounds"></div>
  </section>

  <section>
    <h2>What changed in the engine</h2>
    <ul>
      <li><b>A PIE module in the engine</b> reads handbook notation (<span class="pie">*h₂éwsōs</span>) and our IPA, and plans each feature as a movement of the tract. The browser app gained a PIE mode.</li>
      <li><b>Murmur</b> is a new voice-source channel: the folds keep vibrating with a leaky, slack glottis. Before, any opening wide enough to sound breathy stopped the voice.</li>
      <li><b>Uvular fricatives</b> were silent apart from breath: a bug dropped the frication source of any constriction just past the throat–mouth junction. Fixed, and h₂ has a new obstacle source where the jet meets the soft palate.</li>
      <li><b>r is trilled</b>: explicit tongue-tip contacts with releases between them. The original tap was missed 97% of the time, mostly heard as d; the trill is heard as r about three times in four, and 74% of r's make two or more real tongue contacts.</li>
      <li><b>Voiced stops keep their voicing</b> through the closure (the larynx lowers), so b d g are no longer heard as p t k.</li>
      <li><b>kʷ gʷ release through rounding</b>, the palatals ḱ ǵ have palatal bursts and transitions, /s/ spreads the lips, /w/ no longer closes off against the soft palate.</li>
      <li><b>The PIE pitch accent</b> (the model our VocalTractLab corpus used): a raised-cosine rise on each accented nucleus over a declining line; accent lift 3.4 semitones median.</li>
      <li><b>Timing at the glottis</b>: voicing returns 14–20 ms after p t k s (it took 30), and short vowels keep their middle open (a 55 ms schwa between s and t used to stay closed and voiceless).</li>
      <li><b>Breath colour as a control</b>: h's breath is darker than a stop's release, so it sits in the vowel's formants rather than sounding like s.</li>
    </ul>
  </section>

  <p class="foot">Fork: github.com/davidtorcivia/chimera-vocal-synth, branch pie (Chimera is MIT, by jamesondh). Voice changer: eleven_multilingual_sts_v2 with the corpus settings. Recognizer: wav2vec2-xlsr-53-espeak-cv-ft.</p>
</div>

<script>
const PILOT = __PILOT__;
const fmt = s => isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : '0:00';
const players = [];
function stopAll(except) { for (const p of players) if (p !== except) p.stop(); }

// a player over several versions of the same timeline; switching keeps the position
function Player(host, versions, head) {
  const P = { side: 0, playing: false };
  host.innerHTML = `<div class="head">${head || ''}</div>
    <div class="controls">
      <button class="play" type="button">Play</button>
      <span class="seg" role="group" aria-label="Version">${versions.map((v, i) => `<button type="button" class="${v.cls || ''}" aria-pressed="${i === 0}">${v.label}</button>`).join('')}</span>
      <div class="track" role="slider" tabindex="0" aria-label="Position"><i></i></div>
      <span class="time">0:00 / 0:00</span>
    </div>`;
  const playBtn = host.querySelector('.play'), bar = host.querySelector('.track i'), time = host.querySelector('.time'), track = host.querySelector('.track');
  const btns = [...host.querySelectorAll('.seg button')];
  let auds = [];
  const load = (vs, h) => {
    P.stop && P.stop();
    auds.forEach(a => { a.pause(); a.src = ''; });
    auds = vs.map(v => { const a = new Audio(); a.preload = 'metadata'; a.src = v.src; a.addEventListener('timeupdate', draw); a.addEventListener('loadedmetadata', draw); a.addEventListener('ended', () => { if (a === cur()) { P.stop(); setPos(0); draw(); } }); return a; });
    vs.forEach((v, i) => { btns[i].innerHTML = v.label; });
    if (h !== undefined) host.querySelector('.head').innerHTML = h;
    draw();
  };
  const cur = () => auds[P.side];
  const pos = () => cur() ? cur().currentTime : 0;
  const dur = () => Math.max(...auds.map(a => a.duration || 0));
  const setPos = t => auds.forEach(a => { try { a.currentTime = Math.max(0, Math.min(t, (a.duration || t) - 0.01)); } catch {} });
  function draw() { const d = dur(), t = pos(); bar.style.width = d ? `${Math.min(100, t / d * 100)}%` : '0'; time.textContent = `${fmt(t)} / ${fmt(d)}`; }
  P.stop = () => { auds.forEach(a => a.pause()); P.playing = false; playBtn.textContent = 'Play'; };
  P.play = () => { stopAll(P); setPos(pos()); cur().play().catch(() => {}); P.playing = true; playBtn.textContent = 'Pause'; };
  P.pick = i => {
    if (i === P.side || !auds[i]) return;
    const t = pos(), was = P.playing;
    cur().pause(); P.side = i; setPos(t);
    btns.forEach((b, k) => b.setAttribute('aria-pressed', k === i));
    if (was) cur().play().catch(() => {});
    draw();
  };
  playBtn.onclick = () => P.playing ? P.stop() : P.play();
  btns.forEach((b, i) => b.onclick = () => P.pick(i));
  track.onclick = e => { const r = track.getBoundingClientRect(); setPos((e.clientX - r.left) / r.width * dur()); draw(); };
  track.onkeydown = e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { setPos(pos() + (e.key === 'ArrowRight' ? 1 : -1)); draw(); e.preventDefault(); } };
  P.load = load;
  load(versions);
  players.push(P);
  return P;
}

// ---- the overnight lab
const LAB = __LAB__;
document.getElementById('lab-budget').textContent = `Credits used overnight: ${LAB.spent.toLocaleString()} of ${LAB.budget.toLocaleString()}`;
{
  const best = Math.min(...LAB.tests.filter(t => t.name !== 'vtl').map(t => t.conv));
  document.getElementById('lab-table').innerHTML = `<thead><tr><th>Test</th><th>What changed</th><th>Raw</th><th>Converted</th><th>χ conv.</th><th>s conv.</th><th>h conv.</th></tr></thead><tbody>` +
    LAB.tests.map(t => `<tr><td>${t.label}</td><td class="w">${t.what}</td><td class="n">${t.raw}%</td><td class="n${t.conv === best ? ' best' : ''}">${t.conv}%</td><td class="n">${t.chi}%</td><td class="n">${t.s}%</td><td class="n">${t.h}%</td></tr>`).join('') + '</tbody>';
  const jl = document.getElementById('lab-journal');
  for (const e of LAB.journal) { const li = document.createElement('li'); li.innerHTML = `<time>${e.t}</time><span></span>`; li.querySelector('span').textContent = e.text; jl.append(li); }
}
let labKind = 1, labSel = LAB.sents[0].id;
const labVersions = id => LAB.tests.map(t => ({ label: t.label, src: labKind ? `c/${t.name}/${id}-conv.mp3` : `c/${t.raw_of}/${id}-raw.mp3`, cls: t.name === 'vtl' ? 'v' : 'c' }));
const labHead = s => `<div><div class="words">${s.words}</div><div class="ipa">${s.ipa}</div></div><span class="eyebrow">${s.voice} · ${s.dur} s</span>`;
const labPlayer = Player(document.getElementById('lab-player'), labVersions(labSel), labHead(LAB.sents[0]));
const labList = document.getElementById('lab-list');
function renderLab() {
  labList.innerHTML = '';
  for (const s of LAB.sents) {
    const b = document.createElement('button');
    b.className = 'row'; b.type = 'button'; b.setAttribute('role', 'option'); b.setAttribute('aria-selected', s.id === labSel);
    const nums = LAB.tests.map(t => s.px[t.name] ? s.px[t.name][labKind] : '–');
    b.innerHTML = `<span class="w">${s.words}</span><span class="e">${nums.map((n, i) => i === nums.length - 1 ? `<b>${n}%</b>` : `${n}%`).join(' · ')}</span><span class="meta">${s.voice} · ${s.dur} s · misses ${labKind ? 'after conversion' : 'raw'}, in table order</span>`;
    b.onclick = () => { labSel = s.id; labPlayer.load(labVersions(s.id), labHead(s)); renderLab(); labPlayer.play(); };
    labList.append(b);
  }
}
renderLab();
document.querySelectorAll('#lab-kind button').forEach(b => b.onclick = () => {
  labKind = +b.dataset.k; document.querySelectorAll('#lab-kind button').forEach(x => x.setAttribute('aria-pressed', x === b));
  const s = LAB.sents.find(x => x.id === labSel); labPlayer.load(labVersions(labSel), labHead(s)); renderLab();
});

// ---- the dawn hymn
const hy = document.getElementById('hymn');
for (const [v, title] of [['m', 'Male'], ['f', 'Female']]) {
  const el = document.createElement('div'); el.className = 'player'; hy.append(el);
  Player(el, [
    { label: 'VTL', src: `h/${v}-vtl.mp3`, cls: 'v' },
    { label: 'VTL conv.', src: `h/${v}-vtlc.mp3`, cls: 'v' },
    { label: 'Chimera eve', src: `h/${v}-chim.mp3`, cls: 'c' },
    { label: 'eve conv.', src: `h/${v}-chimc.mp3`, cls: 'c' },
    { label: 'Chimera night', src: `h/${v}-chim3.mp3`, cls: 'c' },
    { label: 'night conv.', src: `h/${v}-chim3c.mp3`, cls: 'c' },
    { label: 'u fix', src: `h/${v}-chim4.mp3`, cls: 'c' },
    { label: 'u fix conv.', src: `h/${v}-chim4c.mp3`, cls: 'c' },
    { label: 'site voice', src: `h/${v}-site.mp3` },
  ], `<div><h3>${title}</h3><div class="ipa">χáwsos léwka déjwi diwés dʱúgəter ‖ n̩sméj léwks sciɣʷwóskʷe …</div></div>`);
}

// ---- the conversion test
const pv = id => [
  { label: 'VTL', src: `p/${id}-vtl.mp3`, cls: 'v' },
  { label: 'VTL conv.', src: `p/${id}-vtlc.mp3`, cls: 'v' },
  { label: 'Chimera pilot', src: `p/${id}-chim.mp3`, cls: 'c' },
  { label: 'pilot conv.', src: `p/${id}-chimc.mp3`, cls: 'c' },
  { label: 'Chimera now', src: `p/${id}-chim2.mp3`, cls: 'c' },
  { label: 'now conv.', src: `p/${id}-chim2c.mp3`, cls: 'c' },
  { label: 'tonight', src: `p/${id}-chim3.mp3`, cls: 'c' },
  { label: 'tonight conv.', src: `p/${id}-chim3c.mp3`, cls: 'c' },
  { label: 'final', src: `p/${id}-chim4.mp3`, cls: 'c' },
  { label: 'final conv.', src: `p/${id}-chim4c.mp3`, cls: 'c' },
];
const headFor = s => `<div><div class="words">${s.words}</div><div class="ipa">${s.ipa}</div></div><span class="eyebrow">${s.voice} · ${s.src}</span>`;
const pilotPlayer = Player(document.getElementById('pilot-player'), pv(PILOT[0].id), headFor(PILOT[0]));
const list = document.getElementById('pilot-list');
let voice = 'all', selected = PILOT[0].id;
function renderList() {
  list.innerHTML = '';
  for (const s of PILOT.filter(s => voice === 'all' || s.voice === voice)) {
    const b = document.createElement('button');
    b.className = 'row'; b.type = 'button'; b.setAttribute('role', 'option'); b.setAttribute('aria-selected', s.id === selected);
    b.innerHTML = `<span class="w">${s.words}</span><span class="e">${s.px[1]}% → ${s.px[3]}% → ${s.px[5]}% → ${s.px[7]}% → <b>${s.px[9]}%</b></span><span class="meta">${s.voice} · ${s.dur.toFixed(1)} s · ${s.src}</span>`;
    b.onclick = () => { selected = s.id; pilotPlayer.load(pv(s.id), headFor(s)); renderList(); pilotPlayer.play(); };
    list.append(b);
  }
}
renderList();
document.querySelectorAll('#voice-filter button').forEach(b => b.onclick = () => {
  voice = b.dataset.v; document.querySelectorAll('#voice-filter button').forEach(x => x.setAttribute('aria-pressed', x === b)); renderList();
});
addEventListener('keydown', e => {
  if (e.target.closest('input,textarea')) return;
  const k = '1234567890'.indexOf(e.key); if (k >= 0) pilotPlayer.pick(k);
});

// ---- passages
const PASS = [['hymn-m', 'Dawn hymn, male', 'χáwsos léwka déjwi diwés dʱúgəter …'], ['hymn-f', 'Dawn hymn, female', 'χáwsos léwka déjwi diwés dʱúgəter …'],
  ['hard-m', 'Hard test, male', 'pətéːr bʱráχteːr dʱugətéːr ɟénətor …'], ['hard-f', 'Hard test, female', 'pətéːr bʱráχteːr dʱugətéːr ɟénətor …']];
const ph = document.getElementById('passages');
for (const [id, title, ipa] of PASS) {
  const el = document.createElement('div'); el.className = 'player'; ph.append(el);
  Player(el, [{ label: 'VTL', src: `s/${id}-vtl.mp3`, cls: 'v' }, { label: 'Chimera, first try', src: `s/${id}-first.mp3`, cls: 'c' }, { label: 'Chimera now', src: `s/${id}-now.mp3`, cls: 'c' }],
    `<div><h3>${title}</h3><div class="ipa">${ipa}</div></div>`);
}

// ---- sound by sound
const SOUNDS = [
  ['kw', 'ákʷa · áka', 'Labiovelar kʷ: the lips round and the vowel starts through a [w].'],
  ['gw', 'ágʷa · ága', 'Labiovelar gʷ, voiced through the closure.'],
  ['gwh', 'ágʷʰa · ága', 'Murmured labiovelar gʷʰ: rounding and breathy release together.'],
  ['bh', 'ábʰa · ába', 'Murmured bʰ: the vowel starts breathy, still voiced.'],
  ['dh', 'ádʰa · áda', 'Murmured dʰ.'],
  ['gh', 'ágʰa · ága', 'Murmured gʰ.'],
  ['kj', 'áḱa · áka', 'Palatal ḱ against velar k.'],
  ['kjo', 'óḱo · óko', 'Clearest beside a back vowel: F2 climbs into ḱ.'],
  ['gjh', 'áǵʰa · ága', 'Murmured palatal ǵʰ.'],
  ['h1', 'áh₁a', 'h₁ as a glottal [h].'],
  ['h2', 'áh₂a', 'h₂ as uvular [χ], with frication of its own.'],
  ['h3', 'áh₃a', 'h₃ as rounded voiced uvular [ʁʷ].'],
  ['r', 'ára · árta', 'r as a short alveolar trill.'],
  ['rs', 'ár̥ta · ál̥ta', 'Syllabic r̥ (a trilled vocoid) and l̥.'],
  ['ms', 'ám̥ta · án̥ta', 'Syllabic m̥ and n̥.'],
  ['w1', '*kʷékʷlos', '"Wheel".'],
  ['w2', '*ǵʰéyōm', '"Winter".'],
  ['w3', '*dʰugh₂tḗr', '"Daughter".'],
  ['w4', '*gʷʰénti', '"Strikes".'],
  ['w5', '*h₂ŕ̥tḱos', '"Bear".'],
  ['w6', '*wĺ̥kʷos', '"Wolf".'],
  ['w7', '*h₃ékʷs', '"Eye".'],
  ['w8', '*h₂éwsōs', '"Dawn".'],
];
const sg = document.getElementById('sounds');
let one = null;
for (const [id, ex, note] of SOUNDS) {
  const c = document.createElement('div'); c.className = 'card';
  c.innerHTML = `<div class="ex">${ex}</div><p>${note}</p><div class="btns"><button type="button" data-b="male">Male</button><button type="button" data-b="female">Female</button></div>`;
  c.querySelectorAll('button').forEach(b => b.onclick = () => { stopAll(null); if (one) one.pause(); one = new Audio(`x/${id}-${b.dataset.b}.mp3`); one.play().catch(() => {}); });
  sg.append(c);
}
</script>
'''
html = html.replace('__LAB__', open('lab-data.json').read())
html = html.replace('__PILOT__', json.dumps(pilot, ensure_ascii=False))
open('page/index.html', 'w').write(html)
print(len(html))
