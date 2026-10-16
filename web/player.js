
(function () {
const REF = window.LOL_REFERENCE, P = window.LOL_PLAYER;
if (!REF || !P) { document.getElementById('pname').textContent = 'reference.js not found'; return; }

const METS = REF.metrics, MK = METS.map(m => m.key);
const DEF_W = {}; METS.forEach(m => DEF_W[m.key] = m.sign);
const ROLE_COLORS = { TOP:'#4f9cf9', JUNGLE:'#e05656', MIDDLE:'#3fb96f', BOTTOM:'#f0a13a', SUPPORT:'#b07cf0' };
const ROLE_ORDER = REF.roles;
const state = { baseline: P.bandAvailable ? 'band' : 'overall', role: 'ALL', w: Object.assign({}, DEF_W) };
for (const k in REF.groups) REF.groups[k].__key = k;

// ---------- reference math ----------
function groupFor(role) {
  if (state.baseline === 'band' && P.band) {
    const g = REF.groups[role + '|' + P.band];
    if (g) return g;
  }
  return REF.groups[role];
}
function pct(q, v) {   // value -> percentile, interpolating the quantile grid
  const n = q.length, step = 100 / (n - 1);
  // Ties get mid-rank, exactly like the Mann-Whitney AUC in build_web.py. Without
  // this, a metric that is 0 in 58% of peer games (solo kills) reports every tied
  // game as 0% and drags the averaged percentile down to nonsense.
  if (v <= q[0]) { let h = 0; while (h + 1 < n && q[h + 1] === q[0]) h++; return h ? h * step / 2 : 0; }
  if (v >= q[n - 1]) { let l = n - 1; while (l - 1 >= 0 && q[l - 1] === q[n - 1]) l--; return 100 - (n - 1 - l) * step / 2; }
  let i = 1; while (i < n && v > q[i]) i++;
  const a = q[i - 1], b = q[i];
  return (i - 1) * step + (b > a ? step * (v - a) / (b - a) : 0);
}
function z(v, grp, k) { const s = grp.m[k]; return (v - s.mean) / s.sd; }
function compOf(vals, grp) {
  let s = 0, d = 0;
  for (let i = 0; i < MK.length; i++) {
    const w = state.w[MK[i]]; if (!w) continue;
    s += w * z(vals[i], grp, MK[i]); d += Math.abs(w);
  }
  return d ? s / d : 0;
}
function isDefaultW() { for (let i = 0; i < MK.length; i++) if (state.w[MK[i]] !== DEF_W[MK[i]]) return false; return true; }

// peer percentile of a custom-weight composite, from the sampled baseline
const scache = new Map();
function sampleComps(grp) {
  const sig = grp.__key + '|' + MK.map(k => state.w[k]).join(',');
  let a = scache.get(sig);
  if (!a) {
    a = [];
    for (const row of grp.z) {
      let s = 0, d = 0;
      for (let i = 0; i < MK.length; i++) {
        const w = state.w[MK[i]]; if (!w) continue;
        s += w * (row[i] / 100 - grp.m[MK[i]].mean) / grp.m[MK[i]].sd; d += Math.abs(w);
      }
      a.push(d ? s / d : 0);
    }
    a.sort((x, y) => x - y); scache.set(sig, a);
  }
  return a;
}
function sampPct(grp, v) {
  const a = sampleComps(grp);
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
  return a.length ? lo / a.length * 100 : 50;
}
function compPct(grp, v) { return isDefaultW() ? pct(grp.comp.q, v) : sampPct(grp, v); }

function rows() {
  let gs = P.games;
  if (state.role !== 'ALL') gs = gs.filter(g => g.role === state.role);
  return gs.map(g => {
    const grp = groupFor(g.role);
    const comp = compOf(g.raw, grp), mate = compOf(g.mate, grp);
    return { g, grp, comp, out: comp - mate, cp: compPct(grp, comp), op: pct(grp.outperform.q, comp - mate) };
  });
}
function avg(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }
function majorityRole(rs) {
  const c = {}; rs.forEach(r => c[r.g.role] = (c[r.g.role] || 0) + 1);
  return Object.keys(c).sort((a, b) => c[b] - c[a])[0];
}
const nf = (v, d) => (v === null || v === undefined || isNaN(v)) ? '-' : v.toFixed(d === undefined ? 1 : d);
const pc = v => Math.round(v) + '%';

// ---------- svg helpers ----------
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text; return e;
}
function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }

// ---------- header + tiles ----------
function header() {
  const pr = P.profile, rank = pr.soloRank || {}, cdn = P.meta.ddragon;
  document.getElementById('pname').textContent = pr.riotId;
  const bits = [];
  if (rank.tier) bits.push(rank.tier + ' ' + (rank.division || '') +
    (rank.leaguePoints != null ? ' \u00b7 ' + rank.leaguePoints + ' LP' : ''));
  if (rank.wins != null) bits.push(rank.wins + '-' + rank.losses + ' W-L');
  bits.push('level ' + (pr.summonerLevel ?? '?'));
  bits.push((pr.region || '').toUpperCase());
  document.getElementById('pmeta').innerHTML = bits.join(' &nbsp;\u00b7&nbsp; ') +
    '<br><span style="font-size:12px">Last ' + P.nGames + ' ranked solo games' +
    (P.meta.trimmedOlder ? ' (of ' + (P.nGames + P.meta.trimmedOlder) + ' stored)' : '') +
    ' \u00b7 collected ' +
    (pr.collectedAtIso || '').slice(0, 10) + ' \u00b7 <a href="index.html">all reports</a> \u00b7 ' +
    '<a href="../index.html">home / explorer</a></span>';
  if (pr.profileIconId != null)
    document.getElementById('avatar').innerHTML =
      '<img src="https://ddragon.leagueoflegends.com/cdn/' + cdn + '/img/profileicon/' + pr.profileIconId + '.png" alt="">';
}

function tiles(rs) {
  const wins = rs.filter(r => r.g.win).length;
  const top20 = rs.filter(r => r.cp >= 80).length;
  const items = [
    [rs.length, 'games in view'],
    [rs.length ? (wins / rs.length * 100).toFixed(1) + '%' : '-', 'win rate'],
    [rs.length ? pc(avg(rs.map(r => r.cp))) : '-', 'score percentile vs peers'],
    [rs.length ? (top20 / rs.length * 100).toFixed(0) + '%' : '-', 'games in peers\u2019 top 20%'],
    [rs.length ? pc(avg(rs.map(r => r.op))) : '-', 'percentile vs own teammates'],
  ];
  document.getElementById('tiles').innerHTML = items.map(([big, lbl]) =>
    '<div class="tile"><div class="big">' + big + '</div><div class="lbl">' + lbl + '</div></div>').join('');
}

// ---------- baseline / role controls ----------
function baselineBtns() {
  const box = document.getElementById('baselineBtns');
  const opts = [['overall', 'All ranks (whole corpus)']];
  if (P.bandAvailable) opts.push(['band', 'My band: ' + P.band]);
  box.innerHTML = '';
  opts.forEach(([v, label]) => {
    const b = document.createElement('button');
    b.className = 'btn' + (state.baseline === v ? ' on' : '');
    b.textContent = label;
    b.onclick = () => { state.baseline = v; render(); };
    box.appendChild(b);
  });
  const rolesInView = state.role === 'ALL' ? ROLE_ORDER.filter(r => P.roles[r]) : [state.role];
  const parts = rolesInView.map(r => {
    const g = groupFor(r);
    return r + ' n=' + g.n.toLocaleString() + (g.__key.includes('|') ? '' : ' (all ranks)');
  });
  document.getElementById('baselineNote').innerHTML =
    'Baseline groups used: ' + parts.join(' \u00b7 ') +
    (state.baseline === 'band' && !P.bandAvailable ? ' <b class="lose">(band baseline unavailable \u2014 using all ranks)</b>' : '') +
    '. Your band is taken from your <i>current</i> league tier and applied to every game here, ' +
    'so games played at a lower tier are scored a bit generously.';
}

function roleBtns() {
  const box = document.getElementById('roleBtns');
  box.innerHTML = '';
  ['ALL'].concat(ROLE_ORDER.filter(r => P.roles[r])).forEach(r => {
    const b = document.createElement('button');
    b.className = 'btn' + (state.role === r ? ' on' : '');
    b.textContent = r === 'ALL' ? 'All roles (' + P.nGames + ')' :
      r + ' ' + P.roles[r].games + 'g ' + Math.round(P.roles[r].wins / P.roles[r].games * 100) + '%W';
    if (r !== 'ALL') b.style.borderColor = ROLE_COLORS[r] + '66';
    b.onclick = () => { state.role = r; render(); };
    box.appendChild(b);
  });
}

// ---------- metric percentile bars ----------
function metricBars(rs) {
  const box = document.getElementById('metricBars');
  const role = majorityRole(rs) || 'JUNGLE';
  const grp = groupFor(role);
  let html = '';
  METS.forEach((m, i) => {
    const ps = rs.map(r => {
      const g = groupFor(r.g.role);
      const p = pct(g.m[m.key].q, r.g.raw[i]);
      return m.sign < 0 ? 100 - p : p;
    });
    if (!ps.length) return;
    const mean = avg(ps);
    // percentile of your *average value* per role, games-weighted: differs from the
    // bar whenever your per-game percentiles are lopsided (one stomp vs nine close games)
    const byRole = {};
    rs.forEach(r => (byRole[r.g.role] = byRole[r.g.role] || []).push(r.g.raw[i]));
    let num = 0, den = 0;
    for (const rl in byRole) {
      const g = groupFor(rl), mv = avg(byRole[rl]);
      let p = pct(g.m[m.key].q, mv); if (m.sign < 0) p = 100 - p;
      num += p * byRole[rl].length; den += byRole[rl].length;
    }
    const aggP = den ? num / den : 50;
    const mine = avg(rs.map(r => r.g.raw[i]));
    // flag metrics where a big block of peers is tied at one value (solo kills,
    // wards killed): the percentile then says less than the tie share does
    const qq = grp.m[m.key].q; let tie = 0;
    while (tie + 1 < qq.length && qq[tie + 1] === qq[0]) tie++;
    const tiePct = tie * (100 / (qq.length - 1));
    html += '<div class="mrow">' +
      '<div>' + m.label + (m.sign < 0 ? ' <span class="note">\u2193</span>' : '') + '</div>' +
      '<div class="pbar" title="average of per-game percentiles ' + pc(mean) +
        '; percentile of your average value ' + pc(aggP) + '">' +
        '<span style="width:' + mean.toFixed(1) + '%"></span>' +
        '<i style="left:' + aggP.toFixed(1) + '%"></i><em>' + pc(mean) + '</em></div>' +
      '<div class="vals">you ' + nf(mine, 2) + ' \u00b7 peers ' + nf(qq[10], 2) +
      (tiePct >= 10 ? ' <span class="note">(' + Math.round(tiePct) + '% tied at ' + nf(qq[0], 2) + ')</span>' : '') +
      '</div></div>';
  });
  box.innerHTML = html || '<div class="note">no games in this filter</div>';
}

// ---------- score per game ----------
function drawTrend(rs) {
  const svg = document.getElementById('trend'); clear(svg);
  const W = svg.clientWidth || 860, H = 240, pad = { l: 34, r: 10, t: 10, b: 22 };
  const n = rs.length; if (!n) return;
  const ys = v => pad.t + (H - pad.t - pad.b) * (1 - v / 100);
  [0, 25, 50, 75, 100].forEach(v => {
    svg.appendChild(el('line', { x1: pad.l, x2: W - pad.r, y1: ys(v), y2: ys(v),
      stroke: v === 50 ? '#3a4252' : '#2a3040', 'stroke-dasharray': v === 50 ? '' : '3 3' }));
    svg.appendChild(el('text', { x: pad.l - 5, y: ys(v) + 3, 'text-anchor': 'end', 'font-size': 10, fill: '#8b93a7' }, v));
  });
  const bw = (W - pad.l - pad.r) / n;
  rs.forEach((r, i) => {
    const x = pad.l + i * bw, top = ys(r.cp), base = ys(0);
    svg.appendChild(el('rect', { x: x + bw * 0.18, y: top, width: Math.max(1.5, bw * 0.64),
      height: Math.max(1, base - top), fill: r.g.win ? '#3fb96f' : '#e05656', opacity: .75 }));
  });
  const win = 5, pts = [];
  for (let i = 0; i < n; i++) {
    const s = rs.slice(Math.max(0, i - win + 1), i + 1);
    pts.push([pad.l + i * bw + bw / 2, ys(avg(s.map(x => x.cp)))]);
  }
  svg.appendChild(el('polyline', { points: pts.map(p => p.join(',')).join(' '),
    fill: 'none', stroke: '#4f9cf9', 'stroke-width': 2 }));
  svg.appendChild(el('text', { x: pad.l, y: H - 5, 'font-size': 10, fill: '#8b93a7' }, 'oldest'));
  svg.appendChild(el('text', { x: W - pad.r, y: H - 5, 'font-size': 10, fill: '#8b93a7', 'text-anchor': 'end' }, 'newest'));
}

function drawHist(id, vals, grp, grid) {
  const svg = document.getElementById(id); clear(svg);
  const W = svg.clientWidth || 860, H = +svg.getAttribute('height') || 240, pad = { l: 26, r: 12, t: 10, b: 22 };
  if (!vals.length) return;
  const q = grid.q, lo = Math.min(q[1], Math.min.apply(null, vals)), hi = Math.max(q[19], Math.max.apply(null, vals));
  const xs = v => pad.l + (W - pad.l - pad.r) * (v - lo) / (hi - lo + 1e-9);
  const NB = 16, counts = new Array(NB).fill(0);
  vals.forEach(v => counts[Math.min(NB - 1, Math.floor((v - lo) / (hi - lo + 1e-9) * NB))]++);
  const mx = Math.max.apply(null, counts) || 1;
  svg.appendChild(el('rect', { x: xs(q[5]), y: pad.t, width: Math.max(2, xs(q[15]) - xs(q[5])),
    height: H - pad.t - pad.b, fill: '#4f9cf9', opacity: .10 }));
  svg.appendChild(el('line', { x1: xs(q[10]), x2: xs(q[10]), y1: pad.t, y2: H - pad.b, stroke: '#f0a13a', 'stroke-width': 2 }));
  counts.forEach((c, b) => {
    const h = c / mx * (H - pad.t - pad.b), x = pad.l + (W - pad.l - pad.r) * b / NB;
    if (c) svg.appendChild(el('rect', { x: x + 1, y: H - pad.b - h, width: (W - pad.l - pad.r) / NB - 2,
      height: h, fill: '#4f9cf9', opacity: .8 }));
  });
  svg.appendChild(el('text', { x: xs(q[10]) + 4, y: pad.t + 10, 'font-size': 10, fill: '#f0a13a' }, 'peer median'));
  [lo, (lo + hi) / 2, hi].forEach(v => svg.appendChild(el('text',
    { x: xs(v), y: H - 6, 'text-anchor': 'middle', 'font-size': 10, fill: '#8b93a7' }, nf(v, 2))));
}

function drawOut(rs) {
  drawHist('outChart', rs.map(r => r.out), groupFor(majorityRole(rs)), groupFor(majorityRole(rs)).outperform);
  const better = rs.filter(r => r.out > 0).length;
  document.getElementById('outNote').innerHTML =
    'You outscored your own teammates in <b>' + better + ' of ' + rs.length + '</b> games \u00b7 average differential ' +
    (avg(rs.map(r => r.out)) >= 0 ? '<b class="win">' : '<b class="lose">') + nf(avg(rs.map(r => r.out)), 2) + '</b> SD \u00b7 ' +
    'median peer differential is 0 by construction.';
}

// ---------- tables ----------
function champTable(rs) {
  const by = {};
  rs.forEach(r => {
    const c = by[r.g.champ] || (by[r.g.champ] = { n: 0, w: 0, cp: 0, role: r.g.role, key: r.g.champKey });
    c.n++; c.w += r.g.win ? 1 : 0; c.cp += r.cp;
  });
  const list = Object.entries(by).sort((a, b) => b[1].n - a[1].n).slice(0, 14);
  const cdn = P.meta.ddragon;
  let html = '<table><tr><th>champion</th><th></th><th>role</th><th>games</th><th>win</th><th>avg score pct</th></tr>';
  list.forEach(([name, c]) => {
    const img = c.key ? '<img class="champ" src="https://ddragon.leagueoflegends.com/cdn/' + cdn +
      '/img/champion/' + c.key + '.png" alt="">' : '';
    html += '<tr><td>' + img + '</td><td style="text-align:left">' + name + '</td><td>' + c.role + '</td><td>' + c.n +
      '</td><td>' + Math.round(c.w / c.n * 100) + '%</td><td>' + pc(c.cp / c.n) + '</td></tr>';
  });
  document.getElementById('champTable').innerHTML =
    (list.length ? html + '</table>' : '<div class="note">no games in this filter</div>');
}

function gameTable(rs) {
  const idx = k => MK.indexOf(k);
  const cols = ['date', 'champion', 'role', 'result', 'KDA', 'CS/min', 'dmg/min', 'vision/min', 'gold/min', 'score pct', 'vs team'];
  let html = '<table><tr>' + cols.map(c => '<th>' + c + '</th>').join('') + '</tr>';
  rs.slice().reverse().slice(0, 20).forEach(r => {
    const g = r.g, i = idx;
    html += '<tr><td>' + g.date + '</td><td>' + g.champ + '</td><td style="color:' + ROLE_COLORS[g.role] + '">' + g.role +
      '</td><td class="' + (g.win ? 'win">W' : 'lose">L') + '</td><td>' + g.k + '/' + g.d + '/' + g.a + '</td>' +
      '<td>' + nf(g.raw[i('csm')], 1) + '</td><td>' + Math.round(g.raw[i('dpm')]).toLocaleString() + '</td>' +
      '<td>' + nf(g.raw[i('vspm')], 1) + '</td><td>' + Math.round(g.raw[i('gpm')]).toLocaleString() + '</td>' +
      '<td>' + pc(r.cp) + '</td><td>' + (r.out >= 0 ? '<span class="win">+' : '<span class="lose">') + nf(r.out, 2) + '</span></td></tr>';
  });
  document.getElementById('gameTable').innerHTML = html + '</table>';
}

// ---------- weights ----------
function weightsBox() {
  const box = document.getElementById('weights');
  if (box.dataset.built) { syncSliders(); return; }
  box.dataset.built = '1';
  box.innerHTML = '';
  METS.forEach(m => {
    const row = document.createElement('div'); row.className = 'wrow';
    row.innerHTML = '<span class="name">' + m.label + '</span>' +
      '<input type="range" min="-3" max="3" step="0.1" value="' + DEF_W[m.key] + '">' +
      '<span class="val">' + DEF_W[m.key].toFixed(1) + '</span>';
    const inp = row.querySelector('input'), val = row.querySelector('.val');
    inp.dataset.key = m.key;
    inp.oninput = () => {
      state.w[m.key] = +inp.value; val.textContent = (+inp.value).toFixed(1);
      scache.clear(); render();
    };
    box.appendChild(row);
  });
  document.getElementById('resetW').onclick = () => {
    METS.forEach(m => state.w[m.key] = m.sign); scache.clear(); syncSliders(); render();
  };
}
function syncSliders() {
  document.querySelectorAll('#weights input').forEach(inp => {
    inp.value = state.w[inp.dataset.key];
    inp.parentElement.querySelector('.val').textContent = (+state.w[inp.dataset.key]).toFixed(1);
  });
}

function methodText() {
  const c = P.corpus, pr = P.profile;
  document.getElementById('method').innerHTML =
    '<b>Baseline.</b> Every metric is z-scored and percentiled against ' + c.nRows.toLocaleString() +
    ' player-games from ' + c.nMatches.toLocaleString() + ' ' + c.region +
    ' ranked solo queue games (collected via the Riot API, baseline frozen ' + c.generated.slice(0, 10) + '). ' +
    'Two baselines are offered: all ranks for your role, and your rank band \u00d7 role. ' +
    'Values are trimmed to the corpus 0.1%/99.9% range before scoring.<br>' +
    '<b>Score.</b> Equal-weight mean of the 14 signed z-scores (deaths and death share count negatively). ' +
    'Percentiles are computed per game inside the role played in that game, then averaged \u2014 averaging raw stats first ' +
    'would let one 25-0 stomp dominate the result. Ties are given mid-rank (the same Mann-Whitney convention the ' +
    'AUC code uses), so a metric that is zero in most peer games does not report every tied game as 0%.<br>' +
    '<b>Teammate differential.</b> Own score minus the mean of your teammates\u2019 scores (same baseline), so team strength ' +
    'cancels out; this is the within-team signal the project\u2019s Model C uses.<br>' +
    '<b>Caveats worth knowing.</b> (1) End-of-game box scores partly <i>re-encode</i> the win \u2014 stomps inflate everything, ' +
    'so read the differential panel and the distribution, not just the win-rate tile. ' +
    '(2) ' + P.nGames + ' games is a small sample: a \u00b15 percentile change is noise. ' +
    '(3) Your band is your <i>current</i> tier (' + (pr.soloRank ? pr.soloRank.tier + ' ' + pr.soloRank.division : 'unranked') +
    ') applied to all games shown. ' + '(4) The corpus is ' + c.region + ' solo queue, so cross-region comparisons inherit ' +
    c.region + '-specific playstyle. ' + '(5) Percentiles describe association with winning in the corpus, not causal carry value.' +
    '<br><b>Refresh.</b> <code>python collect_player.py "' + pr.riotId + '" --region ' + (pr.region || 'na1') +
    '</code> then <code>python build_player_web.py --slug ' + P.slug + '</code>';
}

// ---------- render ----------
let tm = null;
function render() {
  const rs = rows();
  baselineBtns(); roleBtns(); weightsBox();
  tiles(rs);
  if (!rs.length) {
    ['trend', 'hist', 'outChart'].forEach(id => clear(document.getElementById(id)));
    document.getElementById('metricBars').innerHTML = '<div class="note">no games in this filter</div>';
    document.getElementById('champTable').innerHTML = document.getElementById('gameTable').innerHTML = '';
    return;
  }
  metricBars(rs);
  drawTrend(rs);
  const grp = groupFor(majorityRole(rs));
  drawHist('hist', rs.map(r => r.comp), grp, grp.comp);
  drawOut(rs);
  champTable(rs);
  gameTable(rs);
  methodText();
}
header();
render();
window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(render, 150); });
})();
