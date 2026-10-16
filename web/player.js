
(function () {
const REF = window.LOL_REFERENCE, P = window.LOL_PLAYER;
if (!REF || !P) { document.getElementById('pname').textContent = 'reference.js not found'; return; }

const METS = REF.metrics, MK = METS.map(m => m.key);
const DEF_W = {}; METS.forEach(m => DEF_W[m.key] = m.sign);
const ROLE_COLORS = { TOP:'#4f9cf9', JUNGLE:'#e05656', MIDDLE:'#3fb96f', BOTTOM:'#f0a13a', SUPPORT:'#b07cf0' };
const ROLE_ORDER = REF.roles;

// ---------- reference math, axes and per-role weights ----------
const SIGNS = {}; METS.forEach(m => SIGNS[m.key] = m.sign);
const AX = Object.fromEntries(REF.axes.map(a => [a.key, a]));
const AXLBL = Object.fromEntries(REF.axes.map(a => [a.key, a.label]));
const MI = {}; MK.forEach((k, i) => MI[k] = i);
const HAS_PERSONAL = !!(P.personal && (P.personal.band || P.personal.overall));
const state = { baseline: P.bandAvailable ? 'band' : 'overall', role: 'ALL',
                axis: 'C', w: Object.assign({}, DEF_W), custom: false };
for (const k in REF.groups) REF.groups[k].__key = k;

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
// The role's own fitted coefficients (per role!), or +-1 signs for equal weight.
function popW(axKey, role) {
  const o = {};
  if (axKey === 'equal' || !REF.presets || !REF.presets[axKey] || !REF.presets[axKey][role]) {
    MK.forEach(k => o[k] = SIGNS[k]);
    return o;
  }
  const w = REF.presets[axKey][role].w;
  MK.forEach(k => o[k] = w[k] || 0);
  return o;
}
function wNow(role) { return state.custom ? state.w : popW(state.axis, role); }
function useResidual() { return AX[state.axis].kind === 'residual'; }

// z against the peer group, and r = own minus the mean of your four teammates
function prep(g) {
  const grp = groupFor(g.role), w = wNow(g.role);
  const zv = new Array(MK.length), rv = new Array(MK.length);
  for (let i = 0; i < MK.length; i++) {
    const s = grp.m[MK[i]];
    zv[i] = (g.raw[i] - s.mean) / s.sd;
    rv[i] = zv[i] - (g.mate[i] - s.mean) / s.sd;
  }
  return { g, grp, w, z: zv, r: rv };
}
function dot(vecs, w, feats) {
  let s = 0, d = 0;
  for (const f of feats) {
    const ww = w[f] || 0; if (!ww) continue;
    s += ww * vecs[MI[f]]; d += Math.abs(ww);
  }
  return d ? s / d : 0;
}
function scoreOf(p) {
  return dot(useResidual() ? p.r : p.z, p.w, AX[state.axis].feats);
}
function diffOf(p) {                 // same weights, always on the residual axis
  return dot(p.r, p.w, MK);
}
// Percentiles come from exact quantile grids for the four published weightings.
// Custom sliders fall back to sampled peer rows (grp.z / grp.rz hold peer z-scores,
// already standardized, so they are used as-is).
const scache = new Map();
function sampPct(grp, v, key) {
  const sig = key + '|' + grp.__key + '|' + MK.map(k => state.w[k]).join(',');
  let a = scache.get(sig);
  if (!a) {
    a = (grp[key] || []).map(row => {
      let s = 0, d = 0;
      for (let i = 0; i < MK.length; i++) {
        const w = state.w[MK[i]]; if (!w) continue;
        s += w * row[i] / 100; d += Math.abs(w);
      }
      return d ? s / d : 0;
    });
    a.sort((x, y) => x - y); scache.set(sig, a);
  }
  if (!a.length) return 50;
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
  return lo / a.length * 100;
}
function scorePct(p, v) {
  return state.custom ? sampPct(p.grp, v, useResidual() ? 'rz' : 'z')
                      : pct(p.grp.score[state.axis].q, v);
}
function diffPct(p, v) {
  return state.custom ? sampPct(p.grp, v, 'rz') : pct(p.grp.diff[state.axis].q, v);
}

function rows() {
  let gs = P.games;
  if (state.role !== 'ALL') gs = gs.filter(g => g.role === state.role);
  return gs.map(g => {
    const p = prep(g);
    const sc = scoreOf(p), df = diffOf(p);
    return { g, grp: p.grp, w: p.w, sc, df, sp: scorePct(p, sc), dp: diffPct(p, df) };
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
  const top20 = rs.filter(r => r.sp >= 80).length;
  const lbl = state.custom ? 'custom weights' : AXLBL[state.axis];
  const items = [
    [rs.length, 'games in view'],
    [rs.length ? (wins / rs.length * 100).toFixed(1) + '%' : '-', 'win rate'],
    [rs.length ? pc(avg(rs.map(r => r.sp))) : '-', 'score percentile vs peers'],
    [rs.length ? (top20 / rs.length * 100).toFixed(0) + '%' : '-', 'games in peers\u2019 top 20%'],
    [rs.length ? pc(avg(rs.map(r => r.dp))) : '-', 'percentile vs own teammates'],
  ];
  document.getElementById('tiles').innerHTML = items.map(([big, l]) =>
    '<div class="tile"><div class="big">' + big + '</div><div class="lbl">' + l + '</div></div>').join('');
  document.getElementById('tilesNote').textContent =
    'Score = ' + lbl + (useResidual() ? ' on the vs-teammates (residual) axis' : ' on the box-score axis') +
    (state.custom ? ' \u00b7 percentiles read off a sampled peer baseline' : ' \u00b7 percentiles from exact peer quantile grids');
}

// ---------- weighting: corpus role models vs your own fit ----------
function axisBtns() {
  const box = document.getElementById('axisBtns');
  const role = (P.personal && P.personal.role) || majorityRole(rows()) || 'BOTTOM';
  box.innerHTML = '';
  REF.axes.forEach(a => {
    const b = document.createElement('button');
    b.className = 'btn' + (!state.custom && state.axis === a.key ? ' on' : '');
    const auc = REF.fit && REF.fit.auc[a.key] ? REF.fit.auc[a.key][role] : null;
    b.innerHTML = a.label + (auc ? '<br><span class="note">corpus AUC ' + auc.toFixed(2) + '</span>' : '');
    b.title = a.feats.length + ' metrics, ' + a.kind + ' axis';
    b.onclick = () => { state.custom = false; state.axis = a.key; scache.clear(); syncSliders(); render(); };
    box.appendChild(b);
  });
  document.getElementById('axisNote').innerHTML =
    '<b>Equal</b>: mean of the 14 signed z-scores, no fitting. <b>A</b>: raw box scores ' +
    '(inflated by stomps \u2014 high AUC mostly because winning teams look dominant). ' +
    '<b>B</b>: your share of team output. <b>C</b>: each stat\u2019s <i>within-team residual</i> ' +
    '(yours minus your teammates\u2019 mean), which is the axis that asks how good <i>you</i> were. ' +
    'All three model rows are fitted per role on ' + (REF.nRows || 0).toLocaleString() +
    ' player-games, split by match, with the held-out AUC shown on the button for your role (' + role + ').';
}

function modelPanel(rs) {
  const per = (P.personal && (P.personal[state.baseline] || P.personal.overall)) || null;
  const box = document.getElementById('modelPanel');
  if (!per || !per.axes) {
    box.innerHTML = '<div class="note">' + (per && per.unavailable ? per.unavailable
      : 'no personal fit available for this baseline') + '</div>';
    document.getElementById('slopePanel').innerHTML = '';
    document.getElementById('slopeNote').innerHTML = '';
    return;
  }
  const ax = per.axes[state.axis] || per.axes.equal;
  let html = '<table><tr><th>weighting</th><th>metric</th><th>your role\u2019s weight</th>' +
    '<th>your weight (ridge, n=' + per.n + ')</th></tr>';
  const feats = AX[state.axis].feats;
  const rowsH = feats.map(f => [f, ax.wPop[f] || 0, ax.wMine[f] || 0])
    .sort((a, b) => Math.abs(b[2] - b[1]) - Math.abs(a[2] - a[1]));
  rowsH.forEach(([f, wp, wm]) => {
    html += '<tr><td>' + AXLBL[state.axis] + '</td><td>' + LBL(f) + '</td>' +
      '<td>' + (wp >= 0 ? '+' : '') + nf(wp, 2) + '</td>' +
      '<td class="' + (Math.sign(wm) === Math.sign(wp) ? 'ok' : 'no') + '">' +
      (wm >= 0 ? '+' : '') + nf(wm, 2) + '</td></tr>';
  });
  html += '</table>';
  html += '<div class="note" style="margin-top:8px"><b>Same or different?</b> Cosine between the two vectors ' +
    'is <b>' + nf(ax.cosine, 2) + '</b> (1 = you win by doing exactly what your role wins by). ' +
    'Does each actually predict <i>your</i> wins? AUC on your ' + per.n + ' ' + per.role + ' games: ' +
    'role weights <b>' + nf(ax.aucPop, 3) + ' \u00b1' + nf(ax.aucPopSe, 3) + '</b>, ' +
    'weights fitted to you (leave-one-out) <b>' + nf(ax.aucMine, 3) + ' \u00b1' + nf(ax.aucMineSe, 3) + '</b>. ' +
    'With this many games the two are usually within noise of each other \u2014 a personal edge needs a ' +
    'gap bigger than the \u00b1SE before you believe it.</div>';
  html += '<div style="margin-top:8px"><button class="btn" id="useMine">Score me with my own weights</button>' +
    '<button class="btn" id="useRole">Back to role weights</button></div>';
  box.innerHTML = html;
  const um = document.getElementById('useMine'), ur = document.getElementById('useRole');
  if (um) um.onclick = () => {
    state.w = Object.assign({}, ax.wMine); state.custom = true; scache.clear(); syncSliders(); render();
  };
  if (ur) ur.onclick = () => {
    state.custom = false; scache.clear(); syncSliders(); render();
  };

  // --- "what do you actually win with": univariate win slopes, you vs peers ---
  const kind = useResidual() ? 'residual' : 'level';
  const sl = per.slopes[kind], peer = groupFor(per.role).slope[kind];
  let sh = '<table><tr><th>metric</th><th>you (logit / SD)</th><th>peers</th><th>difference</th></tr>';
  const order = MK.slice().sort((a, b) => Math.abs(sl[b][4]) - Math.abs(sl[a][4]));
  order.forEach(m => {
    const [b, se, pb, pse, z] = sl[m];
    const real = Math.abs(z) >= 2;
    sh += '<tr><td>' + LBL(m) + '</td>' +
      '<td>' + (b >= 0 ? '+' : '') + nf(b, 2) + ' <span class="note">\u00b1' + nf(se, 2) + '</span></td>' +
      '<td>' + (pb >= 0 ? '+' : '') + nf(pb, 2) + ' <span class="note">\u00b1' + nf(pse, 2) + '</span></td>' +
      '<td class="' + (real ? (z > 0 ? 'ok' : 'no') : '') + '">' + (real ? '<b>' : '') +
      (z >= 0 ? '+' : '') + nf(z, 1) + ' SE' + (real ? '</b>' : '') + '</td></tr>';
  });
  sh += '</table>';
  document.getElementById('slopePanel').innerHTML = sh;
  document.getElementById('slopeNote').innerHTML =
    'Reading: one extra SD of a stat raises your win odds by <i>you</i> (log-odds), versus <i>peers</i> at ' +
    per.role + ' / ' + (P.band && state.baseline === 'band' ? P.band : 'all ranks').toUpperCase() +
    '. This is the ' + kind + ' axis. The difference column is z = (you \u2212 peers) / combined SE. ' +
    '<b>Multiplicity matters:</b> ' + MK.length + ' metrics are tested at once, so one or two |z| near 2 is exactly ' +
    'what chance produces \u2014 |z| \u2265 2.9 is the level that survives a Bonferroni correction here. ' +
    'Treat the ordering of rows as the hint, and any single bold row as a hypothesis to collect more games against.';
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

const LBL = k => (METS.find(m => m.key === k) || { label: k }).label;

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
    const x = pad.l + i * bw, top = ys(r.sp), base = ys(0);
    svg.appendChild(el('rect', { x: x + bw * 0.18, y: top, width: Math.max(1.5, bw * 0.64),
      height: Math.max(1, base - top), fill: r.g.win ? '#3fb96f' : '#e05656', opacity: .75 }));
  });
  const win = 5, pts = [];
  for (let i = 0; i < n; i++) {
    const s = rs.slice(Math.max(0, i - win + 1), i + 1);
    pts.push([pad.l + i * bw + bw / 2, ys(avg(s.map(x => x.sp)))]);
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
  const role = majorityRole(rs), grp = groupFor(role);
  drawHist('outChart', rs.map(r => r.df), grp, state.custom ? grp.diff.equal : grp.diff[state.axis]);
  const better = rs.filter(r => r.df > 0).length;
  const axName = state.custom ? 'custom weights' : AXLBL[state.axis];
  document.getElementById('outNote').innerHTML =
    'You beat your own teammates in <b>' + better + ' of ' + rs.length + '</b> games \u00b7 average differential ' +
    (avg(rs.map(r => r.df)) >= 0 ? '<b class="win">' : '<b class="lose">') + nf(avg(rs.map(r => r.df)), 2) +
    '</b> SD \u00b7 shading/line are the ' + role + ' peer distribution. Under Model C the score <i>is</i> ' +
    'this differential, so the score panel and this one agree by construction (' + axName + ').';
}

// ---------- tables ----------
function champTable(rs) {
  const by = {};
  rs.forEach(r => {
    const c = by[r.g.champ] || (by[r.g.champ] = { n: 0, w: 0, cp: 0, role: r.g.role, key: r.g.champKey });
    c.n++; c.w += r.g.win ? 1 : 0; c.cp += r.sp;
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
      '<td>' + pc(r.sp) + '</td><td>' + (r.df >= 0 ? '<span class="win">+' : '<span class="lose">') + nf(r.df, 2) + '</span></td></tr>';
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
      state.custom = true;                 // leaving the published weightings
      scache.clear(); render();
    };
    box.appendChild(row);
  });
  document.getElementById('resetW').onclick = () => {
    state.custom = false;
    state.w = popW(state.axis, majorityRole(rows()) || ROLE_ORDER[0]);
    scache.clear(); syncSliders(); render();
  };
}
function syncSliders() {
  // in preset mode the sliders just show the fitted weights for the role in view
  if (!state.custom) state.w = popW(state.axis, majorityRole(rows()) || ROLE_ORDER[0]);
  const scale = state.custom ? 1 : 1;
  document.querySelectorAll('#weights input').forEach(inp => {
    const k = inp.dataset.key, v = (state.w[k] || 0);
    inp.value = Math.max(-3, Math.min(3, v * scale));
    inp.parentElement.querySelector('.val').textContent = (+v).toFixed(2);
  });
  document.getElementById('weightsAreCustom').textContent = state.custom
    ? 'custom weights (not a published model)'
    : 'showing ' + AXLBL[state.axis] + ' for ' + (majorityRole(rows()) || 'your main role');
}

function methodText() {
  const c = P.corpus, pr = P.profile;
  const per = (P.personal && (P.personal[state.baseline] || P.personal.overall)) || null;
  document.getElementById('method').innerHTML =
    '<b>Baseline.</b> Every metric is z-scored and percentiled against ' + c.nRows.toLocaleString() +
    ' player-games from ' + c.nMatches.toLocaleString() + ' ' + c.region +
    ' ranked solo queue games (collected via the Riot API, baseline frozen ' + c.generated.slice(0, 10) + '). ' +
    'Two baselines are offered: all ranks for your role, and your rank band \u00d7 role. ' +
    'Values are trimmed to the corpus 0.1%/99.9% range before scoring.<br>' +
    '<b>Score &amp; weightings.</b> The selected model multiplies each metric by a weight and averages over ' +
    'available metrics. <i>Equal</i> is the neutral signed mean. <i>A/B/C</i> are logistic win models fitted ' +
    '<b>per role</b> on ' + c.nRows.toLocaleString() + ' player-games, train/test split by match, and the button shows ' +
    'each one\u2019s held-out AUC for your role. A reads high (~0.9) mostly because box scores re-encode stomps; B ' +
    '(shares of team output) and C (within-team residuals) are the honest estimands, and C is the one that asks how good ' +
    '<i>you</i> were rather than how good your team looked. Percentiles are computed per game inside the role played in ' +
    'that game, then averaged \u2014 averaging raw stats first would let one 25-0 stomp dominate the result. Ties are given ' +
    'mid-rank (the same Mann-Whitney convention the AUC code uses), so a metric that is zero in most peer games does not ' +
    'report every tied game as 0%.<br>' +
    '<b>Your weights vs your role\u2019s.</b> The comparison refits a win model on your games alone, ridge-shrunk ' +
    '(C=' + (per.ridgeC || 0.2) + ') and scored leave-one-out so it cannot grade data it saw, with bootstrap SEs. ' +
    'With ' + per.n + ' games at ' + per.role + ', its coefficients are noisy by construction: the gap to the role model ' +
    'is a hypothesis about how you differ, not a measurement. Rows in the slope table are the better-powered version ' +
    '(one stat at a time, both sides with SEs).<br>' +
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
  axisBtns(); baselineBtns(); roleBtns(); weightsBox(); syncSliders();
  tiles(rs);
  modelPanel(rs);
  if (!rs.length) {
    ['trend', 'hist', 'outChart'].forEach(id => clear(document.getElementById(id)));
    document.getElementById('metricBars').innerHTML = '<div class="note">no games in this filter</div>';
    document.getElementById('champTable').innerHTML = document.getElementById('gameTable').innerHTML = '';
    return;
  }
  metricBars(rs);
  drawTrend(rs);
  const grp = groupFor(majorityRole(rs));
  drawHist('hist', rs.map(r => r.sc), grp, state.custom ? grp.score.equal : grp.score[state.axis]);
  drawOut(rs);
  champTable(rs);
  gameTable(rs);
  methodText();
}
header();
render();
window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(render, 150); });
})();
