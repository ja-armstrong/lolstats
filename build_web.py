"""
Build the interactive explorer: data/web/index.html (single self-contained file).

Embeds a compact version of the player-game table (z-scores scaled x100),
per-role fitted model presets, and a vanilla-JS app that lets you:
  - filter by role, rank band, minimum game length
  - set your own weights per metric (the 'twiddle coefficients' part)
  - see live AUC, win-rate-by-decile curve, win/loss score distributions,
    and a role x rank impact heatmap recomputed with YOUR weights

Re-run any time after collect_matches/prepare_features to refresh the data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
IN_FILE = ROOT / "data" / "processed" / "player_games.parquet"
OUT_DIR = ROOT / "web"
OUT_DIR.mkdir(exist_ok=True)

APEX_TIERS = {"CHALLENGER", "GRANDMASTER", "MASTER"}
METRICS = ["kda", "kp", "csm", "gpm", "dpm", "dmg_share", "vspm", "wpm",
           "dthpm", "skpm", "gold_share", "cs_share", "vision_share", "death_share"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
BANDS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "APEX"]
LABELS = {
    "kda": "KDA (k+a)/d", "kp": "Kill participation", "csm": "CS per minute",
    "gpm": "Gold per minute", "dpm": "Damage to champs / min",
    "dmg_share": "Damage share of team", "vspm": "Vision score / min",
    "wpm": "Wards placed / min", "dthpm": "Deaths per minute",
    "skpm": "Solo kills / min", "gold_share": "Gold share of team",
    "cs_share": "CS share of team", "vision_share": "Vision share of team",
    "death_share": "Death share of team",
}


def band_of(tier: str) -> str:
    base = str(tier).split("_")[0]
    return "APEX" if base in APEX_TIERS else base


def main():
    df = pd.read_parquet(IN_FILE)
    df = df[df["role"].isin(ROLES)].copy()
    df["band"] = df["tier"].map(band_of)
    df = df[df["band"].isin(BANDS)]

    # per-role fitted presets — fitted on a TRAIN split only, using ONLY the
    # 10 raw-stat metrics. (Adding the 4 team-share metrics lets an LR
    # reconstruct team totals (gold ~ share x team gold) and inflate AUC to
    # ~0.98 by detecting team dominance rather than individual performance.)
    # The page excludes these matches from scoring when a preset is active,
    # so preset AUCs shown in the explorer are honest (out-of-sample).
    LEVEL = METRICS[:10]
    mids, _ = pd.factorize(df["matchId"])
    tr_pos, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                       random_state=42).split(df, groups=mids))
    fit_match_idx = sorted({int(mids[i]) for i in tr_pos})
    fit_mask = np.isin(mids, fit_match_idx)
    presets = {}
    for role in ROLES:
        r = df[fit_mask & (df["role"] == role)]
        X = r[[m + "_zt" for m in LEVEL]].fillna(0.0).values
        y = r["win"].astype(int).values
        model = LogisticRegression(max_iter=2000).fit(X, y)
        presets[role] = {m: round(float(w), 3) for m, w in zip(LEVEL, model.coef_[0])}

    role_idx = {r: i for i, r in enumerate(ROLES)}
    band_idx = {b: i for i, b in enumerate(BANDS)}

    rows = []
    for i, rec in enumerate(df.itertuples(index=False)):
        zs = []
        for m in METRICS:
            v = getattr(rec, m + "_zt")
            zs.append(int(round(v * 100)) if pd.notna(v) else 0)
        rows.append([
            role_idx[rec.role], band_idx[rec.band],
            1 if rec.win else 0, int(rec.minutes * 10), int(mids[i])
        ] + zs)

    data = {
        "metrics": [{"key": m, "label": LABELS[m]} for m in METRICS],
        "roles": ROLES, "bands": BANDS,
        "presets": presets,
        "fitMatches": fit_match_idx,
        "nMatches": int(df["matchId"].nunique()),
        "rows": rows,
    }
    payload = json.dumps(data, separators=(",", ":"))

    html = TEMPLATE.replace("__DATA__", payload)
    out = OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(rows):,} player-games, {data['nMatches']:,} matches)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LoL Role Impact Explorer</title>
<style>
:root{
  --bg:#0f1115;--panel:#171a21;--panel2:#1d222b;--text:#e6e9ef;--dim:#8b93a7;
  --accent:#4f9cf9;--good:#3fb96f;--bad:#e05656;--line:#2a3040;
  --top:#4f9cf9;--jungle:#e05656;--middle:#3fb96f;--bottom:#f0a13a;--support:#b07cf0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
header{padding:18px 24px 10px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--dim);font-size:13px}
#layout{display:grid;grid-template-columns:320px 1fr;gap:16px;padding:0 24px 30px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:0 0 10px}
label.ck{display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer}
input[type=checkbox]{accent-color:var(--accent)}
input[type=range]{width:100%;accent-color:var(--accent)}
.wrow{display:grid;grid-template-columns:150px 1fr 52px;gap:8px;align-items:center;margin:5px 0}
.wrow .val{text-align:right;font-variant-numeric:tabular-nums;color:var(--dim)}
.wrow .name{font-size:12.5px}
button{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:6px;
  padding:5px 10px;cursor:pointer;font-size:12.5px}
button:hover{border-color:var(--accent)}
select{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .big{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .lbl{color:var(--dim);font-size:12px;margin-top:2px}
svg{display:block;width:100%}
.legend{display:flex;gap:14px;color:var(--dim);font-size:12px;margin:2px 0 8px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}
th:first-child,td:first-child{text-align:left}
.heat{display:grid;grid-template-columns:90px repeat(5,1fr);gap:3px;font-size:11px}
.hcell{border-radius:4px;padding:5px 3px;text-align:center;font-variant-numeric:tabular-nums}
.hrole{color:var(--dim);padding:5px 3px;text-align:left;font-size:12px}
.hband{color:var(--dim);text-align:center;padding:2px 0}
.hcell.best{outline:2px solid #fff}
</style>
</head>
<body>
<header>
  <h1>LoL Role Impact Explorer</h1>
  <div class="sub" id="meta"></div>
</header>
<div id="layout">
  <div>
    <div class="panel">
      <h2>Roles</h2><div id="roleBoxes"></div>
    </div>
    <div class="panel">
      <h2>Rank bands</h2><div id="bandBoxes"></div>
    </div>
    <div class="panel">
      <h2>Filters</h2>
      <div style="color:var(--dim);font-size:12px">Min game length: <span id="minLenV">20</span> min</div>
      <input type="range" id="minLen" min="10" max="60" value="20" step="1">
    </div>
    <div class="panel">
      <h2>Your score weights</h2>
      <div style="margin-bottom:8px">
        <select id="preset"></select>
        <button id="resetW" style="margin-left:6px">Reset</button>
      </div>
      <div class="sub" style="font-size:11px;margin:-4px 0 8px">Fitted presets use the 10 raw-stat metrics and are scored out-of-sample (training matches excluded). Heads-up: giving weight to the four <b>share</b> metrics lets a score reconstruct team totals (e.g. gold &asymp; share &times; team gold), which inflates AUC &mdash; that reads team dominance, not individual impact.</div>
      <div id="weights"></div>
    </div>
  </div>
  <div>
    <div class="tiles">
      <div class="tile"><div class="big" id="tN">–</div><div class="lbl">player-games in view</div></div>
      <div class="tile"><div class="big" id="tAuc">–</div><div class="lbl">AUC of your score</div></div>
      <div class="tile"><div class="big" id="tHi">–</div><div class="lbl">win rate, top score decile</div></div>
      <div class="tile"><div class="big" id="tLo">–</div><div class="lbl">win rate, bottom score decile</div></div>
    </div>
    <div class="panel">
      <h2>Win rate by your-score decile</h2>
      <div class="legend"><span><span class="dot" style="background:var(--accent)"></span>win rate per decile of your custom score</span></div>
      <svg id="decChart" height="230"></svg>
    </div>
    <div class="panel">
      <h2>Score distribution — wins vs losses</h2>
      <div class="legend">
        <span><span class="dot" style="background:var(--good)"></span>wins</span>
        <span><span class="dot" style="background:var(--bad)"></span>losses</span>
      </div>
      <svg id="histChart" height="180"></svg>
    </div>
    <div class="panel">
      <h2>Role &times; rank impact heatmap (AUC of your score, per subset)</h2>
      <div id="heat" class="heat"></div>
      <div class="sub" style="margin-top:8px">White outline = best role in that band. Cells with &lt;150 games are dimmed.</div>
    </div>
    <div class="panel">
      <h2>Single-feature AUCs in current view</h2>
      <table id="featTable"></table>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const FIT = new Set(DATA.fitMatches);
const M = DATA.metrics.map(m=>m.key);
const L = Object.fromEntries(DATA.metrics.map(m=>[m.key,m.label]));
const ROLE_COLORS = {TOP:"#4f9cf9",JUNGLE:"#e05656",MIDDLE:"#3fb96f",BOTTOM:"#f0a13a",SUPPORT:"#b07cf0"};
const DEFAULT_W = Object.fromEntries(M.map(k=>[k, (k==="dthpm"||k==="death_share")?-1:1]));

const state = {
  roles:new Set(DATA.roles), bands:new Set(DATA.bands),
  minLen:20, weights:{...DEFAULT_W}
};

// ---------- build controls ----------
const roleBoxes = document.getElementById("roleBoxes");
DATA.roles.forEach(r=>{
  const l=document.createElement("label");l.className="ck";
  l.innerHTML=`<input type="checkbox" checked> <span class="dot" style="background:${ROLE_COLORS[r]}"></span>${r}`;
  l.querySelector("input").onchange=e=>{e.target.checked?state.roles.add(r):state.roles.delete(r);update();};
  roleBoxes.appendChild(l);
});
const bandBoxes = document.getElementById("bandBoxes");
DATA.bands.forEach(b=>{
  const l=document.createElement("label");l.className="ck";
  l.innerHTML=`<input type="checkbox" checked> ${b}`;
  l.querySelector("input").onchange=e=>{e.target.checked?state.bands.add(b):state.bands.delete(b);update();};
  bandBoxes.appendChild(l);
});
const minLen=document.getElementById("minLen");
minLen.oninput=e=>{state.minLen=+e.target.value;document.getElementById("minLenV").textContent=state.minLen;update();};

const wBox=document.getElementById("weights");
M.forEach(k=>{
  const row=document.createElement("div");row.className="wrow";
  row.innerHTML=`<span class="name">${L[k]}</span><input type="range" min="-3" max="3" step="0.1" value="${DEFAULT_W[k]}"><span class="val">${DEFAULT_W[k].toFixed(1)}</span>`;
  const inp=row.querySelector("input"), val=row.querySelector(".val");
  inp.oninput=()=>{state.weights[k]=+inp.value;val.textContent=(+inp.value).toFixed(1);update();};
  wBox.appendChild(row); inp.dataset.key=k;
});
const presetSel=document.getElementById("preset");
presetSel.innerHTML=`<option value="equal">Equal weight (default signs)</option>`+
  DATA.roles.map(r=>`<option value="${r}">Fitted win-model — ${r}</option>`).join("");
presetSel.onchange=()=>{const v=presetSel.value;
  if(v==="equal"){setWeights(DEFAULT_W);}else{setWeights(DATA.presets[v]);}};
function setWeights(w){M.forEach(k=>{state.weights[k]=w[k]??0;});
  [...wBox.querySelectorAll("input")].forEach(inp=>{const k=inp.dataset.key;
    inp.value=state.weights[k];inp.parentElement.querySelector(".val").textContent=state.weights[k].toFixed(1);});
  update();}
document.getElementById("resetW").onclick=()=>{presetSel.value="equal";setWeights(DEFAULT_W);};

document.getElementById("meta").textContent =
  `${DATA.nMatches.toLocaleString()} ranked solo queue matches (NA), ${DATA.rows.length.toLocaleString()} player-games — Riot API`;

// ---------- scoring ----------
function filtered(){
  const out=[];
  for(const r of DATA.rows){
    if(!state.roles.has(DATA.roles[r[0]])) continue;
    if(!state.bands.has(DATA.bands[r[1]])) continue;
    if(r[3] < state.minLen*10) continue;
    out.push(r);
  }
  return out;
}
function scoreOf(r){
  let s=0,norm=0;
  for(let i=0;i<M.length;i++){const w=state.weights[M[i]];if(!w)continue;s+=w*r[5+i]/100;norm+=Math.abs(w);}
  return norm?s/norm:0;
}
function auc(scores,labels){ // labels: 1 win, 0 loss (Mann-Whitney with ties)
  const n=scores.length;if(!n)return null;
  const idx=scores.map((s,i)=>[s,labels[i],i]).sort((a,b)=>a[0]-b[0]);
  let rankSumPos=0,nPos=0,i=0;
  while(i<n){
    let j=i;while(j<n&&idx[j][0]===idx[i][0])j++;
    const avgRank=(i+j+1)/2; // average of ranks i+1..j
    for(let k=i;k<j;k++){if(idx[k][1]===1){rankSumPos+=avgRank;nPos++;}}
    i=j;
  }
  const nNeg=n-nPos;
  if(!nPos||!nNeg)return null;
  return (rankSumPos-nPos*(nPos+1)/2)/(nPos*nNeg);
}

// ---------- charts ----------
function el(tag,attrs,text){const e=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const k in attrs)e.setAttribute(k,attrs[k]);if(text!=null)e.textContent=text;return e;}
function clear(svg){while(svg.firstChild)svg.removeChild(svg.firstChild);}

function drawDecile(rows){
  const svg=document.getElementById("decChart");clear(svg);
  const W=svg.clientWidth||900,H=230,pad={l:46,r:12,t:12,b:28};
  const sc=rows.map(r=>scoreOf(r)), y=rows.map(r=>r[2]);
  const order=sc.map((s,i)=>i).sort((a,b)=>sc[a]-sc[b]);
  const D=10, bins=[];const n=order.length;
  for(let d=0;d<D;d++){
    const a=Math.floor(d*n/D), b=Math.floor((d+1)*n/D)||n;
    let w=0,c=0;for(let k=a;k<b;k++){w+=y[order[k]];c++;}
    bins.push(c?w/c:null);
  }
  const xs=d=>pad.l+(W-pad.l-pad.r)*(d+0.5)/D;
  const ys=v=>pad.t+(H-pad.t-pad.b)*(1-v);
  [0.25,0.5,0.75].forEach(v=>{
    svg.appendChild(el("line",{x1:pad.l,x2:W-pad.r,y1:ys(v),y2:ys(v),stroke:"#2a3040","stroke-dasharray":"3 3"}));
    svg.appendChild(el("text",{x:pad.l-6,y:ys(v)+4,"text-anchor":"end","font-size":10,fill:"#8b93a7"},Math.round(v*100)+"%"));});
  let prev=null;
  bins.forEach((v,d)=>{ if(v==null)return;
    if(prev)svg.appendChild(el("line",{x1:xs(d-1),y1:ys(prev),x2:xs(d),y2:ys(v),stroke:"#4f9cf9","stroke-width":2}));
    svg.appendChild(el("circle",{cx:xs(d),cy:ys(v),r:4,fill:"#4f9cf9"}));
    svg.appendChild(el("text",{x:xs(d),y:H-8,"text-anchor":"middle","font-size":10,fill:"#8b93a7"},"D"+(d+1)));
    prev=v;});
  svg.appendChild(el("text",{x:pad.l-6,y:ys(1)+12,"text-anchor":"end","font-size":10,fill:"#8b93a7"},"100%"));
  svg.appendChild(el("text",{x:pad.l-6,y:ys(0)-2,"text-anchor":"end","font-size":10,fill:"#8b93a7"},"0%"));
}

function drawHist(rows){
  const svg=document.getElementById("histChart");clear(svg);
  const W=svg.clientWidth||900,H=180,pad={l:40,r:12,t:10,b:24};
  const sc=rows.map(r=>scoreOf(r));
  const lo=Math.min(...sc),hi=Math.max(...sc),NB=28;
  const wC=new Array(NB).fill(0),lC=new Array(NB).fill(0);
  rows.forEach((r,i)=>{const b=Math.min(NB-1,Math.floor((sc[i]-lo)/(hi-lo+1e-9)*NB));(r[2]?wC:lC)[b]++;});
  const wTot=wC.reduce((a,b)=>a+b,0)||1,lTot=lC.reduce((a,b)=>a+b,0)||1;
  const mx=Math.max(...wC.map((v,i)=>v/wTot),...lC.map((v,i)=>v/lTot));
  const bw=(W-pad.l-pad.r)/NB;
  for(let b=0;b<NB;b++){
    const h1=wC[b]/wTot/mx*(H-pad.t-pad.b), h2=lC[b]/lTot/mx*(H-pad.t-pad.b);
    svg.appendChild(el("rect",{x:pad.l+b*bw+1,y:H-pad.b-h1,width:bw-2,height:h1,fill:"#3fb96f",opacity:.75}));
    svg.appendChild(el("rect",{x:pad.l+b*bw+1,y:H-pad.b-h2,width:bw-2,height:h2,fill:"#e05656",opacity:.55}));
  }
  for(const v of [lo,(lo+hi)/2,hi])
    svg.appendChild(el("text",{x:pad.l+(W-pad.l-pad.r)*(v-lo)/(hi-lo+1e-9),y:H-6,"text-anchor":"middle","font-size":10,fill:"#8b93a7"},v.toFixed(2)));
}

function drawHeat(rows){
  const box=document.getElementById("heat");box.innerHTML="";
  box.appendChild(Object.assign(document.createElement("div"),{className:"hband",textContent:""}));
  DATA.roles.forEach(r=>box.appendChild(Object.assign(document.createElement("div"),{className:"hband",textContent:r})));
  const grid={};
  DATA.bands.forEach(b=>{
    box.appendChild(Object.assign(document.createElement("div"),{className:"hrole",textContent:b}));
    let best=null,bestV=-1;
    const cellAucs=[];
    DATA.roles.forEach(r=>{
      const sub=rows.filter(x=>DATA.roles[x[0]]===r&&DATA.bands[x[1]]===b);
      let v=null;
      if(sub.length>=150){const sc=sub.map(x=>scoreOf(x));v=auc(sc,sub.map(x=>x[2]));}
      cellAucs.push([r,v,sub.length]);
      if(v!=null&&v>bestV){bestV=v;best=r;}
    });
    cellAucs.forEach(([r,v,n])=>{
      const d=document.createElement("div");d.className="hcell";
      if(v==null){d.style.background="#20242e";d.style.color="#555";d.textContent="–";}
      else{
        const t=Math.max(0,Math.min(1,(v-0.45)/0.25)); // map .45-.70 to color
        d.style.background=`rgba(${Math.round(224-180*t)},${Math.round(86+90*t)},${Math.round(86-40*t)},${0.25+0.6*t})`;
        d.textContent=v.toFixed(3);
        d.title=`${r} @ ${b}: AUC ${v.toFixed(3)}, ${n} games`;
        if(n<300)d.style.opacity=.45;
        if(r===best&&v!=null)d.classList.add("best");
      }
      box.appendChild(d);
    });
  });
}

function drawFeat(rows){
  const t=document.getElementById("featTable");
  let html="<tr><th>metric</th><th>AUC (as-is)</th></tr>";
  const aucs=M.map(k=>{
    const sc=rows.map(r=>r[5+M.indexOf(k)]/100);
    return [k,auc(sc,rows.map(r=>r[2]))];
  }).sort((a,b)=>(b[1]??-1)-(a[1]??-1));
  aucs.forEach(([k,v])=>{html+=`<tr><td>${L[k]}</td><td>${v==null?"–":v.toFixed(3)}</td></tr>`;});
  t.innerHTML=html;
}

let tm=null;
function update(){
  clearTimeout(tm);
  tm=setTimeout(()=>{
    let rows=filtered();
    if(presetSel.value!=="equal")rows=rows.filter(r=>!FIT.has(r[4]));
    document.getElementById("tN").textContent=rows.length.toLocaleString();
    if(rows.length<50){["tAuc","tHi","tLo"].forEach(i=>document.getElementById(i).textContent="–");
      ["decChart","histChart"].forEach(i=>clear(document.getElementById(i)));
      document.getElementById("heat").innerHTML="<div class='hband'>Not enough data for this filter</div>";
      document.getElementById("featTable").innerHTML="";return;}
    const sc=rows.map(r=>scoreOf(r)), y=rows.map(r=>r[2]);
    const a=auc(sc,y);
    document.getElementById("tAuc").textContent=a?a.toFixed(3):"–";
    const order=sc.map((s,i)=>i).sort((x,z)=>sc[x]-sc[z]);
    const n=order.length,D=10;
    const dec=(q)=>{const lo=Math.floor(q*n/D),hi=Math.max(lo+1,Math.floor((q+1)*n/D));
      let w=0,c=0;for(let k=lo;k<hi;k++){w+=y[order[k]];c++;}return c?w/c:null;};
    document.getElementById("tLo").textContent=(dec(0)*100).toFixed(1)+"%";
    document.getElementById("tHi").textContent=(dec(9)*100).toFixed(1)+"%";
    drawDecile(rows);drawHist(rows);drawHeat(rows);drawFeat(rows);
  },60);
}
update();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()