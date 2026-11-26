"""Interactive version of the strategy-trajectory figures: PC1 x PC2 of the
server's weights, with the round t on a slider (or the mouse wheel over the chart).

    python experiments/strategy_trajectories_explorer.py

Reads outputs/strategy_trajectories/projection_<base>.json (written by
plot_strategy_trajectories.py, one per base run) and writes ONE self-contained
page, docs/strategy_trajectories_explorer.html: the data is embedded, so it
opens straight from disk in any browser, no server and no installation needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDER = ["fedavg", "fedavgm", "fedprox", "fedadam", "fedyogi"]


def compact(value, digits: int = 4):
    """Round every float (the page stays small)."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [compact(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: compact(v, digits) for k, v in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/strategy_trajectories")
    parser.add_argument("--html", type=Path,
                        default=REPO_ROOT / "docs/strategy_trajectories_explorer.html")
    args = parser.parse_args()

    files = {p.stem.removeprefix("projection_"): p for p in args.out.glob("projection_*.json")}
    if not files:
        raise SystemExit(f"No projection_<base>.json in {args.out}: "
                         "run experiments/plot_strategy_trajectories.py first.")
    data = {b: compact(json.loads(files[b].read_text()))
            for b in sorted(files, key=lambda b: ORDER.index(b) if b in ORDER else 99)}
    html = TEMPLATE.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(html)
    print(f"saved {args.html} ({len(html) / 1e6:.1f} MB), bases: {', '.join(data)}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy trajectories explorer</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-0: #f6f6f4; --surface-1: #ffffff; --border: #e2e2de;
  --text-primary: #191919; --text-secondary: #666666; --grid: #ececea;
  --fedavg: #257de0; --fedprox: #00b894; --fedavgm: #ec4647;
  --fedadam: #6a4ce2; --fedyogi: #f47230; --start: #8a8a8a;
  --accent: #257de0;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-0: #121212; --surface-1: #191919; --border: #2e2e2e;
    --text-primary: #efefef; --text-secondary: #9c9c9c; --grid: #262626;
    --fedavg: #338ced; --fedprox: #00a888; --fedavgm: #eb6263;
    --fedadam: #9887ef; --fedyogi: #e76624; --start: #8a8a8a; --accent: #338ced;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-0: #121212; --surface-1: #191919; --border: #2e2e2e;
  --text-primary: #efefef; --text-secondary: #9c9c9c; --grid: #262626;
  --fedavg: #338ced; --fedprox: #00a888; --fedavgm: #eb6263;
  --fedadam: #9887ef; --fedyogi: #e76624; --start: #8a8a8a; --accent: #338ced;
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root {
  min-height: 100vh; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  padding: 28px 20px 40px;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 6px; font-weight: 650; letter-spacing: -0.01em; }
.lede { color: var(--text-secondary); margin: 0 0 18px; max-width: 80ch; }
.controls {
  display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; margin-bottom: 14px;
}
.group { display: flex; align-items: center; gap: 8px; }
.group > span { color: var(--text-secondary); font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.05em; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button {
  border: 0; background: transparent; color: var(--text-primary); padding: 6px 12px;
  font: inherit; cursor: pointer;
}
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.time { flex: 1 1 340px; display: flex; align-items: center; gap: 10px; }
.time input { flex: 1; accent-color: var(--accent); }
.play {
  width: 34px; height: 34px; border-radius: 50%; border: 1px solid var(--border);
  background: var(--surface-1); color: var(--text-primary); cursor: pointer; font-size: 13px;
}
.tval { font-variant-numeric: tabular-nums; min-width: 9ch; font-weight: 600; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 14px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; }
.card h2 { font-size: 14px; margin: 0 0 2px; font-weight: 600; }
.card .sub { color: var(--text-secondary); font-size: 12.5px; margin: 0 0 8px; }
svg { display: block; width: 100%; height: auto; user-select: none; }
.axis, .tick { fill: var(--text-secondary); font-size: 11px; }
.axis { font-size: 12px; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.zero { stroke: var(--text-secondary); stroke-opacity: 0.35; stroke-dasharray: 2 3; }
.label { font-size: 12px; fill: var(--text-primary); paint-order: stroke;
  stroke: var(--surface-1); stroke-width: 3px; stroke-linejoin: round; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 10px; font-size: 12.5px; }
.legend label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
.legend input { margin: 0; accent-color: var(--accent); }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 4px; text-align: right; border-bottom: 1px solid var(--border); }
th { color: var(--text-secondary); font-weight: 500; }
th:first-child, td:first-child { text-align: left; }
.sw { display: inline-block; width: 18px; height: 0; border-top: 2px solid; vertical-align: middle;
  margin-right: 6px; }
.sw.dash { border-top-style: dashed; }
.note { color: var(--text-secondary); font-size: 12px; margin-top: 10px; }
.tip {
  position: fixed; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 7px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,.12); opacity: 0; transition: opacity .08s; z-index: 5;
  font-variant-numeric: tabular-nums;
}
.tip b { font-weight: 600; }
.head { display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; }
.head .seg { flex: none; font-size: 12.5px; }
</style>
</head>
<body>
<div class="viz-root">
<div class="wrap">
  <div class="head"><h1>How the server's weights move: FedAvg, FedProx, FedAvgM, FedAdam, FedYogi</h1>
    <div class="seg" id="theme"><button data-v="light">Light</button><button data-v="dark">Dark</button></div></div>
  <p class="lede">Each curve is the global model of one strategy (mean of the seeds), compressed
  from all weights to 2 coordinates with <b>one</b> PCA shared by every curve. Drag the slider,
  press play, or scroll over the chart to move through the rounds. Solid: real runs.
  Dashed, lighter: at every round, the model another strategy <i>would</i> produce from the
  base run's model in one step; the base then continues with its own update.</p>

  <div class="controls">
    <div class="group"><span>Steps from</span><div class="seg" id="base"></div></div>
    <div class="group"><span>View</span><div class="seg" id="view">
      <button data-v="absolute">Absolute</button><button data-v="relative">Minus base</button>
    </div></div>
    <div class="group"><span>Path</span><div class="seg" id="trail">
      <button data-v="ghost">Whole, faded</button><button data-v="sofar">Up to t</button>
    </div></div>
    <div class="time">
      <button class="play" id="play" aria-label="Play">&#9654;</button>
      <input type="range" id="t" min="0" value="0" aria-label="Round">
      <span class="tval" id="tval"></span>
    </div>
  </div>

  <div class="layout">
    <div class="card">
      <h2 id="ctitle"></h2>
      <p class="sub" id="csub"></p>
      <svg id="chart" viewBox="0 0 760 520" role="img" aria-label="PC1 against PC2"></svg>
      <div class="legend" id="legend"></div>
    </div>
    <div class="card">
      <h2>At round <span id="tval2"></span></h2>
      <p class="sub">Mean over seeds. Distance: L2 over all weights, to the base model of
      the same round (no projection).</p>
      <table id="table"></table>
      <h2 style="margin-top:16px">Final test set</h2>
      <p class="sub">Evaluated once, after the last round (mean &plusmn; std).</p>
      <table id="test"></table>
      <p class="note" id="explained"></p>
    </div>
  </div>
</div>
<div class="tip" id="tip"></div>
</div>
<script>
const DATA = /*DATA*/null;
const NAMES = {fedavg: "FedAvg", fedprox: "FedProx", fedavgm: "FedAvgM", fedadam: "FedAdam",
               fedyogi: "FedYogi"};
const ORDER = ["fedavg", "fedprox", "fedavgm", "fedadam", "fedyogi"];
const W = 760, H = 520, M = {l: 70, r: 22, t: 14, b: 44};
const state = {base: Object.keys(DATA)[0], view: "absolute", trail: "ghost", i: 0,
               hidden: new Set(), playing: null};
const $ = id => document.getElementById(id);
const svg = $("chart"), NS = "http://www.w3.org/2000/svg";
const color = (s, light) => light
  ? `color-mix(in oklab, var(--${s}) 50%, var(--surface-1))` : `var(--${s})`;

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function niceTicks(lo, hi, n = 6) {
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
// same number of decimals for every tick of an axis, from its step
const fmtTicks = ts => { const st = ts.length > 1 ? Math.abs(ts[1] - ts[0]) : 1;
  const dec = Math.max(0, -Math.floor(Math.log10(st) + 1e-9) + (st / 10 ** Math.floor(Math.log10(st)) % 1 ? 1 : 0));
  return ts.map(t => t.toFixed(dec)); };

function series() {
  const d = DATA[state.base], v = d[state.view], out = [];
  for (const s of ORDER) if (v.real[s]) out.push({s, cf: false, pts: v.real[s]});
  for (const s of ORDER) if (v.counterfactual[s]) out.push({s, cf: true, pts: v.counterfactual[s]});
  return out;
}

let scale;
function draw() {
  const d = DATA[state.base], all = series(), i = state.i, base = NAMES[d.base];
  svg.textContent = "";
  // fixed extent over ALL rounds, so the axes never move while you scrub
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const {pts} of all) for (const [x, y] of pts) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.08 || 1;
  x0 -= px; x1 += px; y0 -= py; y1 += py;
  const sx = x => M.l + (x - x0) / (x1 - x0) * (W - M.l - M.r);
  const sy = y => H - M.b - (y - y0) / (y1 - y0) * (H - M.t - M.b);
  scale = {sx, sy};
  const g = el("g", {}, svg);
  const xt = niceTicks(x0, x1), yt = niceTicks(y0, y1), xl = fmtTicks(xt), yl = fmtTicks(yt);
  xt.forEach((t, k) => {
    el("line", {x1: sx(t), x2: sx(t), y1: M.t, y2: H - M.b, class: "gridline"}, g);
    el("text", {x: sx(t), y: H - M.b + 16, "text-anchor": "middle", class: "tick"}, g).textContent = xl[k];
  });
  yt.forEach((t, k) => {
    el("line", {x1: M.l, x2: W - M.r, y1: sy(t), y2: sy(t), class: "gridline"}, g);
    el("text", {x: M.l - 8, y: sy(t) + 4, "text-anchor": "end", class: "tick"}, g).textContent = yl[k];
  });
  if (x0 < 0 && x1 > 0) el("line", {x1: sx(0), x2: sx(0), y1: M.t, y2: H - M.b, class: "zero"}, g);
  if (y0 < 0 && y1 > 0) el("line", {x1: M.l, x2: W - M.r, y1: sy(0), y2: sy(0), class: "zero"}, g);
  const ex = d[state.view].explained;
  el("text", {x: (M.l + W - M.r) / 2, y: H - 8, "text-anchor": "middle", class: "axis"}, svg)
    .innerHTML = `<tspan>PC1 (${(ex[0] * 100).toFixed(0)}% of variance)</tspan>`;
  el("text", {x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle", class: "axis",
              transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})`}, svg)
    .innerHTML = `<tspan>PC2 (${(ex[1] * 100).toFixed(0)}% of variance)</tspan>`;

  const path = (pts, a, b) => pts.slice(a, b + 1)
    .map(([x, y], k) => `${k ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
  const visible = all.filter(r => !state.hidden.has(r.s + (r.cf ? ":cf" : "")));
  // faded whole path
  if (state.trail === "ghost") for (const r of visible) {
    el("path", {d: path(r.pts, r.cf ? 1 : 0, r.pts.length - 1), fill: "none",
      stroke: color(r.s, r.cf), "stroke-width": 1.5, "stroke-opacity": 0.22,
      "stroke-dasharray": r.cf ? "5 4" : "none"}, svg);
  }
  // path up to t
  for (const r of visible) if (i >= (r.cf ? 1 : 0)) {
    el("path", {d: path(r.pts, r.cf ? 1 : 0, i), fill: "none", stroke: color(r.s, r.cf),
      "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": r.cf ? "5 4" : "none"}, svg);
  }
  // shared start (same initial model for every strategy)
  const st = visible.find(r => !r.cf);
  if (st) {
    el("circle", {cx: sx(st.pts[0][0]), cy: sy(st.pts[0][1]), r: 4, fill: "var(--start)",
      stroke: "var(--surface-1)", "stroke-width": 2}, svg);
  }
  // heads + direct labels (no two labels on top of each other)
  const heads = [];
  for (const r of visible) {
    if (r.cf && i === 0) continue;
    const [x, y] = r.pts[i];
    el("circle", {cx: sx(x), cy: sy(y), r: r.cf ? 4.5 : 5.5,
      fill: r.cf ? "var(--surface-1)" : color(r.s, false), stroke: r.cf ? color(r.s, true) : "var(--surface-1)",
      "stroke-width": r.cf ? 2.2 : 2}, svg);
    heads.push({x: sx(x), y: sy(y), s: r.s, cf: r.cf});
  }
  // heads closer than 12 px share ONE label (hover the point for the details)
  const groups = [];
  for (const h of heads) {
    const g2 = groups.find(gr => Math.hypot(gr.x - h.x, gr.y - h.y) < 12);
    g2 ? g2.items.push(h) : groups.push({x: h.x, y: h.y, items: [h]});
  }
  const placed = [];
  groups.sort((a, b) => a.y - b.y);
  for (const gr of groups) {
    const real = gr.items.filter(h => !h.cf).map(h => NAMES[h.s]);
    const cfs = gr.items.filter(h => h.cf).map(h => NAMES[h.s]);
    let text = real.join(", ");
    if (cfs.length) text += real.length
      ? (cfs.length > 1 ? ` + ${cfs.length} steps` : ` + ${cfs[0]} step`)
      : (cfs.length > 1 ? `${cfs.length} steps` : `${cfs[0]} step`);
    let ly = Math.min(Math.max(gr.y + 4, M.t + 10), H - M.b - 4);
    for (const p of placed) if (Math.abs(p.x - gr.x) < 110 && Math.abs(p.y - ly) < 14) ly = p.y + 14;
    placed.push({x: gr.x, y: ly});
    const right = gr.x < W - 170;
    el("text", {x: gr.x + (right ? 9 : -9), y: ly, "text-anchor": right ? "start" : "end",
      class: "label", "font-style": real.length ? "normal" : "italic"}, svg).textContent = text;
  }
  $("ctitle").textContent = state.view === "absolute"
    ? "Server weights, PC1 against PC2"
    : `Server weights minus ${base}'s of the same round`;
  $("csub").textContent = state.view === "absolute"
    ? `One PCA over all curves. Dashed: one step of another strategy from ${base}'s model.`
    : `${base} is the origin (0, 0) at every round. Dashed: how far one step of another strategy lands from ${base}'s next model.`;
  const r = d.rounds[i];
  $("tval").textContent = `t = ${r}`;
  $("tval2").textContent = r;
  table();
}

function nearestVal(arr) {
  const d = DATA[state.base], r = d.rounds[state.i];
  let k = -1;
  d.val_rounds.forEach((vr, j) => { if (vr <= r) k = j; });
  return k < 0 ? null : arr[k];
}
function table() {
  const d = DATA[state.base], i = state.i, dist = d.relative.distance;
  let h = `<tr><th>Strategy</th><th>Val acc</th><th>Dist. run</th><th>Dist. 1 step</th></tr>`;
  for (const s of ORDER) {
    const va = nearestVal(d.val_accuracy[s]);
    const dr = s === d.base ? 0 : dist[`${s}:real`]?.[i];
    const dc = dist[`${s}:cf`]?.[i];
    h += `<tr><td><span class="sw" style="border-color:${color(s, false)}"></span>${NAMES[s]}${s === d.base ? " (base)" : ""}</td>
      <td>${va == null ? "&ndash;" : va.toFixed(3)}</td><td>${dr == null ? "&ndash;" : dr.toFixed(2)}</td>
      <td>${dc == null || i === 0 ? "&ndash;" : dc.toFixed(2)}</td></tr>`;
  }
  $("table").innerHTML = h;
  let t = `<tr><th>Strategy</th><th>Accuracy</th><th>Balanced</th></tr>`;
  const ms = a => { const m = a.reduce((x, y) => x + y, 0) / a.length;
    const sd = Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); return `${m.toFixed(3)} &plusmn; ${sd.toFixed(3)}`; };
  for (const s of ORDER) {
    const runs = d.test[s];
    t += `<tr><td><span class="sw" style="border-color:${color(s, false)}"></span>${NAMES[s]}</td>
      <td>${ms(runs.map(x => x.accuracy))}</td><td>${ms(runs.map(x => x.balanced_accuracy))}</td></tr>`;
  }
  $("test").innerHTML = t;
  const ex = d[state.view].explained;
  $("explained").textContent = `The 2 components keep ${((ex[0] + ex[1]) * 100).toFixed(0)}% of the variance of this view; `
    + `the axes have different scales (PC1 spreads much more than PC2). Seeds: ${d.seeds.join(", ")}; `
    + `a point every ${d.rounds[1] - d.rounds[0]} rounds.`;
}

function legend() {
  const d = DATA[state.base];
  let h = "";
  for (const s of ORDER) {
    const key = s;
    h += `<label><input type="checkbox" data-k="${key}" ${state.hidden.has(key) ? "" : "checked"}>
      <span class="sw" style="border-color:${color(s, false)}"></span>${NAMES[s]}</label>`;
  }
  for (const s of d.counterfactuals) {
    const key = s + ":cf";
    h += `<label><input type="checkbox" data-k="${key}" ${state.hidden.has(key) ? "" : "checked"}>
      <span class="sw dash" style="border-color:${color(s, true)}"></span>${NAMES[s]} step from ${NAMES[d.base]}</label>`;
  }
  $("legend").innerHTML = h;
  $("legend").querySelectorAll("input").forEach(c => c.onchange = () => {
    c.checked ? state.hidden.delete(c.dataset.k) : state.hidden.add(c.dataset.k); draw();
  });
}

function seg(id, key, onset) {
  const box = $(id);
  const sync = () => box.querySelectorAll("button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.v === state[key])));
  box.querySelectorAll("button").forEach(b => b.onclick = () => {
    state[key] = b.dataset.v; sync(); if (onset) onset(); draw();
  });
  sync();
}

function setRound(i) {
  const n = DATA[state.base].rounds.length;
  state.i = Math.max(0, Math.min(n - 1, i));
  $("t").value = state.i;
  draw();
}

// --- init ----------------------------------------------------------------
$("base").innerHTML = Object.keys(DATA).map(b => `<button data-v="${b}">${NAMES[b] || b}</button>`).join("");
seg("base", "base", () => { $("t").max = DATA[state.base].rounds.length - 1; legend(); });
seg("view", "view");
seg("trail", "trail");
state.theme = document.documentElement.dataset.theme
  || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
seg("theme", "theme", () => { document.documentElement.dataset.theme = state.theme; });
$("t").max = DATA[state.base].rounds.length - 1;
$("t").oninput = e => setRound(+e.target.value);
svg.addEventListener("wheel", e => {
  e.preventDefault();
  const n = DATA[state.base].rounds.length, stepK = Math.max(1, Math.round(n / 200));
  setRound(state.i + Math.sign(e.deltaY || e.deltaX) * stepK);
}, {passive: false});
$("play").onclick = () => {
  if (state.playing) { clearInterval(state.playing); state.playing = null; $("play").innerHTML = "&#9654;"; return; }
  const n = DATA[state.base].rounds.length;
  if (state.i >= n - 1) state.i = 0;
  $("play").innerHTML = "&#10074;&#10074;";
  state.playing = setInterval(() => {
    if (state.i >= n - 1) { $("play").click(); return; }
    setRound(state.i + Math.max(1, Math.round(n / 250)));
  }, 30);
};
// hover: nearest point of any visible curve, up to the current round
const tip = $("tip");
svg.addEventListener("mousemove", e => {
  const box = svg.getBoundingClientRect(), k = W / box.width;
  const mx = (e.clientX - box.left) * k, my = (e.clientY - box.top) * k;
  const d = DATA[state.base];
  let best = null;
  for (const r of series()) {
    if (state.hidden.has(r.s + (r.cf ? ":cf" : ""))) continue;
    const last = state.trail === "ghost" ? r.pts.length - 1 : state.i;
    for (let j = r.cf ? 1 : 0; j <= last; j++) {
      const dx = scale.sx(r.pts[j][0]) - mx, dy = scale.sy(r.pts[j][1]) - my, dd = dx * dx + dy * dy;
      if (!best || dd < best.dd) best = {dd, r, j};
    }
  }
  if (!best || best.dd > 400) { tip.style.opacity = 0; return; }
  const [x, y] = best.r.pts[best.j];
  tip.innerHTML = `<b>${NAMES[best.r.s]}${best.r.cf ? ` step from ${NAMES[d.base]}` : ""}</b><br>`
    + `round ${d.rounds[best.j]}<br>PC1 ${x.toFixed(2)} &middot; PC2 ${y.toFixed(2)}`;
  tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 12}px`; tip.style.opacity = 1;
});
svg.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
legend();
setRound(DATA[state.base].rounds.length - 1);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
