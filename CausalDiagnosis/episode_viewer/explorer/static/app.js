"use strict";
/* Episode explorer. Reads data/data.json (what an agent could see, plus public run facts), the copies of the allowed
   files under data/files, and, only when the person switches the hidden side on, data/hidden.json and data/hidden_files,
   which the server sends only when it was started with --hidden. An exported page carries the same files inline. Every hidden value is rendered inside an element of the class HID.
   No name of an asset is written here: components, sensor sets, channels, prices and tools come from the data. */

const HID = "hidden-side";
const SVGNS = "http://www.w3.org/2000/svg";
const NR = "not recorded";
const KINDS = {
  reads: { label: "Reads and knowledge", glyph: "R" },
  advice: { label: "Advice", glyph: "A" },
  time: { label: "Time", glyph: "T" },
  orders: { label: "Work orders", glyph: "W" },
  decide: { label: "Diagnosis and finish", glyph: "D" },
};
const TABS = [["reasoning", "Timeline"], ["decision", "What the agent knew and did"], ["graph", "Graph"], ["readings", "Sensor readings"], ["raw", "Files"]];
const S = { levels: {}, hiddenLevels: {}, data: null, hidden: null, showHidden: false, tab: "decision", step: 1, graphMode: "decision",
            layout: { decision: "lanes", asset: "lanes" }, edgeLabels: false, off: { decision: new Set(["then"]), asset: new Set() },
            pos: {}, view: {}, selected: {}, ovFilter: { outcome: "all", fclass: "all" },
            rawAll: false, chainOnly: true, rsVariant: "A", rsAspect: "all", inspFace: "readable", face: {}, insp: null, sheetOpen: false, rdMode: "sensor", rdPlan: undefined, folds: {}, search: {}, lastStep: null };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

/* ---------- small helpers ---------- */
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  setAttrs(e, attrs);
  append(e, kids);
  return e;
}
function s(tag, attrs, ...kids) {
  const e = document.createElementNS(SVGNS, tag);
  setAttrs(e, attrs);
  append(e, kids);
  return e;
}
function setAttrs(e, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "class") e.setAttribute("class", Array.isArray(v) ? v.filter(Boolean).join(" ") : v);
    else e.setAttribute(k, v === true ? "" : v);
  }
}
function append(e, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}
const hid = (node) => { node.classList.add(HID); return node; };
const nr = (what) => h("span", { class: "nr" }, what ? `${what}: ${NR}` : NR);
function num(x, d = 3) {
  if (x === null || x === undefined || x === "") return NR;
  if (typeof x !== "number") return String(x);
  if (Number.isInteger(x)) return String(x);
  if (Math.abs(x) > 0 && Math.abs(x) < 0.001) return x < 0 ? "> -0.001" : "< 0.001";
  return x.toFixed(d).replace(/0+$/, "").replace(/\.$/, "");
}
const pct = (p) => (p === null || p === undefined ? NR : p > 0 && p < 0.001 ? "< 0.001" : p.toFixed(3));
const short = (field) => String(field).split("_")[0];
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const FOLLOW_WORDS = { followed: "followed the advice", deviated: "went against the advice", "no advice": "no advice" };
function kindDot(kind) {
  return h("span", { class: ["kdot", `k-${kind}`], "aria-hidden": "true" }, KINDS[kind] ? KINDS[kind].glyph : "?");
}

/* ---------- raw records: files, positions, the inspector ---------- */
/* Every statement the page derives stands one action away from the record it came from. The records are byte copies
   of the allowed files of the episode and run folders (explorer/data.py), read here as text, and a position is a key path
   located in that text by the parser below, so the inspector shows the file's own bytes, never a re-serialized copy. */
const FILES = {}, SPANS = {}, LINES = {};
/* the exported page carries its files inline in one JSON element; the served page fetches them */
const INLINE = (() => { const el = document.getElementById("explorer-data"); return el ? JSON.parse(el.textContent) : null; })();
function getText(url) {
  if (!INLINE) return fetch(url, { cache: "no-store" });
  const has = Object.prototype.hasOwnProperty.call(INLINE, url);
  return Promise.resolve({ ok: has, status: has ? 200 : 404, text: async () => (has ? INLINE[url] : ""), json: async () => JSON.parse(INLINE[url]) });
}
const TRAJ = "ep/trajectory.json", REL = "ep/released.json";
function isHiddenFile(path) { return /^ep\/hidden\/|^run\/(scores\.csv|ledger\.json)$/.test(path); }
function fileUrl(path) { return `data/${isHiddenFile(path) ? "hidden_files" : "files"}/${path}`; }
async function loadFiles(paths) {
  await Promise.all(paths.map(async (p) => {
    if (FILES[p] != null) return;
    for (let attempt = 0; attempt < 3 && FILES[p] == null; attempt++) {
      try { const r = await getText(fileUrl(p)); FILES[p] = r.ok ? await r.text() : null; if (r.status === 403 || r.status === 404) break; } catch (err) { FILES[p] = null; }
    }
  }));
}
/* start and end offset of every value, by its key path, in one pass over the text */
function spansOf(text) {
  const map = new Map();
  let i = 0;
  const fail = () => { throw new Error(`not JSON at offset ${i}`); };
  const ws = () => { while (i < text.length && (text[i] === " " || text[i] === "\n" || text[i] === "\r" || text[i] === "\t")) i++; };
  const str = () => {
    const s0 = i; i++;
    while (i < text.length && text[i] !== '"') i += text[i] === "\\" ? 2 : 1;
    if (i >= text.length) fail();
    i++;
    return JSON.parse(text.slice(s0, i));
  };
  const val = (path) => {
    ws();
    const s0 = i, ch = text[i];
    if (ch === "{") {
      i++; ws();
      if (text[i] === "}") i++;
      else for (;;) { ws(); if (text[i] !== '"') fail(); const k = str(); ws(); if (text[i] !== ":") fail(); i++; val([...path, k]); ws(); if (text[i] === ",") { i++; continue; } if (text[i] !== "}") fail(); i++; break; }
    } else if (ch === "[") {
      i++; ws();
      let n = 0;
      if (text[i] === "]") i++;
      else for (;;) { val([...path, n++]); ws(); if (text[i] === ",") { i++; continue; } if (text[i] !== "]") fail(); i++; break; }
    } else if (ch === '"') str();
    else { if (i >= text.length) fail(); while (i < text.length && !",]} \t\n\r".includes(text[i])) i++; }
    map.set(JSON.stringify(path), [s0, i]);
  };
  val([]);
  return map;
}
function lineStarts(text) { const a = [0]; for (let k = 0; k < text.length; k++) if (text[k] === "\n") a.push(k + 1); return a; }
function lineOf(starts, off) { let lo = 0, hi = starts.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (starts[m] <= off) lo = m; else hi = m - 1; } return lo; }
function linesOfFile(path) { const t = FILES[path]; return t == null ? null : (LINES[path] ||= lineStarts(t)); }
function lineCount(path) { const t = FILES[path], st = linesOfFile(path); return t == null ? 0 : st.length - (t.endsWith("\n") ? 1 : 0); }
function spansOfFile(path) { if (!(path in SPANS)) { try { SPANS[path] = spansOf(FILES[path]); } catch (err) { SPANS[path] = null; } } return SPANS[path]; }
function pathText(p) { return (p || []).map((k) => (typeof k === "number" ? `[${k}]` : `.${k}`)).join("") || "the whole file"; }
function fmtBytes(b) { return b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`; }
/* a ref: f file, p path of the record shown, m path marked inside it, s path of a JSON-string field to decode, q path
   marked inside the decoded string, l [first, last] line of a text file, dOnly to show the decoded string alone */
function resolveRef(ref) {
  const text = FILES[ref.f];
  if (text == null) return null;
  const starts = linesOfFile(ref.f);
  if (ref.l) {
    const a = starts[ref.l[0] - 1], b = ref.l[1] < starts.length ? starts[ref.l[1]] - 1 : text.length;
    return a == null ? null : { text, starts, start: a, end: b, mark: null, dec: null };
  }
  let base = 0, spans, p = ref.p || [], mp = ref.m, sp = ref.s;
  if (ref.f.endsWith(".jsonl")) {
    const li = p[0];
    if (li == null || li >= starts.length) return null;
    base = starts[li];
    const end = li + 1 < starts.length ? starts[li + 1] - 1 : text.length;
    try { spans = spansOf(text.slice(base, end)); } catch (err) { return null; }
    p = p.slice(1); mp = mp ? mp.slice(1) : null; sp = sp ? sp.slice(1) : null;
  } else spans = spansOfFile(ref.f);
  if (!spans) return null;
  const at = (q) => { const v = spans.get(JSON.stringify(q)); return v ? [v[0] + base, v[1] + base] : null; };
  const rec = at(p);
  if (!rec) return null;
  let dec = null;
  if (sp) {
    const sv = at(sp);
    try {
      const str = sv ? JSON.parse(text.slice(sv[0], sv[1])) : null;
      if (typeof str === "string") {
        let dm = null;
        if (ref.q) { try { dm = spansOf(str).get(JSON.stringify(ref.q)) || null; } catch (err) { dm = null; } }
        dec = { str, mark: dm };
      }
    } catch (err) { dec = null; }
  }
  return { text, starts, start: rec[0], end: ref.u && at(ref.u) ? at(ref.u)[1] : rec[1], mark: mp ? at(mp) : null, dec };
}
function copyButton(textFn) {
  const b = h("button", { type: "button", class: "mini" }, "Copy");
  b.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    try { await navigator.clipboard.writeText(textFn()); b.textContent = "Copied"; } catch (err) { b.textContent = "The browser refused to copy"; }
    setTimeout(() => { b.textContent = "Copy"; }, 1600);
  });
  return b;
}
function fileHref(path, line) {
  const route = parseRoute();
  if (route.view === "run") return `#/run/${encodeURIComponent(route.run)}/files/${encodeURIComponent(path)}${line ? `/${line}` : ""}`;
  return `#/ep/${encodeURIComponent(ep().run)}/${encodeURIComponent(ep().id)}/raw/${S.step}/${encodeURIComponent(path)}${line ? `/${line}` : ""}`;
}
function foldable(inner, big) {
  const box = h("div", { class: ["rr-fold", big ? "folded" : null] }, inner);
  if (!big) return box;
  const b = h("button", { type: "button", class: "mini rr-more" }, "Show all");
  b.addEventListener("click", (ev) => { ev.stopPropagation(); const f = box.classList.toggle("folded"); b.textContent = f ? "Show all" : "Show less"; });
  return h("div", {}, box, b);
}
function rrHead(ref, r) {
  const l0 = lineOf(r.starts, r.start), l1 = lineOf(r.starts, Math.max(r.start, r.end - 1));
  const rec = r.text.slice(r.start, r.end);
  const recPath = (p) => (p.length === 1 && typeof p[0] === "number" ? `record ${p[0] + 1}` : pathText(p));
  const where = ref.l ? `lines ${l0 + 1} to ${l1 + 1}` : `${ref.f.endsWith(".jsonl") ? `line ${ref.p[0] + 1}, ` : ""}${recPath(ref.f.endsWith(".jsonl") ? ref.p.slice(1) : ref.p)}${ref.f.endsWith(".jsonl") ? "" : `, ${l0 === l1 ? `line ${l0 + 1}` : `lines ${l0 + 1} to ${l1 + 1}`}`}`;
  return h("div", { class: "rr-head" },
    h("a", { class: "mono rr-file", href: fileHref(ref.f, l0 + 1), title: "Open this file at this line in the Files tab" }, ref.f.replace(/^ep\//, "").replace(/^run\//, "run/")),
    h("span", { class: "rr-pos" }, `${where} of ${lineCount(ref.f)}`),
    h("span", { class: "rr-size" }, fmtBytes(new TextEncoder().encode(rec).length)), copyButton(() => (r.dec && ref.dOnly ? r.dec.str : rec)));
}
/* the readable face of a record over its source: fields in words, the source one click away */
function recordView(ref) {
  if (!ref) return h("p", { class: "nr" }, NR);
  if (ref.inline !== undefined) return inlineView(ref);
  if ((S.inspFace || "readable") === "source") return rawView(ref);
  const r = resolveRef(ref);
  if (!r) return rawView(ref);
  return h("div", { class: "rawrec" }, rrHead(ref, r), readableRec(ref));
}
function inlineView(ref) {
  const text = ref.text != null ? ref.text : JSON.stringify(ref.inline, null, 2), st = lineStarts(text);
  return h("div", { class: "rawrec" }, h("div", { class: "rr-head" }, h("span", { class: "rr-pos" }, ref.title)),
    (S.inspFace || "readable") === "source" ? foldable(codeLines(text, st, 0, st.length - 1, null, null, 1, true), text.length > 1400)
      : h("div", { class: "rd-face" }, readableObject(ref.inline)));
}
function rawView(ref) {
  if (!ref) return h("p", { class: "nr" }, NR);
  if (ref.inline !== undefined) return inlineView(ref);
  const r = resolveRef(ref);
  if (!r) return h("div", { class: "rawrec" }, h("p", { class: "nr" }, FILES[ref.f] == null ? `${ref.f} is not shared with this page` : `${pathText(ref.p)} is not in ${ref.f}`));
  const l0 = lineOf(r.starts, r.start), l1 = lineOf(r.starts, Math.max(r.start, r.end - 1));
  const rec = r.text.slice(r.start, r.end);
  const isJson = /\.jsonl?$/.test(ref.f);
  const parts = [rrHead(ref, r)];
  if (!ref.dOnly || !r.dec) {
    const m = r.mark || [r.start, r.end];
    parts.push(foldable(codeLines(r.text, r.starts, l0, l1, m[0], m[1], l0 + 1, isJson), rec.length > 1400 || l1 - l0 > 18));
  }
  if (r.dec) {
    const ds = lineStarts(r.dec.str), dm = r.dec.mark;
    parts.push(h("div", { class: "rr-sub" }, "The answer as the agent received it, unpacked from the file."),
      foldable(codeLines(r.dec.str, ds, 0, ds.length - 1, dm ? dm[0] : null, dm ? dm[1] : null, 1, !ref.plain), r.dec.str.length > 1400 || ds.length > 18));
  }
  return h("div", { class: "rawrec" }, parts);
}
const rCall = (n, m) => ({ f: TRAJ, p: [n - 1], m: m ? [n - 1, ...m] : null });
const rAnswer = (n, q, full) => ({ f: TRAJ, p: [n - 1, "response"], s: [n - 1, "response"], q: q || null, dOnly: !full });
const rRules = (...k) => ({ f: REL, p: ["rules", ...k] });
function rTranscript(n) { const li = ep().trLine[n]; return li == null ? null : { f: "ep/transcript.jsonl", p: [li] }; }
function rModel(n) { const c = call(n); return c.llm_index == null ? null : { f: "ep/llm_calls.json", p: [c.llm_index], m: [c.llm_index, "reply"] }; }
function rObs(no, field) { const i = ep().observations.findIndex((o) => o.n === no); return i < 0 ? null : { f: REL, p: ["observations", i], m: field ? ["observations", i, "values", field] : null }; }
function rObsAnswer(no, field) {
  const cn = ep().obsCall[no];
  if (!cn) return null;
  const j = call(cn).served.findIndex((x) => x.n === no);
  return rAnswer(cn, ["result", "served", j, ...(field ? ["values", field] : [])]);
}
function rPred(id) { const i = ep().predictions.findIndex((p) => p.id === id); return i < 0 ? null : { f: "ep/predictions.json", p: [i] }; }
function rStatusBefore(n, field) {
  if (n > 1) return rAnswer(n - 1, [field]);
  const rule = { budget_left: "budget", steps_left: "max_steps" }[field];
  return rule ? rRules(rule) : null;
}
function stepRefs(n) { return [rCall(n, ["tool_args"]), rAnswer(n), rTranscript(n), rModel(n)].filter(Boolean); }
function showRaw(title, refs, note) { S.insp = { title, refs: refs.filter(Boolean), note }; S.sheetOpen = true; drawInspector(); }
function srcable(el, title, refs, note) {
  el.classList.add("src");
  if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "0");
  if (!el.title) el.title = "Show the record in the panel at the side";
  const go1 = (ev) => { ev.stopPropagation(); showRaw(title, typeof refs === "function" ? refs() : refs, note); };
  el.addEventListener("click", go1);
  el.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go1(ev); } });
  return el;
}
/* a number shown rounded keeps its value as the file holds it in its title */
function rv(x, shown) { return h("span", { class: "rnd", title: `exact value in the file ${x}` }, shown); }
function drawInspector() {
  const box = document.getElementById("raw-inspector");
  if (!box) return;
  const I = S.insp || { title: `Step ${S.step}, what the agent sent, what came back, and the model's message`, refs: stepRefs(S.step), auto: true };
  const views = I.refs.map((ref) => (isHiddenFile(ref.f) ? hid(h("div", { class: "hidden-box rr-hidden" }, h("div", { class: "hidden-label" }, "Ground truth record"), recordView(ref))) : recordView(ref)));
  const face = S.inspFace || "readable";
  const faceSw = h("div", { class: "seg small", role: "group", "aria-label": "Plain or raw records" }, [["readable", "Plain"], ["source", "Raw"]].map(([k, t]) =>
    h("button", { type: "button", "aria-pressed": String(face === k), onclick: () => { S.inspFace = k; drawInspector(); } }, t)));
  box.replaceChildren(...[
    h("div", { class: "insp-head" },
      h("h2", {}, "Record"), faceSw,
      h("button", { type: "button", class: "mini sheet-toggle", "aria-expanded": String(!!S.sheetOpen), onclick: () => { S.sheetOpen = !S.sheetOpen; drawInspector(); } }, S.sheetOpen ? "Close" : "Open")),
    h("div", { class: "insp-title" }, I.title,
      I.auto ? h("div", { class: "sub" }, "It shows whatever you click on the page, a step, a line of a card, a point of the graph or a cell of a table.")
        : h("button", { type: "button", class: "mini", onclick: () => { S.insp = null; drawInspector(); } }, "Back to the selected step")),
    I.note ? h("p", { class: "insp-note" }, I.note) : null,
    h("div", { class: "insp-body" }, views.length ? views : h("p", { class: "nr" }, "No file has a record for this."))].filter(Boolean));
  box.classList.toggle("open", !!S.sheetOpen);
}
let tipEl = null;
function showTip(ev, c) {
  if (!tipEl) { tipEl = h("div", { class: "tip", role: "tooltip" }); document.body.append(tipEl); }
  const ans = String(c.answer_text == null ? NR : c.answer_text);
  tipEl.replaceChildren(h("div", { class: "tip-h" }, `Step ${c.n}, ${KINDS[c.kind].label.toLowerCase()}`),
    h("div", { class: "tip-k" }, "sent"), h("div", { class: "tip-x mono" }, `${c.server}.${callText(c)}`),
    h("div", { class: "tip-k" }, "answer, first lines"), h("div", { class: "tip-x mono" }, ans.split("\n").slice(0, 3).join("\n").slice(0, 260) + (ans.length > 260 ? " …" : "")),
    h("div", { class: "tip-k" }, "click for the whole record"));
  const r = ev.target.getBoundingClientRect ? ev.target.getBoundingClientRect() : { left: ev.clientX, bottom: ev.clientY };
  tipEl.style.left = `${Math.min(window.innerWidth - 380, Math.max(8, r.left - 20))}px`;
  tipEl.style.top = `${r.bottom + 8}px`;
  tipEl.hidden = false;
}
function hideTip() { if (tipEl) tipEl.hidden = true; }

/* ---------- data shaping ---------- */
function ep() { return S.data.episode; }
function rules() { return ep().rules; }
function comps() { return rules().components || []; }
function combText(c, full) {
  const r = rules();
  const get = (k, i) => (Array.isArray(c) ? c[i] : c[k]);
  const parts = comps().map((k, i) => [k, get(k, i)]);
  const faulty = parts.filter(([k, v]) => v !== r.healthy[k]);
  if (full) return parts.map(([k, v]) => `${k} ${v}`).join(", ");
  return faulty.length ? andList(faulty.map(([k, v]) => `${k} at level ${v}`)) : "no faulty part";
}
function sameComb(a, b) {
  return comps().every((k, i) => (Array.isArray(a) ? a[i] : a[k]) === (Array.isArray(b) ? b[i] : b[k]));
}
// the normal ranges of the rules: "normal_ranges", or "healthy_bands" where the rules carry that key
function bandKey() { return rules().normal_ranges ? "normal_ranges" : "healthy_bands"; }
function bandOf(field) { return (rules()[bandKey()] || {})[field] || null; }
function outside(field, v) {
  const b = bandOf(field);
  return b && typeof v === "number" ? v < b.low || v > b.high : false;
}
function prepare() {
  const e = ep();
  const obsByN = Object.fromEntries((e.observations || []).map((o) => [o.n, o]));
  for (const c of e.calls) {
    const res = c.answer && c.answer.result;
    c.served = res && Array.isArray(res.served) ? res.served.map((x) => ({ ...x, ...(obsByN[x.n] || {}) })) : null;
    c.plan = e.plan[c.n - 1] || null;
  }
  e.obsCall = {};
  for (const c of e.calls) for (const o of c.served || []) e.obsCall[o.n] = c.n;
  e.rankByN = Object.fromEntries(e.rankings.map((r) => [r.n, r]));
  e.closes = [];
  for (const c of e.calls) for (const n of c.notifications || []) if (n.event === "work_order_closed") e.closes.push({ t: n.t, call: c.n, id: n.id, component: n.component });
  // a work order of AssetOpsBench's servers names its number and the child asset of a component, and the cycle of its call
  // stands for the cycle it was created in
  e.orders = e.calls.filter((c) => c.answer && c.answer.result && c.answer.result.work_order)
    .map((c) => { const w = c.answer.result.work_order; return { ...w, id: w.id || w.wonum, component: w.component || (e.asset_children || {})[w.assetnum], created_t: w.created_t ?? c.time_before, call: c.n }; });
  e.endTime = e.status ? e.status.time : e.calls.length ? e.calls[e.calls.length - 1].time_after : 0;
  // the line of the transcript that records each call, in call order
  e.trLine = {};
  let k = 0;
  (FILES["ep/transcript.jsonl"] || "").split("\n").forEach((ln, i) => {
    try { if (JSON.parse(ln).kind === "call") e.trLine[++k] = i; } catch (err) { /* a blank last line */ }
  });
}
const call = (n) => ep().calls[n - 1];
function obsBefore(n) { return ep().calls.filter((c) => c.n < n).flatMap((c) => c.served || []); }
function lastRanking(n) { const r = ep().rankings.filter((x) => x.n <= n); return r.length ? r[r.length - 1] : null; }
// a run is its folder, which the server routes by; the label of its run.json answers only where no folder has the name
function runOf(name) { return S.data.runs.find((r) => r.folder === name) || S.data.runs.find((r) => r.label === name) || null; }
function stopwatch(c) { return c.duration_ms == null ? NR : `${(c.duration_ms / 1000).toFixed(1)} s`; }
function callText(c) {
  const args = Object.entries(c.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  return `${c.tool}(${args})`;
}
function plainTask(c) { return (c.plan && c.plan.Task) || c.task || c.tool; }

/* ---------- routing and frame ---------- */
function parseRoute() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent).filter(Boolean);
  if (parts[0] === "run" && parts[1]) return { view: "run", run: parts[1], file: parts[2] === "files" ? parts[3] : null, line: +parts[4] || null };
  if (parts[0] === "figure" && parts[2]) return { view: "figure", run: parts[1], ep: parts[2], step: +parts[3] || 1 };
  if (parts[0] === "ep" && parts[2]) return { view: "ep", run: parts[1], ep: parts[2], tab: parts[3], step: +parts[4] || 1, file: parts[5] || null, line: +parts[6] || null };
  return { view: "runs" };
}
function go(hash) { if (location.hash !== hash) location.hash = hash; else render(); }
function epHash(tab = S.tab, step = S.step) { return `#/ep/${encodeURIComponent(ep().run)}/${encodeURIComponent(ep().id)}/${tab}/${step}`; }
function crumbs(route) {
  const c = document.getElementById("crumbs");
  c.replaceChildren();
  const link = (t, href) => h("a", { href }, t);
  const sep = () => " / ";
  c.append(link("Runs", "#/"));
  if (route.run) append(c, [sep(), route.view === "run" ? h("span", {}, route.run) : link(route.run, `#/run/${encodeURIComponent(route.run)}`)]);
  if (route.view === "ep") append(c, [sep(), h("span", {}, route.ep)]);
}
function render() {
  if (!S.prepared) return;
  // a page opened without a hash is the page of its own episode, so it opens on the timeline of the first step
  if (!location.hash) history.replaceState(null, "", epHash("reasoning", 1));
  const route = parseRoute();
  crumbs(route);
  const app = document.getElementById("app");
  app.replaceChildren();
  document.body.classList.toggle("figure-page", route.view === "figure");
  if (route.view === "figure") {
    if (route.run !== ep().run || route.ep !== ep().id) { app.append(viewMissing(route)); return; }
    S.step = Math.min(Math.max(1, route.step), ep().calls.length);
    app.append(viewFigure());
    return;
  }
  if (route.view === "runs") app.append(viewRuns());
  else if (route.view === "run") app.append(viewRun(route.run, route));
  else {
    if (route.run !== ep().run || route.ep !== ep().id) { app.append(viewMissing(route)); return; }
    S.tab = TABS.some(([k]) => k === route.tab) ? route.tab : "reasoning";
    S.step = Math.min(Math.max(1, route.step), ep().calls.length);
    if (S.lastStep !== S.step) { S.insp = null; S.lastStep = S.step; }
    app.append(viewEpisode());
    afterEpisode();
  }
}
function viewMissing(route) {
  // each episode has a page of its own; the served page opens it, an exported page holds one episode alone
  if (!INLINE) { location.href = `../../${encodeURIComponent(route.run)}/${encodeURIComponent(route.ep)}/${location.hash}`; return h("p", { class: "loading" }, "Opening the episode."); }
  return h("div", { class: "panel panel-b" }, h("h1", { class: "page-title" }, "Episode not in this file"),
    h("p", {}, `This file holds one episode. ${route.ep} of ${route.run} is shown by the explorer that python -m CausalDiagnosis episode_viewer explorer serve starts.`));
}

/* ---------- Runs ---------- */
function viewRuns() {
  const rows = S.data.runs.map((r) => {
    const played = r.played.length;
    return h("tr", { class: "link", tabindex: "0", onclick: () => go(`#/run/${encodeURIComponent(r.folder)}`),
                     onkeydown: (e) => { if (e.key === "Enter") go(`#/run/${encodeURIComponent(r.folder)}`); } },
      h("td", {}, h("b", { class: "mono" }, r.label)), h("td", {}, r.agent_label || r.agent),
      h("td", {}, r.config_name), h("td", {}, r.clock_mode), h("td", {}, String(r.guidance_level)),
      h("td", {}, r.diagnostics_server ? "on" : "off"), h("td", {}, `${played} of ${r.episodes_planned}`),
      h("td", {}, r.started || NR));
  });
  return h("section", {},
    h("h1", { class: "page-title" }, "Runs"),
    h("p", { class: "page-lead" }, "Every run in the folder this viewer was started on. Open a run to see its episodes."),
    h("div", { class: "panel table-wrap" }, h("table", { class: "grid" },
      h("thead", {}, h("tr", {}, ["Run", "Agent", "Setting", "How time passes", "Hints given to the agent", "CausalDiagnosis MCP", "Episodes played", "Started"].map((t) => h("th", { scope: "col" }, t)))),
      h("tbody", {}, rows))));
}

/* ---------- Run overview ---------- */
function hiddenGrid(r, folder, bySeed) {
  const box = hid(h("div", { class: "panel" }));
  const H = S.hidden;
  if (!H) { box.append(h("div", { class: "panel-b" }, "Loading the ground truth.")); return box; }
  const rows = (H.score_rows || []).map((x) => {
    const block = JSON.parse(x.block || "null");
    const f = bySeed[+x.seed];
    return { seed: +x.seed, folder: f, comb: block, offset: x.entry_offset === "" ? null : +x.entry_offset,
             pass: x.pass === "True", fclass: x.failure_class || "none" };
  });
  const offsets = [...new Set(rows.map((x) => x.offset))].sort((a, b) => a - b);
  const classes = [...new Set(rows.map((x) => x.fclass))];
  const f = S.ovFilter;
  const keep = (x) => (f.outcome === "all" || (f.outcome === "pass" ? x && x.pass : f.outcome === "fail" ? x && !x.pass : f.outcome === "unplayed" ? !x : !!x))
    && (f.fclass === "all" || (x && x.fclass === f.fclass));
  const nFaulty = (c) => comps().filter((k, i) => c[i] !== rules().healthy[k]).length;
  const states = [...S.data.start_states].sort((a, b) => nFaulty(a) - nFaulty(b));
  const tiles = states.map((comb) => {
    const here = rows.filter((x) => x.comb && sameComb(x.comb, comb));
    const anyKeep = offsets.some((o) => keep(here.find((x) => x.offset === o))) || (!offsets.length && keep(null));
    return h("div", { class: ["cell", here.length ? "played" : "empty", here.some((x) => x.pass) ? "pass" : here.length ? "fail" : null], style: null,
                      "data-dim": anyKeep ? null : "1" },
      h("span", { class: "c-top" }, combText(comb)),
      here.length ? h("span", { class: "c-sub" }, here.map((x) => `seed ${x.seed}, ${x.pass ? "passed" : `failed, ${x.fclass.replace(/_/g, " ")}`}`).join(". ")) : null,
      h("div", { class: "offset-row" }, offsets.map((o) => {
        const x = here.find((y) => y.offset === o);
        if (!x) return h("span", { class: "offset-cell", title: `start point ${o}, not played` }, `start ${o}`);
        return h("button", { type: "button", class: ["offset-cell", x.pass ? "pass" : "fail"],
          title: `start point ${o}, seed ${x.seed}, ${x.pass ? "passed" : `failed, ${x.fclass.replace(/_/g, " ")}`}`,
          onclick: () => go(`#/ep/${encodeURIComponent(folder)}/${encodeURIComponent(x.folder)}/decision/1`) },
          x.pass ? "✓ passed" : "✗ failed", " ", h("span", { class: "fc" }, `start ${o}`));
      })));
  });
  for (const t of tiles) if (t.getAttribute("data-dim")) t.style.opacity = "0.25";
  const sel = (key, opts) => h("select", { onchange: (e) => { S.ovFilter[key] = e.target.value; render(); } },
    opts.map(([v, t]) => h("option", { value: v, selected: S.ovFilter[key] === v ? "" : null }, t)));
  box.append(
    h("div", { class: "panel-h" }, h("h2", {}, h("span", { class: "hidden-label" }, "Ground truth "), "The episodes by the fault they started with and the point of the recording they started at"),
      h("span", { class: "sub" }, `${S.data.start_states.length} starting faults this setting allows, ${rows.length} played`)),
    h("div", { class: "panel-b" },
      h("div", { class: "filters" }, h("label", {}, "Score ", sel("outcome", [["all", "all"], ["pass", "passed"], ["fail", "failed"], ["played", "played"], ["unplayed", "not played"]])),
        h("label", {}, "Why it failed ", sel("fclass", [["all", "all"], ...classes.map((c) => [c, c.replace(/_/g, " ")])])),
        h("span", { class: "sub" }, "Each small box is one start point in the recording, with ✓ passed or ✗ failed. Point at a box for the reason it failed. A dashed box was not played.")),
      h("div", { class: "ov-grid" }, tiles),
      h("p", { class: "note" }, `The played episodes started at ${andList(offsets.map(String)) || "no recorded point"}. The run files do not say which other start points a full run would play, so only these are drawn.`)));
  return box;
}

/* ---------- Episode ---------- */
/* ---------- Decision tab ---------- */
function tabDecision(c) {
  const idle = idlePicked();
  if (idle) return idleStory(idle);
  return h("div", {}, storyLine(c), h("div", { class: "cards" }, cardKnown(c), cardAdvised(c), cardChosen(c), cardResult(c)));
}
/* the line of a no_call step: the turn called no tool, so it knew what the step before it knew */
function idleStory(x) {
  const before = obsBefore(x.after + 1).length;
  const step = (k, t) => h("li", {}, h("b", {}, k), " ", t);
  return h("div", {}, h("ol", { class: "story", "aria-label": "This step in one line" },
    step("Knew", before ? plural(before, "reading") : "no reading yet"), step("Advice", "no advice at this step"),
    step("Did", "called no tool"), step("Got back", "nothing, one step used")),
    h("p", { class: "note" }, `${x.after ? `This turn came after step ${x.after}` : "This turn came before the first step"}, at cycle ${x.time}. ${IDLE_TIP}`));
}
/* one line a newcomer reads first: what was known, advised, chosen and what came back */
function storyLine(c) {
  const before = obsBefore(c.n).length;
  const r = c.advice_in_force && c.kind !== "advice" ? ep().rankByN[c.advice_in_force] : null;
  const res = resultSummary(c)[0];
  const step = (k, t) => h("li", {}, h("b", {}, k), " ", t);
  return h("ol", { class: "story", "aria-label": "This step in one line" },
    step("Knew", before ? plural(before, "reading") : "no reading yet"),
    step("Advice", c.kind === "advice" ? "this step asks for advice" : r ? advisedAction(r).replace(/ \(.*$/, "") : "no advice at this step"),
    step("Did", `${plainTask(c)}${c.follow === "followed" || c.follow === "deviated" ? `, ${FOLLOW_WORDS[c.follow]}` : ""}`),
    step("Got back", res ? res.textContent : NR));
}
function advisedAction(r) {
  const tb = r.top_branch || {};
  if (r.read_first && r.discriminating_test) {
    const t = r.discriminating_test;
    return `Read the ${t.sensor_group || JSON.stringify(t)} sensors${t.cycles ? ` for ${count(t.cycles, "more cycle")}` : ""} first`;
  }
  const plan = tb.repair_plan;
  if (plan && plan.steps) return `Repair the ${andList(plan.steps.map((x) => x.component))} now (costs ${num(plan.price)}, machine working again after ${plural(plan.cycles_to_restored, "cycle")})`;
  return NR;
}
const SERVER_WORDS = { causal_diagnosis: "the CausalDiagnosis MCP", hydraulic_rig: "the machine" };
function barFill(p) { const i = h("i"); i.style.width = `${Math.max(0, Math.min(1, p || 0)) * 100}%`; return i; }
function followReason(c) {
  const r = c.advice_in_force ? ep().rankByN[c.advice_in_force] : null;
  if (!r || !c.follow || c.follow === "no advice") return null;
  const top = r.candidates[0];
  if (c.kind === "decide") return `The agent named ${causeWords(c.args)}, and the most likely cause was ${causeWords(top.combination)}.`;
  if (c.kind === "orders") return `The agent ordered a repair of the ${c.args.component}, and the advice was to repair ${andList(((r.top_branch.repair_plan || {}).steps || []).map((s) => `the ${s.component}`)) || "no part"}.`;
  if (c.kind === "reads") return r.discriminating_test ? `The advice was to read the ${r.discriminating_test.sensor_group} sensors.` : "The advice was to repair now, with no more reading.";
  return null;
}
// the answer of one call to the CausalDiagnosis MCP in plain words, from the fields of the answer; the machine message of
// the answer stays in the record. A tool without a sentence here keeps its message
const upFirst = (s) => s.replace(/^./, (x) => x.toUpperCase());
const pctWords = (p) => (p == null ? NR : p > 0 && p < 0.005 ? "below 1%" : p < 1 && p > 0.995 ? "above 99%" : `${Math.round(p * 100)}%`);
function adviceWords(c) {
  const a = c.answer || {};
  if (c.tool === "get_episode_evidence" && Array.isArray(a.items)) {
    const means = a.items.filter((x) => x.statistic === "mean" && typeof x.in_normal_range === "boolean"), out = means.filter((x) => !x.in_normal_range).length;
    const closed = Array.isArray(a.work_orders_closed) ? a.work_orders_closed.length : 0;
    return `${plural(a.items.length, "value")} from ${plural(a.n_observations || 0, "reading")}, ${out} of ${means.length} sensor averages outside the normal range${closed ? `, ${plural(closed, "repair")} done` : ""}`;
  }
  if (c.tool === "get_episode_evidence" && Array.isArray(a.fields)) {
    // a shorter level writes one long table, a value outside the normal range of its field followed by low or high
    const out = a.fields.flatMap((f) => f.values || []).filter((v) => OUT_MARK.test(v || "")).length;
    const closed = Array.isArray(a.work_orders_closed) ? a.work_orders_closed.length : 0;
    return `${plural((a.readings || []).length, "reading")}, ${plural(out, "value")} outside the normal range${closed ? `, ${plural(closed, "repair")} done` : ""}`;
  }
  if (c.tool === "rank_root_causes" && Array.isArray(a.candidates) && a.candidates.length) {
    const top = a.candidates[0], k = Math.round(a.effective_hypotheses), n0 = nCauses(a), adv = a.advice || {};
    const left = k <= 1 ? `the other ${plural(n0 - 1, "cause")} are ruled out` : `${k} of ${n0} causes are still possible`;
    const next = adv.action === "read" ? `It advises reading ${groupWords(adv.sensor_group)} for ${count(adv.cycles, "more cycle")} before a diagnosis.` : adv.text ? `Its advice is to ${adv.text}.` : "";
    return `${upFirst(causeWords(top.combination))} is the most likely cause, chance ${pctWords(top.probability)}, and ${left}. ${next}`.trim();
  }
  if (c.tool === "estimate_intervention_effect" && a.action) {
    const parts = andList(String(a.action).replace(/^[^:]*:/, "").split(",").map((x) => `the ${x}`));
    return `A repair of ${parts} would fix a fault with chance ${pctWords(a.addresses_fault)}, at price ${num(a.price)} and ${plural(a.duration, "cycle")}`;
  }
  if (c.tool === "verify_outcome") { const row = checkRow(a); if (row) return upFirst(row[2]); }
  return a.message || a.headline || NR;
}
function resultSummary(c) {
  const a = c.answer || {}, res = a.result;
  if (c.refused) return [h("p", { class: "lead result-refused" }, `Refused because ${c.refused}`)];
  if (c.error) return [h("p", { class: "lead result-refused" }, `Failed with ${c.error}`)];
  if (c.served) {
    if (!c.served.length) return [h("p", { class: "lead" }, "The read returned nothing"), h("p", { class: "note" }, res.stopped ? `The machine was stopped for ${plural(res.stopped, "cycle")} of the read, so nothing was recorded.` : "The read returned nothing.")];
    return [h("p", { class: "lead" }, `${plural(c.served.length, "cycle")} of the ${res.sensor_group} sensors came back`), h("ul", {}, c.served.map(obsLine))];
  }
  if (res && res.work_order) { const w = res.work_order; return [h("p", { class: "lead" }, `Work order ${w.id} for the ${w.component} opened, the repair should be done at cycle ${w.expected_close_t}`)]; }
  if (res && res.diagnosis) return [h("p", { class: "lead" }, `The diagnosis was recorded, ${causeWords(res.diagnosis)}`), h("p", { class: "note" }, `evidence cited ${(res.evidence_ids || []).length ? res.evidence_ids.join(", ") : "none"}`)];
  if (c.kind === "advice") return [h("p", { class: "lead" }, adviceWords(c))];
  if (res && "waited" in res) return [h("p", { class: "lead" }, `Waited ${plural(res.waited, "cycle")}, to cycle ${res.time}, until ${String(res.ended_by).replace(/_/g, " ")}`)];
  if (res && typeof res === "object") return [h("dl", { class: "kv" }, Object.entries(res).slice(0, 8).map(([k, v]) => [h("dt", {}, k), h("dd", {}, summarize(v))]))];
  return [h("p", { class: "lead" }, a.text || NR)];
}
// the levels of response_format (mcp_servers/utils/response_levels.py): the agent received the answer at its level
const LEVEL_WORDS = { headline: "headline, one line", concise: "concise, what an agent acts on", detailed: "detailed, every value in compact form", raw: "raw, the whole answer" };
function levelWords(c) {
  const a = c.answer || {}, again = a.unchanged || (a.result || {}).unchanged;
  return `${LEVEL_WORDS[c.level] || c.level}${(c.args || {}).response_format ? "" : ", the default"}${again ? `, ${again}` : ""}`;
}
// every answer of the CausalDiagnosis MCP at the four levels (data.write_levels): the level the agent received, marked, and the
// three others, which the page wrote from the raw answer with the level code of the server, each opening in the record panel
const LEVELS = ["headline", "concise", "detailed", "raw"];
function levelRow(c) {
  const recomputed = !S.levels[c.n] && S.showHidden && S.hiddenLevels[c.n];
  const L = S.levels[c.n] || recomputed;
  const open = (lv) => showRaw(`Step ${c.n}, the answer at the level ${lv}`, [{ inline: JSON.parse(L.texts[lv]), text: L.texts[lv],
    title: lv === L.seen ? `the answer at the level ${lv}, as the agent received it` : `the answer at the level ${lv}, written by the page` }], levelNote(lv, L));
  return [h("dt", {}, "level of the answer"), h("dd", {}, srcable(h("span", {}, levelWords(c)), `Level of the answer of step ${c.n}`, [rCall(c.n, ["tool_args"]), rAnswer(c.n)]),
    L ? (recomputed ? hid : (x) => x)(h("div", { class: "lv-seg", role: "group", "aria-label": "The answer at each level" }, LEVELS.map((lv) => h("button", { type: "button", class: ["mini", lv === L.seen ? "lv-seen" : null],
      title: lv === L.seen ? "the level the agent received" : "written by the page from the raw answer", onclick: () => open(lv) }, lv === L.seen ? `${lv}, seen` : lv))))
      : c.levels ? h("div", { class: "note" }, levelMissing(c)) : null)];
}
function levelNote(lv, L) {
  if (lv === L.seen) return "The agent received the answer at this level.";
  const from = L.source === "recomputed" ? "the raw answer that the page recomputed with the hidden option, since the run kept none"
    : L.seen === "raw" ? "the raw answer that the agent received" : "the raw answer that the CausalDiagnosis MCP kept beside the call";
  return `The page wrote this level from ${from}, with the level code of the CausalDiagnosis MCP. The agent did not see it.`;
}
function levelMissing(c) {
  if (c.levels === "no profile") return "The asset of this run is not registered on this machine, so the page cannot write the other levels.";
  if (S.showHidden && S.hidden && (S.hidden.levels || {})[c.n] === "not recomputed") return "The run kept no raw answer of this call, and the replay with the hidden option did not give back the answer the agent received, so the page shows that level alone.";
  return S.hiddenServed ? "The run kept no raw answer of this call. With the ground truth shown, the page recomputes it from the hidden start." : "The run kept no raw answer of this call, so the page shows the level the agent received alone. A viewer started with the hidden option recomputes it.";
}
// a field that the level of a ranking leaves out (tools/answer_levels.py) is left out of that answer, not unrecorded
function atLevel(r, key, shown) {
  const c = call(r.n), a = (c && c.answer) || {};
  return key in a || !c || !c.level || c.level === "raw" ? shown : h("span", { class: "nr" }, `not in the ${c.level} answer`);
}
const OUT_MARK = /\s(low|high)$/;
function summarize(v) {
  if (Array.isArray(v)) return `${v.length} item${v.length === 1 ? "" : "s"}${v.length && typeof v[0] !== "object" ? `: ${v.slice(0, 6).join(", ")}` : ""}`;
  if (v && typeof v === "object") { const t = JSON.stringify(v); return t.length > 110 ? `${t.slice(0, 108)}…` : t; }
  return String(v);
}
/* ---------- Raw tab ---------- */
/* ---------- Run overview ---------- */
function runKeyLine(key) {
  const sp = spansOfFile("run/run.json"), v = sp && sp.get(JSON.stringify([key]));
  return v ? lineOf(linesOfFile("run/run.json"), v[0]) + 1 : null;
}
function runGroups() {
  const E = ep(), Hd = S.hidden && S.hidden.files ? S.hidden.files : null;
  return [
    { label: `Run folder ${E.run}`, files: E.files.run.map((f) => ({ ...f, path: `run/${f.name}` })) },
    { label: "Run folder, ground truth", hidden: true, files: Hd ? Hd.run.map((f) => ({ ...f, path: `run/${f.name}` })) : [] },
  ];
}
function viewRun(name, route) {
  const r = runOf(name);
  if (!r) return h("p", {}, `No run named ${name} in this folder.`);
  const folder = r.folder;
  const bySeed = {};
  for (const f of r.played) bySeed[parseInt(f.replace(/\D/g, ""), 10)] = f;
  const rowBySeed = Object.fromEntries(r.rows.map((x) => [+x.seed, x]));
  const seeds = r.seeds || [];
  const runLink = (k) => `#/run/${encodeURIComponent(folder)}/files/${encodeURIComponent("run/run.json")}/${runKeyLine(k) || 1}`;
  const publicCells = seeds.map((sd) => {
    const f = bySeed[sd], row = rowBySeed[sd];
    if (!f) return h("div", { class: "cell empty" }, h("span", { class: "c-top" }, `seed ${sd}`), h("span", {}, "not played"));
    const here = folder === ep().run && f === ep().id;
    const calls = here ? ep().calls.length : row ? (+row.tool_calls || 0) + (+row.causal_calls || 0) : null;
    return h("div", { class: "cell played" },
      h("a", { class: "c-top", href: `#/ep/${encodeURIComponent(folder)}/${encodeURIComponent(f)}/decision/1` }, f),
      h("span", { class: "c-sub" }, `agent ${r.agent_label || r.agent}, setting ${r.config_name}`),
      calls == null ? h("span", { class: "c-sub" }, NR) : here
        ? h("a", { class: "c-sub", href: `#/ep/${encodeURIComponent(folder)}/${encodeURIComponent(f)}/raw/1/${encodeURIComponent(TRAJ)}/1`, title: "Open the file this count comes from, one record per step" }, plural(calls, "step"))
        : h("span", { class: "c-sub" }, plural(calls, "step")));
  });
  const counted = h("ul", { class: "counted" }, r.played.map((f) => h("li", {}, h("a", { href: `#/ep/${encodeURIComponent(folder)}/${encodeURIComponent(f)}/decision/1` }, f), ", one folder per episode played")),
    h("li", {}, h("a", { href: runLink("episodes_planned") }, "the planned count, from run.json")));
  const panel = h("div", { class: "panel" },
    h("div", { class: "panel-h" }, h("h2", {}, "Episodes"),
      h("details", { class: "agg" }, h("summary", {}, `${r.played.length} of ${r.episodes_planned} planned episodes played`), counted)),
    h("div", { class: "panel-b" }, h("div", { class: "ov-grid" }, publicCells),
      h("p", { class: "note" }, "Which fault each episode started with, and its score, are part of the ground truth. Switch the ground truth on to see them.")));
  const fact = (k, key, v) => h("a", { href: runLink(key), class: "fact-link", title: `${key} in run.json` }, `${k} `, h("b", {}, v == null ? NR : String(v)));
  const files = h("div", { class: "panel" }, h("div", { class: "panel-h" }, h("h2", {}, "Run files"), h("span", { class: "sub" }, "raw, with line numbers; every fact above links to its line")),
    h("div", { class: "panel-b" }, fileBrowser(runGroups(), route.file, route.line)));
  return h("section", {},
    h("h1", { class: "page-title mono" }, r.label),
    h("div", { class: "facts" }, [["agent", "agent_label", r.agent_label || r.agent], ["setting", "config_name", r.config_name], ["prices", "prices", r.prices],
      ["how time passes", "clock_mode", r.clock_mode], ["hints given", "guidance_level", r.guidance_level], ["CausalDiagnosis MCP", "diagnostics_server", r.diagnostics_server ? "on" : "off"],
      ["model the CausalDiagnosis MCP ranks with", "ranking_model", r.ranking_model], ["what the score checks", "verifier_scope", r.verifier_scope], ["how start states were picked", "starts", r.starts], ["started", "started", r.started], ["ended", "ended", r.ended]]
      .map(([k, key, v]) => fact(k, key, v))),
    servedBlock(folder),
    h("div", { class: "ov-layout" }, panel, hiddenGrid(r, folder, bySeed)), files);
}
// how the open episode was served, from harness.json and episode.json: shown only for a run that flushed the model's
// cache or checked the model at a cut, and folded
function servedBlock(folder) {
  const e = ep(), sv = e.served || {}, fl = sv.cache_flush, pr = sv.cut_probe;
  if (folder !== e.run || (!fl && !pr)) return null;
  const okStatus = (x) => x != null && x >= 200 && x < 300;
  const rows = [
    ["model cache emptied before the episode", fl ? (okStatus(fl.status) ? "yes, the server answered OK" : `no, the server answered ${fl.status == null ? "nothing" : fl.status}`) : "not emptied"],
    ["cached tokens of the first request", fl && fl.first_request_cached_tokens != null ? String(fl.first_request_cached_tokens) : NR],
    ["check of the model at a cut", pr ? (pr.ok ? `${plural(pr.cut_probes, "token")} generated, answered in ${secs(pr.seconds)}` : `no token generated${pr.error ? ` (${pr.error})` : ""}`) : sv.wall_limit ? "none was made" : "no cut"],
    ["episode ended at", sv.episode_ended_at ? new Date(sv.episode_ended_at * 1000).toLocaleString() : NR]];
  return h("details", { class: "panel served" }, h("summary", {}, "How this run was served"),
    h("dl", { class: "kv" }, rows.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    h("p", { class: "note" }, `For episode ${e.id}. The same facts are in harness.json and episode.json in the Files tab.`));
}

/* ---------- Episode ---------- */
function viewEpisode() {
  const e = ep(), r = rules(), run = runOf(e.run) || {};
  const c = call(S.step);
  const fact = (k, v, ref) => { const el = h("span", {}, `${k} `, h("b", {}, String(v))); return ref ? srcable(el, k, [ref]) : el; };
  const head = h("div", { class: "ep-head" },
    h("div", {},
      h("h1", { class: "ep-title" }, `Episode ${e.id} `, h("span", { class: "mono" }, `seed ${e.seed}`)),
      h("div", { class: "facts" }, fact("agent", run.agent_label || run.agent, { f: "run/run.json", p: ["agent_label"] }),
        fact("setting", run.config_name, { f: "run/run.json", p: ["config_name"] }), fact("how time passes", r.clock_mode, rRules("clock_mode")),
        fact("budget", r.budget ?? "none", rRules("budget")), fact("time limit", `${r.horizon} cycles`, rRules("horizon")), fact("step limit", r.max_steps, rRules("max_steps")),
        fact("steps", e.calls.length, { f: TRAJ, p: [] }),
        fact("ended", e.status ? `at cycle ${e.status.time}${e.status.budget_left != null ? ` with ${e.status.budget_left} of the budget left` : ""}` : NR, e.status ? { f: REL, p: ["status"] } : null))),
    h("details", { class: "task" }, h("summary", {}, "Task description the agent received"), h("pre", {}, e.task_description || NR)));
  // the page top: the step list beside the calls and spending lanes (navStrip)
  const tabList = TABS.filter(([k]) => k !== "reasoning");
  if (!tabList.some(([k]) => k === S.tab)) S.tab = tabList[0][0];
  const M = rankingCalls().length ? storyModel() : null;
  const strip = navStrip(M);
  const tabs = h("div", { class: "tabs", role: "tablist", "aria-label": "Views of the step" },
    tabList.map(([k, t], i) => h("button", { role: "tab", id: `tab-${k}`, "aria-selected": String(S.tab === k), "aria-controls": "tabpanel",
      tabindex: S.tab === k ? "0" : "-1", onclick: () => go(epHash(k)) }, t, h("span", { class: "key" }, String(i + 1)))));
  const route = parseRoute();
  const main = h("section", { class: "panel" },
    h("div", { class: "stepbar" },
      h("div", { class: "step-title" }, kindDot(c.kind), `Step ${c.n} of ${e.calls.length}`, h("code", {}, c.tool),
        h("span", { class: "sub" }, `cycle ${c.time_before}${c.cycles ? ` → ${c.time_after}` : ""}`)),
      h("button", { class: "btn", type: "button", disabled: S.step <= 1 ? "" : null, onclick: () => go(epHash(S.tab, S.step - 1)) }, "← Previous"),
      h("button", { class: "btn", type: "button", disabled: S.step >= e.calls.length ? "" : null, onclick: () => go(epHash(S.tab, S.step + 1)) }, "Next →"),
      null),
    tabs,
    h("div", { class: "tabpanel", id: "tabpanel", role: "tabpanel", "aria-labelledby": `tab-${S.tab}` },
      S.tab === "decision" ? tabDecision(c) : S.tab === "reasoning" ? tabReasoning() : S.tab === "graph" ? tabGraph() : S.tab === "readings" ? tabReadings() : tabRaw(route)));
  // the belief over steps and the cost by kind are drawn in the candidates and spending panels of the timeline
  const side = h("div", { class: "side", "aria-label": "Views of the whole episode" }, sideChecks());
  const insp = h("aside", { class: "inspector-panel", id: "raw-inspector", "aria-label": "Raw record of the selection", "aria-live": "polite" });
  return h("section", {}, head, strip, h("div", { class: "ep-body" }, h("div", { class: "ep-main" }, main, side), insp));
}
function afterEpisode() {
  const ov = document.getElementById("ovb-host");
  if (ov) mountOverview(ov);
  const g = document.getElementById("graph-host");
  if (g) mountGraph(g);
  drawInspector();
  const route = parseRoute();
  if (route.line) { const el = document.getElementById(`L${route.line}`); if (el) el.scrollIntoView({ block: "center" }); }
}

/* each card has two faces, Summary and Raw; the switch Raw everywhere turns every card to its raw face */
function face(key, ord, title, cls, summary, rawFn) {
  const raw = S.rawAll || S.face[key] === "raw";
  const seg = h("div", { class: "face-seg", role: "group", "aria-label": `${title}, face` },
    [["summary", "Plain"], ["raw", "Raw"]].map(([f, t]) => h("button", { type: "button", "aria-pressed": String((f === "raw") === raw),
      disabled: S.rawAll && f === "summary" ? "" : null, title: S.rawAll && f === "summary" ? "Raw files is on in the header" : null,
      onclick: () => { S.face[key] = f; render(); } }, t)));
  return h("article", { class: ["card", ...(cls || []), raw ? "raw-on" : null] }, h("h3", {}, h("span", { class: "ord" }, String(ord)), h("span", { class: "card-t" }, title), seg),
    raw ? h("div", { class: "raw-face" }, rawFn()) : summary);
}
function card(ord, title, ...body) {
  return h("article", { class: "card" }, h("h3", {}, h("span", { class: "ord" }, String(ord)), title), body);
}
function obsLine(o) {
  const means = Object.entries(o.values || {}).filter(([k]) => bandOf(k));
  const out = means.filter(([k, v]) => outside(k, v));
  return srcable(h("li", {}, `reading ${o.n} of the ${o.sensor_group} sensors, cycle ${o.t}, ${out.length} of ${means.length} sensors outside the normal range`,
    out.length ? h("div", { class: "chips" }, out.map(([k]) => h("span", { class: "chip out", title: k }, short(k)))) : null),
  `obs/${o.n}`, [rObs(o.n), rObsAnswer(o.n)]);
}
function cardKnown(c) {
  const before = obsBefore(c.n);
  const know = ep().calls.filter((x) => x.n < c.n && x.kind === "reads" && !x.served && !x.refused);
  const advice = ep().calls.filter((x) => x.n < c.n && x.kind === "advice");
  const prev = c.n > 1 ? call(c.n - 1) : null;
  const open = prev && prev.open_work_orders ? prev.open_work_orders : [];
  const notesSoFar = ep().calls.filter((x) => x.n < c.n).flatMap((x) => x.notifications || []);
  const alarms = new Set(notesSoFar.filter((n) => n.event === "alarm").map((n) => n.channel));
  const added = [];
  for (const o of c.served || []) added.push(srcable(h("li", {}, `reading ${o.n} of the ${o.sensor_group} sensors at cycle ${o.t}`), `obs/${o.n}`, [rObs(o.n), rObsAnswer(o.n)]));
  if (c.served && !c.served.length) added.push(srcable(h("li", {}, "a read that returned nothing"), `Step ${c.n}`, [rAnswer(c.n, ["result"])]));
  if (c.kind === "reads" && !c.served && !c.refused) added.push(srcable(h("li", {}, "looked up ", h("code", {}, callText(c))), `Step ${c.n}`, [rAnswer(c.n, ["result"])]));
  if (c.kind === "advice") added.push(srcable(h("li", {}, "advice from the CausalDiagnosis MCP"), `Step ${c.n}`, [rAnswer(c.n)]));
  (c.notifications || []).forEach((n, k) => { if (n.event !== "alarm") added.push(srcable(h("li", {}, n.message), `Notification of step ${c.n}`, [rAnswer(c.n, ["notifications", k])])); });
  const al = (c.notifications || []).filter((n) => n.event === "alarm");
  if (al.length) added.push(srcable(h("li", {}, `${plural(al.length, "alarm")}: `, h("span", { class: "chips" }, al.map((n) => h("span", { class: "chip out", title: n.channel }, short(n.channel))))), `Alarms of step ${c.n}`, [rAnswer(c.n, ["notifications"])]));
  const dd = (txt, title, refs) => srcable(h("dd", {}, txt), title, refs);
  const summary = [
    h("p", { class: "lead" }, before.length ? `${plural(before.length, "reading")}, ${plural(alarms.size, "sensor")} in alarm` : "No reading yet"),
    h("dl", { class: "kv" },
      h("dt", {}, "time"), prev ? dd(`cycle ${c.time_before} of ${rules().horizon}`, "Time before the step", [rAnswer(c.n - 1, ["time"]), rRules("horizon")]) : h("dd", {}, `cycle ${c.time_before} of ${rules().horizon}`),
      h("dt", {}, "budget left"), dd(String(c.budget_before), "Budget left before the step", [rStatusBefore(c.n, "budget_left")]),
      h("dt", {}, "steps left"), dd(String(c.steps_left_before), "Steps left before the step", [rStatusBefore(c.n, "steps_left")]),
      h("dt", {}, "open work orders"), prev ? dd(open.length ? open.map((w) => `${w.component} (${String(w.status).replace(/_/g, " ")})`).join(", ") : "none", "Open work orders before the step", [rAnswer(c.n - 1, ["open_work_orders"])]) : h("dd", {}, "none"),
      h("dt", {}, "looked up"), h("dd", {}, know.length ? know.map((x) => srcable(h("div", { title: callText(x) }, `${x.n} ${stepWords(x)}`), `Step ${x.n}`, [rAnswer(x.n, ["result"])])) : "none"),
      h("dt", {}, "advice asked"), h("dd", {}, advice.length ? advice.map((x) => srcable(h("div", { title: x.tool }, `${x.n} ${stepWords(x)}`), `Step ${x.n}`, [rAnswer(x.n)])) : "none")),
    before.length ? h("ul", {}, before.map(obsLine)) : null,
    c.board != null ? srcable(h("p", { class: "board-line" }, boardLine(c)), `The board of plan ${ep().boards[c.board].plan}`, [boardRef(c.board)],
      "The board is the part of the planning request that the controller writes for the model. Open Raw for the whole request.") : null,
    h("h4", { class: "sub" }, "This step added"),
    added.length ? h("ul", {}, added) : h("p", { class: "note" }, "nothing new")];
  const rawFn = () => {
    const prior = ep().calls.filter((x) => x.n < c.n);
    if (!prior.length) return h("p", { class: "note" }, "No answer had come back before the first step.");
    return [h("p", { class: "note" }, `Every answer the agent had received before step ${c.n}, verbatim, newest first.`),
      prior.slice().reverse().map((x, i) => h("details", { class: "raw-item", open: i < 3 ? "" : null }, h("summary", {}, `step ${x.n} · ${x.tool}`), rawView(rAnswer(x.n))))];
  };
  return face("known", 1, "Known before this step", [], summary, rawFn);
}
/* the board the model of the CausalDiagnosis agent read: the planning request of the plan that made a call (data.boards_of) */
function boardRef(i) { const b = ep().boards[i]; return { f: "ep/llm_calls.json", p: [b.llm_index], s: [b.llm_index, "prompt"], dOnly: true, plain: true, board: i }; }
function boardLine(c) {
  const b = ep().boards[c.board];
  return c.board_before_plan ? `The model read this ranking on the board of plan ${b.plan}, the board it planned from` : `The model planned this step from the board of plan ${b.plan}`;
}
// the board in its sections, the first and the last open, every other folded under its heading; a table cell that ends in low
// or high, a value outside its normal range, is marked
function boardFace(b) {
  const last = b.sections.length - 1;
  return h("div", { class: "rd-face board" },
    h("p", { class: "note" }, `The board of model request ${b.llm_index + 1} of llm_calls.json${b.retry ? ", sent again after a reply that could not be used" : ""}, as the model read it.`),
    b.sections.map((sec, i) => (i === 0 || i === last ? h("section", { class: "bd-sec" }, h("h4", {}, sec.heading), sec.blocks.map(boardBlock))
      : h("details", { class: "bd-sec" }, h("summary", {}, sec.heading), sec.blocks.map(boardBlock)))));
}
function boardBlock(b) {
  if (b.table) {
    const cell = (x) => { const m = /^(.*\S)\s(low|high)$/.exec(x); return h("td", { class: m ? "bd-out" : null }, m ? [m[1], " ", h("b", {}, m[2])] : x); };
    return h("div", { class: "bd-scroll" }, h("table", { class: "bd-table" }, h("thead", {}, h("tr", {}, b.table.header.map((x) => h("th", {}, x)))),
      h("tbody", {}, b.table.rows.map((row) => h("tr", {}, row.map(cell))))));
  }
  if (b.items) return h("ul", {}, b.items.map((x) => h("li", {}, x)));
  return h("p", {}, b.text);
}
// the possible causes with a chance that a ranking leaves out, by its own more_candidates: how many, and their summed
// chance behind a click; nothing where it leaves none out or where the answer does not say
function notListed(r) {
  const more = r.more_candidates;
  if (!more || !(more.count > 0)) return null;
  return h("details", { class: "note not-listed" }, h("summary", {}, `${plural(more.count, "more possible cause")} ${more.count === 1 ? "is" : "are"} not listed`),
    srcable(h("span", {}, `Their summed chance is ${pct(more.mass)}.`), `Possible causes the ranking of step ${r.n} leaves out`, [rAnswer(r.n, ["more_candidates"])]));
}
function rankingBlock(r, compact) {
  const truth = S.hidden && S.hidden.start_state;
  const cands = r.candidates.map((x, i) => [
    srcable(h("span", { class: ["lbl", truth && sameComb(x.combination, truth) ? "truth" : null] }, `${x.rank}. ${combText(x.combination)}`,
      truth && sameComb(x.combination, truth) ? hid(h("span", { class: "hidden-label" }, " ground truth, the real fault")) : null), `Possible cause ${x.rank} of the ranking of step ${r.n}`, [rAnswer(r.n, ["candidates", i])]),
    h("span", { class: "bar", role: "img", "aria-label": `probability ${pct(x.probability)}` }, barFill(x.probability)), h("span", { class: "num" }, rv(x.probability, pct(x.probability)))]);
  const fp = Object.entries(r.fault_probability || {}).map(([k, p]) => [srcable(h("span", { class: "lbl" }, k), `Chance that the ${k} is faulty, step ${r.n}`, [rAnswer(r.n, ["current_fault_probability", k])]),
    h("span", { class: "bar", role: "img", "aria-label": `probability ${pct(p)}` }, barFill(p)), h("span", { class: "num" }, rv(p, pct(p)))]);
  return [srcable(h("p", { class: "lead" }, advisedAction(r)), `Advice of step ${r.n}`, [rAnswer(r.n, ["branches", 0])]),
    srcable(h("div", { class: "sub" }, `Ranking at step ${r.n}, cycle ${r.cycle}, the ${r.candidates.length} most likely of ${nCauses(r)} possible causes, about `, rv(r.perplexity, num(r.perplexity, 2)), " still possible"),
      `Ranking of step ${r.n}`, [rAnswer(r.n, ["effective_hypotheses"])]),
    h("div", { class: "prob" }, cands), notListed(r),
    compact ? null : [h("div", { class: "sub" }, "Chance that each part is faulty now"), h("div", { class: "prob" }, fp),
      h("dl", { class: "kv" }, h("dt", {}, "expected cost if it repairs now"), srcable(h("dd", {}, atLevel(r, "expected_loss_act", rv(r.expected_loss_act, num(r.expected_loss_act, 1)))), "Expected cost if it repairs now", [rAnswer(r.n, ["expected_loss_act"])]),
        h("dt", {}, "expected cost if it reads more first"), srcable(h("dd", {}, atLevel(r, "expected_loss_test", Object.entries(r.expected_loss_test || {}).map(([k, v]) => [`${k} `, rv(v, num(v, 1)), " "]) || NR)), "Expected cost if it reads more first", [rAnswer(r.n, ["expected_loss_test"])]))]];
}
function adviceRefs(c) {
  const r = c.advice_in_force ? ep().rankByN[c.advice_in_force] : null;
  const eff = ep().effects.filter((x) => x.n < c.n && x.time_window && x.time_window.start === c.time_before);
  return [r ? rAnswer(r.n) : null, ...eff.map((x) => rAnswer(x.n))].filter(Boolean);
}
function cardAdvised(c) {
  const run = runOf(ep().run) || {};
  const rawFn = () => {
    const refs = c.kind === "advice" ? [c.advice_in_force && c.advice_in_force !== c.n ? rAnswer(c.advice_in_force) : null].filter(Boolean) : adviceRefs(c);
    if (!refs.length) return h("p", { class: "note" }, c.kind === "advice" ? "This step is itself the question. Its answer is under What came back." : "No advice of the CausalDiagnosis MCP applies to this step.");
    return [h("p", { class: "note" }, "The answers of the CausalDiagnosis MCP that apply to this step, as returned."), refs.map((ref) => rawView(ref))];
  };
  if (!run.diagnostics_server) return face("advised", 2, "Advice at this step", [], h("p", { class: "lead" }, "This run has no CausalDiagnosis MCP"), () => h("p", { class: "note" }, "This run has no CausalDiagnosis MCP."));
  if (c.kind === "advice") {
    const prior = c.advice_in_force && c.advice_in_force !== c.n ? ep().rankByN[c.advice_in_force] : null;
    return face("advised", 2, "Advice at this step", [], [h("p", { class: "lead" }, "At this step the agent asks the CausalDiagnosis MCP"),
      h("p", { class: "note" }, "Its answer is under What came back, and it is the advice for the next steps in the same cycle."),
      prior ? rankingBlock(prior, true) : null], rawFn);
  }
  const r = c.advice_in_force ? ep().rankByN[c.advice_in_force] : null;
  if (!r) {
    const last = lastRanking(c.n);
    return face("advised", 2, "Advice at this step", [], [h("p", { class: "lead" }, "No advice applies to this step"),
      h("p", { class: "note" }, last ? `The last ranking (step ${last.n}, cycle ${last.cycle}) no longer applied once time moved on to cycle ${c.time_before}.` : "The agent has not asked for a ranking yet.")], rawFn);
  }
  const eff = ep().effects.filter((x) => x.n < c.n && x.time_window && x.time_window.start === c.time_before);
  return face("advised", 2, "Advice at this step", [], [rankingBlock(r),
    eff.map((x) => srcable(h("p", { class: "note" }, `Prediction ${x.prediction_id} for ${planWords(x.action)}, chance the machine works again `, rv(x.recovery, pct(x.recovery)), ` (between `, rv(x.recovery_interval_low, pct(x.recovery_interval_low)), " and ", rv(x.recovery_interval_high, pct(x.recovery_interval_high)), ")."),
      `Prediction ${x.prediction_id}`, [rAnswer(x.n), rPred(x.prediction_id)])),
    h("p", { class: "note" }, `The last ranking before this step, at cycle ${r.cycle}.`)], rawFn);
}
function followRefs(c) {
  const r = c.advice_in_force ? ep().rankByN[c.advice_in_force] : null;
  if (!r) return [rCall(c.n, ["tool_args"])];
  const m = c.kind === "decide" ? ["candidates", 0] : c.kind === "orders" ? ["branches", 0, "faulty_set"] : ["separating_test"];
  return [rCall(c.n, ["tool_args"]), rAnswer(r.n, m)];
}
function planRef(c) {
  const t = FILES["ep/plan.txt"];
  if (!t || !c.plan) return null;
  const lines = t.split("\n"), keys = ["Task", "Server", "Tool", "Dependency", "ExpectedOutput"].map((k) => `#${k}${c.n}:`);
  const at = lines.map((ln, i) => (keys.some((k) => ln.startsWith(k)) ? i + 1 : null)).filter(Boolean);
  return at.length ? { f: "ep/plan.txt", l: [Math.min(...at), Math.max(...at)] } : null;
}
function cardChosen(c) {
  const badgeCls = c.follow === "followed" ? "followed" : c.follow === "deviated" ? "deviated" : "none";
  const p = c.plan || {};
  const summary = [
    h("p", { class: "lead" }, plainTask(c)),
    srcable(h("div", { class: "callline" }, callText(c)), `Step ${c.n}, what the agent sent`, [rCall(c.n, ["tool_args"])]),
    h("div", {}, kindDot(c.kind), " ", KINDS[c.kind].label, " · ", c.kind === "advice" ? h("span", { class: "sub" }, "a question, not a decision")
      : srcable(h("span", { class: ["badge", badgeCls] }, FOLLOW_WORDS[c.follow || "no advice"]), `Step ${c.n}, ${FOLLOW_WORDS[c.follow || "no advice"]}`, followRefs(c), followReason(c) ? `${followReason(c)} The same rule decides it in the scores of the CausalDiagnosis MCP.` : null)),
    followReason(c) ? h("p", { class: "note" }, followReason(c)) : null,
    h("dl", { class: "kv" },
      h("dt", {}, "line of the agent's plan"), planRef(c) ? srcable(h("dd", {}, `step ${c.n} of the plan${/^#S\d+$/.test(p.Dependency || "") ? `, after step ${p.Dependency.slice(2)}` : ""}`), `Plan line of step ${c.n}`, [planRef(c)]) : h("dd", {}, NR),
      h("dt", {}, "what the agent expected to see"), h("dd", {}, p.ExpectedOutput ? `“${p.ExpectedOutput}”` : NR),
      h("dt", {}, "answered by"), h("dd", { title: c.server }, SERVER_WORDS[c.server] || c.server),
      h("dt", {}, (runOf(ep().run) || {}).agent === "scripted" ? "the script's message" : "the model's message"), rModel(c.n) ? srcable(h("dd", {}, `${(runOf(ep().run) || {}).agent === "scripted" ? "message" : "model call"} ${c.llm_index + 1} of llm_calls.json`), `Model message of step ${c.n}`, [rModel(c.n)]) : h("dd", {}, NR))];
  const rawFn = () => [h("p", { class: "note" }, "The step as the agent sent it, and its line in the transcript."),
    rawView({ ...rCall(c.n, ["tool_args"]), u: [c.n - 1, "tool_args"] }), rawView(rTranscript(c.n)),
    h("div", { class: "rr-sub" }, "The model message behind this step, prompt and reply."),
    rModel(c.n) ? rawView(rModel(c.n)) : h("p", { class: "nr" }, `model message: ${NR}`)];
  return face("chosen", 3, "What the agent did", ["chosen", `k-${c.kind}`], summary, rawFn);
}
function cardResult(c) {
  const notes = (c.notifications || []).map((n, k) => [n, k]).filter(([n]) => n.event !== "alarm");
  const H = S.hidden;
  let hiddenBox = null;
  if (H && c.env_step) {
    const log = H.log || [];
    const ents = log.map((x, i) => [x, i]).filter(([x]) => x.step === c.env_step);
    const loss = ((H.end || {}).downtime_by_cycle || []).filter((x) => x.t >= c.time_before && x.t < c.time_after).reduce((a, x) => a + x.charge, 0);
    const truth = H.verdict && H.verdict.truth_at_write;
    const endI = log.findIndex((x) => x.event === "end");
    hiddenBox = hid(h("div", { class: "hidden-box" }, h("div", { class: "hidden-label" }, "Ground truth for this step"),
      h("dl", { class: "kv" },
        h("dt", {}, "output lost while the machine was faulty"), srcable(h("dd", {}, c.cycles ? `${loss} over ${plural(c.cycles, "cycle")}` : "none, no time passed"), "Output lost during the step", endI >= 0 ? [{ f: "ep/hidden/log.json", p: [endI], m: [endI, "downtime_by_cycle"] }] : []),
        ents.map(([x, i]) => [h("dt", {}, x.event), srcable(h("dd", {}, `cost ${x.cost}${x.unnecessary ? ", unnecessary" : ""}${x.served ? `, recorded cycles ${x.served.map((y) => y.cycle).join(", ")}` : ""}`), `Hidden log entry ${i}`, [{ f: "ep/hidden/log.json", p: [i] }])]),
        c.answer && c.answer.result && c.answer.result.diagnosis && truth ? [h("dt", {}, "the real fault when the diagnosis was written"), srcable(h("dd", {}, combText(truth, true)), "The real fault when the diagnosis was written", [{ f: "ep/hidden/verdict.json", p: ["truth_at_write"] }])] : null)));
  }
  const summary = [srcable(h("div", {}, resultSummary(c)), `Step ${c.n}: the answer`, [rAnswer(c.n, ["result"])]),
    notes.length ? h("div", {}, h("div", { class: "sub" }, "Notifications in the answer"), h("ul", { class: "notes-list" }, notes.map(([n, k]) => srcable(h("li", {}, n.message), `Notification of step ${c.n}`, [rAnswer(c.n, ["notifications", k])])))) : null,
    h("dl", { class: "kv" },
      h("dt", {}, "cost"), srcable(h("dd", {}, c.price == null ? NR : String(c.price)), `Cost of step ${c.n}`, [rStatusBefore(c.n, "budget_left"), rAnswer(c.n, ["budget_left"])], "The cost is the budget left before the step minus the budget left after it."),
      h("dt", {}, "time taken"), srcable(h("dd", {}, c.cycles ? `${plural(c.cycles, "cycle")}, cycle ${c.time_before} → ${c.time_after}` : "no time"), `Cycles of step ${c.n}`, [c.n > 1 ? rAnswer(c.n - 1, ["time"]) : null, rAnswer(c.n, ["time"])].filter(Boolean)),
      h("dt", {}, "seconds the step took"), srcable(h("dd", {}, stopwatch(c)), `Wall time of step ${c.n}`, [rCall(c.n, ["duration_ms"])]),
      c.level ? levelRow(c) : null,
      c.profile_raw ? [h("dt", {}, "raw form"), srcable(h("dd", {}, "the profile with one entry per field and per tool, rebuilt from this answer"), `Step ${c.n}: the profile in its raw form`,
        [{ inline: c.profile_raw, title: "the raw form of the profile, rebuilt by the page" }], "Every value is in the answer the agent received. The shorter level writes the tool table and the normal ranges as long tables, and the raw form gives them back one entry per field and per tool.")] : null,
      thinkRow(c)),
    hiddenBox];
  const rawFn = () => [h("p", { class: "note" }, "The answer of the tool, word for word."), rawView(rAnswer(c.n, null, true)),
    H && c.env_step ? hid(h("div", { class: "hidden-box" }, h("div", { class: "hidden-label" }, "Ground truth log of this step"),
      (H.log || []).map((x, i) => [x, i]).filter(([x]) => x.step === c.env_step).map(([, i]) => rawView({ f: "ep/hidden/log.json", p: [i] })))) : null];
  return face("result", 4, "What came back", [], summary, rawFn);
}

/* ---------- Raw tab: a file browser of the allowed files ---------- */
function episodeGroups() {
  const E = ep(), Hd = S.hidden && S.hidden.files ? S.hidden.files : null;
  return [
    { label: `Episode folder ${E.id}`, files: E.files.ep.map((f) => ({ ...f, path: `ep/${f.name}` })) },
    { label: `${E.id}/hidden, ground truth`, hidden: true, files: Hd ? Hd.ep.map((f) => ({ ...f, path: `ep/${f.name}` })) : [] },
    { label: `Run folder ${E.run}`, files: E.files.run.map((f) => ({ ...f, path: `run/${f.name}` })) },
    { label: `Run folder, ground truth`, hidden: true, files: Hd ? Hd.run.map((f) => ({ ...f, path: `run/${f.name}` })) : [] },
  ];
}
function tabRaw(route) {
  return fileBrowser(episodeGroups(), route.file, route.line);
}
function fileBrowser(groups, cur, line) {
  const all = groups.flatMap((g) => g.files);
  if (!cur || !all.some((f) => f.path === cur)) cur = all.length ? all[0].path : null;
  const tree = h("nav", { class: "ftree", "aria-label": "Files" }, groups.filter((g) => g.files.length).map((g) => {
    const box = h("div", { class: "fgroup" }, h("div", { class: "fg-h" }, g.label),
      h("ul", {}, g.files.map((f) => h("li", {}, h("a", { href: fileHref(f.path), class: f.path === cur ? "cur" : null, "aria-current": f.path === cur ? "page" : null },
        h("span", { class: "mono" }, f.name.replace(/^hidden\//, "")), h("span", { class: "sub" }, fmtBytes(f.bytes)))))));
    return g.hidden ? hid(box) : box;
  }), h("p", { class: "note" }, "Only files the agent could see are shown. The files that start the tool servers name the real fault, so they are never shown."));
  const curFile = all.find((f) => f.path === cur);
  const fv = cur ? fileView(cur, line) : h("p", { class: "nr" }, "No file.");
  const view = curFile && curFile.note ? h("div", {}, h("p", { class: "note file-note" }, curFile.note), fv) : fv;
  return h("div", { class: "fbrowser" }, tree, isHiddenFile(cur || "") ? hid(view) : view);
}
/* which line of a file belongs to which step */
function stepLines(path) {
  const e = ep(), sp = path.endsWith(".json") ? spansOfFile(path) : null, st = linesOfFile(path), out = new Map();
  const put = (p, n) => { const v = sp && sp.get(JSON.stringify(p)); if (v && n) out.set(lineOf(st, v[0]), n); };
  if (path === TRAJ) e.calls.forEach((c) => put([c.n - 1], c.n));
  else if (path === "ep/llm_calls.json") e.calls.forEach((c) => { if (c.llm_index != null) put([c.llm_index], c.n); });
  else if (path === REL) e.observations.forEach((o, i) => put(["observations", i], e.obsCall[o.n]));
  else if (path === "ep/transcript.jsonl") for (const [n, li] of Object.entries(e.trLine)) out.set(li, +n);
  return out;
}
function foldsOf(path) {
  const sp = path.endsWith(".json") ? spansOfFile(path) : null, st = linesOfFile(path), t = FILES[path], out = new Map();
  if (!sp) return out;
  for (const [, [a, b]] of sp) {
    if (t[a] !== "{" && t[a] !== "[") continue;
    const l0 = lineOf(st, a), l1 = lineOf(st, b - 1);
    if (l1 - l0 >= 2 && !out.has(l0)) out.set(l0, l1);
  }
  return out;
}
function fileView(path, sel) {
  const text = FILES[path];
  if (text == null) return h("section", { class: "fview" }, h("p", { class: "nr" }, `${path} is not shared with this page`));
  const st = linesOfFile(path), n = lineCount(path);
  const steps = stepLines(path), folds = foldsOf(path);
  const closed = (S.folds[path] ||= new Set());
  const q = S.search[path] || "";
  const code = h("div", { class: "fv-code" });
  let hits = 0;
  for (let i = 0; i < n; i++) {
    const a = st[i], b = i + 1 < st.length ? st[i + 1] - 1 : text.length, ln = text.slice(a, b);
    const tx = h("span", { class: "tx" });
    if (q) {
      let k = 0, j;
      const low = ln.toLowerCase(), ql = q.toLowerCase();
      while ((j = low.indexOf(ql, k)) >= 0) { tx.append(ln.slice(k, j), h("mark", { class: "hit" }, ln.slice(j, j + q.length))); k = j + q.length; hits++; }
      tx.append(ln.slice(k));
    } else tx.append(ln);
    const fold = folds.get(i);
    const tog = fold != null ? h("button", { type: "button", class: "fold", "aria-expanded": String(!closed.has(i)), "aria-label": `${closed.has(i) ? "Unfold" : "Fold"} lines ${i + 2} to ${fold}`,
      onclick: () => { closed.has(i) ? closed.delete(i) : closed.add(i); const nv = fileView(path, sel); document.querySelector(".fview").replaceWith(nv); } }, closed.has(i) ? "▸" : "▾") : h("span", { class: "fold" });
    const stepLink = steps.has(i) ? h("a", { class: "stepl", href: `#/ep/${encodeURIComponent(ep().run)}/${encodeURIComponent(ep().id)}/decision/${steps.get(i)}`, title: `This record belongs to step ${steps.get(i)}` }, `step ${steps.get(i)}`) : h("span", { class: "stepl" });
    code.append(h("div", { class: ["fl", i + 1 === sel ? "sel" : null], id: `L${i + 1}` }, stepLink, tog, h("a", { class: "ln", href: fileHref(path, i + 1) }, String(i + 1)), tx,
      closed.has(i) ? h("span", { class: "folded-n" }, ` … ${fold - i - 1} lines folded`) : null));
    if (closed.has(i)) i = fold - 1;
  }
  const input = h("input", { type: "search", class: "fv-search", placeholder: "Search this file", value: q, "aria-label": `Search ${path}` });
  let at = -1;
  input.addEventListener("input", () => { S.search[path] = input.value; const nv = fileView(path, sel); document.querySelector(".fview").replaceWith(nv); const ni = nv.querySelector(".fv-search"); ni.focus(); ni.setSelectionRange(ni.value.length, ni.value.length); });
  input.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    const marks = code.querySelectorAll("mark.hit");
    if (!marks.length) return;
    marks.forEach((m) => m.classList.remove("cur"));
    at = (at + (ev.shiftKey ? -1 : 1) + marks.length) % marks.length;
    marks[at].classList.add("cur"); marks[at].scrollIntoView({ block: "center" });
  });
  const bytes = new TextEncoder().encode(text).length;
  return h("section", { class: "fview" },
    h("div", { class: "fv-head" }, h("b", { class: "mono" }, path.replace(/^ep\//, `${ep().id}/`).replace(/^run\//, `${ep().run}/`)),
      h("span", { class: "sub" }, `${plural(n, "line")}, ${fmtBytes(bytes)}`), input, h("span", { class: "sub" }, q ? `${hits} match${hits === 1 ? "" : "es"}, press Enter for the next` : ""),
      folds.size ? h("button", { type: "button", class: "mini", onclick: () => { if (closed.size) closed.clear(); else for (const k of folds.keys()) if (k > 0) closed.add(k); document.querySelector(".fview").replaceWith(fileView(path, sel)); } }, closed.size ? "Unfold all" : "Fold all") : null,
      copyButton(() => text)),
    code);
}

/* ---------- Readings tab ---------- */
/* ---------- Readings tab: one row per sensor against its normal range ----------
   Each row holds the channel's readings of the first statistic up to the selected step, its normal range, and a bar
   that places them on one scale. The switch above the table draws where the CausalDiagnosis MCP expects each reading
   after a repair plan it estimated. "Exact values" shows the released numbers unrounded. */
function tabReadings() {
  const mode = S.rawAll ? "exact" : S.rdMode;
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Readings as" }, [["sensor", "By sensor"], ["exact", "Exact values"]].map(([k, t]) =>
    h("button", { type: "button", "aria-pressed": String(mode === k), disabled: S.rawAll && k === "sensor" ? "" : null, onclick: () => { S.rdMode = k; render(); } }, t)));
  return h("div", {}, h("div", { class: "graph-tools" }, seg,
    h("span", { class: "sub" }, mode === "exact" ? "The values exactly as the run recorded them. Click a value to see the answer it came in." : "One row per sensor, with its readings up to this step and its normal range.")),
    mode === "exact" ? readingsTable() : readingsBySensor());
}
// "VolumeFlow" reads "Volume flow"; "mm_per_s" reads "mm/s"
const plainWords = (w) => String(w).replace(/([a-z])([A-Z])/g, "$1 $2").replace(/ ([A-Z])(?=[a-z])/g, (m, c) => ` ${c.toLowerCase()}`);
const plainUnit = (u) => (u ? String(u).replace(/_per_/g, "/").replace(/^percent$/, "%").replace(/^C$/, "°C").replace(/^F$/, "°F") : "");
const DEC = 2; // every number of a sensor row shows this many decimals, the readings as the range
const fix = (v) => (typeof v === "number" ? v.toFixed(DEC) : NR);
function sensorName(ch) {
  const c = (ep().channels || {})[ch];
  return c && c.quantity ? { name: `${plainWords(c.quantity)} (${c.id})`, unit: plainUnit(c.unit) } : { name: ch, unit: "" };
}
const planWords = (action) => { const [verb, comp] = String(action || "").split(":"); return comp ? `${verb} the ${comp}` : verb; };
function sensorRows(stat0, known, eff) {
  const e = ep(), r = rules(), RANK = { high: 0, low: 0, normal: 1, none: 2, unread: 3 };
  const chans = [...new Set(Object.values(r.sensor_groups || {}).flat())];
  return chans.map((ch) => {
    const field = `${ch}_${stat0}`, band = bandOf(field);
    const vals = [];
    e.observations.forEach((o, i) => { if (field in (o.values || {})) vals.push({ v: o.values[field], t: o.t, n: o.n, i, later: !known.has(o.n) }); });
    const now = vals.filter((v) => !v.later), last = now[now.length - 1];
    const state = !last ? "unread" : !band ? "none" : last.v > band.high ? "high" : last.v < band.low ? "low" : "normal";
    const pred = eff ? (eff.predicted || []).find((p) => p.field === field) || null : null;
    return { ch, field, band, vals, state, pred };
  }).sort((a, b) => RANK[a.state] - RANK[b.state]);
}
function rangeBar(row, w) {
  const now = row.vals.filter((v) => !v.later);
  const pts = [...now.map((v) => v.v), ...(row.band ? [row.band.low, row.band.high] : []), ...(row.pred ? [row.pred.low, row.pred.high] : [])];
  if (!pts.length) return null;
  let lo = Math.min(...pts), hi = Math.max(...pts);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.12; lo -= pad; hi += pad;
  const X = (v) => 10 + ((v - lo) / (hi - lo)) * (w - 20);
  const out = row.state === "high" || row.state === "low";
  const svg = s("svg", { class: "rdA-bar", width: w, height: 46, viewBox: `0 0 ${w} 46`, role: "img",
    "aria-label": `${sensorName(row.ch).name} ${now.length ? `last read ${num(now[now.length - 1].v)}` : "not read yet"}${row.band ? `, normal range ${num(row.band.low)} to ${num(row.band.high)}` : ""}${row.pred ? `, expected after the repair ${num(row.pred.low)} to ${num(row.pred.high)}` : ""}` },
    s("line", { x1: 10, x2: w - 10, y1: 20, y2: 20, class: "rdA-axis" }));
  if (row.band) svg.append(s("rect", { x: X(row.band.low), y: 13, width: Math.max(2, X(row.band.high) - X(row.band.low)), height: 14, rx: 3, class: "rdA-band" }));
  if (row.pred) {
    const a = X(row.pred.low), b = X(row.pred.high);
    svg.append(s("path", { d: `M${a + 5},8 H${a} V32 H${a + 5} M${b - 5},8 H${b} V32 H${b - 5}`, class: "rdA-pred" }));
  }
  now.forEach((v, i) => svg.append(s("circle", { cx: X(v.v), cy: 20, r: 5, class: ["rdA-pt", out ? "out" : null, i === now.length - 1 ? null : "earlier"] })));
  svg.append(s("text", { x: 10, y: 44, class: "rdA-tick" }, fix(lo)), s("text", { x: w - 10, y: 44, "text-anchor": "end", class: "rdA-tick" }, fix(hi)));
  return svg;
}
function planNote(eff) {
  if (!eff) return null;
  const chk = ep().checks.find((c) => c.prediction_id === eff.prediction_id && c.n <= S.step);
  const done = orderLives(ep()).map((l) => l.close).filter((x) => x != null);
  const asked = chk ? `At step ${chk.n} the agent asked for the readings to be compared with this prediction${checkTail({ ...call(chk.n), answer: chk }, done.length ? Math.max(...done) : null)}` : `By step ${S.step} the agent has not asked for the readings to be compared with this prediction.`;
  const seen = chk && chk.observation ? ` The comparison used reading ${summarize(chk.observation)}.` : "";
  const rec = eff.recovery == null ? "" : ` For the repair to ${planWords(eff.action)}, the chance the CausalDiagnosis MCP gave that the machine works at the first reading after it is ${pct(eff.recovery)}.`;
  const lvl = /no interval carries its level/.test(eff.message || "") ? " The brackets show the expected range, with no stated confidence."
    : eff.alpha != null ? ` The brackets are ${Math.round((1 - eff.alpha) * 100)}% ranges.` : "";
  return srcable(h("div", { class: "rdA-note" }, asked, seen, rec, lvl), "The prediction and its check", [rPred(eff.prediction_id), chk ? rAnswer(chk.n) : null]);
}
function readingsBySensor() {
  const e = ep(), stat0 = (rules().statistics || ["mean"])[0];
  const known = new Set(e.calls.filter((x) => x.n <= S.step).flatMap((x) => (x.served || []).map((o) => o.n)));
  const effects = e.effects.filter((x) => x.n <= S.step);
  // the plan shown follows the newest estimate up to the step until the person picks one; "none" stays picked
  if (S.rdPlan === undefined || (S.rdPlan !== null && !effects.some((x) => x.prediction_id === S.rdPlan))) S.rdPlan = effects.length ? effects[effects.length - 1].prediction_id : undefined;
  const eff = effects.find((x) => x.prediction_id === S.rdPlan) || null;
  const rows = sensorRows(stat0, known, eff), read = rows.filter((x) => x.state !== "unread"), bad = read.filter((x) => x.state === "high" || x.state === "low");
  const narrow = innerWidth < 1000, W = narrow ? 200 : 300;
  const summary = read.length ? `At step ${S.step}, ${bad.length} of the ${plural(read.length, "sensor")} read so far ${bad.length === 1 ? "is" : "are"} outside the normal range.` : `No sensor has been read by step ${S.step}.`;
  const chip = (id, label) => h("button", { type: "button", "aria-pressed": String((S.rdPlan || null) === id), onclick: () => { S.rdPlan = id; render(); } }, label);
  const plans = effects.length ? h("div", { class: "rdA-plans" }, h("span", {}, "Show what a repair should change"),
    h("div", { class: "seg chips", role: "group", "aria-label": "Repair plan" }, chip(null, "none"), effects.map((x) => chip(x.prediction_id, `${planWords(x.action)} (step ${x.n})`)))) : null;
  const TAG = { high: ["above range", "hi"], low: ["below range", "hi"], normal: ["normal", "ok"], none: ["no normal range released", "na"], unread: ["not read yet", "na"] };
  const body = rows.map((x) => {
    const sn = sensorName(x.ch);
    const name = srcable(h("td", { title: x.field }, h("span", { class: "rdA-name" }, sn.name, sn.unit ? h("span", { class: "rdA-unit" }, ` in ${sn.unit}`) : null)),
      `Normal range of ${x.field}`, [x.band ? rRules(bandKey(), x.field) : null]);
    const shown = x.vals.slice(-6), hiddenCount = x.vals.length - shown.length;
    const vals = h("td", { class: "rdA-vals" }, hiddenCount ? h("small", {}, `${hiddenCount} earlier, `) : null, shown.map((v) => srcable(
      h("span", { class: ["rdA-val", v.later ? "later" : null], title: v.later ? "read after the selected step" : null },
        h("span", { class: !v.later && x.band && outside(x.field, v.v) ? "hi" : null }, fix(v.v)), h("small", {}, ` cycle ${v.t}`)),
      `obs/${v.n}, ${x.field}`, [rObs(v.n, x.field), rObsAnswer(v.n, x.field)])), x.vals.length ? null : h("span", { class: "na" }, "not read"));
    const [tt, tc] = TAG[x.state];
    return h("tr", { class: `rdA-row s-${x.state}` }, name, vals, h("td", { class: "rdA-range", "data-head": "Normal range" }, x.band ? `${fix(x.band.low)} to ${fix(x.band.high)}` : "not released"),
      h("td", {}, rangeBar(x, W)), h("td", {}, h("span", { class: `rdA-tag ${tc}` }, tt)));
  });
  const tbl = h("div", { class: "table-wrap" }, h("table", { class: "rdA" },
    h("thead", {}, h("tr", {}, ["Sensor", `Readings (${stat0} of each cycle)`, "Normal range", "Where the readings sit", ""].map((t) => h("th", { scope: "col" }, t)))),
    h("tbody", {}, body)));
  const leg = legend([["l-rdA-band", "normal range"], ["l-rdA-pt", "reading, the faint ones earlier"], ...(eff ? [["l-rdA-pred", "where the CausalDiagnosis MCP expects the reading after the repair"]] : [])]);
  return h("div", { class: "rdA-wrap" }, h("p", { class: "rdA-summary" }, summary), plans, tbl, leg, planNote(eff));
}
function readingsTable() {
  const e = ep(), text = FILES[REL], sp = spansOfFile(REL);
  const exact = (p, v) => { const x = sp && sp.get(JSON.stringify(p)); return x ? text.slice(x[0], x[1]) : String(v); };
  const fields = [];
  for (const o of e.observations) for (const k of Object.keys(o.values || {})) if (!fields.includes(k)) fields.push(k);
  if (!fields.length) return h("p", {}, "No reading in this episode.");
  const head = h("tr", {}, ["reading", "cycle", "sensor set", "step"].map((t) => h("th", { scope: "col" }, t)), fields.map((f) => h("th", { scope: "col", class: "mono" }, f)));
  const band = h("tr", { class: "band-row" }, h("th", { scope: "row", colspan: "4" }, "normal range"), fields.map((f) => {
    const b = bandOf(f);
    if (!b) return h("td", { class: "na" }, "");
    return srcable(h("td", { class: "mono" }, exact(["rules", bandKey(), f, "low"], b.low), " to ", exact(["rules", bandKey(), f, "high"], b.high)), `Normal range of ${f}`, [rRules(bandKey(), f)]);
  }));
  const rows = e.observations.map((o, i) => {
    const cn = e.obsCall[o.n];
    return h("tr", { class: cn > S.step ? "later" : null },
      srcable(h("th", { scope: "row", class: "mono" }, `obs/${o.n}`), `obs/${o.n}`, [{ f: REL, p: ["observations", i] }, rObsAnswer(o.n)]),
      h("td", {}, exact(["observations", i, "t"], o.t)), h("td", {}, o.sensor_group),
      h("td", {}, cn ? h("a", { href: epHash("decision", cn) }, String(cn)) : NR),
      fields.map((f) => {
        if (!(f in (o.values || {}))) return h("td", { class: "na", title: "not in this sensor set" }, "");
        const out = outside(f, o.values[f]);
        return srcable(h("td", { class: ["mono", out ? "out" : null] }, exact(["observations", i, "values", f], o.values[f]), out ? " ▲" : ""),
          `obs/${o.n}, ${f}`, [rObs(o.n, f), rObsAnswer(o.n, f)]);
      }));
  });
  return h("div", { class: "table-wrap rd-table" }, h("table", { class: "grid" }, h("thead", {}, head), h("tbody", {}, band, rows)),
    h("p", { class: "note" }, "▲ outside the normal range. Rows after this step are faint."));
}

/* ---------- graph nodes and edges back to their records ---------- */
function nodeRefs(n) {
  const e = ep(), id = n.id;
  if (id.startsWith("call/")) { const k = +id.slice(5); return [rCall(k, ["tool_args"]), rAnswer(k)]; }
  if (id.startsWith("wo/")) return [rCall(n.call, ["tool_args"]), rAnswer(n.call, ["result", "work_order"])];
  if (id.startsWith("obs/")) { const k = +id.slice(4); return [rObs(k), rObsAnswer(k)]; }
  if (id.startsWith("pred/")) return [rPred(id), n.call ? rAnswer(n.call) : null];
  if (id.startsWith("ev/")) {
    const c = call(n.call);
    if (id.startsWith("ev/alarms/")) return [rAnswer(n.call, ["notifications"])];
    const k = (c.notifications || []).findIndex((x) => id === `ev/${x.event}/${x.id || x.t}`);
    return [rAnswer(n.call, k >= 0 ? ["notifications", k] : ["notifications"])];
  }
  if (id === "verdict") return [{ f: "ep/hidden/verdict.json", p: [] }];
  if (id === "hstart") { const i = ((S.hidden || {}).log || []).findIndex((x) => x.event === "reset"); return i < 0 ? [] : [{ f: "ep/hidden/log.json", p: [i], m: [i, "hidden"] }]; }
  if (id.startsWith("chk/")) return [{ f: "ep/hidden/verdict.json", p: ["checks", id.slice(4)] }];
  if (id.startsWith("comp/")) { const k = id.slice(5); return [rRules("levels", k), rRules("healthy", k)]; }
  if (id.startsWith("set/")) return [rRules("sensor_groups", id.slice(4))];
  if (id.startsWith("ch/")) {
    const ch = id.slice(3), set = Object.entries(rules().sensor_groups || {}).find(([, cs]) => cs.includes(ch));
    const stat0 = (rules().statistics || ["mean"])[0];
    return [set ? rRules("sensor_groups", set[0]) : null, bandOf(`${ch}_${stat0}`) ? rRules(bandKey(), `${ch}_${stat0}`) : null];
  }
  if (id.startsWith("mode/")) {
    const code = id.slice(5);
    for (const c of e.calls) {
      const fm = c.answer && c.answer.result && c.answer.result.failure_modes;
      const k = Array.isArray(fm) ? fm.findIndex((m) => m.code === code) : -1;
      if (k >= 0) return [rAnswer(c.n, ["result", "failure_modes", k])];
    }
  }
  return [];
}
function edgeNote(edge, a, b) {
  if (edge.cls === "e-deviates" || edge.cls === "e-follows") {
    const c = call(a.call);
    return `${a.label} ${edge.label} the advice of step ${b.call}. ${followReason(c) || ""} A diagnosis follows the advice when it names the most likely cause, a repair when it repairs the parts the advice named, and a paid read when it is the read the advice asked for.`;
  }
  return `${a.label} ${edge.label} ${b.label}. The records of both follow.`;
}

/* ---------- figure: one composed view for the docs, the paper and the slides ---------- */
const FIG_W = 1000;
function figureLine(e) {
  const r = runOf(e.run) || {};
  const who = !r.model && r.agent === "scripted" ? "a script, not a model," : `the ${playerWords(e)}`;
  const server = r.diagnostics_server ? "with the CausalDiagnosis MCP" : "without the CausalDiagnosis MCP";
  return `${r.agent === "scripted" ? "A scripted episode" : "An episode"}, played by ${who} ${server}.${r.model ? "" : " A made-up example, not an experimental result."}`;
}
function viewFigure() {
  // the appendix figure: the episode timeline at 1000 px with every text at 15 px or more, public values only
  const e = ep(), M = rankingCalls().length ? storyModel() : null;
  return h("section", { class: "figure fig-reason" },
    h("div", { class: "fig-title" },
      h("h1", {}, "One episode, step by step over the machine's time"),
      h("p", {}, figureLine(e))),
    tlHeadline(e), timelineView({ fig: true }));
}

/* ---------- Reasoning tab helpers: rankings, plans, the reads between two rankings, the chain behind the action ---------- */
const PLANS = "ep/plans.json";
function plansDoc() { try { return FILES[PLANS] ? JSON.parse(FILES[PLANS]) : null; } catch (err) { return null; } }
function rankingCalls() { return ep().calls.filter((c) => c.kind === "advice" && c.answer && Array.isArray(c.answer.candidates)); }
// how many causes the CausalDiagnosis MCP ranks: n_candidates of a raw or detailed answer, else every level combination of the
// machine (n_combinations of the data), which an answer at the level concise does not repeat
function nCauses(a) { return typeof (a || {}).n_candidates === "number" ? a.n_candidates : ep().n_combinations; }
function readsBetween(a, b) { return ep().calls.filter((c) => c.n > a && c.n < b && c.served && c.served.length); }
function keyOf(comb) { return JSON.stringify(Object.keys(comb).sort().map((k) => [k, comb[k]])); }
/* the node ids of the chain behind the final action, for the decision graph's emphasis */
function chainIds() {
  const e = ep(), ids = new Set();
  const diag = e.calls.find((c) => c.answer && c.answer.result && c.answer.result.diagnosis);
  if (!diag) return ids;
  ids.add(`call/${diag.n}`);
  for (const x of String(diag.args.evidence_ids || "").split(",").map((s) => s.trim()).filter(Boolean)) ids.add(x.split("/").slice(0, 2).join("/"));
  const before = rankingCalls().filter((r) => r.n < diag.n).pop();
  if (before) ids.add(`call/${before.n}`);
  const orders = e.calls.filter((c) => c.kind === "orders");
  for (const o of orders) { const w = o.answer && o.answer.result && o.answer.result.work_order; if (w) { ids.add(`wo/${w.id}`); ids.add(`ev/work_order_executed/${w.id}`); ids.add(`ev/work_order_closed/${w.id}`); } if (o.args.prediction_id) ids.add(o.args.prediction_id); }
  for (const c of e.calls) if (c.tool === "verify_outcome") ids.add(`call/${c.n}`);
  return ids;
}

/* ---------- the reasoning view: one branch graph ----------
   The view reads one story model built from the recorded files only (storyModel). No number is on the canvas;
   a click on a mark opens a small panel with its numbers and a link to its record. */
const GROUP_WORDS = { full: "all sensors", basic: "the basic sensors" };
function groupWords(g) { return GROUP_WORDS[g] || `sensor set "${g}"`; }
function readWords(c) { return `read ${groupWords(c.args.sensor_group)}, ${c.args.cycles} cycle${c.args.cycles === 1 ? "" : "s"}`; }
function andList(xs) { return xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`; }
function faultySet(comb) { const r = rules(); return comps().filter((k) => comb[k] !== r.healthy[k]); }
function storyModel() {
  const e = ep(), R = rankingCalls();
  const diag = e.calls.find((c) => c.kind === "decide" && c.answer && c.answer.result && c.answer.result.diagnosis);
  const pre = diag ? R.filter((r) => r.n < diag.n) : R;
  const orders = e.calls.filter((c) => c.kind === "orders");
  const lastOrder = orders.length ? orders[orders.length - 1].n : null;
  const closes = e.closes;
  const lastClose = closes.length ? Math.max(...closes.map((x) => x.call)) : null;
  const afterReads = lastClose ? e.calls.filter((c) => c.n > lastClose && c.kind === "reads" && c.served && c.served.length) : [];
  const check = e.calls.find((c) => c.tool === "verify_outcome");
  const P = plansDoc();
  // lanes: every candidate a ranking before the diagnosis lists, first by first appearance
  const lanes = [];
  pre.forEach((rk) => rk.answer.candidates.forEach((x, j) => {
    const k = keyOf(x.combination);
    let L = lanes.find((l) => l.key === k);
    if (!L) { L = { key: k, comb: x.combination, label: combText(x.combination), faulty: faultySet(x.combination), at: {} }; lanes.push(L); }
    L.at[rk.n] = { p: x.probability, rank: x.rank, j };
  }));
  const n0 = pre.length ? nCauses(pre[0].answer) : 0;
  // when a lane is ruled out: the first ranking that lists it below .001, or that drops it while listing only candidates below .001
  for (const L of lanes) {
    L.first = pre.find((rk) => L.at[rk.n]) || null;
    for (let i = 0; i < pre.length; i++) {
      const rk = pre[i], cur = L.at[rk.n];
      const floorLow = Math.min(...rk.answer.candidates.map((x) => x.probability)) < 1e-3;
      const seenBefore = pre.slice(0, i).some((q) => L.at[q.n]);
      if ((cur && cur.p < 1e-3) || (!cur && seenBefore && floorLow)) { L.out = { at: rk, before: pre[i - 1] || null, listed: !!cur }; break; }
    }
  }
  const chosenPid = orders.length ? orders[0].args.prediction_id : null;
  const at = pre[pre.length - 1] || null;
  const branches = at ? (at.answer.branches || []).map((b, i) => ({ ...b, i, chosen: (b.repair_plan || {}).plan_prediction_id === chosenPid })) : [];
  const chosenB = branches.find((b) => b.chosen) || null;
  const chosenLane = chosenB ? lanes.find((l) => sameSet(l.faulty, chosenB.faulty_set || []) && (!l.out)) : null;
  const gaps = pre.slice(1).map((rk, i) => ({ a: pre[i], b: rk, reads: readsBetween(pre[i].n, rk.n) }));
  const sched = P ? P.plans.map((pl, k) => [pl, k]).filter(([pl]) => pl.schedule && pl.schedule.reads && pl.schedule.reads.length) : [];
  return { e, R, pre, diag, orders, closes, lastOrder, afterReads, check, lanes, n0, branches, chosenB, chosenLane, gaps, sched, at, P, chosenPid };
}
function wrapWords(text, n) { const out = []; let cur = ""; for (const w of text.split(" ")) { if (cur && (cur + " " + w).length > n) { out.push(cur); cur = w; } else cur = cur ? cur + " " + w : w; } if (cur) out.push(cur); return out; }
function sameSet(a, b) { return a.length === b.length && a.every((x) => b.includes(x)); }
function storySentence(M) {
  const parts = [];
  if (M.pre.length) parts.push(`${M.n0} candidate root causes at the start.`);
  const g = M.gaps[0];
  if (g && g.reads.length) {
    const eff = Math.round(g.b.answer.effective_hypotheses);
    parts.push(`${g.reads.length === 1 ? "One read" : `${g.reads.length} reads`} of ${g.reads.map((r) => `${r.args.cycles} cycle${r.args.cycles === 1 ? "" : "s"}`).join(" and ")} left ${eff}.`);
  }
  if (M.orders.length) {
    const comp = [...new Set(M.orders.map((o) => o.args.component))].map((x) => `the ${x}`);
    let s1 = `The agent replaced ${andList(comp)}`;
    if (M.check && M.check.answer.status === "verified") s1 += ", and the reading after the repair confirmed it.";
    else if (M.check) {
      // say when the check came against when the orders closed, as the record has them, rather than quote the status word
      const c = M.check, lives = orderLives(M.e).filter((l) => l.call < c.n);
      const open = lives.filter((l) => l.close == null || l.close > c.time_before);
      const closes = open.map((l) => l.close).filter((x) => x != null);
      s1 += `. It asked for the check of the repair at step ${c.n}, in cycle ${c.time_before}${open.length ? `, while the work order was still open` : ""}.`;
      if (closes.length) s1 += ` The order closed in cycle ${Math.max(...closes)}.`;
    } else s1 += ", and no check of the outcome is recorded.";
    parts.push(s1);
  }
  return parts.join(" ");
}
function infoMark(text) { return h("button", { type: "button", class: "info", "aria-label": "How this is joined", title: text, onclick: (ev) => { ev.currentTarget.classList.toggle("open"); } }, "i", h("span", { class: "info-tx", role: "note" }, text)); }
function legend(items) { return h("ul", { class: "lg" }, items.map(([cls, t]) => h("li", {}, h("span", { class: `lg-m ${cls}`, "aria-hidden": "true" }), t))); }

/* Words for a candidate, in the asset's own component names: "<component> at level 1, <component> at level 2". */
function candWords(comb) {
  const f = faultySet(comb);
  return f.length ? f.map((k) => `${k} at level ${comb[k]}`).join(", ") : "every component healthy";
}
/* The state of a candidate at a ranking, from the server's full belief when the episode holds belief.json (a side file
   the server writes for the viewer and no agent reads), else from the answer alone:
   listed: plausible at .001 or more, ruled out below; not listed: plausible when the plausible count covers every
   candidate, ruled out when the listed plausible candidates already account for the plausible count, else open. */
const BELIEF = "ep/belief.json";
function beliefDoc() { try { return FILES[BELIEF] ? JSON.parse(FILES[BELIEF]) : null; } catch (err) { return null; } }
function laneState(comb, rk) {
  const B = beliefDoc();
  const full = B && (B.rankings || []).find((r) => r.step === rk.n);
  if (full) { const x = full.candidates.find((y) => keyOf(y.combination) === keyOf(comb)); return x ? (x.probability >= 1e-3 ? "on" : "out") : "open"; }
  const listed = rk.answer.candidates.find((x) => keyOf(x.combination) === keyOf(comb));
  if (listed) return listed.probability >= 1e-3 ? "on" : "out";
  const eff = Math.round(rk.answer.effective_hypotheses), n = nCauses(rk.answer);
  const plausListed = rk.answer.candidates.filter((x) => x.probability >= 1e-3).length;
  if (eff >= n) return "on";
  if (plausListed >= eff) return "out";
  return "open";
}
function groupState(states) { return states.every((s) => s === states[0]) ? states[0] : "open"; }
const ACTOR = { rank: ["advice", "server"], read: ["reads", "asset"], diag: ["decide", "agent"], orders: ["orders", "agent"], closes: ["reads", "asset"], check: ["advice", "server"] };
function colSentence(c, M) {
  if (c.kind === "rank") { const a = c.c.answer; return `The server ranked the candidates: ${Math.round(a.effective_hypotheses)} of ${nCauses(a)} still plausible, ${candWords(a.candidates[0].combination)} first.`; }
  if (c.kind === "read") return `The asset returned ${c.c.args.cycles} cycle${c.c.args.cycles === 1 ? "" : "s"} of ${groupWords(c.c.args.sensor_group)}${c.after ? " after the repair" : ""}.`;
  if (c.kind === "diag") return `The agent diagnosed ${candWords(c.c.args)}.`;
  if (c.kind === "orders") return `The agent ordered ${c.cs.length === 1 ? "a repair" : "repairs"} of ${andList(c.cs.map((o) => `the ${o.args.component}`))}.`;
  if (c.kind === "closes") return c.xs.map((x, i) => `${i ? "the" : "The"} ${x.component || x.id} repair finished at cycle ${x.t}`).join(", ") + ".";
  if (c.kind === "check") return c.c.answer.status === "verified" ? "The server checked the outcome and confirmed the repair." : `The server checked the outcome: ${c.c.answer.status}.`;
  return "";
}
function colRows(c) {
  if (c.kind === "rank") { const a = c.c.answer; return [["plausible candidates", `${num(a.effective_hypotheses, 1)} of ${nCauses(a)}`], ["listed in the answer", String(a.candidates.length)], ["cycle", String(a.belief_time)]]; }
  if (c.kind === "read") return [["observations", c.c.served.map((o) => `obs/${o.n}`).join(", ")], ["price", String(c.c.price)], ...(c.serves ? [["tests the candidates ranked", c.serves.slice().sort((a, b) => a - b).join(", ")]] : [])];
  if (c.kind === "diag") return [["evidence cited", `${String(c.c.args.evidence_ids || "").split(",").filter(Boolean).length} channel values`]];
  if (c.kind === "orders") return c.cs.map((o) => [o.args.component, `prediction ${o.args.prediction_id || "none"}`]);
  if (c.kind === "closes") return c.xs.map((x) => [x.component || x.id, `work order ${x.id}, cycle ${x.t}`]);
  if (c.kind === "check") return [["outcome", c.c.answer.status || NR], ["prediction checked", c.c.answer.prediction_id || NR]];
  return [];
}
function colRefs(c) {
  if (c.kind === "rank") return [rAnswer(c.c.n, ["candidates"])];
  if (c.kind === "read") return [rAnswer(c.c.n, ["result"])];
  if (c.kind === "diag") return [rCall(c.c.n, ["tool_args"])];
  if (c.kind === "orders") return c.cs.map((o) => rCall(o.n, ["tool_args"]));
  if (c.kind === "closes") return [...new Set(c.xs.map((x) => x.call))].map((n) => rAnswer(n, ["notifications"]));
  if (c.kind === "check") return [rAnswer(c.c.n, ["status"])];
  return [];
}
/* the panel: the plain sentence first, then its numbers, then the record collapsed */
function openMark(panel, sentence, rows, refs, note) {
  panel.hidden = false;
  const R = (refs || []).filter(Boolean);
  panel.replaceChildren(...[
    h("div", { class: "mk-h" }, h("p", { class: "mk-sent" }, sentence), h("button", { type: "button", class: "mini", "aria-label": "Close", onclick: () => { panel.hidden = true; } }, "Close")),
    rows.length ? h("dl", { class: "mk-kv" }, rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])) : null,
    note ? h("p", { class: "mk-note" }, note) : null,
    R.length ? h("details", { class: "mk-rec" }, h("summary", {}, "The record"), R.map((r) => readableRec(r.m ? { ...r, p: r.m, m: null } : r)),
      h("button", { type: "button", class: "mini", onclick: () => showRaw(sentence, R, note) }, "Open in the inspector")) : null].filter(Boolean));
}
function mark(el, panel, sentence, rows, refs, note) {
  el.setAttribute("tabindex", "0"); el.setAttribute("role", "button"); el.setAttribute("aria-label", sentence); el.classList.add("mk");
  el.append(s("title", {}, sentence));
  const go1 = (ev) => { ev.stopPropagation(); openMark(panel, sentence, rows, refs, note); };
  el.addEventListener("click", go1);
  el.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go1(ev); } });
  return el;
}
function headline(M) {
  const last = M.pre[M.pre.length - 1];
  const facts = [];
  if (M.pre.length) facts.push(["f-elim", `${M.n0} candidates → ${last ? Math.round(last.answer.effective_hypotheses) : NR}`]);
  if (M.orders.length) facts.push(["f-orders", `${M.orders.length} work order${M.orders.length === 1 ? "" : "s"}`]);
  if (M.check) facts.push([M.check.answer.status === "verified" ? "f-ok" : "f-bad", M.check.answer.status === "verified" ? "confirmed" : M.check.answer.status]);
  return h("div", { class: "rs-head2" }, h("p", { class: "rs-story" }, storySentence(M)), h("div", { class: "rs-facts" }, facts.map(([c, t]) => h("span", { class: `fact ${c}` }, t))));
}
function tabReasoning() {
  const R = rankingCalls(), e = ep(), A = tlAxis(e, {});
  const M = R.length ? storyModel() : null;
  return h("div", { class: "reasoning r4 r6" }, M ? tlHeadline(e) : null, timelineView({}));
}

/* the branch graph; opts.fig sets the type at 15.5 px and the width at 1000 px for the appendix figure */
function branchGraph(M, opts) {
  const fig = !!opts.fig;
  const FS = fig ? { verb: 15.5, step: 15, lab: 15.5 } : { verb: 12, step: 11, lab: 13 };
  const panel = h("div", { class: "mk-panel", hidden: "", role: "dialog", "aria-label": "Details" });
  const cols = [];
  M.pre.forEach((rk, i) => {
    if (i) for (const r of M.gaps[i - 1].reads) cols.push({ kind: "read", c: r, verb: `Read ${r.args.cycles} cycle${r.args.cycles === 1 ? "" : "s"}` });
    cols.push({ kind: "rank", c: rk, verb: "Server ranks" });
  });
  if (M.diag) cols.push({ kind: "diag", c: M.diag, verb: "Agent diagnoses" });
  if (M.orders.length) cols.push({ kind: "orders", cs: M.orders, verb: `${M.orders.length} work order${M.orders.length === 1 ? "" : "s"}` });
  if (M.closes.length) cols.push({ kind: "closes", xs: M.closes, verb: "Repairs finish" });
  for (const r of M.afterReads) cols.push({ kind: "read", c: r, verb: "Read after repair", after: true });
  if (M.check) cols.push({ kind: "check", c: M.check, verb: "Server checks" });
  const sched = M.sched.flatMap(([pl]) => pl.schedule.reads);
  cols.forEach((c) => { if (c.kind === "read" && !c.after) { const sc = sched.find((r) => r.sensor_group === c.c.args.sensor_group && r.cycles === c.c.args.cycles); c.serves = sc && sc.serves ? sc.serves : null; } });
  // lanes: the chosen candidate first, then the others a ranking listed, then the rest in one row
  const chosen = M.chosenLane;
  const others = M.lanes.filter((l) => l !== chosen);
  const SHOW = 3;
  const open = !!S.rsMoreOpen && !fig;
  const shown = [chosen, ...(open ? others : others.slice(0, SHOW))].filter(Boolean);
  const hiddenLanes = open ? [] : others.slice(SHOW);
  const unlisted = Math.max(0, M.n0 - M.lanes.length);
  const moreCount = hiddenLanes.length + unlisted;
  const rows = shown.map((L) => ({ L, label: candWords(L.comb), states: M.pre.map((rk) => laneState(L.comb, rk)) }));
  if (moreCount) {
    const unlistedStates = M.pre.map((rk) => laneState({ __unlisted: true }, rk));
    const states = M.pre.map((rk, i) => groupState([...hiddenLanes.map((L) => laneState(L.comb, rk)), ...(unlisted ? [unlistedStates[i]] : [])]));
    rows.push({ more: true, label: open || !hiddenLanes.length ? `${unlisted} more, never listed by the server` : `${moreCount} more candidates`, states });
  }
  const LW = fig ? 262 : 214, GAP = fig ? 92 : 84, LH = fig ? 44 : 38, TOP = fig ? 108 : 92;
  const W = LW + (cols.length - 1) * GAP + 50, H = TOP + rows.length * LH + 16;
  const X = (i) => LW + i * GAP, Y = (j) => TOP + j * LH + LH / 2;
  const svg = s("svg", { class: ["bg", fig ? "bg-fig" : null], viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "How the candidates were narrowed, and the path of the one acted on" });
  svg.style.width = `${W}px`;
  const rankCol = (rk) => cols.findIndex((c) => c.kind === "rank" && c.c.n === rk.n);
  const diagCol = cols.findIndex((c) => c.kind === "diag");
  const lastPre = M.pre.length ? rankCol(M.pre[M.pre.length - 1]) : 0;
  const gCols = s("g"), gLanes = s("g", { class: "asp asp-elim" }), gPar = s("g", { class: "asp asp-par" }), gChain = s("g", { class: "asp asp-chain" }), gReads = s("g", { class: "asp asp-read" });
  // column heads: one chip per step, colored by who acted, with the step number small; the whole column opens its sentence
  cols.forEach((c, i) => {
    const [kind, actor] = ACTOR[c.kind];
    const step = c.c ? c.c.n : c.cs ? c.cs.map((o) => o.n).join(", ") : [...new Set(c.xs.map((x) => x.call))].join(", ");
    const lines = wrapWords(c.verb, fig ? 11 : 10);
    const cw = Math.min(GAP - 8, Math.max(...lines.map((l) => l.length)) * FS.verb * 0.6 + 16), chH = lines.length * (FS.verb + 3) + 10;
    const y0 = TOP - 26 - chH - (fig ? 6 : 0);
    const g = mark(s("g", { class: `bg-chip k-${kind} actor-${actor}` }), panel, colSentence(c, M), colRows(c), colRefs(c));
    g.append(s("rect", { x: X(i) - GAP / 2 + 2, y: y0 - 4, width: GAP - 4, height: H - y0 + 2, class: "bg-colhit" }),
      s("rect", { x: X(i) - cw / 2, y: y0, width: cw, height: chH, rx: 7, class: "bg-chipbox" }),
      ...(fig ? [] : [s("circle", { cx: X(i) - cw / 2 + 2, cy: y0 + 2, r: 7.5, class: "bg-glyphc" }),
        s("text", { x: X(i) - cw / 2 + 2, y: y0 + 5.6, "text-anchor": "middle", class: "bg-glyph" }, KINDS[kind].glyph)]));
    lines.forEach((ln, k) => g.append(s("text", { x: X(i), y: y0 + 4 + FS.verb * 0.9 + k * (FS.verb + 3), "text-anchor": "middle", class: "bg-verb" }, ln)));
    g.append(s("text", { x: X(i), y: TOP - 10, "text-anchor": "middle", class: "bg-step" }, `step ${step}`));
    gCols.append(g);
  });
  // read bands: a read is evidence for every candidate, so its band spans every lane
  cols.forEach((c, i) => {
    if (c.kind !== "read") return;
    const y0 = TOP + 3, y1 = TOP + rows.length * LH - 3;
    const band = mark(s("rect", { x: X(i) - 13, y: y0, width: 26, height: y1 - y0, rx: 7, class: `bg-band${c.after ? " after" : ""}` }), panel, colSentence(c, M), colRows(c), colRefs(c),
      c.serves ? "The agent planned this one read for the tests of several candidates at once." : null);
    (c.after ? gChain : gReads).append(band);
  });
  // lanes
  rows.forEach((row, j) => {
    const y = Y(j);
    const isChosen = row.L && row.L === chosen;
    const parts = row.label.length > (fig ? 26 : 24) && row.label.includes(", ") ? [row.label.slice(0, row.label.indexOf(", ") + 1), row.label.slice(row.label.indexOf(", ") + 2)] : [row.label];
    const lab = s("text", { x: 8, y: y + FS.lab * 0.35 - (parts.length - 1) * FS.lab * 0.55, class: ["bg-lab", row.more ? "more" : null, isChosen ? "chosen" : null] },
      parts.map((pt, k) => s("tspan", { x: 8, dy: k ? FS.lab * 1.1 : 0 }, pt)));
    if (row.more && !fig) {
      const g = s("g", { class: "bg-more", tabindex: "0", role: "button", "aria-expanded": String(open) }, lab, s("title", {}, open ? "Collapse the list" : "Show every candidate"));
      const tog = () => { S.rsMoreOpen = !S.rsMoreOpen; render(); };
      g.addEventListener("click", tog); g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); tog(); } });
      gCols.append(g);
    } else gCols.append(lab);
    const ranks = M.pre.map((rk) => rankCol(rk));
    let xPrev = X(0) - 30, prevState = row.states[0], outX = null;
    ranks.forEach((ci, k) => {
      const st = row.states[k];
      if (k === 0) { gLanes.append(s("line", { x1: xPrev, x2: X(ci), y1: y, y2: y, class: `bg-lane st-${st}${isChosen ? " chosen" : ""}` })); xPrev = X(ci); prevState = st; return; }
      if (prevState !== "out" && st === "out") {
        outX = X(ci) - 30;
        gLanes.append(s("line", { x1: xPrev, x2: outX, y1: y, y2: y, class: `bg-lane st-${prevState}` }));
        const sentence = `${row.more ? `These ${row.label.replace(/^\d+ /, "")}` : candWords(row.L.comb)}: ruled out at the ranking of step ${M.pre[k].n}.`;
        gLanes.append(mark(s("g", { class: "bg-cross" }, s("path", { d: `M${outX - 5},${y - 5} L${outX + 5},${y + 5} M${outX - 5},${y + 5} L${outX + 5},${y - 5}` }), s("rect", { x: outX - 10, y: y - 10, width: 20, height: 20, class: "hit" })),
          panel, sentence, row.L && row.L.at[M.pre[k].n] ? [["probability", pct(row.L.at[M.pre[k].n].p)]] : [["listed", "no: below every candidate the server listed"]],
          row.L && row.L.at[M.pre[k].n] ? [rAnswer(M.pre[k].n, ["candidates", row.L.at[M.pre[k].n].j])] : [rAnswer(M.pre[k].n, ["candidates"])],
          "The belief exists only at a ranking, so the cross sits after the reads that came between two rankings."));
        gLanes.append(s("line", { x1: outX + 8, x2: X(lastPre) + 8, y1: y, y2: y, class: "bg-lane st-gone" }));
      } else if (prevState !== "out") gLanes.append(s("line", { x1: xPrev, x2: X(ci), y1: y, y2: y, class: `bg-lane st-${st === "out" ? prevState : st}${isChosen ? " chosen" : ""}` }));
      xPrev = X(ci); prevState = st;
    });
    // a dot where a ranking lists the candidate as plausible
    if (row.L) for (const rk of M.pre) { const a = row.L.at[rk.n]; if (a && a.p >= 1e-3) gLanes.append(mark(s("circle", { cx: X(rankCol(rk)), cy: y, r: fig ? 4.5 : 3.8, class: "bg-dot" }), panel, `The server ranked ${candWords(row.L.comb)} ${a.rank === 1 ? "first" : `number ${a.rank}`}.`, [["probability", pct(a.p)]], [rAnswer(rk.n, ["candidates", a.j])])); }
    // squares for the repairs the server scored at the last ranking before the diagnosis; the acted one filled
    if (row.L && M.at) {
      const ci = rankCol(M.at);
      for (const b of M.branches) {
        if (!sameSet(row.L.faulty, b.faulty_set || []) || !row.L.at[M.at.n]) continue;
        const rp = b.repair_plan || {}, pred = rp.plan_prediction_id ? M.e.predictions.find((q) => q.id === rp.plan_prediction_id) : null;
        const sz = fig ? 13 : 12;
        gPar.append(mark(s("rect", { x: X(ci) + 9, y: y - sz / 2, width: sz, height: sz, rx: 2, class: `bg-rep${b.chosen ? " acted" : ""}` }), panel,
          b.chosen ? `The agent acted on the repair of ${andList(b.faulty_set || [])}.` : `The server also scored a repair of ${andList(b.faulty_set || [])}; it was not chosen.`,
          [["probability", pct(b.probability)], ["price of the repair", rp.price == null ? NR : num(rp.price, 1)], ["cycles to restored", rp.cycles_to_restored == null ? NR : String(rp.cycles_to_restored)],
            ["predicted probability of recovery", pred ? pct(pred.recovery) : NR], ["expected loss if acted", num(b.expected_loss_if_acted, 1)],
            ...(!b.chosen && M.chosenB ? [["the repair acted on", `expected loss if acted ${num(M.chosenB.expected_loss_if_acted, 1)}`]] : [])],
          [rAnswer(M.at.n, ["branches", b.i]), pred ? rPred(pred.id) : null]));
      }
    }
    // the chosen candidate runs on from its last ranking to the check, the strongest mark on the page
    if (isChosen && diagCol >= 0) {
      gChain.append(s("line", { x1: X(lastPre), x2: X(cols.length - 1), y1: y, y2: y, class: "bg-path" }));
      cols.forEach((c, i) => {
        const sent = colSentence(c, M), rws = colRows(c), refs = colRefs(c);
        if (c.kind === "diag") gChain.append(mark(s("circle", { cx: X(i), cy: y, r: fig ? 8 : 7, class: "bg-diag" }), panel, sent, rws, refs));
        if (c.kind === "orders") c.cs.forEach((o, k) => gChain.append(mark(s("rect", { x: X(i) - 13 + k * 14, y: y - 6, width: 12, height: 12, rx: 2, class: "bg-order" }), panel, `The agent ordered a repair of the ${o.args.component}.`, [["prediction for this repair", o.args.prediction_id || "none"]], [rCall(o.n, ["tool_args"])])));
        if (c.kind === "closes") c.xs.forEach((x, k) => gChain.append(mark(s("circle", { cx: X(i) - 7 + k * 14, cy: y, r: 5, class: "bg-close" }), panel, `The ${x.component || x.id} repair finished at cycle ${x.t}.`, [["work order", x.id]], [rAnswer(x.call, ["notifications"])])));
        if (c.kind === "check") gChain.append(mark(s("g", { class: `bg-check ${c.c.answer.status === "verified" ? "ok" : "bad"}` }, s("circle", { cx: X(i), cy: y, r: fig ? 10 : 9 }),
          s("path", { d: c.c.answer.status === "verified" ? `M${X(i) - 4},${y} L${X(i) - 1},${y + 3} L${X(i) + 4},${y - 3}` : `M${X(i) - 3},${y - 3} L${X(i) + 3},${y + 3} M${X(i) - 3},${y + 3} L${X(i) + 3},${y - 3}` })), panel, sent, rws, refs));
      });
    }
  });
  svg.append(gCols, gReads, gLanes, gPar, gChain);
  const anyOpen = rows.some((r) => r.states.includes("open"));
  const leg = legend([["l-lane", "a candidate still plausible"], ...(anyOpen ? [["l-open", "the files leave it open"]] : []), ["l-cross", "ruled out"], ["l-band", "one read: evidence for every candidate"], ["l-rep", "a repair the server scored (filled: acted on)"], ["l-path", "the candidate acted on, to the check"]]);
  if (fig) return h("div", { class: "rs-a" }, h("div", { class: "bg-wrap" }, svg), leg);
  const aspects = [["all", "Everything", ""],
    ["elim", "Elimination", `${M.n0} candidates were plausible at first. After the read${M.gaps[0] && M.gaps[0].reads.length === 1 ? "" : "s"} before the second ranking, ${M.gaps[0] ? Math.round(M.gaps[0].b.answer.effective_hypotheses) : NR} remained; each cross marks a candidate ruled out.`],
    ["par", "Parallel work", `One read served the tests of several candidates at once, and the server scored ${M.branches.length} repairs side by side; the filled square is the one acted on.`],
    ["chain", "From reading to check", "The thick line follows the candidate acted on, from its ranking through the diagnosis, the work orders, the finished repairs and the read after the repair, to the check."]];
  const asp = S.rsAspect || "all";
  svg.classList.add(`focus-${asp}`);
  const chips = h("div", { class: "seg chips", role: "group", "aria-label": "Show one aspect" }, aspects.map(([k, t]) => h("button", { type: "button", "aria-pressed": String(asp === k), onclick: () => { S.rsAspect = k; render(); } }, t)));
  return h("section", { class: "rs-sec rs-a" },
    h("div", { class: "rs-bar" }, chips, infoMark(beliefDoc() ? "The states come from the server's full belief at each ranking (belief.json)." : "Without the server's full belief, a candidate the answer does not list is plausible when the plausible count covers every candidate, ruled out when the listed candidates already account for it, and open otherwise. A repair square sits on every listed candidate whose faulty components equal the repair's.")),
    h("p", { class: "rs-asp", "aria-live": "polite" }, asp === "all" ? "Click a step, a line or a mark for what happened there." : aspects.find((a) => a[0] === asp)[2]),
    h("div", { class: "bg-wrap" }, svg), panel, leg);
}

/* ---------- the inspector reads like an editor, and every record has a readable face ---------- */
/* One table from keys to words, in the paper's terms. A key not in the table shows as itself, so the records of any
   asset stay readable. */
const KEY_WORDS = {
  time: "cycle", budget_left: "budget left", steps_left: "steps left", open_work_orders: "open work orders", episode_over: "episode over",
  notifications: "notifications", result: "result", served: "readings served", values: "values", sensor_group: "sensor set", cycles: "cycles",
  evidence_ids: "evidence", time_window: "time window", assumptions: "assumptions", uncertainty: "uncertainty", message: "message",
  ranking_model: "model the CausalDiagnosis MCP ranks with", n_candidates: "possible causes", n_states: "states", candidates: "possible causes", combination: "cause",
  current_combination: "cause now", probability: "probability", rank: "rank", effective_hypotheses: "possible causes still left",
  current_fault_probability: "chance a fault is present now", elimination: "reads the CausalDiagnosis MCP can advise", read_batch: "read batch",
  branches: "repairs it scored", faulty_set: "faulty parts", repair_plan: "repair plan", price: "cost", cycles_to_restored: "cycles until the machine works again",
  plan_prediction_id: "prediction for this repair", cost_to_restored: "predicted cost until the machine works again", act_now: "if it repairs now",
  expected_loss_if_acted: "expected cost if it repairs now", expected_loss_if_tested: "expected cost if it reads more first", advice: "advice",
  prediction_id: "prediction", action: "repair", status: "status", created_t: "opened at cycle", expected_close_t: "expected to close at cycle",
  closed_t: "done at cycle", visit: "visit", mode: "mode", source: "source", recovery: "chance the machine works again", recovery_interval: "range",
  alpha: "error level", events: "events", observations: "readings", addresses: "chance it fixes a fault", component: "part", id: "id",
  tool_args: "arguments sent", tool_name: "tool", response: "answer", effective_before: "plausible before", expected_after: "expected after",
  realized_after: "plausible after", downtime: "output lost", cost: "predicted cost", diagnosis: "diagnosis", work_order: "work order",
};
const STAT_WORDS = { mean: "mean", std: "standard deviation", min: "minimum", max: "maximum" };
function keyWords(k) { return KEY_WORDS[k] || String(k).replace(/_/g, " "); }
function channelWord(ch) { const p = String(ch).split("_"); return (p[1] || p[0]).toLowerCase(); }
/* a list of observation value ids, grouped and counted */
function idGroup(ids) {
  const parsed = ids.map((x) => /^obs\/(\d+)\/(.+)_(mean|std|min|max)$/.exec(x)).filter(Boolean);
  if (parsed.length !== ids.length || !ids.length) return null;
  const obs = [...new Set(parsed.map((m) => +m[1]))].sort((a, b) => a - b);
  const chans = [...new Set(parsed.map((m) => channelWord(m[2])))], stats = [...new Set(parsed.map((m) => STAT_WORDS[m[3]] || m[3]))];
  const from = obs.length === 1 ? `reading ${obs[0]}` : `readings ${obs[0]} to ${obs[obs.length - 1]}`;
  return h("span", {}, `${ids.length} values from ${from}, covering ${andList(chans)}, ${andList(stats)}${chans.length > 1 ? " each" : ""} `,
    h("details", { class: "rd-ids" }, h("summary", {}, "the full list"), h("div", { class: "mono" }, ids.join(", "))));
}
function parseMaybeJSON(x) { if (typeof x !== "string") return x; const t = x.trim(); if (!/^[[{]/.test(t)) return x; try { return JSON.parse(t); } catch (err) { return x; } }
function readableValue(v, key, depth) {
  v = parseMaybeJSON(v);
  if (v === null || v === undefined) return h("span", { class: "rd-none" }, "none");
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return num(v);
  if (typeof v === "string") {
    if (key === "action" && /^replace:/.test(v)) return `replace ${andList(v.slice(8).split(","))}`;
    if (/(^|,)obs\/\d+\//.test(v)) { const ids = v.split(",").map((x) => x.trim()).filter(Boolean); const obsIds = ids.filter((x) => x.startsWith("obs/")); const g = idGroup(obsIds); const rest = ids.filter((x) => !x.startsWith("obs/")); if (g) return h("span", {}, rest.length ? `${rest.join(", ")}; ` : "", g); }
    return v.length > 240 ? h("span", {}, v.slice(0, 240), h("details", { class: "rd-ids" }, h("summary", {}, "the rest"), v.slice(240))) : v;
  }
  if (Array.isArray(v)) {
    if (!v.length) return h("span", { class: "rd-none" }, "none");
    if (v.every((x) => typeof x === "string")) { const g = idGroup(v); if (g) return g; return v.length > 6 ? h("span", {}, `${v.length} items: ${v.slice(0, 4).join(", ")} `, h("details", { class: "rd-ids" }, h("summary", {}, "all"), v.join(", "))) : v.join(", "); }
    if (v.every((x) => typeof x === "number")) return v.length === 2 && key === "recovery_interval" ? `${num(v[0])} to ${num(v[1])}` : v.map((x) => num(x)).join(", ");
    if (depth > 2) return `${v.length} items`;
    return h("details", { class: "rd-list", open: v.length <= 3 && depth < 2 ? "" : null }, h("summary", {}, `${keyWords(key || "items")} (${v.length})`),
      h("ol", {}, v.map((x) => h("li", {}, readableValue(x, null, depth + 1)))));
  }
  if (depth > 3) return `${Object.keys(v).length} fields`;
  return readableObject(v, depth + 1);
}
/* sentences for groups of fields that read better together */
function readableObject(o, depth = 0) {
  const rows = [], used = new Set();
  if (typeof o.action === "string" && /^replace:/.test(o.action)) { rows.push(["Repair to make", `replace ${andList(o.action.slice(8).split(","))}`]); used.add("action"); }
  if (typeof o.recovery === "number") {
    const iv = Array.isArray(o.recovery_interval) ? `, between ${num(o.recovery_interval[0])} and ${num(o.recovery_interval[1])}` : "";
    rows.push(["Chance the machine works again", `${o.recovery.toFixed(2)}${iv}${typeof o.alpha === "number" ? ` at error level ${o.alpha}` : ""}`]);
    ["recovery", "recovery_interval", "alpha"].forEach((k) => used.add(k));
  }
  if (typeof o.observations === "number" && typeof o.time === "number") { rows.push(["Made", `after ${plural(o.observations, "reading")}, at cycle ${o.time}`]); used.add("observations"); used.add("time"); }
  if (o.combination && typeof o.probability === "number") { rows.push(["Possible cause", `${combText(o.combination)}, probability ${pct(o.probability)}${o.rank ? `, rank ${o.rank}` : ""}`]); ["combination", "probability", "rank", "current_combination"].forEach((k) => used.add(k)); }
  for (const [k, v] of Object.entries(o)) if (!used.has(k)) rows.push([keyWords(k), readableValue(v, k, depth)]);
  return h("dl", { class: "rd" }, rows.flatMap(([k, v]) => [h("dt", {}, k[0].toUpperCase() + k.slice(1)), h("dd", {}, v)]));
}
function refValue(ref) {
  const r = resolveRef(ref);
  if (!r || ref.l) return { r, ok: false };
  try {
    if (r.dec) {
      let v = JSON.parse(r.dec.str);
      for (const k of ref.q || []) v = v == null ? v : v[k];
      return { r, ok: true, v };
    }
    return { r, ok: true, v: JSON.parse(r.text.slice(r.start, r.end)) };
  } catch (err) { return { r, ok: false }; }
}
function readableRec(ref) {
  if (!ref) return h("p", { class: "nr" }, NR);
  if (ref.board != null) return boardFace(ep().boards[ref.board]);
  const { ok, v } = refValue(ref);
  return ok ? h("div", { class: "rd-face" }, typeof v === "object" && v !== null && !Array.isArray(v) ? readableObject(v) : readableValue(v, ref.q ? ref.q[ref.q.length - 1] : null, 0)) : rawView(ref);
}
/* syntax colors by token kind, as an editor shows JSON; strings never span lines in these files */
const TOK = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false)\b|\b(null)\b|([{}[\],:])/g;
function tokLine(text, isJson) {
  if (!isJson) return [text];
  const out = []; let last = 0, m;
  TOK.lastIndex = 0;
  while ((m = TOK.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1]) { out.push(h("span", { class: m[2] ? "tk-key" : "tk-str" }, m[1])); if (m[2]) out.push(h("span", { class: "tk-p" }, m[2])); }
    else if (m[3]) out.push(h("span", { class: "tk-num" }, m[3]));
    else if (m[4]) out.push(h("span", { class: "tk-bool" }, m[4]));
    else if (m[5]) out.push(h("span", { class: "tk-null" }, m[5]));
    else out.push(h("span", { class: "tk-p" }, m[6]));
    last = TOK.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
/* the matching close line and the item count of every line that opens an object or an array */
function foldMap(text, starts, l0, l1) {
  const map = {}, stack = [];
  for (let li = l0; li <= l1; li++) {
    const a = starts[li], b = li + 1 < starts.length ? starts[li + 1] - 1 : text.length;
    const ln = text.slice(a, b).replace(/"(?:[^"\\]|\\.)*"/g, '""');
    for (const ch of ln) {
      if (ch === "{" || ch === "[") stack.push({ li, n: 0, any: false, kind: ch });
      else if (ch === "}" || ch === "]") { const o = stack.pop(); if (o && o.li !== li) map[o.li] = { end: li, n: o.any ? o.n + 1 : 0, kind: o.kind }; if (stack.length) stack[stack.length - 1].any = true; }
      else if (ch === "," && stack.length) stack[stack.length - 1].n++;
      else if (!/\s/.test(ch) && stack.length) stack[stack.length - 1].any = true;
    }
  }
  return map;
}
function codeLines(text, starts, l0, l1, m0, m1, firstNo, isJson) {
  const box = h("div", { class: "rr-code" });
  const fm = isJson ? foldMap(text, starts, l0, l1) : {};
  const sel = (li) => { const a = starts[li], b = li + 1 < starts.length ? starts[li + 1] - 1 : text.length; return m0 != null && m1 > a && m0 < b; };
  const rowsEl = [];
  for (let li = l0; li <= l1; li++) {
    const a = starts[li], b = li + 1 < starts.length ? starts[li + 1] - 1 : text.length;
    const f = fm[li];
    const tog = f ? h("button", { type: "button", class: "fold-t", "aria-expanded": "true", "aria-label": `Fold ${f.n} ${f.kind === "[" ? "items" : "fields"}` }, "▾") : h("span", { class: "fold-t none" });
    const sum = f ? h("span", { class: "fold-sum", hidden: "" }, ` ${f.n} ${f.kind === "[" ? (f.n === 1 ? "item" : "items") : (f.n === 1 ? "field" : "fields")} `) : null;
    const row = h("div", { class: ["rr-line", sel(li) ? "sel" : null] }, h("span", { class: "ln", "aria-hidden": "true" }, String(li - l0 + firstNo)), tog, h("span", { class: "tx" }, tokLine(text.slice(a, b), isJson), sum));
    rowsEl.push(row);
    if (f) {
      const setFold = (closed) => {
        tog.setAttribute("aria-expanded", String(!closed)); tog.textContent = closed ? "▸" : "▾"; sum.hidden = !closed;
        for (let k = li + 1 - l0; k <= f.end - l0 && k < rowsEl.length; k++) rowsEl[k].hidden = closed;
        if (!closed) for (let k = li + 1; k <= f.end; k++) { const g = rowsEl[k - l0]; if (g && g._closed) for (let q = k + 1 - l0; q <= g._end - l0; q++) rowsEl[q].hidden = true; }
        row._closed = closed;
      };
      row._end = f.end; row._fold = setFold;
      tog.addEventListener("click", (ev) => { ev.stopPropagation(); setFold(!row._closed); });
    }
    box.append(row);
  }
  // a long array opens folded, unless it holds the marked part
  for (let li = l0; li <= l1; li++) {
    const f = fm[li];
    if (f && f.kind === "[" && f.n > 8 && li !== l0) {
      let holds = false; for (let k = li; k <= f.end; k++) if (sel(k)) holds = true;
      if (!holds) rowsEl[li - l0]._fold(true);
    }
  }
  return box;
}

/* ---------- the episode timeline: one shared axis, four stacked panels ----------
   a. candidates still plausible, b. the calls over the cycles they take, c. the asset running or down, d. spending against
   the budget. Environment time runs left to right in cycles. A call that takes no cycle sits in a narrow band at its cycle,
   in call order, so a cycle keeps its width whatever happens in it. A row whose calls all take no cycle (the base row) is
   drawn with the steps as its axis. Hidden values are drawn only in the mode for people, inside the HID class. */
const TL_LANES = [["reads", "Reads"], ["advice", "CausalDiagnosis MCP"], ["time", "Waiting"], ["orders", "Work orders"], ["decide", "Diagnosis and finish"]];
function thinkCycles() {
  try { const E = JSON.parse(FILES["ep/episode.json"] || "null"); return E && Array.isArray(E.think_cycles) ? E.think_cycles : []; } catch (err) { return []; }
}
function orderLives(e) {
  const L = {};
  for (const o of e.orders) L[o.id] = { id: o.id, component: o.component, created: o.created_t, call: o.call, exec: null, close: null, execCall: null, closeCall: null };
  for (const c of e.calls) for (const n of c.notifications || []) {
    if (!L[n.id]) continue;
    if (n.event === "work_order_executed" && L[n.id].exec == null) { L[n.id].exec = n.t; L[n.id].execCall = c.n; }
    if (n.event === "work_order_closed" && L[n.id].close == null) { L[n.id].close = n.t; L[n.id].closeCall = c.n; }
  }
  return Object.values(L);
}
function rankAt(c) { const tw = c.answer && c.answer.time_window; return tw && typeof tw.end === "number" ? tw.end : c.time_before; }
function stepMode(e) { return e.calls.every((c) => c.kind === "time" || c.cycles === 0); }
/* the axis: x of an instant (a call that takes no cycle) and of a span of cycles */
function tlAxis(e, opt) {
  const ZW = opt.fig ? 15 : 13, PAD = 8;
  if (stepMode(e)) {
    const SW = opt.fig ? 40 : 34;
    return { step: true, W: e.calls.length * SW + PAD * 2, xCall: (c) => PAD + (c.n - 0.5) * SW, xStep: (n) => PAD + (n - 0.5) * SW, SW,
      span: (c) => [PAD + (c.n - 1) * SW + 4, PAD + c.n * SW - 4], at: (t) => null, T: 0 };
  }
  const lives = orderLives(e);
  const T = Math.max(1, e.endTime, ...e.calls.map((c) => c.time_after), ...lives.map((l) => l.close || 0));
  const zero = {};
  for (const c of e.calls) if (c.cycles === 0) (zero[c.time_before] = zero[c.time_before] || []).push(c.n);
  const zonesW = Array.from({ length: T + 1 }, (_, t) => ((zero[t] || []).length ? zero[t].length * ZW + 6 : 2)).reduce((a, b) => a + b, 0);
  // the cycles share what the zones leave of the plot width, so a whole episode fits the page without a scroll
  const plotW = (opt.fig ? 960 : 840) - (opt.fig ? 190 : 160) - 16 - 2 * PAD;
  const CW = Math.max(24, Math.min(72, (plotW - zonesW) / T));
  const zoneStart = [], cycStart = [];
  let pos = PAD;
  for (let t = 0; t <= T; t++) {
    zoneStart[t] = pos;
    const z = (zero[t] || []).length;
    pos += z ? z * ZW + 6 : 2;
    cycStart[t] = pos;
    if (t < T) pos += CW;
  }
  const W = pos + PAD;
  const xCall = (c) => (c.cycles === 0 ? zoneStart[c.time_before] + 3 + zero[c.time_before].indexOf(c.n) * ZW + ZW / 2 : (cycStart[c.time_before] + zoneStart[c.time_after]) / 2);
  return { step: false, W, T, zoneStart, cycStart, CW, ZW, zero, xCall,
    span: (c) => (c.cycles === 0 ? [xCall(c) - ZW / 2 + 1, xCall(c) + ZW / 2 - 1] : [cycStart[c.time_before], zoneStart[c.time_after]]),
    // an event at cycle t (a repair starts, a failure): the start of that cycle
    at: (t) => cycStart[Math.min(Math.max(0, t), T)],
    // the end of the call: where spending lands
    end: (c) => (c.cycles === 0 ? xCall(c) : zoneStart[c.time_after]) };
}
/* ---------- the summary of an episode in plain words ----------
   One sentence and one row of labels for any episode, from what the page holds: the last ranking before the diagnosis,
   the diagnosis or the declaration, the work orders, the last check of a repair and the end event. Each clause is said
   only when the record supports it. A check whose status is "verified" only ran; "consistent" says whether the readings
   matched the state the CausalDiagnosis MCP predicted after the repair, and that state can still hold a fault. */
const WORDS10 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
// numbers up to ten as words inside a sentence, as written English has them
const count = (n, w) => `${n >= 0 && n <= 10 ? WORDS10[n] : n} ${w}${n === 1 ? "" : "s"}`;
// the parts a possible cause names: "the fan", "the fan and the belt"; a level only where the part has more than one
// faulty level, since only then does the name alone leave the cause open
function causeWords(comb) {
  const r = rules(), bad = faultySet(comb);
  if (!bad.length) return "no faulty part";
  return andList(bad.map((k) => ((r.levels[k] || []).filter((v) => v !== r.healthy[k]).length > 1 ? `the ${k} at level ${comb[k]}` : `the ${k}`)));
}
// what the CausalDiagnosis MCP predicted the machine would be after the repair
const predWords = (a) => (a.predicted_combination ? (faultySet(a.predicted_combination).length ? `, that ${causeWords(a.predicted_combination)} would still be faulty` : ", a machine with no faulty part") : "");
// per answer of verify_outcome, the agent's request to compare the readings after a repair with the prediction: the
// test, the tail of the sentence, and the label; a null tail is a comparison asked while the order was open, whose
// sentence says how many cycles early it came. The word is "compare" on every screen, so it is never read as the verifier
const CHECK_WORDS = [
  [(a) => a.status === "verified" && a.consistent === true, (a) => `, and the readings matched what it had predicted${predWords(a)}.`, "readings matched the prediction"],
  [(a) => a.status === "verified" && a.consistent === false, (a) => `, and the readings did not match what it had predicted${predWords(a)}.`, "readings did not match the prediction"],
  [(a) => a.status === "verified", () => ", but the comparison gave no answer.", "comparison gave no answer"],
  [(a) => /^not yet observable: the work order is open/.test(a.status), null, "compared too early, the repair was not done"],
  [(a) => /^not yet observable: no reading/.test(a.status), () => ", but no new reading came after the repair, so there was nothing to compare.", "compared before a new reading"],
  [(a) => /^not yet observable/.test(a.status), () => ", but no repair had been done yet, so there was nothing to compare.", "compared before any repair"],
  [(a) => /^not comparable/.test(a.status), () => ", but another repair finished in between, so the comparison could not tell which one worked.", "another repair got in the way"],
  [(a) => /^action refused/.test(a.status), () => ", but the repair had been refused, so there was nothing to compare.", "the repair was refused"],
];
function checkRow(a) { return CHECK_WORDS.find(([t]) => t(a || {})) || null; }
// the tail of a sentence about one comparison; doneAt is the cycle the repair was done, when known
function checkTail(c, doneAt) {
  const a = c.answer || {}, row = checkRow(a);
  if (!row) return a.status ? `, and the CausalDiagnosis MCP answered ${a.status}.` : ".";
  if (row[1]) return row[1](a);
  const gap = doneAt != null ? doneAt - c.time_before : null;
  return gap != null && gap > 0 ? `, ${count(gap, "cycle")} before the repair was done, so there was nothing to compare yet.`
    : ", before the repair was done, so there was nothing to compare yet.";
}
// how a cut at the wall limit ended (wall_cut.cut_end, classed by the facts first), in words
const CUT_WORDS = { wall: "the agent's time ran out", service: "the model service or a tool did not answer" };
const secs = (x) => `${Math.round((+x || 0) * 10) / 10} s`;
// the clock modes that charge the agent's thinking before each step (THINKING_MODES of layers/layer_2_real_time.py)
const THINKING_MODES = ["generation_time", "generated_tokens"];
// the thinking the clock charged in all (episode.json), and what it left out, in the terms of each clock. The wall clock leaves
// out the credited starts of the servers and service of the counted calls, and the failed model requests. The token clock
// charges the tokens the model generated before each counted call (clock/token_clock.py) and no wall-clock seconds,
// so the time of generation, the starts, the service and the failed requests take no part, and no call charges the tokens
// generated after the last counted call
function thinkChip(e) {
  const mode = rules().clock_mode;
  if (!THINKING_MODES.includes(mode)) return null;
  const k = e.clock || {}, credited = k.credited_seconds || {};
  const said = mode === "generated_tokens"
    ? `The clock counted ${secs(k.charged_seconds)} of thinking for the ${plural(k.charged_tokens || 0, "token")} the model generated up to the last step. It counts tokens, not wall-clock time, so the time the model took to generate them, server starts, tool service and failed model requests did not count, and neither did tokens generated after the last step.`
    : `The clock counted ${secs(k.charged_seconds)} of thinking. ${secs((credited.starts || 0) + (credited.calls || 0))} of server starts and tool service did not count, and ${secs(k.failed_model_seconds)} of failed model requests did not count.`;
  return h("details", { class: "fact f-think" }, h("summary", {}, `thinking ${secs(k.charged_seconds)}`),
    h("span", { class: "think-open" }, said));
}
function thinkRow(c) {
  const mode = rules().clock_mode;
  if (!THINKING_MODES.includes(mode) || c.think_seconds == null) return null;
  const tokens = mode === "generated_tokens" ? ` for ${plural(c.think_tokens || 0, "token")}` : "";
  return [h("dt", {}, "thinking before this step"), h("dd", {}, `${secs(c.think_seconds)}${tokens}, ${plural(c.think_cycles || 0, "cycle")}`)];
}
const END_WORDS = {
  horizon: (r) => [`The episode reached its time limit of ${plural(r.horizon, "cycle")} before the agent finished.`, "stopped at the time limit"],
  max_steps: (r) => [`The episode reached its limit of ${plural(r.max_steps, "step")} before the agent finished.`, "stopped at the step limit"],
  budget: () => ["The episode stopped when the budget ran out.", "ran out of budget"],
  failure: () => ["The episode stopped when a part of the machine failed.", "stopped by a failure"],
};
// one plain line on what moves the clock in this run, from the run's own rules: the clock mode and the declared
// durations (has_clock of layers/layer_2_real_time.py reads the same durations)
function clockLine() {
  const r = rules(), d = r.durations || {}, mode = r.clock_mode || "declared";
  const repairs = [...Object.values(d.lead_time || {}), ...Object.values(d.repair_downtime || {})].some((v) => v > 0);
  const movers = [...(d.read_per_cycle > 0 ? ["reads"] : []), "waits", ...(repairs ? ["repairs"] : [])];
  if (mode === "instant") return "In this run every step takes one cycle.";
  if (mode === "generation_time" || mode === "generated_tokens") {
    const how = mode === "generation_time" ? (r.think_seconds_per_cycle ? `, one cycle for each ${num(r.think_seconds_per_cycle)} seconds of it` : "")
      : ", counted from the length of what its model writes";
    return `In this run the agent's thinking time moves the clock too${how}, and so do ${andList(movers)}.`;
  }
  return movers.length > 1 ? `In this run thinking takes no time. Only ${andList(movers)} move the clock.` : "In this run thinking takes no time, and only waiting moves the clock.";
}
// who played the run, from the run's own record: a model agent names its design and model; a player without a model is
// the scripted example or a fixed policy, so the picture itself says that it shows no model's result
function playerWords(e) {
  const run = runOf(e.run) || {}, who = String(run.agent_label || run.agent || "agent").replace(/_/g, " ");
  if (run.model) return `${who} agent, model ${run.model}`;
  return run.agent === "scripted" ? "scripted example, no model" : `${who} policy, no model`;
}
function plainSummary(e) {
  const r = rules(), parts = [], chips = [["f-player", playerWords(e)]];
  const R = rankingCalls(), lives = orderLives(e);
  const decide = e.calls.find((c) => c.kind === "decide" && !c.refused && (c.tool === "submit_diagnosis" || c.tool === "declare_insufficient_evidence"));
  const pre = decide ? R.filter((c) => c.n < decide.n) : R;
  const rk = pre.length ? pre[pre.length - 1] : R.length ? R[R.length - 1] : null;
  if (rk) {
    const n0 = nCauses(R[0].answer), k = Math.round(rk.answer.effective_hypotheses), top = rk.answer.candidates[0].combination;
    parts.push(k === 1 ? `The CausalDiagnosis MCP narrowed ${n0} possible causes down to one, ${causeWords(top)}.`
      : k < n0 ? `The CausalDiagnosis MCP narrowed ${n0} possible causes down to ${k}, with ${causeWords(top)} most likely.`
      : `The CausalDiagnosis MCP could not yet rule out any of the ${n0} possible causes.`);
    chips.push(["f-elim", `${n0} possible causes → ${k}`]);
  }
  const named = decide && decide.tool === "submit_diagnosis";
  let s2 = named ? `The agent diagnosed ${causeWords(decide.args)}${rk && decide.follow === "deviated" ? " instead" : ""}`
    : decide ? "The agent declared that the evidence was not enough to name a cause" : "The agent named no cause";
  const fixed = [...new Set(lives.map((l) => l.component))];
  if (fixed.length) s2 += `${named ? " and" : ", and"} ordered a repair of ${andList(fixed.map((k) => `the ${k}`))}`;
  parts.push(`${s2}.`);
  if (lives.length) chips.push(["f-orders", plural(lives.length, "work order")]);
  const idleN = (e.idle_steps || []).length;
  if (idleN) chips.push(["f-idle", `${idleN} ${idleN === 1 ? "turn" : "turns"} without a tool`]);
  chips.push(["f-time", plural(e.endTime, "cycle")]);
  const left = e.status ? e.status.budget_left : null;
  if (r.budget != null && left != null) chips.push(["f-cost", `spent ${r.budget - left} of ${r.budget}`]);
  const closes = lives.map((l) => l.close).filter((x) => x != null), done = closes.length && closes.length === lives.length ? Math.max(...closes) : null;
  const chk = e.calls.filter((c) => c.tool === "verify_outcome" && c.answer && c.answer.status).pop();
  if (chk) {
    const row = checkRow(chk.answer);
    parts.push(`At step ${chk.n} it asked the CausalDiagnosis MCP to compare the readings with its prediction${checkTail(chk, done)}`);
    chips.push([chk.answer.consistent === true ? "f-ok" : "f-bad", row ? row[2] : chk.answer.status]);
  } else if (lives.length) {
    // the answers of a row without notifications, as AssetOpsBench's base row, never say when an order closed
    const told = e.calls.some((c) => Array.isArray(c.notifications));
    parts.push(done != null ? `The repair was done in cycle ${done}, and the agent did not ask for the readings to be compared with a prediction.`
      : told ? "The repair was not done by the end of the episode." : "The answers of this run do not say when a work order closed.");
  }
  const end = (e.events || []).filter((x) => x.event === "end").pop(), ew = end && END_WORDS[end.reason];
  if (ew) { const [s1, c1] = ew(r); parts.push(s1); chips.push(["f-bad", c1]); }
  const sv = e.served || {};
  if (sv.wall_limit) parts.push(`The episode was cut at the time limit at cycle ${e.endTime} because ${CUT_WORDS[sv.cut_end] || "the harness recorded no reason"}.`);
  return { sentence: parts.join(" "), chips, think: thinkChip(e) };
}
function tlHeadline(e) {
  const p = plainSummary(e);
  return h("div", { class: "rs-head2" }, h("p", { class: "rs-story" }, p.sentence), h("div", { class: "rs-facts" }, p.chips.map(([c, t]) => h("span", { class: `fact ${c}` }, t)), p.think),
    h("p", { class: "rs-clock" }, clockLine()));
}
function shortIds(ids) {
  const xs = String(ids || "").split(",").map((x) => x.trim()).filter(Boolean);
  const byObs = {};
  for (const x of xs) { const m = x.match(/^(obs\/\d+)\//); const k = m ? m[1] : x; byObs[k] = (byObs[k] || 0) + 1; }
  return Object.entries(byObs).map(([k, n]) => (n > 1 ? `${k} (${n} channel values)` : k)).join(", ") || "none";
}
function callSentence(c) {
  const a = c.args || {};
  if (c.tool === "read_sensors") return `The agent read ${groupWords(a.sensor_group)} for ${count(a.cycles, "cycle")}${c.cycles ? `, cycle ${c.time_before} to ${c.time_after}` : ""}.`;
  if (c.kind === "time") return c.cycles ? `The agent waited ${count(c.cycles, "cycle")}, cycle ${c.time_before} to ${c.time_after}.` : "The agent asked for the time.";
  if (c.tool === "submit_diagnosis") return `The agent diagnosed ${causeWords(a)}.`;
  if (c.tool === "create_work_order") return `The agent ordered a repair of the ${a.component}.`;
  if (c.tool === "finish") return "The agent finished the episode.";
  if (c.tool === "verify_outcome") { const done = orderLives(ep()).map((l) => l.close).filter((x) => x != null); return `The agent asked the CausalDiagnosis MCP to compare the readings with its prediction${checkTail(c, done.length ? Math.max(...done) : null)}`; }
  if (c.kind === "advice") return `The agent asked the CausalDiagnosis MCP (${c.tool}), which takes no time and costs nothing.`;
  return `The agent called ${c.tool}${c.cycles ? `, cycle ${c.time_before} to ${c.time_after}` : ", which takes no time"}.`;
}
function callRows(c) {
  const rows = [["step", String(c.n)], ["cycle", c.cycles ? `${c.time_before} → ${c.time_after}` : String(c.time_before)]];
  if (c.price != null) rows.push(["price", String(c.price)], ["budget left after it", String(c.budget_after)]);
  const a = c.args || {};
  if (a.evidence_ids) rows.push(["evidence cited", shortIds(a.evidence_ids)]);
  if (a.prediction_id) {
    const p = ep().predictions.find((q) => q.id === a.prediction_id);
    rows.push(["prediction cited", p ? `${p.id}: chance it recovers ${pct(p.recovery)}${p.recovery_interval ? `, interval ${num(p.recovery_interval[0], 2)} to ${num(p.recovery_interval[1], 2)}` : ""}` : a.prediction_id]);
  }
  return rows;
}
function callRefs(c) { return [rCall(c.n, ["tool_args"]), rAnswer(c.n), c.args && c.args.prediction_id ? rPred(c.args.prediction_id) : null]; }
/* the plans of a ranking: one row per candidate plan with its two predictions, and the read they share, once */
function openRanking(panel, c) {
  const a = c.answer, at = rankAt(c);
  openMark(panel, `The CausalDiagnosis MCP ranked the possible causes on the readings up to cycle ${at}, ${Math.round(a.effective_hypotheses)} of ${nCauses(a)} still left.`,
    [["step", String(c.n)], ["possible causes still left", `${num(a.effective_hypotheses, 1)} of ${nCauses(a)}`], ["most likely", causeWords(a.candidates[0].combination)]],
    [rAnswer(c.n, ["candidates"]), rAnswer(c.n, ["branches"])], "The CausalDiagnosis MCP takes no cycle and charges nothing.");
  const br = a.branches || [];
  if (!br.length) return;
  const P = plansDoc();
  const next = ep().calls.find((x) => x.n > c.n && x.tool === "read_sensors");
  const sched = P && next ? P.plans.map((pl) => pl.schedule).filter((sc) => sc && sc.reads && sc.reads.length)
    .flatMap((sc) => sc.reads).find((r) => r.sensor_group === next.args.sensor_group && r.cycles === next.args.cycles) : null;
  const preds = ep().predictions;
  const tbl = h("table", { class: "grid tl-plans" },
    h("thead", {}, h("tr", {}, ["candidate", "repair", "probability", "chance it recovers (prediction)", "cost to restored (prediction)"].map((t) => h("th", {}, t)))),
    h("tbody", {}, br.map((b) => {
      const rp = b.repair_plan || {}, p = rp.plan_prediction_id ? preds.find((q) => q.id === rp.plan_prediction_id) : null;
      const ctr = b.cost_to_restored, cost = ctr == null ? NR : typeof ctr === "number" ? num(ctr, 1)
        : Object.entries(ctr).map(([k, v]) => `${k.replace(/_/g, " ")} ${num(v, 0)}`).join(", ");
      return h("tr", {}, h("td", {}, candWords(b.combination)), h("td", {}, andList((b.faulty_set || []).map((k) => `the ${k}`)) || "nothing"), h("td", {}, pct(b.probability)),
        h("td", {}, p ? pct(p.recovery) : NR), h("td", {}, cost));
    })));
  const shared = sched ? h("p", { class: "mk-note" }, `One read serves the tests of ${plural((sched.serves || []).length, "plan")} at once: ${groupWords(sched.sensor_group)} for ${plural(sched.cycles, "cycle")}, drawn once in the calls panel (step ${next.n}).`) : null;
  panel.insertBefore(h("div", { class: "tl-plans-wrap" }, shared, tbl), panel.querySelector(".mk-rec"));
}
function timelineView(opt) {
  opt = opt || {};
  const fig = !!opt.fig, people = !fig && !!(S.showHidden && S.hidden && S.hidden.log);
  const e = ep(), A = tlAxis(e, opt), M = rankingCalls().length ? storyModel() : null;
  const FS = fig ? { lab: 15, tick: 15, num: 15 } : { lab: 12.5, tick: 11.5, num: 10.5 };
  const LW = fig ? 190 : 160, PW = A.W, W = LW + PW + 16;
  const panel = h("div", { class: "mk-panel", hidden: "", role: "dialog", "aria-label": "Details" });
  const svg = s("svg", { class: ["tl", fig ? "tl-fig" : null], viewBox: `0 0 ${W} 10`, role: "img", "aria-label": "The episode over the machine's time: possible causes, steps, the machine and spending" });
  const X = (x) => LW + x;
  let y = 6;
  const G = s("g");
  const band = (h0, title, sub) => {
    const g = s("g", { class: "tl-band" });
    g.append(s("text", { x: 6, y: y + FS.lab + 2, class: "tl-ptitle" }, title));
    if (sub) wrapWords(sub, fig ? 22 : 24).forEach((ln, k) => g.append(s("text", { x: 6, y: y + FS.lab * 2 + 6 + k * (FS.tick + 3), class: "tl-psub" }, ln)));
    g.append(s("line", { x1: LW, x2: W - 8, y1: y + h0, y2: y + h0, class: "tl-sep" }));
    G.append(g);
    const top = y; y += h0 + 10; return top;
  };
  // the axis zones that hold calls taking no cycle, drawn first, faint, under every panel
  const zones = s("g", { class: "tl-zones" });
  G.append(zones);
  /* a. candidates still plausible */
  // the plot keeps 96 px; a strip below the line holds the name of the chosen plan, so no step of the line can cross it
  const HaPlot = 96, Ha = HaPlot + 28, ya = band(Ha, "Possible causes", "still left, at each ranking");
  if (M && M.pre.length || (M && M.R.length)) {
    const R = rankingCalls(), n0 = nCauses(R[0].answer);
    const yv = (v) => ya + 10 + (HaPlot - 22) * (1 - Math.log(Math.max(1, v)) / Math.log(Math.max(2, n0)));
    const xr = (c) => (A.step ? A.xStep(c.n) : A.xCall(c));
    // a step line: hold the count until the next ranking, then drop
    let d = ""; R.forEach((c, i) => { const v = c.answer.effective_hypotheses; d += i ? ` H${X(xr(c))} V${yv(v)}` : `M${X(xr(c))} ${yv(v)}`; });
    const endX = X(A.step ? A.xStep(e.calls.length) : A.zoneStart[A.T]);
    d += ` H${endX}`;
    G.append(s("path", { d, class: "tl-cand" }));
    for (const v of [n0, 1]) G.append(s("text", { x: LW - 6, y: yv(v) + 4, "text-anchor": "end", class: "tl-ytick" }, String(v)));
    R.forEach((c) => {
      const v = c.answer.effective_hypotheses;
      G.append(markWith(s("circle", { cx: X(xr(c)), cy: yv(v), r: fig ? 6 : 5.5, class: "tl-rank" }), `The CausalDiagnosis MCP ranked the possible causes, ${Math.round(v)} of ${nCauses(c.answer)} still left.`, () => openRanking(panel, c)));
    });
    const chosen = M.chosenB ? `the agent acted on ${causeWords(M.chosenB.combination)}` : null;
    const lastV = Math.round(R[R.length - 1].answer.effective_hypotheses), firstLast = R.find((c) => Math.round(c.answer.effective_hypotheses) === lastV);
    if (chosen && firstLast) G.append(tlChip(X(PW) - 8, ya + HaPlot + 16, "end", chosen, "tl-chosen", fig ? 15 : 12.5));
    if (M.check) {
      const ok = M.check.answer.consistent === true, cx = X(A.step ? A.xStep(M.check.n) : A.xCall(M.check)), cy = yv(1);
      const g = mark(s("g", { class: `tl-check ${ok ? "ok" : "bad"}` }, s("circle", { cx, cy, r: fig ? 10 : 9 }),
        s("path", { d: ok ? `M${cx - 4},${cy} L${cx - 1},${cy + 3} L${cx + 4},${cy - 3}` : `M${cx - 3},${cy - 3} L${cx + 3},${cy + 3} M${cx - 3},${cy + 3} L${cx + 3},${cy - 3}` })),
        panel, callSentence(M.check), callRows(M.check), callRefs(M.check));
      G.append(g, s("text", { x: cx, y: cy - (fig ? 16 : 14), "text-anchor": "middle", class: `tl-checklab ${ok ? "ok" : "bad"}` }, ok ? "matched" : M.check.answer.consistent === false ? "did not match" : /^not yet observable/.test(M.check.answer.status || "") ? "too early" : "no answer"));
    }
  } else G.append(s("text", { x: LW + 6, y: ya + 40, class: "tl-psub" }, "The agent did not ask the CausalDiagnosis MCP for a ranking in this episode, so nothing is drawn here."));
  /* b. calls over the cycles they take */
  const LH = fig ? 26 : 22, lives = orderLives(e), nOrd = Math.max(1, lives.length);
  const laneH = { reads: LH, advice: LH, time: LH, orders: Math.max(LH, nOrd * (LH - 6) + 6), decide: LH };
  const Hb = TL_LANES.reduce((acc, [k]) => acc + laneH[k], 0) + 8, yb = band(Hb, "Steps", null);
  const laneY = {}; let yy = yb + 4;
  for (const [k, t] of TL_LANES) { laneY[k] = yy; G.append(s("text", { x: LW - 6, y: yy + Math.min(laneH[k], LH) / 2 + 4, "text-anchor": "end", class: `tl-lane k-${k}` }, t)); yy += laneH[k]; }
  const think = thinkCycles();
  for (const c of e.calls) {
    const k = c.kind === "know" ? "reads" : c.kind, y0 = laneY[k] != null ? laneY[k] : laneY.reads;
    if (c.tool === "create_work_order") continue;
    const [x0, x1] = A.span(c);
    const zero = !A.step && c.cycles === 0;
    const el = zero || (A.step && !c.cycles)
      ? s("rect", { x: X(xCallSafe(A, c)) - (k === "advice" ? 1.5 : 3), y: y0 + 2, width: k === "advice" ? 3 : 6, height: LH - 12, rx: 1.5, class: `tl-tick k-${k}` })
      : s("rect", { x: X(x0), y: y0 + 3, width: Math.max(4, x1 - x0), height: LH - 6, rx: 4, class: `tl-block k-${k}` });
    if (opt.nav) {
      if (c.n === S.step) el.classList.add("sel");
      G.append(markWith(el, `Step ${c.n}: ${callSentence(c)} Click to select this step.`, () => pickStep(c.n)));
      if (c.follow === "followed" || c.follow === "deviated") G.append(s("circle", { cx: X(xCallSafe(A, c)) + 4, cy: y0 + 2, r: 2.6, class: c.follow === "followed" ? "tl-fol" : "tl-dev" }));
    } else if (c.kind === "advice" && Array.isArray((c.answer || {}).candidates)) G.append(markWith(el, callSentence(c), () => openRanking(panel, c)));
    else G.append(mark(el, panel, callSentence(c), callRows(c), callRefs(c)));
    const tc = c.env_step ? think[c.env_step - 1] || 0 : 0;
    if (tc > 0 && !A.step) G.append(s("rect", { x: X(A.cycStart[Math.max(0, c.time_before - tc)]), y: y0 + 1, width: X(A.zoneStart[c.time_before]) - X(A.cycStart[Math.max(0, c.time_before - tc)]), height: 3, class: "tl-think" }, s("title", {}, `thinking, ${plural(tc, "cycle")} before step ${c.n}`)));
    // the step number under a tick, inside a block
    if (!fig) G.append(s("text", { x: X(xCallSafe(A, c)), y: zero || (A.step && !c.cycles) ? y0 + LH - 1 : y0 + LH / 2 + 3.5, "text-anchor": "middle", class: "tl-stepnum" }, String(c.n)));
  }
  // work orders: created, waiting for the part, in repair (the asset down), closed
  lives.forEach((l, i) => {
    const y0 = laneY.orders + 3 + i * (LH - 6), hh = LH - 10, c = call(l.call);
    const xc = X(xCallSafe(A, c));
    const xe = l.exec != null ? X(A.step ? A.xStep(l.execCall) : A.at(l.exec)) : null, xz = l.close != null ? X(A.step ? A.xStep(l.closeCall) : A.zoneStart[l.close]) : null;
    if (xe != null) G.append(s("rect", { x: xc, y: y0 + hh / 2 - 1.5, width: Math.max(0, xe - xc), height: 3, class: "tl-wait-part" }));
    if (xe != null && xz != null) G.append(mark(s("rect", { x: xe, y: y0, width: Math.max(4, xz - xe), height: hh, rx: 3, class: "tl-repair" }), panel,
      `The ${l.component} was in repair from cycle ${l.exec} to cycle ${l.close}, and the machine was stopped.`, [["work order", l.id], ["repair starts", `cycle ${l.exec}`], ["closed", `cycle ${l.close}`]],
      [rAnswer(l.execCall, ["notifications"]), rAnswer(l.closeCall, ["notifications"])]));
    const oTick = s("rect", { x: xc - 4, y: y0 - 1, width: 8, height: hh + 2, rx: 2, class: ["tl-tick", "k-orders", opt.nav && c.n === S.step ? "sel" : null] });
    G.append(opt.nav ? markWith(oTick, `Step ${c.n}: ${callSentence(c)} Click to select this step.`, () => pickStep(c.n)) : mark(oTick, panel, callSentence(c), callRows(c), callRefs(c)));
    if (!fig) G.append(s("text", { x: xc, y: y0 + hh + 9, "text-anchor": "middle", class: "tl-stepnum" }, String(c.n)));
    const lw = String(l.component).length * (fig ? 9 : 6.6) + 8, right = X(A.W);
    const lab = (xz != null ? xz : xc) + lw <= right ? { x: (xz != null ? xz : xc) + 6, a: "start", c: "tl-olab" }
      : xe != null && xz != null && xz - xe >= lw ? { x: xz - 4, a: "end", c: "tl-olab tl-olab-in" }
      : { x: xc - 6, a: "end", c: "tl-olab" };
    G.append(s("text", { x: lab.x, y: y0 + hh - 2, "text-anchor": lab.a, class: lab.c }, l.component));
  });
  /* c. the asset */
  const Hc = people ? 46 : 34, yc = band(Hc, "Machine", people ? "running or stopped, and its real state" : "running or stopped");
  const xEnd = X(A.step ? PW - 8 : A.zoneStart[A.T]), x0a = X(A.step ? 8 : A.cycStart[0]);
  G.append(s("rect", { x: x0a, y: yc + 6, width: xEnd - x0a, height: 14, class: "tl-run" }));
  for (const l of lives) if (l.exec != null && l.close != null) {
    const a = X(A.step ? A.xStep(l.execCall) : A.at(l.exec)), b = X(A.step ? A.xStep(l.closeCall) : A.zoneStart[l.close]);
    G.append(s("rect", { x: a, y: yc + 6, width: Math.max(3, b - a), height: 14, class: "tl-down" }));
  }
  G.append(s("text", { x: x0a + 4, y: yc + 17, class: "tl-runlab" }, "running"));
  if (people && !A.step) {
    const H = S.hidden, played = (H.end || {}).played || [];
    const gH = s("g", { class: HID });
    for (const p of played) {
      if (p.t >= A.T) continue;
      const faulty = faultySet(p.combination.length ? Object.fromEntries(comps().map((k, i) => [k, p.combination[i]])) : p.combination);
      gH.append(s("rect", { x: X(A.cycStart[p.t]), y: yc + 24, width: X(A.zoneStart[p.t + 1]) - X(A.cycStart[p.t]), height: 8, class: faulty.length ? "tl-true-bad" : "tl-true-ok" }, s("title", {}, `cycle ${p.t}, ${faulty.length ? `${andList(faulty)} faulty` : "no faulty part"}${p.cycle == null ? ", stopped" : ""}`)));
    }
    (H.log || []).forEach((x, i) => {
      if (x.event !== "failure") return;
      const ft = x.t_start, cx = X(A.at(ft));
      gH.append(mark(s("path", { d: `M${cx},${yc + 2} l5,9 l-10,0 z`, class: "tl-fail" }), panel, `The ${x.component} failed at cycle ${ft}. The agent is told only that a part failed.`, [["penalty", String(x.cost)]], [{ f: "ep/hidden/log.json", p: [i] }]));
    });
    G.append(gH);
  }
  /* d. spending */
  const B = rules().budget;
  const Hd = fig ? 120 : 110, yd = band(Hd, B == null ? "Steps taken" : "Spending",
    B == null ? (e.cost_unit === "calls" ? "this run counts steps, not money" : "this run has prices but no budget")
      : A.step ? "against the budget" : "by kind, against the budget");
  if (!A.step && B != null) {
    const events = e.calls.filter((c) => c.price).map((c) => ({ x: A.end(c), kind: c.kind === "know" ? "reads" : c.kind, v: c.price }));
    const hiddenEv = [];
    if (people) {
      const H = S.hidden;
      for (const x of (H.end || {}).downtime_by_cycle || []) hiddenEv.push({ x0: A.cycStart[x.t], x1: A.zoneStart[Math.min(A.T, x.t + 1)], kind: "loss", v: x.charge });
      (H.log || []).forEach((x) => { if (x.event === "failure" && x.cost) hiddenEv.push({ x0: A.at(x.t_start), x1: A.at(x.t_start), kind: "penalty", v: x.cost }); if (/misdiag/.test(x.event || "") && x.cost) hiddenEv.push({ x0: A.at(x.t_start || 0), x1: A.at(x.t_start || 0), kind: "misdiagnosis", v: x.cost }); });
    }
    const totalPub = events.reduce((a, x) => a + x.v, 0), totalHid = hiddenEv.reduce((a, x) => a + x.v, 0);
    const vmax = Math.max(B, totalPub + totalHid) * 1.04;
    const yv = (v) => yd + Hd - 6 - (Hd - 16) * v / vmax;
    const x0 = X(A.cycStart[0] - 4), x1 = xEnd;
    const kinds = ["reads", "orders", "time", "advice", "decide"].filter((k) => events.some((x) => x.kind === k));
    // breakpoints: every x where any series changes
    const xs = [...new Set([A.cycStart[0] - 4, ...events.map((x) => x.x), ...hiddenEv.flatMap((x) => [x.x0, x.x1]), A.zoneStart[A.T]])].sort((a, b) => a - b);
    const cumAt = (pred, x) => events.filter((ev) => pred(ev) && ev.x <= x + 1e-6).reduce((a, ev) => a + ev.v, 0);
    const hidAt = (kind, x) => hiddenEv.filter((ev) => ev.kind === kind).reduce((a, ev) => a + (x >= ev.x1 ? ev.v : x <= ev.x0 ? 0 : ev.v * (x - ev.x0) / (ev.x1 - ev.x0)), 0);
    const layers = [...kinds.map((k) => ({ k, f: (x) => cumAt((ev) => ev.kind === k, x), cls: `tl-area k-${k}`, hid: false })),
      ...(people ? ["loss", "penalty", "misdiagnosis"].filter((k) => hiddenEv.some((ev) => ev.kind === k)).map((k) => ({ k, f: (x) => hidAt(k, x), cls: `tl-area tl-hid-${k}`, hid: true })) : [])];
    let base = xs.map(() => 0);
    const gHid = s("g", { class: HID });
    for (const L of layers) {
      const top = xs.map((x, i) => base[i] + L.f(x));
      // public spending is a step at the end of the call; production loss accrues across a cycle, so it is a ramp
      const n = xs.length - 1;
      let d = `M${X(xs[0])} ${yv(top[0])} `;
      for (let i = 1; i <= n; i++) d += L.hid ? `L${X(xs[i])} ${yv(top[i])} ` : `H${X(xs[i])} V${yv(top[i])} `;
      d += `L${X(xs[n])} ${yv(base[n])} `;
      for (let i = n; i >= 1; i--) d += L.hid ? `L${X(xs[i - 1])} ${yv(base[i - 1])} ` : `V${yv(base[i - 1])} H${X(xs[i - 1])} `;
      const p = s("path", { d: d + "Z", class: L.cls });
      (L.hid ? gHid : G).append(p);
      base = top;
    }
    if (people) G.append(gHid);
    G.append(s("line", { x1: x0, x2: x1, y1: yv(B), y2: yv(B), class: "tl-budget" }), s("text", { x: x0 + 4, y: yv(B) - 5, class: "tl-budlab" }, `budget ${B}`));
    for (const v of [0, B]) G.append(s("text", { x: LW - 6, y: yv(v) + 4, "text-anchor": "end", class: "tl-ytick" }, String(v)));
    const left = e.status ? e.status.budget_left : B - totalPub;
    // with hidden losses stacked on top, the public total goes inside its own area, under the hatch
    const fsEnd = fig ? 15 : 12;
    G.append(tlChip(x1 - 6, people && totalHid ? yv(totalPub) + fsEnd + 6 : yv(totalPub) - 6, "end", `spent ${totalPub}, ${left} left`, "tl-endlab", fsEnd));
    if (people && totalHid) { const c = tlChip(x1 - 6, yv(totalPub + totalHid) - 6, "end", `with output lost ${totalPub + totalHid}`, "tl-endlab tl-hidlab", fsEnd); c.classList.add(HID); G.append(c); }
  } else if (B != null) {
    // a priced row whose calls take no cycle: the axis counts steps, and the spending climbs at each step that cost money
    const total = e.calls.reduce((a, c) => a + (c.price || 0), 0), vmax = Math.max(B, total) * 1.04, yv = (v) => yd + Hd - 6 - (Hd - 16) * v / vmax;
    let d = `M${X(8)} ${yv(0)}`, spent = 0;
    for (const c of e.calls) if (c.price) { spent += c.price; d += ` H${X(A.xStep(c.n))} V${yv(spent)}`; }
    d += ` H${xEnd}`;
    const left = e.status ? e.status.budget_left : B - total;
    G.append(s("path", { d, class: "tl-calls" }), s("line", { x1: X(8), x2: xEnd, y1: yv(B), y2: yv(B), class: "tl-budget" }),
      s("text", { x: X(12), y: yv(B) - 5, class: "tl-budlab" }, `budget ${B}`),
      tlChip(xEnd - 6, yv(total) - 6, "end", `spent ${total}, ${left} left`, "tl-endlab", fig ? 15 : 12));
  } else {
    const ms = rules().max_steps || e.calls.length, vmax = ms * 1.04, yv = (v) => yd + Hd - 6 - (Hd - 16) * v / vmax;
    const env = e.calls.filter((c) => c.env_step);
    // a run without a budget may stand on an axis of cycles, whose calls end where spending would land
    const at = (c) => (A.step ? A.xStep(c.n) : A.end(c));
    let d = `M${X(8)} ${yv(0)}`;
    env.forEach((c, i) => { d += ` H${X(at(c))} V${yv(i + 1)}`; });
    d += ` H${xEnd}`;
    G.append(s("path", { d, class: "tl-calls" }), s("line", { x1: X(8), x2: xEnd, y1: yv(ms), y2: yv(ms), class: "tl-budget" }),
      s("text", { x: X(12), y: yv(ms) - 5, class: "tl-budlab" }, `step limit ${ms}`), s("text", { x: xEnd - 4, y: yv(env.length) - 6, "text-anchor": "end", class: "tl-endlab" }, `${env.length} steps on the machine`));
  }
  if (opt.nav) {
    const cs = call(S.step), [sx0, sx1] = A.span(cs);
    G.insertBefore(s("rect", { x: X(sx0) - 3, y: 4, width: Math.max(8, sx1 - sx0 + 6), height: y - 8, rx: 4, class: "tl-selcol" }), G.firstChild.nextSibling);
    if (!A.step && rules().budget != null) {
      const B = rules().budget, vmax = Math.max(B, e.calls.reduce((a, c) => a + (c.price || 0), 0)) * 1.04;
      const yv = (v) => yd + Hd - 6 - (Hd - 16) * v / vmax, xs = X(A.end(cs)), spentTo = B - cs.budget_after;
      G.append(s("circle", { cx: xs, cy: yv(spentTo), r: 4, class: "tl-selpt" }),
        tlChip(xs + 8, yv(spentTo) + 16, "start", `after step ${cs.n}: ${cs.budget_after} of ${B} left`, "tl-endlab", fig ? 15 : 12));
    }
  }
  /* the axis */
  const yAx = y + 2;
  const gAx = s("g", { class: "tl-axis" });
  if (A.step) {
    for (const c of e.calls) gAx.append(s("text", { x: X(A.xStep(c.n)), y: yAx + FS.tick, "text-anchor": "middle", class: "tl-axnum" }, String(c.n)));
    gAx.append(s("text", { x: LW - 6, y: yAx + FS.tick, "text-anchor": "end", class: "tl-axlab" }, "step"));
    // the cycle the clock had reached, under each step: it moves only during waits
    let last = null;
    for (const c of e.calls) if (c.time_after !== last) { gAx.append(s("text", { x: X(A.xStep(c.n)), y: yAx + FS.tick * 2 + 6, "text-anchor": "middle", class: "tl-axcyc" }, String(c.time_after))); last = c.time_after; }
    gAx.append(s("text", { x: LW - 6, y: yAx + FS.tick * 2 + 6, "text-anchor": "end", class: "tl-axlab" }, "cycle"));
  } else {
    for (let t = 0; t < A.T; t++) gAx.append(s("text", { x: X((A.cycStart[t] + A.zoneStart[t + 1]) / 2), y: yAx + FS.tick, "text-anchor": "middle", class: "tl-axnum" }, String(t)));
    for (let t = 0; t <= A.T; t++) {
      gAx.append(s("line", { x1: X(A.cycStart[t]), x2: X(A.cycStart[t]), y1: 6, y2: yAx - 2, class: "tl-grid" }));
      if ((A.zero[t] || []).length) zones.append(s("rect", { x: X(A.zoneStart[t]), y: 4, width: A.cycStart[t] - A.zoneStart[t], height: yAx - 6, class: "tl-zone" }));
    }
    gAx.append(s("text", { x: LW - 6, y: yAx + FS.tick, "text-anchor": "end", class: "tl-axlab" }, "cycle"));
  }
  y = yAx + (A.step ? FS.tick * 2 + 12 : FS.tick + 8);
  // the cursor: a vertical line through the four panels and one sentence of what holds at that cycle
  const cur = s("line", { x1: 0, x2: 0, y1: 4, y2: yAx - 2, class: "tl-cursor", visibility: "hidden" });
  const read = fig ? null : h("p", { class: "tl-read", "aria-live": "off" }, A.step ? "Point at the picture: the step, the calls made so far and the asset." : "Point at the picture: the cycle, what is plausible, the asset and the money spent by then.");
  const defs = s("defs", {}, s("pattern", { id: "tl-hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    s("rect", { width: 6, height: 6, class: "tl-hatch-bg" }), s("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "tl-hatch-ln" })));
  // the grid goes under the marks, so a label chip covers the grid lines it sits on
  svg.append(defs, gAx, G, cur);
  svg.setAttribute("viewBox", `0 0 ${W} ${y}`);
  svg.style.width = `${W}px`;
  if (!fig) {
    svg.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width - LW;
      if (px < 0 || px > PW) { cur.setAttribute("visibility", "hidden"); return; }
      cur.setAttribute("x1", X(px)); cur.setAttribute("x2", X(px)); cur.setAttribute("visibility", "visible");
      read.textContent = cursorSentence(e, A, px, lives, people);
    });
    svg.addEventListener("pointerleave", () => cur.setAttribute("visibility", "hidden"));
  }
  const leg = legend([["l-tl-reads", "a read, as long as the cycles it takes"], ["l-tl-mcp", "the CausalDiagnosis MCP, no time and no cost"], ["l-tl-wait", "waiting"],
    ["l-tl-order", "work order, waiting for the spare part, then the repair"], ["l-tl-down", "the machine stopped"],
    ...(opt.nav ? [["l-tl-fol", "green dot, the step followed the advice of the CausalDiagnosis MCP, red dot, it went against that advice"]] : []), ...(people ? [["l-tl-true", "the real state at each cycle, red where some part is faulty, green where none is (ground truth)"], ["l-tl-hid", "output lost that the agent never sees (ground truth)"]] : [])]);
  const wrap = h("div", { class: "bg-wrap tl-wrap" }, svg);
  if (fig) return h("div", { class: "tl-a" }, wrap, leg);
  return h("section", { class: "rs-sec tl-sec" },
    A.step ? h("p", { class: "tl-stepnote" }, e.cost_unit === "calls" ? "This row has no clock for the calls: every declared duration is zero and no call has a price, so the axis counts steps. The clock moved only while the agent waited."
      : "This row has no clock for the calls: every declared duration is zero, so the axis counts steps. The clock moved only while the agent waited.") : null,
    read, wrap, panel, leg);
}
function xCallSafe(A, c) { return A.step ? A.xStep(c.n) : A.xCall(c); }
function cursorSentence(e, A, px, lives, people) {
  if (A.step) {
    const n = Math.min(e.calls.length, Math.max(1, Math.round((px - 8) / A.SW + 0.5)));
    const c = call(n), made = e.calls.filter((x) => x.n <= n && x.env_step).length;
    return `Step ${n} (${c.tool}), cycle ${c.time_after}: ${plural(made, "call")} to the asset so far.`;
  }
  let t = 0;
  for (let k = 0; k < A.T; k++) if (px >= A.zoneStart[k]) t = k;
  const R = rankingCalls().filter((c) => A.xCall(c) <= px);
  const rk = R[R.length - 1];
  const down = lives.some((l) => l.exec != null && l.close != null && t >= l.exec && t < l.close);
  const spent = e.calls.filter((c) => c.price && A.end(c) <= px).reduce((a, c) => a + c.price, 0);
  return `Cycle ${t}: ${rk ? `${Math.round(rk.answer.effective_hypotheses)} of ${nCauses(rk.answer)} candidates plausible` : "no ranking yet"}, the asset ${down ? "down for a repair" : "running"}, ${spent} spent by then.`;
}
function markWith(el, sentence, fn) {
  el.setAttribute("tabindex", "0"); el.setAttribute("role", "button"); el.setAttribute("aria-label", sentence); el.classList.add("mk");
  el.append(s("title", {}, sentence));
  el.addEventListener("click", (ev) => { ev.stopPropagation(); fn(); });
  el.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); ev.stopPropagation(); fn(); } });
  return el;
}
/* a label on a filled chip, so grid lines and hatching never run through its letters; the width is estimated from the
   font size, since the text is not measured before it is drawn */
function tlChip(x, y, anchor, text, cls, fs) {
  const w = text.length * fs * 0.5 + 12, h0 = fs + 6, x0 = anchor === "end" ? x - w + 5 : x - 5;
  return s("g", { class: "tl-chip" }, s("rect", { x: x0, y: y - fs - 1, width: w, height: h0, rx: 4, class: "tl-chipbg" }), s("text", { x, y, "text-anchor": anchor, class: cls }, text));
}

/* a click on a step anywhere in the page top selects it for the tabs and the record below */
function pickStep(n) { S.insp = null; S.sheetOpen = true; go(epHash(S.tab, n)); }
const IDLE_TIP = "The agent's turn called no tool of the machine, it counts as one step.";
// a no_call step opens its line in the Decision tab; the pick holds only while the page stays on the hash it was made on, so
// any other selection drops it
function pickIdle(i) {
  const hash = epHash("decision", Math.max(1, ep().idle_steps[i].after));
  S.insp = null; S.sheetOpen = true; S.idlePick = { i, hash }; go(hash);
}
function idlePicked() { const p = S.idlePick; return p && p.hash === location.hash ? ep().idle_steps[p.i] : null; }

const kindOf = (c) => (c.kind === "know" ? "reads" : c.kind);
/* ---------- the top of an episode page: one axis of cycles for the whole episode ----------
   A step that takes cycles is a bar over them; a step that takes none is a tag at the cycle it happened in, and the
   tags of one cycle read top down in step order. Rows below show each work order, whether the asset runs and the
   budget left. A click on a step selects it for the tabs and the record below. */
// short: the words of a tag on a narrow overview, where a cycle is under 100 px; the step's sentence keeps every word
function stepWords(c, short) {
  const a = c.args || {};
  const nRead = ((rules().sensor_groups || {})[a.sensor_group] || []).length, readWhat = nRead ? plural(nRead, "sensor") : "sensors";
  const S0 = { get_failure_modes: ["Failure modes"], lookup_fix: ["Repair guide"], read_sensors: ["Read", readWhat],
    estimate_intervention_effect: ["Predict repair"], create_work_order: ["Order", `${a.component || ""} repair`], verify_outcome: ["Compare"],
    summarize_changes_since: ["Changes"] };
  const diag = () => { const bad = comps().filter((k) => k in a && a[k] !== rules().healthy[k]); return bad.length ? `${andList(bad)} faulty` : "no faulty part"; };
  const t = {
    get_asset_profile: ["Machine profile"], get_failure_modes: ["Failure modes", a.component ? `of the ${a.component}` : ""], get_fix_history: ["Repair history"],
    lookup_fix: ["Repair guide", a.component ? `for the ${a.component}` : ""], read_sensors: ["Read", readWhat],
    get_episode_evidence: ["Evidence list"], rank_root_causes: ["Rank the causes"],
    summarize_changes_since: ["Changes since", a.cycle != null ? `cycle ${a.cycle}` : ""], submit_diagnosis: ["Diagnosis,", diag()],
    estimate_intervention_effect: ["Predict a", `${andList(String(a.action || "").replace(/^[^:]*:/, "").split(",").filter(Boolean))} repair`], create_work_order: ["Order a", `${a.component} repair`],
    wait_for_notification: ["Wait"], get_current_time: ["Ask the time"], verify_outcome: ["Compare with the prediction"], declare_insufficient_evidence: ["Not enough evidence"], finish: ["Finish"],
  }[c.tool];
  const [w1, w2] = (short && S0[c.tool]) || t || [c.tool.replace(/_/g, " ")];
  return `${w1}${w2 ? ` ${w2}` : ""}${c.refused ? ", refused" : c.error ? ", failed" : ""}`;
}
const takesCycles = (c) => (c.cycles != null ? c.cycles : c.time_after - c.time_before) > 0;
let measureCtx = null;
function textWidth(text, px, weight) {
  measureCtx = measureCtx || document.createElement("canvas").getContext("2d");
  measureCtx.font = `${weight} ${px}px ${getComputedStyle(document.body).fontFamily}`;
  return measureCtx.measureText(text).width;
}
function clipText(text, px, weight, maxw) {
  if (textWidth(text, px, weight) <= maxw) return text;
  let t = text;
  while (t.length > 1 && textWidth(`${t}…`, px, weight) > maxw) t = t.slice(0, -1);
  return t.length > 1 ? `${t}…` : "";
}
function navStrip(M) {
  const e = ep(), host = h("div", { class: "ovb-host", id: "ovb-host" });
  const people = !!(S.showHidden && S.hidden && S.hidden.log);
  const kinds = ["reads", "advice", "time", "orders", "decide"].filter((k) => e.calls.some((c) => kindOf(c) === k));
  const KW = { reads: "read data or look up a record", advice: "asked the CausalDiagnosis MCP, the diagnosis tool", time: "waited or asked the time", orders: "ordered a repair", decide: "diagnosis or finish" };
  const leg = legend([...kinds.map((k) => [`l-ovb k-${k}`, KW[k]]), ["l-ovb-dot ok", "green dot, the step followed the advice of the CausalDiagnosis MCP"], ["l-ovb-dot bad", "red dot, the step went against that advice"],
    ["l-ovb-repair", "repair under way"], ["l-ovb-down", "the machine stopped"], ...(people ? [["l-tl-true", "the real state at each cycle, red where some part is faulty, green where none is (ground truth)"]] : []),
    ["l-ovb-cycle", "a cycle is one time step of the machine"]]);
  return h("section", { class: "panel tl-top", "aria-label": "What happened over time" },
    h("h2", { class: "ovb-title" }, "What happened over time"), tlHeadline(e), host, leg);
}
function mountOverview(host) {
  const e = ep(), people = !!(S.showHidden && S.hidden && S.hidden.log);
  const lives = orderLives(e), B = rules().budget;
  const T = Math.max(1, e.endTime, ...e.calls.map((c) => c.time_after), ...lives.map((l) => l.close || 0));
  // below 1100 px the label column and the least width of a cycle shrink, and where a cycle gets narrow the tags take
  // short words, so that an episode fits a small window without scrolling sideways (a cycle still keeps 60 px)
  const narrow = host.clientWidth < 640, compact = host.clientWidth < 1100, LABW = narrow ? 92 : compact ? 150 : 176, avail = Math.max(200, host.clientWidth - LABW - 12);
  const PL = 4, px = Math.max(narrow ? 96 : compact ? 60 : 120, (avail - PL - 8) / T), X = (t) => PL + t * px;
  const terse = compact && px < 100;  // short tag words only where the cycles are narrow
  const TF = 11.5, TH = 18, TG = 3;
  const ranks = new Set(rankingCalls().map((c) => c.n));
  const chipText = (c) => `${nCauses(c.answer)} possible causes → ${Math.round(c.answer.effective_hypotheses)}`;
  // tags: each cycle's group takes the lowest rows whose earlier tags end before it starts, so no two tags overlap
  const groups = {};
  // a no_call step has no call of its own: its tag follows the call before it, keyed by that call's number and a letter
  const idles = (e.idle_steps || []).map((x, i, all) => ({ idle: i, after: x.after, time_before: x.time,
    n: `${x.after}${"bcdefghijk"[i - all.findIndex((y) => y.after === x.after)] || "+"}` }));
  const tagLabel = (c) => (c.idle != null ? `${c.n} no tool` : `${c.n} ${stepWords(c, terse)}`);
  const seq = (c) => (c.idle != null ? c.after + 0.5 : c.n);
  for (const c of [...e.calls.filter((x) => !takesCycles(x)), ...idles]) (groups[c.time_before] = groups[c.time_before] || []).push(c);
  for (const g of Object.values(groups)) g.sort((a, b) => seq(a) - seq(b));
  const placed = [], tagAt = {};
  let rows = 0;
  for (const t of Object.keys(groups).map(Number).sort((a, b) => a - b)) {
    const g = groups[t], xs = X(t) + 2;
    const ws = g.map((c) => textWidth(tagLabel(c), TF, 600) + 14 + (c.follow === "followed" || c.follow === "deviated" ? 12 : 0));
    const ext = g.map((c, i) => ws[i] + (ranks.has(c.n) ? textWidth(chipText(c), TF, 700) + 14 : 0));
    // a group near the end of the axis moves left so that it ends at the last cycle; its stems stay at its cycle
    const x0 = Math.max(2, Math.min(xs, X(T) - Math.max(...ext)));
    let k = 0;
    while (g.some((c, i) => placed.some((p) => p.row === k + g.length - i && p.x1 + 6 > x0 && p.x0 < x0 + ext[i]))) k++;
    g.forEach((c, i) => { const row = k + g.length - i; tagAt[c.n] = { row, x0, xs, w: ws[i] }; placed.push({ row, x0, x1: x0 + ext[i] }); rows = Math.max(rows, row); });
  }
  const right = Math.max(X(T) + 8, ...placed.map((p) => p.x1 + 8));
  const W = Math.ceil(right);
  const svg = s("svg", { class: "ovb", width: W, role: "img", "aria-label": "Steps, work orders, the machine and the budget on one axis of cycles" });
  const grid = s("g", { class: "ovb-grid" }), G = s("g");
  const bands = [];
  let y = 8;
  const band = (hh, q, sub) => { bands.push({ y, h: hh, q, sub }); const top = y; y += hh + 14; return top; };
  /* a. the steps */
  const ya = band(rows * (TH + TG) + 40, narrow ? "Steps" : "What did the agent do?", "A bar is a step that took time. A tag took no time, and tags at one cycle read top down.");
  const yBar = ya + rows * (TH + TG) + 8;
  const sel = call(S.step);
  const pick = (el, c) => { if (c.n === S.step) el.classList.add("sel"); return markWith(el, `Step ${c.n}: ${callSentence(c)} Click to open this step.`, () => pickStep(c.n)); };
  for (const c of e.calls) {
    const k = kindOf(c), label = `${c.n} ${stepWords(c, terse)}`;
    if (takesCycles(c)) {
      const x0 = X(c.time_before) + 2, w = Math.max(6, X(c.time_after) - X(c.time_before) - 4);
      // a bar too short for its words takes the short words before it clips them
      const fit = textWidth(label, 12, 600) <= w - 12 ? label : `${c.n} ${stepWords(c, true)}`;
      const g = s("g", { class: `ovb-step k-${k}`, "data-n": c.n }, s("rect", { x: x0, y: yBar, width: w, height: 26, rx: 5 }),
        s("text", { x: x0 + 7, y: yBar + 17, class: "ovb-tx" }, clipText(fit, 12, 600, w - 12)));
      G.append(pick(g, c));
      continue;
    }
    const p = tagAt[c.n], ty = yBar - 6 - TH - (p.row - 1) * (TH + TG);
    G.append(s("line", { x1: p.xs, x2: p.xs, y1: ty + TH, y2: yBar - 2, class: `ovb-stem k-${k}` }));
    const g = s("g", { class: `ovb-step k-${k}`, "data-n": c.n }, s("rect", { x: p.x0, y: ty, width: p.w, height: TH, rx: 4 }),
      s("text", { x: p.x0 + 6, y: ty + 13, class: "ovb-tx ovb-tag" }, label));
    if (c.follow === "followed" || c.follow === "deviated") g.append(s("circle", { cx: p.x0 + p.w - 8, cy: ty + TH / 2, r: 4, class: c.follow === "followed" ? "ovb-fol" : "ovb-dev" }));
    G.append(pick(g, c));
    if (ranks.has(c.n)) G.append(s("text", { x: p.x0 + p.w + 8, y: ty + 13, class: "ovb-chip" }, chipText(c)));
  }
  const idleSel = idlePicked();
  for (const c of idles) {
    const p = tagAt[c.n], ty = yBar - 6 - TH - (p.row - 1) * (TH + TG);
    G.append(s("line", { x1: p.xs, x2: p.xs, y1: ty + TH, y2: yBar - 2, class: "ovb-stem k-idle" }));
    const g = s("g", { class: "ovb-step k-idle", "data-idle": c.n }, s("rect", { x: p.x0, y: ty, width: p.w, height: TH, rx: 4 }),
      s("text", { x: p.x0 + 6, y: ty + 13, class: "ovb-tx ovb-tag" }, tagLabel(c)));
    if (idleSel === e.idle_steps[c.idle]) g.classList.add("sel");
    G.append(markWith(g, IDLE_TIP, () => pickIdle(c.idle)));
  }
  /* b. the work orders, one row each */
  const OH = 36, yb = band(Math.max(1, lives.length) * OH, narrow ? "Work order" : "What happened to the work order?", lives.length > 1 ? "one row per order" : "");
  if (!lives.length) G.append(s("text", { x: X(0) + 4, y: yb + 22, class: "ovb-muted" }, "No work order in this episode."));
  lives.forEach((l, i) => {
    const y0 = yb + i * OH, c = call(l.call), t0 = l.created != null ? l.created : c.time_after;
    const tEx = l.exec != null ? l.exec : T, tEnd = l.close != null ? l.close : T;
    G.append(s("text", { x: X(t0) + 4, y: y0 + 12, class: "ovb-olab" }, `${l.component} repair ordered at step ${l.call}${l.exec == null ? ", still waiting for the spare part when the episode ended" : ""}`));
    G.append(s("line", { x1: X(t0), x2: X(tEx), y1: y0 + 25, y2: y0 + 25, class: "ovb-wait" }));
    if (l.exec != null) {
      const w = Math.max(4, X(tEnd) - X(l.exec));
      G.append(s("rect", { x: X(l.exec), y: y0 + 17, width: w, height: 17, rx: 4, class: "ovb-repair" }));
      const tx = l.close != null ? `repair, done at cycle ${l.close}` : "repair, not done when the episode ended";
      G.append(s("text", { x: X(l.exec) + 7, y: y0 + 30, class: "ovb-reptx" }, clipText(tx, 12, 600, w - 12)));
    }
  });
  /* c. the asset */
  const fails = e.calls.flatMap((c) => (c.notifications || []).filter((n) => n.event === "failure").map((n) => ({ t: n.t, step: c.n })));
  const yc = band(18 + 24 + (people ? 12 : 0), narrow ? "Machine" : "Is the machine running?", "");
  const yM = yc + 18;
  G.append(s("rect", { x: X(0), y: yM, width: X(T) - X(0), height: 22, class: "ovb-run" }));
  const downs = lives.filter((l) => l.exec != null).map((l) => [l.exec, l.close != null ? l.close : T]);
  if (!downs.some(([a]) => a === 0)) G.append(s("text", { x: X(0) + 6, y: yM + 15, class: "ovb-runtx" }, "running"));
  for (const [a, b] of downs) {
    G.append(s("rect", { x: X(a), y: yM, width: Math.max(3, X(b) - X(a)), height: 22, class: "ovb-down" }));
    G.append(s("text", { x: X(a) + 6, y: yM + 15, class: "ovb-downtx" }, clipText("stopped for the repair", 12, 400, X(b) - X(a) - 10)));
  }
  for (const f of fails) {
    const xf = X(f.t), tx = `a part failed (the agent learned of it at step ${f.step})`, tw = textWidth(tx, 11, 400);
    G.append(s("path", { d: `M${xf},${yM - 2} l-5,-8 h10 z`, class: "ovb-fail" }));
    G.append(s("text", { x: xf - tw - 8 > X(0) ? xf - 8 : xf + 8, y: yM - 6, "text-anchor": xf - tw - 8 > X(0) ? "end" : "start", class: "ovb-failtx" }, tx));
  }
  if (people) {
    const gH = s("g", { class: HID });
    for (const p of (S.hidden.end || {}).played || []) {
      if (p.t >= T) continue;
      const comb = Array.isArray(p.combination) ? Object.fromEntries(comps().map((k, i) => [k, p.combination[i]])) : p.combination;
      const faulty = faultySet(comb);
      gH.append(s("rect", { x: X(p.t), y: yM + 26, width: px, height: 8, class: faulty.length ? "tl-true-bad" : "tl-true-ok" },
        s("title", {}, `cycle ${p.t}, ${faulty.length ? `${andList(faulty)} faulty` : "no faulty part"}`)));
    }
    G.append(gH);
  }
  /* d. the budget */
  const HB = 120, yd = band(HB, narrow ? "Budget" : "How much budget is left?", "drops where a step costs money");
  if (B == null) G.append(s("text", { x: X(0) + 4, y: yd + 30, class: "ovb-muted" }, "This episode has no budget."));
  else {
    const yv = (v) => yd + 14 + (HB - 28) * (1 - Math.max(0, v) / B);
    G.append(s("line", { x1: X(0), x2: X(T), y1: yv(B), y2: yv(B), class: "ovb-budline" }), s("line", { x1: X(0), x2: X(T), y1: yv(0), y2: yv(0), class: "ovb-zero" }));
    G.append(s("text", { x: X(T), y: yv(B) - 5, "text-anchor": "end", class: "ovb-muted" }, `budget ${B}`));
    let d = `M${X(0)},${yv(B)}`, lastEnd = -1e9, lift = 0;
    for (const c of e.calls) {
      if (!c.price) continue;
      d += ` H${X(c.time_before)} V${yv(c.budget_after)}`;
      const tx = `step ${c.n} cost ${c.price}`, x0 = X(c.time_before) + 4;
      lift = x0 < lastEnd + 8 ? lift + 15 : 0; lastEnd = x0 + textWidth(tx, 12, 700);
      G.append(s("text", { x: x0, y: yv(c.budget_after) + 16 + lift, class: "ovb-drop" }, tx));
    }
    G.append(s("path", { d: `${d} H${X(T)}`, class: "ovb-budget" }));
    const left = e.status ? e.status.budget_left : e.calls.length ? e.calls[e.calls.length - 1].budget_after : B;
    G.append(s("text", { x: X(T) - 4, y: yv(left) - 7, "text-anchor": "end", class: "ovb-left" }, `${left} of ${B} left`));
  }
  /* the axis, the grid and the selected step */
  const yAx = y - 4;
  for (let t = 0; t <= T; t++) {
    grid.append(s("line", { x1: X(t), x2: X(t), y1: 2, y2: yAx - 14, class: "ovb-gl" }));
    grid.append(s("text", { x: t === T ? X(t) - 3 : X(t) + 3, y: yAx + 2, "text-anchor": t === T ? "end" : "start", class: "ovb-ax" }, `cycle ${t}`));
  }
  const xs = X(sel.time_before);
  grid.append(s("line", { x1: xs, x2: xs, y1: 2, y2: yAx - 14, class: "ovb-selline" }));
  svg.setAttribute("height", yAx + 10);
  svg.setAttribute("viewBox", `0 0 ${W} ${yAx + 10}`);
  svg.append(grid, G);
  // the page's content policy refuses style attributes, so sizes go through the style object
  const labs = h("div", { class: "ovb-labs" }, bands.map((b) => {
    const lab = h("div", { class: "ovb-lab" }, h("p", { class: "q" }, b.q), b.sub && !narrow ? h("p", { class: "qsub" }, b.sub) : null);
    lab.style.top = `${b.y}px`;
    return lab;
  }));
  labs.style.width = `${LABW}px`;
  labs.style.height = `${yAx + 10}px`;
  const scroller = h("div", { class: "ovb-scroll" }, svg);
  host.replaceChildren(h("div", { class: "ovb-grid-wrap" }, labs, scroller));
  if (W > scroller.clientWidth + 1) {
    const note = h("p", { class: "qsub ovb-scrollnote" });
    host.prepend(note);
    const upd = () => {
      const left = scroller.scrollLeft, right = left + scroller.clientWidth, ts = [];
      for (let t = 0; t <= T; t++) { const x0 = t === T ? X(t) - 3 - textWidth(`cycle ${t}`, 12, 400) : X(t) + 3; if (x0 >= left && x0 + textWidth(`cycle ${t}`, 12, 400) <= right) ts.push(t); }
      const a = ts.length ? ts[0] : 0, b = ts.length ? ts[ts.length - 1] : 0;
      note.textContent = `Cycles ${a} to ${b} of ${T} in view. Scroll sideways for the rest.`;
    };
    scroller.addEventListener("scroll", upd);
    // keep the selected step in view
    scroller.scrollLeft = Math.max(0, xs - scroller.clientWidth / 3);
    upd();
  }
}

/* ---------- side views ---------- */
function sidePanel(title, sub, ...body) {
  return h("section", { class: "panel" }, h("div", { class: "panel-h" }, h("h2", {}, title), sub ? h("span", { class: "sub" }, sub) : null), h("div", { class: "panel-b" }, body));
}
function sideChecks() {
  const box = hid(h("section", { class: "panel" }));
  const V = S.hidden && S.hidden.verdict;
  if (!V) { box.append(h("div", { class: "panel-b" }, S.hidden ? "This episode has no score yet." : "Loading the ground truth.")); return box; }
  const det = (o) => Object.entries(o).filter(([k]) => k !== "ok").map(([k, v]) => `${k} ${summarize(v)}`).join("; ");
  box.append(h("div", { class: "panel-h" }, h("h2", {}, h("span", { class: "hidden-label" }, "Ground truth "), "how the run was scored"),
      h("span", { class: "sub" }, V.pass ? "passed" : `failed, ${String(V.failure_class).replace(/_/g, " ")}`)),
    h("div", { class: "panel-b" }, h("ul", { class: "checks" }, Object.entries(V.checks || {}).map(([k, o]) => h("li", {},
      h("span", { class: o.ok ? "ok" : "no", "aria-label": o.ok ? "passed" : "failed" }, o.ok ? "✓" : "✗"),
      h("div", {}, h("b", {}, k.replace(/_/g, " ")), h("div", { class: "det" }, det(o)))))),
      h("dl", { class: "kv" }, h("dt", {}, "machine working again"), h("dd", {}, V.recovered ? "yes" : "no"), h("dt", {}, "repairs that were not needed"), h("dd", {}, String(V.unnecessary)),
        h("dt", {}, "the machine's state at the end"), h("dd", {}, V.final ? combText(V.final, true) : NR),
        h("dt", {}, "the fault could be found within the budget"), h("dd", {}, V.separable ? "yes" : "no"))));
  return box;
}

/* ---------- Graph tab ---------- */
const DEC_TYPES = {
  knowledge: { label: "Look-up", glyph: "K", css: "k-know" },
  observation: { label: "Reading", glyph: "O", css: "k-reads" },
  advice: { label: "Advice", glyph: "A", css: "k-advice" },
  decision: { label: "Diagnosis and finish", glyph: "D", css: "k-decide" },
  order: { label: "Work order", glyph: "W", css: "k-orders" },
  time: { label: "Time", glyph: "T", css: "k-time" },
  event: { label: "Event", glyph: "E", css: "k-event" },
  verdict: { label: "Score", glyph: "V", css: "k-verdict", hidden: true },
};
const ASSET_TYPES = {
  component: { label: "Part", glyph: "C", css: "k-know" },
  sensorset: { label: "Sensor set", glyph: "S", css: "k-time" },
  channel: { label: "Sensor", glyph: "", css: "k-reads" },
  mode: { label: "Failure mode", glyph: "F", css: "k-decide" },
};
function decisionModel() {
  const e = ep(), N = [], E = [], byId = {};
  const add = (n) => { if (!byId[n.id]) { byId[n.id] = n; N.push(n); } return byId[n.id]; };
  const edge = (s0, t0, label, cls, hidden) => { const a = byId[s0], b = byId[t0]; if (a && b && a !== b) E.push({ s: a.id, t: b.id, label, cls: cls || label.replace(/\s.*/, ""), hidden }); };
  let prevCall = null;
  for (const c of e.calls) {
    let id;
    if (c.served) {
      if (!c.served.length) { id = `call/${c.n}`; add({ id, type: "observation", t: c.n, call: c.n, label: `${c.args.sensor_group || c.tool} sensors, nothing came back`, props: { call: c.n, tool: callText(c), served: 0, stopped: c.answer.result.stopped } }); }
      for (const o of c.served) {
        const means = Object.entries(o.values || {}).filter(([k]) => bandOf(k));
        id = `obs/${o.n}`;
        add({ id, type: "observation", t: c.n, call: c.n, label: `reading ${o.n}, ${o.sensor_group} sensors, cycle ${o.t}`,
              props: { step: c.n, cycle: o.t, "sensor set": o.sensor_group, values: Object.keys(o.values || {}).length, "sensors outside the normal range": means.filter(([k, v]) => outside(k, v)).map(([k]) => short(k)).join(", ") || "none" } });
      }
    } else {
      const type = c.kind === "reads" ? "knowledge" : c.kind === "advice" ? "advice" : c.kind === "orders" ? "order" : c.kind === "time" ? "time" : "decision";
      const res = c.answer && c.answer.result;
      const wo = res && res.work_order;
      id = wo ? `wo/${wo.id}` : `call/${c.n}`;
      const lbl = wo ? `${c.n} Work order for the ${wo.component}` : `${c.n} ${stepWords(c)}`;
      add({ id, type, t: c.n, call: c.n, label: lbl, props: { step: c.n, tool: callText(c), "followed the advice": c.follow ? FOLLOW_WORDS[c.follow] : null, refused: c.refused, cost: c.price, cycles: c.cycles, message: c.kind === "advice" ? adviceWords(c) : c.answer && c.answer.message } });
      // the call that issued a prediction stands for it; a later check of the same id does not take it over
      const pid = c.kind === "advice" && c.answer.prediction_id ? `pred/${c.answer.prediction_id.replace(/^pred\//, "")}` : null;
      if (pid && !byId[pid]) byId[pid] = byId[id];
    }
    for (const nt of c.notifications || []) {
      if (nt.event === "alarm") continue;
      add({ id: `ev/${nt.event}/${nt.id || nt.t}`, type: "event", t: c.n + 0.45, call: c.n, label: `${nt.event.replace(/_/g, " ")}${nt.id ? ` ${nt.id}` : ""}, cycle ${nt.t}`, props: { ...nt, "came with step": c.n } });
    }
    const al = (c.notifications || []).filter((x) => x.event === "alarm");
    if (al.length) {
      add({ id: `ev/alarms/${c.n}`, type: "event", t: c.n + 0.45, call: c.n, label: `${plural(al.length, "alarm")}`, props: { channels: al.map((x) => short(x.channel)).join(", "), cycles: [...new Set(al.map((x) => x.t))].join(", "), "came with step": c.n } });
      edge(id, `ev/alarms/${c.n}`, "raises");
    }
    if (prevCall) edge(prevCall, id, "then", "then");
    prevCall = id;
  }
  for (const r of e.rankings) {
    const rid = `call/${r.n}`;
    for (const b of r.branches) for (const pid of [b.plan_prediction_id, ...b.steps]) {
      const p = e.predictions.find((x) => x.id === pid);
      if (!p || byId[pid]) continue;
      add({ id: pid, type: "advice", t: r.n + 0.45, call: r.n, label: `Prediction, repair of the ${andList(p.action.replace(/^[^:]*:/, "").split(","))}`, props: { repair: p.action, "chance the machine works again": pct(p.recovery), "chance it fixes a fault": pct(p.addresses), plan: p.branch } });
      edge(rid, pid, "predicts");
    }
  }
  for (const c of e.calls) {
    const a = c.answer || {};
    const id = c.served ? null : (a.result && a.result.work_order ? `wo/${a.result.work_order.id}` : `call/${c.n}`);
    if (c.kind === "advice" && Array.isArray(a.evidence_ids)) {
      const cited = new Set(a.evidence_ids.map((x) => x.split("/").slice(0, 2).join("/")));
      for (const o of cited) if (o.startsWith("obs/")) edge(o, id, "informs");
      if (a.evidence_ids.some((x) => x.startsWith("pred/")) && a.prediction_id) edge(id, a.prediction_id, "checks");
    }
    if (c.follow === "followed" || c.follow === "deviated") edge(id, `call/${c.advice_in_force}`, c.follow === "followed" ? "follows" : "goes against", c.follow === "followed" ? "e-follows" : "e-deviates");
    if (c.args && c.args.prediction_id) edge(id, c.args.prediction_id, "cites");
    for (const ev of String((c.args || {}).evidence_ids || "").split(",").map((x) => x.trim()).filter(Boolean)) edge(id, ev.split("/").slice(0, 2).join("/"), "cites");
  }
  for (const w of e.orders) {
    edge(`wo/${w.id}`, `ev/work_order_executed/${w.id}`, "leads to", "causes");
    edge(`ev/work_order_executed/${w.id}`, `ev/work_order_closed/${w.id}`, "leads to", "causes");
  }
  const V = S.hidden && S.hidden.verdict;
  if (V) {
    const last = e.calls.length + 1.2;
    add({ id: "verdict", type: "verdict", t: last, hidden: true, label: `score, ${V.pass ? "passed" : "failed"}`, props: { passed: V.pass, "reason it failed": V.failure_class, cost: V.cost, "machine working again": V.recovered } });
    add({ id: "hstart", type: "verdict", t: 0.4, hidden: true, label: `real fault at the start, ${combText(S.hidden.start_state)}`, props: { "real fault at the start": combText(S.hidden.start_state, true), "start point in the recording": S.hidden.entry_offset } });
    const diag = e.calls.find((c) => c.answer && c.answer.result && c.answer.result.diagnosis);
    if (diag) edge("hstart", `call/${diag.n}`, sameComb(diag.args, S.hidden.start_state) ? "confirms" : "refutes", sameComb(diag.args, S.hidden.start_state) ? "e-confirms" : "e-refutes", true);
    for (const [k, o] of Object.entries(V.checks || {})) {
      add({ id: `chk/${k}`, type: "verdict", t: last + 0.6, hidden: true, label: `${o.ok ? "✓" : "✗"} ${k.replace(/_/g, " ")}`, props: { check: k, ok: o.ok, ...Object.fromEntries(Object.entries(o).filter(([x]) => x !== "ok").map(([x, v]) => [x, summarize(v)])) } });
      edge("verdict", `chk/${k}`, "has check", "e-has", true);
    }
    if (V.checks.hard && diag) edge(`chk/hard`, `call/${diag.n}`, V.checks.hard.diagnosis_all_correct ? "confirms" : "refutes", V.checks.hard.diagnosis_all_correct ? "e-confirms" : "e-refutes", true);
    for (const w of e.orders) {
      const un = V.checks.outcome && (V.checks.outcome.unnecessary_orders || []).includes(w.id);
      if (V.checks.outcome) edge("chk/outcome", `wo/${w.id}`, un ? "refutes" : "confirms", un ? "e-refutes" : "e-confirms", true);
    }
    if (V.checks.causality && diag) edge("chk/causality", `call/${diag.n}`, V.checks.causality.ok ? "confirms" : "refutes", V.checks.causality.ok ? "e-confirms" : "e-refutes", true);
  }
  for (const n of N) n.sel = n.call === S.step;
  return { nodes: N, edges: E, types: DEC_TYPES, lanes: ["knowledge", "observation", "advice", "decision", "order", "time", "event", "verdict"] };
}
function assetModel() {
  const e = ep(), r = rules(), N = [], E = [], byId = {};
  const add = (n) => { if (!byId[n.id]) { byId[n.id] = n; N.push(n); } return byId[n.id]; };
  const edge = (a, b, label, cls) => { if (byId[a] && byId[b]) E.push({ s: a, t: b, label, cls: cls || label.split(" ")[0] }); };
  const stat0 = (r.statistics || ["mean"])[0];
  const known = e.calls.filter((c) => c.n <= S.step);
  const read = new Set(known.flatMap((c) => (c.served || []).flatMap((o) => Object.keys(o.values || {}))));
  const lastVal = {};
  for (const c of known) for (const o of c.served || []) for (const [k, v] of Object.entries(o.values || {})) lastVal[k] = v;
  const rk = lastRanking(S.step);
  const repaired = new Set(e.closes.filter((x) => x.call <= S.step).map((x) => x.component));
  const truth = S.hidden && S.hidden.start_state;
  comps().forEach((k, i) => {
    const p = rk && rk.fault_probability ? rk.fault_probability[k] : null;
    add({ id: `comp/${k}`, type: "component", label: `${k}${p != null ? ` · ${pct(p)}` : ""}${repaired.has(k) ? " · repaired" : ""}`, susp: p, repaired: repaired.has(k),
          faulty: truth ? truth[i] !== r.healthy[k] : null,
          props: { levels: (r.levels[k] || []).join(", "), "healthy level": r.healthy[k], "chance it is faulty (last ranking)": p == null ? NR : pct(p), "repaired by now": repaired.has(k) ? "yes" : "no", "spare part price": (r.prices.parts || {})[k], "repair order": ((r.processes || {}).priority || r.priority || {})[k] } });
  });
  for (const [set, chans] of Object.entries(r.sensor_groups || {})) {
    add({ id: `set/${set}`, type: "sensorset", label: `${set} sensors (${chans.length})`, props: { sensors: chans.length, "cost per cycle": (r.prices.read || {})[set] } });
    for (const ch of chans) {
      const f = `${ch}_${stat0}`;
      const v = lastVal[f];
      add({ id: `ch/${ch}`, type: "channel", label: sensorName(ch).name, glyph: short(ch), read: read.has(f), out: v != null && outside(f, v),
            props: { name: ch, "read by now": read.has(f) ? "yes" : "no", [`last ${stat0}`]: v == null ? "none" : num(v), "normal range": bandOf(f) ? `${num(bandOf(f).low)} to ${num(bandOf(f).high)}` : NR } });
      edge(`set/${set}`, `ch/${ch}`, "contains");
    }
  }
  const deps = (r.processes || {}).dependencies || r.dependencies || {};
  for (const [k, ds] of Object.entries(deps)) for (const d of ds) edge(`comp/${k}`, `comp/${d}`, "depends on", "e-dep");
  const modes = [];
  for (const c of known) {
    const res = c.answer && c.answer.result;
    if (res && Array.isArray(res.failure_modes)) modes.push(...res.failure_modes.map((m) => ({ ...m, src: `step ${c.n}` })));
    if (c.kind === "advice" && Array.isArray(c.answer.candidates)) for (const cand of c.answer.candidates) for (const f of cand.faults || []) modes.push({ ...f, src: `ranking of step ${c.n}` });
  }
  for (const m of modes) {
    const id = `mode/${m.code}`;
    add({ id, type: "mode", label: m.code.replace(/_/g, " "), props: { part: m.component, level: m.level, severity: m.severity || NR, repair: m.action || NR, "given by": m.src } });
    edge(`comp/${m.component}`, id, "can fail as", "has");
    for (const ch of m.channels || []) edge(id, `ch/${ch}`, "shows in", "indicated");
  }
  return { nodes: N, edges: E, types: ASSET_TYPES, lanes: ["sensorset", "channel", "mode", "component"] };
}
function tabGraph() {
  const mode = S.graphMode;
  const seg = (items, cur, set) => h("div", { class: "seg", role: "group" }, items.map(([k, t]) => h("button", { type: "button", "aria-pressed": String(cur === k), onclick: () => { set(k); render(); } }, t)));
  return h("div", {},
    h("div", { class: "graph-tools" },
      seg([["decision", "What happened"], ["asset", "How the machine is built"]], mode, (k) => { S.graphMode = k; }),
      seg(mode === "decision" ? [["lanes", "Rows by kind"], ["free", "Free layout"]] : [["lanes", "Columns"], ["free", "Free layout"]], S.layout[mode], (k) => { S.layout[mode] = k; delete S.pos[mode]; delete S.view[mode]; }),
      mode === "decision" ? h("label", { class: "sub" }, h("input", { type: "checkbox", checked: S.off.decision.has("then") ? null : "", onchange: (ev) => { ev.target.checked ? S.off.decision.delete("then") : S.off.decision.add("then"); render(); } }), " lines in step order") : null,
      mode === "decision" ? h("label", { class: "sub" }, h("input", { type: "checkbox", checked: S.chainOnly ? "" : null, onchange: (ev) => { S.chainOnly = ev.target.checked; render(); } }), " highlight the steps behind the final action") : null,
      h("label", { class: "sub" }, h("input", { type: "checkbox", checked: S.edgeLabels ? "" : null, onchange: (ev) => { S.edgeLabels = ev.target.checked; render(); } }), " show every line's label"),
      h("span", { class: "sub" }, mode === "decision" ? "One dot for each step, reading, prediction and event, placed left to right by step." : `At step ${S.step}, the sensors read, the sensors outside the normal range, the chance each part is faulty and the repairs done.`)),
    h("div", { id: "graph-legend", class: "legend" }),
    h("div", { class: "graph-wrap" },
      h("div", { class: "canvas", id: "graph-host" }),
      h("div", { class: "inspector", id: "graph-inspector", "aria-live": "polite" }, h("p", { class: "sub" }, "Click a dot, or press Enter on it, to see what it is."))));
}
function mountGraph(host) {
  const mode = S.graphMode;
  const model = mode === "decision" ? decisionModel() : assetModel();
  new GraphView(host, model, mode);
}

/* up to two lines, broken at an underscore or a space, so a tool name stays whole */
function wrapLabel(text, width) {
  if (text.length <= width) return [text];
  const cut = Math.max(text.lastIndexOf(" ", width), text.lastIndexOf("_", width));
  const at = cut > 4 ? cut + 1 : width;
  const rest = text.slice(at);
  return [text.slice(0, at), rest.length > width + 4 ? `${rest.slice(0, width + 3)}\u2026` : rest];
}

class GraphView {
  constructor(host, model, mode) {
    this.host = host; this.model = model; this.mode = mode; this.off = S.off[mode];
    this.byId = Object.fromEntries(model.nodes.map((n) => [n.id, n]));
    this.adj = {};
    for (const e of model.edges) { (this.adj[e.s] ||= new Set()).add(e.t); (this.adj[e.t] ||= new Set()).add(e.s); }
    this.radius = (n) => (n.type === "channel" ? 16 : n.type === "event" ? 12 : 17);
    this.buildLegend();
    this.buildSvg();
    const saved = S.pos[mode];
    if (saved && model.nodes.every((n) => saved[n.id])) model.nodes.forEach((n) => Object.assign(n, saved[n.id]));
    else if (S.layout[mode] === "lanes") this.layoutLanes();
    else { this.layoutLanes(); this.spring(true); }
    this.draw();
    const keep = S.view[mode] && !(mode === "decision" && S.viewStep !== S.step);
    if (keep) { this.tf = S.view[mode]; this.applyTf(); } else if (mode === "decision" && S.layout.decision === "lanes") this.focusStep(); else this.fit();
    S.viewStep = S.step;
    const sel = S.selected[mode] && this.byId[S.selected[mode]];
    if (sel) this.inspect(sel);
    else if (mode === "decision") { const cur = model.nodes.find((n) => n.sel); if (cur) this.inspect(cur, true); }
  }
  visible(n) { return !this.off.has(n.type) && !(n.hidden && !S.showHidden); }
  buildLegend() {
    const lg = document.getElementById("graph-legend");
    const counts = {};
    for (const n of this.model.nodes) counts[n.type] = (counts[n.type] || 0) + 1;
    lg.replaceChildren(...Object.entries(this.model.types).filter(([k]) => counts[k] || !this.model.types[k].hidden).map(([k, t]) => {
      const b = h("button", { type: "button", class: t.css, "aria-pressed": String(!this.off.has(k)), title: `Show or hide ${t.label.toLowerCase()}`,
        onclick: () => { this.off.has(k) ? this.off.delete(k) : this.off.add(k); b.setAttribute("aria-pressed", String(!this.off.has(k))); this.applyFilter(); } },
        h("span", { class: "ldot" }, t.glyph || "•"), t.label, h("span", { class: "cnt" }, String(counts[k] || 0)));
      return t.hidden ? hid(b) : b;
    }));
  }
  buildSvg() {
    const svg = s("svg", { class: S.edgeLabels ? "all-labels" : null, role: "img", "aria-label": `${this.mode} graph with ${this.model.nodes.length} nodes and ${this.model.edges.length} edges` });
    svg.append(s("defs", {}, s("marker", { id: `arr-${this.mode}`, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "8", markerHeight: "8", orient: "auto-start-reverse", markerUnits: "userSpaceOnUse" },
      s("path", { class: "g-marker", d: "M0,0 L10,5 L0,10 z" }))));
    this.view = s("g", {}); this.gl = s("g", {}); this.ge = s("g", {}); this.gn = s("g", {});
    this.view.append(this.gl, this.ge, this.gn); svg.append(this.view);
    this.svg = svg;
    this.host.replaceChildren(svg,
      h("div", { class: "zoom" }, h("button", { type: "button", "aria-label": "Zoom in", onclick: () => this.zoomBy(1.25) }, "+"),
        h("button", { type: "button", "aria-label": "Zoom out", onclick: () => this.zoomBy(0.8) }, "−"),
        h("button", { type: "button", "aria-label": "Fit to view", onclick: () => this.fit() }, "⤢")),
      h("div", { class: "hint" }, "drag a dot, drag the background to move, scroll to zoom"));
    this.tf = { k: 1, x: 0, y: 0 };
    let drag = null;
    svg.addEventListener("pointerdown", (ev) => {
      const g = ev.target.closest(".g-node"), ge = g ? null : ev.target.closest(".g-edge");
      svg.setPointerCapture(ev.pointerId);
      drag = { id: g ? g.dataset.id : null, edge: ge ? +ge.dataset.i : null, x: ev.clientX, y: ev.clientY, moved: false, tx: this.tf.x, ty: this.tf.y };
      if (!g && !ge) svg.classList.add("panning");
    });
    svg.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.hypot(dx, dy) > 3) drag.moved = true;
      if (!drag.moved) return;
      if (drag.id) {
        const n = this.byId[drag.id];
        n.x += (ev.movementX || 0) / this.tf.k; n.y += (ev.movementY || 0) / this.tf.k; n.fixed = true;
        this.positions();
        if (S.layout[this.mode] === "free" && !reduceMotion.matches) this.spring(false);
      } else { this.tf.x = drag.tx + dx; this.tf.y = drag.ty + dy; this.applyTf(); }
    });
    svg.addEventListener("pointerup", () => {
      if (drag && drag.id && !drag.moved) this.inspect(this.byId[drag.id]);
      else if (drag && drag.edge != null && !drag.moved) this.inspectEdge(drag.edge);
      drag = null; svg.classList.remove("panning"); this.save();
    });
    svg.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const r = svg.getBoundingClientRect();
      this.zoomAt(ev.deltaY < 0 ? 1.12 : 1 / 1.12, ev.clientX - r.left, ev.clientY - r.top);
    }, { passive: false });
    svg.addEventListener("pointerover", (ev) => { const g = ev.target.closest(".g-node"); if (g) this.light(g.dataset.id); });
    svg.addEventListener("pointerout", (ev) => { const g = ev.target.closest(".g-node"); if (g) this.light(S.selected[this.mode] || null); });
  }
  layoutLanes() {
    const lanes = this.model.lanes, ns = this.model.nodes;
    if (this.mode === "decision") {
      const X = 96, stack = {};
      lanes.forEach((ln, i) => {
        const here = ns.filter((n) => n.type === ln);
        for (const n of here) { const key = `${ln}/${Math.floor(n.t)}`; const k = (stack[key] = (stack[key] || 0) + 1) - 1; n.x = 150 + Math.floor(n.t) * X; n.lane = i; n.k = k; }
      });
      let y = 50;
      this.laneY = lanes.map((ln) => { const depth = Math.max(1, ...ns.filter((n) => n.type === ln).map((n) => n.k + 1)); const top = y; y += 34 + depth * 60; return { ln, top, h: y - top }; });
      for (const n of ns) n.y = this.laneY[n.lane].top + 30 + n.k * 60;
    } else {
      const cols = { sensorset: 0, channel: 1, mode: 2, component: 3 };
      const groups = {};
      for (const n of ns) (groups[n.type] ||= []).push(n);
      for (const [t, g] of Object.entries(groups)) g.forEach((n, i) => { n.x = 80 + cols[t] * 260; n.y = 40 + (i + 0.5) * (Math.max(560, 42 * g.length) / g.length); });
      this.laneY = null;
    }
  }
  spring(fromScratch) {
    const ns = this.model.nodes.filter((n) => this.visible(n)), es = this.model.edges.filter((e) => e.cls !== "then");
    if (fromScratch) ns.forEach((n, i) => { n.x += Math.cos(i * 2.4) * 30; n.y += Math.sin(i * 2.4) * 30; n.vx = 0; n.vy = 0; });
    let alpha = fromScratch ? 1 : 0.4;
    const step = () => {
      for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
        const a = ns[i], b = ns[j]; let dx = b.x - a.x, dy = b.y - a.y; let d2 = dx * dx + dy * dy || 1;
        if (d2 > 90000) continue;
        const f = (2600 / d2) * alpha, d = Math.sqrt(d2); dx /= d; dy /= d;
        a.vx = (a.vx || 0) - dx * f; a.vy = (a.vy || 0) - dy * f; b.vx = (b.vx || 0) + dx * f; b.vy = (b.vy || 0) + dy * f;
      }
      for (const e of es) {
        const a = this.byId[e.s], b = this.byId[e.t]; if (!this.visible(a) || !this.visible(b)) continue;
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, f = (d - 120) * 0.03 * alpha;
        a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
      }
      const cx = ns.reduce((s0, n) => s0 + n.x, 0) / ns.length, cy = ns.reduce((s0, n) => s0 + n.y, 0) / ns.length;
      for (const n of ns) {
        if (n.fixed) { n.vx = n.vy = 0; continue; }
        n.vx = (n.vx + (cx - n.x) * 0.004 * alpha) * 0.82; n.vy = (n.vy + (cy - n.y) * 0.004 * alpha) * 0.82;
        n.x += Math.max(-30, Math.min(30, n.vx)); n.y += Math.max(-30, Math.min(30, n.vy));
      }
      alpha *= 0.985;
    };
    if (reduceMotion.matches || fromScratch) { for (let i = 0; i < 360; i++) step(); if (!fromScratch) this.positions(); return; }
    cancelAnimationFrame(this.raf);
    const loop = () => { step(); this.positions(); if (alpha > 0.03 && this.svg.isConnected) this.raf = requestAnimationFrame(loop); else this.save(); };
    this.raf = requestAnimationFrame(loop);
  }
  draw() {
    this.gl.replaceChildren(); this.ge.replaceChildren(); this.gn.replaceChildren();
    if (this.laneY && S.layout[this.mode] === "lanes") {
      const xmax = Math.max(...this.model.nodes.map((n) => n.x)) + 60;
      this.laneY.forEach((l, i) => {
        const t = this.model.types[l.ln];
        const g = s("g", { class: t.hidden ? HID : null });
        if (i % 2 === 0) g.append(s("rect", { class: "lane-band", x: 0, y: l.top, width: xmax, height: l.h }));
        g.append(s("text", { class: "lane-label", x: 8, y: l.top + 16 }, t.label));
        this.gl.append(g);
      });
    }
    this.edgeEls = this.model.edges.map((e, i) => {
      const g = s("g", { class: ["g-edge", e.cls && e.cls.startsWith("e-") ? e.cls : `e-${e.cls}`, e.hidden ? HID : null], "data-i": i });
      const hitp = s("path", { class: "e-hit" });
      const p = s("path", { "marker-end": `url(#arr-${this.mode})` });
      const t = e.cls === "then" ? s("text", {}) : s("text", {}, e.label);
      g.append(s("title", {}, "Click to see the records at both ends"), hitp, p, t); this.ge.append(g);
      return { e, g, p, t, hitp };
    });
    this.nodeEls = this.model.nodes.map((n) => {
      const t = this.model.types[n.type], r = this.radius(n);
      const g = s("g", { class: ["g-node", t.css, n.sel ? "sel" : null, n.hidden ? HID : null, n.type === "channel" && !n.read ? "unread" : null],
        "data-id": n.id, tabindex: "0", role: "button", "aria-label": `${t.label}: ${n.label}` });
      g.append(s("title", {}, `${t.label}: ${n.label}`));
      if (n.susp != null) g.append(s("circle", { class: "susp", r: r + 4 + n.susp * 6, opacity: 0.15 + n.susp * 0.5 }));
      g.append(s("circle", { class: "body", r }));
      if (n.read) g.append(s("circle", { class: "ring-read", r: r + 3 }));
      if (n.out) g.append(s("circle", { class: "ring-out", r: r + 3 }));
      if (n.faulty) g.append(hid(s("circle", { class: "faulty-ring", r: r + 9 })));
      if (n.repaired) g.append(s("circle", { class: "repaired", r: 5, cx: r * 0.75, cy: -r * 0.75 }));
      g.append(s("text", { class: "glyph" }, n.glyph != null ? n.glyph : t.glyph));
      if (!n.noLabel) {
        const lines = wrapLabel(n.label, 18);
        const above = this.mode === "decision" && S.layout.decision === "lanes" && Math.floor(n.t) % 2 === 1 && !n.k;
        const y0 = above ? -r - 6 - (lines.length - 1) * 13 : r + 14;
        const t = s("text", { class: "lbl", y: y0 });
        lines.forEach((ln, i) => t.append(s("tspan", { x: 0, dy: i ? 13 : 0 }, ln)));
        g.append(t);
      }
      if (n.out) g.append(s("text", { class: "out-mark", x: r * 0.8, y: -r * 0.7 }, "▲"));
      g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); this.inspect(n); } });
      g.addEventListener("focus", () => this.light(n.id));
      this.gn.append(g);
      return { n, g };
    });
    if (this.mode === "decision" && S.chainOnly) {
      const keep = chainIds();
      if (keep.size) {
        for (const { n, g } of this.nodeEls) g.classList.toggle("offchain", !keep.has(n.id));
        for (const { e, g } of this.edgeEls) g.classList.toggle("offchain", !(keep.has(e.s) && keep.has(e.t)));
      }
    }
    this.applyFilter();
    this.positions();
  }
  positions() {
    for (const { n, g } of this.nodeEls) g.setAttribute("transform", `translate(${n.x},${n.y})`);
    const pairCount = {};
    for (const { e, p, t, hitp } of this.edgeEls) {
      const a = this.byId[e.s], b = this.byId[e.t];
      const key = [e.s, e.t].sort().join("|"); const k = (pairCount[key] = (pairCount[key] || 0) + 1) - 1;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, ux = dx / d, uy = dy / d;
      const bend = (k ? 26 * k : 0) + (S.layout[this.mode] === "lanes" && Math.abs(dy) < 4 && d > 120 ? 22 : 0);
      const mx = (a.x + b.x) / 2 - uy * bend, my = (a.y + b.y) / 2 + ux * bend;
      const ra = this.radius(a) + 2, rb = this.radius(b) + 4;
      const sx = a.x + ux * ra, sy = a.y + uy * ra, ex = b.x - ux * rb, ey = b.y - uy * rb;
      p.setAttribute("d", bend ? `M${sx},${sy} Q${mx},${my} ${ex},${ey}` : `M${sx},${sy} L${ex},${ey}`);
      hitp.setAttribute("d", p.getAttribute("d"));
      t.setAttribute("x", bend ? (a.x + b.x) / 4 + mx / 2 : mx); t.setAttribute("y", (bend ? (a.y + b.y) / 4 + my / 2 : my) - 3);
    }
  }
  applyFilter() {
    for (const { n, g } of this.nodeEls) g.style.display = this.off.has(n.type) ? "none" : "";
    for (const { e, g } of this.edgeEls) g.style.display = this.off.has(this.byId[e.s].type) || this.off.has(this.byId[e.t].type) || this.off.has(e.cls) ? "none" : "";
  }
  light(id) {
    const on = id && this.byId[id];
    this.svg.classList.toggle("dim", !!on);
    const nb = on ? new Set([id, ...(this.adj[id] || [])]) : new Set();
    for (const { n, g } of this.nodeEls) g.classList.toggle("lit", nb.has(n.id));
    for (const { e, g } of this.edgeEls) g.classList.toggle("lit", !!on && (e.s === id || e.t === id));
  }
  inspect(n, quiet) {
    if (!quiet) S.selected[this.mode] = n.id;
    for (const { n: m, g } of this.nodeEls) g.classList.toggle("sel", m.id === n.id || (!S.selected[this.mode] && m.sel));
    this.light(quiet ? null : n.id);
    const t = this.model.types[n.type];
    const box = h("div", {},
      h("div", {}, h("span", { class: ["kdot", t.css] }, t.glyph || "•"), " ", h("span", { class: "sub" }, t.label)),
      h("h4", {}, n.label),
      h("table", {}, h("tbody", {}, Object.entries(n.props || {}).filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", {}, summarize(v)))))),
      h("div", { class: "neigh sub" }, `linked to ${(this.adj[n.id] || new Set()).size}, `,
        [...(this.adj[n.id] || [])].map((x) => this.byId[x].label).join(", ") || "none"),
      n.call ? h("button", { class: "btn", type: "button", onclick: () => go(epHash("decision", n.call)) }, `Open step ${n.call}`) : null);
    const ins = document.getElementById("graph-inspector");
    ins.replaceChildren(n.hidden ? hid(box) : box);
    if (!quiet) showRaw(`${t.label}: ${n.label}`, nodeRefs(n));
  }
  inspectEdge(i) {
    const { e, g } = this.edgeEls[i];
    const a = this.byId[e.s], b = this.byId[e.t];
    this.light(null);
    g.classList.add("lit");
    showRaw(`Edge: ${a.label} ${e.label} ${b.label}`, [...nodeRefs(a), ...nodeRefs(b)], edgeNote(e, a, b));
  }
  /* time lanes open at full size around the selected step, so labels read at the size of the page's small text;
     the fit button shows the whole episode */
  focusStep() {
    const sel = this.model.nodes.find((n) => n.sel);
    const r = this.svg.getBoundingClientRect(), W = r.width || 800, H = r.height || 560;
    const ys = this.model.nodes.filter((n) => this.visible(n)).map((n) => n.y);
    const y0 = Math.min(...ys) - 50, y1 = Math.max(...ys) + 50;
    if (sel) {
      const near = new Set([sel.id]);
      for (const e of this.model.edges) if (!e.hidden && (e.s === sel.id || e.t === sel.id)) near.add(e.s === sel.id ? e.t : e.s);
      const box = this.model.nodes.filter((n) => near.has(n.id) && this.visible(n));
      // a label runs up to about 100 px either side of its dot, 60 px above or below it
      const bx0 = Math.min(...box.map((n) => n.x)) - 100, bx1 = Math.max(...box.map((n) => n.x)) + 100;
      const by0 = Math.min(...box.map((n) => n.y)) - 70, by1 = Math.max(...box.map((n) => n.y)) + 60;
      const kb = Math.min(1, W / (bx1 - bx0), H / (by1 - by0));
      this.tf = { k: kb, x: (W - (bx0 + bx1) * kb) / 2, y: (H - (by0 + by1) * kb) / 2 };
    } else {
      const k = Math.min(1, H / (y1 - y0));
      this.tf = { k, x: 0, y: (H - (y1 - y0) * k) / 2 - y0 * k };
    }
    this.applyTf();
  }
  applyTf() { this.view.setAttribute("transform", `translate(${this.tf.x},${this.tf.y}) scale(${this.tf.k})`); S.view[this.mode] = this.tf; }
  zoomAt(f, px, py) { const k = Math.min(4, Math.max(0.2, this.tf.k * f)); f = k / this.tf.k; this.tf.x = px - (px - this.tf.x) * f; this.tf.y = py - (py - this.tf.y) * f; this.tf.k = k; this.applyTf(); }
  zoomBy(f) { const r = this.svg.getBoundingClientRect(); this.zoomAt(f, r.width / 2, r.height / 2); }
  fit() {
    const ns = this.model.nodes.filter((n) => this.visible(n));
    if (!ns.length) return;
    const xs = ns.map((n) => n.x), ys = ns.map((n) => n.y);
    const x0 = this.laneY ? 0 : Math.min(...xs) - 60, x1 = Math.max(...xs) + 60, y0 = Math.min(...ys) - 50, y1 = Math.max(...ys) + 50;
    const r = this.svg.getBoundingClientRect(), W = r.width || 800, H = r.height || 560;
    const fitK = Math.min(1.3, W / (x1 - x0), H / (y1 - y0)), k = Math.max(0.6, fitK);
    // below the readable floor the view keeps the floor and starts at the top left; the person pans for the rest
    this.tf = fitK >= 0.6 ? { k, x: (W - (x1 - x0) * k) / 2 - x0 * k, y: (H - (y1 - y0) * k) / 2 - y0 * k } : { k, x: -x0 * k, y: -y0 * k };
    this.applyTf();
  }
  save() { S.pos[this.mode] = Object.fromEntries(this.model.nodes.map((n) => [n.id, { x: n.x, y: n.y, fixed: n.fixed }])); }
}

/* the four levels of the answers of the CausalDiagnosis MCP, one file per call (data.write_levels) */
async function loadLevels(sub, ns, into) {
  await Promise.all(ns.map(async (n) => {
    try { const r = await getText(`data/${sub}/${n}.json`); if (r.ok) into[n] = await r.json(); } catch (err) { /* the call keeps its one level */ }
  }));
}
/* ---------- hidden toggle, theme, keys ---------- */
async function setHidden(on) {
  if (on && !S.hidden) {
    try {
      S.hidden = await (await getText("data/hidden.json")).json();
      await loadFiles([...S.hidden.files.ep.map((f) => `ep/${f.name}`), ...S.hidden.files.run.map((f) => `run/${f.name}`)]);
      await loadLevels("hidden_levels", Object.entries(S.hidden.levels || {}).filter(([, v]) => v === "recomputed").map(([n]) => +n), S.hiddenLevels);
    } catch (err) { S.hidden = { error: String(err) }; }
  }
  S.showHidden = on;
  document.body.classList.toggle("show-hidden", on);
  hiddenChip(on);
  if (S.graphMode === "decision") delete S.pos.decision;
  render();
}
function setRawAll(on) {
  S.rawAll = on;
  try { localStorage.setItem("viewer.rawAll", on ? "1" : "0"); } catch (err) { /* storage refused: the switch still works for this visit */ }
  syncRawSeg();
  render();
}
function syncRawSeg() { for (const b of document.querySelectorAll("#raw-seg button")) b.setAttribute("aria-pressed", String((b.dataset.v === "source") === S.rawAll)); }
function hiddenChip(on) {
  const b = document.getElementById("hidden-toggle");
  b.setAttribute("aria-pressed", String(on));
  b.querySelector(".ht-text").textContent = on ? "Ground truth shown" : "Ground truth hidden";
}
function initChrome() {
  document.getElementById("hidden-toggle").addEventListener("click", () => setHidden(!S.showHidden));
  for (const b of document.querySelectorAll("#raw-seg button")) b.addEventListener("click", () => setRawAll(b.dataset.v === "source"));
  const help = document.getElementById("help-btn"), pop = document.getElementById("help-pop");
  help.addEventListener("click", (ev) => { ev.stopPropagation(); pop.hidden = !pop.hidden; help.setAttribute("aria-expanded", String(!pop.hidden)); });
  document.addEventListener("click", (ev) => { if (!pop.hidden && !pop.contains(ev.target)) { pop.hidden = true; help.setAttribute("aria-expanded", "false"); } });
  document.addEventListener("keydown", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tag = (ev.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag)) return;
    const route = parseRoute();
    if (ev.key === "h" || ev.key === "H") { if (S.hiddenServed) setHidden(!S.showHidden); return; }
    if (ev.key === "r" || ev.key === "R") { setRawAll(!S.rawAll); return; }
    if (route.view !== "ep") return;
    if (ev.key === "ArrowRight" && S.step < ep().calls.length) { ev.preventDefault(); go(epHash(S.tab, S.step + 1)); }
    else if (ev.key === "ArrowLeft" && S.step > 1) { ev.preventDefault(); go(epHash(S.tab, S.step - 1)); }
    else if (/^[1-4]$/.test(ev.key)) go(epHash(TABS.filter(([k]) => k !== "reasoning")[+ev.key - 1][0]));
  });
  window.addEventListener("hashchange", render);
}
async function boot() {
  initChrome();
  try {
    S.data = await (await getText("data/data.json")).json();
    await loadFiles([...S.data.episode.files.ep.map((f) => `ep/${f.name}`), ...S.data.episode.files.run.map((f) => `run/${f.name}`)]);
    await loadLevels("levels", S.data.episode.calls.filter((c) => c.levels === "recorded").map((c) => c.n), S.levels);
  } catch (err) {
    document.getElementById("app").replaceChildren(h("p", {}, `The data of this episode could not be read: ${err}. Start the explorer with python -m CausalDiagnosis episode_viewer explorer serve --runs <folder>.`));
    return;
  }
  prepare();
  S.prepared = true;
  try { if (localStorage.getItem("viewer.rawAll") === "1") S.rawAll = true; } catch (err) { /* no storage */ }
  syncRawSeg();
  try {
    S.hiddenServed = (await (await getText("hidden-status.json")).json()).served === true;
  } catch (err) { S.hiddenServed = false; }
  if (!S.hiddenServed) {
    const b = document.getElementById("hidden-toggle");
    b.disabled = true;
    b.classList.add("locked");
    b.title = "The ground truth (the real fault and the score) is shown only when the viewer is started with the hidden option on your own computer.";
    b.querySelector(".ht-text").textContent = "Ground truth hidden";
  }
  document.getElementById("foot").textContent = `Data read on ${S.data.generated} from ${ep().run}/${ep().id}. The ground truth loads only when it is switched on.`;
  render();
  let resizeTimer = null;
  addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { const ov = document.getElementById("ovb-host"); if (ov) mountOverview(ov); }, 150); });
  new MutationObserver(glossBands).observe(document.getElementById("app"), { childList: true, subtree: true });
  glossBands();
}
// A reply may say "healthy band" where this page says "normal range". The records are shown as they are, so the
// line naming the two as one thing sits right after the paragraph, item or table that quotes the phrase, or at the end
// of a raw record whose source lines cannot take a paragraph between them.
const BAND_PHRASE = /healthy band/i;
function bandAnchor(el) {
  const raw = el.closest(".rawrec");
  if (raw) return { node: raw, inside: true };
  // an item or a grid cell takes the line inside it; a paragraph, table, list of terms or source block takes it after
  const item = el.closest("li"), block = el.closest("p, table, dl, pre");
  if (block && (!item || item.contains(block))) return { node: block, inside: false };
  return { node: item || el, inside: true };
}
function hasGloss(a) {
  return a.inside ? [...a.node.children].some((c) => c.classList.contains("band-gloss"))
    : !!(a.node.nextElementSibling && a.node.nextElementSibling.classList.contains("band-gloss"));
}
function glossBands() {
  const walk = document.createTreeWalker(document.getElementById("app"), NodeFilter.SHOW_TEXT);
  const anchors = new Map();
  for (let n = walk.nextNode(); n; n = walk.nextNode()) {
    const el = n.parentElement;
    if (!el || el.closest(".band-gloss, svg") || !BAND_PHRASE.test(n.data)) continue;
    const a = bandAnchor(el);
    anchors.set(a.node, a);
  }
  for (const a of anchors.values()) {
    if (hasGloss(a)) continue;
    const note = h("p", { class: "note band-gloss" }, "\u201chealthy band\u201d in this record means the normal range.");
    if (a.inside) a.node.append(note); else a.node.after(note);
  }
}
boot();
