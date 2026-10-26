// Chart2Code front end. Plain JS, no build step.
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;
const words = (t) => (t.match(/\b\w\w+\b/g) || []).length;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const CHAPTERS = {
  A: "Infectious diseases", B: "Infectious diseases", C: "Neoplasms", D: "Neoplasms and blood disorders",
  E: "Endocrine and metabolic", F: "Mental and behavioural", G: "Nervous system", H: "Eye and ear",
  I: "Circulatory system", J: "Respiratory system", K: "Digestive system", L: "Skin",
  M: "Musculoskeletal", N: "Genitourinary", O: "Pregnancy and childbirth", P: "Perinatal",
  Q: "Congenital", R: "Symptoms and signs", S: "Injury", T: "Injury and poisoning",
  U: "Special purposes (e.g. COVID-19)", Z: "Health status factors",
};
const MODEL_NAMES = { combined: "Combined", tfidf: "TF-IDF", pubmedbert: "PubMedBERT" };
const SOURCE_NAMES = { typed: "Typed", sample: "Test case", upload: "Upload", scan: "Scan", dictation: "Dictation" };

const state = {
  meta: null,
  result: null,       // last /api/predict response
  analysedText: "",   // exact text that result belongs to
  model: "combined",
  selected: null,     // code whose evidence is highlighted
  source: "typed",
  sample: null,
  revealed: false,
  records: [],
  controller: null,
  swipe: false,
};

const note = $("#note");
const backdrop = $("#backdrop");

// ------------------------------------------------------------------ helpers
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = `The server returned an error (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail[0]?.msg || message;
    } catch { /* keep default message */ }
    throw new Error(message);
  }
  return res.status === 204 ? null : res.json();
}

let toastTimer;
function toast(message, { link, error = false } = {}) {
  const el = $("#toast");
  el.className = `toast${error ? " error" : ""}`;
  el.innerHTML = `<span>${esc(message)}</span>${link ? `<a href="${link.href}">${esc(link.text)}</a>` : ""}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, error ? 7000 : 4500);
}

function chapterOf(code) {
  return `Chapter ${code[0]}, ${CHAPTERS[code[0]] || "Other"}`;
}

function autoGrow() {
  note.style.height = "auto";
  note.style.height = `${note.scrollHeight}px`;
}

// ------------------------------------------------------------------ theme
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#themeToggle").setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  try { localStorage.setItem("chart2code-theme", theme); } catch { /* storage unavailable */ }
}
$("#themeToggle").addEventListener("click", () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  setTheme(current === "dark" ? "light" : "dark");
});
try {
  const saved = localStorage.getItem("chart2code-theme");
  if (saved) setTheme(saved);
} catch { /* storage unavailable */ }

// ------------------------------------------------------------------ routing
function route() {
  const view = (location.hash || "#code").slice(1);
  const known = ["code", "records", "codes", "about"].includes(view) ? view : "code";
  for (const v of $$(".view")) v.hidden = v.id !== `view-${known}`;
  $$(".nav a").forEach((a) => (a.dataset.view === known ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  if (known === "records") renderRecords();
  window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ note input
function setNote(text, source, sourceLine, sample = null) {
  note.value = text;
  state.source = source;
  state.sample = sample;
  state.revealed = false;
  $("#sourceLine").textContent = sourceLine;
  renderSampleBanner();
  autoGrow();
  analyse({ swipe: true });
}

function loadSample() {
  const samples = state.meta.samples;
  let next;
  do { next = samples[Math.floor(Math.random() * samples.length)]; } while (samples.length > 1 && next === state.sample);
  setNote(next.note, "sample", `Held-out test case ${next.id}. The models never saw this note while training.`, next);
}

function renderSampleBanner() {
  const s = state.sample;
  $("#sampleBanner").hidden = !s;
  if (!s) return;
  if (!state.revealed) {
    $("#sampleText").textContent = "The true code is hidden. Check the suggestions first, then reveal it.";
    $("#revealBtn").hidden = false;
    return;
  }
  const list = state.result?.views[state.model] || [];
  const rank = list.findIndex((x) => x.code === s.code) + 1;
  const verdict = rank === 1 ? `${MODEL_NAMES[state.model]} ranked it first.`
    : rank > 1 ? `${MODEL_NAMES[state.model]} ranked it #${rank}.`
    : `It is not in ${MODEL_NAMES[state.model]}'s top 5.`;
  $("#sampleText").innerHTML = `<span class="revealed">True code: <span class="code-ref">${esc(s.code)}</span> ${esc(s.diagnosis)}.</span> ${esc(verdict)}`;
  $("#revealBtn").hidden = true;
}

$("#revealBtn").addEventListener("click", () => { state.revealed = true; renderSampleBanner(); });
$("#sampleBtn").addEventListener("click", loadSample);
$("#clearBtn").addEventListener("click", () => {
  setNote("", "typed", "Type or paste a note to begin.");
  note.focus();
});

let typingTimer;
note.addEventListener("input", () => {
  if (!note.value.trim()) { state.sample = null; renderSampleBanner(); }
  if (state.source !== "sample" && state.source !== "typed" && !note.value.trim()) state.source = "typed";
  autoGrow();
  renderBackdrop();               // marks vanish while the text changes
  clearTimeout(typingTimer);
  typingTimer = setTimeout(() => analyse(), 650);
});
window.addEventListener("resize", autoGrow);

// ------------------------------------------------------------------ analysis
async function analyse({ swipe = false } = {}) {
  const text = note.value;
  updateStats();
  if (words(text) < 5) {
    state.controller?.abort();
    state.result = null;
    renderResults();
    return;
  }
  state.controller?.abort();
  const controller = (state.controller = new AbortController());
  $("#panel").classList.add("busy");
  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    if (note.value !== text) return;   // the note changed while waiting; a newer request follows
    state.result = result;
    state.analysedText = text;
    state.selected = result.views[state.model][0].code;
    state.swipe = swipe;
    renderResults();
  } catch (err) {
    if (err.name !== "AbortError") toast(err.message, { error: true });
  } finally {
    if (state.controller === controller) $("#panel").classList.remove("busy");
  }
}

function updateStats() {
  const n = words(note.value);
  const timing = state.result && state.analysedText === note.value ? `, analysed in ${state.result.elapsed_ms} ms` : "";
  $("#noteStats").textContent = n ? `${n} words${timing}` : "";
}

function renderBackdrop() {
  const text = note.value;
  const r = state.result;
  const evidence = r && state.analysedText === text && $("#showEvidence").checked ? r.evidence[state.selected] : null;
  backdrop.classList.remove("swipe");
  if (!evidence) {
    backdrop.textContent = `${text}\n`;
    return;
  }
  const max = Math.max(1e-9, ...evidence.spans.map((s) => Math.abs(s[2])));
  let html = "";
  let pos = 0;
  let i = 0;
  for (const [start, end, score] of evidence.spans) {
    const strength = Math.abs(score) / max;
    if (strength < 0.12) continue;  // keep the page readable: only clear evidence is marked
    html += esc(text.slice(pos, start));
    html += `<mark class="${score > 0 ? "for" : "against"}" style="--a:${(0.22 + 0.68 * strength).toFixed(2)};--i:${i++}">${esc(text.slice(start, end))}</mark>`;
    pos = end;
  }
  backdrop.innerHTML = `${html}${esc(text.slice(pos))}\n`;
  if (state.swipe && !reducedMotion) {
    void backdrop.offsetWidth;   // restart the animation
    backdrop.classList.add("swipe");
  }
  state.swipe = false;
}

function renderResults() {
  const r = state.result;
  $("#emptyPanel").hidden = !!r;
  $("#results").hidden = !r;
  updateStats();
  renderBackdrop();
  renderSampleBanner();
  $("#legendCode").textContent = r ? state.selected : "";
  if (!r) return;

  // agreement banner
  const a = r.agreement;
  const agreement = $("#agreement");
  agreement.className = `agreement ${a.agree ? "agree" : "disagree"}`;
  agreement.innerHTML = a.agree
    ? `<strong>Both models agree on ${esc(a.tfidf_top)}</strong><span>On unseen test notes, the models were right ${pct(a.historical_accuracy)} of the time when they agreed.</span>`
    : `<strong>The models disagree, so review carefully</strong><span>TF-IDF picks ${esc(a.tfidf_top)}, PubMedBERT picks ${esc(a.pubmedbert_top)}. On unseen test notes, the combined pick was right ${pct(a.historical_accuracy)} of the time in this situation.</span>`;

  // ranked suggestions
  const list = r.views[state.model];
  const label = {
    combined: (s) => [`Consensus ${pct(s.score)}`, "100% means both models rank this code first"],
    tfidf: (s) => [`Relative score ${pct(s.score)}`, "Compared with the TF-IDF model's top suggestion"],
    pubmedbert: (s) => [`Confidence ${pct(s.score)}`, "PubMedBERT model probability"],
  }[state.model];
  $("#suggestions").innerHTML = list.map((s, i) => {
    const [meterText, meterTitle] = label(s);
    return `<li class="sugg${s.code === state.selected ? " selected" : ""}" data-code="${esc(s.code)}">
      <button type="button" class="sugg-main" aria-pressed="${s.code === state.selected}" aria-label="Show evidence for ${esc(s.code)}">
        <span class="rank">${i + 1}</span>
        <span class="code">${esc(s.code)}</span>
        <span class="dx">${esc(s.diagnosis)}</span>
        <span class="chapter">${esc(chapterOf(s.code))}</span>
        <span class="meter-row" title="${esc(meterTitle)}"><span class="meter"><span style="width:0" data-w="${Math.max(2, s.score * 100).toFixed(1)}%"></span></span><span class="meter-label">${meterText}</span></span>
        <span class="votes">
          <span class="vote tfidf${s.rank_tfidf === 1 ? " first" : ""}">TF-IDF #${s.rank_tfidf}</span>
          <span class="vote pubmed${s.rank_pubmedbert === 1 ? " first" : ""}">PubMedBERT #${s.rank_pubmedbert}, ${pct(s.prob_pubmedbert)}</span>
        </span>
      </button>
      <button type="button" class="accept" data-accept="${esc(s.code)}">Accept</button>
    </li>`;
  }).join("");
  requestAnimationFrame(() => $$(".meter span[data-w]").forEach((el) => { el.style.width = el.dataset.w; }));

  renderEvidencePanel();

  // similar training cases
  $("#similar").innerHTML = r.similar.map((c) => `<li><details>
      <summary><span class="code-ref">${esc(c.code)}</span><span>${esc(c.diagnosis)}</span><span class="sim">${pct(c.similarity)} similar</span></summary>
      <p>${esc(c.note.slice(0, 600))}${c.note.length > 600 ? "..." : ""}</p>
    </details></li>`).join("");
}

function renderEvidencePanel() {
  const ev = state.result.evidence[state.selected];
  $("#termsCode").textContent = state.selected;
  $("#legendCode").textContent = state.selected;
  $("#terms").innerHTML = ev.top_terms.length
    ? ev.top_terms.map(([t]) => `<span class="chip">${esc(t)}</span>`).join("")
    : `<span class="fine">No words in this note point towards ${esc(state.selected)}.</span>`;
}

function selectCode(code) {
  state.selected = code;
  state.swipe = true;
  for (const li of $$(".sugg")) {
    const on = li.dataset.code === code;
    li.classList.toggle("selected", on);
    $(".sugg-main", li).setAttribute("aria-pressed", on);
  }
  renderEvidencePanel();
  renderBackdrop();
}

$("#suggestions").addEventListener("click", (e) => {
  const accept = e.target.closest("[data-accept]");
  if (accept) { saveRecord(accept.dataset.accept); return; }
  const card = e.target.closest(".sugg");
  if (card) selectCode(card.dataset.code);
});

$$(".segmented button").forEach((btn) => btn.addEventListener("click", () => {
  state.model = btn.dataset.model;
  $$(".segmented button").forEach((b) => b.setAttribute("aria-checked", b === btn));
  if (state.result) {
    state.selected = state.result.views[state.model][0].code;
    state.swipe = true;
    renderResults();
  }
}));

$("#showEvidence").addEventListener("change", renderBackdrop);

// ------------------------------------------------------------------ records
async function saveRecord(finalCode, comment = null) {
  const r = state.result;
  if (!r || state.analysedText !== note.value) { toast("Wait for the suggestions to update, then try again.", { error: true }); return; }
  try {
    const rec = await api("/api/records", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: state.source,
        note: state.analysedText,
        model: state.model,
        suggested_code: r.views[state.model][0].code,
        final_code: finalCode,
        models_agreed: r.agreement.agree,
        sample_id: state.sample?.id ?? null,
        comment,
      }),
    });
    state.records.unshift(rec);
    updateBadge();
    let message = `Saved ${finalCode} to coded records.`;
    if (rec.true_code) message += rec.true_code === finalCode ? " It matches the true code." : ` The true code was ${rec.true_code}.`;
    toast(message, { link: { href: "#records", text: "View records" } });
  } catch (err) {
    toast(err.message, { error: true });
  }
}

$("#assignForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const raw = $("#assignInput").value.trim();
  const code = raw.split(/\s+/)[0].toUpperCase();
  const known = state.meta.all_codes.find((c) => c.code === code);
  if (!known) { toast(`"${raw}" is not a code in this dataset. Pick one from the list.`, { error: true }); return; }
  saveRecord(known.code, "Assigned manually");
  $("#assignInput").value = "";
});

function updateBadge() {
  $("#recordCount").textContent = state.records.length;
}

function renderRecords() {
  const rows = state.records;
  $("#recordsEmpty").hidden = rows.length > 0;
  $("#recordsTableWrap").hidden = rows.length === 0;
  $("#exportBtn").hidden = rows.length === 0;
  const withTruth = rows.filter((r) => r.true_code);
  const stats = [
    [rows.length, "notes coded"],
    [rows.length ? pct(rows.filter((r) => r.accepted).length / rows.length) : "-", "top suggestion accepted"],
    [rows.filter((r) => !r.models_agreed).length, "flagged because the models disagreed"],
    [withTruth.length ? `${withTruth.filter((r) => r.final_code === r.true_code).length}/${withTruth.length}` : "-", "test cases coded correctly"],
  ];
  $("#recordStats").innerHTML = stats.map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  $("#recordsBody").innerHTML = rows.map((r) => {
    const when = new Date(r.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    const pills = [
      r.accepted ? `<span class="pill ok">Top suggestion accepted</span>` : `<span class="pill neutral">Coder override</span>`,
      r.models_agreed ? "" : `<span class="pill warn">Models disagreed</span>`,
      r.true_code ? (r.final_code === r.true_code ? `<span class="pill ok">Matches true code</span>` : `<span class="pill warn">True code ${esc(r.true_code)}</span>`) : "",
    ].join("");
    return `<tr>
      <td>${esc(when)}</td>
      <td>${esc(SOURCE_NAMES[r.source] || r.source)}${r.sample_id ? `<br><span class="fine">${esc(r.sample_id)}</span>` : ""}</td>
      <td class="note-cell">${esc(r.note.slice(0, 160))}${r.note.length > 160 ? "..." : ""}</td>
      <td><span class="code">${esc(r.suggested_code)}</span><span class="fine">${esc(MODEL_NAMES[r.model] || r.model)}</span></td>
      <td><span class="code">${esc(r.final_code)}</span><span class="dx-small">${esc(r.final_diagnosis)}</span></td>
      <td>${pills}</td>
      <td><button type="button" class="delete" data-delete="${r.id}" aria-label="Delete record ${r.id}">Delete</button></td>
    </tr>`;
  }).join("");
}

$("#recordsBody").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-delete]");
  if (!btn) return;
  const id = Number(btn.dataset.delete);
  try {
    await api(`/api/records/${id}`, { method: "DELETE" });
    state.records = state.records.filter((r) => r.id !== id);
    updateBadge();
    renderRecords();
    toast("Record deleted.");
  } catch (err) {
    toast(err.message, { error: true });
  }
});

// ------------------------------------------------------------------ supported codes + metrics
function renderCodes() {
  const query = $("#codeSearch").value.trim().toLowerCase();
  const codes = state.meta.supported_codes.filter((c) => !query
    || c.code.toLowerCase().includes(query) || c.diagnosis.toLowerCase().includes(query)
    || c.top_words.some((w) => w.includes(query)));
  const groups = {};
  for (const c of codes) (groups[c.code[0]] ||= []).push(c);
  $("#codeGroups").innerHTML = Object.keys(groups).sort().map((letter) => `
    <section class="chapter-group">
      <h2><span class="letter">${letter}</span>${esc(CHAPTERS[letter] || "Other")}</h2>
      <div class="code-grid">${groups[letter].map((c) => `
        <article class="code-card">
          <div class="head"><span class="code">${esc(c.code)}</span><span class="count">${c.train_cases} training notes</span></div>
          <p class="dx">${esc(c.diagnosis)}</p>
          <div class="chips">${c.top_words.map((w) => `<span class="chip">${esc(w)}</span>`).join("")}</div>
        </article>`).join("")}
      </div>
    </section>`).join("") || `<p class="no-match">No supported code matches "${esc(query)}".</p>`;
}
$("#codeSearch").addEventListener("input", renderCodes);

function renderMetrics() {
  const m = state.meta.metrics;
  $("#codeCount").textContent = m.classes;
  $("#trainNotes").textContent = m.train_notes;
  $("#testNotes").textContent = m.test_notes;
  const names = Object.keys(m.test);
  const best = (key, get) => Math.max(...names.map((n) => get(n)));
  const cell = (v, top) => `<td class="${Math.abs(v - top) < 1e-9 ? "best" : ""}">${(v * 100).toFixed(1)}%</td>`;
  const bAcc = best("acc", (n) => m.test[n].accuracy), b3 = best("t3", (n) => m.test[n].top3);
  const b5 = best("t5", (n) => m.test[n].top5), bCv = best("cv", (n) => m.cv[n].mean);
  $("#metricsTable tbody").innerHTML = names.map((n) => `<tr><td>${esc(n)}</td>
    ${cell(m.test[n].accuracy, bAcc)}${cell(m.test[n].top3, b3)}${cell(m.test[n].top5, b5)}
    <td class="${Math.abs(m.cv[n].mean - bCv) < 1e-9 ? "best" : ""}">${(m.cv[n].mean * 100).toFixed(1)}% ± ${(m.cv[n].std * 100).toFixed(1)}</td></tr>`).join("");
  const a = m.agreement;
  $("#agreeCallout").textContent = `The two models agreed on ${pct(a.agree_share)} of test notes and were right ${pct(a.accuracy_when_agree)} of the time when they did. When they disagreed, the combined pick was right ${pct(a.accuracy_when_disagree_combined)} of the time, which is why disagreements are flagged for review.`;
}

// ------------------------------------------------------------------ files: upload and drag-and-drop
async function handleFile(file) {
  if (!file) return;
  if (file.type.startsWith("image/")) {
    openScan();
    await showImageFile(file);
    return;
  }
  try {
    const form = new FormData();
    form.append("file", file);
    const { text } = await api("/api/extract", { method: "POST", body: form });
    setNote(text, "upload", `Read from ${file.name}.`);
  } catch (err) {
    toast(err.message, { error: true });
  }
}
$("#fileInput").addEventListener("change", (e) => { handleFile(e.target.files[0]); e.target.value = ""; });

const sheet = $("#sheet");
["dragenter", "dragover"].forEach((t) => sheet.addEventListener(t, (e) => { e.preventDefault(); sheet.classList.add("dragging"); }));
["dragleave", "drop"].forEach((t) => sheet.addEventListener(t, (e) => { e.preventDefault(); if (t === "drop" || !sheet.contains(e.relatedTarget)) sheet.classList.remove("dragging"); }));
sheet.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

// ------------------------------------------------------------------ scanning (camera + OCR)
const dialog = $("#scanDialog");
const video = $("#camVideo");
const canvas = $("#camCanvas");
let stream = null;
let ocrWorker = null;
let scanSource = "scan";

function openScan() {
  $("#ocrText").value = "";
  $("#ocrUse").disabled = true;
  $("#ocrProgress").hidden = true;
  canvas.hidden = true;
  $("#viewfinderHint").hidden = false;
  if (!dialog.open) dialog.showModal();
}
function stopCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.classList.remove("live");
  $("#camCapture").disabled = true;
}
$("#scanBtn").addEventListener("click", openScan);
$("#scanClose").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", stopCamera);

$("#camStart").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast("This browser cannot open the camera here. Use Choose photo instead.", { error: true });
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } } });
    video.srcObject = stream;
    await video.play();
    video.classList.add("live");
    canvas.hidden = true;
    $("#viewfinderHint").hidden = true;
    $("#camCapture").disabled = false;
  } catch {
    toast("Camera access was blocked. Allow the camera in your browser settings, or use Choose photo.", { error: true });
  }
});

$("#camCapture").addEventListener("click", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  stopCamera();
  canvas.hidden = false;
  scanSource = "scan";
  runOcr(canvas);
});

async function showImageFile(file) {
  const bitmap = await createImageBitmap(file);
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  stopCamera();
  canvas.hidden = false;
  $("#viewfinderHint").hidden = true;
  scanSource = "scan";
  runOcr(canvas);
}
$("#photoInput").addEventListener("change", (e) => { showImageFile(e.target.files[0]); e.target.value = ""; });
$("#samplePhoto").addEventListener("click", async () => {
  const blob = await (await fetch("/static/demo/demo_scan.jpg")).blob();
  showImageFile(blob);
});

const OCR_STEPS = {
  "loading tesseract core": "Loading the text recogniser",
  "initializing tesseract": "Starting the text recogniser",
  "loading language traineddata": "Loading the English model",
  "initializing api": "Getting ready",
  "recognizing text": "Reading the page",
};
function onOcrProgress(m) {
  $("#ocrStatus").textContent = `${OCR_STEPS[m.status] || "Working"} ${m.progress ? pct(m.progress) : ""}`.trim();
  $("#ocrBar").style.width = pct(m.status === "recognizing text" ? m.progress : (m.progress || 0) * 0.3);
}

async function runOcr(image) {
  $("#ocrProgress").hidden = false;
  $("#ocrUse").disabled = true;
  $("#ocrText").value = "";
  try {
    ocrWorker ||= await Tesseract.createWorker("eng", 1, {
      workerPath: "/static/vendor/tesseract/worker.min.js",
      corePath: "/static/vendor/tesseract/core",
      langPath: "/static/vendor/tesseract/lang",
      logger: onOcrProgress,
    });
    const { data } = await ocrWorker.recognize(image);
    const text = data.text
      .replace(/-\n(?=\w)/g, "")        // re-join words hyphenated across lines
      .replace(/([^\n])\n(?!\n)/g, "$1 ")  // join wrapped lines; keep paragraph breaks
      .trim();
    $("#ocrText").value = text;
    $("#ocrUse").disabled = !text;
    $("#ocrStatus").textContent = text ? `Read ${words(text)} words (OCR confidence ${Math.round(data.confidence)}%). Check the text, then use it.` : "No text found. Try a sharper, well-lit photo.";
    $("#ocrBar").style.width = "100%";
  } catch (err) {
    $("#ocrStatus").textContent = "Text recognition failed. Try another photo.";
    console.error(err);
  }
}
$("#ocrText").addEventListener("input", () => { $("#ocrUse").disabled = !$("#ocrText").value.trim(); });
$("#ocrUse").addEventListener("click", () => {
  dialog.close();
  setNote($("#ocrText").value.trim(), scanSource, "Read from a scanned page with OCR. Check it for misread words.");
});

// ------------------------------------------------------------------ dictation (Web Speech API, where available)
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (Recognition) {
  const btn = $("#dictateBtn");
  const live = $("#dictationLive");
  const recogniser = new Recognition();
  recogniser.lang = "en-US";
  recogniser.continuous = true;
  recogniser.interimResults = true;
  let listening = false;
  btn.hidden = false;
  recogniser.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const phrase = e.results[i][0].transcript.trim();
      if (e.results[i].isFinal) {
        const sep = note.value && !/\s$/.test(note.value) ? " " : "";
        note.value += `${sep}${phrase.charAt(0).toUpperCase()}${phrase.slice(1)}.`;
        note.dispatchEvent(new Event("input"));
      } else interim += `${phrase} `;
    }
    live.textContent = interim;
    live.hidden = !interim;
  };
  recogniser.onend = () => {
    listening = false;
    btn.classList.remove("listening");
    $("span", btn).textContent = "Dictate";
    live.hidden = true;
  };
  recogniser.onerror = (e) => {
    if (e.error !== "no-speech" && e.error !== "aborted") toast("Dictation stopped: the browser could not use the microphone.", { error: true });
  };
  btn.addEventListener("click", () => {
    if (listening) { recogniser.stop(); return; }
    if (!note.value.trim() || state.source === "sample") {
      setNote("", "dictation", "Dictated with your browser's speech recognition.");
    } else {
      state.source = "dictation";
    }
    recogniser.start();
    listening = true;
    btn.classList.add("listening");
    $("span", btn).textContent = "Stop dictating";
  });
}

// ------------------------------------------------------------------ start
async function init() {
  try {
    const [meta, records] = await Promise.all([api("/api/meta"), api("/api/records")]);
    state.meta = meta;
    state.records = records;
  } catch (err) {
    toast("Could not reach the Chart2Code server. Start it with ./run.sh and reload this page.", { error: true });
    return;
  }
  $("#allCodes").innerHTML = state.meta.all_codes.map((c) => `<option value="${esc(c.code)}">${esc(c.diagnosis)}</option>`).join("");
  updateBadge();
  renderCodes();
  renderMetrics();
  route();
  // open on a real, unseen case so the page shows what it does; ?case=T073 picks a specific one
  const wanted = state.meta.samples.find((s) => s.id === new URLSearchParams(location.search).get("case"));
  if (wanted) setNote(wanted.note, "sample", `Held-out test case ${wanted.id}. The models never saw this note while training.`, wanted);
  else loadSample();
}
init();
