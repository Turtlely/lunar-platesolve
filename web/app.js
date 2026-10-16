// Lunar Plate Solver website.  Plain ES module, no build step.
// The API returns the solved camera (centre C, world->camera rotation Rwc, focal length, principal
// point); every pixel <-> latitude/longitude conversion below uses it, so the page needs no
// further requests for hover readouts or overlays.

const $ = (s) => document.querySelector(s);
const MOON_R = 1737.4;

const EXAMPLES = [
  { file: "AS16-M-1657", title: "Apollo 16" },
  { file: "AS16-M-1347", title: "Apollo 16, oblique" },
  { file: "AS17-M-0748", title: "Apollo 17, high sun" },
  { file: "wac_view", title: "Simulated orbiter" },
];

// ------------------------------------------------------------------ API
function apiBase() {
  const q = new URLSearchParams(location.search).get("api");
  const base = q ?? (window.LUNAR_CONFIG && window.LUNAR_CONFIG.api) ?? "";
  return base.replace(/\/$/, "");
}

const state = {
  file: null, fileUrl: null, image: null, result: null, controller: null, lastOpts: null,
  layers: { matched: true, catalog: false, dets: false, grid: true, names: true },
  view: { s: 1, tx: 0, ty: 0 }, highlight: null, gridCache: null,
};

// ------------------------------------------------------------------ human check (Cloudflare Turnstile)
// When the server has a Turnstile secret, every solve needs a fresh one-use token.  The widget is
// invisible unless Cloudflare wants a click; it then shows in the "Solving" view.
const human = { required: false, id: null, ready: null, pending: null };

function loadTurnstile() {
  const key = window.LUNAR_CONFIG && window.LUNAR_CONFIG.turnstileSiteKey;
  if (!key || human.ready) return human.ready;
  human.ready = new Promise((resolve, reject) => {
    window.onTurnstileLoad = () => {
      human.id = window.turnstile.render("#human", {
        sitekey: key, execution: "execute", appearance: "interaction-only",
        callback: (t) => { human.pending?.resolve(t); human.pending = null; },
        "error-callback": () => { human.pending?.reject(new Error("The human check failed. Reload the page and try again.")); human.pending = null; },
      });
      resolve();
    };
    const s = document.createElement("script");
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit&onload=onTurnstileLoad";
    s.async = true;
    s.onerror = () => reject(new Error("The human check could not load. Check the connection and reload."));
    document.head.appendChild(s);
  });
  return human.ready;
}

async function humanToken() {
  if (!human.required) return "";
  if (!loadTurnstile()) throw new Error("This site has no human-check key. The solver cannot accept requests.");
  await human.ready;
  return new Promise((resolve, reject) => {
    human.pending = { resolve, reject };
    window.turnstile.reset(human.id);
    window.turnstile.execute(human.id);
    setTimeout(() => { if (human.pending) { human.pending = null; reject(new Error("The human check timed out. Try again.")); } }, 60000);
  });
}

async function checkHealth() {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 8000);
    const r = await fetch(apiBase() + "/api/health", { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) throw new Error(r.status);
    const j = await r.json();
    $("#offline").hidden = true;
    human.required = !!j.human_check;
    if (human.required) loadTurnstile()?.catch(() => {});
    $("#refine-row").hidden = !j.refine;                 // high precision needs the WAC map on the server
    if (!j.refine) $("#opt-refine").checked = false;
  } catch (e) {
    $("#offline").hidden = false;
  }
}

// ------------------------------------------------------------------ views
function show(id) {
  for (const v of document.querySelectorAll(".view")) v.hidden = v.id !== id;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ------------------------------------------------------------------ start view
function initStart() {
  const drop = $("#drop"), input = $("#file");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { if (input.files[0]) startSolve(input.files[0]); input.value = ""; });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) startSolve(f); });
  // drops anywhere on the start page
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    if (!$("#view-start").hidden && e.dataTransfer.files[0]) startSolve(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    if ($("#view-start").hidden) return;
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) startSolve(item.getAsFile());
  });

  const grid = $("#examples");
  for (const ex of EXAMPLES) {
    const b = document.createElement("button");
    b.className = "example";
    b.innerHTML = `<img src="examples/${ex.file}_thumb.jpg" alt="" loading="lazy"><span>${ex.title}</span>`;
    b.addEventListener("click", async () => {
      const r = await fetch(`examples/${ex.file}.png`);
      const blob = await r.blob();
      startSolve(new File([blob], ex.file + ".png", { type: "image/png" }));
    });
    grid.appendChild(b);
  }

  const tr = $("#opt-time"), tv = $("#opt-time-v");
  tr.addEventListener("input", () => (tv.textContent = tr.value + " s"));
}

function options() {
  return {
    fov_deg: parseFloat($("#opt-fov").value) || 0,
    parity: $("#opt-parity").value,
    max_time: parseFloat($("#opt-time").value) || 30,
    refine: $("#opt-refine").checked,
  };
}

// ------------------------------------------------------------------ solving
let timer = null;

async function startSolve(file, opts = null) {
  if (!file.type.startsWith("image/") && !/\.(tif|tiff)$/i.test(file.name)) {
    alert("Please choose an image file (PNG, JPEG or TIFF).");
    return;
  }
  if (file.size > 30 * 1024 * 1024) { alert("The image is larger than 30 MB."); return; }
  opts = opts || options();
  state.file = file;
  state.lastOpts = opts;
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.fileUrl = URL.createObjectURL(file);
  $("#work-img").src = state.fileUrl;
  show("view-work");
  const t0 = performance.now();
  clearInterval(timer);
  timer = setInterval(() => { $("#elapsed").textContent = ((performance.now() - t0) / 1000).toFixed(1); }, 100);

  const fd = new FormData();
  fd.append("image", file);
  for (const [k, v] of Object.entries(opts)) fd.append(k, String(v));
  state.controller = new AbortController();
  const imgPromise = loadImage(state.fileUrl);
  try {
    fd.append("turnstile", await humanToken());
    const r = await fetch(apiBase() + "/api/solve", { method: "POST", body: fd, signal: state.controller.signal });
    if (!r.ok) {
      let msg = `The server answered ${r.status}.`;
      try { msg = (await r.json()).detail || msg; } catch (_) { /* not JSON */ }
      throw new Error(msg);
    }
    const res = await r.json();
    state.image = await imgPromise;
    clearInterval(timer);
    state.result = res;
    if (res.ok) showResult(res); else showFail(res);
  } catch (e) {
    clearInterval(timer);
    if (e.name === "AbortError") { show("view-start"); return; }
    state.image = await imgPromise.catch(() => null);
    showFail(null, e.message.includes("fetch") || e.message.includes("NetworkError")
      ? "The solver server cannot be reached. Try again in a minute."
      : e.message);
  }
}

function loadImage(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url; });
}

// ------------------------------------------------------------------ camera model
function makeCamera(cam, geo = null) {
  const { C, Rwc, f, cx, cy, width, mirrored } = cam;
  function ray(u, v) {
    const uc = mirrored ? width - 1 - u : u;
    const a = (uc - cx) / f, b = (v - cy) / f;
    // world direction = Rwc^T [a, b, 1]
    const d = [0, 1, 2].map((i) => Rwc[0][i] * a + Rwc[1][i] * b + Rwc[2][i]);
    const n = Math.hypot(d[0], d[1], d[2]);
    return d.map((x) => x / n);
  }
  function hit(u, v) {
    const d = ray(u, v);
    const b = d[0] * C[0] + d[1] * C[1] + d[2] * C[2];
    const c = C[0] * C[0] + C[1] * C[1] + C[2] * C[2] - MOON_R * MOON_R;
    const disc = b * b - c;
    if (disc < 0) return null;
    const t = -b - Math.sqrt(disc);
    if (t <= 0) return null;
    return [C[0] + t * d[0], C[1] + t * d[1], C[2] + t * d[2]];
  }
  function sphereLatLon(u, v) {
    const p = hit(u, v);
    if (!p) return null;
    const r = Math.hypot(p[0], p[1], p[2]);
    return { lat: Math.asin(p[2] / r) * 180 / Math.PI, lon: ((Math.atan2(p[1], p[0]) * 180 / Math.PI) + 360) % 360 };
  }
  // On the LOLA terrain: bilinear interpolation of the server's grid.  Next to the limb (a grid
  // corner misses the Moon) fall back to the mean sphere.
  function latlon(u, v) {
    if (!geo) return sphereLatLon(u, v);
    const H = cam.height;
    const fx = Math.min(Math.max(u / (width - 1) * geo.nx, 0), geo.nx - 1e-9);
    const fy = Math.min(Math.max(v / (H - 1) * geo.ny, 0), geo.ny - 1e-9);
    const i = Math.floor(fx), j = Math.floor(fy), a = fx - i, b = fy - j;
    const k = [j * (geo.nx + 1) + i, j * (geo.nx + 1) + i + 1, (j + 1) * (geo.nx + 1) + i, (j + 1) * (geo.nx + 1) + i + 1];
    const la = k.map((q) => geo.lat[q]), lo = k.map((q) => geo.lon[q]);
    if (la.some((x) => x === null) || lo.some((x) => x === null)) return sphereLatLon(u, v);
    const lo0 = lo[0];
    const lu = lo.map((x) => lo0 + ((x - lo0 + 540) % 360 - 180));         // unwrap across 0/360
    const w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b];
    const lat = w.reduce((t, x, q) => t + x * la[q], 0);
    const lon = w.reduce((t, x, q) => t + x * lu[q], 0);
    return { lat, lon: (lon % 360 + 360) % 360 };
  }
  return { ray, hit, latlon };
}

const fmtLat = (x, d = 4) => `${Math.abs(x).toFixed(d)}° ${x >= 0 ? "N" : "S"}`;
const fmtLon = (x, d = 4) => { const l = ((x + 180) % 360 + 360) % 360 - 180; return `${Math.abs(l).toFixed(d)}° ${l >= 0 ? "E" : "W"}`; };
const fmtKm = (km) => (km >= 10 ? km.toFixed(1) : km >= 1 ? km.toFixed(2) : km.toFixed(3)) + " km";
const fmtDist = (km) => (km < 1 ? Math.round(km * 1000) + " m" : fmtKm(km));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ------------------------------------------------------------------ result view
function showResult(res) {
  const sol = res.solution;
  state.cam = makeCamera(sol.camera, sol.geo);
  state.highlight = null;
  state.gridCache = null;
  show("view-result");
  renderSummary(res);
  renderNames(sol.features || []);
  setupViewer();
  fitView();
  draw();
  initGlobe(res);
}

function renderSummary(res) {
  const s = res.solution;
  const far = s.method === "disk";
  const sig = s.refined ? s.refined.sigma_altitude_m / 1000 : s.altitude_sigma_km;
  const num = (x) => x !== null && x !== undefined && Number.isFinite(x);
  let alt;
  if (far) alt = s.distance_measured ? `${Math.round(s.altitude_km + MOON_R).toLocaleString("en-US")} km from the centre` : "not measured";
  else alt = `${fmtKm(s.altitude_km)}${num(sig) ? ` ± ${fmtDist(sig)}` : ""}`;
  const rows = [
    [far ? "Seen from lat." : "Latitude", fmtLat(s.camera_lat), "main"],
    [far ? "Seen from lon." : "Longitude", fmtLon(s.camera_lon), "main"],
    [far ? "Distance" : "Altitude", alt, "main"],
    num(s.center_lat) ? ["Image centre", `${fmtLat(s.center_lat, 3)}, ${fmtLon(s.center_lon, 3)}`, "gap"] : null,
    num(s.gsd_m) ? ["Resolution", `${s.gsd_m < 100 ? s.gsd_m.toFixed(1) : Math.round(s.gsd_m)} m/px`] : null,
    !far && num(s.off_nadir_deg) ? ["Tilt", `${s.off_nadir_deg.toFixed(1)}°`] : null,
    num(s.heading_deg) ? ["Image up", `${s.heading_deg.toFixed(1)}° from north`] : null,
    num(s.fov_deg) ? ["Field of view", `${s.fov_deg < 2 ? s.fov_deg.toFixed(3) : s.fov_deg.toFixed(2)}°${s.focal_estimated && !far ? " (est.)" : ""}${far && !s.distance_measured ? " (assumed)" : ""}`] : null,
    ["Craters", `${s.n_matches} of ${res.detections.length} matched`],
    ["Time", `${res.time_s.toFixed(1)} s`],
  ].filter(Boolean);
  if (far) rows.push(["View", "far (disk edge used)"]);
  if (s.catalog) rows.push(["Catalog", s.catalog === "CraterNet" ? "CraterNet (own, 2.6 M craters)" : "Robbins (1.3 M craters)"]);
  $("#summary").innerHTML = rows.map(([k, v, c]) => `<tr${c ? ` class="${c}"` : ""}><th>${k}</th><td>${v}</td></tr>`).join("");
  const note = $("#note");
  const notes = [];
  if (far && !s.distance_measured) notes.push("Far view: the distance cannot be measured from the image alone. "
    + "Enter the field of view under Options to measure it. From Earth it is 356,000–407,000 km.");
  if (s.refine_rejected) notes.push(`High precision skipped: ${s.refine_rejected}.`);
  note.hidden = !notes.length;
  note.textContent = notes.join(" ");
  $("#near").textContent = "";
  const cLat = num(s.center_lat) ? s.center_lat : s.camera_lat, cLon = num(s.center_lon) ? s.center_lon : s.camera_lon;
  lookup(cLat, cLon).then((j) => {
    if (!j) return;
    $("#near").innerHTML = j.inside ? `The image centre is inside <b>${esc(j.nearest)}</b>.`
      : `The image centre is ${fmtDist(j.nearest_km)} from <b>${esc(j.nearest)}</b>.`;
  });
  const copy = $("#copy-pos");
  copy.onclick = () => {
    navigator.clipboard?.writeText(`${s.camera_lat.toFixed(5)}, ${s.camera_lon.toFixed(5)}` + (far && !s.distance_measured ? "" : `, ${s.altitude_km.toFixed(3)} km`));
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy position"), 1400);
  };
}

function renderNames(feats) {
  const ul = $("#names");
  $("#names-count").textContent = feats.length ? `(${feats.length})` : "";
  if (!feats.length) { ul.innerHTML = `<li class="muted">None large enough in this view.</li>`; return; }
  ul.innerHTML = feats.slice(0, 80).map((f, i) =>
    `<li data-i="${i}" title="${esc(f.type)}"><span>${esc(f.name)}</span><span class="s">${f.diam_km ? f.diam_km.toFixed(1) + " km" : ""}</span></li>`).join("");
  ul.querySelectorAll("li[data-i]").forEach((li) => {
    li.addEventListener("mouseenter", () => { state.highlight = +li.dataset.i; draw(); });
    li.addEventListener("mouseleave", () => { state.highlight = null; draw(); });
    li.addEventListener("click", () => {
      const f = feats[+li.dataset.i];
      zoomTo(f.u, f.v, Math.max(3, Math.min(12, 300 / Math.max(f.diam_km ? 40 : 60, 1))));
    });
  });
}

async function lookup(lat, lon) {
  try {
    const r = await fetch(`${apiBase()}/api/lookup?lat=${lat}&lon=${lon}`);
    return r.ok ? await r.json() : null;
  } catch (_) { return null; }
}

// ------------------------------------------------------------------ image viewer
let viewerReady = false;

function setupViewer() {
  if (viewerReady) return;
  viewerReady = true;
  const vw = $("#viewer"), cv = $("#canvas");
  new ResizeObserver(() => { fitView(false); draw(); }).observe(vw);
  for (const box of document.querySelectorAll("input[data-layer]")) {
    box.addEventListener("change", () => { state.layers[box.dataset.layer] = box.checked; draw(); });
  }
  let drag = null, moved = false;
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const { x, y } = local(e);
    const k = Math.exp(-e.deltaY * 0.0015);
    const v = state.view;
    const s2 = Math.min(Math.max(v.s * k, state.minScale * 0.8), 40);
    v.tx = x - (x - v.tx) * (s2 / v.s);
    v.ty = y - (y - v.ty) * (s2 / v.s);
    v.s = s2;
    draw();
  }, { passive: false });
  cv.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, tx: state.view.tx, ty: state.view.ty }; moved = false; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.hypot(dx, dy) > 3) { moved = true; vw.classList.add("dragging"); $("#popup").hidden = true; }
      state.view.tx = drag.tx + dx;
      state.view.ty = drag.ty + dy;
      draw();
    }
    readout(e);
  });
  cv.addEventListener("pointerup", (e) => {
    vw.classList.remove("dragging");
    if (drag && !moved) clickInfo(e);
    drag = null;
  });
  cv.addEventListener("pointerleave", () => { $("#readout").textContent = "Scroll to zoom, drag to pan, click for details."; });
  cv.addEventListener("dblclick", () => { fitView(); draw(); });
}

function local(e) {
  const r = $("#canvas").getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}
function toImage(x, y) { const v = state.view; return [(x - v.tx) / v.s, (y - v.ty) / v.s]; }

function fitView(reset = true) {
  const vw = $("#viewer");
  const W = state.result?.image.width || 1, H = state.result?.image.height || 1;
  const s = Math.min(vw.clientWidth / W, vw.clientHeight / H);
  state.minScale = s;
  if (reset || state.view.s < s) state.view = { s, tx: (vw.clientWidth - W * s) / 2, ty: (vw.clientHeight - H * s) / 2 };
}

function zoomTo(u, v, s) {
  const vw = $("#viewer");
  state.view = { s: state.minScale * s, tx: 0, ty: 0 };
  state.view.tx = vw.clientWidth / 2 - u * state.view.s;
  state.view.ty = vw.clientHeight / 2 - v * state.view.s;
  draw();
}

function readout(e) {
  if (!state.cam) return;
  const { x, y } = local(e);
  const [u, v] = toImage(x, y);
  const W = state.result.image.width, H = state.result.image.height;
  if (u < 0 || v < 0 || u >= W || v >= H) return;
  const ll = state.cam.latlon(u, v);
  $("#readout").innerHTML = ll
    ? `<b>${fmtLat(ll.lat)}, ${fmtLon(ll.lon)}</b>`
    : `off the Moon`;
}

async function clickInfo(e) {
  if (!state.cam) return;
  const { x, y } = local(e);
  const [u, v] = toImage(x, y);
  const ll = state.cam.latlon(u, v);
  const pop = $("#popup");
  if (!ll) { pop.hidden = true; return; }
  const vw = $("#viewer");
  pop.style.left = Math.min(x + 14, vw.clientWidth - 290) + "px";
  pop.style.top = Math.min(y + 14, vw.clientHeight - 110) + "px";
  pop.innerHTML = `<b>${fmtLat(ll.lat)}, ${fmtLon(ll.lon)}</b><br><span class="muted">Looking up…</span>`;
  pop.hidden = false;
  const j = await lookup(ll.lat, ll.lon);
  if (!j || pop.hidden) return;
  const where = j.inside ? `Inside <b>${esc(j.nearest)}</b>` : `${fmtDist(j.nearest_km)} from <b>${esc(j.nearest)}</b>`;
  pop.innerHTML = `<b>${fmtLat(ll.lat)}, ${fmtLon(ll.lon)}</b><br>Elevation ${j.elevation_km >= 0 ? "+" : ""}${j.elevation_km.toFixed(2)} km<br>${where}`;
}

// ------------------------------------------------------------------ drawing
function niceStep(span) {
  const raw = span / 5;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * p) return m * p;
  return 10 * p;
}

function buildGrid() {
  // lat/lon on a coarse pixel grid, then iso-lines by marching squares
  const W = state.result.image.width, H = state.result.image.height;
  const n = Math.max(2, Math.round(Math.max(W, H) / 160 * 20));
  const nx = Math.min(n, 140), ny = Math.max(2, Math.round(nx * H / W));
  const lat = new Float64Array((nx + 1) * (ny + 1)), lon = new Float64Array((nx + 1) * (ny + 1));
  const c = state.result.solution;
  const lon0 = c.center_lon ?? c.camera_lon;
  let lmin = 1e9, lmax = -1e9, omin = 1e9, omax = -1e9;
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
    const ll = state.cam.latlon(i * (W - 1) / nx, j * (H - 1) / ny);
    const k = j * (nx + 1) + i;
    if (!ll) { lat[k] = NaN; lon[k] = NaN; continue; }
    lat[k] = ll.lat;
    lon[k] = lon0 + ((ll.lon - lon0 + 540) % 360 - 180);
    lmin = Math.min(lmin, lat[k]); lmax = Math.max(lmax, lat[k]);
    omin = Math.min(omin, lon[k]); omax = Math.max(omax, lon[k]);
  }
  const step = niceStep(Math.max(lmax - lmin, (omax - omin) * Math.cos(((lmin + lmax) / 2) * Math.PI / 180), 1e-4));
  const lines = [];
  for (const [arr, lo, hi, kind] of [[lat, lmin, lmax, "lat"], [lon, omin, omax, "lon"]]) {
    const st = kind === "lon" ? step / Math.max(Math.cos(((lmin + lmax) / 2) * Math.PI / 180), 0.2) : step;
    const s2 = niceStep(st * 5);
    for (let L = Math.ceil(lo / s2) * s2; L <= hi; L += s2) {
      const segs = [];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const idx = [j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i];
        const vals = idx.map((q) => arr[q] - L);
        if (vals.some(Number.isNaN)) continue;
        const corners = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
        const pts = [];
        for (let e = 0; e < 4; e++) {
          const a = vals[e], b = vals[(e + 1) % 4];
          if ((a < 0) !== (b < 0)) {
            const t = a / (a - b);
            const p = corners[e], q = corners[(e + 1) % 4];
            pts.push([(p[0] + t * (q[0] - p[0])) * (W - 1) / nx, (p[1] + t * (q[1] - p[1])) * (H - 1) / ny]);
          }
        }
        if (pts.length === 2) segs.push(pts);
      }
      if (segs.length) lines.push({ kind, value: L, segs });
    }
  }
  return lines;
}

// target: optional {g, cw, ch, view} to draw somewhere else (used for the full-resolution download)
function draw(target = null) {
  const res = state.result;
  if (!res || !res.solution || !state.image) return;
  let g, cw, ch, view;
  if (target) {
    ({ g, cw, ch, view } = target);
  } else {
    const cv = $("#canvas"), vw = $("#viewer");
    const dpr = window.devicePixelRatio || 1;
    cw = vw.clientWidth; ch = vw.clientHeight; view = state.view;
    if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); }
    g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  g.fillStyle = "#000";
  g.fillRect(0, 0, cw, ch);
  const { s, tx, ty } = view;
  g.save();
  g.translate(tx, ty);
  g.scale(s, s);
  g.imageSmoothingQuality = "high";
  g.drawImage(state.image, 0, 0, res.image.width, res.image.height);
  g.restore();
  const P = (u, v) => [tx + u * s, ty + v * s];
  const sol = res.solution;
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();

  if (state.layers.grid) {
    if (!state.gridCache) state.gridCache = buildGrid();
    const gboxes = [];
    for (const ln of state.gridCache) {
      g.strokeStyle = ln.kind === "lat" ? "rgba(124,199,255,.75)" : "rgba(214,150,255,.75)";
      g.lineWidth = 1;
      g.beginPath();
      let lab = null;
      for (const [a, b] of ln.segs) {
        const p = P(...a), q = P(...b);
        g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]);
        if (!lab || (ln.kind === "lat" ? p[0] < lab[0] : p[1] > lab[1])) lab = p;
      }
      g.stroke();
      if (lab) {
        const txt = ln.kind === "lat" ? fmtLat(ln.value, decimals(ln.value)) : fmtLon(ln.value, decimals(ln.value));
        const lx = Math.max(4, Math.min(lab[0] + 3, cw - 70)), ly = Math.max(14, Math.min(lab[1] - 3, ch - 4));
        g.font = "500 11px 'IBM Plex Sans', sans-serif";
        const bx = [lx - 2, ly - 12, lx + g.measureText(txt).width + 2, ly + 3];
        if (!gboxes.some((b) => !(bx[2] < b[0] || bx[0] > b[2] || bx[3] < b[1] || bx[1] > b[3]))) {
          gboxes.push(bx);
          label(g, txt, lx, ly, ln.kind === "lat" ? "#bfe3ff" : "#ead1ff", 11);
        }
      }
    }
  }
  if (state.layers.dets) {
    g.strokeStyle = col("--dets");
    g.lineWidth = 1;
    for (const [x, y, r] of res.detections) { const p = P(x, y); g.beginPath(); g.arc(p[0], p[1], Math.max(1.5, r * s), 0, 7); g.stroke(); }
  }
  if (state.layers.catalog) {
    g.strokeStyle = "rgba(255,255,255,.9)";
    g.lineWidth = 1;
    for (const [x, y, r] of sol.matched_catalog || []) { const p = P(x, y); g.beginPath(); g.arc(p[0], p[1], Math.max(2, r * s), 0, 7); g.stroke(); }
  }
  if (state.layers.matched) {
    g.strokeStyle = col("--matched");
    g.lineWidth = 2;
    for (const i of sol.matched) { const [x, y, r] = res.detections[i] || []; if (x === undefined) continue; const p = P(x, y); g.beginPath(); g.arc(p[0], p[1], Math.max(2, r * s), 0, 7); g.stroke(); }
  }
  if (state.layers.names || state.highlight !== null) {
    const boxes = [];
    (sol.features || []).forEach((f, i) => {
      const hl = state.highlight === i;
      if (!state.layers.names && !hl) return;
      // declutter: a name appears when its feature is large enough at the current zoom
      if (!hl && f.d_px && f.d_px * s < 18) return;
      g.strokeStyle = hl ? "#ffffff" : col("--names");
      g.lineWidth = hl ? 2.5 : 1.2;
      const p = P(f.u, f.v);
      if (f.crater && f.outline && f.outline.length > 3) {
        g.beginPath();
        f.outline.forEach(([u, v], k) => { const q = P(u, v); k ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
        g.closePath();
        g.stroke();
      } else {
        g.beginPath(); g.moveTo(p[0], p[1] - 6); g.lineTo(p[0] + 6, p[1]); g.lineTo(p[0], p[1] + 6); g.lineTo(p[0] - 6, p[1]); g.closePath(); g.stroke();
      }
      g.font = `${hl ? 600 : 500} 12.5px 'IBM Plex Sans', sans-serif`;
      const w = g.measureText(f.name).width;
      const bx = [p[0] - w / 2, p[1] - 22, p[0] + w / 2, p[1] - 6];
      if (!hl && boxes.some((b) => !(bx[2] < b[0] || bx[0] > b[2] || bx[3] < b[1] || bx[1] > b[3]))) return;
      boxes.push(bx);
      label(g, f.name, bx[0], p[1] - 9, hl ? "#ffffff" : col("--names"), 12.5, hl ? 600 : 500);
    });
  }
}

function decimals(step) { const s = Math.abs(step % 1); return s === 0 ? 0 : s.toString().split(".")[1]?.length > 2 ? 3 : (s.toString().split(".")[1]?.length || 1); }

function label(g, text, x, y, color, size = 12, weight = 500) {
  g.font = `${weight} ${size}px 'IBM Plex Sans', sans-serif`;
  g.lineWidth = 3.5;
  g.strokeStyle = "rgba(0,0,0,.85)";
  g.lineJoin = "round";
  g.strokeText(text, x, y);
  g.fillStyle = color;
  g.fillText(text, x, y);
}

// ------------------------------------------------------------------ failure view
function showFail(res, message) {
  show("view-fail");
  $("#fail-title").textContent = res ? "No match" : "Something went wrong";
  $("#fail-msg").textContent = message || (res
    ? `The solver found ${res.detections.length} craters but could not match their pattern to the catalog.`
    : "");
  $("#fail-tips").hidden = !res;
  const cv = $("#fail-canvas");
  const im = state.image;
  $("#fail-count").textContent = res ? `Detected craters (${res.detections.length})` : "";
  if (!im) { cv.hidden = true; return; }
  cv.hidden = false;
  const W = res ? res.image.width : im.naturalWidth, H = res ? res.image.height : im.naturalHeight;
  const k = Math.min(1, 900 / Math.max(W, H));
  cv.width = Math.round(W * k); cv.height = Math.round(H * k);
  const g = cv.getContext("2d");
  g.drawImage(im, 0, 0, cv.width, cv.height);
  if (res) {
    g.strokeStyle = "#ff9f43"; g.lineWidth = 1.2;
    for (const [x, y, r] of res.detections) { g.beginPath(); g.arc(x * k, y * k, Math.max(1.5, r * k), 0, 7); g.stroke(); }
  }
}

// ------------------------------------------------------------------ globe (three.js, loaded on demand)
let globe = null;

async function initGlobe(res) {
  const host = $("#globe");
  try {
    if (!globe) globe = await createGlobe(host);
    globe.show(res.solution);
  } catch (e) {
    console.error(e);
    host.innerHTML = `<p class="globe-msg">The 3-D view could not load.</p>`;
  }
}

async function createGlobe(host) {
  const THREE = await import("three");
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
  host.innerHTML = "";
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.0005, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 0.004;          // distance from the orbit target (a surface point when zoomed in)
  controls.maxDistance = 12;
  controls.rotateSpeed = 0.5;
  controls.zoomSpeed = 0.9;

  // stars
  const sg = new THREE.BufferGeometry();
  const sp = new Float32Array(3000 * 3);
  for (let i = 0; i < sp.length; i += 3) {
    const v = new THREE.Vector3().randomDirection().multiplyScalar(40 + Math.random() * 20);
    sp.set([v.x, v.y, v.z], i);
  }
  sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
  scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ color: 0x8890a0, size: 0.06 })));

  const loader = new THREE.TextureLoader();
  const tex = loader.load("assets/moon_1k.jpg", () => {
    const hi = loader.load("assets/moon_4k.jpg", () => { mat.map = hi; mat.needsUpdate = true; });
    hi.colorSpace = THREE.SRGBColorSpace;
    hi.anisotropy = renderer.capabilities.getMaxAnisotropy();
  });
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({ map: tex });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 160, 96), mat));

  // body frame (x: lon 0, y: lon 90 E, z: north) -> three.js (x, z, -y)
  const T = (p, k = 1) => new THREE.Vector3(p[0] / MOON_R * k, p[2] / MOON_R * k, -p[1] / MOON_R * k);
  const ll2v = (lat, lon, k = 1.0004) => {
    const a = lat * Math.PI / 180, o = lon * Math.PI / 180;
    return new THREE.Vector3(Math.cos(a) * Math.cos(o) * k, Math.sin(a) * k, -Math.cos(a) * Math.sin(o) * k);
  };
  const group = new THREE.Group();
  scene.add(group);

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host);
  resize();

  let fly = null;
  // fly the view to look at `target` (a point near the surface) from `pos`
  function flyTo(pos, target, ms = 1600) {
    fly = { t0: performance.now(), ms, from: camera.position.clone(), to: pos.clone(),
            fromT: controls.target.clone(), toT: target.clone() };
  }
  function tick(now) {
    if (fly) {
      let t = Math.min(1, (now - fly.t0) / fly.ms);
      t = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      // move along the sphere, not through it
      const r = Math.max(1.002, fly.from.length() * (1 - t) + fly.to.length() * t);
      camera.position.copy(fly.from.clone().normalize().lerp(fly.to.clone().normalize(), t).normalize().multiplyScalar(r));
      controls.target.copy(fly.fromT.clone().lerp(fly.toT, t));
      if (t >= 1) fly = null;
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  camera.position.set(0, 0.6, 3.2);
  requestAnimationFrame(tick);

  let current = null;
  $("#globe-zoom").addEventListener("click", () => current && flyTo(current.close, current.look));
  $("#globe-whole").addEventListener("click", () => current && flyTo(current.far, new THREE.Vector3()));

  return {
    show(sol) {
      group.clear();
      const fp = sol.footprint || [];
      const red = new THREE.LineBasicMaterial({ color: 0xff5a5a });
      if (fp.length > 2) {
        const pts = fp.map(([a, o]) => ll2v(a, o));
        pts.push(pts[0]);
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), red));
      }
      const C = T(sol.camera.C);
      const camMat = new THREE.MeshBasicMaterial({ color: 0x4fb3ff });
      const center = ll2v(sol.center_lat ?? sol.camera_lat, sol.center_lon ?? sol.camera_lon, 1);
      const Cn = C.clone().normalize();
      if (C.length() > 4) {
        // far view (thousands of km or more): the camera is off the scene; an arrow from the point
        // below it shows the direction it looked from
        const base = Cn.clone().multiplyScalar(1.002), tip = Cn.clone().multiplyScalar(1.9);
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([base, tip]),
          new THREE.LineBasicMaterial({ color: 0x4fb3ff })));
        const head = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.12, 16), camMat);
        head.position.copy(tip);
        head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), Cn);
        group.add(head);
        // look from 35 degrees beside the camera direction, so that the arrow is seen from the side
        let side = new THREE.Vector3(0, 1, 0).cross(Cn);
        if (side.length() < 1e-6) side = new THREE.Vector3(1, 0, 0);
        side.normalize();
        const a35 = 35 * Math.PI / 180;
        const view = Cn.clone().multiplyScalar(Math.cos(a35)).add(side.multiplyScalar(Math.sin(a35))).multiplyScalar(3.2);
        current = { look: new THREE.Vector3(), close: view, far: Cn.clone().multiplyScalar(3.8) };
        camera.position.copy(current.far);
        controls.target.set(0, 0, 0);
        setTimeout(() => flyTo(current.close, current.look, 1600), 300);
        return;
      }
      const dot = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), camMat);
      group.add(dot);
      // view cone: lines from the camera to the four image corners
      const cam = makeCamera(sol.camera);
      const W = sol.camera.width, H = sol.camera.height;
      const coneMat = new THREE.LineBasicMaterial({ color: 0x4fb3ff, transparent: true, opacity: 0.75 });
      for (const [u, v] of [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]) {
        const p = cam.hit(u, v);
        if (!p) continue;
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([C, T(p, 1.0004)]), coneMat));
      }
      // nadir line
      const nadir = Cn.clone().multiplyScalar(1.0004);
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([C, nadir]),
        new THREE.LineDashedMaterial({ color: 0x9aa3b5, dashSize: 0.004, gapSize: 0.004 })).computeLineDistances());
      let ext = 0.01;
      for (const [a, o] of fp) ext = Math.max(ext, ll2v(a, o, 1).distanceTo(center));
      const reach = Math.min(Math.max(ext, C.distanceTo(center) * 0.6), 1.5);
      dot.scale.setScalar(Math.max(0.002, reach * 0.05));
      // oblique close view from behind the spacecraft: look at the middle between the footprint
      // centre and the camera's nadir, from 55 degrees off the vertical, so that the view cone
      // shows the altitude
      const look = center.clone().add(Cn).multiplyScalar(0.5).normalize();
      const up = look.clone();
      let h = Cn.clone().sub(up.clone().multiplyScalar(up.dot(Cn)));    // towards the camera, along the surface
      if (h.length() < 1e-3) {                                          // nadir view: look from the south
        h = new THREE.Vector3(0, -1, 0).sub(up.clone().multiplyScalar(-up.y));
        if (h.length() < 1e-6) h = new THREE.Vector3(1, 0, 0);
      }
      h.normalize();
      const d = Math.max(2.2 * Math.max(C.distanceTo(center), C.length() - 1), 0.02);
      const a = 55 * Math.PI / 180;
      const close = look.clone().add(up.clone().multiplyScalar(d * Math.cos(a))).add(h.multiplyScalar(d * Math.sin(a)));
      current = { look, close, far: look.clone().multiplyScalar(3.2) };
      camera.position.copy(current.far);
      controls.target.set(0, 0, 0);
      setTimeout(() => flyTo(current.close, current.look, 2400), 500);
    },
  };
}

// ------------------------------------------------------------------ downloads and navigation
function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function initResultActions() {
  $("#again").addEventListener("click", () => show("view-start"));
  $("#fail-back").addEventListener("click", () => show("view-start"));
  $("#fail-retry").addEventListener("click", () => {
    if (!state.file) return show("view-start");
    const o = { ...(state.lastOpts || options()) };
    o.max_time = 60;
    startSolve(state.file, o);
  });
  $("#cancel").addEventListener("click", () => state.controller?.abort());
  $("#home-link").addEventListener("click", (e) => { e.preventDefault(); show("view-start"); });
  $("#dl-json").addEventListener("click", () => {
    const base = (state.file?.name || "image").replace(/\.[^.]+$/, "");
    download(base + "_solution.json", new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" }));
  });
  $("#dl-png").addEventListener("click", () => {
    const res = state.result;
    const off = document.createElement("canvas");
    off.width = res.image.width; off.height = res.image.height;
    draw({ g: off.getContext("2d"), cw: off.width, ch: off.height, view: { s: 1, tx: 0, ty: 0 } });
    off.toBlob((b) => download((state.file?.name || "image").replace(/\.[^.]+$/, "") + "_solved.png", b));
  });
}

// ------------------------------------------------------------------ boot
function boot() {
  initStart();
  initResultActions();
  checkHealth();
  setInterval(checkHealth, 60000);
}
boot();
