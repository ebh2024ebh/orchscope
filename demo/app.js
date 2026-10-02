/*
 * Scanning-campaign replay: offline demo accompanying the ICC paper on
 * real-time inference of orchestrated scanning campaigns from darknet data.
 *
 * Runs straight from disk (file://). data.js sets window.CAMPAIGN_DATA and
 * this script renders it with DOM nodes and <canvas>. There are no network
 * requests, frameworks or external assets. Every data-derived string reaches
 * the page through textContent, and every field is optional: anything
 * missing renders as a dash or is skipped.
 *
 * window.CAMPAIGN_DATA = {
 *   meta: {title, telescope, window_start (ISO 8601), hours, n_profiles,
 *          n_campaigns, n_orchestrated, anonymized, note, synthetic?},
 *   hourly: [{h, records, sources}, ...],            one row per hour
 *   campaigns: [{
 *     id, bots, port, proto, tool, strategy, orchestrated, z, pph_med,
 *     coverage, first_h, last_h, first_detected_h, asns,
 *     fingerprint: {TTL, "IP ID", "TCP window", "TCP options",
 *                   "Seq number", "Source port"},
 *     countries: {CC: bots, ...},
 *     activity: [bots active in each hour],          length meta.hours
 *     points: [[hour (float), offset inside the monitored block 0..1], ...],
 *     openjev: {tool: {label: p}, service: {...}, cls: {...},
 *               coordinated: p_yes,
 *               severity: {"can wait", "this week", "today", "right now"}}
 *   }, ...]
 * }
 */
(function () {
  'use strict';

  // ------------------------------------------------------------ constants
  const SEV_KEYS = ['can wait', 'this week', 'today', 'right now'];
  const SEV_NAMES = ['Can wait', 'This week', 'Today', 'Right now'];
  const FP_FIELDS = ['TTL', 'IP ID', 'TCP window', 'TCP options', 'Seq number', 'Source port'];
  const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const HOUR_MS = 3600000;
  const THEME_KEY = 'campaign-demo-theme';
  const NEW_HOURS = 6;            // "New" tag for campaigns reported this recently
  const THUMB = 16;               // scrubber thumb size in px, see style.css
  const SPARK_W = 84;
  const SPARK_H = 22;
  const FLASH_MS = 1600;
  const NDASH = '\u2013';
  const MDASH = '\u2014';
  const DOT = '\u00b7';
  const MINUS = '\u2212';
  const ELLIPSIS = '\u2026';
  const DASH = NDASH;             // shown for missing values

  const SORTS = {
    bots: { dir: -1, get: (c) => c.bots },
    severity: { dir: -1, get: (c) => c.sevExp },
    z: { dir: -1, get: (c) => c.z },
    detect: { dir: 1, get: (c) => c.det },
  };

  // ------------------------------------------------------------- helpers
  const doc = document;
  const byId = (id) => doc.getElementById(id);

  function isObj(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }
  function obj(v) { return isObj(v) ? v : {}; }
  function arr(v) { return Array.isArray(v) ? v : []; }

  function finite(v) {
    if (typeof v === 'number') return Number.isFinite(v) ? v : null;
    if (typeof v === 'string' && v.trim() !== '') {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  }

  function str(v) {
    if (typeof v === 'string') {
      const s = v.trim();
      return s ? s : null;
    }
    if (typeof v === 'number' && Number.isFinite(v)) return String(v);
    if (typeof v === 'boolean') return v ? 'yes' : 'no';
    return null;
  }

  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
  function plural(n, one, many) { return n === 1 ? one : many; }
  function pad2(n) { return n < 10 ? '0' + n : String(n); }
  function capFirst(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

  /** Create an element. props: class, text, title or any attribute. */
  function el(tag, props, kids) {
    const node = doc.createElement(tag);
    if (props) {
      Object.keys(props).forEach((key) => {
        const v = props[key];
        if (v === null || v === undefined || v === false) return;
        if (key === 'class') node.className = v;
        else if (key === 'text') node.textContent = String(v);
        else if (key === 'title') node.title = String(v);
        else node.setAttribute(key, v === true ? '' : String(v));
      });
    }
    return append(node, kids);
  }

  function append(node, kids) {
    if (kids === null || kids === undefined || kids === false) return node;
    (Array.isArray(kids) ? kids : [kids]).forEach((k) => {
      if (k === null || k === undefined || k === false) return;
      node.appendChild(typeof k === 'object' ? k : doc.createTextNode(String(k)));
    });
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function setText(node, s) {
    if (node && node.textContent !== s) node.textContent = s;
  }

  // ---------------------------------------------------------- formatting
  function fmtInt(n) {
    const v = finite(n);
    return v === null ? DASH : Math.round(v).toLocaleString('en-US');
  }

  function fmtCompact(n) {
    const v = finite(n);
    if (v === null) return DASH;
    const a = Math.abs(v);
    const cut = (x, digits) => x.toFixed(digits).replace(/\.0$/, '');
    if (a >= 1e9) return cut(v / 1e9, a >= 1e11 ? 0 : 1) + 'B';
    if (a >= 1e6) return cut(v / 1e6, a >= 1e8 ? 0 : 1) + 'M';
    if (a >= 1e4) return cut(v / 1e3, a >= 1e5 ? 0 : 1) + 'K';
    return fmtInt(v);
  }

  function fmtPct(p) {
    const v = finite(p);
    if (v === null) return DASH;
    const x = v * 100;
    if (x === 0) return '0%';
    if (x >= 99.95 && x < 100) return '99.9%';
    return parseFloat(x.toPrecision(3)) + '%';
  }

  function fmtShare(p) {
    const v = finite(p);
    return v === null ? DASH : (v * 100).toFixed(1) + '%';
  }

  function fmtProb(p) {
    const v = finite(p);
    if (v === null) return DASH;
    if (v > 0 && v < 0.005) return '<0.01';
    return v.toFixed(2);
  }

  function fmtZ(z) {
    const v = finite(z);
    if (v === null) return DASH;
    return (v < 0 ? MINUS : '') + Math.abs(v).toFixed(1);
  }

  function fmtRate(x) {
    const v = finite(x);
    if (v === null) return DASH;
    if (v >= 100) return fmtInt(v);
    return v >= 10 ? v.toFixed(1) : v.toFixed(2);
  }

  // Time. Hour h covers [h, h+1) after window_start; all times are UTC.
  function dateAt(h) { return new Date(model.t0 + h * HOUR_MS); }

  function isMidnight(h) {
    if (model.t0 === null) return h % 24 === 0;
    const d = dateAt(h);
    return d.getUTCHours() === 0 && d.getUTCMinutes() === 0;
  }

  function absHour(h) {
    return model.t0 === null ? h : Math.round(model.t0 / HOUR_MS) + h;
  }

  function fmtClock(h) {
    if (model.t0 === null) return 'h' + Math.floor(h);
    const d = dateAt(h);
    return pad2(d.getUTCHours()) + ':' + pad2(d.getUTCMinutes());
  }

  function fmtDay(h) {
    if (model.t0 === null) return 'Day ' + (Math.floor(h / 24) + 1);
    const d = dateAt(h);
    return DOW[d.getUTCDay()] + ' ' + d.getUTCDate();
  }

  function fmtDayMonth(h) {
    if (model.t0 === null) return 'Day ' + (Math.floor(h / 24) + 1);
    const d = dateAt(h);
    return DOW[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' + MON[d.getUTCMonth()];
  }

  /** " UTC" suffix, empty when the data carries no usable window_start. */
  function tz() { return model.t0 === null ? '' : ' UTC'; }

  /** "Thu 24 Sep 14:00" (start of hour h). */
  function fmtWhen(h) {
    if (model.t0 === null) return 'hour ' + (Math.floor(h) + 1);
    return fmtDayMonth(h) + ' ' + fmtClock(h);
  }

  /** "Thu 24 Sep 2026, 14:00-15:00 UTC" for hour h. */
  function fmtHourRange(h) {
    if (model.t0 === null) return 'Hour ' + (h + 1) + ' of ' + model.H;
    const d = dateAt(h);
    return DOW[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + ' ' +
      d.getUTCFullYear() + ', ' + fmtClock(h) + NDASH + fmtClock(h + 1) + ' UTC';
  }

  /** Fractional hour to "Thu 24 Sep 14:23". */
  function fmtInstant(t) {
    if (model.t0 === null) return 'hour ' + t.toFixed(2);
    const d = new Date(model.t0 + t * HOUR_MS);
    return DOW[d.getUTCDay()] + ' ' + d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + ' ' +
      pad2(d.getUTCHours()) + ':' + pad2(d.getUTCMinutes());
  }

  function fmtPortProto(port, proto) {
    if (!port && !proto) return 'unknown port';
    if (!port) return proto;
    if (!proto) return port;
    return /^[0-9][0-9+,\s-]*$/.test(port) ? port + '/' + proto : port + ' ' + DOT + ' ' + proto;
  }

  let regionNames = null;
  try {
    if (typeof Intl === 'object' && typeof Intl.DisplayNames === 'function') {
      regionNames = new Intl.DisplayNames(['en'], { type: 'region' });
    }
  } catch (e) { regionNames = null; }
  const REGION_SHORT = { HK: 'Hong Kong', MO: 'Macao', US: 'United States', GB: 'United Kingdom', KR: 'South Korea' };

  function countryName(cc) {
    const code = String(cc).toUpperCase();
    if (REGION_SHORT[code]) return REGION_SHORT[code];
    if (regionNames && /^[A-Z]{2}$/.test(code)) {
      try {
        const n = regionNames.of(code);
        if (n && n !== code) return n;
      } catch (e) { /* unknown code: fall through */ }
    }
    return code;
  }

  // ------------------------------------------------------------ the model
  let model = null;

  function buildModel(raw) {
    const src = obj(raw);
    const m = obj(src.meta);
    const rawHourly = arr(src.hourly);
    const rawCamps = arr(src.campaigns);

    let H = finite(m.hours);
    H = H !== null && H >= 1 ? Math.floor(H) : 0;
    if (!H) {
      let longest = rawHourly.length;
      rawCamps.forEach((c) => { longest = Math.max(longest, arr(obj(c).activity).length); });
      H = longest || 168;
    }
    H = Math.min(H, 24 * 366);

    const ws = str(m.window_start);
    const parsed = ws ? Date.parse(ws) : NaN;
    const t0 = Number.isFinite(parsed) ? parsed : null;

    const hourly = [];
    for (let h = 0; h < H; h++) hourly.push({ records: null, sources: null });
    rawHourly.forEach((row, i) => {
      const r = obj(row);
      const hv = finite(r.h);
      const h = hv !== null ? Math.floor(hv) : i;
      if (h >= 0 && h < H) hourly[h] = { records: finite(r.records), sources: finite(r.sources) };
    });

    const used = new Set();
    const campaigns = [];
    rawCamps.forEach((rc, i) => {
      if (isObj(rc)) campaigns.push(normCampaign(rc, i, H, used));
    });

    const meta = {
      title: str(m.title),
      telescope: str(m.telescope),
      note: str(m.note),
      nProfiles: finite(m.n_profiles),
      nCampaigns: finite(m.n_campaigns),
      nOrch: finite(m.n_orchestrated),
      anonymized: m.anonymized !== false,
      synthetic: m.synthetic === true,
    };

    return {
      H, t0, hourly, campaigns, meta,
      byId: new Map(campaigns.map((c) => [c.id, c])),
      totalBots: campaigns.reduce((s, c) => s + c.bots, 0),
      nOrchData: campaigns.filter((c) => c.orch).length,
      trackStats: {
        records: rangeOf(hourly.map((r) => r.records)),
        sources: rangeOf(hourly.map((r) => r.sources)),
      },
    };
  }

  function rangeOf(values) {
    let lo = Infinity;
    let hi = -Infinity;
    values.forEach((v) => {
      if (v === null) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
    if (!Number.isFinite(lo)) return null;
    const span = hi - lo || Math.abs(hi) || 1;
    return { lo: lo - span * 0.12, hi: hi + span * 0.06 };
  }

  function normCampaign(rc, index, H, used) {
    let id = finite(rc.id);
    id = id !== null ? Math.round(id) : null;
    if (id === null || used.has(id)) {
      id = index + 1;
      while (used.has(id)) id++;
    }
    used.add(id);

    const activity = new Array(H).fill(0);
    arr(rc.activity).forEach((v, h) => {
      if (h >= H) return;
      const n = finite(v);
      activity[h] = n !== null && n > 0 ? n : 0;
    });
    let firstAct = -1;
    let lastAct = -1;
    let maxAct = 0;
    let peakH = -1;
    for (let h = 0; h < H; h++) {
      const a = activity[h];
      if (a > 0) {
        if (firstAct < 0) firstAct = h;
        lastAct = h;
      }
      if (a > maxAct) { maxAct = a; peakH = h; }
    }

    let firstH = finite(rc.first_h);
    let lastH = finite(rc.last_h);
    firstH = firstH !== null ? clamp(Math.floor(firstH), 0, H - 1) : (firstAct >= 0 ? firstAct : 0);
    lastH = lastH !== null ? clamp(Math.floor(lastH), 0, H - 1) : (lastAct >= 0 ? lastAct : H - 1);
    if (lastH < firstH) { const t = lastH; lastH = firstH; firstH = t; }
    let det = finite(rc.first_detected_h);
    det = det !== null ? clamp(Math.floor(det), 0, H - 1) : firstH;

    const botsIn = finite(rc.bots);
    const bots = Math.max(0, Math.round(botsIn !== null ? botsIn : maxAct));
    const port = str(rc.port);
    const protoRaw = str(rc.proto);
    const proto = protoRaw ? protoRaw.toUpperCase() : null;
    const tool = str(rc.tool);
    const strategy = str(rc.strategy);
    const orch = rc.orchestrated === true;
    // null: too few peers on the port to test coordination (undetermined)
    const undet = rc.orchestrated === null;

    const fpIn = obj(rc.fingerprint);
    const fingerprint = FP_FIELDS.map((k) => [k, str(fpIn[k])]);
    Object.keys(fpIn).forEach((k) => {
      if (FP_FIELDS.indexOf(k) >= 0) return;
      const v = str(fpIn[k]);
      if (v !== null) fingerprint.push([k, v]);
    });

    const countries = [];
    const cIn = obj(rc.countries);
    Object.keys(cIn).forEach((k) => {
      const n = finite(cIn[k]);
      if (n !== null && n > 0) countries.push([String(k), n]);
    });
    countries.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1));

    const points = [];
    arr(rc.points).forEach((p) => {
      if (!Array.isArray(p) || p.length < 2) return;
      const t = finite(p[0]);
      const u = finite(p[1]);
      if (t === null || u === null || t < 0 || t > H || u < 0 || u > 1) return;
      points.push([t, u]);
    });
    points.sort((a, b) => a[0] - b[0]);

    const oj = normOpenjev(rc.openjev);
    const label = 'C' + id;
    const portLabel = fmtPortProto(port, proto);
    const searchText = [port, proto, tool, strategy, orch ? 'orchestrated' : (undet ? 'undetermined' : 'independent')]
      .filter(Boolean).join(' ').toLowerCase();

    return {
      id, label, bots, port, proto, portLabel, tool, strategy, orch, undet,
      z: finite(rc.z), pph: finite(rc.pph_med), coverage: finite(rc.coverage), asns: finite(rc.asns),
      firstH, lastH, det, activity, maxAct, peakH, fingerprint, countries,
      countrySum: countries.reduce((s, x) => s + x[1], 0),
      points, oj,
      sevExp: oj ? oj.sevExp : null,
      sevLevel: oj ? oj.sevLevel : null,
      searchText,
      strip: null,
      stripVersion: -1,
    };
  }

  function normDist(v) {
    const o = obj(v);
    const out = [];
    Object.keys(o).forEach((k) => {
      const p = finite(o[k]);
      if (p !== null && p >= 0) out.push([String(k), Math.min(1, p)]);
    });
    out.sort((a, b) => b[1] - a[1]);
    return out;
  }

  function normOpenjev(v) {
    if (!isObj(v)) return null;
    const tool = normDist(v.tool);
    const service = normDist(v.service);
    const cls = normDist(v.cls);
    const coordinated = finite(v.coordinated);

    const sevIn = {};
    Object.keys(obj(v.severity)).forEach((k) => { sevIn[k.trim().toLowerCase()] = v.severity[k]; });
    let sev = SEV_KEYS.map((k) => {
      const p = finite(sevIn[k]);
      return p !== null && p > 0 ? p : 0;
    });
    const total = sev.reduce((a, b) => a + b, 0);
    let sevExp = null;
    let sevLevel = null;
    if (total > 0) {
      sev = sev.map((p) => p / total);
      sevExp = sev.reduce((acc, p, i) => acc + p * i, 0);
      sevLevel = clamp(Math.round(sevExp), 0, 3);
    } else {
      sev = null;
    }

    if (!tool.length && !service.length && !cls.length && coordinated === null && !sev) return null;
    return {
      tool, service, cls,
      coordinated: coordinated !== null ? clamp(coordinated, 0, 1) : null,
      sev, sevExp, sevLevel,
    };
  }

  // --------------------------------------------------------------- state
  const state = {
    hour: 0,
    playing: false,
    speed: 1,
    acc: 0,
    lastTs: null,
    sortKey: 'bots',
    sortDir: -1,
    orchOnly: false,
    proto: null,
    query: '',
    selectedId: null,
    theme: 'light',
  };

  let C = {};                  // theme colors read from CSS custom properties
  let FONT = 'system-ui, sans-serif';
  let themeVersion = 0;
  let visible = [];            // campaigns in the list, display order
  let reportedIds = new Set();
  let counts = { all: 0, orch: 0, TCP: 0, UDP: 0 };
  let listSig = null;
  let rafId = 0;
  const flashUntil = new Map();
  const ui = {};

  function isReported(c) { return state.hour >= c.det; }
  function isNew(c) { return isReported(c) && state.hour - c.det < NEW_HOURS; }
  function colorOf(c) { return c.orch ? C.orch : (c.undet ? C.undet : C.indep); }
  function orchWord(c) { return c.orch ? 'Orchestrated' : (c.undet ? 'Undetermined' : 'Independent'); }


  // ---------------------------------------------------------------- theme
  function readTheme() {
    const cs = getComputedStyle(doc.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    C = {
      text: v('--text'), text2: v('--text-2'), text3: v('--text-3'),
      grid: v('--grid'), axis: v('--axis'), border: v('--border'), track: v('--track'),
      surface: v('--surface'), orch: v('--c-orch'), indep: v('--c-indep'), undet: v('--c-undet'),
      model: v('--c-model'), sev: v('--c-sev'), neutral: v('--c-neutral'),
      accent: v('--accent'), cursor: v('--cursor'), dim: v('--dim'),
      selBg: v('--sel-bg'), hoverBg: v('--hover-bg'),
    };
    FONT = getComputedStyle(doc.body).fontFamily || FONT;
    themeVersion++;
  }

  function setTheme(theme, persist) {
    state.theme = theme === 'dark' ? 'dark' : 'light';
    doc.documentElement.setAttribute('data-theme', state.theme);
    ui.themeToggle.setAttribute('aria-pressed', state.theme === 'dark' ? 'true' : 'false');
    if (persist) {
      try { window.localStorage.setItem(THEME_KEY, state.theme); } catch (e) { /* ignore */ }
    }
    readTheme();
  }

  function rgbOf(color) {
    const s = String(color).trim();
    let m = /^#([0-9a-f]{6})$/i.exec(s);
    if (m) {
      const n = parseInt(m[1], 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    m = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(s);
    if (m) return [parseInt(m[1] + m[1], 16), parseInt(m[2] + m[2], 16), parseInt(m[3] + m[3], 16)];
    m = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(s);
    if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
    return [128, 128, 128];
  }

  function alpha(color, a) {
    const rgb = rgbOf(color);
    return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + a + ')';
  }

  // --------------------------------------------------------------- canvas
  /** Size a canvas for the device pixel ratio. fixedWidth also sets CSS width. */
  function setupCanvas(canvas, w, h, fixedWidth) {
    const dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
    const pw = Math.max(1, Math.round(w * dpr));
    const ph = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== pw) canvas.width = pw;
    if (canvas.height !== ph) canvas.height = ph;
    if (fixedWidth) canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  function crisp(x) { return Math.round(x) + 0.5; }

  function hline(ctx, x0, x1, y, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
  }

  function vline(ctx, x, y0, y1, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
  }

  function roundRectPath(ctx, x, y, w, h, r) {
    const rr = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.arcTo(x + w, y, x + w, y + rr, rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr);
    ctx.lineTo(x + rr, y + h);
    ctx.arcTo(x, y + h, x, y + h - rr, rr);
    ctx.lineTo(x, y + rr);
    ctx.arcTo(x, y, x + rr, y, rr);
    ctx.closePath();
  }

  function fitText(ctx, text, maxW) {
    if (ctx.measureText(text).width <= maxW) return text;
    let s = text;
    while (s.length > 1 && ctx.measureText(s + ELLIPSIS).width > maxW) s = s.slice(0, -1);
    return s + ELLIPSIS;
  }

  /** Vertical cursor line plus the "after the cursor" wash inside a plot. */
  function drawCursor(ctx, x, left, right, top, bottom, washOnly) {
    if (x >= right) return;
    const cx = Math.max(left, x);
    ctx.fillStyle = C.dim;
    ctx.fillRect(cx, top, right - cx, bottom - top);
    if (x >= left && !washOnly) {
      ctx.fillStyle = C.cursor;
      ctx.fillRect(Math.round(x) - 1, top, 2, bottom - top);
    }
  }

  /** Time ticks and labels along the bottom of a plot spanning [d0, d1] hours. */
  function drawTimeAxis(ctx, d0, d1, X, left, right, top, bottom) {
    const span = d1 - d0;
    const maxTicks = Math.max(2, Math.floor((right - left) / 46));
    const steps = [1, 2, 3, 6, 12, 24, 48, 72, 168];
    let step = steps[steps.length - 1];
    for (let i = 0; i < steps.length; i++) {
      if (span / steps[i] <= maxTicks) { step = steps[i]; break; }
    }
    ctx.font = '11px ' + FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let h = Math.ceil(d0); h <= Math.floor(d1); h++) {
      if (((absHour(h) % step) + step) % step !== 0) continue;
      const x = crisp(X(h));
      vline(ctx, x, top, bottom, C.grid);
      const label = step >= 24 || isMidnight(h) ? fmtDay(h) : fmtClock(h);
      const tw = ctx.measureText(label).width;
      if (x - tw / 2 < left - 18 || x + tw / 2 > right + 10) continue;
      ctx.fillStyle = C.text3;
      ctx.fillText(label, x, bottom + 6);
    }
    hline(ctx, left, right, crisp(bottom), C.axis);
  }

  function niceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const m = v / p;
    const steps = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
    for (let i = 0; i < steps.length; i++) if (m <= steps[i] + 1e-9) return steps[i] * p;
    return 10 * p;
  }

  /** Campaign span used by the detail charts: at least 12 hours wide. */
  function spanDomain(c) {
    let d0 = c.firstH;
    let d1 = c.lastH + 1;
    const minSpan = Math.min(12, model.H);
    if (d1 - d0 < minSpan) {
      const mid = (d0 + d1) / 2;
      d0 = mid - minSpan / 2;
      d1 = mid + minSpan / 2;
      if (d0 < 0) { d1 -= d0; d0 = 0; }
      if (d1 > model.H) { d0 -= d1 - model.H; d1 = model.H; }
      d0 = Math.max(0, d0);
    }
    return [d0, d1];
  }

  // -------------------------------------------------------------- tooltip
  function showTip(clientX, clientY, kids) {
    const tip = ui.tooltip;
    clear(tip);
    append(tip, kids);
    tip.hidden = false;
    const pad = 14;
    const r = tip.getBoundingClientRect();
    let x = clientX + pad;
    let y = clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = clientX - pad - r.width;
    if (y + r.height > window.innerHeight - 8) y = clientY - pad - r.height;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  }

  function hideTip() { if (ui.tooltip) ui.tooltip.hidden = true; }

  function tipTitle(text) { return el('div', { class: 'tt-title', text: text }); }
  function tipLine(text) { return el('div', { class: 'tt-row', text: text }); }
  function tipValue(color, value, label) {
    const key = el('span', { class: 'tt-key' });
    key.style.background = color;
    return el('div', { class: 'tt-row' }, [
      key, el('span', { class: 'tt-value', text: value }), label ? el('span', { text: label }) : null,
    ]);
  }

  // ------------------------------------------------------- list filtering
  function matches(c, terms) {
    for (let i = 0; i < terms.length; i++) {
      const t = terms[i];
      const idm = /^[c#](\d+)$/.exec(t);
      if (idm) {
        if (String(c.id) !== idm[1]) return false;
      } else if (c.searchText.indexOf(t) === -1) {
        return false;
      }
    }
    return true;
  }

  function compareCampaigns(a, b) {
    const s = SORTS[state.sortKey] || SORTS.bots;
    const va = s.get(a);
    const vb = s.get(b);
    const na = va === null || va === undefined;
    const nb = vb === null || vb === undefined;
    if (na !== nb) return na ? 1 : -1;      // missing values sort last
    if (!na && va !== vb) return (va < vb ? -1 : 1) * state.sortDir;
    return (b.bots - a.bots) || (a.id - b.id);
  }

  function refreshVisible() {
    const terms = state.query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const reported = model.campaigns.filter((c) => isReported(c) && matches(c, terms));
    counts = {
      all: reported.length,
      orch: reported.filter((c) => c.orch).length,
      TCP: reported.filter((c) => c.proto === 'TCP').length,
      UDP: reported.filter((c) => c.proto === 'UDP').length,
    };
    visible = reported.filter((c) => (!state.orchOnly || c.orch) && (!state.proto || c.proto === state.proto));
    visible.sort(compareCampaigns);
    updateChips();
    const shown = visible.length;
    const rep = reportedIds.size;
    setText(ui.listCount, shown === rep ? rep + ' reported' : shown + ' shown of ' + rep + ' reported');
  }

  function updateChips() {
    ui.chips.forEach((b) => {
      const f = b.getAttribute('data-filter');
      let on;
      if (f === 'all') on = !state.orchOnly && !state.proto;
      else if (f === 'orch') on = state.orchOnly;
      else on = state.proto === f;
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
      const n = b.querySelector('.chip-n');
      if (n) setText(n, counts[f] !== undefined ? String(counts[f]) : '');
    });
  }

  // ------------------------------------------------------------ list view
  function sevGlyph(level) {
    const g = el('span', { class: 'sev-glyph', 'aria-hidden': 'true' });
    for (let i = 0; i < 4; i++) g.appendChild(el('i', { class: level !== null && i <= level ? 'on' : null }));
    return g;
  }

  function orchTag(c) {
    return el('span', { class: 'tag ' + (c.orch ? 'tag-orch' : (c.undet ? 'tag-undet' : 'tag-indep')) }, [
      el('span', { class: 'dot', 'aria-hidden': 'true' }), orchWord(c),
    ]);
  }

  function sevTag(c) {
    if (c.sevLevel === null) return null;
    return el('span', {
      class: 'tag tag-sev',
      title: 'Expected severity ' + c.sevExp.toFixed(2) + ' on a 0' + NDASH + '3 scale',
    }, [sevGlyph(c.sevLevel), SEV_NAMES[c.sevLevel]]);
  }

  function rowLabel(c) {
    const parts = [c.label, c.portLabel, c.tool || 'unattributed tool', fmtInt(c.bots) + ' ' + plural(c.bots, 'bot', 'bots'),
      orchWord(c).toLowerCase()];
    if (c.sevLevel !== null) parts.push('severity ' + SEV_NAMES[c.sevLevel].toLowerCase());
    if (c.z !== null) parts.push('synchrony z ' + fmtZ(c.z));
    parts.push('first reported ' + fmtWhen(c.det) + tz());
    if (isNew(c)) parts.push('new');
    return parts.join(', ');
  }

  function buildRow(c, tabId, now) {
    const sel = c.id === state.selectedId;
    const li = el('li', {
      class: 'crow',
      role: 'option',
      id: 'opt-' + c.id,
      'data-id': String(c.id),
      'aria-selected': sel ? 'true' : 'false',
      'aria-label': rowLabel(c),
      tabindex: c.id === tabId ? '0' : '-1',
    });

    const flashEnd = flashUntil.get(c.id);
    if (flashEnd && flashEnd > now) {
      li.classList.add('flash');
      li.style.animationDelay = -(FLASH_MS - (flashEnd - now)) + 'ms';
    }

    const meta = el('div', { class: 'crow-meta' }, [
      el('span', { class: 'crow-status ' + (c.orch ? 'is-orch' : (c.undet ? 'is-undet' : 'is-indep')) }, [
        el('span', { class: 'dot', 'aria-hidden': 'true' }), orchWord(c),
      ]),
      el('span', { class: 'crow-sev' }, [
        sevGlyph(c.sevLevel), c.sevLevel !== null ? SEV_NAMES[c.sevLevel] : 'Severity n/a',
      ]),
    ]);

    li.title = 'First reported in the ' + fmtWhen(c.det) + tz() + ' batch';
    append(li, [
      el('div', { class: 'crow-head' }, [
        el('span', { class: 'crow-id', text: c.label }),
        el('span', { class: 'crow-port', text: c.portLabel }),
        isNew(c) ? el('span', { class: 'tag tag-new', text: 'New' }) : null,
      ]),
      el('div', { class: 'crow-bots' }, [el('strong', { text: fmtInt(c.bots) }), ' ' + plural(c.bots, 'bot', 'bots')]),
      el('div', { class: 'crow-tool', text: c.tool || 'Unattributed tool' }),
      el('canvas', { class: 'crow-spark', 'aria-hidden': 'true' }),
      meta,
      el('div', { class: 'crow-z', text: 'z ' + fmtZ(c.z) }),
    ]);
    return li;
  }

  function drawRowSpark(canvas, c) {
    if (!canvas || !c) return;
    const ctx = setupCanvas(canvas, SPARK_W, SPARK_H, true);
    const H = model.H;
    const max = c.maxAct || 1;
    const X = (h) => (h + 0.5) / H * SPARK_W;
    const Y = (v) => SPARK_H - 2 - (v / max) * (SPARK_H - 5);
    const color = colorOf(c);
    ctx.fillStyle = C.grid;
    ctx.fillRect(0, SPARK_H - 1, SPARK_W, 1);

    ctx.beginPath();
    ctx.moveTo(X(c.firstH), SPARK_H - 1);
    for (let h = c.firstH; h <= c.lastH; h++) ctx.lineTo(X(h), Y(c.activity[h]));
    ctx.lineTo(X(c.lastH), SPARK_H - 1);
    ctx.closePath();
    ctx.fillStyle = alpha(color, 0.18);
    ctx.fill();

    const end = Math.min(c.lastH, state.hour);
    if (end >= c.firstH) {
      ctx.beginPath();
      for (let h = c.firstH; h <= end; h++) {
        if (h === c.firstH) ctx.moveTo(X(h), Y(c.activity[h]));
        else ctx.lineTo(X(h), Y(c.activity[h]));
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = 'round';
      ctx.stroke();
    }
    drawCursor(ctx, (state.hour + 1) / H * SPARK_W, 0, SPARK_W, 0, SPARK_H, true);
  }

  function focusedRowId() {
    const a = doc.activeElement;
    if (a && a.parentNode === ui.list && a.getAttribute('data-id')) return Number(a.getAttribute('data-id'));
    return null;
  }

  function rowEl(id) { return byId('opt-' + id); }

  function renderList() {
    const sig = visible.map((c) => c.id + (isNew(c) ? 'n' : '')).join(',') + '|' + state.selectedId;
    if (sig !== listSig) {
      listSig = sig;
      const hadFocus = focusedRowId();
      const listFocused = doc.activeElement === ui.list;
      const tabId = visible.some((c) => c.id === state.selectedId) ? state.selectedId
        : (visible.length ? visible[0].id : null);
      const now = Date.now();
      const frag = doc.createDocumentFragment();
      visible.forEach((c) => frag.appendChild(buildRow(c, tabId, now)));
      clear(ui.list);
      ui.list.appendChild(frag);
      if (hadFocus !== null) {
        const r = rowEl(hadFocus);
        if (r) r.focus({ preventScroll: true });
        else ui.list.focus({ preventScroll: true });
      } else if (listFocused) {
        ui.list.focus({ preventScroll: true });
      }
    }
    Array.prototype.forEach.call(ui.list.children, (li) => {
      drawRowSpark(li.querySelector('canvas'), model.byId.get(Number(li.getAttribute('data-id'))));
    });
    renderListEmpty();
  }

  function renderListEmpty() {
    const box = ui.listEmpty;
    if (visible.length) { box.hidden = true; return; }
    clear(box);
    if (!reportedIds.size) {
      append(box, [
        el('strong', { text: 'No campaigns reported yet' }),
        el('p', { text: 'Campaigns appear here as the online engine reports them, one hourly batch at a time. Press Play or move the cursor forward.' }),
      ]);
    } else {
      const btn = el('button', { type: 'button', class: 'btn', text: 'Clear search and filters' });
      btn.addEventListener('click', () => {
        state.query = '';
        ui.search.value = '';
        state.orchOnly = false;
        state.proto = null;
        refreshAll();
      });
      append(box, [
        el('strong', { text: 'No campaigns match' }),
        el('p', { text: reportedIds.size + ' ' + plural(reportedIds.size, 'campaign is', 'campaigns are') + ' reported at this hour, but none match the current search and filters.' }),
        btn,
      ]);
    }
    box.hidden = false;
  }

  function scrollRowIntoView(row) {
    const body = ui.listBody;
    if (!row || !body) return;
    const top = row.offsetTop;
    const bottom = top + row.offsetHeight;
    if (top < body.scrollTop) body.scrollTop = top;
    else if (bottom > body.scrollTop + body.clientHeight) body.scrollTop = bottom - body.clientHeight;
  }

  function onListKey(e) {
    if (!visible.length) return;
    const ids = visible.map((c) => c.id);
    let idx = ids.indexOf(state.selectedId);
    const focused = focusedRowId();
    if (focused !== null) idx = ids.indexOf(focused);
    let next;
    switch (e.key) {
      case 'ArrowDown': next = idx < 0 ? 0 : Math.min(ids.length - 1, idx + 1); break;
      case 'ArrowUp': next = idx < 0 ? 0 : Math.max(0, idx - 1); break;
      case 'Home': next = 0; break;
      case 'End': next = ids.length - 1; break;
      case 'PageDown': next = Math.min(ids.length - 1, Math.max(0, idx) + 8); break;
      case 'PageUp': next = Math.max(0, idx - 8); break;
      case 'Enter':
      case ' ':
        next = idx < 0 ? 0 : idx;
        break;
      default:
        return;
    }
    e.preventDefault();
    e.stopPropagation();
    select(ids[next], { focus: true, revealTimeline: true });
  }

  // ------------------------------------------------------------ selection
  function select(id, opts) {
    const o = opts || {};
    if (!model.byId.has(id)) return;
    const changed = state.selectedId !== id;
    state.selectedId = id;
    const tabId = visible.some((c) => c.id === id) ? id : (visible.length ? visible[0].id : null);
    Array.prototype.forEach.call(ui.list.children, (li) => {
      const rid = Number(li.getAttribute('data-id'));
      li.setAttribute('aria-selected', rid === id ? 'true' : 'false');
      li.tabIndex = rid === tabId ? 0 : -1;
    });
    listSig = visible.map((c) => c.id + (isNew(c) ? 'n' : '')).join(',') + '|' + state.selectedId;
    const row = rowEl(id);
    if (o.focus && row) row.focus({ preventScroll: true });
    if ((o.focus || o.revealList) && row) scrollRowIntoView(row);
    if (o.revealTimeline) tlReveal(id);
    if (changed) {
      renderDetail();
      tlDraw();
      writeHashSoon();
    }
    if (o.revealDetail) revealDetail();
  }

  /** In the stacked (narrow) layouts, bring the detail panel into view after a click. */
  function revealDetail() {
    if (isFixedLayout()) return;
    const panel = doc.querySelector('.panel-detail');
    if (!panel) return;
    const r = panel.getBoundingClientRect();
    if (r.top < window.innerHeight * 0.8 && r.bottom > 80) return;
    let smooth = true;
    try { smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { smooth = false; }
    panel.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
  }

  // ------------------------------------------------------------ timeline
  const TL = { rowH: 26, barH: 12, gutter: 120, padR: 14, axisH: 30, width: 0, viewH: 0, rows: [], hover: -1 };

  function isFixedLayout() {
    return getComputedStyle(doc.documentElement).getPropertyValue('--fixed-layout').trim() === '1';
  }

  function tlLayout() {
    TL.rows = visible.slice();
    const scroll = ui.tlScroll;
    const totalH = TL.rows.length * TL.rowH;
    let viewH;
    if (isFixedLayout()) {
      scroll.style.height = '';
      viewH = scroll.clientHeight;
    } else {
      const cap = Math.max(240, Math.round(window.innerHeight * 0.6));
      viewH = clamp(totalH, TL.rowH * 4, cap);
      scroll.style.height = viewH + 'px';
    }
    TL.viewH = Math.max(1, viewH);
    ui.tlSpacer.style.height = Math.max(0, totalH - TL.viewH) + 'px';
    ui.tlCanvas.style.height = TL.viewH + 'px';
    TL.width = Math.max(120, scroll.clientWidth);
    TL.gutter = TL.width < 520 ? 92 : 120;
    if (TL.hover >= TL.rows.length) TL.hover = -1;

    const box = ui.tlEmpty;
    if (TL.rows.length) {
      box.hidden = true;
    } else {
      clear(box);
      append(box, el('p', {
        text: reportedIds.size ? 'No campaigns match the current search and filters.'
          : 'No campaigns reported yet at this hour.',
      }));
      box.hidden = false;
    }
    ui.tlCanvas.setAttribute('aria-label', 'Timeline of ' + TL.rows.length + ' ' +
      plural(TL.rows.length, 'campaign', 'campaigns') + ' from first to last activity. The campaign list holds the same campaigns.');
  }

  function tlGeom() {
    const x0 = TL.gutter;
    const pw = Math.max(10, TL.width - TL.gutter - TL.padR);
    const k = pw / model.H;
    return { x0, pw, k, X: (h) => x0 + h * k };
  }

  function barStrip(c) {
    if (c.strip && c.stripVersion === themeVersion) return c.strip;
    const H = model.H;
    const cv = doc.createElement('canvas');
    cv.width = H;
    cv.height = 1;
    const g = cv.getContext('2d');
    const img = g.createImageData(H, 1);
    const rgb = rgbOf(colorOf(c));
    const max = c.maxAct || 1;
    for (let h = 0; h < H; h++) {
      let a = 0;
      if (h >= c.firstH && h <= c.lastH) {
        const v = c.activity[h];
        a = v > 0 ? 0.24 + 0.76 * Math.pow(v / max, 0.6) : 0.08;
      }
      const o = h * 4;
      img.data[o] = rgb[0];
      img.data[o + 1] = rgb[1];
      img.data[o + 2] = rgb[2];
      img.data[o + 3] = Math.round(a * 255);
    }
    g.putImageData(img, 0, 0);
    c.strip = cv;
    c.stripVersion = themeVersion;
    return cv;
  }

  let dayCache = null;
  function dayTicks() {
    if (!dayCache) {
      dayCache = [];
      for (let h = 0; h < model.H; h++) if (isMidnight(h)) dayCache.push(h);
    }
    return dayCache;
  }

  function tlDraw() {
    if (!model || !TL.width) return;
    const w = TL.width;
    const vh = TL.viewH;
    const ctx = setupCanvas(ui.tlCanvas, w, vh, true);
    const g = tlGeom();
    const st = ui.tlScroll.scrollTop;

    dayTicks().forEach((d) => vline(ctx, crisp(g.X(d)), 0, vh, C.grid));

    const first = Math.max(0, Math.floor(st / TL.rowH));
    const last = Math.min(TL.rows.length - 1, Math.floor((st + vh) / TL.rowH));
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    for (let i = first; i <= last; i++) {
      const c = TL.rows[i];
      const top = i * TL.rowH - st;
      const mid = top + TL.rowH / 2;
      const sel = c.id === state.selectedId;
      if (sel) {
        ctx.fillStyle = C.selBg;
        ctx.fillRect(0, top, w, TL.rowH);
        ctx.fillStyle = C.accent;
        ctx.fillRect(0, top, 3, TL.rowH);
      } else if (i === TL.hover) {
        ctx.fillStyle = C.hoverBg;
        ctx.fillRect(0, top, w, TL.rowH);
      }

      ctx.font = '650 12px ' + FONT;
      ctx.fillStyle = sel ? C.text : C.text2;
      ctx.fillText(c.label, 10, mid);
      const idW = ctx.measureText(c.label).width;
      ctx.font = (sel ? '600 ' : '') + '12px ' + FONT;
      ctx.fillStyle = sel ? C.text : C.text3;
      ctx.fillText(fitText(ctx, c.port || c.proto || '', TL.gutter - 22 - idW), 16 + idW, mid);

      const bx0 = g.X(c.firstH);
      const bx1 = g.X(c.lastH + 1);
      const by = mid - TL.barH / 2;
      ctx.save();
      roundRectPath(ctx, bx0, by, Math.max(3, bx1 - bx0), TL.barH, 3);
      ctx.clip();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(barStrip(c), c.firstH, 0, c.lastH - c.firstH + 1, 1, bx0, by, Math.max(3, bx1 - bx0), TL.barH);
      ctx.restore();

      const dx = Math.round(g.X(c.det + 1));
      ctx.fillStyle = C.surface;
      ctx.fillRect(dx - 2, by - 4, 4, TL.barH + 8);
      ctx.fillStyle = C.cursor;
      ctx.fillRect(dx - 1, by - 4, 2, TL.barH + 8);
    }
    drawCursor(ctx, g.X(state.hour + 1), g.x0, w, 0, vh);
  }

  function tlDrawAxis() {
    if (!model || !TL.width) return;
    const w = TL.width;
    const h = TL.axisH;
    const ctx = setupCanvas(ui.tlAxis, w, h, true);
    const g = tlGeom();
    const H = model.H;

    ctx.fillStyle = C.axis;
    for (let hh = 0; hh <= H; hh++) {
      if (!isMidnight(hh) && ((absHour(hh) % 6) + 6) % 6 === 0) ctx.fillRect(Math.round(g.X(hh)), h - 4, 1, 4);
    }
    ctx.font = '11px ' + FONT;
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    const dayW = 24 * g.k;
    dayTicks().forEach((d) => {
      const x = Math.round(g.X(d));
      ctx.fillStyle = C.axis;
      ctx.fillRect(x, h - 10, 1, 10);
      const label = dayW >= 52 ? fmtDay(d) : (model.t0 === null ? String(Math.floor(d / 24) + 1) : String(dateAt(d).getUTCDate()));
      if (x + 4 + ctx.measureText(label).width <= g.x0 + g.pw) {
        ctx.fillStyle = C.text3;
        ctx.fillText(label, x + 4, h - 12);
      }
    });
    ctx.fillStyle = C.text3;
    if (model.t0 !== null) ctx.fillText('UTC', 10, h - 12);

    const cx = g.X(state.hour + 1);
    drawCursor(ctx, cx, g.x0, w, 0, h);
    const tag = fmtClock(state.hour + 1);
    ctx.font = '650 11px ' + FONT;
    const tw = ctx.measureText(tag).width + 12;
    const tx = clamp(cx - tw / 2, g.x0 - 6, w - tw - 1);
    ctx.fillStyle = C.cursor;
    roundRectPath(ctx, tx, 2, tw, 16, 4);
    ctx.fill();
    ctx.fillStyle = C.surface;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(tag, tx + tw / 2, 10.5);
  }

  function tlReveal(id) {
    const i = TL.rows.findIndex((c) => c.id === id);
    if (i < 0) return;
    const sc = ui.tlScroll;
    const top = i * TL.rowH;
    if (top < sc.scrollTop) sc.scrollTop = top;
    else if (top + TL.rowH > sc.scrollTop + TL.viewH) sc.scrollTop = top + TL.rowH - TL.viewH;
  }

  function tlRowAt(clientY) {
    const r = ui.tlCanvas.getBoundingClientRect();
    const i = Math.floor((clientY - r.top + ui.tlScroll.scrollTop) / TL.rowH);
    return i >= 0 && i < TL.rows.length ? i : -1;
  }

  function tlHourAt(clientX, canvas) {
    const r = canvas.getBoundingClientRect();
    const g = tlGeom();
    return (clientX - r.left - g.x0) / g.k;
  }

  function onTlMove(e) {
    const i = tlRowAt(e.clientY);
    if (i !== TL.hover) { TL.hover = i; tlDraw(); }
    if (i < 0) { hideTip(); return; }
    const c = TL.rows[i];
    const hf = tlHourAt(e.clientX, ui.tlCanvas);
    const h = Math.floor(hf);
    const kids = [
      tipTitle(c.label + ' ' + DOT + ' ' + c.portLabel),
      tipLine(c.tool || 'Unattributed tool'),
    ];
    if (h >= c.firstH && h <= c.lastH) {
      kids.push(tipValue(colorOf(c), fmtInt(c.activity[h]), plural(c.activity[h], 'bot', 'bots') + ' active'));
      kids.push(tipLine(fmtHourRange(h) + (h > state.hour ? ' (after the cursor)' : '')));
    } else {
      kids.push(tipLine('Active ' + fmtWhen(c.firstH) + ' ' + NDASH + ' ' + fmtWhen(c.lastH + 1) + tz()));
    }
    kids.push(tipLine(orchWord(c) + ', z ' + fmtZ(c.z) +
      ' ' + DOT + ' first report ' + fmtWhen(c.det)));
    showTip(e.clientX, e.clientY, kids);
  }

  function onTlLeave() {
    if (TL.hover !== -1) { TL.hover = -1; tlDraw(); }
    hideTip();
  }

  function onTlClick(e) {
    const i = tlRowAt(e.clientY);
    if (i < 0) return;
    select(TL.rows[i].id, { revealList: true, revealDetail: true });
  }

  let axisDrag = false;
  function axisToHour(e) {
    const t = tlHourAt(e.clientX, ui.tlAxis);
    setHour(Math.round(t) - 1);
  }

  // --------------------------------------------------------- hour tracks
  function drawTrack(canvas, key) {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight || 20;
    if (!w) return;
    const ctx = setupCanvas(canvas, w, h, false);
    const st = model.trackStats[key];
    const H = model.H;
    const inset = THUMB / 2;
    const pw = w - 2 * inset;
    const X = (i) => inset + (H > 1 ? i / (H - 1) : 0.5) * pw;
    hline(ctx, inset, w - inset, h - 0.5, C.grid);
    if (!st) return;
    const Y = (v) => 2 + (1 - (v - st.lo) / (st.hi - st.lo)) * (h - 4);
    const end = state.hour;

    let seg = [];
    const flush = () => {
      if (!seg.length) return;
      ctx.beginPath();
      ctx.moveTo(X(seg[0][0]), h);
      seg.forEach((p) => ctx.lineTo(X(p[0]), Y(p[1])));
      ctx.lineTo(X(seg[seg.length - 1][0]), h);
      ctx.closePath();
      ctx.fillStyle = alpha(C.neutral, 0.16);
      ctx.fill();
      ctx.beginPath();
      seg.forEach((p, j) => (j ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
      ctx.strokeStyle = C.neutral;
      ctx.lineWidth = 1.25;
      ctx.lineJoin = 'round';
      ctx.stroke();
      seg = [];
    };
    for (let i = 0; i <= end; i++) {
      const v = model.hourly[i][key];
      if (v === null) flush();
      else seg.push([i, v]);
    }
    flush();

    const last = model.hourly[end][key];
    if (last !== null) {
      ctx.beginPath();
      ctx.arc(X(end), Y(last), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = C.surface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(X(end), Y(last), 2.2, 0, Math.PI * 2);
      ctx.fillStyle = C.text;
      ctx.fill();
    }
  }

  function trackHourAt(canvas, clientX) {
    const r = canvas.getBoundingClientRect();
    const pw = r.width - THUMB;
    const H = model.H;
    return clamp(Math.round((clientX - r.left - THUMB / 2) / (pw || 1) * (H - 1)), 0, H - 1);
  }

  function onTrackMove(e) {
    const canvas = e.currentTarget;
    const h = trackHourAt(canvas, e.clientX);
    if (canvas.hasPointerCapture && canvas.hasPointerCapture(e.pointerId)) {
      setHour(h);
      return;
    }
    const row = model.hourly[h];
    const kids = [tipTitle(fmtHourRange(h))];
    if (h > state.hour) {
      kids.push(tipLine('After the cursor. Click to move the cursor here.'));
    } else {
      kids.push(tipValue(C.neutral, fmtInt(row.records), 'records'));
      kids.push(tipValue(C.neutral, fmtInt(row.sources), 'unique sources'));
    }
    showTip(e.clientX, e.clientY, kids);
  }

  function onTrackDown(e) {
    if (e.button !== 0) return;
    const canvas = e.currentTarget;
    hideTip();
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    setHour(trackHourAt(canvas, e.clientX));
  }

  function onTrackUp(e) {
    const canvas = e.currentTarget;
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    writeHashSoon();
  }

  // ------------------------------------------------------------- tiles
  function renderTiles() {
    const rep = model.campaigns.filter(isReported);
    const meta = model.meta;
    const totalCamp = meta.nCampaigns !== null ? meta.nCampaigns : model.campaigns.length;
    const totalOrch = meta.nOrch !== null ? meta.nOrch : model.nOrchData;
    setText(ui.tProfiles, fmtInt(meta.nProfiles));
    setText(ui.tCampaigns, fmtInt(rep.length));
    setText(ui.tCampaignsSub, 'of ' + fmtInt(totalCamp) + ' in total');
    ui.tCampaignsSub.title = model.campaigns.length < totalCamp
      ? 'This demo includes ' + fmtInt(model.campaigns.length) + ' of the ' + fmtInt(totalCamp) + ' campaigns.' : '';
    setText(ui.tOrch, fmtInt(rep.filter((c) => c.orch).length));
    setText(ui.tOrchSub, 'of ' + fmtInt(totalOrch) + ' in total');
    ui.tOrchSub.title = 'Campaigns that pass the calibrated coordination test';
    setText(ui.tBots, fmtInt(rep.reduce((s, c) => s + c.bots, 0)));
    setText(ui.tBotsSub, 'of ' + fmtInt(model.totalBots) + ' in total');
  }

  // --------------------------------------------------------- controls
  function updateControls() {
    const h = state.hour;
    const H = model.H;
    setText(ui.timeMain, fmtHourRange(h));
    const n = reportedIds.size;
    const reportedText = n + ' ' + plural(n, 'campaign', 'campaigns') + ' reported so far';
    setText(ui.timeSub, model.t0 === null ? reportedText
      : 'Hour ' + (h + 1) + ' of ' + H + ' ' + DOT + ' ' + reportedText);
    if (String(h) !== ui.scrubber.value) ui.scrubber.value = String(h);
    ui.scrubber.setAttribute('aria-valuetext', fmtHourRange(h));
    ui.scrubber.style.setProperty('--pct', (H > 1 ? (h / (H - 1)) * 100 : 100).toFixed(2) + '%');
    const row = model.hourly[h];
    setText(ui.valRecords, fmtCompact(row.records));
    setText(ui.valSources, fmtCompact(row.sources));
    setText(ui.valHour, (h + 1) + '/' + H);
    drawTrack(ui.sparkRecords, 'records');
    drawTrack(ui.sparkSources, 'sources');
  }

  function updateFeed() {
    let best = null;
    model.campaigns.forEach((c) => {
      if (c.det > state.hour) return;
      if (!best || c.det > best.det || (c.det === best.det && c.bots > best.bots)) best = c;
    });
    if (!best) {
      setText(ui.feed, 'No campaign reported yet. Press Play to watch the engine report them.');
      return;
    }
    const same = model.campaigns.filter((c) => c.det === best.det).length;
    setText(ui.feed, 'Latest report: ' + best.label + ' (' + best.portLabel + ', ' +
      (best.tool || 'unattributed') + ', ' + fmtInt(best.bots) + ' ' + plural(best.bots, 'bot', 'bots') + ') in the ' +
      fmtWhen(best.det) + ' batch' + (same > 1 ? ', with ' + (same - 1) + ' more' : ''));
  }

  function updatePlayButton() {
    ui.playBtn.classList.toggle('is-playing', state.playing);
    ui.playBtn.setAttribute('aria-label', state.playing ? 'Pause replay' : 'Play replay');
    setText(ui.playLabel, state.playing ? 'Pause' : (state.hour >= model.H - 1 ? 'Replay' : 'Play'));
  }

  function updateSpeedButtons() {
    ui.speedBtns.forEach((b) => {
      b.setAttribute('aria-pressed', Number(b.getAttribute('data-speed')) === state.speed ? 'true' : 'false');
    });
  }

  function updateSortButton() {
    const asc = state.sortDir === 1;
    ui.sortDir.setAttribute('data-dir', asc ? 'asc' : 'desc');
    ui.sortDir.setAttribute('aria-label', asc ? 'Sorted ascending; switch to descending' : 'Sorted descending; switch to ascending');
    ui.sortDir.title = asc ? 'Ascending' : 'Descending';
  }

  // --------------------------------------------------------- detail view
  // id is undefined until the first render, null while nothing is selected
  let detail = { id: undefined, reported: null, scatter: null, activity: null };

  function currentCampaign() {
    return state.selectedId !== null ? model.byId.get(state.selectedId) || null : null;
  }

  function section(title, sub, kids) {
    const s = el('section', { class: 'd-section' }, [el('h3', { text: title })]);
    if (sub) s.appendChild(el('p', { class: 'd-sub', text: sub }));
    return append(s, kids);
  }

  function emptyBlock(title, text, extra) {
    return el('div', { class: 'empty' }, [el('strong', { text: title }), el('p', { text: text }), extra || null]);
  }

  function renderDetail() {
    const body = ui.detailBody;
    hideTip();
    clear(body);
    const c = currentCampaign();
    detail = { id: c ? c.id : null, reported: c ? isReported(c) : null, scatter: null, activity: null };
    setText(ui.detailTitle, c ? 'Campaign ' + c.label : 'Campaign detail');

    if (!c) {
      body.appendChild(emptyBlock('No campaign selected',
        'Choose a campaign in the list, or click a bar in the timeline, to see its signature, probing strategy, activity, origin and AI assessment.'));
      return;
    }
    if (!isReported(c)) {
      const btn = el('button', { type: 'button', class: 'btn', text: 'Jump to its first report' });
      btn.addEventListener('click', () => {
        setPlaying(false);
        setHour(c.det);
        writeHashSoon();
      });
      body.appendChild(emptyBlock(c.label + ' is not reported yet',
        'At this point of the replay the online engine has not reported this campaign. It is first reported in the ' +
        fmtWhen(c.det) + tz() + ' batch.', btn));
      return;
    }

    const lat = c.det - c.firstH;
    append(body, [
      el('div', { class: 'd-head' }, [
        el('div', { class: 'd-head-main' }, [
          el('div', { class: 'd-port', text: c.portLabel }),
          el('div', { class: 'd-tool', text: c.tool || 'Unattributed tool' }),
        ]),
        el('div', { class: 'd-tags' }, [orchTag(c), sevTag(c)]),
      ]),
      el('p', {
        class: 'd-report',
        text: 'First reported in the ' + fmtWhen(c.det) + tz() + ' batch, ' +
          (lat <= 0 ? 'within its first hour of activity.' : lat + ' ' + plural(lat, 'hour', 'hours') + ' after its first activity.'),
      }),
      factsBlock(c),
      signatureSection(c),
    ]);

    detail.scatter = scatterSection(c);
    detail.activity = activitySection(c);
    append(body, [detail.scatter.node, detail.activity.node, countrySection(c), openjevSection(c)]);
    drawDetailCharts();
  }

  function updateDetail() {
    const c = currentCampaign();
    if (!c) {
      if (detail.id !== null) renderDetail();
      return;
    }
    if (c.id !== detail.id || isReported(c) !== detail.reported) {
      renderDetail();
      return;
    }
    drawDetailCharts();
  }

  function fact(label, value, sub) {
    return el('div', { class: 'fact' }, [
      el('div', { class: 'fact-label', text: label }),
      el('div', { class: 'fact-value', text: value }),
      el('div', { class: 'fact-sub', text: sub, title: sub }),
    ]);
  }

  function factsBlock(c) {
    const nC = c.countries.length;
    const origin = (c.asns !== null ? fmtInt(c.asns) + ' ' + plural(c.asns, 'ASN', 'ASNs') : 'ASNs n/a') +
      ', ' + nC + ' ' + plural(nC, 'country', 'countries');
    return el('div', { class: 'facts' }, [
      fact('Bots', fmtInt(c.bots), origin),
      fact('Synchrony z', fmtZ(c.z), c.orch ? 'passes the coordination test' : (c.undet ? 'too few peers on this port to test' : 'no coordination signal')),
      fact('Coverage', fmtPct(c.coverage), 'of the monitored block'),
      fact('Rate', fmtRate(c.pph), 'packets / h per bot (median)'),
    ]);
  }

  function sigRow(label, value, mono) {
    const td = el('td', { class: value === null ? 'na' : (mono ? 'mono' : null), text: value === null ? DASH : value });
    return el('tr', null, [el('th', { scope: 'row', text: label }), td]);
  }

  function signatureSection(c) {
    const dur = c.lastH - c.firstH + 1;
    const rows = [
      ['Target port(s)', c.port, true],
      ['Protocol', c.proto, true],
      ['Best attribution', c.tool, false],
      ['Probing strategy', c.strategy, false],
      ['Rate', c.pph !== null ? fmtRate(c.pph) + ' packets / h per bot (median, at the telescope)' : null, false],
      ['Coverage', c.coverage !== null ? fmtPct(c.coverage) + ' of the monitored block' : null, false],
      ['Source ASNs', c.asns !== null ? fmtInt(c.asns) : null, false],
      ['Active', fmtWhen(c.firstH) + ' ' + NDASH + ' ' + fmtWhen(c.lastH + 1) + tz() + ' (' + dur + ' h)', false],
      ['Peak', c.peakH >= 0 ? fmtInt(c.maxAct) + ' ' + plural(c.maxAct, 'bot', 'bots') + ' in the ' + fmtWhen(c.peakH) + ' hour' : null, false],
    ];
    const general = el('tbody');
    rows.forEach((r) => general.appendChild(sigRow(r[0], r[1], r[2])));
    const fpBody = el('tbody', null, [el('tr', { class: 'group' }, [
      el('th', { scope: 'rowgroup', colspan: '2', text: 'Header fingerprint (from sampled packets)' }),
    ])]);
    c.fingerprint.forEach((f) => fpBody.appendChild(sigRow(f[0], f[1], true)));
    const table = el('table', { class: 'sig' }, [
      el('caption', { class: 'sr-only', text: 'Signature of campaign ' + c.label }), general, fpBody,
    ]);
    return section('Signature', null, table);
  }

  // probing-strategy scatter
  function scatterSection(c) {
    const canvas = el('canvas', {
      role: 'img',
      'aria-label': 'Probing strategy of ' + c.label + ': ' + c.points.length +
        ' sampled destinations plotted by time and relative position in the monitored block' +
        (c.strategy ? '; strategy ' + c.strategy : '') + '.',
    });
    const foot = el('p', { class: 'chart-foot' });
    const sub = (c.strategy ? capFirst(c.strategy) + '. ' : '') +
      'Each dot is a sampled destination: time on x, relative position inside the monitored block on y. No addresses are shown.';
    const node = section('Probing strategy', sub, [el('div', { class: 'chart' }, [canvas]), foot]);
    const sc = { node, canvas, foot, c, geom: null };
    canvas.addEventListener('pointermove', (e) => onScatterMove(e, sc));
    canvas.addEventListener('pointerleave', hideTip);
    return sc;
  }

  function drawScatter(sc) {
    const canvas = sc.canvas;
    const c = sc.c;
    const w = canvas.clientWidth;
    if (!w) return;
    const h = w < 360 ? 196 : 224;
    const ctx = setupCanvas(canvas, w, h, false);
    const m = { l: 40, r: 8, t: 8, b: 24 };
    const pw = Math.max(10, w - m.l - m.r);
    const ph = h - m.t - m.b;
    const d = spanDomain(c);
    const X = (t) => m.l + ((t - d[0]) / (d[1] - d[0])) * pw;
    const Y = (u) => m.t + (1 - u) * ph;

    ctx.font = '11px ' + FONT;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [0.25, 0.5, 0.75, 1].forEach((u) => {
      const y = crisp(Y(u));
      hline(ctx, m.l, m.l + pw, y, C.grid);
      ctx.fillStyle = C.text3;
      ctx.fillText(Math.round(u * 100) + '%', m.l - 6, y);
    });
    ctx.fillStyle = C.text3;
    ctx.fillText('0%', m.l - 6, Y(0));
    drawTimeAxis(ctx, d[0], d[1], X, m.l, m.l + pw, m.t, m.t + ph);

    const cut = state.hour + 1;
    let seen = 0;
    ctx.fillStyle = alpha(colorOf(c), 0.78);
    ctx.beginPath();
    c.points.forEach((p) => {
      const x = X(p[0]);
      const y = Y(p[1]);
      ctx.moveTo(x + 2.2, y);
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      if (p[0] < cut) seen++;
    });
    ctx.fill();
    drawCursor(ctx, X(cut), m.l, m.l + pw, m.t, m.t + ph);

    sc.geom = { X, Y, m, pw, ph };
    setText(sc.foot, c.points.length
      ? fmtInt(seen) + ' of ' + fmtInt(c.points.length) + ' sampled destinations observed by the cursor; later samples are dimmed.'
      : 'No sampled destinations are available for this campaign.');
  }

  function onScatterMove(e, sc) {
    if (!sc.geom || !sc.c.points.length) return;
    const r = sc.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    let best = null;
    let bestD = 12 * 12;
    sc.c.points.forEach((p) => {
      const dx = sc.geom.X(p[0]) - x;
      const dy = sc.geom.Y(p[1]) - y;
      const dd = dx * dx + dy * dy;
      if (dd < bestD) { bestD = dd; best = p; }
    });
    if (!best) { hideTip(); return; }
    const kids = [
      tipTitle('Sampled destination'),
      tipValue(colorOf(sc.c), fmtPct(best[1]), 'into the monitored block'),
      tipLine(fmtInstant(best[0]) + tz() + (best[0] >= state.hour + 1 ? ' (after the cursor)' : '')),
    ];
    showTip(e.clientX, e.clientY, kids);
  }

  // activity chart
  function activitySection(c) {
    const canvas = el('canvas', {
      role: 'img',
      'aria-label': 'Bots active per hour for ' + c.label + '. Peak ' + fmtInt(c.maxAct) + ' ' + plural(c.maxAct, 'bot', 'bots') +
        (c.peakH >= 0 ? ' in the ' + fmtWhen(c.peakH) + tz() + ' hour' : '') + '. Hourly values are in the table below.',
    });
    const details = el('details', { class: 'data-table' }, [el('summary', { text: 'Show hourly values' })]);
    details.addEventListener('toggle', () => {
      if (!details.open || details.querySelector('table')) return;
      const tbody = el('tbody');
      for (let h = c.firstH; h <= c.lastH; h++) {
        tbody.appendChild(el('tr', null, [el('td', { text: fmtWhen(h) }), el('td', { text: fmtInt(c.activity[h]) })]));
      }
      details.appendChild(el('div', { class: 'table-wrap' }, [
        el('table', null, [
          el('thead', null, [el('tr', null, [el('th', { scope: 'col', text: model.t0 === null ? 'Hour' : 'Hour (UTC)' }), el('th', { scope: 'col', text: 'Bots active' })])]),
          tbody,
        ]),
      ]));
    });
    const node = section('Activity', 'Bots active per hour over the campaign\u2019s lifetime.', [
      el('div', { class: 'chart' }, [canvas]), details,
    ]);
    const ac = { node, canvas, c, geom: null, hover: null };
    canvas.addEventListener('pointermove', (e) => onActivityMove(e, ac));
    canvas.addEventListener('pointerleave', () => {
      ac.hover = null;
      drawActivity(ac);
      hideTip();
    });
    return ac;
  }

  function drawActivity(ac) {
    const canvas = ac.canvas;
    const c = ac.c;
    const w = canvas.clientWidth;
    if (!w) return;
    const h = 150;
    const ctx = setupCanvas(canvas, w, h, false);
    const m = { l: 40, r: 8, t: 10, b: 24 };
    const pw = Math.max(10, w - m.l - m.r);
    const ph = h - m.t - m.b;
    const d = spanDomain(c);
    const yMax = niceMax(c.maxAct);
    const X = (t) => m.l + ((t - d[0]) / (d[1] - d[0])) * pw;
    const Y = (v) => m.t + (1 - v / yMax) * ph;
    const color = colorOf(c);

    ctx.font = '11px ' + FONT;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [0.5, 1].forEach((f) => {
      const y = crisp(Y(yMax * f));
      hline(ctx, m.l, m.l + pw, y, C.grid);
      ctx.fillStyle = C.text3;
      ctx.fillText(fmtCompact(yMax * f), m.l - 6, y);
    });
    ctx.fillStyle = C.text3;
    ctx.fillText('0', m.l - 6, Y(0));
    drawTimeAxis(ctx, d[0], d[1], X, m.l, m.l + pw, m.t, m.t + ph);

    const h0 = Math.max(0, Math.floor(d[0]));
    const h1 = Math.min(model.H - 1, Math.ceil(d[1]) - 1);
    ctx.beginPath();
    ctx.moveTo(X(h0 + 0.5), Y(0));
    for (let hh = h0; hh <= h1; hh++) ctx.lineTo(X(hh + 0.5), Y(c.activity[hh]));
    ctx.lineTo(X(h1 + 0.5), Y(0));
    ctx.closePath();
    ctx.fillStyle = alpha(color, 0.14);
    ctx.fill();
    ctx.beginPath();
    for (let hh = h0; hh <= h1; hh++) {
      if (hh === h0) ctx.moveTo(X(hh + 0.5), Y(c.activity[hh]));
      else ctx.lineTo(X(hh + 0.5), Y(c.activity[hh]));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();

    drawCursor(ctx, X(state.hour + 1), m.l, m.l + pw, m.t, m.t + ph);

    if (ac.hover !== null && ac.hover >= h0 && ac.hover <= h1) {
      const x = crisp(X(ac.hover + 0.5));
      vline(ctx, x, m.t, m.t + ph, C.text3);
      const y = Y(c.activity[ac.hover]);
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = C.surface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
    ac.geom = { X, m, pw, d, h0, h1 };
  }

  function onActivityMove(e, ac) {
    if (!ac.geom) return;
    const r = ac.canvas.getBoundingClientRect();
    const g = ac.geom;
    const t = g.d[0] + ((e.clientX - r.left - g.m.l) / g.pw) * (g.d[1] - g.d[0]);
    const hh = clamp(Math.floor(t), g.h0, g.h1);
    if (hh !== ac.hover) {
      ac.hover = hh;
      drawActivity(ac);
    }
    const v = ac.c.activity[hh];
    showTip(e.clientX, e.clientY, [
      tipValue(colorOf(ac.c), fmtInt(v), plural(v, 'bot', 'bots') + ' active'),
      tipLine(fmtHourRange(hh) + (hh > state.hour ? ' (after the cursor)' : '')),
    ]);
  }

  function drawDetailCharts() {
    if (detail.scatter) drawScatter(detail.scatter);
    if (detail.activity) drawActivity(detail.activity);
  }

  // origin
  function countrySection(c) {
    if (!c.countries.length) {
      return section('Origin', null, el('p', { class: 'd-sub', text: 'No country data is available for this campaign.' }));
    }
    const top = c.countries.slice(0, 8);
    const max = top[0][1];
    const list = el('ol', { class: 'hbars ' + (c.orch ? 'is-orch' : 'is-indep') });
    top.forEach((entry) => {
      const cc = entry[0];
      const n = entry[1];
      const name = countryName(cc);
      const fill = el('span', { class: 'hbar-fill' });
      fill.style.width = Math.max(1, (n / max) * 100).toFixed(1) + '%';
      list.appendChild(el('li', { class: 'hbar' }, [
        el('span', { class: 'hbar-label', title: name + ' (' + cc + ')' }, [name, el('span', { class: 'cc', text: ' ' + cc })]),
        el('span', { class: 'hbar-track', 'aria-hidden': 'true' }, [fill]),
        el('span', { class: 'hbar-val', text: fmtInt(n) + ' ' + DOT + ' ' + fmtShare(n / (c.countrySum || 1)) }),
      ]));
    });
    const kids = [list];
    const rest = c.countries.slice(8);
    if (rest.length) {
      const bots = rest.reduce((s, x) => s + x[1], 0);
      kids.push(el('p', {
        class: 'more-note',
        text: '+ ' + rest.length + ' more ' + plural(rest.length, 'country', 'countries') + ' with ' + fmtInt(bots) + ' ' + plural(bots, 'bot', 'bots') + '.',
      }));
    }
    const sub = 'Bots by country of origin, top ' + top.length + ' of ' + c.countries.length +
      (c.asns !== null ? ', spread over ' + fmtInt(c.asns) + ' ' + plural(c.asns, 'ASN', 'ASNs') : '') + '.';
    return section('Origin', sub, kids);
  }

  // openjev
  function probRow(label, p, isTop, isRest) {
    const fill = el('span', { class: 'hbar-fill' });
    fill.style.width = (clamp(p, 0, 1) * 100).toFixed(1) + '%';
    return el('li', { class: 'hbar' + (isTop ? ' is-top' : '') + (isRest ? ' is-rest' : '') }, [
      el('span', { class: 'hbar-label', text: label, title: label }),
      el('span', { class: 'hbar-track', 'aria-hidden': 'true' }, [fill]),
      el('span', { class: 'hbar-val', text: fmtProb(p) }),
    ]);
  }

  function distBlock(title, entries) {
    const top = entries.slice(0, 4);
    const rest = entries.slice(4);
    const restSum = rest.reduce((s, x) => s + x[1], 0);
    const list = el('ol', { class: 'hbars hbars-prob' });
    top.forEach((x, i) => list.appendChild(probRow(x[0], x[1], i === 0, false)));
    if (rest.length && restSum >= 0.005) {
      list.appendChild(probRow(rest.length + ' other ' + plural(rest.length, 'option', 'options'), restSum, false, true));
    }
    return el('div', { class: 'oj-block' }, [el('h4', { text: title }), list]);
  }

  function coordBlock(c, p) {
    const fill = el('span', { class: 'meter-fill' });
    fill.style.width = (p * 100).toFixed(1) + '%';
    const agree = (p >= 0.5) === c.orch;
    return el('div', { class: 'oj-block' }, [
      el('h4', { text: 'Coordinated? (probability of yes)' }),
      el('div', { class: 'hbar is-top' }, [
        el('span', { class: 'hbar-label', text: 'Yes' }),
        el('span', { class: 'meter', 'aria-hidden': 'true' }, [fill, el('span', { class: 'meter-mark' })]),
        el('span', { class: 'hbar-val', text: fmtProb(p) }),
      ]),
      c.undet
        ? el('p', { class: 'oj-verdict', text: 'The statistical coordination test could not be ' +
            'applied: the port has too few other sources to serve as peers. The tick marks 0.5.' })
        : el('p', { class: 'oj-verdict' }, [
          el('strong', { text: agree ? 'Agrees' : 'Disagrees' }),
          ' with the statistical coordination test, which marks this campaign ' +
            (c.orch ? 'orchestrated' : 'independent') + ' (z = ' + fmtZ(c.z) + '). The tick marks 0.5.',
        ]),
    ]);
  }

  function sevBlock(oj) {
    const list = el('ol', { class: 'hbars hbars-prob oj-sev' });
    let arg = 0;
    oj.sev.forEach((p, i) => { if (p > oj.sev[arg]) arg = i; });
    oj.sev.forEach((p, i) => list.appendChild(probRow(SEV_NAMES[i], p, i === arg, false)));
    return el('div', { class: 'oj-block' }, [
      el('h4', { text: 'Severity' }),
      list,
      el('p', {
        class: 'oj-verdict',
        text: 'Expected severity ' + oj.sevExp.toFixed(2) + ' on a 0' + NDASH + '3 scale (closest level: ' +
          SEV_NAMES[oj.sevLevel].toLowerCase() + '). The list sorts by this value.',
      }),
    ]);
  }

  function openjevSection(c) {
    const oj = c.oj;
    const kids = [];
    if (!oj) {
      kids.push(el('p', { class: 'd-sub', text: 'No openjev assessment is available for this campaign.' }));
    } else {
      if (oj.tool.length) kids.push(distBlock('Likely tool or family', oj.tool));
      if (oj.service.length) kids.push(distBlock('Target service', oj.service));
      if (oj.cls.length) kids.push(distBlock('Campaign class', oj.cls));
      if (oj.coordinated !== null) kids.push(coordBlock(c, oj.coordinated));
      if (oj.sev) kids.push(sevBlock(oj));
    }
    kids.push(el('p', {
      class: 'oj-note',
      text: 'These are calibrated probabilities from openjev, an AI decision model that scores every option from its logits. ' +
        'They express model confidence, not ground truth, and give a second opinion beside the statistical tests.',
    }));
    const s = section('AI assessment (openjev)', 'Decision-model outputs for this campaign.', kids);
    s.classList.add('oj');
    return s;
  }

  // --------------------------------------------------------- hour updates
  function setHour(h) {
    const nh = clamp(Math.round(Number(h) || 0), 0, model.H - 1);
    if (nh === state.hour) return;
    const forward = nh > state.hour;
    state.hour = nh;
    onHourChanged(forward);
  }

  function onHourChanged(forward) {
    const prev = reportedIds;
    reportedIds = new Set(model.campaigns.filter(isReported).map((c) => c.id));
    if (forward) {
      const now = Date.now();
      reportedIds.forEach((id) => { if (!prev.has(id)) flashUntil.set(id, now + FLASH_MS); });
    }
    refreshVisible();
    renderList();
    renderTiles();
    tlLayout();
    tlDraw();
    tlDrawAxis();
    updateDetail();
    updateControls();
    updateFeed();
    updatePlayButton();
  }

  function refreshAll() {
    reportedIds = new Set(model.campaigns.filter(isReported).map((c) => c.id));
    refreshVisible();
    renderList();
    renderTiles();
    tlLayout();
    tlDraw();
    tlDrawAxis();
    updateDetail();
    updateControls();
    updateFeed();
    updatePlayButton();
  }

  // --------------------------------------------------------------- replay
  function setPlaying(on) {
    if (on === state.playing) return;
    state.playing = on;
    if (on) {
      if (state.hour >= model.H - 1) setHour(0);
      state.acc = 0;
      state.lastTs = null;
      rafId = window.requestAnimationFrame(tick);
    } else {
      window.cancelAnimationFrame(rafId);
      writeHashSoon();
    }
    updatePlayButton();
  }

  function tick(ts) {
    if (!state.playing) return;
    if (state.lastTs !== null) {
      const dt = Math.min(0.25, Math.max(0, (ts - state.lastTs) / 1000));
      state.acc += dt * state.speed;
      if (state.acc >= 1) {
        const steps = Math.floor(state.acc);
        state.acc -= steps;
        const target = Math.min(model.H - 1, state.hour + steps);
        setHour(target);
        if (target >= model.H - 1) {
          setPlaying(false);
          return;
        }
      }
    }
    state.lastTs = ts;
    rafId = window.requestAnimationFrame(tick);
  }

  // ----------------------------------------------------------- deep links
  let lastHash = '';
  let hashTimer = 0;

  function readHash() {
    const raw = (window.location.hash || '').replace(/^#/, '');
    const out = { c: null, h: null };
    raw.split('&').forEach((part) => {
      const kv = part.split('=');
      if (kv.length !== 2) return;
      const v = finite(decodeURIComponentSafe(kv[1]));
      if (v === null) return;
      if (kv[0] === 'c') out.c = Math.round(v);
      if (kv[0] === 'h') out.h = Math.round(v);
    });
    return out;
  }

  function decodeURIComponentSafe(s) {
    try { return decodeURIComponent(s); } catch (e) { return ''; }
  }

  function writeHash() {
    if (!model) return;
    const parts = [];
    if (state.selectedId !== null) parts.push('c=' + state.selectedId);
    parts.push('h=' + state.hour);
    const hash = '#' + parts.join('&');
    if (hash === window.location.hash) return;
    lastHash = hash;
    try {
      window.history.replaceState(null, '', hash);
    } catch (e) {
      try { window.location.replace(hash); } catch (e2) { /* ignore */ }
    }
  }

  function writeHashSoon() {
    window.clearTimeout(hashTimer);
    hashTimer = window.setTimeout(() => { if (!state.playing) writeHash(); }, 350);
  }

  function onHashChange() {
    if (window.location.hash === lastHash) return;
    const hs = readHash();
    if (hs.h !== null) setHour(hs.h);
    if (hs.c !== null && model.byId.has(hs.c)) select(hs.c, { revealList: true, revealTimeline: true });
  }

  // ------------------------------------------------------------- startup
  function cacheElements() {
    [
      'appTitle', 'telescope', 'windowText', 'synthBadge', 'themeToggle',
      'tProfiles', 'tProfilesSub', 'tCampaigns', 'tCampaignsSub', 'tOrch', 'tOrchSub', 'tBots', 'tBotsSub',
      'controls', 'playBtn', 'playLabel', 'timeMain', 'timeSub', 'feed',
      'sparkRecords', 'sparkSources', 'valRecords', 'valSources', 'scrubber', 'valHour',
      'listCount', 'search', 'sortKey', 'sortDir', 'listBody', 'listEmpty',
      'tlWrap', 'tlAxis', 'tlScroll', 'tlCanvas', 'tlSpacer', 'tlEmpty',
      'detailTitle', 'detailBody', 'metaNote', 'tooltip', 'main',
    ].forEach((id) => { ui[id] = byId(id); });
    ui.list = byId('campaignList');
    ui.chips = Array.prototype.slice.call(doc.querySelectorAll('#chips .chip'));
    ui.speedBtns = Array.prototype.slice.call(doc.querySelectorAll('#speedGroup .seg-btn'));
  }

  function showFatal(message) {
    if (!ui.main) return;
    clear(ui.main);
    ui.main.appendChild(el('div', { class: 'fatal', role: 'alert' }, [
      el('h2', { text: 'The replay data could not be loaded' }),
      el('p', { text: message }),
    ]));
  }

  function renderStatic() {
    const meta = model.meta;
    if (meta.title) setText(ui.appTitle, meta.title);
    setText(ui.telescope, meta.telescope || 'Network telescope');
    let windowText;
    if (model.t0 !== null) {
      const endYear = dateAt(model.H - 1).getUTCFullYear();
      windowText = fmtDayMonth(0) + ' ' + NDASH + ' ' + fmtDayMonth(model.H - 1) + ' ' + endYear +
        ' ' + DOT + ' ' + model.H + ' hourly batches (UTC)';
    } else {
      windowText = model.H + ' hourly batches';
    }
    setText(ui.windowText, windowText);
    ui.synthBadge.hidden = !meta.synthetic;
    if (meta.synthetic && meta.note) ui.synthBadge.title = meta.note;
    if (!meta.anonymized) {
      const anon = doc.querySelector('.badge-anon');
      if (anon) {
        clear(anon);
        anon.classList.add('badge-synthetic');
        append(anon, 'Data file is not marked as anonymized');
      }
    }
    setText(ui.metaNote, meta.note ? 'About the data: ' + meta.note : '');
    ui.scrubber.min = '0';
    ui.scrubber.max = String(model.H - 1);
    ui.scrubber.value = String(state.hour);
    ui.sortKey.value = state.sortKey;
    updateSortButton();
    updateSpeedButtons();
  }

  function bindEvents() {
    ui.playBtn.addEventListener('click', () => setPlaying(!state.playing));
    ui.speedBtns.forEach((b) => b.addEventListener('click', () => {
      state.speed = Number(b.getAttribute('data-speed')) || 1;
      updateSpeedButtons();
    }));
    ui.scrubber.addEventListener('input', () => setHour(Number(ui.scrubber.value)));
    ui.scrubber.addEventListener('change', writeHashSoon);

    [ui.sparkRecords, ui.sparkSources].forEach((cv) => {
      cv.addEventListener('pointerdown', onTrackDown);
      cv.addEventListener('pointermove', onTrackMove);
      cv.addEventListener('pointerup', onTrackUp);
      cv.addEventListener('pointercancel', onTrackUp);
      cv.addEventListener('pointerleave', hideTip);
    });

    ui.themeToggle.addEventListener('click', () => {
      setTheme(state.theme === 'dark' ? 'light' : 'dark', true);
      redrawAll();
    });

    ui.search.addEventListener('input', () => {
      state.query = ui.search.value;
      refreshAll();
    });
    ui.chips.forEach((b) => b.addEventListener('click', () => {
      const f = b.getAttribute('data-filter');
      if (f === 'all') {
        state.orchOnly = false;
        state.proto = null;
      } else if (f === 'orch') {
        state.orchOnly = !state.orchOnly;
      } else {
        state.proto = state.proto === f ? null : f;
      }
      refreshAll();
    }));
    ui.sortKey.addEventListener('change', () => {
      state.sortKey = SORTS[ui.sortKey.value] ? ui.sortKey.value : 'bots';
      state.sortDir = SORTS[state.sortKey].dir;
      updateSortButton();
      refreshAll();
    });
    ui.sortDir.addEventListener('click', () => {
      state.sortDir = -state.sortDir;
      updateSortButton();
      refreshAll();
    });

    ui.list.addEventListener('keydown', onListKey);
    ui.list.addEventListener('click', (e) => {
      const li = e.target.closest ? e.target.closest('li.crow') : null;
      if (li) select(Number(li.getAttribute('data-id')), { focus: true, revealTimeline: true, revealDetail: true });
    });

    ui.tlCanvas.addEventListener('pointermove', onTlMove);
    ui.tlCanvas.addEventListener('pointerleave', onTlLeave);
    ui.tlCanvas.addEventListener('click', onTlClick);
    let scrollQueued = false;
    ui.tlScroll.addEventListener('scroll', () => {
      if (scrollQueued) return;
      scrollQueued = true;
      window.requestAnimationFrame(() => { scrollQueued = false; tlDraw(); });
      hideTip();
    });
    ui.tlAxis.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      axisDrag = true;
      try { ui.tlAxis.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      axisToHour(e);
    });
    ui.tlAxis.addEventListener('pointermove', (e) => { if (axisDrag) axisToHour(e); });
    const endDrag = (e) => {
      if (!axisDrag) return;
      axisDrag = false;
      try { ui.tlAxis.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      writeHashSoon();
    };
    ui.tlAxis.addEventListener('pointerup', endDrag);
    ui.tlAxis.addEventListener('pointercancel', endDrag);

    doc.addEventListener('keydown', onGlobalKey);
    window.addEventListener('hashchange', onHashChange);
    window.addEventListener('blur', hideTip);
  }

  function onGlobalKey(e) {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    const tag = t && t.tagName ? t.tagName : '';
    const typing = tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (t && t.isContentEditable);
    const inList = t && t.closest ? !!t.closest('#campaignList') : false;
    if (e.key === ' ' || e.key === 'Spacebar' || e.key === 'k' || e.key === 'K') {
      if (typing || inList || tag === 'BUTTON' || tag === 'SUMMARY' || tag === 'A') return;
      e.preventDefault();
      setPlaying(!state.playing);
      return;
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      if (typing || inList) return;
      e.preventDefault();
      setHour(state.hour + (e.shiftKey ? 24 : 1) * (e.key === 'ArrowLeft' ? -1 : 1));
      writeHashSoon();
    }
  }

  function measureControls() {
    if (ui.controls) doc.documentElement.style.setProperty('--controls-h', ui.controls.offsetHeight + 'px');
  }

  function redrawAll() {
    measureControls();
    tlLayout();
    tlDraw();
    tlDrawAxis();
    updateControls();
    renderList();
    drawDetailCharts();
  }

  function observeResize() {
    let queued = false;
    const run = () => {
      queued = false;
      redrawAll();
    };
    const schedule = () => {
      if (queued) return;
      queued = true;
      window.requestAnimationFrame(run);
    };
    if (typeof window.ResizeObserver === 'function') {
      const ro = new window.ResizeObserver(schedule);
      [ui.controls, ui.tlWrap, ui.detailBody, ui.listBody].forEach((n) => { if (n) ro.observe(n); });
    }
    window.addEventListener('resize', schedule);
  }

  function init() {
    cacheElements();
    const raw = window.CAMPAIGN_DATA;
    if (!isObj(raw)) {
      showFatal('data.js did not load or did not define window.CAMPAIGN_DATA. Keep index.html, style.css, app.js and data.js together in one folder, then reload the page.');
      return;
    }
    model = buildModel(raw);

    let stored = null;
    try { stored = window.localStorage.getItem(THEME_KEY); } catch (e) { stored = null; }
    setTheme(stored === 'dark' ? 'dark' : 'light', false);

    state.hour = model.H - 1;
    const hs = readHash();
    if (hs.h !== null) state.hour = clamp(hs.h, 0, model.H - 1);
    renderStatic();
    bindEvents();
    measureControls();

    reportedIds = new Set(model.campaigns.filter(isReported).map((c) => c.id));
    refreshVisible();
    if (hs.c !== null && model.byId.has(hs.c)) state.selectedId = hs.c;
    else if (visible.length) state.selectedId = visible[0].id;
    else if (model.campaigns.length) state.selectedId = model.campaigns[0].id;

    refreshAll();
    observeResize();
    if (hs.c !== null) {
      const row = rowEl(state.selectedId);
      if (row) scrollRowIntoView(row);
      tlReveal(state.selectedId);
      tlDraw();
    }
  }

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init);
  else init();
})();
