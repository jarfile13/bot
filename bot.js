const http = require("http");
const fs = require("fs");
const path = require("path");

const TG_TOKEN = "8606506994:AAE-g9SYVmUKzehn2FHaS2GikRU1rOufBFE";
const VOCAB_SIZE = 256;
const HIDDEN = 64;
const EMB = 16;
const LR = 0.03;
const MAX_LEN = 200;
const GEN_LEN = 80;
const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
const MAX_DELTA = 0.5;
const SNAPSHOT_DIR = "/tmp/snapshots";
const MAX_SNAPSHOTS = 5;

const PORT = process.env.PORT || 3000;
http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("ok");
}).listen(PORT, () => console.log("Health check on port " + PORT));

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeWeights(seed = 42) {
  const rnd = mulberry32(seed);
  const randn = () => (rnd() * 2 - 1) * 0.1;
  const W = {
    emb: new Float32Array(VOCAB_SIZE * EMB).map(randn),
    Wx: new Float32Array(HIDDEN * EMB).map(randn),
    Wh: new Float32Array(HIDDEN * HIDDEN).map(randn),
    bh: new Float32Array(HIDDEN),
    Wy: new Float32Array(VOCAB_SIZE * HIDDEN).map(randn),
    by: new Float32Array(VOCAB_SIZE),
  };
  const M = {}, V = {};
  for (const k in W) {
    M[k] = new Float32Array(W[k].length);
    V[k] = new Float32Array(W[k].length);
  }
  return { W, M, V, t: 0, h: new Float32Array(HIDDEN), temp: 1.0 };
}

function softmaxInPlace(logits, temp = 1.0) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.exp((logits[i] - max) / temp);
    sum += logits[i];
  }
  for (let i = 0; i < logits.length; i++) logits[i] /= sum;
}

function forwardStep(W, xIdx, hPrev) {
  const xOff = xIdx * EMB;
  const h = new Float32Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) {
    let s = W.bh[i];
    const wx = i * EMB;
    for (let j = 0; j < EMB; j++) s += W.Wx[wx + j] * W.emb[xOff + j];
    const wh = i * HIDDEN;
    for (let j = 0; j < HIDDEN; j++) s += W.Wh[wh + j] * hPrev[j];
    h[i] = Math.tanh(s);
  }
  const logits = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) {
    let s = W.by[i];
    const wy = i * HIDDEN;
    for (let j = 0; j < HIDDEN; j++) s += W.Wy[wy + j] * h[j];
    logits[i] = s;
  }
  return { h, logits };
}

function trainStep(model, xIdx, yIdx, hPrev) {
  const { W, M, V } = model;
  const { h, logits } = forwardStep(W, xIdx, hPrev);
  softmaxInPlace(logits, 1.0);
  const loss = -Math.log(logits[yIdx] + 1e-9);
  logits[yIdx] -= 1;

  const dWy = new Float32Array(VOCAB_SIZE * HIDDEN);
  const dby = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) {
    const g = logits[i];
    dby[i] = g;
    const off = i * HIDDEN;
    for (let j = 0; j < HIDDEN; j++) dWy[off + j] = g * h[j];
  }
  const dh = new Float32Array(HIDDEN);
  for (let i = 0; i < VOCAB_SIZE; i++) {
    const g = logits[i];
    if (g === 0) continue;
    const off = i * HIDDEN;
    for (let j = 0; j < HIDDEN; j++) dh[j] += g * W.Wy[off + j];
  }
  const dhRaw = new Float32Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) dhRaw[i] = dh[i] * (1 - h[i] * h[i]);

  const dWx = new Float32Array(HIDDEN * EMB);
  const dWh = new Float32Array(HIDDEN * HIDDEN);
  const dbh = new Float32Array(HIDDEN);
  const demb = new Float32Array(EMB);
  const xOff = xIdx * EMB;

  for (let i = 0; i < HIDDEN; i++) {
    const g = dhRaw[i];
    dbh[i] = g;
    const wx = i * EMB;
    for (let j = 0; j < EMB; j++) dWx[wx + j] = g * W.emb[xOff + j];
    const wh = i * HIDDEN;
    for (let j = 0; j < HIDDEN; j++) dWh[wh + j] = g * hPrev[j];
  }
  for (let j = 0; j < EMB; j++) {
    let s = 0;
    for (let i = 0; i < HIDDEN; i++) s += dhRaw[i] * W.Wx[i * EMB + j];
    demb[j] = s;
  }

  model.t++;
  const bc1 = 1 - Math.pow(B1, model.t);
  const bc2 = 1 - Math.pow(B2, model.t);

  const applyAdam = (P, G, m, v) => {
    for (let k = 0; k < P.length; k++) {
      let g = G[k];
      if (!Number.isFinite(g)) g = 0;
      if (g > 1) g = 1; else if (g < -1) g = -1;
      m[k] = B1 * m[k] + (1 - B1) * g;
      v[k] = B2 * v[k] + (1 - B2) * g * g;
      const mh = m[k] / bc1;
      const vh = v[k] / bc2;
      let d = LR * mh / (Math.sqrt(vh) + EPS);
      if (!Number.isFinite(d)) d = 0;
      if (d > MAX_DELTA) d = MAX_DELTA;
      else if (d < -MAX_DELTA) d = -MAX_DELTA;
      P[k] -= d;
    }
  };

  applyAdam(W.Wy, dWy, M.Wy, V.Wy);
  applyAdam(W.by, dby, M.by, V.by);
  applyAdam(W.Wx, dWx, M.Wx, V.Wx);
  applyAdam(W.Wh, dWh, M.Wh, V.Wh);
  applyAdam(W.bh, dbh, M.bh, V.bh);
  for (let j = 0; j < EMB; j++) {
    const k = xOff + j;
    let g = demb[j];
    if (!Number.isFinite(g)) g = 0;
    if (g > 1) g = 1; else if (g < -1) g = -1;
    M.emb[k] = B1 * M.emb[k] + (1 - B1) * g;
    V.emb[k] = B2 * V.emb[k] + (1 - B2) * g * g;
    const mh = M.emb[k] / bc1;
    const vh = V.emb[k] / bc2;
    let d = LR * mh / (Math.sqrt(vh) + EPS);
    if (!Number.isFinite(d)) d = 0;
    if (d > MAX_DELTA) d = MAX_DELTA;
    else if (d < -MAX_DELTA) d = -MAX_DELTA;
    W.emb[k] -= d;
  }
  return { loss, h };
}

function bytesToStr(bytes) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(bytes));
  } catch {
    return null;
  }
}

function generate(W, seedText, length, temp = 1.0) {
  const enc = new TextEncoder();
  let h = new Float32Array(HIDDEN);
  let lastByte = 32;

  for (const b of enc.encode(seedText)) {
    const r = forwardStep(W, b, h);
    h = r.h;
    lastByte = b;
  }

  const outBytes = [];
  for (let i = 0; i < length; i++) {
    const r = forwardStep(W, lastByte, h);
    h = r.h;
    const probs = r.logits;
    softmaxInPlace(probs, temp);
    let p = Math.random(), acc = 0, next = 0;
    for (let k = 0; k < VOCAB_SIZE; k++) {
      acc += probs[k];
      if (p <= acc) { next = k; break; }
    }
    outBytes.push(next);
    lastByte = next;
  }

  let result = seedText;
  let buf = [];
  for (const b of outBytes) {
    buf.push(b);
    const s = bytesToStr(buf);
    if (s !== null) {
      result += s;
      buf = [];
    }
    if (buf.length > 4) buf = [];
  }
  return result;
}

async function tg(method, payload) {
  const url = `https://api.telegram.org/bot${TG_TOKEN}/${method}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

async function send(chatId, text) {
  try {
    return await tg("sendMessage", {
      chat_id: chatId,
      text: String(text).slice(0, 4000),
    });
  } catch (e) {
    console.error("send failed:", e);
  }
}

async function getUpdates(offset) {
  try {
    const url = `https://api.telegram.org/bot${TG_TOKEN}/getUpdates?timeout=30&offset=${offset}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!data.ok) {
      console.error("getUpdates error:", data.description);
      return [];
    }
    return data.result || [];
  } catch (e) {
    console.error("getUpdates failed:", e);
    return [];
  }
}

const models = new Map();
const pendingReset = new Set();

function packModel(m) {
  const parts = [
    m.W.emb, m.W.Wx, m.W.Wh, m.W.bh, m.W.Wy, m.W.by,
    m.M.emb, m.M.Wx, m.M.Wh, m.M.bh, m.M.Wy, m.M.by,
    m.V.emb, m.V.Wx, m.V.Wh, m.V.bh, m.V.Wy, m.V.by,
  ];
  let total = 2;
  for (const arr of parts) total += arr.length;
  const view = new Float32Array(total);
  let off = 0;
  for (const arr of parts) { view.set(arr, off); off += arr.length; }
  view[off++] = m.t;
  view[off] = m.temp || 1.0;
  return Buffer.from(view.buffer);
}

function unpackModel(buf) {
  const view = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
  const m = makeWeights(1);
  const layout = [
    ["W","emb",VOCAB_SIZE*EMB],["W","Wx",HIDDEN*EMB],["W","Wh",HIDDEN*HIDDEN],["W","bh",HIDDEN],["W","Wy",VOCAB_SIZE*HIDDEN],["W","by",VOCAB_SIZE],
    ["M","emb",VOCAB_SIZE*EMB],["M","Wx",HIDDEN*EMB],["M","Wh",HIDDEN*HIDDEN],["M","bh",HIDDEN],["M","Wy",VOCAB_SIZE*HIDDEN],["M","by",VOCAB_SIZE],
    ["V","emb",VOCAB_SIZE*EMB],["V","Wx",HIDDEN*EMB],["V","Wh",HIDDEN*HIDDEN],["V","bh",HIDDEN],["V","Wy",VOCAB_SIZE*HIDDEN],["V","by",VOCAB_SIZE],
  ];
  let off = 0;
  for (const [a,b,len] of layout) { m[a][b].set(view.subarray(off, off+len)); off += len; }
  m.t = view[off++] | 0;
  m.temp = view[off] || 1.0;
  return m;
}

function snapshotDir(chatId) {
  return path.join(SNAPSHOT_DIR, String(chatId));
}

function saveSnapshot(chatId, m, label) {
  try {
    const dir = snapshotDir(chatId);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const files = fs.readdirSync(dir).sort();
    while (files.length >= MAX_SNAPSHOTS) {
      fs.unlinkSync(path.join(dir, files.shift()));
    }
    const fname = String(Date.now()) + "_" + label + ".bin";
    fs.writeFileSync(path.join(dir, fname), packModel(m));
  } catch (e) {
    console.error("snapshot failed:", e.message);
  }
}

function loadLastSnapshot(chatId) {
  try {
    const dir = snapshotDir(chatId);
    if (!fs.existsSync(dir)) return null;
    const files = fs.readdirSync(dir).sort();
    if (files.length === 0) return null;
    const last = files[files.length - 1];
    const buf = fs.readFileSync(path.join(dir, last));
    fs.unlinkSync(path.join(dir, last));
    return { model: unpackModel(buf), name: last };
  } catch (e) {
    console.error("load snapshot failed:", e.message);
    return null;
  }
}

function getModel(chatId) {
  let m = models.get(chatId);
  if (!m) {
    m = makeWeights((Date.now() ^ chatId) & 0xffff);
    models.set(chatId, m);
  }
  return m;
}

async function processUpdate(update) {
  const msg = update.message;
  if (!msg || !msg.text) return;
  const chatId = msg.chat.id;
  const text = msg.text.trim();
  if (!text) return;

  if (pendingReset.has(chatId)) {
    pendingReset.delete(chatId);
    if (text.toLowerCase() === "да" || text === "/yes") {
      models.delete(chatId);
      try { fs.rmSync(snapshotDir(chatId), { recursive: true, force: true }); } catch {}
      return send(chatId, "Модель сброшена. Начинаем заново.");
    }
    return send(chatId, "Отмена. Модель сохранена.");
  }

  if (text === "/start" || text === "/help") {
    return send(chatId,
      "RNN-нейросеть. Команды:\n\n" +
      "Просто отправь текст - учусь на нём.\n" +
      "Чем больше повторов одной фразы, тем лучше запомню.\n\n" +
      "/gen привет - сгенерирую продолжение\n" +
      "/gen 5 привет - 5 вариантов\n" +
      "/top привет - что я хочу сказать дальше\n" +
      "/temp 0.5 - температура (0.3 точно, 1.5 безумно)\n" +
      "/stats - статистика\n" +
      "/undo - откатить последнее обучение\n" +
      "/reset - сбросить модель\n\n" +
      "Если случайно отправил мусор - /undo вернёт модель назад."
    );
  }

  if (text === "/reset") {
    pendingReset.add(chatId);
    return send(chatId, "Точно сбросить модель? Напиши 'да' для подтверждения.");
  }

  if (text === "/undo") {
    const snap = loadLastSnapshot(chatId);
    if (!snap) return send(chatId, "Нечего откатывать. Снапшотов нет.");
    models.set(chatId, snap.model);
    return send(chatId,
      "Откатил последнее обучение.\n" +
      "Шагов теперь: " + snap.model.t + "\n" +
      "Loss вернулся к предыдущему значению."
    );
  }

  if (text === "/stats" || text === "/info") {
    const m = models.get(chatId);
    if (!m) return send(chatId, "Модель пустая. Отправь текст для обучения.");
    const params = VOCAB_SIZE*EMB + HIDDEN*EMB + HIDDEN*HIDDEN + HIDDEN + VOCAB_SIZE*HIDDEN + VOCAB_SIZE;
    let snaps = 0;
    try {
      const dir = snapshotDir(chatId);
      if (fs.existsSync(dir)) snaps = fs.readdirSync(dir).length;
    } catch {}
    return send(chatId,
      "Параметров: " + params.toLocaleString() + "\n" +
      "Шагов: " + m.t + "\n" +
      "Vocab: " + VOCAB_SIZE + "\n" +
      "Hidden: " + HIDDEN + "\n" +
      "Embedding: " + EMB + "\n" +
      "LR: " + LR + "\n" +
      "Temp: " + m.temp.toFixed(2) + "\n" +
      "Снапшотов для отката: " + snaps
    );
  }

  if (text.startsWith("/temp")) {
    const val = parseFloat(text.slice(5).trim());
    if (isNaN(val) || val < 0.1 || val > 3.0) {
      return send(chatId, "Использование: /temp 0.5 (от 0.1 до 3.0)");
    }
    const m = getModel(chatId);
    m.temp = val;
    return send(chatId, "Температура: " + val.toFixed(2) + "\n" +
      (val < 0.7 ? "Точная генерация" : val > 1.3 ? "Безумная генерация" : "Сбалансированная"));
  }

  if (text.startsWith("/top")) {
    const seed = text.slice(4).trim().slice(0, 50);
    const m = models.get(chatId);
    if (!m) return send(chatId, "Сначала обучи меня.");
    const enc = new TextEncoder();
    let h = new Float32Array(HIDDEN);
    let lastByte = 32;
    for (const b of enc.encode(seed)) {
      const r = forwardStep(m.W, b, h);
      h = r.h;
      lastByte = b;
    }
    const r = forwardStep(m.W, lastByte, h);
    const probs = r.logits;
    softmaxInPlace(probs, m.temp);

    const pairs = [];
    for (let i = 0; i < VOCAB_SIZE; i++) pairs.push([i, probs[i]]);
    pairs.sort((a, b) => b[1] - a[1]);

    let out = "После '" + seed + "' топ-8 байт:\n\n";
    for (let i = 0; i < 8; i++) {
      const [byte, prob] = pairs[i];
      let single = "?";
      try {
        single = new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array([byte]));
      } catch { single = "(полбайта)"; }
      if (single === "\n") single = "\\n";
      if (single === " ") single = "[пробел]";
      if (single === "(полбайта)") {
        out += (i + 1) + ". байт " + byte + " (часть символа) — " + (prob * 100).toFixed(1) + "%\n";
      } else {
        out += (i + 1) + ". '" + single + "' (байт " + byte + ") — " + (prob * 100).toFixed(1) + "%\n";
      }
    }

    const topTwo = pairs.slice(0, 2).map(p => p[0]);
    const combined = bytesToStr(topTwo);
    if (combined !== null && combined.length > 0) {
      out += "\nЕсли взять топ-2 байта вместе: '" + combined + "'";
    }
    return send(chatId, out);
  }

  if (text.startsWith("/gen")) {
    const rest = text.slice(4).trim();
    const m = models.get(chatId);
    if (!m) return send(chatId, "Сначала обучи меня текстом.");

    const mNum = rest.match(/^(\d+)\s*(.*)$/);
    if (mNum) {
      const count = Math.min(parseInt(mNum[1]), 10);
      const seed = mNum[2].slice(0, 50) || " ";
      let out = "";
      for (let i = 0; i < count; i++) {
        out += (i + 1) + ") " + generate(m.W, seed, 60, m.temp) + "\n\n";
      }
      return send(chatId, out);
    }

    const out = generate(m.W, rest.slice(0, 50) || " ", GEN_LEN, m.temp);
    return send(chatId, out);
  }

  const trainText = text.slice(0, MAX_LEN);
  const m = getModel(chatId);

  if (m.t > 0) {
    saveSnapshot(chatId, m, "before_train");
  }

  const bytes = new TextEncoder().encode(trainText);
  let totalLoss = 0, steps = 0;
  for (let i = 0; i < bytes.length - 1; i++) {
    const x = bytes[i];
    const y = bytes[i + 1];
    const r = trainStep(m, x, y, m.h);
    m.h = r.h;
    totalLoss += r.loss;
    steps++;
  }
  const avg = (totalLoss / Math.max(steps, 1)).toFixed(3);
  return send(chatId,
    "Обучился на " + steps + " байтах.\n" +
    "Loss: " + avg + "\n" +
    "Всего шагов: " + m.t + "\n" +
    "(если это был мусор — /undo)"
  );
}

async function pollLoop() {
  console.log("Bot polling started");
  let offset = 0;
  while (true) {
    try {
      const updates = await getUpdates(offset);
      for (const u of updates) {
        offset = u.update_id + 1;
        try { await processUpdate(u); }
        catch (e) { console.error("process error:", e); }
      }
    } catch (e) {
      console.error("poll loop error:", e);
      await new Promise(r => setTimeout(r, 3000));
    }
  }
}

pollLoop();
