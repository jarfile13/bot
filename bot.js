const TG_TOKEN = "8606506994:AAEq8_0WBOlsyxbZF2vC0PhZb8ok5lJ6OA4";
const VOCAB_SIZE = 128;
const HIDDEN = 32;
const EMB = 8;
const LR = 0.03;
const MAX_LEN = 300;
const GEN_LEN = 120;
const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
const MAX_DELTA = 0.5;

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
  return { W, M, V, t: 0 };
}

function packModel(m) {
  const parts = [
    m.W.emb, m.W.Wx, m.W.Wh, m.W.bh, m.W.Wy, m.W.by,
    m.M.emb, m.M.Wx, m.M.Wh, m.M.bh, m.M.Wy, m.M.by,
    m.V.emb, m.V.Wx, m.V.Wh, m.V.bh, m.V.Wy, m.V.by,
  ];
  let total = 1;
  for (const arr of parts) total += arr.length;
  const view = new Float32Array(total);
  let off = 0;
  for (const arr of parts) { view.set(arr, off); off += arr.length; }
  view[off] = m.t;
  const bytes = new Uint8Array(view.buffer);
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

function unpackModel(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new Float32Array(bytes.buffer);
  const m = makeWeights(1);
  const layout = [
    ["W","emb",VOCAB_SIZE*EMB],["W","Wx",HIDDEN*EMB],["W","Wh",HIDDEN*HIDDEN],["W","bh",HIDDEN],["W","Wy",VOCAB_SIZE*HIDDEN],["W","by",VOCAB_SIZE],
    ["M","emb",VOCAB_SIZE*EMB],["M","Wx",HIDDEN*EMB],["M","Wh",HIDDEN*HIDDEN],["M","bh",HIDDEN],["M","Wy",VOCAB_SIZE*HIDDEN],["M","by",VOCAB_SIZE],
    ["V","emb",VOCAB_SIZE*EMB],["V","Wx",HIDDEN*EMB],["V","Wh",HIDDEN*HIDDEN],["V","bh",HIDDEN],["V","Wy",VOCAB_SIZE*HIDDEN],["V","by",VOCAB_SIZE],
  ];
  let off = 0;
  for (const [a,b,len] of layout) { m[a][b].set(view.subarray(off, off+len)); off += len; }
  m.t = view[off] | 0;
  return m;
}

function softmaxInPlace(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { logits[i] = Math.exp(logits[i] - max); sum += logits[i]; }
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
  softmaxInPlace(logits);
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

function generate(W, seedText, length) {
  let h = new Float32Array(HIDDEN);
  let lastIdx = 32;
  for (const ch of seedText) {
    const idx = ch.charCodeAt(0) % VOCAB_SIZE;
    const r = forwardStep(W, idx, h);
    h = r.h;
    lastIdx = idx;
  }
  let out = "";
  for (let i = 0; i < length; i++) {
    const r = forwardStep(W, lastIdx, h);
    h = r.h;
    const probs = r.logits;
    softmaxInPlace(probs);
    let p = Math.random(), acc = 0, next = 0;
    for (let k = 0; k < VOCAB_SIZE; k++) {
      acc += probs[k];
      if (p <= acc) { next = k; break; }
    }
    out += String.fromCharCode(next);
    lastIdx = next;
  }
  return out;
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
  const url = `https://api.telegram.org/bot${TG_TOKEN}/getUpdates?timeout=10&offset=${offset}`;
  const res = await fetch(url);
  const data = await res.json();
  return data.result || [];
}

let model = null;
let offset = 0;

async function processUpdate(update) {
  const msg = update.message;
  if (!msg || !msg.text) return;
  const chatId = msg.chat.id;
  const text = msg.text.trim();

  if (text === "/start" || text === "/help") {
    return send(chatId, "Отправь текст - я обучусь. /gen [начало] - сгенерировать. /info - статистика. /reset - сброс.");
  }
  if (text === "/reset") {
    model = null;
    return send(chatId, "Модель сброшена.");
  }
  if (text === "/info") {
    return send(chatId, "Модель: HIDDEN=" + HIDDEN + ", шагов=" + (model ? model.t : 0));
  }
  if (text.startsWith("/gen")) {
    const seed = text.slice(4).trim().slice(0, 100) || " ";
    if (!model) return send(chatId, "Сначала обучи меня текстом.");
    const out = generate(model.W, seed, GEN_LEN);
    return send(chatId, seed + out);
  }

  const trainText = text.slice(0, MAX_LEN);
  if (!model) model = makeWeights(Date.now() & 0xffff);
  if (!model.h) model.h = new Float32Array(HIDDEN);
  let totalLoss = 0, steps = 0;
  for (let i = 0; i < trainText.length - 1; i++) {
    const x = trainText.charCodeAt(i) % VOCAB_SIZE;
    const y = trainText.charCodeAt(i + 1) % VOCAB_SIZE;
    const r = trainStep(model, x, y, model.h);
    model.h = r.h;
    totalLoss += r.loss;
    steps++;
  }
  const avg = (totalLoss / Math.max(steps, 1)).toFixed(3);
  return send(chatId, "Обучился на " + steps + " символах. Loss: " + avg + ". Всего шагов: " + model.t);
}

async function main() {
  console.log("Bot started");
  const updates = await getUpdates(offset);
  for (const u of updates) {
    offset = u.update_id + 1;
    try {
      await processUpdate(u);
    } catch (e) {
      console.error("process error:", e);
    }
  }
  console.log("Done");
}

main();
