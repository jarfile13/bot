const http = require("http");
const fs = require("fs");
const path = require("path");

const TG_TOKEN = "8606506994:AAE-g9SYVmUKzehn2FHaS2GikRU1rOufBFE";
const VOCAB_SIZE = 256;
const HIDDEN = 64;
const EMB = 16;
const LR = 0.03;
const MAX_LEN = 500;
const GEN_LEN = 80;
const B1 = 0.9, B2 = 0.999, EPS = 1e-8;
const MAX_DELTA = 0.5;
const TEMP = 0.7;
const SNAPSHOT_DIR = "/tmp/snapshots";
const MAX_SNAPSHOTS = 5;
const MAX_FILE_SIZE = 2 * 1024 * 1024;
const END_MARK = "\u0003";
const MAX_EPOCHS = 200;
const EARLY_STOP_PATIENCE = 5;

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
  return { W, M, V, t: 0, h: new Float32Array(HIDDEN) };
}

function softmaxInPlace(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    logits[i] = Math.exp((logits[i] - max) / TEMP);
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

function bytesToStr(bytes) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array(bytes));
  } catch {
    return null;
  }
}

function generate(W, seedText, length) {
  const enc = new TextEncoder();
  let h = new Float32Array(HIDDEN);
  let lastByte = 32;

  const actualSeed = seedText || END_MARK;
  for (const b of enc.encode(actualSeed)) {
    const r = forwardStep(W, b, h);
    h = r.h;
    lastByte = b;
  }

  const outBytes = [];
  for (let i = 0; i < length; i++) {
    const r = forwardStep(W, lastByte, h);
    h = r.h;
    const probs = r.logits;
    softmaxInPlace(probs);
    let p = Math.random(), acc = 0, next = 0;
    for (let k = 0; k < VOCAB_SIZE; k++) {
      acc += probs[k];
      if (p <= acc) { next = k; break; }
    }
    outBytes.push(next);
    lastByte = next;
  }

  let result = "";
  let buf = [];
  for (const b of outBytes) {
    buf.push(b);
    const s = bytesToStr(buf);
    if (s !== null) {
      if (s === END_MARK) {
        result += "\n";
      } else {
        result += s;
      }
      buf = [];
    }
    if (buf.length > 4) buf = [];
  }
  return seedText ? seedText + result : result;
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

async function getFile(fileId) {
  try {
    const r = await tg("getFile", { file_id: fileId });
    if (!r.ok) return null;
    const filePath = r.result.file_path;
    const url = `https://api.telegram.org/file/bot${TG_TOKEN}/${filePath}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.text();
  } catch (e) {
    console.error("getFile failed:", e);
    return null;
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
  let total = 1;
  for (const arr of parts) total += arr.length;
  const view = new Float32Array(total);
  let off = 0;
  for (const arr of parts) { view.set(arr, off); off += arr.length; }
  view[off] = m.t;
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
  m.t = view[off] | 0;
  return m;
}

function snapshotDir(chatId) {
  return path.join(SNAPSHOT_DIR, String(chatId));
}

function saveSnapshot(chatId, m, label, loss) {
  try {
    const dir = snapshotDir(chatId);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const files = fs.readdirSync(dir).sort();
    while (files.length >= MAX_SNAPSHOTS) {
      fs.unlinkSync(path.join(dir, files.shift()));
    }
    const lossStr = loss !== undefined ? loss.toFixed(3) : "start";
    const fname = String(Date.now()) + "_" + label + "_loss" + lossStr + ".bin";
    fs.writeFileSync(path.join(dir, fname), packModel(m));
  } catch (e) {
    console.error("snapshot failed:", e.message);
  }
}

function loadSnapshotByIndex(chatId, index) {
  try {
    const dir = snapshotDir(chatId);
    if (!fs.existsSync(dir)) return null;
    const files = fs.readdirSync(dir).sort();
    if (index < 0 || index >= files.length) return null;
    const f = files[index];
    const buf = fs.readFileSync(path.join(dir, f));
    return { model: unpackModel(buf), name: f };
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

function shuffle(arr, rnd) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function trainOnPhrases(m, phrases, epochs, onEpoch) {
  const enc = new TextEncoder();
  const endBytes = enc.encode(END_MARK);
  const rnd = mulberry32(Date.now() & 0xffff);

  const encodedPhrases = phrases.map(p => {
    const bytes = enc.encode(p);
    const full = new Uint8Array(bytes.length + endBytes.length);
    full.set(bytes, 0);
    full.set(endBytes, bytes.length);
    return full;
  });

  let totalLoss = 0, steps = 0;
  let bestLoss = Infinity;
  let patienceLeft = EARLY_STOP_PATIENCE;
  let stopped = false;
  let epochHistory = [];

  for (let epoch = 0; epoch < epochs && !stopped; epoch++) {
    const order = shuffle(encodedPhrases, rnd);
    let epochLoss = 0, epochSteps = 0;

    for (const bytes of order) {
      for (let i = 0; i < bytes.length - 1; i++) {
        const r = trainStep(m, bytes[i], bytes[i + 1], m.h);
        m.h = r.h;
        epochLoss += r.loss;
        epochSteps++;
      }
    }

    const avgEpoch = epochLoss / Math.max(epochSteps, 1);
    totalLoss += epochLoss;
    steps += epochSteps;
    epochHistory.push(avgEpoch);

    if (onEpoch && (epoch + 1) % 20 === 0) {
      onEpoch(epoch + 1, epochs, avgEpoch);
    }

    if (avgEpoch < bestLoss - 0.001) {
      bestLoss = avgEpoch;
      patienceLeft = EARLY_STOP_PATIENCE;
    } else {
      patienceLeft--;
      if (patienceLeft <= 0) {
        stopped = true;
      }
    }
  }

  return { avg: totalLoss / Math.max(steps, 1), steps, best: bestLoss, stopped, epochs: epochHistory.length, history: epochHistory };
}

async function processUpdate(update) {
  const msg = update.message;
  if (!msg) return;
  const chatId = msg.chat.id;

  if (pendingReset.has(chatId) && msg.text) {
    const t = msg.text.trim().toLowerCase();
    pendingReset.delete(chatId);
    if (t === "да" || t === "/yes") {
      models.delete(chatId);
      try { fs.rmSync(snapshotDir(chatId), { recursive: true, force: true }); } catch {}
      return send(chatId, "Модель сброшена.");
    }
    return send(chatId, "Отмена.");
  }

  if (msg.document) {
    const doc = msg.document;
    const name = (doc.file_name || "").toLowerCase();
    if (!name.endsWith(".txt")) return send(chatId, "Только .txt.");
    if (doc.file_size && doc.file_size > MAX_FILE_SIZE) return send(chatId, "Файл больше 2 МБ.");
    await send(chatId, "Скачиваю...");
    const content = await getFile(doc.file_id);
    if (!content || content.length < 2) return send(chatId, "Пусто или не скачалось.");

    const lines = content.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0).slice(0, 50);
    if (!lines.length) return send(chatId, "Нет строк для обучения.");

    const m = getModel(chatId);
    saveSnapshot(chatId, m, "beforefile");

    const result = trainOnPhrases(m, lines, 100, async (e, total, loss) => {
      if (e % 40 === 0) await send(chatId, "Эпоха " + e + "/" + total + ", loss: " + loss.toFixed(3));
    });

    saveSnapshot(chatId, m, "afterfile", result.avg);

    return send(chatId,
      "Файл обработан.\n" +
      "Строк: " + lines.length + "\n" +
      "Эпох: " + result.epochs + (result.stopped ? " (ранний стоп)" : "") + "\n" +
      "Loss: " + result.avg.toFixed(3) + "\n" +
      "Лучший: " + result.best.toFixed(3) + "\n" +
      "Шагов: " + m.t
    );
  }

  if (!msg.text) return;
  const text = msg.text.trim();
  if (!text) return;

  if (text === "/start" || text === "/help") {
    return send(chatId,
      "RNN. Команды:\n\n" +
      "/learn фраза1 | фраза2 | фраза3 - учить фразы\n" +
      "/gen [начало] - генерировать\n" +
      "/stats - статистика\n" +
      "/undo - откатить\n" +
      "/reset - сбросить\n\n" +
      "Как учить: пиши фразы через |. Бот учит их по кругу и не забывает.\n\n" +
      "Пример:\n" +
      "/learn привет как дела | хорошо а у тебя | тоже хорошо"
    );
  }

  if (text === "/reset") {
    pendingReset.add(chatId);
    return send(chatId, "Точно сбросить? Напиши 'да'.");
  }

  if (text === "/undo") {
    const dir = snapshotDir(chatId);
    if (!fs.existsSync(dir)) return send(chatId, "Нечего откатывать.");
    const files = fs.readdirSync(dir).sort();
    if (!files.length) return send(chatId, "Нечего откатывать.");
    const last = files[files.length - 1];
    fs.unlinkSync(path.join(dir, last));
    const remaining = fs.readdirSync(dir).sort();
    if (!remaining.length) return send(chatId, "Откатил до пустой модели.");
    const target = remaining[remaining.length - 1];
    const buf = fs.readFileSync(path.join(dir, target));
    models.set(chatId, unpackModel(buf));
    return send(chatId, "Откатил. Шагов: " + models.get(chatId).t);
  }

  if (text === "/stats") {
    const m = models.get(chatId);
    if (!m) return send(chatId, "Модель пустая. Используй /learn");
    const params = VOCAB_SIZE*EMB + HIDDEN*EMB + HIDDEN*HIDDEN + HIDDEN + VOCAB_SIZE*HIDDEN + VOCAB_SIZE;
    let snaps = 0;
    try {
      const dir = snapshotDir(chatId);
      if (fs.existsSync(dir)) snaps = fs.readdirSync(dir).length;
    } catch {}
    return send(chatId,
      "Параметров: " + params.toLocaleString() + "\n" +
      "Шагов: " + m.t + "\n" +
      "Hidden: " + HIDDEN + "\n" +
      "Embedding: " + EMB + "\n" +
      "LR: " + LR + "\n" +
      "Откатов доступно: " + snaps
    );
  }

  if (text.startsWith("/learn")) {
    let rest = text.slice(6).trim();
    if (!rest) {
      return send(chatId,
        "Использование:\n" +
        "/learn фраза1 | фраза2 | фраза3\n" +
        "/learn 100 фраза1 | фраза2  (100 эпох)\n\n" +
        "Фразы разделяй |. Бот учит их в случайном порядке, не забывая."
      );
    }

    let epochs = 100;
    const mNum = rest.match(/^(\d+)\s+(.+)$/s);
    if (mNum) {
      epochs = Math.min(parseInt(mNum[1]), MAX_EPOCHS);
      rest = mNum[2];
    }

    const phrases = rest.split("|").map(p => p.trim()).filter(p => p.length > 0);
    if (!phrases.length) return send(chatId, "Нужна хотя бы одна фраза.");
    if (phrases.length > 20) return send(chatId, "Максимум 20 фраз.");
    for (let i = 0; i < phrases.length; i++) {
      if (phrases[i].length > 100) phrases[i] = phrases[i].slice(0, 100);
    }

    const m = getModel(chatId);
    saveSnapshot(chatId, m, "beforelearn");

    const result = trainOnPhrases(m, phrases, epochs, async (e, total, loss) => {
      if (e % 40 === 0) await send(chatId, "Эпоха " + e + "/" + total + ", loss: " + loss.toFixed(3));
    });

    saveSnapshot(chatId, m, "afterlearn", result.avg);

    const firstWord = phrases[0].split(" ")[0];
    return send(chatId,
      "Обучил " + phrases.length + " фраз.\n" +
      "Эпох: " + result.epochs + (result.stopped ? " (ранний стоп)" : "") + "\n" +
      "Loss: " + result.avg.toFixed(3) + "\n" +
      "Лучший: " + result.best.toFixed(3) + "\n" +
      "Шагов: " + m.t + "\n\n" +
      "Проверь: /gen " + firstWord
    );
  }

  if (text.startsWith("/gen")) {
    const seed = text.slice(4).trim().slice(0, 50);
    const m = models.get(chatId);
    if (!m) return send(chatId, "Сначала обучи: /learn");
    return send(chatId, generate(m.W, seed, GEN_LEN));
  }

  return send(chatId, "Неизвестная команда. /help");
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
