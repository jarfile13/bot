const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BOT_TOKEN = process.env.BOT_TOKEN;
const APP_ID = process.env.B4A_APP_ID;
const MASTER_KEY = process.env.B4A_MASTER_KEY;
const SERVER_HOST = 'parseapi.back4app.com';

const EMB_DIM = 64;
const WINDOW = 5;
const MIN_WORD_FREQ = 2;
const MIN_COOC = 2;
const SIM_THRESHOLD = 0.72;
const MIN_SIM = 0.55;
const LIST_PAGE = 20;
const MAX_BRAINS = 200;
const PAGE_SIZE = 100;
const DATA_DIR = process.env.DATA_DIR || './data';

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

const health = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK');
});
health.listen(3000, () => console.log('Health on 3000'));

const brains = new Map();
const brainOrder = [];
let offset = 0;

function lc(s) {
  return String(s).toLowerCase().replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function tokenize(s) {
  return lc(s).split(' ').filter(w => w.length >= 2);
}

function scopeOf(chatId, isGroup) {
  return isGroup ? `c:${chatId}` : `u:${chatId}`;
}

function safeName(scope) {
  return scope.replace(/[^\w]/g, '_');
}

function dbKey(scope, q) { return scope + ':' + lc(q); }
function synKey(scope, w) { return scope + ':@syn:' + lc(w); }

function makeBrain(scope) {
  return {
    scope,
    pairs: [],
    syn: new Map(),
    loaded: false,
    loading: null,
    vocab: new Map(),
    idf: new Map(),
    wordVec: new Map(),
    ready: false,
    dirty: true,
    file: path.join(DATA_DIR, safeName(scope) + '.json'),
  };
}

function touchBrain(scope) {
  const i = brainOrder.indexOf(scope);
  if (i >= 0) brainOrder.splice(i, 1);
  brainOrder.push(scope);
  while (brainOrder.length > MAX_BRAINS) {
    const drop = brainOrder.shift();
    if (drop !== scope) brains.delete(drop);
  }
}

function getBrain(scope) {
  let b = brains.get(scope);
  if (!b) { b = makeBrain(scope); brains.set(scope, b); }
  touchBrain(scope);
  return b;
}

function expand(brain, word) {
  const out = new Set([word]);
  const s = brain.syn.get(word);
  if (s) for (const x of s) out.add(x);
  return out;
}

function buildVocab(brain) {
  const df = new Map();
  for (const p of brain.pairs) {
    const seen = new Set();
    for (const w of tokenize(p.question)) {
      if (!seen.has(w)) { seen.add(w); df.set(w, (df.get(w) || 0) + 1); }
    }
  }
  const vocab = new Map();
  let id = 0;
  for (const [w, c] of df) {
    if (c >= MIN_WORD_FREQ) vocab.set(w, id++);
  }
  brain.vocab = vocab;
  brain.df = df;
  const N = brain.pairs.length || 1;
  const idf = new Map();
  for (const [w, c] of df) {
    idf.set(w, Math.log((N + 1) / (c + 1)) + 1);
  }
  brain.idf = idf;
}

function buildCooc(brain) {
  const V = brain.vocab.size;
  if (!V) { brain.wordVec = new Map(); return; }
  const rows = new Map();
  const addEntry = (a, b) => {
    let r = rows.get(a);
    if (!r) { r = new Map(); rows.set(a, r); }
    r.set(b, (r.get(b) || 0) + 1);
  };
  for (const p of brain.pairs) {
    const ws = tokenize(p.question).filter(w => brain.vocab.has(w));
    for (let i = 0; i < ws.length; i++) {
      for (let j = i + 1; j < ws.length && j <= i + WINDOW; j++) {
        addEntry(ws[i], ws[j]);
        addEntry(ws[j], ws[i]);
      }
    }
  }
  const total = new Map();
  let sum = 0;
  for (const [w, m] of rows) {
    let t = 0;
    for (const c of m.values()) t += c;
    total.set(w, t);
    sum += t;
  }
  if (!sum) { brain.wordVec = new Map(); return; }

  const sparse = [];
  for (const [w, m] of rows) {
    const vec = [];
    const tw = total.get(w) || 1;
    for (const [c, cnt] of m) {
      if (cnt < MIN_COOC) continue;
      const tc = total.get(c) || 1;
      const pmi = Math.log((cnt * sum) / (tw * tc));
      const ppmi = Math.max(0, pmi);
      if (ppmi > 0 && brain.vocab.has(c)) vec.push([brain.vocab.get(c), ppmi]);
    }
    if (vec.length) sparse.push([brain.vocab.get(w), vec]);
  }

  const dim = Math.min(EMB_DIM, V);
  const emb = svdPowerIter(sparse, V, dim, 20);
  const wordVec = new Map();
  for (const [w, id] of brain.vocab) {
    const v = emb[id];
    if (!v) continue;
    wordVec.set(w, v);
  }
  brain.wordVec = wordVec;
}

function svdPowerIter(sparse, V, k, iters) {
  const rows = new Map();
  for (const [r, vec] of sparse) rows.set(r, vec);

  const result = new Map();
  let residualRows = new Map(rows);

  for (let comp = 0; comp < k; comp++) {
    let v = new Float64Array(V);
    for (let i = 0; i < V; i++) v[i] = Math.random() - 0.5;

    for (let it = 0; it < iters; it++) {
      const u = new Float64Array(V);
      for (const [, vec] of residualRows) {
        let dot = 0;
        for (const [c, val] of vec) dot += val * v[c];
        if (dot === 0) continue;
        for (const [c, val] of vec) u[c] += val * dot;
      }
      let norm = 0;
      for (let i = 0; i < V; i++) norm += u[i] * u[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < V; i++) v[i] = u[i] / norm;
    }

    let sigma = 0;
    {
      const u = new Float64Array(V);
      for (const [, vec] of residualRows) {
        let dot = 0;
        for (const [c, val] of vec) dot += val * v[c];
        if (dot === 0) continue;
        for (const [c, val] of vec) u[c] += val * dot;
      }
      for (let i = 0; i < V; i++) sigma += u[i] * v[i];
    }

    if (Math.abs(sigma) < 1e-6) break;

    const newResidual = new Map();
    for (const [r, vec] of residualRows) {
      let vr = 0;
      for (const [c, val] of vec) vr += val * v[c];
      const updated = [];
      for (const [c, val] of vec) updated.push([c, val - sigma * vr * v[c]]);
      newResidual.set(r, updated);
    }
    residualRows = newResidual;

    for (let i = 0; i < V; i++) {
      if (!result.has(i)) result.set(i, new Float64Array(k));
      result.get(i)[comp] = v[i] * Math.sqrt(Math.abs(sigma));
    }
  }

  const out = [];
  for (let i = 0; i < V; i++) out.push(result.get(i) || new Float64Array(k));
  return out;
}

function embedText(brain, text) {
  if (!brain.wordVec || !brain.wordVec.size) return null;
  const ws = tokenize(text);
  if (!ws.length) return null;
  const acc = new Float64Array(EMB_DIM);
  let count = 0;
  for (const w of ws) {
    for (const v of expand(brain, w)) {
      const vec = brain.wordVec.get(v);
      if (!vec) continue;
      const weight = brain.idf.get(v) || 1;
      for (let i = 0; i < vec.length; i++) acc[i] += vec[i] * weight;
      count++;
    }
  }
  if (!count) return null;
  let norm = 0;
  for (let i = 0; i < acc.length; i++) norm += acc[i] * acc[i];
  norm = Math.sqrt(norm) || 1;
  const out = [];
  for (let i = 0; i < acc.length; i++) out.push(acc[i] / norm);
  return out;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function rebuildEmbeddings(brain) {
  buildVocab(brain);
  buildCooc(brain);
  brain.ready = true;
  brain.dirty = false;
}

function reindexVectors(brain) {
  for (const p of brain.pairs) {
    const v = embedText(brain, p.question);
    p.vector = v || new Float64Array(EMB_DIM);
  }
}

function findBestPairs(brain, queryVec, limit) {
  if (!brain.ready || !queryVec) return [];
  const scored = [];
  for (let i = 0; i < brain.pairs.length; i++) {
    const p = brain.pairs[i];
    if (!p.vector) continue;
    const sim = cosine(queryVec, p.vector);
    if (sim < MIN_SIM) continue;
    scored.push({ idx: i, score: sim, raw: sim });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

function answerFor(brain, input) {
  if (!brain.pairs.length || !brain.ready) return null;
  const qVec = embedText(brain, input);
  if (!qVec) return null;
  const best = findBestPairs(brain, qVec, 3);
  if (!best.length) return null;
  const top = best[0];
  if (top.score >= SIM_THRESHOLD) return { answer: brain.pairs[top.idx].answer, kind: 'exact', score: top.score };
  if (top.score >= MIN_SIM) return { answer: brain.pairs[top.idx].answer, kind: 'close', score: top.score };
  return null;
}

function b4aRequest(method, urlPath, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const headers = {
      'X-Parse-Application-Id': APP_ID,
      'X-Parse-Master-Key': MASTER_KEY,
    };
    if (data) {
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = Buffer.byteLength(data);
    }
    const req = https.request({ hostname: SERVER_HOST, path: urlPath, method, headers }, (res) => {
      let buf = '';
      res.on('data', c => buf += c);
      res.on('end', () => {
        try {
          const json = JSON.parse(buf || '{}');
          if (res.statusCode >= 400) return reject(new Error(json.error || ('HTTP ' + res.statusCode)));
          resolve(json);
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

async function dbFind(scope, q) {
  const where = encodeURIComponent(JSON.stringify({ question: dbKey(scope, q) }));
  const r = await b4aRequest('GET', `/classes/Knowledge?where=${where}&limit=1`);
  return r.results && r.results[0] ? r.results[0] : null;
}

async function dbSavePair(scope, q, a) {
  const existing = await dbFind(scope, q);
  const body = { question: dbKey(scope, q), answer: a, originalQuestion: q, scope };
  if (existing) await b4aRequest('PUT', `/classes/Knowledge/${existing.objectId}`, body);
  else await b4aRequest('POST', '/classes/Knowledge', body);
}

async function dbDeletePair(scope, q) {
  const existing = await dbFind(scope, q);
  if (existing) await b4aRequest('DELETE', `/classes/Knowledge/${existing.objectId}`);
}

async function dbSaveSyn(scope, word, set) {
  const existing = await dbFind(scope, '@syn:' + word);
  const body = { question: synKey(scope, word), answer: [...set].join(','), originalQuestion: word, scope };
  if (existing) await b4aRequest('PUT', `/classes/Knowledge/${existing.objectId}`, body);
  else await b4aRequest('POST', '/classes/Knowledge', body);
}

async function dbLoadScope(scope) {
  const prefix = scope + ':';
  const synPrefix = scope + ':@syn:';
  const pairs = [];
  const syn = new Map();
  let skip = 0;
  while (true) {
    const where = encodeURIComponent(JSON.stringify({
      question: { $regex: '^' + prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') }
    }));
    const r = await b4aRequest('GET', `/classes/Knowledge?where=${where}&limit=${PAGE_SIZE}&skip=${skip}&order=createdAt`);
    const results = r.results || [];
    if (!results.length) break;
    for (const obj of results) {
      const raw = obj.question || '';
      if (raw.startsWith(synPrefix)) {
        const word = raw.slice(synPrefix.length);
        const ans = obj.answer || '';
        const set = new Set();
        for (const s of ans.split(',')) {
          const t = s.trim();
          if (t) set.add(t);
        }
        syn.set(word, set);
      } else if (raw.startsWith(prefix)) {
        const clean = raw.slice(prefix.length);
        pairs.push({ question: obj.originalQuestion || clean, answer: obj.answer, scope });
      }
    }
    if (results.length < PAGE_SIZE) break;
    skip += PAGE_SIZE;
  }
  return { pairs, syn };
}

async function persistLocal(brain) {
  const data = {
    scope: brain.scope,
    pairs: brain.pairs.map(p => ({ question: p.question, answer: p.answer, scope: p.scope })),
    syn: [...brain.syn.entries()].map(([w, set]) => [w, [...set]]),
  };
  await fs.promises.writeFile(brain.file, JSON.stringify(data), 'utf8');
}

async function loadLocal(brain) {
  try {
    const raw = await fs.promises.readFile(brain.file, 'utf8');
    const data = JSON.parse(raw);
    if (data.scope !== brain.scope) return false;
    brain.pairs = (data.pairs || []).map(p => ({ ...p, scope: p.scope || brain.scope }));
    brain.syn = new Map();
    for (const [w, arr] of (data.syn || [])) brain.syn.set(w, new Set(arr));
    return true;
  } catch (e) { return false; }
}

async function ensureLoaded(brain) {
  if (brain.loaded) {
    if (brain.dirty) { rebuildEmbeddings(brain); reindexVectors(brain); }
    return brain;
  }
  if (brain.loading) return brain.loading;
  const p = (async () => {
    try {
      const local = await loadLocal(brain);
      if (!local) {
        const { pairs, syn } = await dbLoadScope(brain.scope);
        brain.pairs = pairs;
        brain.syn = new Map();
        for (const [w, set] of syn) brain.syn.set(w, new Set(set));
        await persistLocal(brain);
      }
      brain.loaded = true;
      rebuildEmbeddings(brain);
      reindexVectors(brain);
      console.log('Loaded', brain.scope, 'pairs:', brain.pairs.length, 'vocab:', brain.vocab.size);
      return brain;
    } finally {
      brain.loading = null;
    }
  })();
  brain.loading = p;
  return p;
}

function api(method, payload) {
  return new Promise((resolve, reject) => {
    if (!BOT_TOKEN) return reject(new Error('no token'));
    const data = JSON.stringify(payload || {});
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${BOT_TOKEN}/${method}`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    }, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(e); } });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function send(chat, text) {
  try { await api('sendMessage', { chat_id: chat, text }); }
  catch (e) { console.error('send', e.message); }
}

async function sendDocument(chat, filename, content) {
  return new Promise((resolve, reject) => {
    const boundary = '----bot' + Date.now();
    const head = `--${boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n${chat}\r\n--${boundary}\r\nContent-Disposition: form-data; name="document"; filename="${filename}"\r\nContent-Type: application/json\r\n\r\n`;
    const tail = `\r\n--${boundary}--\r\n`;
    const body = Buffer.concat([Buffer.from(head, 'utf8'), Buffer.from(content, 'utf8'), Buffer.from(tail, 'utf8')]);
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${BOT_TOKEN}/sendDocument`,
      method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}`, 'Content-Length': body.length },
    }, (res) => {
      let b = '';
      res.on('data', c => b += c);
      res.on('end', () => { try { resolve(JSON.parse(b)); } catch (e) { reject(e); } });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function downloadFile(fileId) {
  const r = await api('getFile', { file_id: fileId });
  if (!r || !r.ok) throw new Error('getFile failed');
  const filePath = r.result.file_path;
  return new Promise((resolve, reject) => {
    https.get(`https://api.telegram.org/file/bot${BOT_TOKEN}/${filePath}`, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => resolve(body));
    }).on('error', reject);
  });
}

const HELP = [
  'Бот с самообучающимися эмбеддингами (чистый JS).',
  '',
  '/teach вопрос = ответ',
  '/learn слово = синоним1, синоним2',
  '/delete вопрос',
  '/forget слово = синоним',
  '/list [стр] — список пар',
  '/syn [стр] — список синонимов',
  '/syn слово — синонимы конкретного слова',
  '/stats',
  '/reindex — пересчитать эмбеддинги',
  '/export — выгрузить свой мозг',
  '/import (реплай на JSON) — залить мозг',
].join('\n');

async function handle(chat, text, isGroup, replyTo) {
  const t = text.trim();
  const scope = scopeOf(chat, isGroup);
  const brain = getBrain(scope);

  if (t === '/start' || t === '/help') return send(chat, HELP);

  if (t === '/stats') {
    await ensureLoaded(brain);
    const links = [...brain.syn.values()].reduce((s, set) => s + set.size, 0);
    return send(chat, `Скоуп: ${scope}\nПар: ${brain.pairs.length}\nСлов в словаре: ${brain.vocab.size}\nВекторов: ${brain.wordVec.size}\nСинонимов: ${brain.syn.size} (связей ${links})\nРазмерность: ${EMB_DIM}`);
  }

  if (t === '/reindex') {
    await ensureLoaded(brain);
    rebuildEmbeddings(brain);
    reindexVectors(brain);
    await persistLocal(brain);
    return send(chat, `Пересчитал. Слов: ${brain.vocab.size}, векторов: ${brain.wordVec.size}`);
  }

  if (t === '/list' || t.startsWith('/list ')) {
    await ensureLoaded(brain);
    if (!brain.pairs.length) return send(chat, 'Пусто.');
    const arg = t.slice(5).trim();
    const totalPages = Math.max(1, Math.ceil(brain.pairs.length / LIST_PAGE));
    let page = 1;
    if (arg) {
      const n = parseInt(arg, 10);
      if (!Number.isFinite(n) || n < 1) return send(chat, `Всего страниц: ${totalPages}`);
      page = Math.min(n, totalPages);
    }
    const start = (page - 1) * LIST_PAGE;
    const slice = brain.pairs.slice(start, start + LIST_PAGE);
    let out = `Стр. ${page} из ${totalPages} (всего ${brain.pairs.length})\n\n`;
    for (let i = 0; i < slice.length; i++) {
      out += `${start + i + 1}. ${slice[i].question} = ${slice[i].answer}\n`;
    }
    if (out.length > 3900) out = out.slice(0, 3900) + '...';
    return send(chat, out);
  }

  if (t === '/syn' || t.startsWith('/syn ')) {
    await ensureLoaded(brain);
    const arg = t.slice(4).trim();

    if (arg && !/^\d+$/.test(arg)) {
      const w = lc(arg);
      const s = brain.syn.get(w);
      if (!s || !s.size) return send(chat, `У "${w}" нет синонимов.`);
      return send(chat, `${w} ↔ ${[...s].join(', ')}`);
    }

    const entries = [...brain.syn.entries()].filter(([, set]) => set.size);
    if (!entries.length) return send(chat, 'Синонимов нет.');
    entries.sort((a, b) => a[0].localeCompare(b[0]));
    const totalPages = Math.max(1, Math.ceil(entries.length / LIST_PAGE));
    let page = 1;
    if (arg) {
      const n = parseInt(arg, 10);
      if (!Number.isFinite(n) || n < 1) return send(chat, `Всего страниц: ${totalPages}`);
      page = Math.min(n, totalPages);
    }
    const start = (page - 1) * LIST_PAGE;
    const slice = entries.slice(start, start + LIST_PAGE);
    let out = `Синонимы. Стр. ${page} из ${totalPages} (всего ${entries.length})\n\n`;
    for (let i = 0; i < slice.length; i++) {
      out += `${start + i + 1}. ${slice[i][0]} ↔ ${[...slice[i][1]].join(', ')}\n`;
    }
    if (out.length > 3900) out = out.slice(0, 3900) + '...';
    return send(chat, out);
  }

  if (t === '/export') {
    await ensureLoaded(brain);
    const data = {
      scope,
      exportedAt: new Date().toISOString(),
      pairs: brain.pairs.filter(p => p.scope === scope).map(p => ({ q: p.question, a: p.answer })),
      syn: [...brain.syn.entries()].filter(([, set]) => set.size).map(([w, set]) => ({ w, s: [...set] })),
    };
    const filename = `brain_${safeName(scope)}.json`;
    try { await sendDocument(chat, filename, JSON.stringify(data, null, 2)); }
    catch (e) { console.error('export', e.message); return send(chat, 'Ошибка экспорта.'); }
    return;
  }

  if (t === '/import' || t.startsWith('/import')) {
    if (!replyTo || !replyTo.document) return send(chat, 'Ответь /import на JSON-файл.');
    try {
      const content = await downloadFile(replyTo.document.file_id);
      const data = JSON.parse(content);
      const pairs = Array.isArray(data.pairs) ? data.pairs : [];
      const syn = Array.isArray(data.syn) ? data.syn : [];
      await ensureLoaded(brain);
      let added = 0;
      for (const p of pairs) {
        if (!p || !p.q || !p.a) continue;
        const key = lc(p.q);
        const idx = brain.pairs.findIndex(x => lc(x.question) === key);
        if (idx >= 0) continue;
        await dbSavePair(scope, p.q, p.a);
        brain.pairs.push({ question: p.q, answer: p.a, scope });
        added++;
      }
      for (const item of syn) {
        if (!item || !item.w || !Array.isArray(item.s)) continue;
        const w = lc(item.w);
        if (!brain.syn.has(w)) brain.syn.set(w, new Set());
        const target = brain.syn.get(w);
        for (const s of item.s) target.add(lc(s));
        await dbSaveSyn(scope, w, target);
      }
      brain.dirty = true;
      rebuildEmbeddings(brain);
      reindexVectors(brain);
      await persistLocal(brain);
      return send(chat, `Импортировано пар: ${added}, синонимов: ${syn.length}`);
    } catch (e) {
      console.error('import', e.message);
      return send(chat, 'Ошибка импорта.');
    }
  }

  if (t.startsWith('/teach')) {
    const rest = t.slice(6).trim();
    const parts = rest.split('=');
    if (parts.length < 2) return send(chat, 'Формат: /teach вопрос = ответ');
    const q = parts[0].trim();
    const a = parts.slice(1).join('=').trim();
    if (!q || !a) return send(chat, 'Пусто.');
    await ensureLoaded(brain);
    try {
      await dbSavePair(scope, q, a);
      const key = lc(q);
      const idx = brain.pairs.findIndex(p => lc(p.question) === key);
      const newPair = { question: q, answer: a, scope };
      if (idx >= 0) brain.pairs[idx] = newPair;
      else brain.pairs.push(newPair);
      brain.dirty = true;
      rebuildEmbeddings(brain);
      reindexVectors(brain);
      await persistLocal(brain);
      return send(chat, `Запомнил: ${q} = ${a}`);
    } catch (e) {
      console.error('teach', e.message);
      return send(chat, 'Ошибка сохранения.');
    }
  }

  if (t.startsWith('/delete')) {
    const rest = t.slice(7).trim();
    if (!rest) return send(chat, 'Формат: /delete вопрос');
    await ensureLoaded(brain);
    const key = lc(rest);
    const idx = brain.pairs.findIndex(p => lc(p.question) === key);
    if (idx < 0) return send(chat, `Не нашёл: ${key}`);
    try {
      await dbDeletePair(scope, brain.pairs[idx].question);
      brain.pairs.splice(idx, 1);
      brain.dirty = true;
      rebuildEmbeddings(brain);
      reindexVectors(brain);
      await persistLocal(brain);
      return send(chat, `Удалил: ${rest}`);
    } catch (e) {
      console.error('delete', e.message);
      return send(chat, 'Ошибка удаления.');
    }
  }

  if (t.startsWith('/learn')) {
    const rest = t.slice(6).trim();
    const parts = rest.split('=');
    if (parts.length < 2) return send(chat, 'Формат: /learn слово = синоним1, синоним2');
    const main = lc(parts[0].trim());
    const others = parts[1].split(',').map(s => lc(s.trim())).filter(Boolean);
    if (!main || !others.length) return send(chat, 'Пусто.');
    await ensureLoaded(brain);
    if (!brain.syn.has(main)) brain.syn.set(main, new Set());
    const sa = brain.syn.get(main);
    for (const o of others) {
      sa.add(o);
      if (!brain.syn.has(o)) brain.syn.set(o, new Set());
      brain.syn.get(o).add(main);
    }
    try {
      await dbSaveSyn(scope, main, sa);
      for (const o of others) {
        const s = brain.syn.get(o);
        if (s) await dbSaveSyn(scope, o, s);
      }
      reindexVectors(brain);
      await persistLocal(brain);
      return send(chat, `Связал: ${main} ↔ ${others.join(', ')}`);
    } catch (e) {
      console.error('learn', e.message);
      return send(chat, 'Ошибка сохранения.');
    }
  }

  if (t.startsWith('/forget')) {
    const rest = t.slice(7).trim();
    const parts = rest.split('=');
    if (parts.length < 2) return send(chat, 'Формат: /forget слово = синоним');
    const a = lc(parts[0].trim());
    const b = lc(parts[1].trim());
    if (!a || !b) return send(chat, 'Пусто.');
    await ensureLoaded(brain);
    const sa = brain.syn.get(a), sb = brain.syn.get(b);
    if (sa) sa.delete(b);
    if (sb) sb.delete(a);
    try {
      if (sa) await dbSaveSyn(scope, a, sa);
      if (sb) await dbSaveSyn(scope, b, sb);
      reindexVectors(brain);
      await persistLocal(brain);
      return send(chat, `Разъединил: ${a} ✕ ${b}`);
    } catch (e) {
      console.error('forget', e.message);
      return send(chat, 'Ошибка.');
    }
  }

  await ensureLoaded(brain);
  const res = answerFor(brain, t);
  if (!res) return send(chat, 'Не знаю. Научи: /teach вопрос = ответ');
  return send(chat, res.answer);
}

async function poll() {
  while (true) {
    try {
      const r = await api('getUpdates', { offset, timeout: 30, allowed_updates: ['message'] });
      if (r && r.ok && Array.isArray(r.result)) {
        for (const u of r.result) {
          offset = u.update_id + 1;
          const m = u.message;
          if (!m || !m.text) continue;
          const isGroup = m.chat.type === 'group' || m.chat.type === 'supergroup';
          try { await handle(m.chat.id, m.text, isGroup, m.reply_to_message); }
          catch (e) { console.error('handle', e.message); }
        }
      }
    } catch (e) {
      console.error('poll', e.message);
      await new Promise(r => setTimeout(r, 3000));
    }
  }
}

async function start() {
  if (!BOT_TOKEN) { console.error('BOT_TOKEN not set'); process.exit(1); }
  if (!APP_ID || !MASTER_KEY) { console.error('B4A keys not set'); process.exit(1); }
  console.log('Bot started');
  poll();
}

start();
