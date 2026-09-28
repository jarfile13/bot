const https = require('https');
const http = require('http');
const Parse = require('parse/node');

const BOT_TOKEN = process.env.BOT_TOKEN;
const APP_ID = process.env.B4A_APP_ID;
const JS_KEY = process.env.B4A_JS_KEY;
const MASTER_KEY = process.env.B4A_MASTER_KEY;

const DIM = 50;
const MIN_SCORE = 0.35;
const HIGH_CONFIDENCE = 0.7;
const MAX_PAIRS = 5000;
const MAX_SYN = 30;
const LIST_PAGE = 20;
const MAX_BRAINS = 200;
const REBUILD_EVERY = 10;
const PPMI_WINDOW = 5;

const health = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK');
});
health.listen(3000, () => console.log('Health on 3000'));

Parse.initialize(APP_ID, JS_KEY, MASTER_KEY);
Parse.serverURL = 'https://parseapi.back4app.com/';

const brains = new Map();
const brainOrder = [];
let offset = 0;

function lc(s) {
  return String(s).toLowerCase().replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function words(s) {
  return lc(s).split(' ').filter(w => w.length > 1);
}

function dbKey(owner, q) { return owner + ':' + lc(q); }
function synKey(owner, w) { return owner + ':@syn:' + lc(w); }

function makeBrain(owner) {
  return {
    owner,
    pairs: [],
    syn: new Map(),
    vocab: new Map(),
    ppmi: null,
    vectors: null,
    pairVectors: null,
    dirty: 0,
    loaded: false,
    loading: null,
  };
}

function touchBrain(owner) {
  const i = brainOrder.indexOf(owner);
  if (i >= 0) brainOrder.splice(i, 1);
  brainOrder.push(owner);
  while (brainOrder.length > MAX_BRAINS) {
    const drop = brainOrder.shift();
    if (drop !== owner) brains.delete(drop);
  }
}

function getBrain(owner) {
  let b = brains.get(owner);
  if (!b) { b = makeBrain(owner); brains.set(owner, b); }
  touchBrain(owner);
  return b;
}

function expandSyn(brain, word) {
  const out = new Set([word]);
  const s = brain.syn.get(word);
  if (s) for (const x of s) out.add(x);
  return out;
}

function buildVocab(brain) {
  brain.vocab = new Map();
  for (const p of brain.pairs) {
    const ws = words(p.question).concat(words(p.answer));
    for (const w of ws) {
      if (!brain.vocab.has(w)) brain.vocab.set(w, brain.vocab.size);
    }
  }
}

function buildPPMI(brain) {
  const V = brain.vocab.size;
  if (!V) {
    brain.ppmi = null;
    return;
  }

  const cooc = new Map();
  const wordTotal = new Float64Array(V);
  let totalPairs = 0;

  for (const p of brain.pairs) {
    const ws = words(p.question + ' ' + p.answer).map(w => brain.vocab.get(w)).filter(x => x !== undefined);
    totalPairs++;
    for (let i = 0; i < ws.length; i++) {
      wordTotal[ws[i]]++;
      for (let j = i + 1; j < Math.min(i + 1 + PPMI_WINDOW, ws.length); j++) {
        const a = ws[i], b = ws[j];
        const key = a < b ? a * V + b : b * V + a;
        cooc.set(key, (cooc.get(key) || 0) + 1);
      }
    }
  }

  const ppmi = new Map();
  const totalCooc = [...cooc.values()].reduce((s, v) => s + v, 0) || 1;

  for (const [key, count] of cooc) {
    const a = Math.floor(key / V);
    const b = key % V;
    const pA = wordTotal[a] / totalCooc;
    const pB = wordTotal[b] / totalCooc;
    const pAB = count / totalCooc;
    if (!pA || !pB || !pAB) continue;
    const pmi = Math.log(pAB / (pA * pB));
    if (pmi > 0) ppmi.set(key, pmi);
  }

  brain.ppmi = { V, ppmi, wordTotal, totalCooc };
}

function powerIteration(M, dim, iterations = 30) {
  const n = M.length;
  const Q = [];
  const R = [];

  let A = M.map(row => row.slice());

  for (let k = 0; k < dim; k++) {
    let v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.random();
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (!norm) break;
    for (let i = 0; i < n; i++) v[i] /= norm;

    for (let it = 0; it < iterations; it++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        const row = A[i];
        for (let j = 0; j < n; j++) s += row[j] * v[j];
        w[i] = s;
      }
      let nn = 0;
      for (let i = 0; i < n; i++) nn += w[i] * w[i];
      nn = Math.sqrt(nn);
      if (nn < 1e-12) break;
      for (let i = 0; i < n; i++) v[i] = w[i] / nn;
    }

    const Av = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      const row = A[i];
      for (let j = 0; j < n; j++) s += row[j] * v[j];
      Av[i] = s;
    }
    let lambda = 0;
    for (let i = 0; i < n; i++) lambda += v[i] * Av[i];

    Q.push(Array.from(v));
    R.push(lambda);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        A[i][j] -= lambda * v[i] * v[j];
      }
    }
  }

  return { Q, R };
}

function computeVectors(brain) {
  const V = brain.vocab.size;
  if (!V || !brain.ppmi) {
    brain.vectors = null;
    brain.pairVectors = null;
    return;
  }

  const dim = Math.min(DIM, Math.max(2, V - 1));

  const M = [];
  for (let i = 0; i < V; i++) M.push(new Float64Array(V));

  for (const [key, val] of brain.ppmi.ppmi) {
    const a = Math.floor(key / V);
    const b = key % V;
    M[a][b] = val;
    M[b][a] = val;
  }

  const { Q } = powerIteration(M, dim);

  const vectors = new Map();
  for (let i = 0; i < V; i++) {
    const vec = new Float64Array(dim);
    for (let k = 0; k < dim; k++) vec[k] = Q[k][i];
    let norm = 0;
    for (let k = 0; k < dim; k++) norm += vec[k] * vec[k];
    norm = Math.sqrt(norm) || 1;
    for (let k = 0; k < dim; k++) vec[k] /= norm;
    vectors.set(i, vec);
  }
  brain.vectors = vectors;

  const pairVectors = [];
  for (const p of brain.pairs) {
    const ws = words(p.question);
    const vec = new Float64Array(dim);
    let count = 0;
    for (const w of ws) {
      for (const v of expandSyn(brain, w)) {
        const id = brain.vocab.get(v);
        if (id === undefined) continue;
        const wv = vectors.get(id);
        if (!wv) continue;
        for (let k = 0; k < dim; k++) vec[k] += wv[k];
        count++;
      }
    }
    if (count) {
      let norm = 0;
      for (let k = 0; k < dim; k++) norm += vec[k] * vec[k];
      norm = Math.sqrt(norm) || 1;
      for (let k = 0; k < dim; k++) vec[k] /= norm;
    }
    pairVectors.push(vec);
  }
  brain.pairVectors = pairVectors;
}

function questionVector(brain, input) {
  if (!brain.vectors) return null;
  const ws = words(input);
  if (!ws.length) return null;
  const dim = brain.vectors.values().next().value.length;
  const vec = new Float64Array(dim);
  let count = 0;
  for (const w of ws) {
    for (const v of expandSyn(brain, w)) {
      const id = brain.vocab.get(v);
      if (id === undefined) continue;
      const wv = brain.vectors.get(id);
      if (!wv) continue;
      for (let k = 0; k < dim; k++) vec[k] += wv[k];
      count++;
    }
  }
  if (!count) return null;
  let norm = 0;
  for (let k = 0; k < dim; k++) norm += vec[k] * vec[k];
  norm = Math.sqrt(norm) || 1;
  for (let k = 0; k < dim; k++) vec[k] /= norm;
  return vec;
}

function cosVec(a, b) {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function findTopPairs(brain, qVec, limit) {
  if (!brain.pairVectors) return [];
  const scored = [];
  for (let i = 0; i < brain.pairVectors.length; i++) {
    const score = cosVec(qVec, brain.pairVectors[i]);
    if (score > 0) scored.push({ idx: i, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

function bridgeGenerate(brain, input, topPairs) {
  const qWords = words(input);
  const candidates = [];

  for (const tp of topPairs) {
    const pair = brain.pairs[tp.idx];
    const ansWords = words(pair.answer);
    if (!ansWords.length) continue;

    let relevance = 0;
    for (const aw of ansWords) {
      for (const qw of qWords) {
        if (brain.syn && brain.syn.get(qw)?.has(aw)) { relevance += 1; break; }
        if (aw === qw) { relevance += 1; break; }
      }
    }
    candidates.push({
      text: pair.answer,
      score: tp.score + relevance * 0.1,
    });
  }

  if (!candidates.length) return null;
  candidates.sort((a, b) => b.score - a.score);

  const top = candidates[0];
  if (candidates.length >= 2) {
    const second = candidates[1];
    if (second.text !== top.text && second.score >= top.score * 0.7) {
      const tWords = words(top.text);
      const sWords = words(second.text);
      const shared = tWords.filter(w => sWords.includes(w)).length;
      if (shared < tWords.length * 0.5) {
        return top.text + ' ' + second.text;
      }
    }
  }
  return top.text;
}

function answerFor(brain, input) {
  if (!brain.pairs.length) return null;

  if (brain.dirty >= REBUILD_EVERY) {
    rebuildBrainVectors(brain);
  }

  if (!brain.vectors) {
    rebuildBrainVectors(brain);
    if (!brain.vectors) return null;
  }

  const qVec = questionVector(brain, input);
  if (!qVec) {
    const qWords = words(input);
    for (let i = 0; i < brain.pairs.length; i++) {
      const pWords = words(brain.pairs[i].question);
      const overlap = pWords.filter(w => qWords.includes(w)).length;
      if (overlap > 0) return { answer: brain.pairs[i].answer, kind: 'fallback' };
    }
    return null;
  }

  const top = findTopPairs(brain, qVec, 5);
  if (!top.length) return null;

  const best = top[0];
  if (best.score >= HIGH_CONFIDENCE) {
    return { answer: brain.pairs[best.idx].answer, kind: 'vector-exact', score: best.score };
  }
  if (best.score >= MIN_SCORE) {
    const gen = bridgeGenerate(brain, input, top);
    if (gen) return { answer: gen, kind: 'vector-generated', score: best.score };
    return { answer: brain.pairs[best.idx].answer, kind: 'vector-exact', score: best.score };
  }

  return null;
}

function rebuildBrainVectors(brain) {
  try {
    buildVocab(brain);
    buildPPMI(brain);
    computeVectors(brain);
    brain.dirty = 0;
    console.log('Vectors rebuilt for', brain.owner, 'vocab:', brain.vocab.size, 'pairs:', brain.pairs.length);
  } catch (e) {
    console.error('rebuild error:', e.message);
  }
}

async function dbSavePair(owner, q, a) {
  const K = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(K);
  query.equalTo('question', dbKey(owner, q));
  let obj = await query.first({ useMasterKey: true });
  if (!obj) {
    obj = new K();
    obj.set('question', dbKey(owner, q));
  }
  obj.set('answer', a);
  obj.set('originalQuestion', q);
  return obj.save(null, { useMasterKey: true });
}

async function dbDeletePair(owner, q) {
  const K = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(K);
  query.equalTo('question', dbKey(owner, q));
  const obj = await query.first({ useMasterKey: true });
  if (obj) await obj.destroy({ useMasterKey: true });
}

async function dbSaveSyn(owner, word, set) {
  const K = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(K);
  query.equalTo('question', synKey(owner, word));
  let obj = await query.first({ useMasterKey: true });
  if (!obj) {
    obj = new K();
    obj.set('question', synKey(owner, word));
  }
  obj.set('answer', [...set].join(','));
  obj.set('originalQuestion', word);
  return obj.save(null, { useMasterKey: true });
}

async function dbLoadAll(owner) {
  const K = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(K);
  query.startsWith('question', owner + ':');
  query.limit(10000);
  const results = await query.find({ useMasterKey: true });
  const prefix = owner + ':';
  const synPrefix = owner + ':@syn:';
  const pairs = [];
  const syn = new Map();
  for (const r of results) {
    const raw = r.get('question') || '';
    if (raw.startsWith(synPrefix)) {
      const word = raw.slice(synPrefix.length);
      const ans = r.get('answer') || '';
      const set = new Set();
      for (const s of ans.split(',')) {
        const t = s.trim();
        if (t) set.add(t);
      }
      syn.set(word, set);
    } else if (raw.startsWith(prefix)) {
      const clean = raw.slice(prefix.length);
      pairs.push({
        question: r.get('originalQuestion') || clean,
        answer: r.get('answer'),
      });
    }
  }
  return { pairs, syn };
}

async function ensureLoaded(brain) {
  if (brain.loaded) return brain;
  if (brain.loading) return brain.loading;
  brain.loading = (async () => {
    const { pairs, syn } = await dbLoadAll(brain.owner);
    brain.pairs = pairs.map(p => ({ question: p.question, answer: p.answer }));
    brain.syn = new Map();
    for (const [w, set] of syn) brain.syn.set(w, new Set(set));
    brain.loaded = true;
    brain.dirty = pairs.length;
    console.log('Loaded', brain.owner, 'pairs:', brain.pairs.length);
    return brain;
  })();
  try { return await brain.loading; }
  finally { brain.loading = null; }
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

const HELP = [
  'Смысловой бот. У каждого свой мозг.',
  '',
  '/teach вопрос = ответ',
  '/learn слово = синоним1, синоним2',
  '/delete вопрос',
  '/forget слово = синоним',
  '/list [стр]',
  '/synonyms слово',
  '/rebuild — пересчитать смысловые векторы',
  '/stats',
  '',
  'Чем больше похожих пар — тем умнее бот.',
].join('\n');

async function handle(chat, text) {
  const t = text.trim();
  const owner = String(chat);
  const brain = getBrain(owner);

  if (t === '/start' || t === '/help') return send(chat, HELP);

  if (t === '/rebuild') {
    await ensureLoaded(brain);
    await send(chat, 'Пересчитываю смысловые векторы...');
    rebuildBrainVectors(brain);
    return send(chat, `Готово. Слов: ${brain.vocab.size}, размерность: ${brain.vectors ? brain.vectors.values().next().value.length : 0}`);
  }

  if (t === '/stats') {
    await ensureLoaded(brain);
    const links = [...brain.syn.values()].reduce((s, set) => s + set.size, 0);
    const dim = brain.vectors ? brain.vectors.values().next().value.length : 0;
    return send(chat, `Пар: ${brain.pairs.length}\nСлов: ${brain.vocab.size}\nСинонимов: ${brain.syn.size} (${links} связей)\nВектор: ${dim}D\nDirty: ${brain.dirty}`);
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

  if (t === '/synonyms' || t.startsWith('/synonyms ')) {
    await ensureLoaded(brain);
    const w = lc(t.slice(9).trim());
    if (!w) return send(chat, 'Формат: /synonyms слово');
    const s = brain.syn.get(w);
    if (!s || !s.size) return send(chat, `У "${w}" нет синонимов.`);
    return send(chat, `${w} ↔ ${[...s].join(', ')}`);
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
      await dbSavePair(owner, q, a);
      const key = lc(q);
      const idx = brain.pairs.findIndex(p => lc(p.question) === key);
      if (idx >= 0) {
        brain.pairs[idx].question = q;
        brain.pairs[idx].answer = a;
      } else {
        if (brain.pairs.length >= MAX_PAIRS) return send(chat, 'Слишком много пар.');
        brain.pairs.push({ question: q, answer: a });
      }
      brain.dirty++;
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
      await dbDeletePair(owner, brain.pairs[idx].question);
      brain.pairs.splice(idx, 1);
      brain.dirty++;
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
    for (const o of others) {
      if (!brain.syn.has(o)) brain.syn.set(o, new Set());
      if (brain.syn.get(main).size < MAX_SYN) brain.syn.get(main).add(o);
      if (brain.syn.get(o).size < MAX_SYN) brain.syn.get(o).add(main);
    }
    try {
      await dbSaveSyn(owner, main, brain.syn.get(main));
      for (const o of others) {
        if (brain.syn.get(o)) await dbSaveSyn(owner, o, brain.syn.get(o));
      }
      brain.dirty++;
      return send(chat, `Связал: ${main} ↔ ${others.join(', ')}`);
    } catch (e) {
      console.error('learn', e.message);
      return send(chat, 'Ошибка.');
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
    if (brain.syn.get(a)) brain.syn.get(a).delete(b);
    if (brain.syn.get(b)) brain.syn.get(b).delete(a);
    try {
      if (brain.syn.get(a)) await dbSaveSyn(owner, a, brain.syn.get(a));
      if (brain.syn.get(b)) await dbSaveSyn(owner, b, brain.syn.get(b));
      brain.dirty++;
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
          try { await handle(m.chat.id, m.text); }
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
  if (!APP_ID || !JS_KEY || !MASTER_KEY) { console.error('B4A keys not set'); process.exit(1); }
  console.log('Semantic bot started');
  poll();
}

start();
