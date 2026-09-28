const https = require('https');
const http = require('http');
const Parse = require('parse/node');

const BOT_TOKEN = process.env.BOT_TOKEN;
const APP_ID = process.env.B4A_APP_ID;
const JS_KEY = process.env.B4A_JS_KEY;
const MASTER_KEY = process.env.B4A_MASTER_KEY;

const MIN_SCORE = 0.35;
const MIN_GEN_LEN = 2;
const MAX_GEN_LEN = 25;
const MAX_PAIRS = 5000;
const MAX_SYN = 30;
const LIST_PAGE = 20;
const MAX_BRAINS = 200;

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
  return lc(s).split(' ').filter(w => w.length > 0);
}

function dbKey(owner, q) { return owner + ':' + lc(q); }
function synKey(owner, w) { return owner + ':@syn:' + lc(w); }

function makeBrain(owner) {
  return {
    owner,
    pairs: [],
    syn: new Map(),
    wordIndex: new Map(),
    df: new Map(),
    markov: new Map(),
    starters: [],
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

function expand(brain, word) {
  const out = new Set([word]);
  const s = brain.syn.get(word);
  if (s) for (const x of s) out.add(x);
  return out;
}

function idf(brain, w) {
  const df = brain.df.get(w) || 0;
  if (df === 0) return 0;
  return Math.log(1 + brain.pairs.length / df);
}

function indexPair(brain, idx, question, answer) {
  const qw = words(question);
  brain.pairs[idx].qWords = qw;
  const seen = new Set();
  for (const w of qw) {
    if (!brain.wordIndex.has(w)) brain.wordIndex.set(w, new Set());
    brain.wordIndex.get(w).add(idx);
    if (!seen.has(w)) {
      seen.add(w);
      brain.df.set(w, (brain.df.get(w) || 0) + 1);
    }
  }
  indexMarkov(brain, answer);
}

function indexMarkov(brain, text) {
  const ws = words(text);
  if (!ws.length) return;
  brain.starters.push(ws[0]);
  for (let i = 0; i < ws.length; i++) {
    const cur = ws[i];
    const next = ws[i + 1] || null;
    if (!brain.markov.has(cur)) brain.markov.set(cur, new Map());
    if (next) {
      const m = brain.markov.get(cur);
      m.set(next, (m.get(next) || 0) + 1);
    }
  }
}

function unindexPair(brain, idx) {
  const p = brain.pairs[idx];
  if (!p || !p.qWords) return;
  const seen = new Set();
  for (const w of p.qWords) {
    const s = brain.wordIndex.get(w);
    if (s) { s.delete(idx); if (!s.size) brain.wordIndex.delete(w); }
    if (!seen.has(w)) {
      seen.add(w);
      const d = (brain.df.get(w) || 0) - 1;
      if (d <= 0) brain.df.delete(w);
      else brain.df.set(w, d);
    }
  }
  p.qWords = null;
}

function rebuildMarkov(brain) {
  brain.markov.clear();
  brain.starters = [];
  for (const p of brain.pairs) {
    if (p.answer) indexMarkov(brain, p.answer);
  }
}

function rebuildIndex(brain) {
  brain.wordIndex.clear();
  brain.df.clear();
  for (let i = 0; i < brain.pairs.length; i++) {
    brain.pairs[i].qWords = null;
    indexPair(brain, i, brain.pairs[i].question, brain.pairs[i].answer);
  }
  rebuildMarkov(brain);
}

function addSyn(brain, a, b) {
  if (a === b) return;
  if (!brain.syn.has(a)) brain.syn.set(a, new Set());
  if (!brain.syn.has(b)) brain.syn.set(b, new Set());
  const sa = brain.syn.get(a), sb = brain.syn.get(b);
  if (sa.size < MAX_SYN) sa.add(b);
  if (sb.size < MAX_SYN) sb.add(a);
}

function delSyn(brain, a, b) {
  const sa = brain.syn.get(a), sb = brain.syn.get(b);
  if (sa) sa.delete(b);
  if (sb) sb.delete(a);
}

function scorePair(brain, queryWords, pairIdx) {
  const pair = brain.pairs[pairIdx];
  if (!pair || !pair.qWords || !pair.qWords.length) return 0;
  const qSet = new Set();
  for (const w of queryWords) for (const v of expand(brain, w)) qSet.add(v);
  const pairSet = new Set(pair.qWords);
  let dot = 0, qNorm = 0, pNorm = 0;
  for (const w of qSet) {
    const i = idf(brain, w);
    qNorm += i * i;
    if (pairSet.has(w)) dot += i * i;
  }
  for (const w of pairSet) {
    const i = idf(brain, w);
    pNorm += i * i;
  }
  if (!qNorm || !pNorm) return 0;
  return dot / (Math.sqrt(qNorm) * Math.sqrt(pNorm));
}

function findBestPairs(brain, queryWords, limit) {
  const candidates = new Set();
  for (const w of queryWords) {
    for (const v of expand(brain, w)) {
      const idxs = brain.wordIndex.get(v);
      if (idxs) for (const i of idxs) candidates.add(i);
    }
  }
  const scored = [];
  for (const i of candidates) {
    const s = scorePair(brain, queryWords, i);
    if (s > 0) scored.push({ idx: i, score: s });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

function pickNext(brain, word) {
  const m = brain.markov.get(word);
  if (!m || !m.size) return null;
  let total = 0;
  for (const c of m.values()) total += c;
  let r = Math.random() * total;
  for (const [next, count] of m) {
    r -= count;
    if (r <= 0) return next;
  }
  return null;
}

function generateFromSeed(brain, seed) {
  if (!brain.markov.has(seed)) return null;
  const out = [seed];
  let cur = seed;
  for (let i = 0; i < MAX_GEN_LEN; i++) {
    const next = pickNext(brain, cur);
    if (!next) break;
    out.push(next);
    cur = next;
    if (out.length >= MIN_GEN_LEN && Math.random() < 0.15) break;
  }
  if (out.length < MIN_GEN_LEN) return null;
  return out.join(' ');
}

function pickSeedFromPairs(brain, bestPairs) {
  const seeds = [];
  for (const bp of bestPairs) {
    const ans = brain.pairs[bp.idx].answer;
    const ws = words(ans);
    if (ws.length) seeds.push({ word: ws[0], score: bp.score });
  }
  if (!seeds.length) return null;
  seeds.sort((a, b) => b.score - a.score);
  return seeds[0].word;
}

function generateAnswer(brain, queryWords, bestPairs) {
  const seed = pickSeedFromPairs(brain, bestPairs);
  if (!seed) return null;
  const gen = generateFromSeed(brain, seed);
  if (gen) return gen;
  return null;
}

function glueFromPairs(brain, bestPairs) {
  const parts = [];
  for (const bp of bestPairs.slice(0, 2)) {
    const ans = brain.pairs[bp.idx].answer;
    if (ans && !parts.includes(ans)) parts.push(ans);
  }
  if (!parts.length) return null;
  return parts.join(' ');
}

function answerFor(brain, input) {
  if (!brain.pairs.length) return null;
  const qWords = words(input);
  if (!qWords.length) return null;

  const bestPairs = findBestPairs(brain, qWords, 5);

  if (bestPairs.length) {
    const top = bestPairs[0];
    if (top.score >= 0.65) {
      return { answer: brain.pairs[top.idx].answer, kind: 'exact', score: top.score };
    }
    if (top.score >= MIN_SCORE) {
      const gen = generateAnswer(brain, qWords, bestPairs);
      if (gen) return { answer: gen, kind: 'generated', score: top.score };
      const glue = glueFromPairs(brain, bestPairs);
      if (glue) return { answer: glue, kind: 'glued', score: top.score };
      return { answer: brain.pairs[top.idx].answer, kind: 'exact', score: top.score };
    }
  }

  const gen = generateFromRandomSeed(brain);
  if (gen) return { answer: gen, kind: 'random', score: 0 };
  return null;
}

function generateFromRandomSeed(brain) {
  if (!brain.starters.length) return null;
  const seed = brain.starters[Math.floor(Math.random() * brain.starters.length)];
  return generateFromSeed(brain, seed);
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
    brain.pairs = pairs.map(p => ({ question: p.question, answer: p.answer, qWords: null }));
    brain.syn = new Map();
    for (const [w, set] of syn) brain.syn.set(w, new Set(set));
    rebuildIndex(brain);
    brain.loaded = true;
    console.log('Loaded', brain.owner, 'pairs:', brain.pairs.length, 'syn:', brain.syn.size, 'markov:', brain.markov.size);
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
  'Свой генеративный бот. У каждого свой мозг.',
  '',
  '/teach вопрос = ответ — научить',
  '/learn слово = синоним1, синоним2',
  '/delete вопрос',
  '/forget слово = синоним',
  '/list [стр]',
  '/synonyms слово',
  '/stats',
  '',
  'Если точного ответа нет — бот сам сгенерирует из выученного.',
].join('\n');

async function handle(chat, text) {
  const t = text.trim();
  const owner = String(chat);
  const brain = getBrain(owner);

  if (t === '/start' || t === '/help') return send(chat, HELP);

  if (t === '/stats') {
    await ensureLoaded(brain);
    const links = [...brain.syn.values()].reduce((s, set) => s + set.size, 0);
    return send(chat, `Пар: ${brain.pairs.length}\nСлов с синонимами: ${brain.syn.size}\nСвязей: ${links}\nСлов в цепи Маркова: ${brain.markov.size}`);
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
        unindexPair(brain, idx);
        brain.pairs[idx].question = q;
        brain.pairs[idx].answer = a;
        indexPair(brain, idx, q, a);
        rebuildMarkov(brain);
      } else {
        if (brain.pairs.length >= MAX_PAIRS) return send(chat, 'Слишком много пар.');
        const newIdx = brain.pairs.length;
        brain.pairs.push({ question: q, answer: a, qWords: null });
        indexPair(brain, newIdx, q, a);
      }
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
      unindexPair(brain, idx);
      brain.pairs.splice(idx, 1);
      rebuildIndex(brain);
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
    for (const o of others) addSyn(brain, main, o);
    try {
      await dbSaveSyn(owner, main, brain.syn.get(main) || new Set());
      for (const o of others) {
        const s = brain.syn.get(o);
        if (s) await dbSaveSyn(owner, o, s);
      }
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
    delSyn(brain, a, b);
    try {
      const sa = brain.syn.get(a), sb = brain.syn.get(b);
      if (sa) await dbSaveSyn(owner, a, sa);
      if (sb) await dbSaveSyn(owner, b, sb);
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
  console.log('Generative bot started');
  poll();
}

start();
