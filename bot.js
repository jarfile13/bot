const https = require('https');
const http = require('http');
const Parse = require('parse/node');

const BOT_TOKEN = process.env.BOT_TOKEN;
const APP_ID = process.env.B4A_APP_ID;
const JS_KEY = process.env.B4A_JS_KEY;
const MASTER_KEY = process.env.B4A_MASTER_KEY;

const BM25_K1 = 1.5;
const BM25_B = 0.6;
const MIN_SCORE = 0.6;
const MIN_MARGIN = 0.15;
const MIN_COVERAGE = 0.34;
const MIN_SINGLE_LEN = 2;
const MAX_PAIRS_PER_USER = 5000;
const CACHE_LIMIT = 300;
const CANDIDATE_LIMIT = 800;
const MAX_ACTIVE_BRAINS = 200;
const TEACH_RATE_LIMIT = 20;
const TEACH_RATE_WINDOW = 60 * 1000;

const healthServer = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK');
});
healthServer.listen(3000, () => console.log('Health check server on port 3000'));

Parse.initialize(APP_ID, JS_KEY, MASTER_KEY);
Parse.serverURL = 'https://parseapi.back4app.com/';

const brains = new Map();
const brainOrder = [];

let offset = 0;

const SUFFIXES = [
  'иями','ями','ами','ией','иях','ях','ов','ев','ий','ый','ой','ая','ое','ые','ыми','ими',
  'ешь','ишь','ете','ите','ешься','ишься','ется','ится','ются','атся','ться','тся',
  'ого','его','ому','ему','ыми','ими','ать','ять','еть','ить','ыть','уть',
  'ах','ях','ам','ям','ом','ем','ой','ей','ую','юю','ии','ия','ие','ые','ая','яя',
  'ть','ся','ла','ло','ли','ны','на','но','ет','ут','ют','ат','ят','ал','ил','ел',
  'а','я','у','ю','о','е','ы','и','й','ь'
];

function lc(s) {
  return String(s)
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function stem(w) {
  if (w.length <= 3) return w;
  for (const suf of SUFFIXES) {
    if (w.length - suf.length >= 3 && w.endsWith(suf)) {
      return w.slice(0, w.length - suf.length);
    }
  }
  return w;
}

function words(s) {
  const n = lc(s);
  if (!n) return [];
  const out = [];
  for (const w of n.split(' ')) if (w.length > 0) out.push(w);
  return out;
}

function terms(text) {
  const ws = words(text);
  const out = [];
  for (let i = 0; i < ws.length; i++) {
    out.push(stem(ws[i]));
    if (i + 1 < ws.length) out.push(stem(ws[i]) + '_' + stem(ws[i + 1]));
  }
  return out;
}

function levenshtein(a, b) {
  if (a === b) return 0;
  const al = a.length, bl = b.length;
  if (!al) return bl;
  if (!bl) return al;
  if (Math.abs(al - bl) > 2) return Infinity;
  let prev = new Array(bl + 1);
  let cur = new Array(bl + 1);
  for (let j = 0; j <= bl; j++) prev[j] = j;
  for (let i = 1; i <= al; i++) {
    cur[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= bl; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    const t = prev; prev = cur; cur = t;
  }
  return prev[bl];
}

function makeBrain(ownerId) {
  return {
    ownerId,
    pairs: [],
    wordToPairs: new Map(),
    docFreq: new Map(),
    byLen: new Map(),
    vocab: new Set(),
    cache: new Map(),
    totalDocLen: 0,
    cacheHits: 0,
    cacheMiss: 0,
    loaded: false,
    loading: null,
    teachTimestamps: [],
  };
}

function touchBrain(ownerId) {
  const idx = brainOrder.indexOf(ownerId);
  if (idx >= 0) brainOrder.splice(idx, 1);
  brainOrder.push(ownerId);
  while (brainOrder.length > MAX_ACTIVE_BRAINS) {
    const drop = brainOrder.shift();
    if (drop !== ownerId) brains.delete(drop);
  }
}

function getBrain(ownerId) {
  let b = brains.get(ownerId);
  if (!b) {
    b = makeBrain(ownerId);
    brains.set(ownerId, b);
  }
  touchBrain(ownerId);
  return b;
}

function registerWord(brain, w) {
  if (!brain.vocab.has(w)) {
    brain.vocab.add(w);
    if (!brain.byLen.has(w.length)) brain.byLen.set(w.length, new Set());
    brain.byLen.get(w.length).add(w);
  }
}

function findClosestKnown(brain, word) {
  if (brain.vocab.has(word)) return { word, dist: 0 };
  if (word.length < 3) return null;
  let best = null;
  let bestDist = Infinity;
  const maxDist = word.length <= 5 ? 1 : word.length <= 8 ? 2 : 3;
  for (let len = word.length - 2; len <= word.length + 2; len++) {
    const bucket = brain.byLen.get(len);
    if (!bucket) continue;
    for (const k of bucket) {
      const d = levenshtein(word, k);
      if (d < bestDist) {
        bestDist = d;
        best = k;
        if (d === 1) return { word: best, dist: 1 };
      }
    }
  }
  if (best && bestDist <= maxDist) return { word: best, dist: bestDist };
  return null;
}

function idf(brain, term) {
  const df = brain.docFreq.get(term) || 0;
  if (df === 0) return 0;
  return Math.log(1 + (brain.pairs.length - df + 0.5) / (df + 0.5));
}

function indexPair(brain, idx, question) {
  const ts = terms(question);
  brain.pairs[idx].tokens = ts;
  brain.pairs[idx].len = ts.length;
  brain.totalDocLen += ts.length;
  const seen = new Set();
  for (const t of ts) {
    registerWord(brain, t);
    if (!brain.wordToPairs.has(t)) brain.wordToPairs.set(t, new Set());
    brain.wordToPairs.get(t).add(idx);
    if (!seen.has(t)) {
      seen.add(t);
      brain.docFreq.set(t, (brain.docFreq.get(t) || 0) + 1);
    }
  }
}

function unindexPair(brain, idx) {
  const p = brain.pairs[idx];
  if (!p || !p.tokens) return;
  brain.totalDocLen -= p.len;
  const seen = new Set();
  for (const t of p.tokens) {
    const s = brain.wordToPairs.get(t);
    if (s) {
      s.delete(idx);
      if (!s.size) brain.wordToPairs.delete(t);
    }
    if (!seen.has(t)) {
      seen.add(t);
      const df = (brain.docFreq.get(t) || 0) - 1;
      if (df <= 0) brain.docFreq.delete(t);
      else brain.docFreq.set(t, df);
    }
  }
  p.tokens = null;
}

function avgDocLen(brain) {
  return brain.pairs.length ? brain.totalDocLen / brain.pairs.length : 1;
}

function scoreBM25(brain, inputTerms, pairIdx) {
  const pair = brain.pairs[pairIdx];
  if (!pair || !pair.tokens) return { score: 0, matched: 0 };

  const qCount = new Map();
  for (const t of inputTerms) qCount.set(t, (qCount.get(t) || 0) + 1);

  const docCount = new Map();
  for (const t of pair.tokens) docCount.set(t, (docCount.get(t) || 0) + 1);

  const avgdl = avgDocLen(brain) || 1;
  let score = 0;
  let matched = 0;

  for (const [t, qc] of qCount) {
    const f = docCount.get(t);
    if (!f) continue;
    matched++;
    const w = idf(brain, t);
    const denom = f + BM25_K1 * (1 - BM25_B + BM25_B * (pair.len / avgdl));
    score += w * (f * (BM25_K1 + 1)) / denom * Math.min(qc, 2);
  }

  const norm = Math.sqrt(inputTerms.length) || 1;
  return { score: score / norm, matched };
}

function resolveInput(brain, input) {
  const rawTerms = terms(input);
  if (!rawTerms.length) return null;

  const resolved = [];
  for (const t of rawTerms) {
    if (brain.vocab.has(t)) {
      resolved.push({ term: t, penalty: 1 });
      continue;
    }
    const hit = findClosestKnown(brain, t);
    if (hit) {
      resolved.push({ term: hit.word, penalty: hit.dist === 0 ? 1 : hit.dist === 1 ? 0.85 : 0.7 });
      continue;
    }
    if (t.includes('_')) {
      const parts = t.split('_');
      let ok = true;
      const mapped = [];
      for (const p of parts) {
        if (brain.vocab.has(p)) mapped.push(p);
        else {
          const h = findClosestKnown(brain, p);
          if (h) mapped.push(h.word);
          else { ok = false; break; }
        }
      }
      if (ok && mapped.length) {
        resolved.push({ term: mapped.join('_'), penalty: 0.7 });
        continue;
      }
    }
    if (t.length >= MIN_SINGLE_LEN) {
      resolved.push({ term: t, penalty: 0.4 });
    }
  }

  return resolved.length ? resolved : null;
}

function answerFor(brain, input) {
  if (!brain.pairs.length) return null;
  const key = lc(input);

  if (brain.cache.has(key)) {
    brain.cacheHits++;
    const v = brain.cache.get(key);
    brain.cache.delete(key);
    brain.cache.set(key, v);
    return v;
  }
  brain.cacheMiss++;

  const resolved = resolveInput(brain, input);
  if (!resolved) {
    if (brain.cache.size >= CACHE_LIMIT) brain.cache.delete(brain.cache.keys().next().value);
    brain.cache.set(key, null);
    return null;
  }

  const inputTerms = resolved.map(r => r.term);
  const penaltyMap = new Map();
  for (const r of resolved) {
    const prev = penaltyMap.get(r.term) || 0;
    if (r.penalty > prev) penaltyMap.set(r.term, r.penalty);
  }

  const candidates = new Set();
  for (const t of inputTerms) {
    const idxs = brain.wordToPairs.get(t);
    if (idxs) for (const i of idxs) {
      candidates.add(i);
      if (candidates.size > CANDIDATE_LIMIT) break;
    }
    if (candidates.size > CANDIDATE_LIMIT) break;
  }

  if (!candidates.size) {
    if (brain.cache.size >= CACHE_LIMIT) brain.cache.delete(brain.cache.keys().next().value);
    brain.cache.set(key, null);
    return null;
  }

  const scored = [];
  for (const i of candidates) {
    const { score, matched } = scoreBM25(brain, inputTerms, i);
    if (score <= 0) continue;
    const coverage = matched / inputTerms.length;
    if (coverage < MIN_COVERAGE) continue;
    let final = score;
    for (const t of inputTerms) {
      const p = penaltyMap.get(t);
      if (p && p < 1) final *= p;
    }
    scored.push({ idx: i, score: final, coverage });
  }

  if (!scored.length) {
    if (brain.cache.size >= CACHE_LIMIT) brain.cache.delete(brain.cache.keys().next().value);
    brain.cache.set(key, null);
    return null;
  }

  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1];

  let result = null;
  if (best.score >= MIN_SCORE && (!second || best.score - second.score >= MIN_MARGIN)) {
    result = { pair: brain.pairs[best.idx], score: best.score, coverage: best.coverage };
  } else if (best.score >= MIN_SCORE && second && best.score - second.score < MIN_MARGIN) {
    result = { ambiguous: scored.slice(0, 3).map(s => brain.pairs[s.idx]) };
  }

  if (brain.cache.size >= CACHE_LIMIT) brain.cache.delete(brain.cache.keys().next().value);
  brain.cache.set(key, result);
  return result;
}

function addPairLocal(brain, q, a) {
  const idx = brain.pairs.length;
  brain.pairs.push({ question: q, answer: a, tokens: null, len: 0 });
  indexPair(brain, idx, q);
  return idx;
}

function updatePairLocal(brain, idx, q, a) {
  unindexPair(brain, idx);
  brain.pairs[idx].question = q;
  brain.pairs[idx].answer = a;
  indexPair(brain, idx, q);
}

function rebuildLocalIndex(brain) {
  brain.wordToPairs.clear();
  brain.docFreq.clear();
  brain.byLen.clear();
  brain.vocab.clear();
  brain.totalDocLen = 0;
  brain.cache.clear();
  for (let i = 0; i < brain.pairs.length; i++) {
    brain.pairs[i].tokens = null;
    brain.pairs[i].len = 0;
    indexPair(brain, i, brain.pairs[i].question);
  }
}

async function savePairToDb(ownerId, question, answer) {
  const Knowledge = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(Knowledge);
  query.equalTo('ownerId', ownerId);
  query.equalTo('question', lc(question));
  let obj = await query.first({ useMasterKey: true });
  if (!obj) {
    obj = new Knowledge();
    obj.set('ownerId', ownerId);
    obj.set('question', lc(question));
  }
  obj.set('answer', answer);
  obj.set('originalQuestion', question);
  return obj.save(null, { useMasterKey: true });
}

async function deletePairFromDb(ownerId, question) {
  const Knowledge = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(Knowledge);
  query.equalTo('ownerId', ownerId);
  query.equalTo('question', lc(question));
  const obj = await query.first({ useMasterKey: true });
  if (obj) await obj.destroy({ useMasterKey: true });
}

async function loadPairsForOwner(ownerId) {
  const Knowledge = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(Knowledge);
  query.equalTo('ownerId', ownerId);
  query.limit(10000);
  const results = await query.find({ useMasterKey: true });
  return results.map((r) => ({
    question: r.get('originalQuestion') || r.get('question'),
    answer: r.get('answer'),
  }));
}

async function ensureBrainLoaded(brain) {
  if (brain.loaded) return brain;
  if (brain.loading) return brain.loading;
  brain.loading = (async () => {
    const rows = await loadPairsForOwner(brain.ownerId);
    brain.pairs = rows.map(r => ({ question: r.question, answer: r.answer, tokens: null, len: 0 }));
    rebuildLocalIndex(brain);
    brain.loaded = true;
    console.log('Brain loaded for', brain.ownerId, 'pairs:', brain.pairs.length);
    return brain;
  })();
  try {
    return await brain.loading;
  } finally {
    brain.loading = null;
  }
}

function checkTeachRate(brain) {
  const now = Date.now();
  brain.teachTimestamps = brain.teachTimestamps.filter(t => now - t < TEACH_RATE_WINDOW);
  if (brain.teachTimestamps.length >= TEACH_RATE_LIMIT) return false;
  brain.teachTimestamps.push(now);
  return true;
}

function api(method, payload) {
  return new Promise((resolve, reject) => {
    if (!BOT_TOKEN) return reject(new Error('BOT_TOKEN not set'));
    const data = JSON.stringify(payload || {});
    const req = https.request(
      {
        hostname: 'api.telegram.org',
        path: `/bot${BOT_TOKEN}/${method}`,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data),
        },
      },
      (res) => {
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            reject(e);
          }
        });
      }
    );
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function send(chatId, text) {
  try {
    await api('sendMessage', { chat_id: chatId, text });
  } catch (e) {
    console.error('send error:', e.message);
  }
}

const HELP = [
  'Личный бот. У каждого свой список.',
  '',
  'Команды:',
  '/help - справка',
  '/teach вопрос = ответ - научить',
  '/delete вопрос - забыть',
  '/list - что знаю',
  '/stats - статистика',
  '',
  'Твои пары видишь только ты.',
].join('\n');

async function handle(chatId, text) {
  const t = text.trim();
  const ownerId = String(chatId);
  const brain = getBrain(ownerId);

  if (t === '/start' || t === '/help') {
    await send(chatId, HELP);
    return;
  }

  if (t === '/stats') {
    await ensureBrainLoaded(brain);
    const top = [...brain.docFreq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)
      .map(([w, c]) => `${w}(${c})`).join(', ');
    const hitRate = brain.cacheHits + brain.cacheMiss
      ? (brain.cacheHits / (brain.cacheHits + brain.cacheMiss) * 100).toFixed(1)
      : '0.0';
    await send(
      chatId,
      `Твоих пар: ${brain.pairs.length}\nСлов: ${brain.vocab.size}\nСр.длина: ${avgDocLen(brain).toFixed(1)}\nКэш: ${brain.cache.size} (hit ${hitRate}%)\nТоп: ${top || '-'}`
    );
    return;
  }

  if (t === '/list') {
    await ensureBrainLoaded(brain);
    if (!brain.pairs.length) {
      await send(chatId, 'У тебя пусто.');
      return;
    }
    const lines = brain.pairs.slice(-100).map((p, i) => `${i + 1}. ${p.question} = ${p.answer}`);
    let buf = '';
    for (const line of lines) {
      if ((buf + line + '\n').length > 3500) {
        await send(chatId, buf);
        buf = '';
      }
      buf += line + '\n';
    }
    if (buf) await send(chatId, buf);
    return;
  }

  if (t.startsWith('/teach')) {
    const rest = t.slice('/teach'.length).trim();
    const parts = rest.split('=');
    if (parts.length < 2) {
      await send(chatId, 'Формат: /teach вопрос = ответ');
      return;
    }
    const q = parts[0].trim();
    const a = parts.slice(1).join('=').trim();
    if (!q || !a) {
      await send(chatId, 'Пусто.');
      return;
    }
    await ensureBrainLoaded(brain);
    if (!checkTeachRate(brain)) {
      await send(chatId, 'Слишком часто. Подожди минуту.');
      return;
    }
    try {
      await savePairToDb(ownerId, q, a);
      const key = lc(q);
      const existingIdx = brain.pairs.findIndex((p) => lc(p.question) === key);
      if (existingIdx >= 0) {
        updatePairLocal(brain, existingIdx, q, a);
      } else {
        if (brain.pairs.length >= MAX_PAIRS_PER_USER) {
          await send(chatId, 'У тебя слишком много пар, удали что-нибудь.');
          return;
        }
        addPairLocal(brain, q, a);
      }
      brain.cache.clear();
      await send(chatId, `Запомнил: ${q} = ${a}`);
    } catch (e) {
      console.error('teach error:', e.message);
      await send(chatId, 'Ошибка сохранения.');
    }
    return;
  }

  if (t.startsWith('/delete')) {
    const rest = t.slice('/delete'.length).trim();
    if (!rest) {
      await send(chatId, 'Формат: /delete вопрос');
      return;
    }
    await ensureBrainLoaded(brain);
    const key = lc(rest);
    const idx = brain.pairs.findIndex((p) => lc(p.question) === key);
    if (idx < 0) {
      await send(chatId, `Не нашёл у тебя: ${key}`);
      return;
    }
    try {
      await deletePairFromDb(ownerId, brain.pairs[idx].question);
      brain.pairs.splice(idx, 1);
      rebuildLocalIndex(brain);
      await send(chatId, `Удалил: ${rest}`);
    } catch (e) {
      console.error('delete error:', e.message);
      await send(chatId, 'Ошибка удаления.');
    }
    return;
  }

  await ensureBrainLoaded(brain);
  const res = answerFor(brain, t);
  if (!res) {
    await send(chatId, 'Не знаю. Научи: /teach вопрос = ответ');
    return;
  }
  if (res.ambiguous) {
    const opts = res.ambiguous.map((p, i) => `${i + 1}. ${p.answer}`).join('\n');
    await send(chatId, `Уточни, я знаю несколько вариантов:\n${opts}`);
    return;
  }
  await send(chatId, res.pair.answer);
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
          try {
            await handle(m.chat.id, m.text);
          } catch (e) {
            console.error('handle error:', e.message);
          }
        }
      }
    } catch (e) {
      console.error('poll error:', e.message);
      await new Promise((res) => setTimeout(res, 3000));
    }
  }
}

async function start() {
  if (!BOT_TOKEN) {
    console.error('BOT_TOKEN not set');
    process.exit(1);
  }
  if (!APP_ID || !JS_KEY || !MASTER_KEY) {
    console.error('Back4App keys not set');
    process.exit(1);
  }

  console.log('Bot started, multi-tenant mode');
  poll();
}

start();
