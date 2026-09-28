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
const MAX_PAIRS = 20000;
const CACHE_LIMIT = 500;
const CANDIDATE_LIMIT = 800;

const healthServer = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK');
});
healthServer.listen(3000, () => console.log('Health check server on port 3000'));

Parse.initialize(APP_ID, JS_KEY, MASTER_KEY);
Parse.serverURL = 'https://parseapi.back4app.com/';

const pairs = [];
const wordToPairs = new Map();
const docFreq = new Map();
const byLen = new Map();
const vocab = new Set();
const cache = new Map();

let totalDocLen = 0;
let offset = 0;
let cacheHits = 0;
let cacheMiss = 0;

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
  for (const w of n.split(' ')) {
    if (w.length > 0) out.push(w);
  }
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

function registerWord(w) {
  if (!vocab.has(w)) {
    vocab.add(w);
    if (!byLen.has(w.length)) byLen.set(w.length, new Set());
    byLen.get(w.length).add(w);
  }
}

function findClosestKnown(word) {
  if (vocab.has(word)) return { word, dist: 0 };
  if (word.length < 3) return null;
  let best = null;
  let bestDist = Infinity;
  const maxDist = word.length <= 5 ? 1 : word.length <= 8 ? 2 : 3;
  for (let len = word.length - 2; len <= word.length + 2; len++) {
    const bucket = byLen.get(len);
    if (!bucket) continue;
    for (const k of bucket) {
      if (Math.abs(k.charCodeAt(0) - word.charCodeAt(0)) > 0 && k[0] !== word[0] && bestDist > 0) {
        // быстрый префиксный отсев, но не блокируем полностью
      }
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

function idf(term) {
  const df = docFreq.get(term) || 0;
  if (df === 0) return 0;
  return Math.log(1 + (pairs.length - df + 0.5) / (df + 0.5));
}

function indexPair(idx, question) {
  const ts = terms(question);
  pairs[idx].tokens = ts;
  pairs[idx].len = ts.length;
  totalDocLen += ts.length;
  const seen = new Set();
  for (const t of ts) {
    registerWord(t);
    if (!wordToPairs.has(t)) wordToPairs.set(t, new Set());
    wordToPairs.get(t).add(idx);
    if (!seen.has(t)) {
      seen.add(t);
      docFreq.set(t, (docFreq.get(t) || 0) + 1);
    }
  }
}

function unindexPair(idx) {
  const p = pairs[idx];
  if (!p || !p.tokens) return;
  totalDocLen -= p.len;
  const seen = new Set();
  for (const t of p.tokens) {
    const s = wordToPairs.get(t);
    if (s) {
      s.delete(idx);
      if (!s.size) wordToPairs.delete(t);
    }
    if (!seen.has(t)) {
      seen.add(t);
      const df = (docFreq.get(t) || 0) - 1;
      if (df <= 0) docFreq.delete(t);
      else docFreq.set(t, df);
    }
  }
  p.tokens = null;
}

function avgDocLen() {
  return pairs.length ? totalDocLen / pairs.length : 1;
}

function scoreBM25(inputTerms, pairIdx) {
  const pair = pairs[pairIdx];
  if (!pair || !pair.tokens) return { score: 0, matched: 0 };

  const qCount = new Map();
  for (const t of inputTerms) qCount.set(t, (qCount.get(t) || 0) + 1);

  const docCount = new Map();
  for (const t of pair.tokens) docCount.set(t, (docCount.get(t) || 0) + 1);

  const avgdl = avgDocLen() || 1;
  let score = 0;
  let matched = 0;

  for (const [t, qc] of qCount) {
    const f = docCount.get(t);
    if (!f) continue;
    matched++;
    const w = idf(t);
    const denom = f + BM25_K1 * (1 - BM25_B + BM25_B * (pair.len / avgdl));
    score += w * (f * (BM25_K1 + 1)) / denom * Math.min(qc, 2);
  }

  const norm = Math.sqrt(inputTerms.length) || 1;
  return { score: score / norm, matched };
}

function resolveInput(input) {
  const rawTerms = terms(input);
  if (!rawTerms.length) return null;

  const resolved = [];
  for (const t of rawTerms) {
    if (vocab.has(t)) {
      resolved.push({ term: t, penalty: 1 });
      continue;
    }
    const hit = findClosestKnown(t);
    if (hit) {
      resolved.push({ term: hit.word, penalty: hit.dist === 0 ? 1 : hit.dist === 1 ? 0.85 : 0.7 });
      continue;
    }
    if (t.includes('_')) {
      const parts = t.split('_');
      let ok = true;
      const mapped = [];
      for (const p of parts) {
        if (vocab.has(p)) mapped.push(p);
        else {
          const h = findClosestKnown(p);
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

function answerFor(input) {
  if (!pairs.length) return null;
  const key = lc(input);

  if (cache.has(key)) {
    cacheHits++;
    const v = cache.get(key);
    cache.delete(key);
    cache.set(key, v);
    return v;
  }
  cacheMiss++;

  const resolved = resolveInput(input);
  if (!resolved) {
    if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value);
    cache.set(key, null);
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
    const idxs = wordToPairs.get(t);
    if (idxs) for (const i of idxs) {
      candidates.add(i);
      if (candidates.size > CANDIDATE_LIMIT) break;
    }
    if (candidates.size > CANDIDATE_LIMIT) break;
  }

  if (!candidates.size) {
    if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value);
    cache.set(key, null);
    return null;
  }

  const scored = [];
  for (const i of candidates) {
    const { score, matched } = scoreBM25(inputTerms, i);
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
    if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value);
    cache.set(key, null);
    return null;
  }

  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1];

  let result = null;
  if (best.score >= MIN_SCORE && (!second || best.score - second.score >= MIN_MARGIN)) {
    result = { pair: pairs[best.idx], score: best.score, coverage: best.coverage };
  } else if (best.score >= MIN_SCORE && second && best.score - second.score < MIN_MARGIN) {
    result = { ambiguous: scored.slice(0, 3).map(s => pairs[s.idx]) };
  }

  if (cache.size >= CACHE_LIMIT) cache.delete(cache.keys().next().value);
  cache.set(key, result);
  return result;
}

async function savePairToDb(question, answer) {
  const Knowledge = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(Knowledge);
  query.equalTo('question', lc(question));
  let obj = await query.first({ useMasterKey: true });
  if (!obj) {
    obj = new Knowledge();
    obj.set('question', lc(question));
  }
  obj.set('answer', answer);
  obj.set('originalQuestion', question);
  return obj.save(null, { useMasterKey: true });
}

async function deletePairFromDb(question) {
  const Knowledge = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(Knowledge);
  query.equalTo('question', lc(question));
  const obj = await query.first({ useMasterKey: true });
  if (obj) await obj.destroy({ useMasterKey: true });
}

async function loadAllFromDb() {
  const Knowledge = Parse.Object.extend('Knowledge');
  const query = new Parse.Query(Knowledge);
  query.limit(10000);
  const results = await query.find({ useMasterKey: true });
  return results.map((r) => ({
    id: r.id,
    question: r.get('originalQuestion') || r.get('question'),
    answer: r.get('answer'),
  }));
}

function addPair(q, a) {
  const idx = pairs.length;
  pairs.push({ question: q, answer: a, tokens: null, len: 0 });
  indexPair(idx, q);
  return idx;
}

function updatePair(idx, q, a) {
  unindexPair(idx);
  pairs[idx].question = q;
  pairs[idx].answer = a;
  indexPair(idx, q);
}

function rebuildIndex() {
  wordToPairs.clear();
  docFreq.clear();
  byLen.clear();
  vocab.clear();
  totalDocLen = 0;
  cache.clear();
  for (let i = 0; i < pairs.length; i++) {
    pairs[i].tokens = null;
    pairs[i].len = 0;
    indexPair(i, pairs[i].question);
  }
}

async function rebuildFromDb() {
  pairs.length = 0;
  const dbPairs = await loadAllFromDb();
  for (const p of dbPairs) pairs.push({ question: p.question, answer: p.answer, tokens: null, len: 0 });
  rebuildIndex();
  console.log('Loaded from DB:', pairs.length, 'pairs');
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
  'Команды:',
  '/help - справка',
  '/teach вопрос = ответ - научить',
  '/delete вопрос - забыть',
  '/list - что знаю',
  '/stats - статистика',
  '',
  'Данные хранятся в Back4App, перезапуск не страшен.',
].join('\n');

async function handle(chatId, text) {
  const t = text.trim();

  if (t === '/start' || t === '/help') {
    await send(chatId, HELP);
    return;
  }

  if (t === '/stats') {
    const top = [...docFreq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)
      .map(([w, c]) => `${w}(${c})`).join(', ');
    const hitRate = cacheHits + cacheMiss ? (cacheHits / (cacheHits + cacheMiss) * 100).toFixed(1) : '0.0';
    await send(
      chatId,
      `Пар: ${pairs.length}\nСлов: ${vocab.size}\nСр.длина: ${avgDocLen().toFixed(1)}\nКэш: ${cache.size} (hit ${hitRate}%)\nТоп: ${top || '-'}`
    );
    return;
  }

  if (t === '/list') {
    if (!pairs.length) {
      await send(chatId, 'Пусто.');
      return;
    }
    const lines = pairs.slice(-100).map((p, i) => `${i + 1}. ${p.question} = ${p.answer}`);
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
    try {
      await savePairToDb(q, a);
      const key = lc(q);
      const existingIdx = pairs.findIndex((p) => lc(p.question) === key);
      if (existingIdx >= 0) {
        updatePair(existingIdx, q, a);
      } else {
        if (pairs.length >= MAX_PAIRS) {
          await send(chatId, 'Слишком много пар, удали что-нибудь.');
          return;
        }
        addPair(q, a);
      }
      cache.clear();
      await send(chatId, `Запомнил: ${q} = ${a}`);
    } catch (e) {
      console.error('teach error:', e.message);
      await send(chatId, 'Ошибка сохранения в базу.');
    }
    return;
  }

  if (t.startsWith('/delete')) {
    const rest = t.slice('/delete'.length).trim();
    if (!rest) {
      await send(chatId, 'Формат: /delete вопрос');
      return;
    }
    const key = lc(rest);
    const idx = pairs.findIndex((p) => lc(p.question) === key);
    if (idx < 0) {
      await send(chatId, `Не нашёл: ${key}`);
      return;
    }
    try {
      await deletePairFromDb(pairs[idx].question);
      pairs.splice(idx, 1);
      rebuildIndex();
      await send(chatId, `Удалил: ${rest}`);
    } catch (e) {
      console.error('delete error:', e.message);
      await send(chatId, 'Ошибка удаления.');
    }
    return;
  }

  const res = answerFor(t);
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

  try {
    await rebuildFromDb();
    if (pairs.length === 0) {
      console.log('Empty DB, seeding...');
      const seeds = [
        ['привет', 'Здорово друг'],
        ['здравствуй', 'Здорово друг'],
        ['как дела', 'Отлично а у тебя'],
        ['пока', 'До встречи'],
      ];
      for (const [q, a] of seeds) await savePairToDb(q, a);
      await rebuildFromDb();
    }
    console.log('Brain ready, pairs:', pairs.length);
  } catch (e) {
    console.error('start error:', e.message);
  }

  console.log('Bot started');
  poll();
}

start();
