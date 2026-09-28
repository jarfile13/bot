const https = require('https');
const http = require('http');
const Parse = require('parse/node');

const BOT_TOKEN = process.env.BOT_TOKEN;
const APP_ID = process.env.B4A_APP_ID;
const JS_KEY = process.env.B4A_JS_KEY;
const MASTER_KEY = process.env.B4A_MASTER_KEY;

const MIN_SCORE = 0.35;
const MIN_SINGLE_LEN = 5;
const MIN_IDF_FOR_INPUT = 0.8;
const MAX_PAIRS = 20000;
const CACHE_LIMIT = 500;

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
const answerCache = new Map();

let offset = 0;

function lc(s) {
  return String(s)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function words(s) {
  const n = lc(s);
  if (!n) return [];
  const out = [];
  for (const w of n.split(' ')) {
    if (w.length > 1) out.push(w);
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
      const v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      cur[j] = v;
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
  if (vocab.has(word)) return word;
  if (word.length < 3) return null;
  let best = null;
  let bestDist = Infinity;
  const maxDist = Math.max(1, Math.floor(word.length / 3));
  for (let len = word.length - 2; len <= word.length + 2; len++) {
    const bucket = byLen.get(len);
    if (!bucket) continue;
    for (const k of bucket) {
      const d = levenshtein(word, k);
      if (d < bestDist) {
        bestDist = d;
        best = k;
        if (d === 1) return best;
      }
    }
  }
  if (best && bestDist <= maxDist) return best;
  return null;
}

function tokenizeForIndex(text) {
  const out = [];
  for (const w of words(text)) {
    const k = findClosestKnown(w);
    out.push(k || w);
  }
  return out;
}

function indexPair(idx, question) {
  const qw = tokenizeForIndex(question);
  const seen = new Set();
  for (const w of qw) {
    registerWord(w);
    if (!wordToPairs.has(w)) wordToPairs.set(w, new Set());
    wordToPairs.get(w).add(idx);
    if (!seen.has(w)) {
      seen.add(w);
      docFreq.set(w, (docFreq.get(w) || 0) + 1);
    }
  }
}

function removePairFromIndex(idx, question) {
  const qw = tokenizeForIndex(question);
  const seen = new Set();
  for (const w of qw) {
    const s = wordToPairs.get(w);
    if (s) {
      s.delete(idx);
      if (!s.size) wordToPairs.delete(w);
    }
    if (!seen.has(w)) {
      seen.add(w);
      const df = (docFreq.get(w) || 0) - 1;
      if (df <= 0) docFreq.delete(w);
      else docFreq.set(w, df);
    }
  }
}

function idf(w) {
  const df = docFreq.get(w) || 0;
  if (df === 0) return 0;
  return Math.log(1 + pairs.length / df);
}

function scorePair(inputTokens, pairIdx) {
  const pair = pairs[pairIdx];
  if (!pair) return 0;
  const qTokens = pair.tokens;
  if (!qTokens.length) return 0;

  const qCount = new Map();
  for (const t of qTokens) qCount.set(t, (qCount.get(t) || 0) + 1);

  const iCount = new Map();
  for (const t of inputTokens) iCount.set(t, (iCount.get(t) || 0) + 1);

  let dot = 0;
  let qNorm = 0;
  let iNorm = 0;

  for (const [t, qc] of qCount) {
    const w = idf(t);
    qNorm += (qc * w) * (qc * w);
    const ic = iCount.get(t) || 0;
    if (ic) dot += (qc * w) * (ic * w);
  }
  for (const [t, ic] of iCount) {
    const w = idf(t);
    iNorm += (ic * w) * (ic * w);
  }
  if (qNorm === 0 || iNorm === 0) return 0;
  return dot / (Math.sqrt(qNorm) * Math.sqrt(iNorm));
}

function answerFor(input) {
  if (!pairs.length) return null;
  const cacheKey = lc(input);
  if (answerCache.has(cacheKey)) return answerCache.get(cacheKey);

  const rawTokens = words(input);
  if (!rawTokens.length) return null;
  if (rawTokens.length === 1 && rawTokens[0].length < MIN_SINGLE_LEN) {
    answerCache.set(cacheKey, null);
    return null;
  }

  const inputTokens = [];
  for (const w of rawTokens) {
    const k = findClosestKnown(w);
    const token = k || w;
    if (idf(token) < MIN_IDF_FOR_INPUT) continue;
    inputTokens.push(token);
  }
  if (!inputTokens.length) {
    answerCache.set(cacheKey, null);
    return null;
  }

  const candidateSet = new Set();
  for (const t of inputTokens) {
    const idxs = wordToPairs.get(t);
    if (idxs) for (const i of idxs) candidateSet.add(i);
    if (candidateSet.size > 500) break;
  }
  if (!candidateSet.size) {
    answerCache.set(cacheKey, null);
    return null;
  }

  let best = null;
  let bestScore = 0;
  for (const i of candidateSet) {
    const s = scorePair(inputTokens, i);
    if (s > bestScore) {
      bestScore = s;
      best = i;
    }
  }

  let result = null;
  if (best !== null && bestScore >= MIN_SCORE) {
    result = { pair: pairs[best], score: bestScore };
  }

  if (answerCache.size >= CACHE_LIMIT) {
    const firstKey = answerCache.keys().next().value;
    answerCache.delete(firstKey);
  }
  answerCache.set(cacheKey, result);
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
  const tokens = tokenizeForIndex(q);
  const idx = pairs.length;
  pairs.push({ question: q, answer: a, tokens });
  indexPair(idx, q);
  return idx;
}

function updatePair(idx, q, a) {
  removePairFromIndex(idx, pairs[idx].question);
  const tokens = tokenizeForIndex(q);
  pairs[idx] = { question: q, answer: a, tokens };
  indexPair(idx, q);
}

function reindexFrom(startIdx) {
  for (let i = startIdx; i < pairs.length; i++) {
    indexPair(i, pairs[i].question);
  }
}

async function rebuildFromDb() {
  pairs.length = 0;
  wordToPairs.clear();
  docFreq.clear();
  byLen.clear();
  vocab.clear();
  answerCache.clear();

  const dbPairs = await loadAllFromDb();
  for (const p of dbPairs) addPair(p.question, p.answer);
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
    await send(
      chatId,
      `Пар: ${pairs.length}\nСлов: ${vocab.size}\nКэш: ${answerCache.size}`
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
      answerCache.clear();
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
      removePairFromIndex(idx, pairs[idx].question);
      pairs.splice(idx, 1);
      wordToPairs.clear();
      docFreq.clear();
      for (let i = 0; i < pairs.length; i++) indexPair(i, pairs[i].question);
      answerCache.clear();
      await send(chatId, `Удалил: ${rest}`);
    } catch (e) {
      console.error('delete error:', e.message);
      await send(chatId, 'Ошибка удаления.');
    }
    return;
  }

  const res = answerFor(t);
  if (res) {
    await send(chatId, res.pair.answer);
  } else {
    await send(chatId, 'Не знаю. Научи: /teach вопрос = ответ');
  }
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
