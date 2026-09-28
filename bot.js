const https = require('https');
const http = require('http');
const Parse = require('parse/node');

const BOT_TOKEN = process.env.BOT_TOKEN;

const APP_ID = process.env.B4A_APP_ID;
const JS_KEY = process.env.B4A_JS_KEY;
const MASTER_KEY = process.env.B4A_MASTER_KEY;

const healthServer = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK');
});
healthServer.listen(3000, () => {
  console.log('Health check server on port 3000');
});

Parse.initialize(APP_ID, JS_KEY, MASTER_KEY);
Parse.serverURL = 'https://parseapi.back4app.com/';

const MAX_NEIGHBORS = 20;
const DECAY_THRESHOLD = 0.05;
const DECAY_EVERY = 200;

const nodes = new Map();
const edges = new Map();
const pairs = [];
const wordToPairs = new Map();

let offset = 0;
let teachesSinceDecay = 0;
let lastPairForChat = new Map();

function lc(s) {
  return String(s).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function words(s) {
  const n = lc(s);
  return n ? n.split(' ').filter((w) => w.length > 1) : [];
}

function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const prev = new Array(b.length + 1);
  const cur = new Array(b.length + 1);
  for (let j = 0; j <= b.length; j++) prev[j] = j;
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    for (let j = 0; j <= b.length; j++) prev[j] = cur[j];
  }
  return prev[b.length];
}

function ensureNode(w) {
  if (!nodes.has(w)) nodes.set(w, 0);
  if (!edges.has(w)) edges.set(w, new Map());
}

function link(a, b, strength) {
  if (a === b) return;
  ensureNode(a);
  ensureNode(b);
  const ea = edges.get(a);
  const eb = edges.get(b);
  ea.set(b, Math.min(1, (ea.get(b) || 0) + strength));
  eb.set(a, Math.min(1, (eb.get(a) || 0) + strength));
}

function charge(w, amount) {
  ensureNode(w);
  nodes.set(w, Math.min(1, nodes.get(w) + amount));
}

function indexWord(word, pairIndex) {
  if (!wordToPairs.has(word)) wordToPairs.set(word, new Set());
  wordToPairs.get(word).add(pairIndex);
}

function teachGraph(question, answer) {
  const qw = words(question);
  const aw = words(answer);
  if (!qw.length || !aw.length) return;

  for (const q of qw) {
    charge(q, 1.0);
    for (const a of aw) link(q, a, 0.3);
  }
  for (const a of aw) charge(a, 0.5);
  for (let i = 0; i < qw.length; i++)
    for (let j = i + 1; j < qw.length; j++) link(qw[i], qw[j], 0.5);
  for (let i = 0; i < aw.length; i++)
    for (let j = i + 1; j < aw.length; j++) link(aw[i], aw[j], 0.5);
}

function decay() {
  for (const [word, neighbors] of edges) {
    const entries = [...neighbors.entries()];
    for (const [nWord, strength] of entries) {
      const newStrength = strength * 0.98;
      if (newStrength < DECAY_THRESHOLD) {
        neighbors.delete(nWord);
        const reverse = edges.get(nWord);
        if (reverse) reverse.delete(word);
      } else {
        neighbors.set(nWord, newStrength);
      }
    }
    if (neighbors.size > MAX_NEIGHBORS) {
      const sorted = [...neighbors.entries()].sort((a, b) => b[1] - a[1]);
      const keep = sorted.slice(0, MAX_NEIGHBORS);
      neighbors.clear();
      for (const [w, s] of keep) neighbors.set(w, s);
    }
  }

  for (const [word, chargeVal] of nodes) {
    const newCharge = chargeVal * 0.95;
    if (newCharge < DECAY_THRESHOLD) nodes.delete(word);
    else nodes.set(word, newCharge);
  }
}

function findClosestKnown(word) {
  if (nodes.has(word)) return word;
  let best = null;
  let bestDist = Infinity;
  for (const k of nodes.keys()) {
    if (Math.abs(k.length - word.length) > 2) continue;
    const d = levenshtein(word, k);
    if (d < bestDist) {
      bestDist = d;
      best = k;
      if (d === 1) break;
    }
  }
  if (best && bestDist <= Math.max(1, Math.floor(word.length / 3))) return best;
  return null;
}

function spreadWave(startWords) {
  const activation = new Map();
  let frontier = [];

  for (const w of startWords) {
    const known = findClosestKnown(w);
    if (known) {
      activation.set(known, (activation.get(known) || 0) + 1);
      frontier.push({ word: known, energy: 1, depth: 0 });
    }
  }

  const MAX_DEPTH = 3;
  const DECAY_FACTOR = 0.5;
  const visited = new Map();

  while (frontier.length) {
    const next = [];
    for (const { word, energy, depth } of frontier) {
      if (depth >= MAX_DEPTH) continue;
      const prevE = visited.get(word) || 0;
      if (prevE >= energy) continue;
      visited.set(word, energy);

      const neighbors = edges.get(word);
      if (!neighbors) continue;
      for (const [nWord, strength] of neighbors) {
        const newEnergy = energy * strength * DECAY_FACTOR;
        if (newEnergy < 0.05) continue;
        activation.set(nWord, (activation.get(nWord) || 0) + newEnergy);
        next.push({ word: nWord, energy: newEnergy, depth: depth + 1 });
      }
    }
    frontier = next;
  }

  return activation;
}

function answerFor(input) {
  if (!pairs.length) return null;
  const iw = words(input);
  if (!iw.length) return null;

  const activation = spreadWave(iw);

  const candidateSet = new Set();
  for (const [word, energy] of activation) {
    if (energy < 0.1) continue;
    const idxs = wordToPairs.get(word);
    if (idxs) for (const i of idxs) candidateSet.add(i);
  }
  if (!candidateSet.size) return null;

  let bestPair = null;
  let bestScore = 0;

  for (const i of candidateSet) {
    const pair = pairs[i];
    if (!pair) continue;
    const qw = words(pair.question);
    let score = 0;
    for (const w of qw) {
      const known = findClosestKnown(w);
      if (known) score += activation.get(known) || 0;
    }
    score /= Math.sqrt(qw.length) || 1;
    if (score > bestScore) {
      bestScore = score;
      bestPair = pair;
    }
  }

  if (bestPair && bestScore >= 0.4) return { pair: bestPair, score: bestScore };
  return null;
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

async function rebuildFromDb() {
  pairs.length = 0;
  wordToPairs.clear();
  nodes.clear();
  edges.clear();

  const dbPairs = await loadAllFromDb();
  for (let i = 0; i < dbPairs.length; i++) {
    const p = dbPairs[i];
    pairs.push({ question: p.question, answer: p.answer });
    teachGraph(p.question, p.answer);
    const qw = words(p.question);
    for (const q of qw) indexWord(q, i);
  }
  console.log('Loaded from DB:', pairs.length, 'pairs');
}

function api(method, payload) {
  return new Promise((resolve, reject) => {
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
    const totalEdges = [...edges.values()].reduce((s, m) => s + m.size, 0) / 2;
    await send(
      chatId,
      `Пар: ${pairs.length}\nСлов: ${nodes.size}\nСвязей: ${Math.floor(totalEdges)}`
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
    const sep = rest.indexOf('=');
    if (sep === -1) {
      await send(chatId, 'Формат: /teach вопрос = ответ');
      return;
    }
    const q = rest.slice(0, sep).trim();
    const a = rest.slice(sep + 1).trim();
    if (!q || !a) {
      await send(chatId, 'Пусто.');
      return;
    }
    try {
      await savePairToDb(q, a);

      const existingIdx = pairs.findIndex((p) => lc(p.question) === lc(q));
      if (existingIdx >= 0) {
        pairs[existingIdx] = { question: q, answer: a };
      } else {
        pairs.push({ question: q, answer: a });
      }
      teachGraph(q, a);

      const idx = pairs.length - 1;
      for (const w of words(q)) indexWord(w, idx);

      teachesSinceDecay++;
      if (teachesSinceDecay >= DECAY_EVERY) {
        decay();
        teachesSinceDecay = 0;
      }

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
    if (idx >= 0) {
      try {
        await deletePairFromDb(pairs[idx].question);
        await rebuildFromDb();
        await send(chatId, `Удалил: ${rest}`);
      } catch (e) {
        console.error('delete error:', e.message);
        await send(chatId, 'Ошибка удаления.');
      }
      return;
    }
    await send(chatId, `Не нашёл: ${key}`);
    return;
  }

  const res = answerFor(t);
  if (res) {
    lastPairForChat.set(chatId, res.pair);
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
          await handle(m.chat.id, m.text);
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
      for (const [q, a] of seeds) {
        await savePairToDb(q, a);
      }
      await rebuildFromDb();
    }

    console.log('Bot started');
    poll();
  } catch (e) {
    console.error('start error:', e.message);
    process.exit(1);
  }
}

start();
