const https = require('https');

const BOT_TOKEN = process.env.BOT_TOKEN;

const nodes = new Map();
const edges = new Map();
const pairs = [];

let offset = 0;
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

function teach(question, answer) {
  const qw = words(question);
  const aw = words(answer);
  if (!qw.length || !aw.length) return false;

  for (const q of qw) {
    charge(q, 1.0);
    for (const a of aw) link(q, a, 0.3);
  }
  for (const a of aw) {
    charge(a, 0.5);
  }
  for (let i = 0; i < qw.length; i++) {
    for (let j = i + 1; j < qw.length; j++) link(qw[i], qw[j], 0.5);
  }
  for (let i = 0; i < aw.length; i++) {
    for (let j = i + 1; j < aw.length; j++) link(aw[i], aw[j], 0.5);
  }

  const existing = pairs.findIndex((p) => lc(p.question) === lc(question));
  if (existing >= 0) pairs[existing] = { question, answer };
  else pairs.push({ question, answer });

  return true;
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
  const DECAY = 0.5;
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
        const newEnergy = energy * strength * DECAY;
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

  let bestPair = null;
  let bestScore = 0;

  for (const pair of pairs) {
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

function reinforce(pair, positive) {
  const qw = words(pair.question);
  const aw = words(pair.answer);
  const factor = positive ? 1.15 : 0.85;
  for (const q of qw) {
    ensureNode(q);
    nodes.set(q, Math.min(1, nodes.get(q) * factor));
    for (const a of aw) {
      const ea = edges.get(q);
      if (ea && ea.has(a)) {
        ea.set(a, Math.min(1, ea.get(a) * factor));
      }
    }
  }
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
  '/stats - статистика графа',
  '+ - ответ понравился (усилить)',
  '- - ответ не понравился (ослабить)',
  '',
  'Бот сам сближает похожие слова и переносит смысл.',
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
      `Пар: ${pairs.length}\nСлов в графе: ${nodes.size}\nСвязей: ${[...edges.values()].reduce((s, m) => s + m.size, 0) / 2}`
    );
    return;
  }

  if (t === '/list') {
    if (!pairs.length) {
      await send(chatId, 'Пусто.');
      return;
    }
    const lines = pairs.map((p, i) => `${i + 1}. ${p.question} = ${p.answer}`);
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
    teach(q, a);
    await send(chatId, `Запомнил: ${q} = ${a}`);
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
      const removed = pairs.splice(idx, 1)[0];
      await send(chatId, `Удалил: ${removed.question}`);
      return;
    }
    const res = answerFor(rest);
    if (res && res.score >= 0.6) {
      const removedIdx = pairs.indexOf(res.pair);
      pairs.splice(removedIdx, 1);
      await send(chatId, `Удалил: ${res.pair.question}`);
      return;
    }
    await send(chatId, `Не нашёл: ${key}`);
    return;
  }

  if (t === '+' || t === '-') {
    const last = lastPairForChat.get(chatId);
    if (!last) {
      await send(chatId, 'Нет последнего ответа.');
      return;
    }
    reinforce(last, t === '+');
    await send(chatId, t === '+' ? 'Усилил.' : 'Ослабил.');
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

if (!BOT_TOKEN) {
  console.error('BOT_TOKEN not set');
  process.exit(1);
}

teach('привет', 'Здорово друг');
teach('здравствуй', 'Здорово друг');
teach('как дела', 'Отлично а у тебя');
teach('пока', 'До встречи');

console.log('Bot started');
poll();
