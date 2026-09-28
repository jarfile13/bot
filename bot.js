const https = require('https');

const BOT_TOKEN = process.env.BOT_TOKEN;

const knowledge = {};
let lastAnswerChat = {};
let offset = 0;

function lc(s) {
  return String(s).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function words(s) {
  const n = lc(s);
  return n ? n.split(' ').filter((w) => w.length > 1) : [];
}

function longestCommonSubstring(a, b) {
  a = lc(a);
  b = lc(b);
  if (!a || !b) return 0;
  let max = 0;
  const prev = new Array(b.length + 1).fill(0);
  const cur = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        cur[j] = prev[j - 1] + 1;
        if (cur[j] > max) max = cur[j];
      } else {
        cur[j] = 0;
      }
    }
    for (let j = 0; j <= b.length; j++) {
      prev[j] = cur[j];
      cur[j] = 0;
    }
  }
  return max;
}

function keywordScore(input, question) {
  const iw = words(input);
  const qw = words(question);
  if (!iw.length || !qw.length) return 0;
  let score = 0;
  for (const q of qw) {
    for (const i of iw) {
      if (i === q) {
        score += 2 + q.length * 0.1;
        break;
      }
      if (q.length > 3 && i.length > 3 && (i.startsWith(q) || q.startsWith(i))) {
        score += 1 + Math.min(q.length, i.length) * 0.05;
        break;
      }
    }
  }
  return score;
}

function findAnswer(input) {
  const keys = Object.keys(knowledge);
  if (!keys.length) return null;

  let bestKey = null;
  let bestScore = 0;
  for (const k of keys) {
    const s = keywordScore(input, k);
    if (s > bestScore) {
      bestScore = s;
      bestKey = k;
    }
  }
  if (bestKey && bestScore >= 2) {
    return knowledge[bestKey];
  }

  const inputLc = lc(input);
  if (inputLc.length < 4) return null;
  let bestSub = 0;
  let bestBySub = null;
  for (const k of keys) {
    const l = longestCommonSubstring(inputLc, lc(k));
    if (l > bestSub) {
      bestSub = l;
      bestBySub = k;
    }
  }
  if (bestBySub && bestSub >= 4 && bestSub / inputLc.length >= 0.5) {
    return knowledge[bestBySub];
  }
  return null;
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
  '/stats - сколько знаю',
  '',
  'Просто пиши вопрос - отвечу, если знаю.',
].join('\n');

async function handle(chatId, text) {
  const t = text.trim();

  if (t === '/start' || t === '/help') {
    await send(chatId, HELP);
    return;
  }

  if (t === '/stats') {
    await send(chatId, `Знаю пар: ${Object.keys(knowledge).length}`);
    return;
  }

  if (t === '/list') {
    const keys = Object.keys(knowledge);
    if (!keys.length) {
      await send(chatId, 'Пусто.');
      return;
    }
    const lines = keys.map((k, i) => `${i + 1}. ${k} = ${knowledge[k]}`);
    const chunkSize = 3500;
    let buf = '';
    for (const line of lines) {
      if ((buf + line + '\n').length > chunkSize) {
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
    const q = lc(rest.slice(0, sep));
    const a = rest.slice(sep + 1).trim();
    if (!q || !a) {
      await send(chatId, 'Пусто.');
      return;
    }
    knowledge[q] = a;
    await send(chatId, `Запомнил: ${q} = ${a}`);
    return;
  }

  if (t.startsWith('/delete')) {
    const rest = t.slice('/delete'.length).trim();
    if (!rest) {
      await send(chatId, 'Формат: /delete вопрос');
      return;
    }
    const k = lc(rest);
    if (knowledge[k]) {
      delete knowledge[k];
      await send(chatId, `Удалил: ${k}`);
      return;
    }
    const keys = Object.keys(knowledge);
    let found = null;
    let bestSub = 0;
    for (const key of keys) {
      const l = longestCommonSubstring(k, key);
      if (l > bestSub) {
        bestSub = l;
        found = key;
      }
    }
    if (found && bestSub >= 4) {
      delete knowledge[found];
      await send(chatId, `Удалил: ${found}`);
    } else {
      await send(chatId, `Не нашёл: ${k}`);
    }
    return;
  }

  const answer = findAnswer(t);
  if (answer) {
    lastAnswerChat[chatId] = answer;
    await send(chatId, answer);
  } else {
    await send(chatId, 'Не знаю. Научи: /teach вопрос = ответ');
  }
}

async function poll() {
  while (true) {
    try {
      const res = await api('getUpdates', {
        offset,
        timeout: 30,
        allowed_updates: ['message'],
      });
      if (res && res.ok && Array.isArray(res.result)) {
        for (const u of res.result) {
          offset = u.update_id + 1;
          const msg = u.message;
          if (!msg || !msg.text) continue;
          await handle(msg.chat.id, msg.text);
        }
      }
    } catch (e) {
      console.error('poll error:', e.message);
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
}

if (!BOT_TOKEN) {
  console.error('BOT_TOKEN not set');
  process.exit(1);
}

console.log('Bot started');
poll();
