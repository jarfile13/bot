javascript

const Parse = require('parse/node');

Parse.initialize(
  process.env.B4A_APP_ID,
  process.env.B4A_JS_KEY,
  process.env.B4A_MASTER_KEY
);
Parse.serverURL = 'https://parseapi.back4app.com';

async function getAll() {
  const query = new Parse.Query('Knowledge');
  query.limit(10000);
  const results = await query.find({ useMasterKey: true });
  const map = {};
  for (const obj of results) {
    map[obj.get('question')] = obj.get('answer');
  }
  return map;
}

async function upsert(question, answer) {
  const query = new Parse.Query('Knowledge');
  query.equalTo('question', question);
  let obj = await query.first({ useMasterKey: true });
  if (!obj) {
    obj = new Parse.Object('Knowledge');
    obj.set('question', question);
  }
  obj.set('answer', answer);
  await obj.save(null, { useMasterKey: true });
}

async function remove(question) {
  const query = new Parse.Query('Knowledge');
  query.equalTo('question', question);
  const obj = await query.first({ useMasterKey: true });
  if (obj) await obj.destroy({ useMasterKey: true });
}

module.exports = { getAll, upsert, remove };

bot.js
javascript

const https = require('https');
const brainDb = require('./brain');

const BOT_TOKEN = process.env.BOT_TOKEN;
const SIMILARITY_THRESHOLD = 0.5;

let brain = {};

function normalize(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokens(text) {
  const n = normalize(text);
  if (!n) return new Set();
  return new Set(n.split(' '));
}

function similarity(a, b) {
  const ta = tokens(a);
  const tb = tokens(b);
  if (ta.size === 0 || tb.size === 0) return 0;
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter++;
  const union = ta.size + tb.size - inter;
  return union === 0 ? 0 : inter / union;
}

function findAnswer(text) {
  const key = normalize(text);
  if (!key) return null;
  if (brain[key]) return brain[key];
  let best = null;
  let bestScore = 0;
  for (const k of Object.keys(brain)) {
    const s = similarity(k, key);
    if (s > bestScore) {
      bestScore = s;
      best = k;
    }
  }
  if (best && bestScore >= SIMILARITY_THRESHOLD) return brain[best];
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

async function sendMessage(chatId, text) {
  try {
    await api('sendMessage', {
      chat_id: chatId,
      text,
      disable_web_page_preview: true,
    });
  } catch (e) {
    console.error('sendMessage error:', e.message);
  }
}

const HELP = [
  'Доступные команды:',
  '/help - показать справку',
  '/teach вопрос | ответ - обучить бота',
  '/delete вопрос - удалить знание',
  '/list - показать все знания',
  '/stats - статистика',
  '',
  'Просто напиши вопрос, и я отвечу, если знаю.',
].join('\n');

async function handleCommand(chatId, text) {
  const trimmed = text.trim();

  if (trimmed === '/start' || trimmed === '/help') {
    await sendMessage(chatId, HELP);
    return true;
  }

  if (trimmed === '/list') {
    const keys = Object.keys(brain);
    if (keys.length === 0) {
      await sendMessage(chatId, 'Пока ничего не знаю.');
      return true;
    }
    const lines = keys.map((k, i) => `${i + 1}. ${k} -> ${brain[k]}`);
    await sendMessage(chatId, lines.join('\n'));
    return true;
  }

  if (trimmed === '/stats') {
    const count = Object.keys(brain).length;
    await sendMessage(chatId, `Записей в базе знаний: ${count}`);
    return true;
  }

  if (trimmed.startsWith('/teach')) {
    const rest = trimmed.slice('/teach'.length).trim();
    const sep = rest.indexOf('|');
    if (sep === -1) {
      await sendMessage(chatId, 'Формат: /teach вопрос | ответ');
      return true;
    }
    const q = rest.slice(0, sep).trim();
    const a = rest.slice(sep + 1).trim();
    if (!q || !a) {
      await sendMessage(chatId, 'Вопрос и ответ не должны быть пустыми.');
      return true;
    }
    const key = normalize(q);
    try {
      await brainDb.upsert(key, a);
      brain[key] = a;
      await sendMessage(chatId, `Запомнил: ${key} -> ${a}`);
    } catch (e) {
      console.error('upsert error:', e.message);
      await sendMessage(chatId, 'Ошибка сохранения. Попробуй позже.');
    }
    return true;
  }

  if (trimmed.startsWith('/delete')) {
    const rest = trimmed.slice('/delete'.length).trim();
    if (!rest) {
      await sendMessage(chatId, 'Формат: /delete вопрос');
      return true;
    }
    const key = normalize(rest);
    if (brain[key]) {
      try {
        await brainDb.remove(key);
        delete brain[key];
        await sendMessage(chatId, `Удалил: ${key}`);
      } catch (e) {
        console.error('remove error:', e.message);
        await sendMessage(chatId, 'Ошибка удаления. Попробуй позже.');
      }
    } else {
      await sendMessage(chatId, `Не найдено: ${key}`);
    }
    return true;
  }

  return false;
}

let offset = 0;
let polling = false;

async function poll() {
  if (polling) return;
  polling = true;
  while (true) {
    try {
      const res = await api('getUpdates', {
        offset,
        timeout: 30,
        allowed_updates: ['message'],
      });
      if (res && res.ok && Array.isArray(res.result)) {
        for (const update of res.result) {
          offset = update.update_id + 1;
          const msg = update.message;
          if (!msg || !msg.text) continue;
          const chatId = msg.chat.id;
          const text = msg.text;

          const handled = await handleCommand(chatId, text);
          if (handled) continue;

          const answer = findAnswer(text);
          if (answer) {
            await sendMessage(chatId, answer);
          } else {
            await sendMessage(
              chatId,
              'Не знаю ответа. Обучи меня: /teach вопрос | ответ'
            );
          }
        }
      }
    } catch (e) {
      console.error('poll error:', e.message);
      await new Promise((r) => setTimeout(r, 3000));
    }
  }
}

async function start() {
  if (!BOT_TOKEN) {
    console.error('BOT_TOKEN is not set');
    process.exit(1);
  }
  if (!process.env.B4A_APP_ID || !process.env.B4A_JS_KEY || !process.env.B4A_MASTER_KEY) {
    console.error('Back4App keys are not set');
    process.exit(1);
  }
  try {
    brain = await brainDb.getAll();
    console.log('Brain loaded from Back4App, records:', Object.keys(brain).length);
  } catch (e) {
    console.error('Failed to load brain:', e.message);
    brain = {};
  }
  console.log('Bot started');
  poll();
}

start();

Dockerfile
dockerfile

FROM node:20-slim
WORKDIR /app
COPY package.json .
RUN npm install --omit=dev
COPY bot.js brain.js .
EXPOSE 3000
CMD ["node", "bot.js"]

Переменные окружения в Back4App Containers

В настройках приложения Back4App Containers добавь:

    BOT_TOKEN — токен бота от @BotFather

    B4A_APP_ID — Application ID

    B4A_JS_KEY — JavaScript Key

    B4A_MASTER_KEY — Master Key

Проверка

    Залей файлы в репозиторий, Back4App соберёт и запустит контейнер.

    В логах контейнера должно быть Brain loaded from Back4App, records: 0 и Bot started.

    В Telegram: /teach Привет | Здравствуй!

    Напиши Привет — должен ответить.

    В Back4App Dashboard → Database → Knowledge появится запись.

    Перезапусти контейнер — знания останутся.

