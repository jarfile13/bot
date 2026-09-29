const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BOT_TOKEN = process.env.BOT_TOKEN;
if (!BOT_TOKEN) {
  console.error('[КРИТИЧЕСКАЯ ОШИБКА] Переменная BOT_TOKEN не задана!');
  process.exit(1);
}

const PORT = process.env.PORT || 3000;
const DATA_DIR = process.env.DATA_DIR || './data';

try {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
} catch (e) {
  console.error('[ОШИБКА] Не удалось создать папку данных:', e.message);
}

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT);

const userMemories = new Map();
const lastMessages = new Map();
const pendingReset = new Map();
const pendingForget = new Map();
const lastList = new Map();

const PAGE_SIZE = 10;

function getUserMemoryFile(userId) {
  return path.join(DATA_DIR, `mem_${userId}.json`);
}

function loadUserMemory(userId) {
  if (userMemories.has(userId)) return userMemories.get(userId);
  const file = getUserMemoryFile(userId);
  let memory = [];
  if (fs.existsSync(file)) {
    try { memory = JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) { memory = []; }
  }
  userMemories.set(userId, memory);
  return memory;
}

function saveUserMemory(userId, memory) {
  userMemories.set(userId, memory);
  try {
    fs.writeFileSync(getUserMemoryFile(userId), JSON.stringify(memory, null, 2), 'utf8');
  } catch (e) {
    console.error('[ОШИБКА ЗАПИСИ ФАЙЛА]', e.message);
  }
}

function cleanText(str) {
  return String(str)
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(str) {
  const words = cleanText(str).split(' ').filter(word => word.length >= 1);
  const ngrams = [];
  const text = cleanText(str);
  for (let i = 0; i < text.length - 2; i++) {
    ngrams.push(text.substring(i, i + 3));
  }
  return { words, ngrams };
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function sendMessage(chatId, text, extra) {
  const payload = { chat_id: chatId, text: text };
  if (extra) Object.assign(payload, extra);
  const data = JSON.stringify(payload);
  const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }
  });
  req.on('error', (e) => console.error('[ОШИБКА ОТПРАВКИ]', e.message));
  req.write(data);
  req.end();
}

function sendDocument(chatId, filePath, caption) {
  try {
    const boundary = '----WebKitFormBoundary' + Math.random().toString(36).substring(2);
    const filename = path.basename(filePath);
    const fileData = fs.readFileSync(filePath);

    let header = `--${boundary}\r\n`;
    header += `Content-Disposition: form-data; name="chat_id"\r\n\r\n${chatId}\r\n`;
    header += `--${boundary}\r\n`;
    header += `Content-Disposition: form-data; name="caption"\r\n\r\n${caption}\r\n`;
    header += `--${boundary}\r\n`;
    header += `Content-Disposition: form-data; name="document"; filename="${filename}"\r\n`;
    header += `Content-Type: application/json\r\n\r\n`;

    const footer = `\r\n--${boundary}--\r\n`;

    const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/sendDocument`, {
      method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}` }
    });

    req.on('error', (e) => console.error('[ОШИБКА ОТПРАВКИ ДОКУМЕНТА]', e.message));
    req.write(header);
    req.write(fileData);
    req.write(footer);
    req.end();
  } catch (e) {
    console.error('[ОШИБКА ЭКСПОРТА]', e.message);
  }
}

function downloadFile(fileId, callback) {
  https.get(`https://api.telegram.org/bot${BOT_TOKEN}/getFile?file_id=${fileId}`, (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      try {
        const json = JSON.parse(data);
        if (json.ok && json.result.file_path) {
          https.get(`https://api.telegram.org/file/bot${BOT_TOKEN}/${json.result.file_path}`, (fileRes) => {
            const chunks = [];
            fileRes.on('data', chunk => chunks.push(chunk));
            fileRes.on('end', () => callback(Buffer.concat(chunks).toString('utf8')));
          });
        } else {
          console.error('[ОШИБКА ТГ]', json.description);
        }
      } catch (e) {
        console.error('[ОШИБКА СКАЧИВАНИЯ]', e.message);
      }
    });
  }).on('error', (e) => console.error('[ОШИБКА ЗАПРОСА]', e.message));
}

function findBestAnswer(userTokens, memory) {
  if (userTokens.words.length === 0) return null;
  let bestMatch = null;
  let maxScore = 0.15;

  for (const entry of memory) {
    const wordInter = userTokens.words.filter(w => entry.tokens.words.includes(w));
    const wordUni = new Set([...userTokens.words, ...entry.tokens.words]);
    const wordScore = wordInter.length / wordUni.size;

    const ngramInter = userTokens.ngrams.filter(n => entry.tokens.ngrams.includes(n));
    const ngramUni = new Set([...userTokens.ngrams, ...entry.tokens.ngrams]);
    const ngramScore = ngramUni.size > 0 ? ngramInter.length / ngramUni.size : 0;

    const finalScore = (wordScore * 0.4) + (ngramScore * 0.6);

    if (finalScore > maxScore) {
      maxScore = finalScore;
      bestMatch = entry;
    }
  }

  if (bestMatch && bestMatch.answers.length > 0) {
    return bestMatch.answers[Math.floor(Math.random() * bestMatch.answers.length)];
  }
  return null;
}

function learn(userId, question, answer, memory) {
  const qTokens = tokenize(question);
  if (qTokens.words.length === 0 || !answer.trim()) return;

  const existing = memory.find(entry => {
    const inter = qTokens.words.filter(w => entry.tokens.words.includes(w));
    const uni = new Set([...qTokens.words, ...entry.tokens.words]);
    return (inter.length / uni.size) > 0.85;
  });

  if (existing) {
    if (!existing.answers.includes(answer.trim())) {
      existing.answers.push(answer.trim());
    }
  } else {
    memory.push({
      tokens: qTokens,
      text: question.trim(),
      answers: [answer.trim()],
      created: Date.now(),
      hits: 0
    });
  }

  saveUserMemory(userId, memory);
}

function getMemoryStats(memory) {
  let answersCount = 0;
  let totalLength = 0;
  let oldest = null;
  let newest = null;
  let withCreated = 0;

  for (const e of memory) {
    answersCount += e.answers.length;
    totalLength += e.text.length;
    if (e.created) {
      withCreated++;
      if (oldest === null || e.created < oldest) oldest = e.created;
      if (newest === null || e.created > newest) newest = e.created;
    }
  }

  return {
    phrases: memory.length,
    answers: answersCount,
    avgAnswers: memory.length ? (answersCount / memory.length).toFixed(2) : '0',
    avgLength: memory.length ? Math.round(totalLength / memory.length) : 0,
    oldest,
    newest,
    withCreated
  };
}

function formatDate(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  const pad = n => String(n).padStart(2, '0');
  return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function showHelp(chatId) {
  const text =
    '<b>Персональный текстовый ИИ</b>\n\n' +
    'Я учусь в процессе общения. Каждое твоё сообщение становится ответом на моё предыдущее.\n\n' +
    '<b>Управление знаниями</b>\n' +
    '/list [стр] — список всех фраз с номерами\n' +
    '/show &lt;номер&gt; — подробно о фразе\n' +
    '/edit &lt;номер&gt; — заменить ответ у фразы\n' +
    '/del &lt;номер&gt; — удалить фразу по номеру\n' +
    '/forget &lt;текст&gt; — удалить по тексту\n' +
    '/find &lt;текст&gt; — найти фразы по подстроке\n\n' +
    '<b>Данные</b>\n' +
    '/export — скачать базу знаний (JSON)\n' +
    '/import — как импортировать (отправь JSON-файл)\n' +
    '/stats — подробная статистика\n' +
    '/reset — полностью стереть память (с подтверждением)\n\n' +
    '<b>Прочее</b>\n' +
    '/help — эта справка\n' +
    '/cancel — отменить текущее действие';

  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showList(chatId, userId, memory, page) {
  if (memory.length === 0) {
    sendMessage(chatId, 'База знаний пуста.');
    return;
  }
  const totalPages = Math.ceil(memory.length / PAGE_SIZE);
  const p = Math.max(1, Math.min(page, totalPages));
  const start = (p - 1) * PAGE_SIZE;
  const slice = memory.slice(start, start + PAGE_SIZE);

  let text = `<b>База знаний</b> (стр. ${p}/${totalPages}, всего: ${memory.length})\n\n`;
  slice.forEach((e, i) => {
    const num = start + i + 1;
    const preview = escapeHtml(e.text.length > 60 ? e.text.substring(0, 60) + '…' : e.text);
    const cnt = e.answers.length;
    text += `<b>${num}.</b> ${preview}\n     └ ответов: ${cnt}\n`;
  });

  text += `\n/show &lt;номер&gt; · /edit &lt;номер&gt; · /del &lt;номер&gt;`;
  if (totalPages > 1) {
    text += `\n\nСледующая: /list ${p + 1}`;
  }

  lastList.set(userId, p);
  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showEntry(chatId, memory, num) {
  const idx = num - 1;
  if (idx < 0 || idx >= memory.length) {
    sendMessage(chatId, `Фразы с номером ${num} нет. Всего: ${memory.length}`);
    return;
  }
  const e = memory[idx];
  let text = `<b>Фраза #${num}</b>\n\n`;
  text += `<b>Вопрос:</b>\n${escapeHtml(e.text)}\n\n`;
  text += `<b>Ответы (${e.answers.length}):</b>\n`;
  e.answers.forEach((a, i) => {
    text += `${i + 1}. ${escapeHtml(a.length > 200 ? a.substring(0, 200) + '…' : a)}\n`;
  });
  if (e.created) text += `\nСоздано: ${formatDate(e.created)}`;

  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function findEntries(chatId, memory, query) {
  const q = cleanText(query);
  if (!q) {
    sendMessage(chatId, 'Укажи текст для поиска: /find &lt;текст&gt;', { parse_mode: 'HTML' });
    return;
  }
  const results = [];
  memory.forEach((e, i) => {
    if (cleanText(e.text).includes(q) || e.answers.some(a => cleanText(a).includes(q))) {
      results.push({ num: i + 1, entry: e });
    }
  });

  if (results.length === 0) {
    sendMessage(chatId, `Ничего не найдено по запросу: "${escapeHtml(query)}"`, { parse_mode: 'HTML' });
    return;
  }

  let text = `<b>Найдено: ${results.length}</b>\n\n`;
  results.slice(0, 15).forEach(r => {
    const preview = escapeHtml(r.entry.text.length > 70 ? r.entry.text.substring(0, 70) + '…' : r.entry.text);
    text += `<b>#${r.num}</b> ${preview}\n`;
  });
  if (results.length > 15) text += `\n…и ещё ${results.length - 15}`;
  text += `\n\nИспользуй /show &lt;номер&gt; для подробностей`;

  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showStats(chatId, userId, memory) {
  const s = getMemoryStats(memory);
  const file = getUserMemoryFile(userId);
  let fileSize = 0;
  if (fs.existsSync(file)) {
    try { fileSize = fs.statSync(file).size; } catch (e) {}
  }

  let text = '<b>Статистика базы знаний</b>\n\n';
  text += `Уникальных фраз: <b>${s.phrases}</b>\n`;
  text += `Всего ответов: <b>${s.answers}</b>\n`;
  text += `В среднем ответов на фразу: <b>${s.avgAnswers}</b>\n`;
  text += `Средняя длина фразы: <b>${s.avgLength}</b> симв.\n`;
  text += `Размер файла: <b>${(fileSize / 1024).toFixed(1)}</b> КБ\n`;
  if (s.oldest) text += `\nПервая запись: ${formatDate(s.oldest)}\n`;
  if (s.newest) text += `Последняя запись: ${formatDate(s.newest)}\n`;
  if (s.withCreated < s.phrases) {
    text += `\n${s.phrases - s.withCreated} старых фраз без даты создания`;
  }

  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function handleMessage(chatId, userId, text, document) {
  let memory = loadUserMemory(userId);

  if (pendingReset.get(userId)) {
    const answer = (text || '').trim().toLowerCase();
    if (answer === 'да' || answer === 'yes' || answer === 'y') {
      const file = getUserMemoryFile(userId);
      if (fs.existsSync(file)) { try { fs.unlinkSync(file); } catch (e) {} }
      userMemories.delete(userId);
      lastMessages.delete(userId);
      pendingReset.delete(userId);
      sendMessage(chatId, 'Память полностью очищена.');
    } else {
      pendingReset.delete(userId);
      sendMessage(chatId, 'Отменено. Память не тронута.');
    }
    return;
  }

  if (pendingForget.get(userId)) {
    const state = pendingForget.get(userId);
    if (text === '/cancel') {
      pendingForget.delete(userId);
      sendMessage(chatId, 'Отменено.');
      return;
    }
    const idx = state.num - 1;
    pendingForget.delete(userId);
    if (idx < 0 || idx >= memory.length) {
      sendMessage(chatId, `Фразы с номером ${state.num} больше нет.`);
      return;
    }
    memory.splice(idx, 1);
    saveUserMemory(userId, memory);
    sendMessage(chatId, `Фраза #${state.num} удалена.`);
    return;
  }

  if (document && document.file_name && document.file_name.endsWith('.json')) {
    downloadFile(document.file_id, (content) => {
      try {
        const importedData = JSON.parse(content);
        if (Array.isArray(importedData) && importedData.every(e => e.tokens && Array.isArray(e.answers))) {
          saveUserMemory(userId, importedData);
          sendMessage(chatId, `Импортировано ${importedData.length} фраз.`);
        } else {
          sendMessage(chatId, 'Некорректный формат файла.');
        }
      } catch (e) {
        sendMessage(chatId, 'Ошибка при чтении файла.');
      }
    });
    return;
  }

  if (text && text.startsWith('/')) {
    const parts = text.trim().split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const arg = parts.slice(1).join(' ');

    if (cmd === '/start' || cmd === '/help') { showHelp(chatId); return; }

    if (cmd === '/cancel') {
      pendingReset.delete(userId);
      pendingForget.delete(userId);
      sendMessage(chatId, 'Действие отменено.');
      return;
    }

    if (cmd === '/reset') {
      pendingReset.set(userId, true);
      sendMessage(chatId, '<b>Ты уверен?</b>\nВся база знаний будет безвозвратно удалена.\n\nНапиши <b>да</b> для подтверждения или что-то другое для отмены.', { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/export') {
      if (memory.length === 0) { sendMessage(chatId, 'База знаний пуста.'); return; }
      sendDocument(chatId, getUserMemoryFile(userId), `Экспорт базы знаний (${memory.length} фраз)`);
      return;
    }

    if (cmd === '/import') {
      sendMessage(chatId, 'Отправь JSON-файл, ранее полученный через /export.\n\nТекущая база будет заменена.');
      return;
    }

    if (cmd === '/stats') { showStats(chatId, userId, memory); return; }

    if (cmd === '/list') {
      const page = parseInt(arg, 10) || 1;
      showList(chatId, userId, memory, page);
      return;
    }

    if (cmd === '/show') {
      const num = parseInt(arg, 10);
      if (!num) { sendMessage(chatId, 'Использование: /show &lt;номер&gt;', { parse_mode: 'HTML' }); return; }
      showEntry(chatId, memory, num);
      return;
    }

    if (cmd === '/find') {
      findEntries(chatId, memory, arg);
      return;
    }

    if (cmd === '/del') {
      const num = parseInt(arg, 10);
      if (!num || num < 1 || num > memory.length) {
        sendMessage(chatId, `Укажи корректный номер (1–${memory.length}).`);
        return;
      }
      pendingForget.set(userId, { num });
      const preview = escapeHtml(memory[num - 1].text.substring(0, 80));
      sendMessage(chatId, `Удалить фразу #${num}?\n\n"${preview}"\n\nНапиши /cancel чтобы отменить.`, { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/edit') {
      const num = parseInt(arg, 10);
      if (!num || num < 1 || num > memory.length) {
        sendMessage(chatId, `Укажи корректный номер (1–${memory.length}).`);
        return;
      }
      sendMessage(chatId, `Чтобы добавить ответ к фразе #${num}, просто напиши его следующим сообщением (без слэша). Он добавится как новый вариант.`);
      lastMessages.set(userId, memory[num - 1].text);
      return;
    }

    if (cmd === '/forget') {
      const target = arg.trim();
      if (!target) { sendMessage(chatId, 'Использование: /forget &lt;текст&gt;', { parse_mode: 'HTML' }); return; }
      const cleanedTarget = cleanText(target);
      const before = memory.length;
      memory = memory.filter(entry => cleanText(entry.text) !== cleanedTarget);
      if (memory.length < before) {
        saveUserMemory(userId, memory);
        sendMessage(chatId, `Удалено фраз: ${before - memory.length}.`);
      } else {
        sendMessage(chatId, `Фраза "${escapeHtml(target)}" не найдена. Попробуй /find`, { parse_mode: 'HTML' });
      }
      return;
    }

    sendMessage(chatId, 'Неизвестная команда. Смотри /help');
    return;
  }

  if (!text) return;

  const cleanTextStr = text.trim();
  const tokens = tokenize(cleanTextStr);

  const prevBotMessage = lastMessages.get(userId);
  if (prevBotMessage) {
    learn(userId, prevBotMessage, cleanTextStr, memory);
  }

  const aiResponse = findBestAnswer(tokens, memory);

  if (aiResponse) {
    sendMessage(chatId, aiResponse);
    lastMessages.set(userId, aiResponse);
  } else {
    sendMessage(chatId, cleanTextStr);
    lastMessages.set(userId, cleanTextStr);
  }
}

let offset = 0;
let polling = false;
function getUpdates() {
  if (polling) return;
  polling = true;
  const url = `https://api.telegram.org/bot${BOT_TOKEN}/getUpdates?offset=${offset}&timeout=25`;
  https.get(url, (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      polling = false;
      try {
        const json = JSON.parse(data);
        if (json.ok) {
          if (json.result.length > 0) {
            for (const update of json.result) {
              offset = update.update_id + 1;
              if (update.message) {
                handleMessage(update.message.chat.id, update.message.from.id, update.message.text, update.message.document);
              }
            }
          }
        } else {
          console.error('[TELEGRAM API]', json.description);
        }
      } catch (e) {
        console.error('[ОБРАБОТКА ПАКЕТА]', e.message);
      }
      setTimeout(getUpdates, 300);
    });
  }).on('error', (e) => {
    polling = false;
    setTimeout(getUpdates, 3000);
  });
}

getUpdates();
