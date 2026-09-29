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
  if (!Array.isArray(memory)) memory = [];
  memory = memory.filter(e =>
    e &&
    e.tokens &&
    Array.isArray(e.tokens.words) &&
    Array.isArray(e.tokens.ngrams) &&
    Array.isArray(e.answers) &&
    typeof e.text === 'string'
  );
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
  const text = cleanText(str);
  const words = text.split(' ').filter(w => w.length >= 1);
  const ngrams = [];
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

function similarity(userTokens, entryTokens) {
  const wInter = userTokens.words.filter(w => entryTokens.words.includes(w));
  const wUni = new Set([...userTokens.words, ...entryTokens.words]);
  const wScore = wUni.size > 0 ? wInter.length / wUni.size : 0;

  const nInter = userTokens.ngrams.filter(n => entryTokens.ngrams.includes(n));
  const nUni = new Set([...userTokens.ngrams, ...entryTokens.ngrams]);
  const nScore = nUni.size > 0 ? nInter.length / nUni.size : 0;

  return (wScore * 0.4) + (nScore * 0.6);
}

function findBestAnswer(userTokens, memory) {
  if (userTokens.words.length === 0 || memory.length === 0) return null;
  let bestMatch = null;
  let maxScore = 0.35;

  for (const entry of memory) {
    const s = similarity(userTokens, entry.tokens);
    if (s > maxScore) {
      maxScore = s;
      bestMatch = entry;
    }
  }

  if (bestMatch && bestMatch.answers.length > 0) {
    bestMatch.hits = (bestMatch.hits || 0) + 1;
    return bestMatch.answers[Math.floor(Math.random() * bestMatch.answers.length)];
  }
  return null;
}

function learn(userId, question, answer, memory) {
  const q = String(question || '').trim();
  const a = String(answer || '').trim();
  if (!q || !a) return false;

  const qTokens = tokenize(q);
  if (qTokens.words.length === 0) return false;

  let existing = null;
  let bestScore = 0.85;
  for (const entry of memory) {
    const s = similarity(qTokens, entry.tokens);
    if (s > bestScore) {
      bestScore = s;
      existing = entry;
    }
  }

  if (existing) {
    if (!existing.answers.includes(a)) existing.answers.push(a);
  } else {
    memory.push({
      tokens: qTokens,
      text: q,
      answers: [a],
      created: Date.now(),
      hits: 0
    });
  }

  saveUserMemory(userId, memory);
  return true;
}

function getMemoryStats(memory) {
  let answersCount = 0;
  let totalLength = 0;
  let oldest = null;
  let newest = null;
  let withCreated = 0;
  let totalHits = 0;

  for (const e of memory) {
    answersCount += e.answers.length;
    totalLength += e.text.length;
    totalHits += e.hits || 0;
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
    oldest, newest, withCreated, totalHits
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
    '<b>Ассоциативный текстовый ИИ</b>\n\n' +
    'Логика работы:\n' +
    '1. Ты пишешь фразу.\n' +
    '2. Если я её знаю — отвечаю из памяти.\n' +
    '3. Если не знаю — повторяю её эхом и ЖДУ твой ответ.\n' +
    '4. Твой следующий ответ я запоминаю как ответ на эту фразу.\n\n' +
    '<b>Команды</b>\n' +
    '/help — справка и сброс зависшего контекста\n' +
    '/teach вопрос = ответ — обучить сразу, без эха\n' +
    '/list [стр] — список фраз\n' +
    '/show &lt;номер&gt; — подробно о фразе\n' +
    '/del &lt;номер&gt; — удалить фразу\n' +
    '/forget &lt;текст&gt; — удалить по точному тексту\n' +
    '/find &lt;текст&gt; — поиск по подстроке\n' +
    '/export — скачать базу\n' +
    '/import — как импортировать (отправь JSON)\n' +
    '/stats — статистика\n' +
    '/reset — стереть всё (с подтверждением)\n' +
    '/cancel — отменить действие / сбросить контекст';

  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showList(chatId, memory, page) {
  if (memory.length === 0) { sendMessage(chatId, 'База пуста.'); return; }
  const PAGE_SIZE = 10;
  const totalPages = Math.ceil(memory.length / PAGE_SIZE);
  const p = Math.max(1, Math.min(page, totalPages));
  const start = (p - 1) * PAGE_SIZE;
  const slice = memory.slice(start, start + PAGE_SIZE);

  let text = `<b>База знаний</b> (стр. ${p}/${totalPages}, всего: ${memory.length})\n\n`;
  slice.forEach((e, i) => {
    const num = start + i + 1;
    const preview = escapeHtml(e.text.length > 60 ? e.text.substring(0, 60) + '…' : e.text);
    text += `<b>${num}.</b> ${preview}\n     ответов: ${e.answers.length}\n`;
  });
  if (totalPages > 1) text += `\nСледующая: /list ${p + 1}`;
  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showEntry(chatId, memory, num) {
  const idx = num - 1;
  if (idx < 0 || idx >= memory.length) {
    sendMessage(chatId, `Фразы #${num} нет. Всего: ${memory.length}`);
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
  if (e.hits) text += `\nИспользований: ${e.hits}`;
  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function findEntries(chatId, memory, query) {
  const q = cleanText(query);
  if (!q) { sendMessage(chatId, 'Использование: /find &lt;текст&gt;', { parse_mode: 'HTML' }); return; }
  const results = [];
  memory.forEach((e, i) => {
    if (cleanText(e.text).includes(q) || e.answers.some(a => cleanText(a).includes(q))) {
      results.push({ num: i + 1, entry: e });
    }
  });
  if (results.length === 0) {
    sendMessage(chatId, `Ничего не найдено: "${escapeHtml(query)}"`, { parse_mode: 'HTML' });
    return;
  }
  let text = `<b>Найдено: ${results.length}</b>\n\n`;
  results.slice(0, 15).forEach(r => {
    const preview = escapeHtml(r.entry.text.length > 70 ? r.entry.text.substring(0, 70) + '…' : r.entry.text);
    text += `<b>#${r.num}</b> ${preview}\n`;
  });
  if (results.length > 15) text += `\n…и ещё ${results.length - 15}`;
  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showStats(chatId, userId, memory) {
  const s = getMemoryStats(memory);
  const file = getUserMemoryFile(userId);
  let fileSize = 0;
  if (fs.existsSync(file)) { try { fileSize = fs.statSync(file).size; } catch (e) {} }

  let text = '<b>Статистика</b>\n\n';
  text += `Уникальных фраз: <b>${s.phrases}</b>\n`;
  text += `Всего ответов: <b>${s.answers}</b>\n`;
  text += `Средне ответов на фразу: <b>${s.avgAnswers}</b>\n`;
  text += `Средняя длина вопроса: <b>${s.avgLength}</b> симв.\n`;
  text += `Всего использований памяти: <b>${s.totalHits}</b>\n`;
  text += `Размер файла: <b>${(fileSize / 1024).toFixed(1)}</b> КБ\n`;
  if (s.oldest) text += `\nПервая запись: ${formatDate(s.oldest)}\n`;
  if (s.newest) text += `Последняя запись: ${formatDate(s.newest)}\n`;
  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function parseTeach(text) {
  const m = text.match(/^\/teach\s+(.+?)\s*=\s*(.+)$/s);
  if (m) return { q: m[1].trim(), a: m[2].trim() };
  const m2 = text.match(/^запомни[:\s]+(.+?)\s*=\s*(.+)$/i);
  if (m2) return { q: m2[1].trim(), a: m2[2].trim() };
  return null;
}

const pendingReset = new Map();
const pendingDel = new Map();
const teachBuffer = new Map();

function handleMessage(chatId, userId, text, document) {
  let memory = loadUserMemory(userId);

  if (pendingReset.get(userId)) {
    const a = (text || '').trim().toLowerCase();
    if (a === 'да' || a === 'yes' || a === 'y') {
      const file = getUserMemoryFile(userId);
      if (fs.existsSync(file)) { try { fs.unlinkSync(file); } catch (e) {} }
      userMemories.delete(userId);
      lastMessages.delete(userId);
      pendingReset.delete(userId);
      sendMessage(chatId, 'Память полностью очищена.');
    } else {
      pendingReset.delete(userId);
      sendMessage(chatId, 'Отменено.');
    }
    return;
  }

  if (pendingDel.get(userId)) {
    if (text === '/cancel') {
      pendingDel.delete(userId);
      sendMessage(chatId, 'Отменено.');
      return;
    }
    const num = pendingDel.get(userId);
    pendingDel.delete(userId);
    const idx = num - 1;
    if (idx < 0 || idx >= memory.length) {
      sendMessage(chatId, `Фразы #${num} больше нет.`);
      return;
    }
    memory.splice(idx, 1);
    saveUserMemory(userId, memory);
    sendMessage(chatId, `Фраза #${num} удалена.`);
    return;
  }

  if (teachBuffer.get(userId)) {
    const q = teachBuffer.get(userId);
    teachBuffer.delete(userId);
    if (text === '/cancel') { sendMessage(chatId, 'Отменено.'); return; }
    if (learn(userId, q, text, memory)) {
      sendMessage(chatId, `Запомнил: "${q}" → "${text}"`);
    } else {
      sendMessage(chatId, 'Не удалось сохранить.');
    }
    return;
  }

  if (document && document.file_name && document.file_name.endsWith('.json')) {
    downloadFile(document.file_id, (content) => {
      try {
        const data = JSON.parse(content);
        if (Array.isArray(data)) {
          const valid = data.filter(e => e && e.tokens && Array.isArray(e.tokens.words) && Array.isArray(e.tokens.ngrams) && Array.isArray(e.answers));
          saveUserMemory(userId, valid);
          lastMessages.delete(userId);
          sendMessage(chatId, `Импортировано ${valid.length} фраз.`);
        } else {
          sendMessage(chatId, 'Некорректный формат файла.');
        }
      } catch (e) {
        sendMessage(chatId, 'Ошибка чтения файла.');
      }
    });
    return;
  }

  if (text && text.startsWith('/')) {
    const parts = text.trim().split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const arg = parts.slice(1).join(' ');

    if (cmd === '/start' || cmd === '/help') {
      lastMessages.delete(userId);
      pendingReset.delete(userId);
      pendingDel.delete(userId);
      teachBuffer.delete(userId);
      showHelp(chatId);
      return;
    }

    if (cmd === '/cancel') {
      lastMessages.delete(userId);
      pendingReset.delete(userId);
      pendingDel.delete(userId);
      teachBuffer.delete(userId);
      sendMessage(chatId, 'Контекст сброшен.');
      return;
    }

    if (cmd === '/reset') {
      pendingReset.set(userId, true);
      sendMessage(chatId, '<b>Уверен?</b> Вся база будет удалена.\n\nНапиши <b>да</b> для подтверждения.', { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/export') {
      if (memory.length === 0) { sendMessage(chatId, 'База пуста.'); return; }
      sendDocument(chatId, getUserMemoryFile(userId), `Экспорт базы (${memory.length} фраз)`);
      return;
    }

    if (cmd === '/import') {
      sendMessage(chatId, 'Отправь JSON-файл, полученный через /export.');
      return;
    }

    if (cmd === '/stats') { showStats(chatId, userId, memory); return; }

    if (cmd === '/list') {
      showList(chatId, memory, parseInt(arg, 10) || 1);
      return;
    }

    if (cmd === '/show') {
      const num = parseInt(arg, 10);
      if (!num) { sendMessage(chatId, 'Использование: /show &lt;номер&gt;', { parse_mode: 'HTML' }); return; }
      showEntry(chatId, memory, num);
      return;
    }

    if (cmd === '/find') { findEntries(chatId, memory, arg); return; }

    if (cmd === '/del') {
      const num = parseInt(arg, 10);
      if (!num || num < 1 || num > memory.length) {
        sendMessage(chatId, `Укажи номер 1–${memory.length}.`);
        return;
      }
      pendingDel.set(userId, num);
      const preview = escapeHtml(memory[num - 1].text.substring(0, 80));
      sendMessage(chatId, `Удалить #${num}?\n\n"${preview}"\n\n/cancel — отмена, любое другое — подтвердить.`, { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/teach') {
      const parsed = parseTeach(text);
      if (parsed) {
        if (learn(userId, parsed.q, parsed.a, memory)) {
          sendMessage(chatId, `Запомнил: "${parsed.q}" → "${parsed.a}"`);
        } else {
          sendMessage(chatId, 'Не удалось сохранить.');
        }
        return;
      }
      if (!arg) {
        sendMessage(chatId, 'Использование: /teach вопрос = ответ\nИли: /teach вопрос (и следующим сообщением — ответ)');
        return;
      }
      teachBuffer.set(userId, arg.trim());
      sendMessage(chatId, `Какой ответ на "${arg.trim()}"? Напиши следующим сообщением. /cancel — отмена.`);
      return;
    }

    if (cmd === '/forget') {
      const target = arg.trim();
      if (!target) { sendMessage(chatId, 'Использование: /forget &lt;текст&gt;', { parse_mode: 'HTML' }); return; }
      const cleanedTarget = cleanText(target);
      const before = memory.length;
      memory = memory.filter(e => cleanText(e.text) !== cleanedTarget);
      if (memory.length < before) {
        saveUserMemory(userId, memory);
        lastMessages.delete(userId);
        sendMessage(chatId, `Удалено фраз: ${before - memory.length}.`);
      } else {
        sendMessage(chatId, `Не найдено: "${escapeHtml(target)}". Попробуй /find`, { parse_mode: 'HTML' });
      }
      return;
    }

    sendMessage(chatId, 'Неизвестная команда. Смотри /help');
    return;
  }

  if (!text) return;

  const trimmed = text.trim();

  const parsedTeach = parseTeach(trimmed);
  if (parsedTeach) {
    if (learn(userId, parsedTeach.q, parsedTeach.a, memory)) {
      sendMessage(chatId, `Запомнил: "${parsedTeach.q}" → "${parsedTeach.a}"`);
    } else {
      sendMessage(chatId, 'Не удалось сохранить.');
    }
    return;
  }

  // ---- АССОЦИАТИВНАЯ ЛОГИКА ----

  // 1. Если ждём ответ на эхо — учим и СРАЗУ сбрасываем контекст
  const prevBotMessage = lastMessages.get(userId);
  if (prevBotMessage) {
    lastMessages.delete(userId);
    if (learn(userId, prevBotMessage, trimmed, memory)) {
      sendMessage(chatId, 'Понял, записал.');
    } else {
      sendMessage(chatId, 'Не удалось записать.');
    }
    return;
  }

  // 2. Ищем ответ в памяти
  const tokens = tokenize(trimmed);
  const aiResponse = findBestAnswer(tokens, memory);

  if (aiResponse) {
    // Знакомая фраза — отвечаем, контекст НЕ ставим (учиться не надо)
    sendMessage(chatId, aiResponse);
    return;
  }

  // 3. Незнакомая — эхо + ставим контекст ожидания ответа
  sendMessage(chatId, trimmed);
  lastMessages.set(userId, trimmed);
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
                handleMessage(
                  update.message.chat.id,
                  update.message.from.id,
                  update.message.text,
                  update.message.document
                );
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
