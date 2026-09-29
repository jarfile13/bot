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

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT, () => {
  console.log(`[HTTP] Сервер слушает порт ${PORT}`);
});

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

function sendMessage(chatId, text) {
  const data = JSON.stringify({ chat_id: chatId, text: text });
  const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }
  });
  req.on('error', (e) => console.error('[ОШИБКА ОТПРАВКИ СООБЩЕНИЯ]', e.message));
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
    console.error('[ОШИБКА ФАЙЛА ПРИ ЭКСПОРТЕ]', e.message);
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
            let fileContent = '';
            fileRes.on('data', chunk => fileContent += chunk);
            fileRes.on('end', () => callback(fileContent));
          });
        } else {
          console.error('[ОШИБКА ТГ] Не удалось получить путь к файлу:', json.description);
        }
      } catch (e) {
        console.error('[ОШИБКА СКАЧИВАНИЯ]', e.message);
      }
    });
  }).on('error', (e) => console.error('[ОШИБКА ЗАПРОСА СКАЧИВАНИЯ]', e.message));
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

  let existing = memory.find(entry => {
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
      answers: [answer.trim()]
    });
  }

  saveUserMemory(userId, memory);
}

function handleMessage(chatId, userId, text, document) {
  let memory = loadUserMemory(userId);

  if (document && document.file_name && document.file_name.endsWith('.json')) {
    downloadFile(document.file_id, (content) => {
      try {
        const importedData = JSON.parse(content);
        if (Array.isArray(importedData)) {
          saveUserMemory(userId, importedData);
          sendMessage(chatId, 'Твоя личная база знаний успешно импортирована');
        } else {
          sendMessage(chatId, 'Некорректный формат файла базы знаний');
        }
      } catch (e) {
        sendMessage(chatId, 'Ошибка при чтении файла');
      }
    });
    return;
  }

  if (text && text.startsWith('/')) {
    if (text === '/start' || text === '/help') {
      sendMessage(chatId, 'Привет! Я учусь прямо в процессе нашего общения.\n\nКоманды:\n/start или /help - Вывод этой справки\n/reset - Полностью стереть свою память\n/export - Скачать свою базу знаний\n/forget [фраза] - Забыть конкретную фразу\n/stats - Посмотреть объем памяти');
      return;
    }
    if (text === '/reset') {
      const file = getUserMemoryFile(userId);
      if (fs.existsSync(file)) fs.unlinkSync(file);
      userMemories.delete(userId);
      lastMessages.delete(userId);
      sendMessage(chatId, 'Твоя личная память полностью очищена');
      return;
    }
    if (text === '/export') {
      if (memory.length === 0) {
        sendMessage(chatId, 'Твоя база знаний пока пуста для экспорта');
        return;
      }
      sendDocument(chatId, getUserMemoryFile(userId), 'Экспорт твоей личной базы знаний ИИ');
      return;
    }
    if (text.startsWith('/forget ')) {
      const target = text.substring(8).trim();
      const cleanedTarget = cleanText(target);
      const initialLength = memory.length;
      memory = memory.filter(entry => cleanText(entry.text) !== cleanedTarget);
      if (memory.length < initialLength) {
        saveUserMemory(userId, memory);
        sendMessage(chatId, `Воспоминание "${target}" успешно удалено из твоей базы`);
      } else {
        sendMessage(chatId, `Воспоминание "${target}" не найдено в твоей базе`);
      }
      return;
    }
    if (text === '/stats') {
      let answersCount = 0;
      memory.forEach(e => answersCount += e.answers.length);
      sendMessage(chatId, `Личная статистика ИИ:\nУникальных фраз: ${memory.length}\nВсего вариантов ответов: ${answersCount}`);
      return;
    }
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
          console.error('[ОШИБКА TELEGRAM API]', json.description);
          if (json.error_code === 409) {
            console.error('[КОНФЛИКТ] Удалите webhook или запустите только один экземпляр бота');
          }
        }
      } catch (e) {
        console.error('[ОШИБКА ОБРАБОТКИ ПАКЕТА]', e.message);
      }
      setTimeout(getUpdates, 500);
    });
  }).on('error', (e) => {
    polling = false;
    console.error('[ОШИБКА СЕТИ getUpdates]', e.message);
    setTimeout(getUpdates, 3000);
  });
}

console.log('[РОБОТ] Запущен. Ожидание сообщений...');
getUpdates();
