const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BOT_TOKEN = process.env.BOT_TOKEN;
const DATA_DIR = process.env.DATA_DIR || './data';
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(3000);

const memoryFile = path.join(DATA_DIR, 'ai_memory.json');
let memory = [];

if (fs.existsSync(memoryFile)) {
  try { memory = JSON.parse(fs.readFileSync(memoryFile, 'utf8')); } catch (e) { memory = []; }
}

const lastMessages = new Map();

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
  const req = https.request(`https://telegram.org{BOT_TOKEN}/sendMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': data.length }
  });
  req.write(data);
  req.end();
}

function sendDocument(chatId, filePath, caption) {
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

  const req = https.request(`https://telegram.org{BOT_TOKEN}/sendDocument`, {
    method: 'POST',
    headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}` }
  });

  req.write(header);
  req.write(fileData);
  req.write(footer);
  req.end();
}

function downloadFile(fileId, callback) {
  https.get(`https://telegram.org{BOT_TOKEN}/getFile?file_id=${fileId}`, (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      try {
        const json = JSON.parse(data);
        if (json.ok && json.result.file_path) {
          https.get(`https://telegram.org{BOT_TOKEN}/${json.result.file_path}`, (fileRes) => {
            let fileContent = '';
            fileRes.on('data', chunk => fileContent += chunk);
            fileRes.on('end', () => callback(fileContent));
          });
        }
      } catch (e) {}
    });
  });
}

function findBestAnswer(userTokens) {
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

function learn(question, answer) {
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

  fs.writeFileSync(memoryFile, JSON.stringify(memory, null, 2), 'utf8');
}

function handleMessage(chatId, text, document) {
  if (document && document.file_name && document.file_name.endsWith('.json')) {
    downloadFile(document.file_id, (content) => {
      try {
        const importedData = JSON.parse(content);
        if (Array.isArray(importedData)) {
          memory = importedData;
          fs.writeFileSync(memoryFile, JSON.stringify(memory, null, 2), 'utf8');
          sendMessage(chatId, 'База знаний успешно импортирована');
        } else {
          sendMessage(chatId, 'Некорректный формат базы знаний');
        }
      } catch (e) {
        sendMessage(chatId, 'Ошибка при чтении файла');
      }
    });
    return;
  }

  if (text && text.startsWith('/')) {
    if (text === '/reset') {
      memory = [];
      if (fs.existsSync(memoryFile)) fs.unlinkSync(memoryFile);
      lastMessages.delete(chatId);
      sendMessage(chatId, 'Память полностью очищена');
      return;
    }
    if (text === '/export') {
      if (memory.length === 0) {
        sendMessage(chatId, 'База знаний пока пуста для экспорта');
        return;
      }
      sendDocument(chatId, memoryFile, 'Экспорт базы знаний текстового ИИ');
      return;
    }
    if (text.startsWith('/forget ')) {
      const target = text.substring(8).trim();
      const cleanedTarget = cleanText(target);
      const initialLength = memory.length;
      memory = memory.filter(entry => cleanText(entry.text) !== cleanedTarget);
      if (memory.length < initialLength) {
        fs.writeFileSync(memoryFile, JSON.stringify(memory, null, 2), 'utf8');
        sendMessage(chatId, `Воспоминание "${target}" успешно удалено`);
      } else {
        sendMessage(chatId, `Воспоминание "${target}" не найдено`);
      }
      return;
    }
    if (text === '/stats') {
      let answersCount = 0;
      memory.forEach(e => answersCount += e.answers.length);
      sendMessage(chatId, `Статистика ИИ:\nУникальных фраз: ${memory.length}\nВсего вариантов ответов: ${answersCount}`);
      return;
    }
    if (text === '/help') {
      sendMessage(chatId, 'Команды:\n/reset - Стереть память\n/export - Скачать базу знаний\n/forget [фраза] - Забыть конкретную фразу\n/stats - Посмотреть объем памяти');
      return;
    }
  }

  if (!text) return;

  const cleanTextStr = text.trim();
  const tokens = tokenize(cleanTextStr);

  const prevBotMessage = lastMessages.get(chatId);
  if (prevBotMessage) {
    learn(prevBotMessage, cleanTextStr);
  }

  const aiResponse = findBestAnswer(tokens);

  if (aiResponse) {
    sendMessage(chatId, aiResponse);
    lastMessages.set(chatId, aiResponse);
  } else {
    sendMessage(chatId, cleanTextStr);
    lastMessages.set(chatId, cleanTextStr);
  }
}

let offset = 0;
function getUpdates() {
  https.get(`https://telegram.org{BOT_TOKEN}/getUpdates?offset=${offset}&timeout=30`, (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      try {
        const json = JSON.parse(data);
        if (json.ok && json.result.length > 0) {
          for (const update of json.result) {
            offset = update.update_id + 1;
            if (update.message) {
              handleMessage(update.message.chat.id, update.message.text, update.message.document);
            }
          }
        }
      } catch (e) {}
      getUpdates();
    });
  }).on('error', () => setTimeout(getUpdates, 1000));
}

getUpdates();
