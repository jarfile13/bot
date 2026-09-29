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
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT);

const userData = new Map();
const pending = new Map();
const pendingReset = new Map();

const HIDDEN_DIM = 24;
const LR = 0.15;
const EPOCHS = 300;
const CONFIDENCE = 0.72;
const L2 = 0.0001;

function cleanText(str) {
  return String(str).toLowerCase().replace(/ё/g, 'е').replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function getWords(str) {
  return cleanText(str).split(' ').filter(w => w.length >= 1);
}

function userFile(userId) { return path.join(DATA_DIR, `user_${userId}.json`); }

function emptyUserData() {
  return { dataset: [], intents: [], vocabulary: [] };
}

function loadUserData(userId) {
  if (userData.has(userId)) return userData.get(userId);
  const file = userFile(userId);
  let data = emptyUserData();
  if (fs.existsSync(file)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (parsed && Array.isArray(parsed.dataset)) data = parsed;
    } catch (e) { data = emptyUserData(); }
  }
  userData.set(userId, data);
  return data;
}

function saveUserData(userId) {
  const data = userData.get(userId);
  if (!data) return;
  try { fs.writeFileSync(userFile(userId), JSON.stringify(data, null, 2), 'utf8'); } catch (e) {}
}

function rebuildVocabulary(data) {
  const vocab = [];
  const seen = new Set();
  for (const ex of data.dataset) {
    for (const w of getWords(ex.q)) {
      if (!seen.has(w)) { seen.add(w); vocab.push(w); }
    }
  }
  data.vocabulary = vocab;
}

function rebuildIntents(data) {
  const intents = [];
  const seen = new Set();
  for (const ex of data.dataset) {
    const a = ex.a.trim();
    if (!seen.has(a)) { seen.add(a); intents.push(a); }
  }
  data.intents = intents;
}

function bagOfWords(words, vocab) {
  const bag = new Array(vocab.length).fill(0);
  const idx = new Map();
  vocab.forEach((w, i) => idx.set(w, i));
  for (const w of words) {
    const i = idx.get(w);
    if (i !== undefined) bag[i] = 1;
  }
  return bag;
}

function sigmoid(x) {
  if (x < -50) return 0;
  if (x > 50) return 1;
  return 1 / (1 + Math.exp(-x));
}

function initWeights(inputDim, hiddenDim, outputDim) {
  const rng = () => (Math.random() * 2 - 1);
  const limitIH = Math.sqrt(6 / (inputDim + hiddenDim));
  const limitHO = Math.sqrt(6 / (hiddenDim + outputDim));
  return {
    wih: Array.from({ length: inputDim }, () => Array.from({ length: hiddenDim }, () => rng() * limitIH)),
    who: Array.from({ length: hiddenDim }, () => Array.from({ length: outputDim }, () => rng() * limitHO)),
    bh: new Array(hiddenDim).fill(0),
    bo: new Array(outputDim).fill(0)
  };
}

function forward(bag, W) {
  const hidden = new Array(W.bh.length).fill(0);
  for (let h = 0; h < W.bh.length; h++) {
    let s = W.bh[h];
    for (let i = 0; i < bag.length; i++) if (bag[i]) s += W.wih[i][h];
    hidden[h] = sigmoid(s);
  }
  const output = new Array(W.bo.length).fill(0);
  for (let o = 0; o < W.bo.length; o++) {
    let s = W.bo[o];
    for (let h = 0; h < hidden.length; h++) s += hidden[h] * W.who[h][o];
    output[o] = sigmoid(s);
  }
  return { hidden, output };
}

function trainFull(data) {
  const vocab = data.vocabulary;
  const intents = data.intents;
  if (vocab.length === 0 || intents.length === 0 || data.dataset.length === 0) return null;

  const inputDim = vocab.length;
  const hiddenDim = HIDDEN_DIM;
  const outputDim = intents.length;

  const W = initWeights(inputDim, hiddenDim, outputDim);

  const examples = data.dataset.map(ex => ({
    bag: bagOfWords(getWords(ex.q), vocab),
    target: intents.indexOf(ex.a.trim())
  })).filter(e => e.target !== -1);

  if (examples.length === 0) return null;

  let firstErr = 0, lastErr = 0;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    let epochErr = 0;
    const order = [...examples.keys()].sort(() => Math.random() - 0.5);
    for (const idx of order) {
      const { bag, target } = examples[idx];
      const { hidden, output } = forward(bag, W);

      const outErr = new Array(outputDim);
      for (let o = 0; o < outputDim; o++) {
        const t = o === target ? 1 : 0;
        outErr[o] = t - output[o];
        epochErr += outErr[o] * outErr[o];
      }

      const hidErr = new Array(hiddenDim).fill(0);
      for (let h = 0; h < hiddenDim; h++) {
        let e = 0;
        for (let o = 0; o < outputDim; o++) e += outErr[o] * output[o] * (1 - output[o]) * W.who[h][o];
        hidErr[h] = e;
      }

      for (let o = 0; o < outputDim; o++) {
        const g = outErr[o] * output[o] * (1 - output[o]) * LR;
        W.bo[o] += g;
        for (let h = 0; h < hiddenDim; h++) {
          W.who[h][o] += g * hidden[h] - L2 * W.who[h][o];
        }
      }

      for (let h = 0; h < hiddenDim; h++) {
        const g = hidErr[h] * hidden[h] * (1 - hidden[h]) * LR;
        W.bh[h] += g;
        for (let i = 0; i < inputDim; i++) {
          if (bag[i]) W.wih[i][h] += g - L2 * W.wih[i][h];
        }
      }
    }
    epochErr = Math.sqrt(epochErr / examples.length);
    if (epoch === 0) firstErr = epochErr;
    lastErr = epochErr;
  }

  return { W, firstErr, lastErr, examples: examples.length, inputDim, outputDim };
}

function predict(data, W, words) {
  const bag = bagOfWords(words, data.vocabulary);
  const total = bag.reduce((a, b) => a + b, 0);
  if (total === 0) return null;
  const { output } = forward(bag, W);
  let maxIdx = -1, maxVal = -1;
  for (let i = 0; i < output.length; i++) {
    if (output[i] > maxVal) { maxVal = output[i]; maxIdx = i; }
  }
  return { idx: maxIdx, val: maxVal, output };
}

function sendMessage(chatId, text, extra) {
  const payload = { chat_id: chatId, text: text };
  if (extra) Object.assign(payload, extra);
  const body = JSON.stringify(payload);
  const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
  });
  req.on('error', () => {});
  req.write(body);
  req.end();
}

function sendDocument(chatId, filePath, caption) {
  try {
    const boundary = '----Boundary' + Math.random().toString(36).slice(2);
    const filename = path.basename(filePath);
    const fileData = fs.readFileSync(filePath);
    const parts = [];
    parts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n${chatId}\r\n`));
    parts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n${caption}\r\n`));
    parts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="document"; filename="${filename}"\r\nContent-Type: application/json\r\n\r\n`));
    parts.push(fileData);
    parts.push(Buffer.from(`\r\n--${boundary}--\r\n`));
    const body = Buffer.concat(parts);
    const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/sendDocument`, {
      method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}`, 'Content-Length': body.length }
    });
    req.on('error', () => {});
    req.write(body);
    req.end();
  } catch (e) {}
}

function downloadFile(fileId, callback) {
  https.get(`https://api.telegram.org/bot${BOT_TOKEN}/getFile?file_id=${fileId}`, (res) => {
    let data = '';
    res.on('data', c => data += c);
    res.on('end', () => {
      try {
        const json = JSON.parse(data);
        if (json.ok && json.result.file_path) {
          https.get(`https://api.telegram.org/file/bot${BOT_TOKEN}/${json.result.file_path}`, fr => {
            const chunks = [];
            fr.on('data', c => chunks.push(c));
            fr.on('end', () => callback(Buffer.concat(chunks).toString('utf8')));
          });
        } else callback(null);
      } catch (e) { callback(null); }
    });
  }).on('error', () => callback(null));
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function addExample(userId, question, answer) {
  const data = loadUserData(userId);
  const q = question.trim();
  const a = answer.trim();
  if (!q || !a) return null;

  const existingIdx = data.dataset.findIndex(ex => cleanText(ex.q) === cleanText(q) && ex.a.trim() === a);
  let addedNew = false;
  if (existingIdx === -1) {
    data.dataset.push({ q, a });
    addedNew = true;
  }

  const beforeIntents = data.intents.length;
  const beforeVocab = data.vocabulary.length;

  rebuildVocabulary(data);
  rebuildIntents(data);

  const result = trainFull(data);
  if (!result) return null;

  data.W = result.W;
  saveUserData(userId);

  const addedIntents = data.intents.length - beforeIntents;
  const addedVocab = data.vocabulary.length - beforeVocab;

  const parts = [];
  if (addedNew) parts.push('пара добавлена');
  else parts.push('пара уже была');
  if (addedIntents > 0) parts.push(`+${addedIntents} интент`);
  if (addedVocab > 0) parts.push(`+${addedVocab} слов`);
  parts.push(`ошибка ${result.firstErr.toFixed(2)}→${result.lastErr.toFixed(2)}`);
  return parts.join(' · ');
}

function retrain(userId) {
  const data = loadUserData(userId);
  if (data.dataset.length === 0) return null;
  rebuildVocabulary(data);
  rebuildIntents(data);
  const result = trainFull(data);
  if (!result) return null;
  data.W = result.W;
  saveUserData(userId);
  return result;
}

function predictFor(userId, text) {
  const data = loadUserData(userId);
  if (!data.W || data.vocabulary.length === 0 || data.intents.length === 0) return null;
  const words = getWords(text);
  const pred = predict(data, data.W, words);
  if (!pred) return null;
  if (pred.val < CONFIDENCE) return { low: true, val: pred.val, response: data.intents[pred.idx] };
  return { low: false, val: pred.val, response: data.intents[pred.idx] };
}

function showHelp(chatId) {
  const text =
    '<b>Нейросеть на JS</b>\n\n' +
    'Обучается на парах «вопрос → ответ». Данные хранятся отдельно, сеть переобучается с нуля после каждой пары — поэтому не забывает старое.\n\n' +
    '<b>Обучение</b>\n' +
    'Просто напиши фразу — я повторю её и жду ответ.\n' +
    'Ответь — запомню пару.\n\n' +
    'Быстро:\n' +
    '<code>/teach вопрос = ответ</code>\n\n' +
    '<b>Управление</b>\n' +
    '/list — все пары\n' +
    '/show &lt;номер&gt; — подробности\n' +
    '/del &lt;номер&gt; — удалить пару\n' +
    '/find &lt;текст&gt; — поиск\n' +
    '/forget &lt;текст&gt; — удалить по вопросу\n' +
    '/retrain — переобучить сеть\n\n' +
    '<b>Данные</b>\n' +
    '/export — скачать базу\n' +
    '/import — как импортировать (отправь JSON)\n' +
    '/stats — статистика\n' +
    '/reset — стереть всё\n\n' +
    '/cancel — отмена';

  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showList(chatId, data, page) {
  if (data.dataset.length === 0) { sendMessage(chatId, 'база пуста'); return; }
  const PAGE = 10;
  const totalPages = Math.ceil(data.dataset.length / PAGE);
  const p = Math.max(1, Math.min(page, totalPages));
  const start = (p - 1) * PAGE;
  const slice = data.dataset.slice(start, start + PAGE);

  let out = `<b>Пары</b> (стр. ${p}/${totalPages}, всего ${data.dataset.length})\n\n`;
  slice.forEach((ex, i) => {
    const num = start + i + 1;
    const q = ex.q.length > 45 ? ex.q.slice(0, 45) + '…' : ex.q;
    const a = ex.a.length > 45 ? ex.a.slice(0, 45) + '…' : ex.a;
    out += `<b>${num}.</b> ${escapeHtml(q)}\n     → ${escapeHtml(a)}\n`;
  });
  if (totalPages > 1) out += `\n/list ${p + 1}`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function showEntry(chatId, data, num) {
  const idx = num - 1;
  if (idx < 0 || idx >= data.dataset.length) {
    sendMessage(chatId, `нет пары #${num}. всего: ${data.dataset.length}`);
    return;
  }
  const ex = data.dataset[idx];
  let out = `<b>Пара #${num}</b>\n\n`;
  out += `<b>Вопрос:</b> ${escapeHtml(ex.q)}\n\n`;
  out += `<b>Ответ:</b> ${escapeHtml(ex.a)}`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function findPairs(chatId, data, query) {
  const q = cleanText(query);
  if (!q) { sendMessage(chatId, 'формат: /find текст'); return; }
  const results = [];
  data.dataset.forEach((ex, i) => {
    if (cleanText(ex.q).includes(q) || cleanText(ex.a).includes(q)) {
      results.push({ num: i + 1, q: ex.q, a: ex.a });
    }
  });
  if (results.length === 0) {
    sendMessage(chatId, `не найдено: "${escapeHtml(query)}"`, { parse_mode: 'HTML' });
    return;
  }
  let out = `<b>Найдено: ${results.length}</b>\n\n`;
  results.slice(0, 15).forEach(r => {
    const q = r.q.length > 40 ? r.q.slice(0, 40) + '…' : r.q;
    const a = r.a.length > 40 ? r.a.slice(0, 40) + '…' : r.a;
    out += `<b>#${r.num}</b> ${escapeHtml(q)} → ${escapeHtml(a)}\n`;
  });
  if (results.length > 15) out += `\n…и ещё ${results.length - 15}`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function showStats(chatId, data) {
  const totalWeights = data.W ?
    (data.W.wih.length * HIDDEN_DIM + HIDDEN_DIM * (data.W.bo.length || 0)) : 0;

  let out = '<b>Статистика</b>\n\n';
  out += `Пар в базе: <b>${data.dataset.length}</b>\n`;
  out += `Слов в словаре: <b>${data.vocabulary.length}</b>\n`;
  out += `Интентов (ответов): <b>${data.intents.length}</b>\n`;
  out += `Нейронов скрытого слоя: <b>${HIDDEN_DIM}</b>\n`;
  out += `Синапсов: <b>${totalWeights}</b>\n`;
  out += `Порог уверенности: <b>${(CONFIDENCE * 100).toFixed(0)}%</b>`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function handleMessage(chatId, userId, text, document) {
  const data = loadUserData(userId);

  if (document && document.file_name && document.file_name.endsWith('.json')) {
    downloadFile(document.file_id, (content) => {
      if (!content) { sendMessage(chatId, 'не удалось скачать'); return; }
      try {
        const imported = JSON.parse(content);
        if (Array.isArray(imported.dataset)) {
          data.dataset = imported.dataset;
          rebuildVocabulary(data);
          rebuildIntents(data);
          const r = trainFull(data);
          if (r) { data.W = r.W; saveUserData(userId); }
          pending.delete(userId);
          sendMessage(chatId, `импортировано пар: ${data.dataset.length}, обучено заново`);
        } else if (Array.isArray(imported)) {
          data.dataset = imported.filter(e => e.q && e.a);
          rebuildVocabulary(data);
          rebuildIntents(data);
          const r = trainFull(data);
          if (r) { data.W = r.W; saveUserData(userId); }
          sendMessage(chatId, `импортировано пар: ${data.dataset.length}`);
        } else {
          sendMessage(chatId, 'неверный формат');
        }
      } catch (e) { sendMessage(chatId, 'ошибка чтения файла'); }
    });
    return;
  }

  if (text && text.startsWith('/')) {
    const trimmed = text.trim();
    const parts = trimmed.split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const arg = parts.slice(1).join(' ');

    if (cmd === '/start' || cmd === '/help') {
      pending.delete(userId);
      pendingReset.delete(userId);
      showHelp(chatId);
      return;
    }

    if (cmd === '/cancel') {
      pending.delete(userId);
      pendingReset.delete(userId);
      sendMessage(chatId, 'отменено');
      return;
    }

    if (cmd === '/reset') {
      pendingReset.set(userId, true);
      sendMessage(chatId, 'Уверен? Вся база и сеть будут удалены.\n\nНапиши <b>да</b> для подтверждения.', { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/export') {
      if (data.dataset.length === 0) { sendMessage(chatId, 'база пуста'); return; }
      sendDocument(chatId, userFile(userId), `база пар (${data.dataset.length})`);
      return;
    }

    if (cmd === '/import') {
      sendMessage(chatId, 'Отправь JSON-файл с полем dataset (из /export).\n\nТекущая база будет заменена.');
      return;
    }

    if (cmd === '/stats') { showStats(chatId, data); return; }

    if (cmd === '/list') {
      showList(chatId, data, parseInt(arg, 10) || 1);
      return;
    }

    if (cmd === '/show') {
      const num = parseInt(arg, 10);
      if (!num) { sendMessage(chatId, 'формат: /show <номер>'); return; }
      showEntry(chatId, data, num);
      return;
    }

    if (cmd === '/find') { findPairs(chatId, data, arg); return; }

    if (cmd === '/del') {
      const num = parseInt(arg, 10);
      if (!num || num < 1 || num > data.dataset.length) {
        sendMessage(chatId, `укажи номер 1–${data.dataset.length}`);
        return;
      }
      const removed = data.dataset.splice(num - 1, 1)[0];
      rebuildVocabulary(data);
      rebuildIntents(data);
      const r = trainFull(data);
      if (r) { data.W = r.W; saveUserData(userId); }
      sendMessage(chatId, `удалено #${num}: "${escapeHtml(removed.q.slice(0, 60))}"`, { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/forget') {
      const target = arg.trim();
      if (!target) { sendMessage(chatId, 'формат: /forget вопрос'); return; }
      const before = data.dataset.length;
      data.dataset = data.dataset.filter(ex => cleanText(ex.q) !== cleanText(target));
      if (data.dataset.length === before) {
        sendMessage(chatId, `не найдено: "${escapeHtml(target)}". попробуй /find`, { parse_mode: 'HTML' });
        return;
      }
      rebuildVocabulary(data);
      rebuildIntents(data);
      const r = trainFull(data);
      if (r) { data.W = r.W; saveUserData(userId); }
      sendMessage(chatId, `удалено пар: ${before - data.dataset.length}`);
      return;
    }

    if (cmd === '/retrain') {
      const r = retrain(userId);
      if (!r) { sendMessage(chatId, 'нечего обучать'); return; }
      sendMessage(chatId, `переобучено: ${r.examples} пар, ошибка ${r.firstErr.toFixed(2)}→${r.lastErr.toFixed(2)}`);
      return;
    }

    if (cmd === '/teach') {
      const m = trimmed.match(/^\/teach\s+(.+?)\s*=\s*(.+)$/s);
      if (m) {
        const report = addExample(userId, m[1].trim(), m[2].trim());
        if (report) sendMessage(chatId, 'запомнил: ' + report);
        else sendMessage(chatId, 'не удалось');
        return;
      }
      if (!arg) {
        sendMessage(chatId, 'формат:\n/teach вопрос = ответ\nили\n/teach вопрос (и следующим сообщением — ответ)');
        return;
      }
      pending.set(userId, { type: 'teach', question: arg.trim() });
      sendMessage(chatId, `Какой ответ на "${escapeHtml(arg.trim())}"? Напиши следующим сообщением. /cancel — отмена.`, { parse_mode: 'HTML' });
      return;
    }

    sendMessage(chatId, 'неизвестная команда, смотри /help');
    return;
  }

  if (!text) return;

  const trimmedText = text.trim();
  const rawWords = getWords(trimmedText);

  if (pendingReset.get(userId)) {
    const a = trimmedText.toLowerCase();
    if (a === 'да' || a === 'yes' || a === 'y') {
      const file = userFile(userId);
      if (fs.existsSync(file)) { try { fs.unlinkSync(file); } catch (e) {} }
      userData.delete(userId);
      pendingReset.delete(userId);
      sendMessage(chatId, 'всё стёрто');
    } else {
      pendingReset.delete(userId);
      sendMessage(chatId, 'отменено');
    }
    return;
  }

  const p = pending.get(userId);
  if (p && p.type === 'teach') {
    pending.delete(userId);
    const report = addExample(userId, p.question, trimmedText);
    if (report) sendMessage(chatId, 'запомнил: ' + report);
    else sendMessage(chatId, 'не удалось');
    return;
  }

  if (p && p.type === 'echo') {
    pending.delete(userId);
    const report = addExample(userId, p.question, trimmedText);
    if (report) sendMessage(chatId, 'запомнил: ' + report);
    else sendMessage(chatId, 'не удалось');
    return;
  }

  if (rawWords.length === 0) return;

  if (data.dataset.length === 0 || !data.W) {
    pending.set(userId, { type: 'echo', question: trimmedText });
    sendMessage(chatId, escapeHtml(trimmedText) + '\n\n<i>чему учить?</i>', { parse_mode: 'HTML' });
    return;
  }

  const pred = predictFor(userId, trimmedText);
  if (pred && !pred.low) {
    sendMessage(chatId, pred.response);
    return;
  }

  pending.set(userId, { type: 'echo', question: trimmedText });
  const conf = pred ? ` (уверенность ${(pred.val * 100).toFixed(0)}%)` : ' (нет сигнала)';
  sendMessage(chatId, escapeHtml(trimmedText) + `\n\n<i>не уверен${conf}. чему учить?</i>`, { parse_mode: 'HTML' });
}

let offset = 0;
let polling = false;

function getUpdates() {
  if (polling) return;
  polling = true;
  const url = `https://api.telegram.org/bot${BOT_TOKEN}/getUpdates?offset=${offset}&timeout=25`;
  https.get(url, (res) => {
    let data = '';
    res.on('data', c => data += c);
    res.on('end', () => {
      polling = false;
      try {
        const json = JSON.parse(data);
        if (json.ok && json.result && json.result.length > 0) {
          for (const update of json.result) {
            offset = update.update_id + 1;
            if (update.message) {
              handleMessage(update.message.chat.id, update.message.from.id, update.message.text, update.message.document);
            }
          }
        }
      } catch (e) {}
      setTimeout(getUpdates, 300);
    });
  }).on('error', () => {
    polling = false;
    setTimeout(getUpdates, 3000);
  });
}

getUpdates();
