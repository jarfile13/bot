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

const userNetworks = new Map();
const pending = new Map();
const pendingReset = new Map();

function cleanText(str) {
  return String(str).toLowerCase().replace(/ё/g, 'е').replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function getWords(str) {
  return cleanText(str).split(' ').filter(w => w.length >= 1);
}

function initNetwork() {
  return {
    vocabulary: [],
    intents: [],
    weights_ih: [],
    weights_ho: [],
    bias_h: [],
    bias_o: []
  };
}

function getUserNetFile(userId) {
  return path.join(DATA_DIR, `net_${userId}.json`);
}

function loadUserNet(userId) {
  if (userNetworks.has(userId)) return userNetworks.get(userId);
  const file = getUserNetFile(userId);
  let net = initNetwork();
  if (fs.existsSync(file)) {
    try { net = JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) { net = initNetwork(); }
  }
  userNetworks.set(userId, net);
  return net;
}

function saveUserNet(userId, net) {
  userNetworks.set(userId, net);
  try { fs.writeFileSync(getUserNetFile(userId), JSON.stringify(net, null, 2), 'utf8'); } catch (e) {}
}

function buildBagOfWords(words, vocabulary) {
  const bag = new Array(vocabulary.length).fill(0);
  words.forEach(w => {
    const idx = vocabulary.indexOf(w);
    if (idx !== -1) bag[idx] = 1;
  });
  return bag;
}

function sigmoid(x) {
  if (x < -50) return 0;
  if (x > 50) return 1;
  return 1 / (1 + Math.exp(-x));
}

function dSigmoid(y) {
  return y * (1 - y);
}

function forward(inputBag, net) {
  const hiddenDim = net.bias_h.length;
  const outputDim = net.bias_o.length;

  const hidden = new Array(hiddenDim).fill(0);
  for (let h = 0; h < hiddenDim; h++) {
    let sum = net.bias_h[h];
    for (let i = 0; i < inputBag.length; i++) {
      sum += inputBag[i] * net.weights_ih[i][h];
    }
    hidden[h] = sigmoid(sum);
  }

  const output = new Array(outputDim).fill(0);
  for (let o = 0; o < outputDim; o++) {
    let sum = net.bias_o[o];
    for (let h = 0; h < hiddenDim; h++) {
      sum += hidden[h] * net.weights_ho[h][o];
    }
    output[o] = sigmoid(sum);
  }

  return { hidden, output };
}

function trainNetwork(inputBag, targetIdx, net, lr = 0.6, iterations = 40) {
  const hiddenDim = net.bias_h.length;
  const outputDim = net.bias_o.length;
  const targets = new Array(outputDim).fill(0);
  targets[targetIdx] = 1;

  let firstErr = 0;
  let lastErr = 0;

  for (let iter = 0; iter < iterations; iter++) {
    const { hidden, output } = forward(inputBag, net);

    const outputErrors = new Array(outputDim);
    let errSum = 0;
    for (let o = 0; o < outputDim; o++) {
      outputErrors[o] = targets[o] - output[o];
      errSum += outputErrors[o] * outputErrors[o];
    }
    errSum = Math.sqrt(errSum);
    if (iter === 0) firstErr = errSum;
    lastErr = errSum;

    const hiddenErrors = new Array(hiddenDim).fill(0);
    for (let h = 0; h < hiddenDim; h++) {
      let error = 0;
      for (let o = 0; o < outputDim; o++) {
        error += outputErrors[o] * dSigmoid(output[o]) * net.weights_ho[h][o];
      }
      hiddenErrors[h] = error;
    }

    for (let o = 0; o < outputDim; o++) {
      const gradient = outputErrors[o] * dSigmoid(output[o]) * lr;
      net.bias_o[o] += gradient;
      for (let h = 0; h < hiddenDim; h++) {
        net.weights_ho[h][o] += gradient * hidden[h];
      }
    }

    for (let h = 0; h < hiddenDim; h++) {
      const gradient = hiddenErrors[h] * dSigmoid(hidden[h]) * lr;
      net.bias_h[h] += gradient;
      for (let i = 0; i < inputBag.length; i++) {
        net.weights_ih[i][h] += gradient * inputBag[i];
      }
    }
  }

  return { firstErr, lastErr };
}

function updateNetworkStructure(net, newWords, newResponse) {
  let vocabChanged = false;
  newWords.forEach(w => {
    if (!net.vocabulary.includes(w)) {
      net.vocabulary.push(w);
      vocabChanged = true;
    }
  });

  let intentIdx = net.intents.findIndex(id => id.response === newResponse);
  let isNewIntent = false;
  if (intentIdx === -1) {
    net.intents.push({ response: newResponse });
    intentIdx = net.intents.length - 1;
    isNewIntent = true;
  }

  const inputDim = net.vocabulary.length;
  const outputDim = net.intents.length;
  const hiddenDim = 16;

  const needRebuild = vocabChanged || net.bias_h.length === 0 || net.bias_o.length !== outputDim ||
                      net.weights_ih.length !== inputDim || net.weights_ho.length !== hiddenDim;

  if (needRebuild) {
    const oldWeightsIH = net.weights_ih;
    net.weights_ih = Array.from({ length: inputDim }, () => new Array(hiddenDim).fill(0));
    for (let i = 0; i < inputDim; i++) {
      for (let h = 0; h < hiddenDim; h++) {
        if (oldWeightsIH && oldWeightsIH[i] && oldWeightsIH[i][h] !== undefined) {
          net.weights_ih[i][h] = oldWeightsIH[i][h];
        } else {
          net.weights_ih[i][h] = (Math.random() * 2 - 1) * 0.1;
        }
      }
    }

    if (net.bias_h.length !== hiddenDim) {
      const oldBiasH = net.bias_h;
      net.bias_h = Array.from({ length: hiddenDim }, (_, h) => {
        if (oldBiasH && oldBiasH[h] !== undefined) return oldBiasH[h];
        return (Math.random() * 2 - 1) * 0.1;
      });
    }

    const oldWeightsHO = net.weights_ho;
    net.weights_ho = Array.from({ length: hiddenDim }, () => new Array(outputDim).fill(0));
    for (let h = 0; h < hiddenDim; h++) {
      for (let o = 0; o < outputDim; o++) {
        if (oldWeightsHO && oldWeightsHO[h] && oldWeightsHO[h][o] !== undefined) {
          net.weights_ho[h][o] = oldWeightsHO[h][o];
        } else {
          net.weights_ho[h][o] = (Math.random() * 2 - 1) * 0.1;
        }
      }
    }

    const oldBiasO = net.bias_o;
    net.bias_o = new Array(outputDim).fill(0);
    for (let o = 0; o < outputDim; o++) {
      if (oldBiasO && oldBiasO[o] !== undefined) {
        net.bias_o[o] = oldBiasO[o];
      } else {
        net.bias_o[o] = (Math.random() * 2 - 1) * 0.1;
      }
    }
  }

  return { intentIdx, isNewIntent, vocabChanged };
}

function sendMessage(chatId, text, extra) {
  const payload = { chat_id: chatId, text: text };
  if (extra) Object.assign(payload, extra);
  const data = JSON.stringify(payload);
  const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }
  });
  req.on('error', () => {});
  req.write(data);
  req.end();
}

function sendDocument(chatId, filePath, caption) {
  try {
    const boundary = '----WebKitFormBoundary' + Math.random().toString(36).substring(2);
    const filename = path.basename(filePath);
    const fileData = fs.readFileSync(filePath);
    let header = `--${boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n${chatId}\r\n`;
    header += `--${boundary}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n${caption}\r\n`;
    header += `--${boundary}\r\nContent-Disposition: form-data; name="document"; filename="${filename}"\r\nContent-Type: application/json\r\n\r\n`;
    const footer = `\r\n--${boundary}--\r\n`;
    const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/sendDocument`, {
      method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}` }
    });
    req.on('error', () => {});
    req.write(header);
    req.write(fileData);
    req.write(footer);
    req.end();
  } catch (e) {}
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
            let content = '';
            fileRes.on('data', chunk => content += chunk);
            fileRes.on('end', () => callback(content));
          });
        } else {
          callback(null);
        }
      } catch (e) { callback(null); }
    });
  }).on('error', () => callback(null));
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function teachAndReport(userId, question, answer) {
  const net = loadUserNet(userId);
  const words = getWords(question);
  if (words.length === 0 || !answer.trim()) return null;

  const beforeVocab = net.vocabulary.length;
  const beforeIntents = net.intents.length;

  const info = updateNetworkStructure(net, words, answer.trim());
  const refreshedNet = loadUserNet(userId);
  const bag = buildBagOfWords(words, refreshedNet.vocabulary);
  const { firstErr, lastErr } = trainNetwork(bag, info.intentIdx, refreshedNet, 0.6, 40);
  saveUserNet(userId, refreshedNet);

  const addedWords = refreshedNet.vocabulary.length - beforeVocab;
  const addedIntents = refreshedNet.intents.length - beforeIntents;
  const parts = [];
  if (addedIntents > 0) parts.push(`+${addedIntents} интент`);
  if (addedWords > 0) parts.push(`+${addedWords} слов`);
  parts.push(`ошибка ${firstErr.toFixed(2)}→${lastErr.toFixed(2)}`);
  return parts.join(' · ');
}

function showHelp(chatId) {
  const text =
    '<b>Нейросеть на JS (backpropagation)</b>\n\n' +
    '<b>Обучение</b>\n' +
    'Напиши фразу — я повторю её и буду ждать ответ.\n' +
    'Ответь — запомню пару «вопрос → ответ».\n\n' +
    'Или сразу:\n' +
    '<code>/teach вопрос = ответ</code>\n' +
    '<code>/teach вопрос</code> — и следующим сообщением ответ\n\n' +
    '<b>Управление</b>\n' +
    '/list — все выученные пары\n' +
    '/show &lt;номер&gt; — подробности о паре\n' +
    '/del &lt;номер&gt; — удалить пару\n' +
    '/find &lt;текст&gt; — поиск по выученному\n' +
    '/forget &lt;текст&gt; — удалить по точному вопросу\n\n' +
    '<b>Данные</b>\n' +
    '/export — скачать веса сети (JSON)\n' +
    '/import — как импортировать (отправь JSON)\n' +
    '/stats — статистика сети\n' +
    '/reset — стереть всё (с подтверждением)\n\n' +
    '<b>Прочее</b>\n' +
    '/cancel — отменить текущее действие';

  sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function getIntents(net) {
  return net.intents.map((it, i) => ({ idx: i, response: it.response }));
}

function findIntentByResponse(net, text) {
  const t = cleanText(text);
  return net.intents.findIndex(it => cleanText(it.response) === t);
}

function showList(chatId, net, page) {
  if (net.intents.length === 0) {
    sendMessage(chatId, 'сеть пуста');
    return;
  }
  const PAGE = 10;
  const totalPages = Math.ceil(net.intents.length / PAGE);
  const p = Math.max(1, Math.min(page, totalPages));
  const start = (p - 1) * PAGE;
  const slice = net.intents.slice(start, start + PAGE);

  let out = `<b>Выученные ответы</b> (стр. ${p}/${totalPages}, всего ${net.intents.length})\n\n`;
  slice.forEach((it, i) => {
    const num = start + i + 1;
    const preview = it.response.length > 80 ? it.response.slice(0, 80) + '…' : it.response;
    out += `<b>${num}.</b> ${escapeHtml(preview)}\n`;
  });
  if (totalPages > 1) out += `\n/list ${p + 1}`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function showEntry(chatId, net, num) {
  const idx = num - 1;
  if (idx < 0 || idx >= net.intents.length) {
    sendMessage(chatId, `нет интента #${num}. всего: ${net.intents.length}`);
    return;
  }
  const it = net.intents[idx];
  let out = `<b>Интент #${num}</b>\n\n`;
  out += `Ответ: ${escapeHtml(it.response)}\n\n`;
  out += `Слов в словаре: ${net.vocabulary.length}\n`;
  out += `Всего интентов: ${net.intents.length}`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function findPairs(chatId, net, query) {
  const q = cleanText(query);
  if (!q) { sendMessage(chatId, 'формат: /find текст'); return; }
  const results = [];
  net.intents.forEach((it, i) => {
    if (cleanText(it.response).includes(q)) {
      results.push({ num: i + 1, response: it.response });
    }
  });
  if (results.length === 0) {
    sendMessage(chatId, `не найдено: "${escapeHtml(query)}"`, { parse_mode: 'HTML' });
    return;
  }
  let out = `<b>Найдено: ${results.length}</b>\n\n`;
  results.slice(0, 15).forEach(r => {
    const preview = r.response.length > 70 ? r.response.slice(0, 70) + '…' : r.response;
    out += `<b>#${r.num}</b> ${escapeHtml(preview)}\n`;
  });
  if (results.length > 15) out += `\n…и ещё ${results.length - 15}`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function showStats(chatId, net) {
  const totalWeights = net.weights_ih.reduce((s, row) => s + row.length, 0) +
                       net.weights_ho.reduce((s, row) => s + row.length, 0);
  let out = '<b>Статистика сети</b>\n\n';
  out += `Слов в словаре: <b>${net.vocabulary.length}</b>\n`;
  out += `Интентов (ответов): <b>${net.intents.length}</b>\n`;
  out += `Нейронов скрытого слоя: <b>${net.bias_h.length}</b>\n`;
  out += `Синапсов: <b>${totalWeights}</b>\n`;
  out += `Параметров (весов и смещений): <b>${totalWeights + net.bias_h.length + net.bias_o.length}</b>`;
  sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function handleMessage(chatId, userId, text, document) {
  let net = loadUserNet(userId);

  if (document && document.file_name && document.file_name.endsWith('.json')) {
    downloadFile(document.file_id, (content) => {
      if (!content) { sendMessage(chatId, 'не удалось скачать файл'); return; }
      try {
        const imported = JSON.parse(content);
        if (imported.weights_ih && imported.intents && imported.vocabulary) {
          saveUserNet(userId, imported);
          pending.delete(userId);
          pendingReset.delete(userId);
          sendMessage(chatId, `сеть загружена: ${imported.vocabulary.length} слов, ${imported.intents.length} интентов`);
        } else {
          sendMessage(chatId, 'неверный формат файла');
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
      sendMessage(chatId, 'Уверен? Вся сеть будет удалена.\n\nНапиши <b>да</b> для подтверждения.', { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/export') {
      if (net.vocabulary.length === 0) { sendMessage(chatId, 'сеть пуста'); return; }
      sendDocument(chatId, getUserNetFile(userId), `веса сети (${net.vocabulary.length} слов, ${net.intents.length} интентов)`);
      return;
    }

    if (cmd === '/import') {
      sendMessage(chatId, 'Отправь JSON-файл с весами сети (полученный через /export).\n\nТекущая сеть будет заменена.');
      return;
    }

    if (cmd === '/stats') { showStats(chatId, net); return; }

    if (cmd === '/list') {
      const page = parseInt(arg, 10) || 1;
      showList(chatId, net, page);
      return;
    }

    if (cmd === '/show') {
      const num = parseInt(arg, 10);
      if (!num) { sendMessage(chatId, 'формат: /show <номер>'); return; }
      showEntry(chatId, net, num);
      return;
    }

    if (cmd === '/find') {
      findPairs(chatId, net, arg);
      return;
    }

    if (cmd === '/del') {
      const num = parseInt(arg, 10);
      if (!num || num < 1 || num > net.intents.length) {
        sendMessage(chatId, `укажи номер 1–${net.intents.length}`);
        return;
      }
      const removed = net.intents.splice(num - 1, 1)[0];
      saveUserNet(userId, net);
      sendMessage(chatId, `удалён интент #${num}: "${escapeHtml(removed.response.slice(0, 80))}"`, { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/forget') {
      const target = arg.trim();
      if (!target) { sendMessage(chatId, 'формат: /forget текст ответа'); return; }
      const idx = findIntentByResponse(net, target);
      if (idx === -1) {
        sendMessage(chatId, `не найдено: "${escapeHtml(target)}". попробуй /find`, { parse_mode: 'HTML' });
        return;
      }
      net.intents.splice(idx, 1);
      saveUserNet(userId, net);
      sendMessage(chatId, `удалено: "${escapeHtml(target)}"`, { parse_mode: 'HTML' });
      return;
    }

    if (cmd === '/teach') {
      const m = trimmed.match(/^\/teach\s+(.+?)\s*=\s*(.+)$/s);
      if (m) {
        const report = teachAndReport(userId, m[1].trim(), m[2].trim());
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

  const rawWords = getWords(text);
  const trimmedText = text.trim();

  if (pendingReset.get(userId)) {
    const a = trimmedText.toLowerCase();
    if (a === 'да' || a === 'yes' || a === 'y') {
      const file = getUserNetFile(userId);
      if (fs.existsSync(file)) { try { fs.unlinkSync(file); } catch (e) {} }
      userNetworks.delete(userId);
      pendingReset.delete(userId);
      sendMessage(chatId, 'сеть стёрта');
    } else {
      pendingReset.delete(userId);
      sendMessage(chatId, 'отменено');
    }
    return;
  }

  const p = pending.get(userId);
  if (p) {
    if (p.type === 'teach') {
      pending.delete(userId);
      const report = teachAndReport(userId, p.question, trimmedText);
      if (report) sendMessage(chatId, 'запомнил: ' + report);
      else sendMessage(chatId, 'не удалось');
      return;
    }
    if (p.type === 'echo') {
      pending.delete(userId);
      const report = teachAndReport(userId, p.question, trimmedText);
      if (report) sendMessage(chatId, 'запомнил: ' + report);
      else sendMessage(chatId, 'не удалось');
      return;
    }
  }

  if (rawWords.length === 0) return;

  if (net.vocabulary.length === 0 || net.intents.length === 0) {
    pending.set(userId, { type: 'echo', question: trimmedText });
    sendMessage(chatId, escapeHtml(trimmedText) + '\n\n<i>чему учить?</i>', { parse_mode: 'HTML' });
    return;
  }

  const bag = buildBagOfWords(rawWords, net.vocabulary);
  const totalInBag = bag.reduce((a, b) => a + b, 0);
  if (totalInBag === 0) {
    pending.set(userId, { type: 'echo', question: trimmedText });
    sendMessage(chatId, escapeHtml(trimmedText) + '\n\n<i>новые слова. чему учить?</i>', { parse_mode: 'HTML' });
    return;
  }

  const { output } = forward(bag, net);
  let maxIdx = -1;
  let maxVal = -1;
  for (let i = 0; i < output.length; i++) {
    if (output[i] > maxVal) {
      maxVal = output[i];
      maxIdx = i;
    }
  }

  if (maxIdx !== -1 && maxVal > 0.65) {
    sendMessage(chatId, net.intents[maxIdx].response);
  } else {
    pending.set(userId, { type: 'echo', question: trimmedText });
    const conf = maxVal > 0 ? ` (уверенность ${(maxVal * 100).toFixed(0)}%)` : '';
    sendMessage(chatId, escapeHtml(trimmedText) + `\n\n<i>не уверен${conf}. чему учить?</i>`, { parse_mode: 'HTML' });
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
