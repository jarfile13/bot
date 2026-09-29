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
        }
      } catch (e) {}
    });
  });
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatTrainStats(info, before, after) {
  const lines = [];
  if (info.isNewIntent) lines.push('новый интент');
  if (info.vocabChanged) lines.push('словарь пополнен');
  const errInfo = `ошибка ${before.toFixed(3)} → ${after.toFixed(3)}`;
  lines.push(errInfo);
  return lines.join(' · ');
}

function handleMessage(chatId, userId, text, document) {
  let net = loadUserNet(userId);

  if (document && document.file_name && document.file_name.endsWith('.json')) {
    downloadFile(document.file_id, (content) => {
      try {
        const imported = JSON.parse(content);
        if (imported.weights_ih && imported.intents) {
          saveUserNet(userId, imported);
          pending.delete(userId);
          sendMessage(chatId, 'сеть загружена');
        } else {
          sendMessage(chatId, 'неверный формат');
        }
      } catch (e) { sendMessage(chatId, 'ошибка чтения'); }
    });
    return;
  }

  if (text && text.startsWith('/')) {
    const trimmed = text.trim();
    const cmd = trimmed.split(/\s+/)[0].toLowerCase();

    if (cmd === '/start' || cmd === '/help') {
      pending.delete(userId);
      sendMessage(chatId,
        '<b>Нейросеть на JS</b>\n\n' +
        '<b>Обучение</b>\n' +
        'Просто напиши фразу — я повторю и буду ждать ответ.\n' +
        'Ответь — запомню пару.\n\n' +
        'Или сразу: <code>/teach вопрос = ответ</code>\n\n' +
        '<b>Команды</b>\n' +
        '/stats — статистика сети\n' +
        '/export — скачать веса\n' +
        '/reset — стереть сеть\n' +
        '/cancel — отменить обучение',
        { parse_mode: 'HTML' });
      return;
    }
    if (cmd === '/cancel') {
      pending.delete(userId);
      sendMessage(chatId, 'отменено');
      return;
    }
    if (cmd === '/reset') {
      const file = getUserNetFile(userId);
      if (fs.existsSync(file)) fs.unlinkSync(file);
      userNetworks.delete(userId);
      pending.delete(userId);
      sendMessage(chatId, 'сеть стёрта');
      return;
    }
    if (cmd === '/export') {
      if (net.vocabulary.length === 0) { sendMessage(chatId, 'сеть пуста'); return; }
      sendDocument(chatId, getUserNetFile(userId), 'веса сети');
      return;
    }
    if (cmd === '/stats') {
      const totalWeights = net.weights_ih.reduce((s, row) => s + row.length, 0) +
                          net.weights_ho.reduce((s, row) => s + row.length, 0);
      let out = '<b>Сеть</b>\n\n';
      out += `Слов в словаре: ${net.vocabulary.length}\n`;
      out += `Интентов: ${net.intents.length}\n`;
      out += `Нейронов скрытого слоя: ${net.bias_h.length}\n`;
      out += `Синапсов: ${totalWeights}\n`;
      sendMessage(chatId, out, { parse_mode: 'HTML' });
      return;
    }
    if (cmd === '/teach') {
      const m = trimmed.match(/^\/teach\s+(.+?)\s*=\s*(.+)$/s);
      if (!m) {
        sendMessage(chatId, 'формат: /teach вопрос = ответ');
        return;
      }
      const q = m[1].trim();
      const a = m[2].trim();
      const words = getWords(q);
      if (words.length === 0) { sendMessage(chatId, 'вопрос пустой'); return; }

      const beforeNet = JSON.parse(JSON.stringify(net));
      const beforeIntents = beforeNet.intents.length;
      const beforeVocab = beforeNet.vocabulary.length;

      const info = updateNetworkStructure(net, words, a);
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
      sendMessage(chatId, parts.join(' · '));
      return;
    }
    sendMessage(chatId, 'неизвестная команда, смотри /help');
    return;
  }

  if (!text) return;

  const rawWords = getWords(text);
  const trimmedText = text.trim();

  const p = pending.get(userId);
  if (p) {
    pending.delete(userId);
    if (rawWords.length === 0) { sendMessage(chatId, 'пусто'); return; }

    const beforeVocab = net.vocabulary.length;
    const beforeIntents = net.intents.length;

    const info = updateNetworkStructure(net, p.words, trimmedText);
    const refreshedNet = loadUserNet(userId);
    const bag = buildBagOfWords(p.words, refreshedNet.vocabulary);
    const { firstErr, lastErr } = trainNetwork(bag, info.intentIdx, refreshedNet, 0.6, 40);
    saveUserNet(userId, refreshedNet);

    const addedWords = refreshedNet.vocabulary.length - beforeVocab;
    const addedIntents = refreshedNet.intents.length - beforeIntents;
    const parts = [];
    if (addedIntents > 0) parts.push(`+${addedIntents} интент`);
    if (addedWords > 0) parts.push(`+${addedWords} слов`);
    parts.push(`ошибка ${firstErr.toFixed(2)}→${lastErr.toFixed(2)}`);
    sendMessage(chatId, 'запомнил: ' + parts.join(' · '));
    return;
  }

  if (net.vocabulary.length === 0 || net.intents.length === 0) {
    pending.set(userId, { words: rawWords });
    sendMessage(chatId, escapeHtml(trimmedText) + '\n\n<i>чему учить?</i>', { parse_mode: 'HTML' });
    return;
  }

  const bag = buildBagOfWords(rawWords, net.vocabulary);
  const totalInBag = bag.reduce((a, b) => a + b, 0);
  if (totalInBag === 0) {
    pending.set(userId, { words: rawWords });
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
    pending.set(userId, { words: rawWords });
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
