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
const DATA_FILE = process.env.DATA_FILE || './ai_brain.json';
const OFFSET_FILE = './ai_brain.offset.json';

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT);

let aiBrain = { vocabulary: {}, relations: {}, lastInput: {} };

if (fs.existsSync(DATA_FILE)) {
    try {
        const parsed = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
        aiBrain = {
            vocabulary: parsed.vocabulary || {},
            relations: parsed.relations || {},
            lastInput: parsed.lastInput || {}
        };
    } catch (e) {
        console.error('[ОШИБКА] Не удалось прочитать', DATA_FILE, e.message);
    }
}

let saveTimer = null;
let saving = false;

function saveBrainNow() {
    if (saving) return;
    saving = true;
    const tmp = DATA_FILE + '.tmp';
    try {
        fs.writeFileSync(tmp, JSON.stringify(aiBrain, null, 2), 'utf8');
        fs.renameSync(tmp, DATA_FILE);
    } catch (e) {
        console.error('[ОШИБКА] saveBrain:', e.message);
    } finally {
        saving = false;
    }
}

function saveBrain() {
    if (saveTimer) return;
    saveTimer = setTimeout(() => {
        saveTimer = null;
        saveBrainNow();
    }, 2000);
}

process.on('SIGINT', () => { saveBrainNow(); process.exit(0); });
process.on('SIGTERM', () => { saveBrainNow(); process.exit(0); });

function loadOffset() {
    try {
        if (fs.existsSync(OFFSET_FILE)) {
            const o = JSON.parse(fs.readFileSync(OFFSET_FILE, 'utf8'));
            return Number(o.offset) || 0;
        }
    } catch {}
    return 0;
}

function saveOffset(offset) {
    try {
        fs.writeFileSync(OFFSET_FILE, JSON.stringify({ offset }), 'utf8');
    } catch {}
}

function cleanText(text) {
    return String(text)
        .toLowerCase()
        .replace(/ё/g, 'е')
        .replace(/[.,\/#!$%^&*;:{}=\-_`~()?\[\]"'«»…]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function trainAI(input, output) {
    const inClean = cleanText(input);
    const outClean = String(output).trim();
    if (!inClean || !outClean) return false;

    if (!aiBrain.relations[inClean]) {
        aiBrain.relations[inClean] = {};
    }

    if (!aiBrain.relations[inClean][outClean]) {
        aiBrain.relations[inClean][outClean] = 1;
    } else {
        aiBrain.relations[inClean][outClean] += 1;
    }

    for (const w of inClean.split(' ')) {
        if (!w) continue;
        aiBrain.vocabulary[w] = (aiBrain.vocabulary[w] || 0) + 1;
    }

    saveBrain();
    return true;
}

function pickBestAnswer(options) {
    let bestKey = null;
    let bestVal = -1;
    for (const k in options) {
        if (options[k] > bestVal) {
            bestVal = options[k];
            bestKey = k;
        }
    }
    return bestKey;
}

function thinkAI(userInput) {
    const cleanInput = cleanText(userInput);

    if (aiBrain.relations[cleanInput]) {
        const options = aiBrain.relations[cleanInput];
        const best = pickBestAnswer(options);
        if (best) return best;
    }

    let bestMatch = null;
    let maxScore = 0;
    const inputWords = cleanInput.split(' ').filter(Boolean);

    if (inputWords.length === 0) {
        return "Я пока не знаю, что ответить. Обучи меня! Напиши: /train [ответ]";
    }

    for (const knownInput in aiBrain.relations) {
        const knownWords = knownInput.split(' ');
        let intersection = 0;
        for (const word of inputWords) {
            if (knownWords.includes(word)) intersection++;
        }

        if (intersection > maxScore) {
            maxScore = intersection;
            const options = aiBrain.relations[knownInput];
            bestMatch = pickBestAnswer(options);
        }
    }

    if (maxScore === 0 || !bestMatch) {
        return "Я пока не знаю, что ответить. Обучи меня! Напиши: /train [ответ]";
    }

    return bestMatch;
}

function apiRequest(method, data) {
    return new Promise((resolve) => {
        const dataString = JSON.stringify(data);
        const options = {
            hostname: 'api.telegram.org',
            path: `/bot${BOT_TOKEN}/${method}`,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(dataString)
            },
            timeout: 40000
        };
        const req = https.request(options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try { resolve(JSON.parse(body)); }
                catch { resolve({ ok: false, error: 'bad_json' }); }
            });
        });
        req.on('error', (e) => resolve({ ok: false, error: e.message }));
        req.on('timeout', () => { req.destroy(); resolve({ ok: false, error: 'timeout' }); });
        req.write(dataString);
        req.end();
    });
}

function sendMessage(chatId, text, extra) {
    const payload = { chat_id: chatId, text: text };
    if (extra) Object.assign(payload, extra);
    return apiRequest('sendMessage', payload);
}

function sendDocument(chatId, filePath, caption) {
    return new Promise((resolve) => {
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
                headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}`, 'Content-Length': body.length },
                timeout: 60000
            }, (res) => {
                let d = '';
                res.on('data', c => d += c);
                res.on('end', () => resolve({ ok: true }));
            });
            req.on('error', () => resolve({ ok: false }));
            req.on('timeout', () => { req.destroy(); resolve({ ok: false }); });
            req.write(body);
            req.end();
        } catch (e) { resolve({ ok: false }); }
    });
}

function downloadFile(fileId) {
    return new Promise((resolve) => {
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
                            fr.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
                        }).on('error', () => resolve(null));
                    } else resolve(null);
                } catch { resolve(null); }
            });
        }).on('error', () => resolve(null));
    });
}

const pendingReset = new Map();

function showHelp(chatId) {
    const text =
        '<b>Команды</b>\n\n' +
        '<code>/train ответ</code> — обучить на последнем сообщении\n' +
        '<code>/list [стр]</code> — список выученных пар\n' +
        '<code>/show &lt;номер&gt;</code> — подробности пары\n' +
        '<code>/del &lt;номер&gt;</code> — удалить пару\n' +
        '<code>/forget &lt;фраза&gt;</code> — удалить по фразе\n' +
        '<code>/stats</code> — статистика\n' +
        '<code>/export</code> — скачать базу\n' +
        '<code>/import</code> — импорт базы\n' +
        '<code>/reset</code> — стереть всё\n' +
        '<code>/cancel</code> — отмена';
    sendMessage(chatId, text, { parse_mode: 'HTML' });
}

function showStats(chatId) {
    const relationCount = Object.keys(aiBrain.relations).length;
    const answerSet = new Set();
    for (const inKey in aiBrain.relations) {
        for (const outKey in aiBrain.relations[inKey]) answerSet.add(outKey);
    }
    const vocabCount = Object.keys(aiBrain.vocabulary).length;
    let fileSize = 0;
    try { fileSize = fs.statSync(DATA_FILE).size; } catch {}
    const sizeKb = (fileSize / 1024).toFixed(1);

    let out = '<b>Статистика</b>\n\n';
    out += `Фраз: <b>${relationCount}</b>\n`;
    out += `Ответов: <b>${answerSet.size}</b>\n`;
    out += `Слов: <b>${vocabCount}</b>\n`;
    out += `Размер: <b>${sizeKb} КБ</b>`;
    sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function showList(chatId, page) {
    const keys = Object.keys(aiBrain.relations);
    if (keys.length === 0) { sendMessage(chatId, 'база пуста'); return; }
    const PAGE = 10;
    const totalPages = Math.ceil(keys.length / PAGE);
    const p = Math.max(1, Math.min(page, totalPages));
    const start = (p - 1) * PAGE;
    const slice = keys.slice(start, start + PAGE);

    let out = `<b>Пары</b> (${p}/${totalPages}, всего ${keys.length})\n\n`;
    slice.forEach((key, i) => {
        const num = start + i + 1;
        const best = pickBestAnswer(aiBrain.relations[key]);
        const q = key.length > 45 ? key.slice(0, 45) + '…' : key;
        const a = best.length > 45 ? best.slice(0, 45) + '…' : best;
        out += `<b>${num}.</b> ${escapeHtml(q)}\n     ${escapeHtml(a)}\n`;
    });
    if (totalPages > 1) out += `\n/list ${p + 1}`;
    sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function showEntry(chatId, num) {
    const keys = Object.keys(aiBrain.relations);
    const idx = num - 1;
    if (idx < 0 || idx >= keys.length) {
        sendMessage(chatId, `нет пары #${num}. всего: ${keys.length}`);
        return;
    }
    const key = keys[idx];
    const options = aiBrain.relations[key];
    let out = `<b>Пара #${num}</b>\n\n`;
    out += `<b>Вопрос:</b> ${escapeHtml(key)}\n\n`;
    out += `<b>Ответы:</b>\n`;
    for (const ans in options) {
        out += `  ${options[ans]}× ${escapeHtml(ans)}\n`;
    }
    sendMessage(chatId, out, { parse_mode: 'HTML' });
}

function deleteByIndex(chatId, num) {
    const keys = Object.keys(aiBrain.relations);
    const idx = num - 1;
    if (idx < 0 || idx >= keys.length) {
        sendMessage(chatId, `укажи номер 1–${keys.length}`);
        return;
    }
    const key = keys[idx];
    delete aiBrain.relations[key];
    saveBrain();
    sendMessage(chatId, `удалено #${num}`);
}

function deleteByText(chatId, phrase) {
    const key = cleanText(phrase);
    if (!key) { sendMessage(chatId, 'формат: /forget фраза'); return; }
    if (!aiBrain.relations[key]) {
        sendMessage(chatId, 'не найдено');
        return;
    }
    delete aiBrain.relations[key];
    saveBrain();
    sendMessage(chatId, 'удалено');
}

async function handleMessage(msg) {
    const chatId = msg.chat.id;
    const userId = msg.from.id;

    if (msg.document && msg.document.file_name && msg.document.file_name.endsWith('.json')) {
        const content = await downloadFile(msg.document.file_id);
        if (!content) { sendMessage(chatId, 'не удалось скачать'); return; }
        try {
            const imported = JSON.parse(content);
            if (imported && typeof imported === 'object' && imported.relations) {
                aiBrain.relations = imported.relations || {};
                aiBrain.vocabulary = imported.vocabulary || {};
                aiBrain.lastInput = {};
                saveBrainNow();
                sendMessage(chatId, `импортировано: ${Object.keys(aiBrain.relations).length}`);
            } else {
                sendMessage(chatId, 'неверный формат');
            }
        } catch { sendMessage(chatId, 'ошибка чтения'); }
        return;
    }

    const text = (msg.text || '').trim();
    if (!text) return;

    if (text.startsWith('/')) {
        const parts = text.split(/\s+/);
        const cmd = parts[0].toLowerCase();
        const arg = parts.slice(1).join(' ').trim();

        if (cmd === '/start' || cmd === '/help') {
            pendingReset.delete(userId);
            showHelp(chatId);
            return;
        }

        if (cmd === '/cancel') {
            pendingReset.delete(userId);
            sendMessage(chatId, 'отменено');
            return;
        }

        if (cmd === '/stats') { showStats(chatId); return; }

        if (cmd === '/list') {
            showList(chatId, parseInt(arg, 10) || 1);
            return;
        }

        if (cmd === '/show') {
            const num = parseInt(arg, 10);
            if (!num) { sendMessage(chatId, 'формат: /show <номер>'); return; }
            showEntry(chatId, num);
            return;
        }

        if (cmd === '/del') {
            const num = parseInt(arg, 10);
            if (!num) { sendMessage(chatId, 'формат: /del <номер>'); return; }
            deleteByIndex(chatId, num);
            return;
        }

        if (cmd === '/forget') {
            if (!arg) { sendMessage(chatId, 'формат: /forget фраза'); return; }
            deleteByText(chatId, arg);
            return;
        }

        if (cmd === '/export') {
            const count = Object.keys(aiBrain.relations).length;
            if (count === 0) { sendMessage(chatId, 'база пуста'); return; }
            saveBrainNow();
            await sendDocument(chatId, DATA_FILE, `база (${count})`);
            return;
        }

        if (cmd === '/import') {
            sendMessage(chatId, 'отправь JSON-файл из /export');
            return;
        }

        if (cmd === '/reset') {
            pendingReset.set(userId, true);
            sendMessage(chatId, 'напиши да для подтверждения');
            return;
        }

        if (cmd === '/train') {
            const correctOutput = text.replace(/^\/train\s*/, '').trim();
            const lastQuestion = aiBrain.lastInput[userId];

            if (!lastQuestion) {
                sendMessage(chatId, "сначала задай вопрос");
            } else if (!correctOutput) {
                sendMessage(chatId, "формат: /train ответ");
            } else {
                trainAI(lastQuestion, correctOutput);
                sendMessage(chatId, `запомнил: "${lastQuestion}" -> "${correctOutput}"`);
            }
            return;
        }

        sendMessage(chatId, 'неизвестная команда, /start');
        return;
    }

    if (pendingReset.get(userId)) {
        const a = text.toLowerCase();
        if (a === 'да' || a === 'yes' || a === 'y') {
            aiBrain = { vocabulary: {}, relations: {}, lastInput: {} };
            saveBrainNow();
            pendingReset.delete(userId);
            sendMessage(chatId, 'стёрто');
        } else {
            pendingReset.delete(userId);
            sendMessage(chatId, 'отменено');
        }
        return;
    }

    aiBrain.lastInput[userId] = text;
    saveBrain();
    const aiAnswer = thinkAI(text);
    sendMessage(chatId, aiAnswer);
}

let polling = false;

async function getUpdates(offset) {
    if (polling) return;
    polling = true;

    const res = await apiRequest('getUpdates', {
        offset,
        timeout: 30,
        allowed_updates: ['message']
    });

    polling = false;

    if (res && res.ok && Array.isArray(res.result) && res.result.length > 0) {
        for (const update of res.result) {
            offset = update.update_id + 1;
            saveOffset(offset);
            if (update.message) {
                try {
                    await handleMessage(update.message);
                } catch (e) {
                    console.error('[ОШИБКА] handleMessage:', e.message);
                }
            }
        }
        setTimeout(() => getUpdates(offset), 50);
    } else {
        const delay = (res && res.ok) ? 50 : 3000;
        setTimeout(() => getUpdates(offset), delay);
    }
}

const startOffset = loadOffset();
console.log("ИИ-Бот запущен. Offset:", startOffset);
getUpdates(startOffset);
