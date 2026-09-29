const https = require('https');
const http = require('http');
const fs = require('fs');

const BOT_TOKEN = process.env.BOT_TOKEN;
if (!BOT_TOKEN) {
    console.error('BOT_TOKEN is not set');
    process.exit(1);
}

const PORT = process.env.PORT || 3000;
const DATA_FILE = process.env.DATA_FILE || './ai_brain.json';
const OFFSET_FILE = process.env.OFFSET_FILE || './ai_brain.offset.json';

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT);

let aiBrain = {
    vocabulary: {},
    relations: {},
    originals: {},
    lastInput: {},
    synonyms: {},
    context: {}
};

if (fs.existsSync(DATA_FILE)) {
    try {
        const parsed = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
        aiBrain = {
            vocabulary: parsed.vocabulary || {},
            relations: parsed.relations || {},
            originals: parsed.originals || {},
            lastInput: parsed.lastInput || {},
            synonyms: parsed.synonyms || {},
            context: parsed.context || {}
        };
    } catch (e) {
        console.error('Failed to read', DATA_FILE, e.message);
    }
}

let saveTimer = null;
let saving = false;

function saveBrainNow() {
    if (saving) return;
    saving = true;
    const tmp = DATA_FILE + '.tmp';
    try {
        const toSave = {
            vocabulary: aiBrain.vocabulary,
            relations: aiBrain.relations,
            originals: aiBrain.originals,
            lastInput: aiBrain.lastInput,
            synonyms: aiBrain.synonyms,
            context: aiBrain.context
        };
        fs.writeFileSync(tmp, JSON.stringify(toSave, null, 2), 'utf8');
        fs.renameSync(tmp, DATA_FILE);
    } catch (e) {
        console.error('saveBrain error:', e.message);
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

const ENDINGS = [
    'иями', 'ями', 'ами', 'ией', 'иям', 'ием', 'иях', 'ию', 'ия', 'ий', 'ый', 'ой', 'ая', 'ое', 'ые', 'ими', 'ыми',
    'ов', 'ев', 'ам', 'ям', 'ах', 'ях', 'ом', 'ем', 'ой', 'ей', 'ую', 'юю', 'ишь', 'ешь', 'ете', 'ите', 'ует', 'уют',
    'ал', 'ял', 'ил', 'ел', 'ла', 'ло', 'ли', 'ть', 'ся', 'сь', 'а', 'я', 'о', 'е', 'у', 'ю', 'ы', 'и', 'й', 'ь'
];

function stem(word) {
    let w = word;
    if (w.length <= 3) return w;
    for (const end of ENDINGS) {
        if (w.length - end.length >= 3 && w.endsWith(end)) {
            w = w.slice(0, w.length - end.length);
            break;
        }
    }
    return w;
}

function tokenize(text) {
    const clean = cleanText(text);
    if (!clean) return [];
    const words = clean.split(' ').filter(Boolean);
    const out = [];
    for (const w of words) {
        const syn = aiBrain.synonyms[w];
        const base = (syn && syn.length) ? syn[0] : w;
        out.push(stem(base));
    }
    return out;
}

function trainAI(input, output) {
    const inTokens = tokenize(input);
    const outClean = String(output).trim();
    if (!inTokens.length || !outClean) return false;

    const key = inTokens.join(' ');
    const original = cleanText(input);

    if (!aiBrain.relations[key]) {
        aiBrain.relations[key] = {};
    }

    if (!aiBrain.relations[key][outClean]) {
        aiBrain.relations[key][outClean] = 1;
    } else {
        aiBrain.relations[key][outClean] += 1;
    }

    if (!aiBrain.originals[key]) {
        aiBrain.originals[key] = original;
    }

    for (const w of inTokens) {
        aiBrain.vocabulary[w] = (aiBrain.vocabulary[w] || 0) + 1;
    }

    saveBrain();
    return true;
}

function wordWeight(word) {
    const total = Object.values(aiBrain.vocabulary).reduce((a, b) => a + b, 0) || 1;
    const freq = aiBrain.vocabulary[word] || 0;
    return Math.log((total + 1) / (freq + 1)) + 1;
}

function pickWeighted(options) {
    const keys = Object.keys(options);
    if (!keys.length) return null;
    let total = 0;
    for (const k of keys) total += options[k];
    let r = Math.random() * total;
    for (const k of keys) {
        r -= options[k];
        if (r <= 0) return k;
    }
    return keys[keys.length - 1];
}

function scoreMatch(inputTokens, knownKey) {
    const knownTokens = knownKey.split(' ');
    let score = 0;
    for (const t of inputTokens) {
        if (knownTokens.includes(t)) {
            score += wordWeight(t);
        }
    }
    return score;
}

function thinkAI(userInput, userId) {
    const inputTokens = tokenize(userInput);
    if (!inputTokens.length) {
        return 'Я пока не знаю, что ответить. Обучи меня: /train вопрос = ответ';
    }

    const key = inputTokens.join(' ');
    if (aiBrain.relations[key]) {
        const best = pickWeighted(aiBrain.relations[key]);
        if (best) return best;
    }

    let bestKey = null;
    let bestScore = 0;
    for (const knownKey in aiBrain.relations) {
        const s = scoreMatch(inputTokens, knownKey);
        if (s > bestScore) {
            bestScore = s;
            bestKey = knownKey;
        }
    }

    if (bestKey && bestScore > 0) {
        const best = pickWeighted(aiBrain.relations[bestKey]);
        if (best) return best;
    }

    if (userId) {
        const ctx = aiBrain.context[userId] || [];
        for (let i = ctx.length - 1; i >= 0; i--) {
            const ctxTokens = tokenize(ctx[i]);
            for (const knownKey in aiBrain.relations) {
                const s = scoreMatch(ctxTokens, knownKey);
                if (s > bestScore) {
                    bestScore = s;
                    bestKey = knownKey;
                }
            }
        }
        if (bestKey && bestScore > 0) {
            const best = pickWeighted(aiBrain.relations[bestKey]);
            if (best) return best;
        }
    }

    return null;
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

function sendMessage(chatId, text) {
    return apiRequest('sendMessage', { chat_id: chatId, text });
}

const PAGE_SIZE = 10;
const MAX_LEN = 3500;

function chunkMessage(text, limit) {
    if (text.length <= limit) return [text];
    const parts = [];
    let current = '';
    const lines = text.split('\n');
    for (const line of lines) {
        if ((current + line + '\n').length > limit) {
            parts.push(current);
            current = '';
        }
        current += line + '\n';
    }
    if (current) parts.push(current);
    return parts;
}

function buildListPage(page) {
    const items = [];
    for (const q in aiBrain.relations) {
        items.push({ q, opts: aiBrain.relations[q] });
    }
    items.sort((a, b) => a.q.localeCompare(b.q));

    const totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
    if (page < 0) page = 0;
    if (page >= totalPages) page = totalPages - 1;

    const slice = items.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

    let out = 'Стр. ' + (page + 1) + '/' + totalPages + ' | всего пар: ' + items.length + '\n\n';

    slice.forEach((it, i) => {
        const num = page * PAGE_SIZE + i + 1;
        const original = aiBrain.originals[it.q] || it.q;
        out += num + '. ' + original + '\n';
        const opts = Object.keys(it.opts);
        for (const a of opts) {
            out += '   -> ' + a + ' (' + it.opts[a] + ')\n';
        }
        const syns = aiBrain.synonyms[it.q];
        if (syns && syns.length) {
            out += '   ~ ' + syns.join(', ') + '\n';
        }
        out += '\n';
    });

    return out;
}

function buildSynListPage(page) {
    const seen = new Set();
    const groups = [];
    for (const word in aiBrain.synonyms) {
        if (seen.has(word)) continue;
        const list = aiBrain.synonyms[word];
        if (!list || !list.length) continue;
        const group = [word, ...list];
        for (const g of group) seen.add(g);
        groups.push(group);
    }
    groups.sort((a, b) => a[0].localeCompare(b[0]));

    const totalPages = Math.max(1, Math.ceil(groups.length / PAGE_SIZE));
    if (page < 0) page = 0;
    if (page >= totalPages) page = totalPages - 1;

    const slice = groups.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

    let out = 'Синонимы | стр. ' + (page + 1) + '/' + totalPages + ' | всего групп: ' + groups.length + '\n\n';

    slice.forEach((group, i) => {
        const num = page * PAGE_SIZE + i + 1;
        out += num + '. ' + group[0] + '\n';
        if (group.length > 1) {
            out += '   ~ ' + group.slice(1).join(', ') + '\n';
        }
        out += '\n';
    });

    return out;
}

function getStats() {
    const pairs = Object.keys(aiBrain.relations).length;
    let answers = 0;
    let examples = 0;
    for (const q in aiBrain.relations) {
        const opts = aiBrain.relations[q];
        const keys = Object.keys(opts);
        answers += keys.length;
        for (const a of keys) examples += opts[a];
    }
    const words = Object.keys(aiBrain.vocabulary).length;
    const syns = Object.keys(aiBrain.synonyms).length;
    return { pairs, answers, examples, words, syns };
}

function deletePair(question) {
    const tokens = tokenize(question);
    const key = tokens.join(' ');
    if (!key) return false;
    if (!aiBrain.relations[key]) return false;
    delete aiBrain.relations[key];
    delete aiBrain.originals[key];
    saveBrainNow();
    return true;
}

function deleteAnswer(question, answer) {
    const tokens = tokenize(question);
    const key = tokens.join(' ');
    if (!key || !aiBrain.relations[key]) return false;
    const outClean = String(answer).trim();
    if (!aiBrain.relations[key][outClean]) return false;
    delete aiBrain.relations[key][outClean];
    if (!Object.keys(aiBrain.relations[key]).length) {
        delete aiBrain.relations[key];
        delete aiBrain.originals[key];
    }
    saveBrainNow();
    return true;
}

function addSynonyms(word, list) {
    const w = cleanText(word);
    if (!w) return false;
    const syns = list.map(s => cleanText(s)).filter(Boolean);
    if (!syns.length) return false;

    if (!aiBrain.synonyms[w]) aiBrain.synonyms[w] = [];
    for (const s of syns) {
        if (!aiBrain.synonyms[w].includes(s)) {
            aiBrain.synonyms[w].push(s);
        }
        if (!aiBrain.synonyms[s]) aiBrain.synonyms[s] = [w];
        else if (!aiBrain.synonyms[s].includes(w)) {
            aiBrain.synonyms[s].push(w);
        }
    }
    saveBrain();
    return true;
}

function deleteAllSynonyms(word) {
    const w = cleanText(word);
    if (!w || !aiBrain.synonyms[w]) return false;
    const list = [...aiBrain.synonyms[w]];
    delete aiBrain.synonyms[w];
    for (const s of list) {
        if (aiBrain.synonyms[s]) {
            aiBrain.synonyms[s] = aiBrain.synonyms[s].filter(x => x !== w);
            if (!aiBrain.synonyms[s].length) delete aiBrain.synonyms[s];
        }
    }
    saveBrainNow();
    return true;
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
            if (update.message && typeof update.message.text === 'string') {
                try {
                    await handleMessage(update.message);
                } catch (e) {
                    console.error('handleMessage error:', e.message);
                }
            }
        }
        setTimeout(() => getUpdates(offset), 50);
    } else {
        const delay = (res && res.ok) ? 50 : 3000;
        setTimeout(() => getUpdates(offset), delay);
    }
}

async function handleMessage(msg) {
    const chatId = msg.chat.id;
    const userId = msg.from.id;
    const text = msg.text.trim();
    if (!text) return;

    if (text === '/start') {
        await sendMessage(chatId,
            'Как учить:\n' +
            '/train вопрос = ответ\n' +
            '/train привет = Привет!\n\n' +
            'Синонимы:\n' +
            '/syn слово = синоним1, синоним2\n' +
            '/syn del слово\n' +
            '/synlist - список синонимов\n\n' +
            'Смотреть:\n' +
            '/list - стр. 1\n' +
            '/list 2 - стр. 2\n\n' +
            'Удалять:\n' +
            '/del вопрос - всю пару\n' +
            '/del вопрос = ответ - один ответ\n\n' +
            'Ещё:\n' +
            '/stats - статистика\n' +
            '/reset - стереть всё\n' +
            '/export - скачать базу\n' +
            '/import - загрузить базу'
        );
        return;
    }

    if (text === '/reset') {
        aiBrain = { vocabulary: {}, relations: {}, originals: {}, lastInput: {}, synonyms: {}, context: {} };
        saveBrainNow();
        await sendMessage(chatId, 'Память очищена.');
        return;
    }

    if (text === '/stats') {
        const s = getStats();
        await sendMessage(chatId,
            'Вопросов: ' + s.pairs + '\n' +
            'Ответов: ' + s.answers + '\n' +
            'Примеров: ' + s.examples + '\n' +
            'Слов: ' + s.words + '\n' +
            'Синонимов: ' + s.syns
        );
        return;
    }

    if (text === '/list' || text.startsWith('/list ')) {
        if (!Object.keys(aiBrain.relations).length) {
            await sendMessage(chatId, 'База пуста.');
            return;
        }
        const arg = text.slice(5).trim();
        let page = 0;
        if (arg) {
            const n = parseInt(arg, 10);
            if (isNaN(n) || n < 1) {
                await sendMessage(chatId, 'Использование: /list или /list n');
                return;
            }
            page = n - 1;
        }
        const totalPages = Math.max(1, Math.ceil(Object.keys(aiBrain.relations).length / PAGE_SIZE));
        if (page >= totalPages) {
            await sendMessage(chatId, 'Всего страниц: ' + totalPages);
            return;
        }
        const out = buildListPage(page);
        const parts = chunkMessage(out, MAX_LEN);
        for (const part of parts) {
            await sendMessage(chatId, part);
        }
        return;
    }

    if (text === '/synlist' || text.startsWith('/synlist ')) {
        const hasSyns = Object.keys(aiBrain.synonyms).length > 0;
        if (!hasSyns) {
            await sendMessage(chatId, 'Синонимов нет.');
            return;
        }
        const arg = text.slice(8).trim();
        let page = 0;
        if (arg) {
            const n = parseInt(arg, 10);
            if (isNaN(n) || n < 1) {
                await sendMessage(chatId, 'Использование: /synlist или /synlist n');
                return;
            }
            page = n - 1;
        }
        const out = buildSynListPage(page);
        const parts = chunkMessage(out, MAX_LEN);
        for (const part of parts) {
            await sendMessage(chatId, part);
        }
        return;
    }

    if (text.startsWith('/del ')) {
        const arg = text.slice(5).trim();
        if (!arg) {
            await sendMessage(chatId, 'Использование:\n/del вопрос\n/del вопрос = ответ');
            return;
        }
        if (arg.includes('=')) {
            const idx = arg.indexOf('=');
            const q = arg.slice(0, idx).trim();
            const a = arg.slice(idx + 1).trim();
            const ok = deleteAnswer(q, a);
            await sendMessage(chatId, ok ? 'Ответ удалён.' : 'Не найдено.');
            return;
        }
        const ok = deletePair(arg);
        await sendMessage(chatId, ok ? 'Пара удалена.' : 'Не найдено.');
        return;
    }

    if (text.startsWith('/syn')) {
        const rest = text.replace(/^\/syn\s*/, '').trim();

        if (rest.startsWith('del ')) {
            const arg = rest.slice(4).trim();
            if (!arg) {
                await sendMessage(chatId, 'Использование: /syn del слово');
                return;
            }
            const ok = deleteAllSynonyms(arg);
            await sendMessage(chatId, ok ? 'Все синонимы слова удалены.' : 'Не найдено.');
            return;
        }

        if (!rest.includes('=')) {
            await sendMessage(chatId, 'Использование: /syn слово = синоним1, синоним2');
            return;
        }
        const idx = rest.indexOf('=');
        const word = rest.slice(0, idx).trim();
        const listRaw = rest.slice(idx + 1).trim();
        const list = listRaw.split(',').map(s => s.trim()).filter(Boolean);
        if (!word || !list.length) {
            await sendMessage(chatId, 'Использование: /syn слово = синоним1, синоним2');
            return;
        }
        addSynonyms(word, list);
        await sendMessage(chatId, 'Синонимы добавлены: ' + word + ' = ' + list.join(', '));
        return;
    }

    if (text === '/export') {
        try {
            const buf = Buffer.from(JSON.stringify({
                vocabulary: aiBrain.vocabulary,
                relations: aiBrain.relations,
                originals: aiBrain.originals,
                synonyms: aiBrain.synonyms,
                context: aiBrain.context,
                lastInput: aiBrain.lastInput
            }, null, 2), 'utf8');
            const boundary = '----AIBrainBoundary' + Date.now();
            const head = '--' + boundary + '\r\n' +
                'Content-Disposition: form-data; name="document"; filename="ai_brain.json"\r\n' +
                'Content-Type: application/json\r\n\r\n';
            const tail = '\r\n--' + boundary + '--\r\n';
            const body = Buffer.concat([Buffer.from(head, 'utf8'), buf, Buffer.from(tail, 'utf8')]);

            await new Promise((resolve) => {
                const options = {
                    hostname: 'api.telegram.org',
                    path: `/bot${BOT_TOKEN}/sendDocument`,
                    method: 'POST',
                    headers: {
                        'Content-Type': 'multipart/form-data; boundary=' + boundary,
                        'Content-Length': body.length
                    },
                    timeout: 40000
                };
                const req = https.request(options, (res) => {
                    let b = '';
                    res.on('data', c => b += c);
                    res.on('end', () => resolve());
                });
                req.on('error', () => resolve());
                req.on('timeout', () => { req.destroy(); resolve(); });
                req.write(body);
                req.end();
            });
        } catch (e) {
            await sendMessage(chatId, 'Ошибка экспорта.');
        }
        return;
    }

    if (text === '/import') {
        await sendMessage(chatId, 'Отправь JSON-файл с базой.');
        return;
    }

    if (msg.document) {
        try {
            const fileId = msg.document.file_id;
            const fileRes = await apiRequest('getFile', { file_id: fileId });
            if (!fileRes.ok) {
                await sendMessage(chatId, 'Не удалось получить файл.');
                return;
            }
            const filePath = fileRes.result.file_path;
            const fileData = await new Promise((resolve) => {
                https.get(`https://api.telegram.org/file/bot${BOT_TOKEN}/${filePath}`, (res) => {
                    let chunks = [];
                    res.on('data', c => chunks.push(c));
                    res.on('end', () => resolve(Buffer.concat(chunks)));
                }).on('error', () => resolve(null));
            });
            if (!fileData) {
                await sendMessage(chatId, 'Не удалось скачать файл.');
                return;
            }
            const parsed = JSON.parse(fileData.toString('utf8'));
            aiBrain = {
                vocabulary: parsed.vocabulary || {},
                relations: parsed.relations || {},
                originals: parsed.originals || {},
                lastInput: parsed.lastInput || {},
                synonyms: parsed.synonyms || {},
                context: parsed.context || {}
            };
            saveBrainNow();
            await sendMessage(chatId, 'База загружена.');
        } catch (e) {
            await sendMessage(chatId, 'Ошибка импорта.');
        }
        return;
    }

    if (text.startsWith('/train')) {
        const rest = text.replace(/^\/train\s*/, '').trim();

        if (rest.includes('=')) {
            const idx = rest.indexOf('=');
            const question = rest.slice(0, idx).trim();
            const answer = rest.slice(idx + 1).trim();
            if (!question || !answer) {
                await sendMessage(chatId, 'Использование: /train вопрос = ответ');
                return;
            }
            trainAI(question, answer);
            await sendMessage(chatId, 'Выучено: ' + question + ' -> ' + answer);
            return;
        }

        const correctOutput = rest;
        const lastQuestion = aiBrain.lastInput[userId];

        if (!lastQuestion) {
            await sendMessage(chatId, 'Сначала спроси меня о чем-нибудь.');
        } else if (!correctOutput) {
            await sendMessage(chatId, 'Использование: /train вопрос = ответ');
        } else {
            trainAI(lastQuestion, correctOutput);
            await sendMessage(chatId, 'Выучено: ' + lastQuestion + ' -> ' + correctOutput);
        }
        return;
    }

    aiBrain.lastInput[userId] = text;

    if (!aiBrain.context[userId]) aiBrain.context[userId] = [];
    aiBrain.context[userId].push(text);
    if (aiBrain.context[userId].length > 5) aiBrain.context[userId].shift();

    saveBrain();

    const aiAnswer = thinkAI(text, userId);

    if (aiAnswer) {
        await sendMessage(chatId, aiAnswer);
    } else {
        await sendMessage(chatId, 'Я пока не знаю, что ответить. Обучи меня: /train вопрос = ответ');
    }
}

const startOffset = loadOffset();
console.log('Bot started. Offset:', startOffset);
getUpdates(startOffset);
