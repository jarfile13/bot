const https = require('https');
const http = require('http');
const fs = require('fs');

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

function sendMessage(chatId, text) {
    return apiRequest('sendMessage', { chat_id: chatId, text });
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

async function handleMessage(msg) {
    const chatId = msg.chat.id;
    const userId = msg.from.id;
    const text = msg.text.trim();
    if (!text) return;

    if (text.startsWith('/train')) {
        const correctOutput = text.replace(/^\/train\s*/, '').trim();
        const lastQuestion = aiBrain.lastInput[userId];

        if (!lastQuestion) {
            await sendMessage(chatId, "Сначала спроси меня о чем-нибудь, чтобы я запомнил контекст!");
        } else if (!correctOutput) {
            await sendMessage(chatId, "Использование: напиши /train ТЕКСТ_ОТВЕТА");
        } else {
            trainAI(lastQuestion, correctOutput);
            await sendMessage(chatId, `Успешно! Теперь на фразу "${lastQuestion}" я буду отвечать: "${correctOutput}"`);
        }
        return;
    }

    aiBrain.lastInput[userId] = text;
    saveBrain();
    const aiAnswer = thinkAI(text);
    await sendMessage(chatId, aiAnswer);
}

const startOffset = loadOffset();
console.log("ИИ-Бот запущен. Offset:", startOffset);
getUpdates(startOffset);
