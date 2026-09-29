const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BOT_TOKEN = process.env.BOT_TOKEN;
if (!BOT_TOKEN) { console.error('BOT_TOKEN is not set'); process.exit(1); }

const PORT = process.env.PORT || 3000;
const DATA_DIR = process.env.DATA_DIR || './data';
const VEC_FILE = path.join(DATA_DIR, 'vectors.bin');
const VEC_META = path.join(DATA_DIR, 'vectors.json');
const SENTS_FILE = path.join(DATA_DIR, 'sentences.json');
const OFFSET_FILE = path.join(DATA_DIR, 'offset.json');
const CORPUS_FILE = path.join(DATA_DIR, 'corpus.txt');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT);

let vectors = { dim: 50, words: {}, trained: false, tokens: 0 };
let sentences = [];
let sentVecs = [];

if (fs.existsSync(VEC_META) && fs.existsSync(VEC_FILE)) {
    try {
        const meta = JSON.parse(fs.readFileSync(VEC_META, 'utf8'));
        vectors.dim = meta.dim;
        vectors.trained = meta.trained;
        vectors.tokens = meta.tokens;
        const buf = fs.readFileSync(VEC_FILE);
        const dim = meta.dim;
        const words = meta.words;
        const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
        let off = 0;
        for (const w of words) {
            vectors.words[w] = arr.subarray(off, off + dim);
            off += dim;
        }
    } catch (e) { console.error('vec read error:', e.message); }
}

if (fs.existsSync(SENTS_FILE)) {
    try {
        const data = JSON.parse(fs.readFileSync(SENTS_FILE, 'utf8'));
        sentences = data.sentences || [];
        sentVecs = (data.sentVecs || []).map(a => Float32Array.from(a));
    } catch (e) { console.error('sent read error:', e.message); }
}

function saveVectorsNow() {
    try {
        const words = Object.keys(vectors.words);
        const dim = vectors.dim;
        const arr = new Float32Array(words.length * dim);
        let off = 0;
        for (const w of words) {
            const v = vectors.words[w];
            for (let i = 0; i < dim; i++) arr[off + i] = v[i];
            off += dim;
        }
        fs.writeFileSync(VEC_FILE, Buffer.from(arr.buffer));
        fs.writeFileSync(VEC_META, JSON.stringify({ dim, words, trained: vectors.trained, tokens: vectors.tokens }), 'utf8');
    } catch (e) { console.error('saveVectors error:', e.message); }
}

function saveSentencesNow() {
    try {
        const data = { sentences, sentVecs: sentVecs.map(v => Array.from(v)) };
        fs.writeFileSync(SENTS_FILE, JSON.stringify(data), 'utf8');
    } catch (e) { console.error('saveSentences error:', e.message); }
}

process.on('SIGINT', () => { saveVectorsNow(); saveSentencesNow(); process.exit(0); });
process.on('SIGTERM', () => { saveVectorsNow(); saveSentencesNow(); process.exit(0); });

function loadOffset() {
    try {
        if (fs.existsSync(OFFSET_FILE)) return Number(JSON.parse(fs.readFileSync(OFFSET_FILE, 'utf8')).offset) || 0;
    } catch {}
    return 0;
}

function saveOffset(o) {
    try { fs.writeFileSync(OFFSET_FILE, JSON.stringify({ offset: o }), 'utf8'); } catch {}
}

function clean(text) {
    return String(text).toLowerCase().replace(/ё/g, 'е').replace(/[^а-яa-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
}

function rawTokens(text) {
    const c = clean(text);
    if (!c) return [];
    return c.split(' ').filter(Boolean);
}

function randVec(dim) {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = (Math.random() - 0.5) / dim;
    return v;
}

function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function norm(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * a[i];
    return Math.sqrt(s) || 1;
}

function cosine(a, b) {
    return dot(a, b) / (norm(a) * norm(b));
}

function sigmoid(x) {
    if (x > 6) return 1;
    if (x < -6) return 0;
    return 1 / (1 + Math.exp(-x));
}

function buildCorpus() {
    return fs.existsSync(CORPUS_FILE) ? fs.readFileSync(CORPUS_FILE, 'utf8') : '';
}

function appendCorpus(t) {
    fs.appendFileSync(CORPUS_FILE, '\n' + t, 'utf8');
}

function trainWord2Vec(opts) {
    const dim = opts.dim || 50;
    const win = opts.window || 3;
    const neg = opts.negative || 3;
    const lr0 = opts.lr || 0.025;
    const epochs = opts.epochs || 3;
    const minCount = opts.minCount || 2;

    const raw = buildCorpus();
    if (!raw) return { ok: false, error: 'корпус пуст' };

    const rawSents = raw.split(/[\n.!?]+/).map(s => rawTokens(s)).filter(s => s.length > 1);
    const freq = {};
    for (const s of rawSents) for (const w of s) freq[w] = (freq[w] || 0) + 1;

    const vocab = Object.keys(freq).filter(w => freq[w] >= minCount);
    const vocabSet = new Set(vocab);
    const total = vocab.reduce((a, w) => a + freq[w], 0);

    const keepProb = {};
    for (const w of vocab) {
        const f = freq[w] / total;
        keepProb[w] = Math.min(1, (Math.sqrt(f / 0.001) + 1) * 0.001 / f);
    }

    const negTable = [];
    for (const w of vocab) {
        const p = Math.pow(freq[w] / total, 0.75);
        const count = Math.max(1, Math.round(p * 50000));
        for (let i = 0; i < count; i++) negTable.push(w);
    }

    vectors.dim = dim;
    vectors.words = {};
    for (const w of vocab) vectors.words[w] = randVec(dim);

    let tokensCount = 0;
    for (const s of rawSents) for (const w of s) if (vocabSet.has(w)) tokensCount++;

    for (let epoch = 0; epoch < epochs; epoch++) {
        const lr = lr0 * (1 - epoch / epochs);
        for (const s of rawSents) {
            const filtered = [];
            for (const w of s) {
                if (vocabSet.has(w) && Math.random() < keepProb[w]) filtered.push(w);
            }
            for (let i = 0; i < filtered.length; i++) {
                const c = filtered[i];
                const cv = vectors.words[c];
                const ws = Math.max(0, i - win);
                const we = Math.min(filtered.length - 1, i + win);
                for (let j = ws; j <= we; j++) {
                    if (j === i) continue;
                    const u = filtered[j];
                    const uv = vectors.words[u];
                    let sc = 0;
                    for (let d = 0; d < dim; d++) sc += cv[d] * uv[d];
                    const g = (1 - sigmoid(sc)) * lr;
                    for (let d = 0; d < dim; d++) {
                        const grad = g * uv[d];
                        uv[d] += g * cv[d];
                        cv[d] += grad;
                    }
                    for (let k = 0; k < neg; k++) {
                        const nw = negTable[(Math.random() * negTable.length) | 0];
                        if (nw === u) continue;
                        const nv = vectors.words[nw];
                        let ns = 0;
                        for (let d = 0; d < dim; d++) ns += cv[d] * nv[d];
                        const ng = (0 - sigmoid(ns)) * lr;
                        for (let d = 0; d < dim; d++) {
                            const grad = ng * nv[d];
                            nv[d] += ng * cv[d];
                            cv[d] += grad;
                        }
                    }
                }
            }
        }
    }

    vectors.trained = true;
    vectors.tokens = tokensCount;
    saveVectorsNow();

    sentences = rawSents.map(s => s.join(' '));
    sentVecs = [];
    for (const s of rawSents) sentVecs.push(sentVecFromTokens(s));
    saveSentencesNow();

    return { ok: true, vocab: vocab.length, tokens: tokensCount, sentences: sentences.length };
}

function sentVecFromTokens(tokenList) {
    const dim = vectors.dim;
    const v = new Float32Array(dim);
    let count = 0;
    for (const w of tokenList) {
        const wv = vectors.words[w];
        if (!wv) continue;
        for (let i = 0; i < dim; i++) v[i] += wv[i];
        count++;
    }
    if (!count) return null;
    for (let i = 0; i < dim; i++) v[i] /= count;
    return v;
}

function sentVec(text) {
    return sentVecFromTokens(rawTokens(text));
}

function search(query, top) {
    const qv = sentVec(query);
    if (!qv) return [];
    const out = [];
    for (let i = 0; i < sentences.length; i++) {
        const sv = sentVecs[i];
        if (!sv) continue;
        out.push({ i, text: sentences[i], score: cosine(qv, sv) });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, top || 5);
}

function nearest(word, top) {
    const w = clean(word);
    if (!vectors.words[w]) return [];
    const base = vectors.words[w];
    const out = [];
    for (const k in vectors.words) {
        if (k === w) continue;
        out.push({ word: k, score: cosine(base, vectors.words[k]) });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, top || 10);
}

function apiRequest(method, data) {
    return new Promise((resolve) => {
        const ds = JSON.stringify(data);
        const opts = {
            hostname: 'api.telegram.org',
            path: `/bot${BOT_TOKEN}/${method}`,
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(ds) },
            timeout: 40000
        };
        const req = https.request(opts, (res) => {
            let body = '';
            res.on('data', c => body += c);
            res.on('end', () => { try { resolve(JSON.parse(body)); } catch { resolve({ ok: false }); } });
        });
        req.on('error', () => resolve({ ok: false }));
        req.on('timeout', () => { req.destroy(); resolve({ ok: false }); });
        req.write(ds);
        req.end();
    });
}

function send(chatId, text) {
    return apiRequest('sendMessage', { chat_id: chatId, text: String(text).slice(0, 4000) });
}

let pendingReset = false;
let polling = false;
let training = false;
let writing = {};   // userId -> true, если идёт запись в корпус

async function getUpdates(offset) {
    if (polling) return;
    polling = true;
    const res = await apiRequest('getUpdates', { offset, timeout: 30, allowed_updates: ['message'] });
    polling = false;
    if (res && res.ok && res.result && res.result.length) {
        for (const u of res.result) {
            offset = u.update_id + 1;
            saveOffset(offset);
            if (u.message && (typeof u.message.text === 'string' || u.message.document)) {
                try { await handle(u.message); } catch (e) { console.error('handle:', e.message); }
            }
        }
        setTimeout(() => getUpdates(offset), 50);
    } else {
        setTimeout(() => getUpdates(offset), (res && res.ok) ? 50 : 3000);
    }
}

async function handle(msg) {
    const chatId = msg.chat.id;
    const userId = msg.from.id;
    const text = (msg.text || '').trim();

    if (text === '/start') {
        await send(chatId,
            'Бот на семантической модели.\n\n' +
            'Как учить:\n' +
            '/add — начать писать текст в корпус. Пиши сообщения подряд, они добавятся.\n' +
            '/stop — закончить запись.\n' +
            '/retrain — обучить модель на том, что записал.\n\n' +
            'Или можно просто кинуть .txt файл в чат — он тоже пойдёт в корпус.\n\n' +
            'Как спрашивать:\n' +
            'Просто пиши боту вопрос или фразу — он найдёт самое близкое по смыслу в корпусе.\n' +
            '/ask вопрос — показать топ-3 близких предложения.\n' +
            '/similar слово — похожие по смыслу слова.\n\n' +
            'Другое:\n' +
            '/stats — что в модели.\n' +
            '/reset — стереть всё (с подтверждением "да").\n' +
            '/export — скачать корпус.\n' +
            '/import — загрузить корпус из файла.'
        );
        return;
    }

    if (pendingReset) {
        if (text.toLowerCase() === 'да') {
            vectors = { dim: 50, words: {}, trained: false, tokens: 0 };
            sentences = [];
            sentVecs = [];
            writing = {};
            pendingReset = false;
            try { fs.unlinkSync(CORPUS_FILE); } catch {}
            try { fs.unlinkSync(VEC_FILE); } catch {}
            try { fs.unlinkSync(VEC_META); } catch {}
            try { fs.unlinkSync(SENTS_FILE); } catch {}
            await send(chatId, 'Стёрто.');
        } else {
            pendingReset = false;
            await send(chatId, 'Отменено.');
        }
        return;
    }

    if (text === '/reset') {
        pendingReset = true;
        await send(chatId, 'Точно стереть всё? Напиши "да".');
        return;
    }

    if (writing[userId]) {
        if (text === '/stop') {
            writing[userId] = false;
            await send(chatId, 'Запись закончена. Теперь /retrain.');
            return;
        }
        if (text) {
            appendCorpus(text);
            await send(chatId, 'Добавлено.');
            return;
        }
    }

    if (text === '/add') {
        writing[userId] = true;
        await send(chatId, 'Пиши текст. Каждое сообщение пойдёт в корпус. Когда закончишь — /stop.');
        return;
    }

    if (text === '/stop') {
        await send(chatId, 'Ты не в режиме записи. Начни с /add.');
        return;
    }

    if (text === '/stats') {
        await send(chatId,
            'Слов в модели: ' + Object.keys(vectors.words).length + '\n' +
            'Размер вектора: ' + vectors.dim + '\n' +
            'Токенов обучено: ' + (vectors.tokens || 0) + '\n' +
            'Предложений: ' + sentences.length + '\n' +
            'Обучена: ' + (vectors.trained ? 'да' : 'нет') + '\n' +
            'Корпус: ' + (fs.existsSync(CORPUS_FILE) ? (fs.statSync(CORPUS_FILE).size + ' байт') : 'пусто'));
        return;
    }

    if (text === '/retrain') {
        if (training) { await send(chatId, 'Уже обучается.'); return; }
        training = true;
        await send(chatId, 'Обучаю...');
        try {
            const r = trainWord2Vec({});
            await send(chatId, r.ok ? ('Готово. Слов: ' + r.vocab + ', токенов: ' + r.tokens + ', предложений: ' + r.sentences) : ('Ошибка: ' + r.error));
        } catch (e) {
            await send(chatId, 'Ошибка: ' + e.message);
        }
        training = false;
        return;
    }

    if (text.startsWith('/ask ')) {
        const q = text.slice(5).trim();
        if (!q) { await send(chatId, '/ask вопрос'); return; }
        if (!vectors.trained) { await send(chatId, 'Сначала /retrain.'); return; }
        const res = search(q, 3);
        if (!res.length) { await send(chatId, 'Ничего не нашёл.'); return; }
        let out = 'Ближайшее по смыслу:\n\n';
        for (const r of res) out += '[' + r.score.toFixed(3) + '] ' + r.text + '\n\n';
        await send(chatId, out);
        return;
    }

    if (text.startsWith('/similar ')) {
        const w = text.slice(9).trim();
        if (!vectors.trained) { await send(chatId, 'Сначала /retrain.'); return; }
        const list = nearest(w, 15);
        if (!list.length) { await send(chatId, 'Нет в модели.'); return; }
        let out = 'Похожие на "' + w + '":\n';
        for (const it of list) out += it.word + ' (' + it.score.toFixed(3) + ')\n';
        await send(chatId, out);
        return;
    }

    if (text === '/export') {
        try {
            const corpus = fs.existsSync(CORPUS_FILE) ? fs.readFileSync(CORPUS_FILE, 'utf8') : '';
            const buf = Buffer.from(JSON.stringify({ corpus }), 'utf8');
            const b = '----B' + Date.now();
            const head = '--' + b + '\r\nContent-Disposition: form-data; name="document"; filename="corpus.json"\r\nContent-Type: application/json\r\n\r\n';
            const tail = '\r\n--' + b + '--\r\n';
            const body = Buffer.concat([Buffer.from(head, 'utf8'), buf, Buffer.from(tail, 'utf8')]);
            await new Promise((resolve) => {
                const opts = {
                    hostname: 'api.telegram.org',
                    path: `/bot${BOT_TOKEN}/sendDocument`,
                    method: 'POST',
                    headers: { 'Content-Type': 'multipart/form-data; boundary=' + b, 'Content-Length': body.length },
                    timeout: 40000
                };
                const req = https.request(opts, (res) => { res.on('data', () => {}); res.on('end', () => resolve()); });
                req.on('error', () => resolve());
                req.on('timeout', () => { req.destroy(); resolve(); });
                req.write(body);
                req.end();
            });
        } catch {}
        return;
    }

    if (text === '/import') {
        await send(chatId, 'Кинь corpus.json.');
        return;
    }

    if (msg.document) {
        try {
            const fr = await apiRequest('getFile', { file_id: msg.document.file_id });
            if (!fr.ok) { await send(chatId, 'Ошибка.'); return; }
            const fd = await new Promise((resolve) => {
                https.get(`https://api.telegram.org/file/bot${BOT_TOKEN}/${fr.result.file_path}`, (res) => {
                    const c = [];
                    res.on('data', x => c.push(x));
                    res.on('end', () => resolve(Buffer.concat(c)));
                }).on('error', () => resolve(null));
            });
            if (!fd) { await send(chatId, 'Ошибка.'); return; }
            const name = (msg.document.file_name || '').toLowerCase();
            if (name.endsWith('.txt')) {
                appendCorpus(fd.toString('utf8'));
                await send(chatId, 'Добавлено. Корпус: ' + fs.statSync(CORPUS_FILE).size + ' байт. Теперь /retrain.');
                return;
            }
            if (name.endsWith('.json')) {
                const p = JSON.parse(fd.toString('utf8'));
                if (p.corpus) {
                    fs.writeFileSync(CORPUS_FILE, p.corpus, 'utf8');
                    await send(chatId, 'Корпус загружен. Теперь /retrain.');
                    return;
                }
            }
            await send(chatId, 'Неизвестный файл.');
        } catch (e) { await send(chatId, 'Ошибка: ' + e.message); }
        return;
    }

    if (!text) return;

    if (!vectors.trained) {
        await send(chatId, 'Модель не обучена. Напиши /add, накидай текст, потом /retrain.');
        return;
    }

    const res = search(text, 1);
    if (!res.length || res[0].score < 0.3) {
        await send(chatId, 'Ничего близкого не нашёл.');
        return;
    }
    await send(chatId, res[0].text);
}

const start = loadOffset();
console.log('Bot started. Offset:', start);
getUpdates(start);
