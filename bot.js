const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BOT_TOKEN = process.env.BOT_TOKEN;
if (!BOT_TOKEN) { console.error('BOT_TOKEN is not set'); process.exit(1); }

const PORT = process.env.PORT || 3000;
const DATA_DIR = process.env.DATA_DIR || './data';
const DATA_FILE = path.join(DATA_DIR, 'brain.json');
const VEC_FILE = path.join(DATA_DIR, 'vectors.bin');
const VEC_META = path.join(DATA_DIR, 'vectors.json');
const OFFSET_FILE = path.join(DATA_DIR, 'offset.json');
const CORPUS_FILE = path.join(DATA_DIR, 'corpus.txt');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT);

let brain = { vocabulary: {}, relations: {}, lastInput: {}, synonyms: {}, context: [] };
let vectors = { dim: 50, words: {}, trained: false, tokens: 0 };

if (fs.existsSync(DATA_FILE)) {
    try { brain = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8')); } catch (e) {}
}
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

let saveTimer = null, saving = false;

function saveBrainNow() {
    if (saving) return;
    saving = true;
    try {
        const tmp = DATA_FILE + '.tmp';
        fs.writeFileSync(tmp, JSON.stringify(brain), 'utf8');
        fs.renameSync(tmp, DATA_FILE);
    } catch (e) {} finally { saving = false; }
}

function saveBrain() {
    if (saveTimer) return;
    saveTimer = setTimeout(() => { saveTimer = null; saveBrainNow(); }, 2000);
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

process.on('SIGINT', () => { saveBrainNow(); saveVectorsNow(); process.exit(0); });
process.on('SIGTERM', () => { saveBrainNow(); saveVectorsNow(); process.exit(0); });

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

const ENDINGS = ['иями','ями','ами','ией','иям','ием','иях','ию','ия','ий','ый','ой','ая','ое','ые','ими','ыми','ов','ев','ам','ям','ах','ях','ом','ем','ой','ей','ую','юю','ишь','ешь','ете','ите','ует','уют','ал','ял','ил','ел','ла','ло','ли','ть','ся','сь','а','я','о','е','у','ю','ы','и','й','ь'];

function stem(w) {
    if (w.length <= 3) return w;
    for (const e of ENDINGS) {
        if (w.length - e.length >= 3 && w.endsWith(e)) return w.slice(0, w.length - e.length);
    }
    return w;
}

function rawTokens(text) {
    const c = clean(text);
    if (!c) return [];
    return c.split(' ').filter(Boolean).map(stem);
}

function tokens(text) {
    const r = rawTokens(text);
    const out = [];
    for (const w of r) {
        const s = brain.synonyms[w];
        out.push(s && s.length ? stem(s[0]) : w);
    }
    return out;
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
    if (!raw) return { ok: false, error: 'corpus empty' };

    const sentences = raw.split(/[\n.!?]+/).map(s => rawTokens(s)).filter(s => s.length > 1);
    const freq = {};
    for (const s of sentences) for (const w of s) freq[w] = (freq[w] || 0) + 1;

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
    for (const s of sentences) for (const w of s) if (vocabSet.has(w)) tokensCount++;

    for (let epoch = 0; epoch < epochs; epoch++) {
        const lr = lr0 * (1 - epoch / epochs);
        for (const s of sentences) {
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
    return { ok: true, vocab: vocab.length, tokens: tokensCount, sentences: sentences.length };
}

function nearest(word, top) {
    const w = stem(clean(word));
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

function sentVec(text) {
    const t = rawTokens(text).filter(w => vectors.words[w]);
    if (!t.length) return null;
    const dim = vectors.dim;
    const v = new Float32Array(dim);
    for (const w of t) {
        const wv = vectors.words[w];
        for (let i = 0; i < dim; i++) v[i] += wv[i];
    }
    for (let i = 0; i < dim; i++) v[i] /= t.length;
    return v;
}

function semSearch(query, top) {
    const qv = sentVec(query);
    if (!qv) return [];
    const out = [];
    for (const key in brain.relations) {
        const ov = sentVec(key);
        if (!ov) continue;
        out.push({ key, score: cosine(qv, ov) });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, top || 5);
}

function pickWeighted(o) {
    const keys = Object.keys(o);
    if (!keys.length) return null;
    let total = 0;
    for (const k of keys) total += o[k];
    let r = Math.random() * total;
    for (const k of keys) {
        r -= o[k];
        if (r <= 0) return k;
    }
    return keys[keys.length - 1];
}

function trainAI(input, output) {
    const t = tokens(input);
    const a = String(output).trim();
    if (!t.length || !a) return false;
    const key = t.join(' ');
    if (!brain.relations[key]) brain.relations[key] = {};
    brain.relations[key][a] = (brain.relations[key][a] || 0) + 1;
    for (const w of t) brain.vocabulary[w] = (brain.vocabulary[w] || 0) + 1;
    saveBrain();
    return true;
}

function wordWeight(w) {
    const total = Object.values(brain.vocabulary).reduce((a, b) => a + b, 0) || 1;
    return Math.log((total + 1) / ((brain.vocabulary[w] || 0) + 1)) + 1;
}

function lexScore(inputT, knownKey) {
    const known = knownKey.split(' ');
    let s = 0;
    for (const t of inputT) if (known.includes(t)) s += wordWeight(t);
    return s;
}

function thinkAI(text, userId) {
    const t = tokens(text);
    if (!t.length) return null;
    const key = t.join(' ');
    if (brain.relations[key]) {
        const b = pickWeighted(brain.relations[key]);
        if (b) return b;
    }
    if (vectors.trained) {
        for (const s of semSearch(text, 3)) {
            if (s.score < 0.5) continue;
            const b = pickWeighted(brain.relations[s.key]);
            if (b) return b;
        }
    }
    let bestKey = null, bestScore = 0;
    for (const k in brain.relations) {
        const s = lexScore(t, k);
        if (s > bestScore) { bestScore = s; bestKey = k; }
    }
    if (bestKey && bestScore > 0) {
        const b = pickWeighted(brain.relations[bestKey]);
        if (b) return b;
    }
    return null;
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
    return apiRequest('sendMessage', { chat_id: chatId, text: text.slice(0, 4000) });
}

const PAGE = 10;

function listPage(page) {
    const items = Object.keys(brain.relations).sort();
    const total = Math.max(1, Math.ceil(items.length / PAGE));
    if (page < 0) page = 0;
    if (page >= total) page = total - 1;
    const slice = items.slice(page * PAGE, page * PAGE + PAGE);
    let out = 'Пар: ' + items.length + ' | стр. ' + (page + 1) + '/' + total + '\n\n';
    slice.forEach((k, i) => {
        out += (page * PAGE + i + 1) + '. ' + k + '\n';
        for (const a of Object.keys(brain.relations[k])) out += '   -> ' + a + ' (' + brain.relations[k][a] + ')\n';
        const syns = brain.synonyms[k];
        if (syns && syns.length) out += '   ~ ' + syns.join(', ') + '\n';
        out += '\n';
    });
    return out;
}

function synPage(page) {
    const seen = new Set();
    const groups = [];
    for (const w in brain.synonyms) {
        if (seen.has(w)) continue;
        const list = brain.synonyms[w];
        if (!list || !list.length) continue;
        const g = [w, ...list];
        for (const x of g) seen.add(x);
        groups.push(g);
    }
    groups.sort((a, b) => a[0].localeCompare(b[0]));
    const total = Math.max(1, Math.ceil(groups.length / PAGE));
    if (page < 0) page = 0;
    if (page >= total) page = total - 1;
    const slice = groups.slice(page * PAGE, page * PAGE + PAGE);
    let out = 'Групп: ' + groups.length + ' | стр. ' + (page + 1) + '/' + total + '\n\n';
    slice.forEach((g, i) => {
        out += (page * PAGE + i + 1) + '. ' + g[0] + '\n';
        if (g.length > 1) out += '   ~ ' + g.slice(1).join(', ') + '\n';
        out += '\n';
    });
    return out;
}

function delPair(q) {
    const key = tokens(q).join(' ');
    if (!key || !brain.relations[key]) return false;
    delete brain.relations[key];
    saveBrainNow();
    return true;
}

function delAnswer(q, a) {
    const key = tokens(q).join(' ');
    if (!key || !brain.relations[key]) return false;
    const ans = String(a).trim();
    if (!brain.relations[key][ans]) return false;
    delete brain.relations[key][ans];
    if (!Object.keys(brain.relations[key]).length) delete brain.relations[key];
    saveBrainNow();
    return true;
}

function addSyns(word, list) {
    const w = clean(word);
    if (!w) return false;
    const syns = list.map(clean).filter(Boolean);
    if (!syns.length) return false;
    if (!brain.synonyms[w]) brain.synonyms[w] = [];
    for (const s of syns) {
        if (!brain.synonyms[w].includes(s)) brain.synonyms[w].push(s);
        if (!brain.synonyms[s]) brain.synonyms[s] = [w];
        else if (!brain.synonyms[s].includes(w)) brain.synonyms[s].push(w);
    }
    saveBrain();
    return true;
}

function delSyns(word) {
    const w = clean(word);
    if (!w || !brain.synonyms[w]) return false;
    const list = [...brain.synonyms[w]];
    delete brain.synonyms[w];
    for (const s of list) {
        if (brain.synonyms[s]) {
            brain.synonyms[s] = brain.synonyms[s].filter(x => x !== w);
            if (!brain.synonyms[s].length) delete brain.synonyms[s];
        }
    }
    saveBrainNow();
    return true;
}

let pendingReset = false;
let polling = false;

async function getUpdates(offset) {
    if (polling) return;
    polling = true;
    const res = await apiRequest('getUpdates', { offset, timeout: 30, allowed_updates: ['message'] });
    polling = false;
    if (res && res.ok && res.result && res.result.length) {
        for (const u of res.result) {
            offset = u.update_id + 1;
            saveOffset(offset);
            if (u.message && typeof u.message.text === 'string') {
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
    const text = msg.text.trim();
    if (!text) return;

    if (text === '/start') {
        await send(chatId,
            'Команды бота:\n\n' +
            '/train вопрос = ответ — обучить бота. Можно повторять с тем же вопросом и разными ответами, тогда бот будет выбирать случайно.\n\n' +
            '/syn слово = синоним1, синоним2 — добавить синонимы. Слова из одной группы бот считает одинаковыми.\n' +
            '/syn del слово — удалить все синонимы слова.\n' +
            '/synlist [n] — список групп синонимов. Без n — первая страница, с n — страница n.\n\n' +
            '/list [n] — список выученных пар. Показывает все ответы и веса. Листается через /list 2, /list 3 и т.д.\n\n' +
            '/del вопрос — удалить всю пару со всеми ответами.\n' +
            '/del вопрос = ответ — удалить только один конкретный ответ.\n\n' +
            '/learn — отправить .txt файл для обучения модели. Бот добавит текст в корпус.\n' +
            '/retrain — обучить модель на собранных текстах. Занимает время, зависит от объёма.\n' +
            '/similar слово — показать слова, близкие по смыслу (после обучения модели).\n\n' +
            '/stats — статистика: пары, ответы, слова, синонимы, размер модели.\n' +
            '/reset — стереть всё. Спросит подтверждение, надо написать "да".\n\n' +
            '/export — скачать базу и модель файлом.\n' +
            '/import — загрузить базу из файла.\n\n' +
            'Как учить: пиши /train привет = Привет! Потом просто напиши боту "привет" — он ответит.\n' +
            'Если хочешь несколько ответов — повтори /train с тем же вопросом и другим ответом.\n' +
            'Если хочешь, чтобы бот понимал смысл — кинь ему .txt с текстами и сделай /retrain.'
        );
        return;
    }

    if (pendingReset) {
        if (text.toLowerCase() === 'да') {
            brain = { vocabulary: {}, relations: {}, lastInput: {}, synonyms: {}, context: [] };
            pendingReset = false;
            saveBrainNow();
            await send(chatId, 'Память очищена.');
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

    if (text === '/train' || text.startsWith('/train ')) {
        const rest = text.replace(/^\/train\s*/, '').trim();
        if (rest.includes('=')) {
            const i = rest.indexOf('=');
            const q = rest.slice(0, i).trim();
            const a = rest.slice(i + 1).trim();
            if (q && a) {
                trainAI(q, a);
                await send(chatId, 'OK');
            } else await send(chatId, '/train вопрос = ответ');
            return;
        }
        const last = brain.lastInput[userId];
        if (!last) await send(chatId, 'Сначала спроси.');
        else if (!rest) await send(chatId, '/train вопрос = ответ');
        else {
            trainAI(last, rest);
            await send(chatId, 'OK');
        }
        return;
    }

    if (text.startsWith('/syn')) {
        const rest = text.replace(/^\/syn\s*/, '').trim();
        if (rest.startsWith('del ')) {
            const w = rest.slice(4).trim();
            const ok = delSyns(w);
            await send(chatId, ok ? 'Удалено.' : 'Не найдено.');
            return;
        }
        if (!rest.includes('=')) {
            await send(chatId, '/syn слово = синоним1, синоним2');
            return;
        }
        const i = rest.indexOf('=');
        const w = rest.slice(0, i).trim();
        const list = rest.slice(i + 1).split(',').map(s => s.trim()).filter(Boolean);
        addSyns(w, list);
        await send(chatId, 'OK');
        return;
    }

    if (text === '/synlist' || text.startsWith('/synlist ')) {
        if (!Object.keys(brain.synonyms).length) { await send(chatId, 'Пусто.'); return; }
        const p = parseInt(text.slice(8).trim(), 10);
        await send(chatId, synPage(isNaN(p) ? 0 : p - 1));
        return;
    }

    if (text === '/list' || text.startsWith('/list ')) {
        if (!Object.keys(brain.relations).length) { await send(chatId, 'Пусто.'); return; }
        const p = parseInt(text.slice(5).trim(), 10);
        await send(chatId, listPage(isNaN(p) ? 0 : p - 1));
        return;
    }

    if (text.startsWith('/del ')) {
        const arg = text.slice(5).trim();
        if (!arg) { await send(chatId, '/del вопрос  ИЛИ  /del вопрос = ответ'); return; }
        if (arg.includes('=')) {
            const i = arg.indexOf('=');
            const ok = delAnswer(arg.slice(0, i).trim(), arg.slice(i + 1).trim());
            await send(chatId, ok ? 'Удалено.' : 'Не найдено.');
        } else {
            const ok = delPair(arg);
            await send(chatId, ok ? 'Удалено.' : 'Не найдено.');
        }
        return;
    }

    if (text === '/stats') {
        const pairs = Object.keys(brain.relations).length;
        let ans = 0, ex = 0;
        for (const q in brain.relations) {
            const o = brain.relations[q];
            ans += Object.keys(o).length;
            for (const a in o) ex += o[a];
        }
        await send(chatId,
            'Пар: ' + pairs + '\n' +
            'Ответов: ' + ans + '\n' +
            'Примеров: ' + ex + '\n' +
            'Слов: ' + Object.keys(brain.vocabulary).length + '\n' +
            'Синонимов: ' + Object.keys(brain.synonyms).length + '\n' +
            'Векторов: ' + Object.keys(vectors.words).length + ' (dim ' + vectors.dim + ')\n' +
            'Обучена: ' + (vectors.trained ? 'да' : 'нет'));
        return;
    }

    if (text === '/learn') {
        await send(chatId, 'Кинь .txt файл.');
        return;
    }

    if (text === '/retrain') {
        await send(chatId, 'Обучаю... Это займёт время.');
        const r = trainWord2Vec({});
        await send(chatId, r.ok ? ('Готово. Слов: ' + r.vocab + ', токенов: ' + r.tokens) : ('Ошибка: ' + r.error));
        return;
    }

    if (text.startsWith('/similar ')) {
        const w = text.slice(9).trim();
        const list = nearest(w, 15);
        if (!list.length) { await send(chatId, 'Нет в модели.'); return; }
        let out = 'Похожие на "' + w + '":\n';
        for (const it of list) out += it.word + ' (' + it.score.toFixed(3) + ')\n';
        await send(chatId, out);
        return;
    }

    if (text === '/export') {
        try {
            const buf = Buffer.from(JSON.stringify({ brain, vectorsMeta: { dim: vectors.dim, trained: vectors.trained, tokens: vectors.tokens } }, null, 2), 'utf8');
            const b = '----B' + Date.now();
            const head = '--' + b + '\r\nContent-Disposition: form-data; name="document"; filename="brain.json"\r\nContent-Type: application/json\r\n\r\n';
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

    if (text === '/import') { await send(chatId, 'Кинь brain.json.'); return; }

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
                await send(chatId, 'Добавлено. Теперь /retrain.');
                return;
            }
            const p = JSON.parse(fd.toString('utf8'));
            if (p.brain) brain = p.brain;
            saveBrainNow();
            await send(chatId, 'OK');
        } catch (e) { await send(chatId, 'Ошибка: ' + e.message); }
        return;
    }

    brain.lastInput[userId] = text;
    brain.context.push(text);
    if (brain.context.length > 5) brain.context.shift();
    saveBrain();

    const a = thinkAI(text, userId);
    await send(chatId, a || 'Не знаю. /train вопрос = ответ');
}

const start = loadOffset();
console.log('Bot started. Offset:', start);
getUpdates(start);
