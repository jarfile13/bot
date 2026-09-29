const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BOT_TOKEN = process.env.BOT_TOKEN;
if (!BOT_TOKEN) {
  console.error('BOT_TOKEN не задан');
  process.exit(1);
}

const PORT = process.env.PORT || 3000;
const DATA_DIR = process.env.DATA_DIR || './data';
const MODEL_DIR = path.join(DATA_DIR, 'models');

for (const d of [DATA_DIR, MODEL_DIR]) {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
}

http.createServer((req, res) => { res.writeHead(200); res.end('OK'); }).listen(PORT);

function uniq(arr) { return [...new Set(arr)]; }
function sum(arr) { return arr.reduce((a, b) => a + b, 0); }
function log2(x) { return Math.log(x) / Math.LN2; }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function tokenize(str) {
  return String(str)
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter(w => w.length > 0);
}

function clean(str) {
  return String(str).toLowerCase().replace(/ё/g, 'е').replace(/\s+/g, ' ').trim();
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function saveJSON(file, obj) {
  try { fs.writeFileSync(file, JSON.stringify(obj), 'utf8'); } catch (e) {}
}
function loadJSON(file, def) {
  if (!fs.existsSync(file)) return def;
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) { return def; }
}

class NaiveBayes {
  constructor() {
    this.classDocs = {};
    this.classWords = {};
    this.vocab = new Set();
    this.totalDocs = 0;
  }

  train(intent, text) {
    const words = tokenize(text);
    if (words.length === 0) return;
    if (!this.classDocs[intent]) this.classDocs[intent] = 0;
    if (!this.classWords[intent]) this.classWords[intent] = {};
    this.classDocs[intent]++;
    this.totalDocs++;
    for (const w of words) {
      this.vocab.add(w);
      this.classWords[intent][w] = (this.classWords[intent][w] || 0) + 1;
    }
  }

  predict(text) {
    const words = tokenize(text);
    if (words.length === 0 || this.totalDocs === 0) return null;
    const V = this.vocab.size || 1;
    const scores = {};
    for (const intent in this.classDocs) {
      const prior = Math.log(this.classDocs[intent] / this.totalDocs);
      const wordCounts = this.classWords[intent];
      const totalWords = sum(Object.values(wordCounts)) || 1;
      let score = prior;
      for (const w of words) {
        const c = wordCounts[w] || 0;
        score += Math.log((c + 1) / (totalWords + V));
      }
      scores[intent] = score;
    }
    const best = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
    const maxS = scores[best];
    let z = 0;
    const probs = {};
    for (const k in scores) { probs[k] = Math.exp(scores[k] - maxS); z += probs[k]; }
    for (const k in probs) probs[k] /= z;
    return { intent: best, confidence: probs[best], probs };
  }

  toJSON() {
    return {
      classDocs: this.classDocs,
      classWords: this.classWords,
      vocab: [...this.vocab],
      totalDocs: this.totalDocs
    };
  }

  static fromJSON(j) {
    const nb = new NaiveBayes();
    if (!j) return nb;
    nb.classDocs = j.classDocs || {};
    nb.classWords = j.classWords || {};
    nb.vocab = new Set(j.vocab || []);
    nb.totalDocs = j.totalDocs || 0;
    return nb;
  }
}

class TfIdf {
  constructor() {
    this.docs = [];
    this.df = {};
    this.N = 0;
  }

  add(id, text, tokens) {
    const tf = {};
    for (const w of tokens) tf[w] = (tf[w] || 0) + 1;
    this.docs.push({ id, text, tokens, tf });
    for (const w of Object.keys(tf)) this.df[w] = (this.df[w] || 0) + 1;
    this.N++;
  }

  idf(w) {
    return Math.log((this.N + 1) / ((this.df[w] || 0) + 1)) + 1;
  }

  vec(tf) {
    const v = {};
    let norm = 0;
    for (const w in tf) {
      const x = tf[w] * this.idf(w);
      v[w] = x;
      norm += x * x;
    }
    norm = Math.sqrt(norm) || 1;
    for (const w in v) v[w] /= norm;
    return v;
  }

  query(tokens) {
    const tf = {};
    for (const w of tokens) tf[w] = (tf[w] || 0) + 1;
    return this.vec(tf);
  }

  cosine(a, b) {
    let s = 0;
    for (const w in a) if (b[w]) s += a[w] * b[w];
    return s;
  }

  search(tokens, topK = 5) {
    const q = this.query(tokens);
    const scored = this.docs.map(d => ({ id: d.id, score: this.cosine(q, this.vec(d.tf)) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }
}

function matchTemplate(pattern, text) {
  const parts = pattern.split(/\{(\w+)\}/);
  let regexStr = '^';
  const slots = [];
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      slots.push(parts[i]);
      regexStr += '(.+?)';
    } else {
      regexStr += parts[i].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
  }
  regexStr += '$';
  const re = new RegExp(regexStr, 'i');
  const m = clean(text).match(re);
  if (!m) return null;
  const result = {};
  slots.forEach((name, i) => { result[name] = m[i + 1].trim(); });
  return result;
}

function fillTemplate(tpl, slots) {
  return tpl.replace(/\{(\w+)\}/g, (_, name) => slots[name] !== undefined ? slots[name] : '');
}

class AssocGraph {
  constructor() {
    this.edges = {};
    this.nodeCount = {};
  }

  addPair(a, b, w = 1) {
    if (!a || !b || a === b) return;
    if (!this.edges[a]) this.edges[a] = {};
    if (!this.edges[b]) this.edges[b] = {};
    this.edges[a][b] = (this.edges[a][b] || 0) + w;
    this.edges[b][a] = (this.edges[b][a] || 0) + w;
    this.nodeCount[a] = (this.nodeCount[a] || 0) + 1;
    this.nodeCount[b] = (this.nodeCount[b] || 0) + 1;
  }

  spread(seeds, steps = 2, decay = 0.6) {
    const act = {};
    for (const s of seeds) act[s] = (act[s] || 0) + 1;
    let frontier = [...seeds];
    for (let step = 0; step < steps; step++) {
      const next = {};
      for (const node of frontier) {
        const a = act[node] || 0;
        const nbrs = this.edges[node];
        if (!nbrs) continue;
        const total = sum(Object.values(nbrs)) || 1;
        for (const nb in nbrs) {
          const contrib = a * (nbrs[nb] / total) * decay;
          next[nb] = (next[nb] || 0) + contrib;
        }
      }
      for (const k in next) act[k] = (act[k] || 0) + next[k];
      frontier = Object.keys(next);
      if (frontier.length === 0) break;
    }
    return act;
  }
}

class NGramLM {
  constructor(order = 2) {
    this.order = order;
    this.bigrams = {};
    this.trigrams = {};
  }

  train(tokens) {
    for (let i = 0; i < tokens.length - 1; i++) {
      const a = tokens[i], b = tokens[i + 1];
      if (!this.bigrams[a]) this.bigrams[a] = {};
      this.bigrams[a][b] = (this.bigrams[a][b] || 0) + 1;
    }
    for (let i = 0; i < tokens.length - 2; i++) {
      const key = tokens[i] + ' ' + tokens[i + 1];
      const c = tokens[i + 2];
      if (!this.trigrams[key]) this.trigrams[key] = {};
      this.trigrams[key][c] = (this.trigrams[key][c] || 0) + 1;
    }
  }

  next(word1, word2) {
    if (word1 && word2) {
      const key = word1 + ' ' + word2;
      const dist = this.trigrams[key];
      if (dist) {
        const best = Object.keys(dist).reduce((a, b) => dist[a] > dist[b] ? a : b);
        return best;
      }
    }
    if (word2) {
      const dist = this.bigrams[word2];
      if (dist) {
        const best = Object.keys(dist).reduce((a, b) => dist[a] > dist[b] ? a : b);
        return best;
      }
    }
    return null;
  }

  continueText(tokens, maxWords = 5) {
    const out = [...tokens];
    for (let i = 0; i < maxWords; i++) {
      const n = out.length;
      const w1 = n >= 2 ? out[n - 2] : null;
      const w2 = n >= 1 ? out[n - 1] : null;
      const nxt = this.next(w1, w2);
      if (!nxt) break;
      out.push(nxt);
    }
    return out;
  }
}

class Embeddings {
  constructor(dim = 30, window = 3) {
    this.dim = dim;
    this.window = window;
    this.vocab = [];
    this.w2i = {};
    this.vectors = {};
  }

  build(corpusTokens) {
    const freq = {};
    for (const tokens of corpusTokens) {
      for (const w of tokens) freq[w] = (freq[w] || 0) + 1;
    }
    this.vocab = Object.keys(freq).filter(w => freq[w] >= 2);
    this.vocab.forEach((w, i) => { this.w2i[w] = i; });
    const V = this.vocab.length;
    if (V < 2) return;

    const cooc = {};
    const rowSum = new Array(V).fill(0);
    const colSum = new Array(V).fill(0);
    let total = 0;
    for (const tokens of corpusTokens) {
      for (let i = 0; i < tokens.length; i++) {
        const wi = this.w2i[tokens[i]];
        if (wi === undefined) continue;
        for (let j = Math.max(0, i - this.window); j <= Math.min(tokens.length - 1, i + this.window); j++) {
          if (i === j) continue;
          const wj = this.w2i[tokens[j]];
          if (wj === undefined) continue;
          const key = wi + ',' + wj;
          cooc[key] = (cooc[key] || 0) + 1;
          rowSum[wi]++;
          colSum[wj]++;
          total++;
        }
      }
    }

    const ppmi = {};
    for (const key in cooc) {
      const [i, j] = key.split(',').map(Number);
      const pmi = log2((cooc[key] * total) / (rowSum[i] * colSum[j]));
      const v = Math.max(0, pmi);
      if (v > 0) ppmi[key] = v;
    }

    const vecs = [];
    for (let i = 0; i < V; i++) {
      const v = new Array(this.dim);
      for (let d = 0; d < this.dim; d++) v[d] = Math.random() * 0.01;
      vecs.push(v);
    }

    for (let d = 0; d < this.dim; d++) {
      for (let prev = 0; prev < d; prev++) {
        let dot = 0;
        for (let i = 0; i < V; i++) dot += vecs[i][d] * vecs[i][prev];
        for (let i = 0; i < V; i++) vecs[i][d] -= dot * vecs[i][prev];
      }
      let norm = 0;
      for (let i = 0; i < V; i++) norm += vecs[i][d] * vecs[i][d];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < V; i++) vecs[i][d] /= norm;
    }

    for (let iter = 0; iter < 30; iter++) {
      for (let d = 0; d < this.dim; d++) {
        const y = new Array(V).fill(0);
        for (const key in ppmi) {
          const [i, j] = key.split(',').map(Number);
          y[i] += ppmi[key] * vecs[j][d];
        }
        const x = new Array(V).fill(0);
        for (const key in ppmi) {
          const [i, j] = key.split(',').map(Number);
          x[j] += ppmi[key] * y[i];
        }
        for (let prev = 0; prev < d; prev++) {
          let dot = 0;
          for (let i = 0; i < V; i++) dot += x[i] * vecs[i][prev];
          for (let i = 0; i < V; i++) x[i] -= dot * vecs[i][prev];
        }
        let norm = 0;
        for (let i = 0; i < V; i++) norm += x[i] * x[i];
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < V; i++) vecs[i][d] = x[i] / norm;
      }
    }

    for (let i = 0; i < V; i++) this.vectors[this.vocab[i]] = vecs[i];
  }

  vec(word) { return this.vectors[word] || null; }

  cos(a, b) {
    let s = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return s / (Math.sqrt(na) * Math.sqrt(nb) || 1);
  }

  similar(word, topK = 5) {
    const v = this.vec(word);
    if (!v) return [];
    const scores = [];
    for (const w in this.vectors) {
      if (w === word) continue;
      scores.push([w, this.cos(v, this.vectors[w])]);
    }
    scores.sort((a, b) => b[1] - a[1]);
    return scores.slice(0, topK);
  }

  sentenceVec(tokens) {
    const vs = tokens.map(t => this.vec(t)).filter(Boolean);
    if (vs.length === 0) return null;
    const dim = vs[0].length;
    const out = new Array(dim).fill(0);
    for (const v of vs) for (let i = 0; i < dim; i++) out[i] += v[i];
    for (let i = 0; i < dim; i++) out[i] /= vs.length;
    return out;
  }
}

const nb = NaiveBayes.fromJSON(loadJSON(path.join(MODEL_DIR, 'nb.json'), null));
const emb = new Embeddings(30, 3);
{
  const saved = loadJSON(path.join(MODEL_DIR, 'emb.json'), null);
  if (saved) {
    emb.vocab = saved.vocab || [];
    emb.w2i = saved.w2i || {};
    emb.vectors = saved.vectors || {};
    emb.dim = saved.dim || 30;
    emb.window = saved.window || 3;
  }
}
const lm = new NGramLM(3);
{
  const saved = loadJSON(path.join(MODEL_DIR, 'lm.json'), null);
  if (saved) { lm.bigrams = saved.bigrams || {}; lm.trigrams = saved.trigrams || {}; }
}
const graph = new AssocGraph();
{
  const saved = loadJSON(path.join(MODEL_DIR, 'graph.json'), null);
  if (saved) { graph.edges = saved.edges || {}; graph.nodeCount = saved.nodeCount || {}; }
}

function saveModels() {
  saveJSON(path.join(MODEL_DIR, 'nb.json'), nb.toJSON());
  saveJSON(path.join(MODEL_DIR, 'emb.json'), {
    vocab: emb.vocab, w2i: emb.w2i, vectors: emb.vectors, dim: emb.dim, window: emb.window
  });
  saveJSON(path.join(MODEL_DIR, 'lm.json'), { bigrams: lm.bigrams, trigrams: lm.trigrams });
  saveJSON(path.join(MODEL_DIR, 'graph.json'), { edges: graph.edges, nodeCount: graph.nodeCount });
}

const factsFile = path.join(DATA_DIR, 'facts.json');
const templatesFile = path.join(DATA_DIR, 'templates.json');
let facts = loadJSON(factsFile, {});
let templates = loadJSON(templatesFile, []);

function saveFacts() { saveJSON(factsFile, facts); }
function saveTemplates() { saveJSON(templatesFile, templates); }

const userMemoryDir = path.join(DATA_DIR, 'users');
if (!fs.existsSync(userMemoryDir)) fs.mkdirSync(userMemoryDir, { recursive: true });

function userFile(userId) { return path.join(userMemoryDir, `${userId}.json`); }
const userMemoryCache = new Map();

function getUserMemory(userId) {
  if (userMemoryCache.has(userId)) return userMemoryCache.get(userId);
  const data = loadJSON(userFile(userId), { pairs: [], history: [] });
  if (!Array.isArray(data.pairs)) data.pairs = [];
  if (!Array.isArray(data.history)) data.history = [];
  userMemoryCache.set(userId, data);
  return data;
}
function saveUserMemory(userId) {
  const data = userMemoryCache.get(userId);
  if (data) saveJSON(userFile(userId), data);
}

const pending = new Map();
const lastBotMsg = new Map();

function trainIntent(intent, examples) {
  for (const ex of examples) nb.train(intent, ex);
}

function bootstrapIntents() {
  if (nb.totalDocs > 0) return;
  trainIntent('greeting', ['привет', 'здравствуй', 'здорово', 'хай', 'добрый день', 'добрый вечер', 'доброе утро', 'приветствую']);
  trainIntent('farewell', ['пока', 'до свидания', 'до связи', 'прощай', 'увидимся']);
  trainIntent('thanks', ['спасибо', 'благодарю', 'спс', 'пасибо']);
  trainIntent('howareyou', ['как дела', 'как ты', 'как жизнь', 'как настроение', 'что нового']);
  trainIntent('whoareyou', ['кто ты', 'как тебя зовут', 'ты кто', 'что ты такое', 'твое имя']);
  trainIntent('help', ['помоги', 'помощь', 'что делать', 'подскажи']);
  trainIntent('math', ['сколько будет', 'посчитай', 'вычисли', 'чему равно', 'реши пример']);
  saveModels();
}
bootstrapIntents();

const intentReplies = {
  greeting: ['Привет.', 'Здравствуй.', 'Приветствую.'],
  farewell: ['До связи.', 'Пока.', 'Увидимся.'],
  thanks: ['Пожалуйста.', 'Не за что.', 'Обращайся.'],
  howareyou: ['Работаю в штатном режиме.', 'Всё стабильно.', 'Нормально, а у тебя?'],
  whoareyou: ['Я самообучающийся текстовый бот.', 'Я ассоциативный ИИ, учусь на наших сообщениях.'],
  help: ['Опиши задачу подробнее.', 'Что именно нужно?', 'Сформулируй вопрос точнее.']
};

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

const wordToNum = {
  'ноль': 0, 'один': 1, 'одна': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4,
  'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9, 'десять': 10,
  'одиннадцать': 11, 'двенадцать': 12, 'тринадцать': 13, 'четырнадцать': 14,
  'пятнадцать': 15, 'шестнадцать': 16, 'семнадцать': 17, 'восемнадцать': 18,
  'девятнадцать': 19, 'двадцать': 20, 'тридцать': 30, 'сорок': 40, 'пятьдесят': 50,
  'шестьдесят': 60, 'семьдесят': 70, 'восемьдесят': 80, 'девяносто': 90,
  'сто': 100, 'двести': 200, 'триста': 300, 'четыреста': 400, 'пятьсот': 500,
  'тысяча': 1000
};

function wordsToNumbers(text) {
  const tokens = clean(text).split(' ');
  const out = [];
  for (const t of tokens) {
    if (wordToNum[t] !== undefined) out.push(String(wordToNum[t]));
    else out.push(t);
  }
  return out.join(' ');
}

function tryMath(text) {
  const t = wordsToNumbers(text)
    .replace(/плюс/g, '+').replace(/минус/g, '-')
    .replace(/умножить на/g, '*').replace(/умножить/g, '*')
    .replace(/разделить на/g, '/').replace(/делить на/g, '/')
    .replace(/[^\d+\-*/().\s]/g, ' ')
    .replace(/\s+/g, ' ').trim();
  const m = t.match(/(\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)+)/);
  if (!m) return null;
  try {
    const expr = m[1].replace(/\s+/g, '');
    if (!/^[\d+\-*/().]+$/.test(expr)) return null;
    const val = Function('"use strict";return (' + expr + ')')();
    if (typeof val !== 'number' || !isFinite(val)) return null;
    const r = Math.round(val * 1e10) / 1e10;
    return `Ответ: ${r}`;
  } catch (e) { return null; }
}

function tryRandomNumber(text) {
  const t = clean(text);
  const digitWordMatch = t.match(/(\d+)\s*[- ]?\s*значн/);
  let digits = digitWordMatch ? parseInt(digitWordMatch[1], 10) : null;
  if (!digits) {
    const map = { 'однозначн': 1, 'двузначн': 2, 'двухзначн': 2, 'трехзначн': 3, 'трёхзначн': 3,
      'четырехзначн': 4, 'четырёхзначн': 4, 'пятизначн': 5, 'шестизначн': 6, 'семизначн': 7,
      'восьмизначн': 8, 'девятизначн': 9, 'десятизначн': 10 };
    for (const k in map) if (t.includes(k)) { digits = map[k]; break; }
  }
  const rangeMatch = t.match(/от\s+(\d+)\s+до\s+(\d+)/);
  if (rangeMatch) {
    const lo = parseInt(rangeMatch[1], 10), hi = parseInt(rangeMatch[2], 10);
    if (hi >= lo) return String(lo + Math.floor(Math.random() * (hi - lo + 1)));
  }
  const wantsNum = /(числ|цифр|рандом|случайн)/.test(t);
  if (!digits || !wantsNum || digits < 1 || digits > 15) return null;
  let s = '';
  for (let i = 0; i < digits; i++) s += i === 0 ? String(1 + Math.floor(Math.random() * 9)) : String(Math.floor(Math.random() * 10));
  return s;
}

function detectIntent(text) {
  const p = nb.predict(text);
  return p && p.confidence > 0.55 ? p : null;
}

function findFact(text) {
  const t = clean(text);
  for (const tpl of templates) {
    const slots = matchTemplate(tpl.pattern, t);
    if (slots) {
      return fillTemplate(tpl.reply, slots);
    }
  }
  const m = t.match(/^(?:что такое|кто такой|кто такая|расскажи про|расскажи о)\s+(.+)$/);
  if (m) {
    const key = clean(m[1]);
    if (facts[key]) return facts[key];
    if (facts[key.replace(/^(а|the)\s+/, '')]) return facts[key.replace(/^(а|the)\s+/, '')];
    return null;
  }
  const m2 = t.match(/^(.+?)\s+(?:это|—)\s+(.+)$/);
  if (m2) {
    const key = clean(m2[1]);
    if (facts[key]) return facts[key];
  }
  return null;
}

function findInMemory(userId, text) {
  const mem = getUserMemory(userId);
  if (mem.pairs.length === 0) return null;

  const tfidf = new TfIdf();
  mem.pairs.forEach((p, i) => tfidf.add(i, p.q, tokenize(p.q)));
  const tokens = tokenize(text);
  const results = tfidf.search(tokens, 3);

  const act = graph.spread(tokens, 2, 0.5);
  const expandedTokens = uniq([...tokens, ...Object.keys(act).filter(w => act[w] > 0.1)]);

  const qv = emb.sentenceVec(expandedTokens);

  let best = null, bestScore = 0.35;
  for (const r of results) {
    const p = mem.pairs[r.id];
    let score = r.score;
    if (qv) {
      const pv = emb.sentenceVec(tokenize(p.q));
      if (pv) score = score * 0.6 + emb.cos(qv, pv) * 0.4;
    }
    if (score > bestScore) { bestScore = score; best = p; }
  }

  if (best && best.a && best.a.length > 0) {
    best.hits = (best.hits || 0) + 1;
    saveUserMemory(userId);
    return pick(best.a);
  }
  return null;
}

function teachPair(userId, question, answer) {
  const mem = getUserMemory(userId);
  const qTokens = tokenize(question);
  if (qTokens.length === 0 || !answer.trim()) return false;

  const qv = emb.sentenceVec(qTokens);
  let existing = null, bestSim = 0.85;
  for (const p of mem.pairs) {
    const pv = emb.sentenceVec(tokenize(p.q));
    if (!pv || !qv) {
      const inter = qTokens.filter(w => tokenize(p.q).includes(w)).length;
      const uni = uniq([...qTokens, ...tokenize(p.q)]).length || 1;
      if (inter / uni > bestSim) { bestSim = inter / uni; existing = p; }
    } else {
      const sim = emb.cos(qv, pv);
      if (sim > bestSim) { bestSim = sim; existing = p; }
    }
  }

  if (existing) {
    if (!existing.a.includes(answer.trim())) existing.a.push(answer.trim());
  } else {
    mem.pairs.push({ q: question.trim(), a: [answer.trim()], created: Date.now(), hits: 0 });
  }

  for (let i = 0; i < qTokens.length; i++) {
    for (let j = i + 1; j < qTokens.length; j++) graph.addPair(qTokens[i], qTokens[j], 1);
  }
  for (const t of tokenize(answer)) {
    for (const q of qTokens) graph.addPair(q, t, 0.5);
  }

  saveUserMemory(userId);
  saveModels();
  return true;
}

function tg(method, payload, isForm = false) {
  return new Promise((resolve, reject) => {
    let body, headers;
    if (isForm) {
      body = payload;
      headers = { 'Content-Type': payload.contentType };
    } else {
      body = JSON.stringify(payload);
      headers = { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) };
    }
    const req = https.request(`https://api.telegram.org/bot${BOT_TOKEN}/${method}`, { method: 'POST', headers });
    req.on('error', reject);
    req.on('response', res => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => { try { resolve(JSON.parse(d)); } catch (e) { resolve(d); } });
    });
    req.write(body);
    req.end();
  });
}

function send(chatId, text, extra) {
  const payload = { chat_id: chatId, text: String(text).slice(0, 4000) };
  if (extra) Object.assign(payload, extra);
  return tg('sendMessage', payload);
}

async function sendDocument(chatId, filePath, caption) {
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
  return tg('sendDocument', { body, contentType: `multipart/form-data; boundary=${boundary}` }, true);
}

async function downloadFile(fileId) {
  const r = await tg('getFile', { file_id: fileId });
  if (!r || !r.ok) return null;
  return new Promise((resolve) => {
    https.get(`https://api.telegram.org/file/bot${BOT_TOKEN}/${r.result.file_path}`, res => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    }).on('error', () => resolve(null));
  });
}

async function handleMessage(chatId, userId, text, document) {
  if (document && document.file_name) {
    if (document.file_name.endsWith('.json')) {
      const content = await downloadFile(document.file_id);
      if (!content) return send(chatId, 'Не удалось скачать файл.');
      try {
        const data = JSON.parse(content);
        if (Array.isArray(data.pairs)) {
          const mem = getUserMemory(userId);
          mem.pairs = mem.pairs.concat(data.pairs);
          saveUserMemory(userId);
          return send(chatId, `Импортировано пар: ${data.pairs.length}`);
        }
        if (Array.isArray(data)) {
          const mem = getUserMemory(userId);
          mem.pairs = mem.pairs.concat(data);
          saveUserMemory(userId);
          return send(chatId, `Импортировано пар: ${data.length}`);
        }
        return send(chatId, 'Неизвестный формат.');
      } catch (e) { return send(chatId, 'Ошибка чтения JSON.'); }
    }
    if (document.file_name.endsWith('.txt')) {
      const content = await downloadFile(document.file_id);
      if (!content) return send(chatId, 'Не удалось скачать файл.');
      const tokens = tokenize(content);
      lm.train(tokens);
      for (let i = 0; i < tokens.length; i++) {
        for (let j = i + 1; j < Math.min(tokens.length, i + 5); j++) {
          graph.addPair(tokens[i], tokens[j], 1);
        }
      }
      const sentences = content.split(/[.!?]+/).map(s => tokenize(s)).filter(s => s.length >= 3);
      emb.build(sentences);
      saveModels();
      return send(chatId, `Текст обработан: ${tokens.length} слов, предложений ${sentences.length}. Модель обновлена.`);
    }
    return send(chatId, 'Поддерживаются .json и .txt.');
  }

  if (!text) return;
  const trimmed = text.trim();
  const mem = getUserMemory(userId);

  if (trimmed.startsWith('/')) {
    const parts = trimmed.split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const arg = parts.slice(1).join(' ');

    if (cmd === '/start' || cmd === '/help') {
      return send(chatId,
        '<b>Самообучающийся бот</b>\n\n' +
        '<b>Обучение</b>\n' +
        '/teach вопрос = ответ\n' +
        '/fact ключ = значение\n' +
        '/template что такое {X} = Не знаю, что такое {X}\n' +
        '/text — пришли .txt, обучится на нём\n' +
        '/import — пришли .json для импорта\n\n' +
        '<b>Управление</b>\n' +
        '/list — список пар\n' +
        '/del &lt;номер&gt;\n' +
        '/find &lt;текст&gt;\n' +
        '/export — выгрузить память\n' +
        '/stats — статистика моделей\n' +
        '/similar &lt;слово&gt; — похожие слова\n' +
        '/continue &lt;начало&gt; — предсказать продолжение\n' +
        '/reset — стереть память\n' +
        '/cancel — сброс',
        { parse_mode: 'HTML' });
    }

    if (cmd === '/cancel') {
      pending.delete(userId);
      return send(chatId, 'Сброшено.');
    }

    if (cmd === '/reset') {
      pending.set(userId, { type: 'reset' });
      return send(chatId, 'Уверен? Напиши «да».');
    }

    if (cmd === '/teach') {
      const m = trimmed.match(/^\/teach\s+(.+?)\s*=\s*(.+)$/s);
      if (m) {
        if (teachPair(userId, m[1].trim(), m[2].trim())) return send(chatId, 'Запомнил.');
        return send(chatId, 'Не удалось.');
      }
      if (!arg) return send(chatId, 'Использование: /teach вопрос = ответ');
      pending.set(userId, { type: 'teach', q: arg });
      return send(chatId, `Какой ответ на "${arg}"?`);
    }

    if (cmd === '/fact') {
      const m = trimmed.match(/^\/fact\s+(.+?)\s*=\s*(.+)$/s);
      if (!m) return send(chatId, 'Использование: /fact ключ = значение');
      facts[clean(m[1])] = m[2].trim();
      saveFacts();
      return send(chatId, 'Факт сохранён.');
    }

    if (cmd === '/template') {
      const m = trimmed.match(/^\/template\s+(.+?)\s*=\s*(.+)$/s);
      if (!m) return send(chatId, 'Использование: /template шаблон с {X} = ответ с {X}');
      templates.push({ pattern: clean(m[1]), reply: m[2].trim() });
      saveTemplates();
      return send(chatId, 'Шаблон добавлен.');
    }

    if (cmd === '/text') return send(chatId, 'Пришли .txt файл.');
    if (cmd === '/import') return send(chatId, 'Пришли .json файл.');

    if (cmd === '/list') {
      if (mem.pairs.length === 0) return send(chatId, 'Пусто.');
      const PAGE = 10;
      const totalPages = Math.ceil(mem.pairs.length / PAGE);
      const p = clamp(parseInt(arg, 10) || 1, 1, totalPages);
      const slice = mem.pairs.slice((p - 1) * PAGE, p * PAGE);
      let out = `<b>Пары</b> (${p}/${totalPages}, всего ${mem.pairs.length})\n\n`;
      slice.forEach((pr, i) => {
        const num = (p - 1) * PAGE + i + 1;
        const preview = escapeHtml(pr.q.length > 60 ? pr.q.slice(0, 60) + '…' : pr.q);
        out += `<b>${num}.</b> ${preview}\n`;
      });
      if (totalPages > 1) out += `\n/list ${p + 1}`;
      return send(chatId, out, { parse_mode: 'HTML' });
    }

    if (cmd === '/del') {
      const num = parseInt(arg, 10);
      if (!num || num < 1 || num > mem.pairs.length) return send(chatId, 'Неверный номер.');
      mem.pairs.splice(num - 1, 1);
      saveUserMemory(userId);
      return send(chatId, `Удалено #${num}.`);
    }

    if (cmd === '/find') {
      if (!arg) return send(chatId, 'Использование: /find текст');
      const tokens = tokenize(arg);
      const tfidf = new TfIdf();
      mem.pairs.forEach((p, i) => tfidf.add(i, p.q, tokenize(p.q)));
      const res = tfidf.search(tokens, 10).filter(r => r.score > 0.05);
      if (res.length === 0) return send(chatId, 'Ничего не найдено.');
      let out = '<b>Найдено:</b>\n\n';
      for (const r of res) {
        const pr = mem.pairs[r.id];
        out += `<b>#${r.id + 1}</b> [${r.score.toFixed(2)}] ${escapeHtml(pr.q.slice(0, 80))}\n`;
      }
      return send(chatId, out, { parse_mode: 'HTML' });
    }

    if (cmd === '/export') {
      const file = userFile(userId);
      if (!fs.existsSync(file)) return send(chatId, 'Пусто.');
      await sendDocument(chatId, file, 'Экспорт памяти');
      return;
    }

    if (cmd === '/stats') {
      const s = {
        pairs: mem.pairs.length,
        intents: Object.keys(nb.classDocs).length,
        intentDocs: nb.totalDocs,
        vocab: nb.vocab.size,
        embVocab: Object.keys(emb.vectors).length,
        embDim: emb.dim,
        graphNodes: Object.keys(graph.edges).length,
        graphEdges: Object.values(graph.edges).reduce((s, o) => s + Object.keys(o).length, 0) / 2,
        bigrams: Object.keys(lm.bigrams).length,
        trigrams: Object.keys(lm.trigrams).length,
        facts: Object.keys(facts).length,
        templates: templates.length
      };
      let out = '<b>Модели</b>\n\n';
      out += `Пар в памяти: ${s.pairs}\n`;
      out += `Намерений: ${s.intents} (примеров: ${s.intentDocs})\n`;
      out += `Словарь Байеса: ${s.vocab}\n`;
      out += `Эмбеддинги: ${s.embVocab} слов × ${s.embDim} мер\n`;
      out += `Граф: ${s.graphNodes} узлов, ${s.graphEdges} связей\n`;
      out += `N-грамм: ${s.bigrams} биграмм, ${s.trigrams} триграмм\n`;
      out += `Фактов: ${s.facts}\n`;
      out += `Шаблонов: ${s.templates}\n`;
      return send(chatId, out, { parse_mode: 'HTML' });
    }

    if (cmd === '/similar') {
      if (!arg) return send(chatId, 'Использование: /similar слово');
      const res = emb.similar(clean(arg).split(/\s+/)[0], 10);
      if (res.length === 0) return send(chatId, 'Нет данных. Сначала обучи на тексте (/text).');
      let out = `<b>Похожие на «${escapeHtml(arg)}»:</b>\n\n`;
      res.forEach(([w, s]) => { out += `${escapeHtml(w)} — ${s.toFixed(3)}\n`; });
      return send(chatId, out, { parse_mode: 'HTML' });
    }

    if (cmd === '/continue') {
      if (!arg) return send(chatId, 'Использование: /continue начало фразы');
      const tokens = tokenize(arg);
      const result = lm.continueText(tokens, 8);
      const added = result.slice(tokens.length);
      if (added.length === 0) return send(chatId, 'Не знаю продолжения. Обучи на тексте.');
      return send(chatId, result.join(' '));
    }

    return send(chatId, 'Неизвестная команда. /help');
  }

  const p = pending.get(userId);
  if (p) {
    if (p.type === 'reset') {
      if (trimmed.toLowerCase() === 'да' || trimmed.toLowerCase() === 'yes') {
        mem.pairs = [];
        mem.history = [];
        saveUserMemory(userId);
        pending.delete(userId);
        return send(chatId, 'Память очищена.');
      }
      pending.delete(userId);
      return send(chatId, 'Отменено.');
    }
    if (p.type === 'teach') {
      pending.delete(userId);
      if (teachPair(userId, p.q, trimmed)) return send(chatId, 'Запомнил.');
      return send(chatId, 'Не удалось.');
    }
  }

  const lastBot = lastBotMsg.get(userId);
  if (lastBot && lastBot.type === 'echo') {
    lastBotMsg.delete(userId);
    if (teachPair(userId, lastBot.q, trimmed)) return send(chatId, 'Понял, записал.');
  }

  const math = tryMath(trimmed);
  if (math) { lastBotMsg.set(userId, { type: 'answer' }); return send(chatId, math); }

  const rnd = tryRandomNumber(trimmed);
  if (rnd !== null) { lastBotMsg.set(userId, { type: 'answer' }); return send(chatId, rnd); }

  const intentRes = detectIntent(trimmed);
  if (intentRes && intentReplies[intentRes.intent]) {
    const reply = pick(intentReplies[intentRes.intent]);
    lastBotMsg.set(userId, { type: 'answer' });
    return send(chatId, reply);
  }

  const fact = findFact(trimmed);
  if (fact) { lastBotMsg.set(userId, { type: 'answer' }); return send(chatId, fact); }

  const memAns = findInMemory(userId, trimmed);
  if (memAns) {
    lastBotMsg.set(userId, { type: 'answer' });
    return send(chatId, memAns);
  }

  lastBotMsg.set(userId, { type: 'echo', q: trimmed });
  return send(chatId, trimmed);
}

let offset = 0;
let polling = false;

function getUpdates() {
  if (polling) return;
  polling = true;
  const url = `https://api.telegram.org/bot${BOT_TOKEN}/getUpdates?offset=${offset}&timeout=25`;
  https.get(url, res => {
    let data = '';
    res.on('data', c => data += c);
    res.on('end', () => {
      polling = false;
      try {
        const json = JSON.parse(data);
        if (json.ok && json.result) {
          for (const u of json.result) {
            offset = u.update_id + 1;
            if (u.message) {
              handleMessage(u.message.chat.id, u.message.from.id, u.message.text, u.message.document)
                .catch(e => console.error('handle error', e.message));
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
