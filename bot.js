const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const BOT_TOKEN = process.env.BOT_TOKEN;
const DATA_DIR = process.env.DATA_DIR || './data';

const SIM_THRESHOLD = 0.55;
const MIN_SIM = 0.25;
const MIN_COVERAGE = 0.25;
const BM25_K1 = 1.5;
const BM25_B = 0.75;
const LIST_PAGE = 20;
const MAX_BRAINS = 200;
const DEBUG = true;

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

const health = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('OK');
});
health.listen(3000, () => console.log('[HEALTH] on 3000'));

const brains = new Map();
const brainOrder = [];
let offset = 0;

function log(...args) {
  if (DEBUG) console.log('[BOT]', ...args);
}

function lc(s) {
  return String(s).toLowerCase().replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function tokenize(s) {
  return lc(s).split(' ').filter(w => w.length >= 2);
}

function scopeOf(chatId, isGroup) {
  return isGroup ? `c:${chatId}` : `u:${chatId}`;
}

function safeName(scope) {
  return scope.replace(/[^\w]/g, '_');
}

function makeBrain(scope) {
  return {
    scope,
    pairs: [],
    syn: new Map(),
    anchors: new Set(),
    loaded: false,
    loading: null,
    df: new Map(),
    idf: new Map(),
    avgLen: 5,
    dirty: true,
    file: path.join(DATA_DIR, safeName(scope) + '.json'),
  };
}

function touchBrain(scope) {
  const i = brainOrder.indexOf(scope);
  if (i >= 0) brainOrder.splice(i, 1);
  brainOrder.push(scope);
  while (brainOrder.length > MAX_BRAINS) {
    const drop = brainOrder.shift();
    if (drop !== scope) brains.delete(drop);
  }
}

function getBrain(scope) {
  let b = brains.get(scope);
  if (!b) { b = makeBrain(scope); brains.set(scope, b); }
  touchBrain(scope);
  return b;
}

function expand(brain, word) {
  const out = new Set([word]);
  const s = brain.syn.get(word);
  if (s) for (const x of s) out.add(x);
  return out;
}

function expandAll(brain, ws) {
  const out = new Set();
  for (const w of ws) for (const v of expand(brain, w)) out.add(v);
  return out;
}

function buildSynGroups(brain) {
  const parent = new Map();
  const find = (x) => {
    if (!parent.has(x)) parent.set(x, x);
    let r = x;
    while (parent.get(r) !== r) r = parent.get(r);
    while (parent.get(x) !== r) { const n = parent.get(x); parent.set(x, r); x = n; }
    return r;
  };
  const union = (a, b) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };

  for (const [w, set] of brain.syn) {
    find(w);
    for (const s of set) {
      find(s);
      union(w, s);
    }
  }

  const clusters = new Map();
  for (const w of parent.keys()) {
    const r = find(w);
    if (!clusters.has(r)) clusters.set(r, []);
    clusters.get(r).push(w);
  }

  const groups = [];
  for (const [, members] of clusters) {
    if (members.length < 2) continue;
    const anchors = members.filter(m => brain.anchors.has(m));
    let main;
    if (anchors.length) {
      main = anchors.sort((a, b) => a.length - b.length || a.localeCompare(b))[0];
    } else {
      main = members.slice().sort((a, b) => a.length - b.length || a.localeCompare(b))[0];
    }
    groups.push({ main, members: members.slice().sort(), hasAnchor: anchors.length > 0 });
  }
  return groups;
}

function sortedSynGroups(brain) {
  const groups = buildSynGroups(brain);
  groups.sort((a, b) => {
    if (a.hasAnchor !== b.hasAnchor) return a.hasAnchor ? -1 : 1;
    return a.main.localeCompare(b.main);
  });
  return groups;
}

function buildStats(brain) {
  const df = new Map();
  let totalLen = 0;
  for (const p of brain.pairs) {
    const ws = tokenize(p.question);
    totalLen += ws.length;
    const seen = new Set();
    for (const w of ws) {
      if (!seen.has(w)) { seen.add(w); df.set(w, (df.get(w) || 0) + 1); }
    }
  }
  const N = brain.pairs.length || 1;
  const idf = new Map();
  for (const [w, c] of df) {
    idf.set(w, Math.log((N - c + 0.5) / (c + 0.5) + 1));
  }
  brain.df = df;
  brain.idf = idf;
  brain.avgLen = brain.pairs.length ? totalLen / brain.pairs.length : 5;
  log('buildStats', brain.scope, 'words:', df.size, 'pairs:', brain.pairs.length, 'avgLen:', brain.avgLen.toFixed(2));
}

function idfOf(brain, w) {
  return brain.idf.get(w) || 0;
}

function bm25(brain, querySet, pairWords) {
  if (!pairWords.length || !querySet.size) return 0;
  const dl = pairWords.length;
  const avgdl = brain.avgLen || 5;
  const counts = new Map();
  for (const w of pairWords) counts.set(w, (counts.get(w) || 0) + 1);
  let score = 0;
  for (const w of querySet) {
    const tf = counts.get(w) || 0;
    if (!tf) continue;
    const idf = idfOf(brain, w);
    score += idf * (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl));
  }
  return score;
}

function coverage(brain, queryWords, querySet, pairSet) {
  if (!queryWords.length) return 0;
  let totalW = 0;
  let foundW = 0;
  const seen = new Set();
  for (const w of queryWords) {
    if (seen.has(w)) continue;
    seen.add(w);
    const idf = idfOf(brain, w);
    if (idf <= 0) continue;
    totalW += idf;
    let hit = false;
    for (const v of expand(brain, w)) {
      if (pairSet.has(v)) { hit = true; break; }
    }
    if (hit) foundW += idf;
  }
  return totalW ? foundW / totalW : 0;
}

function rebuild(brain) {
  buildStats(brain);
  brain.dirty = false;
}

function computeScore(brain, queryWords, querySet, inputKey, pair) {
  const pWords = tokenize(pair.question);
  if (!pWords.length) return null;
  const pSet = new Set(pWords);
  const bm = bm25(brain, querySet, pWords);
  const cov = coverage(brain, queryWords, querySet, pSet);
  const exact = lc(pair.question) === inputKey ? 1 : 0;
  const idfSum = [...querySet].reduce((s, w) => s + idfOf(brain, w), 0) || 1;
  const bmNorm = bm / (bm + idfSum);
  const score = 0.6 * bmNorm + 0.3 * cov + 0.1 * exact;
  return { score, bm, bmNorm, cov, exact };
}

function answerFor(brain, input) {
  if (!brain.pairs.length) return null;
  const key = lc(input);
  const queryWords = tokenize(input);
  if (!queryWords.length) return null;
  const querySet = expandAll(brain, queryWords);

  const scored = [];
  for (let i = 0; i < brain.pairs.length; i++) {
    const s = computeScore(brain, queryWords, querySet, key, brain.pairs[i]);
    if (!s) continue;
    if (s.cov < MIN_COVERAGE && s.exact === 0) continue;
    scored.push({ idx: i, ...s });
  }

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, 3);
  log('answerFor:', input, '=>', JSON.stringify(top.map(t => ({
    q: brain.pairs[t.idx].question,
    s: Number(t.score.toFixed(4)),
    bm: Number(t.bm.toFixed(3)),
    cov: Number(t.cov.toFixed(3)),
    exact: t.exact,
  }))));

  if (!top.length) return null;
  const best = top[0];
  if (best.score >= SIM_THRESHOLD) return { answer: brain.pairs[best.idx].answer, kind: 'exact', score: best.score };
  if (best.score >= MIN_SIM) return { answer: brain.pairs[best.idx].answer, kind: 'close', score: best.score };
  return null;
}

async function persistLocal(brain) {
  const data = {
    scope: brain.scope,
    pairs: brain.pairs.map(p => ({ question: p.question, answer: p.answer, scope: p.scope })),
    syn: [...brain.syn.entries()].map(([w, set]) => [w, [...set]]),
    anchors: [...brain.anchors],
  };
  const tmp = brain.file + '.tmp';
  await fs.promises.writeFile(tmp, JSON.stringify(data), 'utf8');
  await fs.promises.rename(tmp, brain.file);
}

async function loadLocal(brain) {
  try {
    const raw = await fs.promises.readFile(brain.file, 'utf8');
    const data = JSON.parse(raw);
    if (data.scope !== brain.scope) return false;
    brain.pairs = (data.pairs || []).map(p => ({ ...p, scope: p.scope || brain.scope }));
    brain.syn = new Map();
    for (const [w, arr] of (data.syn || [])) brain.syn.set(w, new Set(arr));
    brain.anchors = new Set(data.anchors || []);
    return true;
  } catch (e) { return false; }
}

async function ensureLoaded(brain) {
  if (brain.loaded) {
    if (brain.dirty) rebuild(brain);
    return brain;
  }
  if (brain.loading) return brain.loading;
  const p = (async () => {
    try {
      const local = await loadLocal(brain);
      if (!local) {
        brain.pairs = [];
        brain.syn = new Map();
        brain.anchors = new Set();
        await persistLocal(brain);
      }
      brain.loaded = true;
      rebuild(brain);
      log('Loaded', brain.scope, 'pairs:', brain.pairs.length, 'words:', brain.df.size);
      return brain;
    } finally {
      brain.loading = null;
    }
  })();
  brain.loading = p;
  return p;
}

function api(method, payload) {
  return new Promise((resolve, reject) => {
    if (!BOT_TOKEN) return reject(new Error('no token'));
    const data = JSON.stringify(payload || {});
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${BOT_TOKEN}/${method}`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    }, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(e); } });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function send(chat, text) {
  try {
    const r = await api('sendMessage', { chat_id: chat, text });
    if (!r || !r.ok) console.error('[SEND FAIL]', JSON.stringify(r));
  } catch (e) { console.error('[SEND ERROR]', e.message); }
}

async function sendDocument(chat, filename, content) {
  return new Promise((resolve, reject) => {
    const boundary = '----bot' + Date.now();
    const head = `--${boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n${chat}\r\n--${boundary}\r\nContent-Disposition: form-data; name="document"; filename="${filename}"\r\nContent-Type: application/json\r\n\r\n`;
    const tail = `\r\n--${boundary}--\r\n`;
    const body = Buffer.concat([Buffer.from(head, 'utf8'), Buffer.from(content, 'utf8'), Buffer.from(tail, 'utf8')]);
    const req = https.request({
      hostname: 'api.telegram.org',
      path: `/bot${BOT_TOKEN}/sendDocument`,
      method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}`, 'Content-Length': body.length },
    }, (res) => {
      let b = '';
      res.on('data', c => b += c);
      res.on('end', () => { try { resolve(JSON.parse(b)); } catch (e) { reject(e); } });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function downloadFile(fileId) {
  const r = await api('getFile', { file_id: fileId });
  if (!r || !r.ok) throw new Error('getFile failed');
  const filePath = r.result.file_path;
  return new Promise((resolve, reject) => {
    https.get(`https://api.telegram.org/file/bot${BOT_TOKEN}/${filePath}`, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => resolve(body));
    }).on('error', reject);
  });
}

const HELP = [
  'Бот с поисковым движком BM25 (чистый JS, без хардкода слов).',
  '',
  '/teach вопрос = ответ',
  '/learn слово = синоним1, синоним2',
  '/delete <вопрос> — удалить пару',
  '/delete <слово> — удалить всю группу синонимов',
  '/delete <номер> [номер ...] — удалить группы из /syn по номерам',
  '/list [стр] — список пар',
  '/syn [стр] — список синонимов',
  '/syn слово — синонимы конкретного слова',
  '/stats',
  '/debug слово — что видит бот',
  '/reindex — пересчитать статистику',
  '/export — выгрузить свой мозг',
  '/import (реплай на JSON) — залить мозг',
].join('\n');

async function handle(chat, text, isGroup, replyTo) {
  const t = text.trim();
  const scope = scopeOf(chat, isGroup);
  const brain = getBrain(scope);

  try {
    if (t === '/start' || t === '/help') return send(chat, HELP);

    if (t === '/stats') {
      await ensureLoaded(brain);
      const links = [...brain.syn.values()].reduce((s, set) => s + set.size, 0);
      return send(chat, [
        `Скоуп: ${scope}`,
        `Пар: ${brain.pairs.length}`,
        `Уникальных слов: ${brain.df.size}`,
        `Синонимов: ${brain.syn.size} (связей ${links})`,
        `Якорей: ${brain.anchors.size}`,
        `Средняя длина: ${brain.avgLen.toFixed(2)}`,
      ].join('\n'));
    }

    if (t.startsWith('/debug ')) {
      await ensureLoaded(brain);
      const word = t.slice(7).trim();
      const lcW = lc(word);
      const syn = brain.syn.get(lcW);
      const df = brain.df.get(lcW) || 0;
      const idf = idfOf(brain, lcW);
      const queryWords = tokenize(word);
      const querySet = expandAll(brain, queryWords);
      const scored = [];
      for (let i = 0; i < brain.pairs.length; i++) {
        const s = computeScore(brain, queryWords, querySet, lcW, brain.pairs[i]);
        if (!s) continue;
        scored.push({ idx: i, ...s });
      }
      scored.sort((a, b) => b.score - a.score);
      const top = scored.slice(0, 5);
      const lines = [
        `Слово: "${lcW}"`,
        `Частота (df): ${df}`,
        `IDF: ${idf.toFixed(3)}`,
        `Якорь: ${brain.anchors.has(lcW)}`,
        `Синонимы: ${syn && syn.size ? [...syn].join(', ') : '—'}`,
        `Слов в запросе: ${queryWords.length}, с синонимами: ${querySet.size}`,
      ];
      for (const s of top) {
        lines.push(`  • ${brain.pairs[s.idx].question} → ${s.score.toFixed(4)} (bm=${s.bm.toFixed(3)}, cov=${s.cov.toFixed(3)}, exact=${s.exact})`);
      }
      return send(chat, lines.join('\n'));
    }

    if (t === '/reindex') {
      await ensureLoaded(brain);
      rebuild(brain);
      await persistLocal(brain);
      return send(chat, `Пересчитал. Слов: ${brain.df.size}, пар: ${brain.pairs.length}`);
    }

    if (t === '/list' || t.startsWith('/list ')) {
      await ensureLoaded(brain);
      if (!brain.pairs.length) return send(chat, 'Пусто.');
      const arg = t.slice(5).trim();
      const totalPages = Math.max(1, Math.ceil(brain.pairs.length / LIST_PAGE));
      let page = 1;
      if (arg) {
        const n = parseInt(arg, 10);
        if (!Number.isFinite(n) || n < 1) return send(chat, `Всего страниц: ${totalPages}`);
        page = Math.min(n, totalPages);
      }
      const start = (page - 1) * LIST_PAGE;
      const slice = brain.pairs.slice(start, start + LIST_PAGE);
      let out = `Стр. ${page} из ${totalPages} (всего ${brain.pairs.length})\n\n`;
      for (let i = 0; i < slice.length; i++) {
        out += `${start + i + 1}. ${slice[i].question} = ${slice[i].answer}\n`;
      }
      if (out.length > 3900) out = out.slice(0, 3900) + '...';
      return send(chat, out);
    }

    if (t === '/syn' || t.startsWith('/syn ')) {
      await ensureLoaded(brain);
      const arg = t.slice(4).trim();

      if (arg && !/^\d+$/.test(arg)) {
        const w = lc(arg);
        const group = buildSynGroups(brain).find(g => g.members.includes(w));
        if (!group) return send(chat, `У "${w}" нет синонимов.`);
        return send(chat, `${group.main} = ${group.members.filter(m => m !== group.main).join(', ')}`);
      }

      const groups = sortedSynGroups(brain);
      if (!groups.length) return send(chat, 'Синонимов нет.');
      const totalPages = Math.max(1, Math.ceil(groups.length / LIST_PAGE));
      let page = 1;
      if (arg) {
        const n = parseInt(arg, 10);
        if (!Number.isFinite(n) || n < 1) return send(chat, `Всего страниц: ${totalPages}`);
        page = Math.min(n, totalPages);
      }
      const start = (page - 1) * LIST_PAGE;
      const slice = groups.slice(start, start + LIST_PAGE);
      let out = `Синонимы. Стр. ${page} из ${totalPages} (всего ${groups.length} групп)\n\n`;
      for (let i = 0; i < slice.length; i++) {
        out += `${start + i + 1}. ${slice[i].main} = ${slice[i].members.filter(m => m !== slice[i].main).join(', ')}\n`;
      }
      if (out.length > 3900) out = out.slice(0, 3900) + '...';
      return send(chat, out);
    }

    if (t === '/export') {
      await ensureLoaded(brain);
      const data = {
        scope,
        exportedAt: new Date().toISOString(),
        pairs: brain.pairs.map(p => ({ q: p.question, a: p.answer })),
        syn: [...brain.syn.entries()].filter(([, set]) => set.size).map(([w, set]) => ({ w, s: [...set] })),
        anchors: [...brain.anchors],
      };
      const filename = `brain_${safeName(scope)}.json`;
      try { await sendDocument(chat, filename, JSON.stringify(data, null, 2)); }
      catch (e) { console.error('export', e.message); return send(chat, 'Ошибка экспорта.'); }
      return;
    }

    if (t === '/import' || t.startsWith('/import')) {
      if (!replyTo || !replyTo.document) return send(chat, 'Ответь /import на JSON-файл.');
      try {
        const content = await downloadFile(replyTo.document.file_id);
        const data = JSON.parse(content);
        const pairs = Array.isArray(data.pairs) ? data.pairs : [];
        const syn = Array.isArray(data.syn) ? data.syn : [];
        const anchors = Array.isArray(data.anchors) ? data.anchors : [];
        await ensureLoaded(brain);
        let added = 0;
        for (const p of pairs) {
          if (!p || !p.q || !p.a) continue;
          const key = lc(p.q);
          const idx = brain.pairs.findIndex(x => lc(x.question) === key);
          if (idx >= 0) continue;
          brain.pairs.push({ question: p.q, answer: p.a, scope });
          added++;
        }
        for (const item of syn) {
          if (!item || !item.w || !Array.isArray(item.s)) continue;
          const w = lc(item.w);
          if (!brain.syn.has(w)) brain.syn.set(w, new Set());
          const target = brain.syn.get(w);
          for (const s of item.s) target.add(lc(s));
        }
        for (const a of anchors) brain.anchors.add(lc(a));
        brain.dirty = true;
        rebuild(brain);
        await persistLocal(brain);
        return send(chat, `Импортировано пар: ${added}, синонимов: ${syn.length}, якорей: ${anchors.length}`);
      } catch (e) {
        console.error('import', e.message);
        return send(chat, 'Ошибка импорта: ' + e.message);
      }
    }

    if (t.startsWith('/teach')) {
      const rest = t.slice(6).trim();
      const parts = rest.split('=');
      if (parts.length < 2) return send(chat, 'Формат: /teach вопрос = ответ');
      const q = parts[0].trim();
      const a = parts.slice(1).join('=').trim();
      if (!q || !a) return send(chat, 'Пусто.');
      await ensureLoaded(brain);
      try {
        const key = lc(q);
        const idx = brain.pairs.findIndex(p => lc(p.question) === key);
        const newPair = { question: q, answer: a, scope };
        if (idx >= 0) brain.pairs[idx] = newPair;
        else brain.pairs.push(newPair);
        brain.dirty = true;
        rebuild(brain);
        await persistLocal(brain);
        return send(chat, `Запомнил: ${q} = ${a}\nПар: ${brain.pairs.length}, слов: ${brain.df.size}`);
      } catch (e) {
        console.error('teach', e.message);
        return send(chat, 'Ошибка сохранения: ' + e.message);
      }
    }

    if (t.startsWith('/delete')) {
      const rest = t.slice(7).trim();
      if (!rest) return send(chat, 'Формат: /delete <вопрос> или /delete <номер> [номер ...]');
      await ensureLoaded(brain);

      if (/^[\d\s]+$/.test(rest)) {
        const nums = rest.split(/\s+/).map(n => parseInt(n, 10)).filter(n => Number.isFinite(n) && n > 0);
        if (!nums.length) return send(chat, 'Нет номеров.');
        const groups = sortedSynGroups(brain);
        const toDelete = [];
        const seen = new Set();
        for (const n of nums) {
          if (n < 1 || n > groups.length) continue;
          const g = groups[n - 1];
          const sig = g.members.slice().sort().join('|');
          if (seen.has(sig)) continue;
          seen.add(sig);
          toDelete.push(g);
        }
        if (!toDelete.length) return send(chat, 'Номера вне диапазона.');

        let removedWords = 0;
        for (const g of toDelete) {
          for (const m of g.members) {
            brain.syn.delete(m);
            brain.anchors.delete(m);
            removedWords++;
          }
        }
        for (const [, set] of brain.syn) {
          for (const g of toDelete) {
            for (const m of g.members) set.delete(m);
          }
        }

        brain.dirty = true;
        rebuild(brain);
        await persistLocal(brain);
        const mains = toDelete.map(g => g.main).join(', ');
        return send(chat, `Удалил групп: ${toDelete.length} (${mains}). Слов убрано: ${removedWords}`);
      }

      const key = lc(rest);

      const idx = brain.pairs.findIndex(p => lc(p.question) === key);
      if (idx >= 0) {
        try {
          brain.pairs.splice(idx, 1);
          brain.dirty = true;
          rebuild(brain);
          await persistLocal(brain);
          return send(chat, `Удалил пару: ${rest}`);
        } catch (e) {
          console.error('delete pair', e.message);
          return send(chat, 'Ошибка удаления пары: ' + e.message);
        }
      }

      const groups = buildSynGroups(brain);
      const group = groups.find(g => g.members.includes(key));
      if (group) {
        for (const m of group.members) {
          brain.syn.delete(m);
          brain.anchors.delete(m);
        }
        for (const [, set] of brain.syn) {
          for (const m of group.members) set.delete(m);
        }
        brain.dirty = true;
        rebuild(brain);
        await persistLocal(brain);
        return send(chat, `Удалил группу синонимов: ${group.main} (${group.members.length} слов)`);
      }

      return send(chat, `Не нашёл: ${key}`);
    }

    if (t.startsWith('/learn')) {
      const rest = t.slice(6).trim();
      const parts = rest.split('=');
      if (parts.length < 2) return send(chat, 'Формат: /learn слово = синоним1, синоним2');
      const main = lc(parts[0].trim());
      const others = parts[1].split(',').map(s => lc(s.trim())).filter(Boolean);
      if (!main || !others.length) return send(chat, 'Пусто.');
      await ensureLoaded(brain);

      if (!brain.syn.has(main)) brain.syn.set(main, new Set());
      const sa = brain.syn.get(main);
      for (const o of others) {
        sa.add(o);
        if (!brain.syn.has(o)) brain.syn.set(o, new Set());
        brain.syn.get(o).add(main);
      }

      if (!brain.anchors.has(main)) {
        let existingAnchor = null;
        for (const a of brain.anchors) {
          if (a === main || sa.has(a)) { existingAnchor = a; break; }
        }
        if (existingAnchor) {
          brain.anchors.delete(existingAnchor);
          brain.anchors.add(main);
        } else {
          brain.anchors.add(main);
        }
      }

      try {
        brain.dirty = true;
        rebuild(brain);
        await persistLocal(brain);
        return send(chat, `Связал: ${main} ↔ ${others.join(', ')}`);
      } catch (e) {
        console.error('learn', e.message);
        return send(chat, 'Ошибка сохранения: ' + e.message);
      }
    }

    await ensureLoaded(brain);
    const res = answerFor(brain, t);
    if (!res) return send(chat, 'Не знаю. Научи: /teach вопрос = ответ');
    return send(chat, res.answer);
  } catch (e) {
    console.error('HANDLE ERROR', e);
    return send(chat, 'Ошибка: ' + e.message);
  }
}

async function poll() {
  while (true) {
    try {
      const r = await api('getUpdates', { offset, timeout: 30, allowed_updates: ['message'] });
      if (r && r.ok && Array.isArray(r.result)) {
        for (const u of r.result) {
          offset = u.update_id + 1;
          const m = u.message;
          if (!m || !m.text) continue;
          const isGroup = m.chat.type === 'group' || m.chat.type === 'supergroup';
          try { await handle(m.chat.id, m.text, isGroup, m.reply_to_message); }
          catch (e) { console.error('handle', e.message); }
        }
      } else if (r && !r.ok) {
        console.error('[POLL] telegram error', JSON.stringify(r));
      }
    } catch (e) {
      console.error('poll', e.message);
      await new Promise(r => setTimeout(r, 3000));
    }
  }
}

async function start() {
  if (!BOT_TOKEN) { console.error('BOT_TOKEN not set'); process.exit(1); }
  console.log('[START] bot polling. data dir:', DATA_DIR);
  poll();
}

start();
