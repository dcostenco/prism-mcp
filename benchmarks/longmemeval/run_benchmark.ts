/**
 * LongMemEval Benchmark Harness for Prism MCP
 * 
 * Runs the LongMemEval-S benchmark (500 questions, ICLR 2025) against
 * Prism's memory retrieval system to measure R@K and QA accuracy.
 * 
 * Uses @libsql/client (same as Prism core) for the benchmark database.
 * 
 * Usage:
 *   npx tsx benchmarks/longmemeval/run_benchmark.ts [--limit N] [--skip-qa]
 */

import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync, unlinkSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createClient } from '@libsql/client';
import crypto from 'crypto';

// ── Types ──────────────────────────────────────────────────────────────────────
interface Turn {
  role: 'user' | 'assistant';
  content: string;
  has_answer?: boolean;
}

interface EvalInstance {
  question_id: string;
  question_type: string;
  question: string;
  question_date: string;
  answer: string;
  answer_session_ids: number[];
  haystack_dates: string[];
  haystack_session_ids: number[];
  haystack_sessions: Turn[][];
}

interface RetrievalResult {
  session_idx: number;
  score: number;
  summary: string;
}

// ── Config ─────────────────────────────────────────────────────────────────────
const BENCHMARK_DIR = dirname(fileURLToPath(import.meta.url));
const DATA_FILE = join(BENCHMARK_DIR, 'data', 'longmemeval_s_cleaned.json');
const OUTPUT_DIR = join(BENCHMARK_DIR, 'results');
const DB_PATH = join(BENCHMARK_DIR, 'benchmark.db');
const TOP_K = 5;

// ── CLI args ───────────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
let LIMIT = 500;
let SKIP_QA = false;
let MODEL = 'gpt-4o-mini';

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--limit' && args[i + 1]) LIMIT = parseInt(args[i + 1]);
  if (args[i] === '--model' && args[i + 1]) MODEL = args[i + 1];
  if (args[i] === '--skip-qa') SKIP_QA = true;
}

// ── Embedding ──────────────────────────────────────────────────────────────────
async function generateEmbedding(text: string): Promise<number[]> {
  try {
    const resp = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'nomic-embed-text', prompt: text.slice(0, 8192) }),
    });
    if (resp.ok) {
      const data = await resp.json() as any;
      return data.embedding;
    }
  } catch { /* fall through */ }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('No embedding provider. Start Ollama or set OPENAI_API_KEY');
  const resp = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model: 'text-embedding-3-small', input: text.slice(0, 8192) }),
  });
  const data = await resp.json() as any;
  return data.data[0].embedding;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, nA = 0, nB = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; nA += a[i] * a[i]; nB += b[i] * b[i]; }
  return dot / (Math.sqrt(nA) * Math.sqrt(nB));
}

function embToHex(emb: number[]): string {
  const buf = Buffer.alloc(emb.length * 4);
  for (let i = 0; i < emb.length; i++) buf.writeFloatLE(emb[i], i * 4);
  return buf.toString('hex');
}

function hexToEmb(hex: string): number[] {
  const buf = Buffer.from(hex, 'hex');
  const emb: number[] = [];
  for (let i = 0; i < buf.length; i += 4) emb.push(buf.readFloatLE(i));
  return emb;
}

// ── Extract keywords ───────────────────────────────────────────────────────────
function extractKeywords(text: string): string {
  const stopWords = new Set(['this','that','with','from','have','been','would','could','should','about','their','there','where','when','what','which','will','them','then','than','they','your','more','some','also','into','just','very','like','know','want','think','make','made','does','much','well','back','even','only','come','over','such','take','most','good','each','sure','help','need','here','going','really','right']);
  const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 3 && !stopWords.has(w));
  const freq: Record<string, number> = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([w]) => w).join(' ');
}

// ── Database ───────────────────────────────────────────────────────────────────
async function createDB() {
  if (existsSync(DB_PATH)) unlinkSync(DB_PATH);
  if (existsSync(DB_PATH + '-wal')) unlinkSync(DB_PATH + '-wal');
  if (existsSync(DB_PATH + '-shm')) unlinkSync(DB_PATH + '-shm');

  const db = createClient({ url: `file:${DB_PATH}` });
  
  await db.executeMultiple(`
    CREATE TABLE IF NOT EXISTS entries (
      id TEXT PRIMARY KEY,
      session_id INTEGER NOT NULL,
      summary TEXT NOT NULL,
      keywords TEXT,
      embedding TEXT,
      created_at TEXT
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(summary, keywords, content='entries', content_rowid='rowid');
  `);

  return db;
}

// ── Ingest ─────────────────────────────────────────────────────────────────────
async function ingestSessions(
  db: ReturnType<typeof createClient> extends Promise<infer T> ? T : never,
  sessions: Turn[][],
  dates: string[],
  sessionIds: number[]
): Promise<Map<string, number[]>> {
  const embeddingMap = new Map<string, number[]>();

  // Insert entries
  for (let i = 0; i < sessions.length; i++) {
    const session = sessions[i];
    const summary = session.map(t => `${t.role}: ${t.content}`).join('\n');
    const keywords = extractKeywords(summary);
    const id = crypto.randomBytes(8).toString('hex');
    
    await db.execute({
      sql: 'INSERT INTO entries (id, session_id, summary, keywords, created_at) VALUES (?, ?, ?, ?, ?)',
      args: [id, sessionIds[i], summary, keywords, dates[i] || ''],
    });

    // FTS insert — use a simple approach with rowid
    await db.execute({
      sql: 'INSERT INTO entries_fts (rowid, summary, keywords) VALUES (?, ?, ?)',
      args: [i + 1, summary, keywords],
    });
  }

  // Generate embeddings in parallel batches
  const batchSize = 10;
  for (let i = 0; i < sessions.length; i += batchSize) {
    const batch = sessions.slice(i, i + batchSize);
    const batchIds: string[] = [];
    
    const rows = await db.execute({
      sql: 'SELECT id, summary FROM entries ORDER BY rowid LIMIT ? OFFSET ?',
      args: [batchSize, i],
    });

    const embeddings = await Promise.all(
      rows.rows.map(r => generateEmbedding((r.summary as string).slice(0, 2000)))
    );

    for (let j = 0; j < rows.rows.length; j++) {
      const id = rows.rows[j].id as string;
      const hex = embToHex(embeddings[j]);
      await db.execute({ sql: 'UPDATE entries SET embedding = ? WHERE id = ?', args: [hex, id] });
      embeddingMap.set(id, embeddings[j]);
    }
  }

  return embeddingMap;
}

// ── Retrieve ───────────────────────────────────────────────────────────────────
async function retrieve(
  db: any,
  question: string,
  topK: number,
  embeddingMap: Map<string, number[]>
): Promise<{ session_idx: number; score: number; summary: string }[]> {

  const results: Map<number, { score: number; summary: string }> = new Map();

  // 1. FTS5 search  
  const ftsTerms = question.replace(/['"?.,!]/g, '').split(/\s+/).filter(w => w.length > 3);
  if (ftsTerms.length > 0) {
    const ftsQuery = ftsTerms.slice(0, 8).join(' OR ');
    try {
      const ftsRows = await db.execute({
        sql: `SELECT e.session_id, e.summary, rank as score
              FROM entries_fts fts
              JOIN entries e ON e.rowid = fts.rowid
              WHERE entries_fts MATCH ?
              ORDER BY rank
              LIMIT ?`,
        args: [ftsQuery, topK * 3],
      });
      for (const r of ftsRows.rows) {
        const sid = r.session_id as number;
        if (!results.has(sid) || Math.abs(r.score as number) > (results.get(sid)?.score || 0)) {
          results.set(sid, { score: Math.abs(r.score as number) * 0.3, summary: r.summary as string });
        }
      }
    } catch { /* FTS may throw on weird queries */ }
  }

  // 2. Vector similarity
  const queryEmb = await generateEmbedding(question);
  
  const allRows = await db.execute('SELECT id, session_id, summary, embedding FROM entries WHERE embedding IS NOT NULL');
  
  const vectorScores: { session_id: number; score: number; summary: string }[] = [];
  for (const row of allRows.rows) {
    const id = row.id as string;
    let entryEmb: number[];
    
    if (embeddingMap.has(id)) {
      entryEmb = embeddingMap.get(id)!;
    } else {
      entryEmb = hexToEmb(row.embedding as string);
    }
    
    const sim = cosineSimilarity(queryEmb, entryEmb);
    vectorScores.push({ session_id: row.session_id as number, score: sim, summary: row.summary as string });
  }

  vectorScores.sort((a, b) => b.score - a.score);

  // Merge: vector scores are primary
  const seen = new Set<number>();
  const merged: { session_idx: number; score: number; summary: string }[] = [];

  for (const v of vectorScores) {
    if (!seen.has(v.session_id)) {
      seen.add(v.session_id);
      // Boost if also found by FTS
      const ftsBoost = results.has(v.session_id) ? 0.1 : 0;
      merged.push({ session_idx: v.session_id, score: v.score + ftsBoost, summary: v.summary });
    }
  }

  // Add FTS-only results
  for (const [sid, data] of results) {
    if (!seen.has(sid)) {
      seen.add(sid);
      merged.push({ session_idx: sid, score: data.score, summary: data.summary });
    }
  }

  merged.sort((a, b) => b.score - a.score);
  return merged.slice(0, topK);
}

// ── QA ─────────────────────────────────────────────────────────────────────────
async function generateAnswer(question: string, retrieved: RetrievalResult[]): Promise<string> {
  const context = retrieved
    .map((r, i) => `--- Session ${i + 1} ---\n${r.summary.slice(0, 3000)}`)
    .join('\n\n');

  const prompt = `Based on the following conversation history, answer the question concisely. If the answer cannot be determined, say "I don't know."

${context}

Question: ${question}
Answer:`;

  try {
    const resp = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'qwen2.5-coder:14b', prompt, stream: false, options: { num_predict: 200, temperature: 0.1 } }),
    });
    if (resp.ok) {
      const data = await resp.json() as any;
      return data.response?.trim() || "I don't know";
    }
  } catch { /* fall through */ }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return 'NO_API_KEY';
  const resp = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: 'system', content: 'Answer concisely based on conversation history. If unsure, say "I don\'t know."' },
        { role: 'user', content: prompt },
      ],
      max_tokens: 200, temperature: 0.1,
    }),
  });
  const data = await resp.json() as any;
  return data.choices?.[0]?.message?.content?.trim() || "I don't know";
}

// ── Metrics ────────────────────────────────────────────────────────────────────
function computeMetrics(results: any[]) {
  const metrics: Record<string, { total: number; hits: number; recall?: string }> = { overall: { total: 0, hits: 0 } };

  for (const r of results) {
    if (r.question_id.endsWith('_abs')) continue; // skip abstention for retrieval eval
    const hit = r.answer_session_ids.some((id: number) => r.retrieved_session_ids.includes(id));
    
    metrics.overall.total++;
    metrics.overall.hits += hit ? 1 : 0;
    
    if (!metrics[r.question_type]) metrics[r.question_type] = { total: 0, hits: 0 };
    metrics[r.question_type].total++;
    metrics[r.question_type].hits += hit ? 1 : 0;
  }

  for (const [k, v] of Object.entries(metrics)) {
    v.recall = v.total > 0 ? (v.hits / v.total * 100).toFixed(1) + '%' : 'N/A';
  }
  return metrics;
}

// ── Main ───────────────────────────────────────────────────────────────────────
async function main() {
  console.log('╔══════════════════════════════════════════════════╗');
  console.log('║  LongMemEval Benchmark — Prism MCP              ║');
  console.log('║  ICLR 2025 · 500 Questions · 5 Memory Abilities ║');
  console.log('╚══════════════════════════════════════════════════╝\n');

  const dataset: EvalInstance[] = JSON.parse(readFileSync(DATA_FILE, 'utf-8'));
  const evalSet = dataset.slice(0, LIMIT);
  console.log(`Dataset: ${evalSet.length}/${dataset.length} questions | Top-K: ${TOP_K} | QA: ${SKIP_QA ? 'SKIP' : MODEL}\n`);

  if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const hypothesisFile = join(OUTPUT_DIR, `hypothesis_${ts}.jsonl`);
  const metricsFile = join(OUTPUT_DIR, `metrics_${ts}.json`);

  const allResults: any[] = [];
  let completed = 0;
  const t0 = Date.now();

  for (const inst of evalSet) {
    completed++;
    const pct = ((completed / evalSet.length) * 100).toFixed(0);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
    console.log(`[${completed}/${evalSet.length}] ${pct}% ${elapsed}s — ${inst.question_id} (${inst.question_type})`);

    try {
      // Fresh DB per question
      const db = await createDB();

      // 1. Ingest
      process.stdout.write(`  Ingesting ${inst.haystack_sessions.length} sessions... `);
      const embMap = await ingestSessions(db, inst.haystack_sessions, inst.haystack_dates, inst.haystack_session_ids);
      console.log('✓');

      // 2. Retrieve
      process.stdout.write(`  Retrieving top-${TOP_K}... `);
      const retrieved = await retrieve(db, inst.question, TOP_K, embMap);
      const retrievedIds = retrieved.map(r => r.session_idx);
      const hit = inst.answer_session_ids.some(id => retrievedIds.includes(id));
      console.log(`${hit ? '✅' : '❌'} [${retrievedIds.join(',')}] vs [${inst.answer_session_ids.join(',')}]`);

      allResults.push({
        question_id: inst.question_id,
        question_type: inst.question_type,
        answer_session_ids: inst.answer_session_ids,
        retrieved_session_ids: retrievedIds,
        hit,
      });

      // 3. QA
      let hypothesis = 'SKIPPED';
      if (!SKIP_QA) {
        hypothesis = await generateAnswer(inst.question, retrieved);
      }
      appendFileSync(hypothesisFile, JSON.stringify({ question_id: inst.question_id, hypothesis }) + '\n');

      db.close();
    } catch (err: any) {
      console.error(`  ❌ Error: ${err.message}`);
      allResults.push({
        question_id: inst.question_id,
        question_type: inst.question_type,
        answer_session_ids: inst.answer_session_ids,
        retrieved_session_ids: [],
        hit: false,
      });
    }
  }

  // Results
  const metrics = computeMetrics(allResults);
  const totalTime = ((Date.now() - t0) / 1000).toFixed(1);

  console.log('\n' + '═'.repeat(60));
  console.log('  RESULTS — LongMemEval-S (Prism MCP)');
  console.log('═'.repeat(60));
  console.log(`  Questions: ${evalSet.length} | Time: ${totalTime}s\n`);
  console.log('  Session-Level Recall (R@5):');
  for (const [type, m] of Object.entries(metrics)) {
    console.log(`    ${type.padEnd(30)} ${m.recall} (${m.hits}/${m.total})`);
  }
  console.log('═'.repeat(60));

  writeFileSync(metricsFile, JSON.stringify({ metrics, config: { limit: LIMIT, model: MODEL, topK: TOP_K, elapsed: totalTime } }, null, 2));
  console.log(`\nSaved: ${metricsFile}`);

  // Cleanup
  if (existsSync(DB_PATH)) unlinkSync(DB_PATH);
  if (existsSync(DB_PATH + '-wal')) unlinkSync(DB_PATH + '-wal');
  if (existsSync(DB_PATH + '-shm')) unlinkSync(DB_PATH + '-shm');
}

main().catch(console.error);
