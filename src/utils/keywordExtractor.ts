/**
 * Keyword Extractor — Lightweight Auto-Categorization
 *
 * Extracts meaningful keywords from session summaries using pure text processing.
 * Zero external dependencies, ~1ms execution time.
 *
 * How it works:
 *   1. Normalize text (lowercase, strip punctuation)
 *   2. Remove common English stopwords
 *   3. Score remaining terms by frequency
 *   4. Assign category tags based on keyword → category mappings
 *   5. Return top keywords + matched categories
 *
 * This powers the "knowledge accumulation" feature:
 *   - Keywords are saved alongside every ledger/handoff entry
 *   - The knowledge_search tool queries these keywords using PostgreSQL GIN indexes
 *   - Similar sessions naturally cluster by shared keywords
 */

// ─── Stopwords ───────────────────────────────────────────────────
// Common English words that carry no semantic value for categorization.
// Keeping this list concise but effective.

const STOPWORDS = new Set([
  // Articles & determiners
  "a", "an", "the", "this", "that", "these", "those",
  // Pronouns
  "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
  "she", "her", "it", "its", "they", "them", "their",
  // Prepositions
  "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about",
  "into", "through", "during", "before", "after", "above", "below", "between",
  "under", "over",
  // Conjunctions
  "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
  // Verbs (common/auxiliary)
  "is", "am", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "shall",
  "should", "may", "might", "must", "can", "could",
  "get", "got", "make", "made", "let",
  // Common action words too generic to categorize
  "use", "used", "using", "set", "add", "added", "adding",
  "create", "created", "creating", "update", "updated", "updating",
  "change", "changed", "work", "worked", "working",
  // Misc
  "not", "no", "also", "just", "only", "more", "most", "very", "well",
  "now", "then", "here", "there", "when", "where", "how", "what", "which",
  "who", "all", "each", "every", "any", "some", "new", "first", "last",
  "next", "one", "two", "three",
  // Agent-specific noise
  "session", "project", "file", "files", "code", "based", "need", "needed",
  "ensure", "implement", "implemented", "implementation",
]);

// ─── Category Mappings ───────────────────────────────────────────
// Each category is defined by signal words that, when found in text,
// indicate the session falls into that category.

const CATEGORY_SIGNALS: Record<string, string[]> = {
  debugging: [
    "bug", "fix", "error", "issue", "debug", "crash", "broken", "fail",
    "failure", "exception", "stack", "trace", "stacktrace", "troubleshoot",
    "diagnose", "regression",
  ],
  architecture: [
    "architecture", "design", "pattern", "refactor", "restructure", "module",
    "component", "layer", "abstraction", "separation", "microservice",
    "monolith", "schema", "model", "entity",
  ],
  deployment: [
    "deploy", "deployment", "ci", "cd", "pipeline", "docker", "container",
    "kubernetes", "k8s", "release", "staging", "production", "build",
    "bundle", "publish",
  ],
  testing: [
    "test", "testing", "spec", "assert", "mock", "stub", "coverage",
    "unit", "integration", "e2e", "benchmark", "validate", "verification",
  ],
  configuration: [
    "config", "configuration", "env", "environment", "setup", "install",
    "dependency", "dependencies", "package", "version", "migrate",
    "migration", "settings", "preferences",
  ],
  "api-integration": [
    "api", "endpoint", "rest", "graphql", "webhook", "oauth", "auth",
    "authentication", "authorization", "token", "request", "response",
    "fetch", "client", "server", "http", "grpc",
  ],
  "data-migration": [
    "data", "database", "sql", "query", "table", "column", "index",
    "migration", "seed", "backup", "restore", "etl", "transform",
    "import", "export", "csv", "json",
  ],
  security: [
    "security", "vulnerability", "encrypt", "encryption", "ssl", "tls",
    "certificate", "firewall", "cors", "xss", "csrf", "injection",
    "sanitize", "permission", "role", "access",
  ],
  performance: [
    "performance", "optimize", "optimization", "cache", "caching", "latency",
    "throughput", "memory", "leak", "profiling", "bottleneck", "slow",
    "speed", "fast", "efficient",
  ],
  documentation: [
    "documentation", "docs", "readme", "guide", "tutorial", "comment",
    "jsdoc", "typedoc", "swagger", "openapi", "changelog",
  ],
  "ai-ml": [
    "ai", "ml", "llm", "gpt", "model", "embedding", "vector", "prompt",
    "inference", "training", "fine-tune", "rag", "agent", "mcp",
    "gemini", "openai", "anthropic", "claude",
  ],
  "ui-frontend": [
    "ui", "ux", "frontend", "css", "html", "react", "vue", "angular",
    "component", "layout", "responsive", "animation", "style", "theme",
    "dark", "light", "mobile", "desktop",
  ],
  resume: [
    "resume", "cv", "job", "position", "role", "application", "hiring",
    "interview", "candidate", "qualification", "experience", "skill",
  ],
};

// ─── Public API ──────────────────────────────────────────────────

export interface ExtractionResult {
  /** Top extracted keywords, sorted by frequency (max 15) */
  keywords: string[];
  /** Matched categories based on signal words */
  categories: string[];
}

/**
 * Extract keywords and categories from text.
 *
 * @param text - The text to analyze (typically a session summary)
 * @returns Keywords and categories — both as string arrays suitable for TEXT[]
 *
 * Performance: ~0.5ms for a 500-word summary on a modern machine.
 */
export function extractKeywords(text: string): ExtractionResult {
  if (!text || typeof text !== "string") {
    return { keywords: [], categories: [] };
  }

  // Step 1: Normalize — lowercase, strip non-alphanumeric (keep hyphens for compound words)
  const normalized = text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  // Step 2: Tokenize and filter stopwords
  const words = normalized.split(" ").filter(
    (w) => w.length > 2 && !STOPWORDS.has(w)
  );

  // Step 3: Count term frequency
  const freq = new Map<string, number>();
  for (const word of words) {
    freq.set(word, (freq.get(word) || 0) + 1);
  }

  // Step 4: Sort by frequency, take top 15
  const keywords = [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([word]) => word);

  // Step 5: Match categories based on signal words present in the full word set
  const wordSet = new Set(words);
  const categories: string[] = [];

  for (const [category, signals] of Object.entries(CATEGORY_SIGNALS)) {
    const matchCount = signals.filter((s) => wordSet.has(s)).length;
    // Require at least 2 signal matches to assign a category (reduces noise)
    if (matchCount >= 2) {
      categories.push(category);
    }
  }

  // If no category matched with 2+ signals, try with 1 match as fallback
  if (categories.length === 0) {
    for (const [category, signals] of Object.entries(CATEGORY_SIGNALS)) {
      if (signals.some((s) => wordSet.has(s))) {
        categories.push(category);
        break; // Take only the first single-signal match
      }
    }
  }

  return { keywords, categories };
}

/**
 * Combine keywords and categories into a single TEXT[] array for storage.
 * Categories are prefixed with "cat:" to distinguish them from regular keywords.
 *
 * Example output: ["typescript", "supabase", "handler", "cat:api-integration", "cat:testing"]
 */
export function toKeywordArray(text: string): string[] {
  const { keywords, categories } = extractKeywords(text);
  return [
    ...keywords,
    ...categories.map((c) => `cat:${c}`),
  ];
}
