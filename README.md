# 🧠 Prism MCP

**Persistent memory for AI agents.**

A Model Context Protocol server that gives Claude, Cursor, and other AI tools a Mind Palace — long-term memory that survives across sessions, with semantic search, cognitive routing, and a visual dashboard.

[![npm](https://img.shields.io/npm/v/prism-mcp-server?color=cb0000&label=npm)](https://www.npmjs.com/package/prism-mcp-server)
[![MCP Registry](https://img.shields.io/badge/MCP_Registry-listed-00ADD8)](https://github.com/modelcontextprotocol/servers)
[![Smithery](https://img.shields.io/badge/Smithery-listed-6B4FBB)](https://smithery.ai/server/@dcostenco/prism-mcp)
[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](LICENSE)

🌐 **Translations:** [Español](docs/i18n/README_es.md) · [Français](docs/i18n/README_fr.md) · [Português](docs/i18n/README_pt.md) · [Română](docs/i18n/README_ro.md) · [Українська](docs/i18n/README_uk.md) · [Русский](docs/i18n/README_ru.md) · [Deutsch](docs/i18n/README_de.md) · [日本語](docs/i18n/README_ja.md) · [한국어](docs/i18n/README_ko.md) · [中文](docs/i18n/README_zh.md) · [العربية](docs/i18n/README_ar.md)

---

## What Prism does

### 💾 Your AI remembers across sessions
Every conversation feeds the Mind Palace. Next session, your AI agent loads the right context automatically — no re-explaining.

### 🔍 Semantic search over your history
Ask "what did I decide about the auth flow last month?" and get the answer with citations. Vector search + keyword + graph traversal.

### 🧬 Cognitive routing
Different memory types live in different stores: episodic (what happened), semantic (what's true), procedural (how to do X). The router picks where to store and where to retrieve.

### 🛡 Local-first
Free tier runs entirely on your machine — SQLite, local embedding model, no API keys, no cloud. Paid tier adds cloud sync via Synalux portal.

### ⚡ Zero-search retrieval
Holographic Reduced Representations (HRR) for instant similarity lookups without an index. ~5ms over 100K memories.

### 🌐 Multi-agent Hivemind
Multiple AI agents share the same Mind Palace. Each agent has a role (dev / qa / pm / etc.) and sees scoped context. Heartbeat + roster for coordination.

---

## Get started

```bash
# Install globally
npm install -g prism-mcp-server

# Or use npx (no install)
npx prism-mcp-server
```

Add to Claude Desktop / Cursor config:

```json
{
  "mcpServers": {
    "prism": {
      "command": "npx",
      "args": ["-y", "prism-mcp-server"]
    }
  }
}
```

That's it. Open Claude / Cursor and your AI now has memory.

More setup details in [`docs/SETUP_GEMINI.md`](docs/SETUP_GEMINI.md).

---

## How AI agents use it

| Tool | What it does |
|---|---|
| `session_load_context` | Recover prior session's state on boot |
| `session_save_ledger` | Append immutable session log entry |
| `session_save_handoff` | Save live state for the next session |
| `knowledge_search` | Semantic + keyword search over all memories |
| `query_memory_natural` | Natural-language Q&A over your Mind Palace |
| `extract_entities` | Pull people / projects / decisions from text |
| `session_synthesize_edges` | Auto-link related memories into a graph |

(35+ tools total — full TypeScript signatures in `src/tools/`. Architecture overview in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).)

---

## Plans

| | Free (local) | Paid (Synalux portal) |
|---|---|---|
| Local SQLite memory | ✅ | ✅ |
| Semantic search | ✅ (local embedding) | ✅ (cloud-backed) |
| Cross-device sync | — | ✅ |
| Hivemind multi-agent | ✅ local team | ✅ + cloud roster |
| Auto-Scholar (web research → memory) | — | ✅ |
| HRR Zero-Search retrieval | ✅ | ✅ |
| Custom domains / SSO | — | Enterprise |

The thin-client architecture: when authenticated to Synalux, prism-mcp routes through the portal for paid features. When not authenticated (or `PRISM_FORCE_LOCAL=1`), runs purely local. Same binary.

[Pricing →](https://synalux.ai/pricing)

---

## What you can build with it

- **Persistent coding assistant** that remembers your codebase, your decisions, your team's conventions
- **Research agent** that builds knowledge over time — Auto-Scholar pipeline ingests papers / docs and synthesizes
- **Clinical scribe** that retains patient context across visits (HIPAA-compliant cloud + local)
- **Customer support agent** that learns from every ticket
- **Writing assistant** that knows your voice, your prior drafts, and what you've already published

---

## Companion: Prism Coder IDE

Standalone desktop AI IDE built on Prism's memory backend. macOS / Windows. Local-first 7B model handles routine edits; Standard+ tiers route to Claude Sonnet 4; Enterprise gets Claude Opus 4.

[Download Prism Coder IDE →](https://github.com/dcostenco/prism-coder/releases/latest)

---

## 🆕 Prism as Foundation (v14.0.0)

As of v14.0.0, Prism's algorithm exports are a **stable public contract** under SemVer. External systems can port `actrActivation.ts` (ACT-R cognitive decay), `spreadingActivation.ts` (the 0.7 similarity + 0.3 activation hybrid score), `routerExperience.ts` (experience bias with `MIN_SAMPLES=5` cold-start gate), `compactionHandler.ts` (the 25KB prompt-budget cap), and `graphMetrics.ts` (warning ratios) with citations and pin a Prism version.

The first reference consumer: an audit hooks framework that ports every threshold with a `# config.ts:317` style comment. **327 tests in that framework pin the cited Prism constants** — divergence from this repo is caught automatically.

See [`docs/WOW_FEATURES.md`](docs/WOW_FEATURES.md) for the algorithm catalogue. Release notes in [`docs/releases/v14.0.0-prism-as-foundation.md`](docs/releases/v14.0.0-prism-as-foundation.md).

---

<details>
<summary>📚 Architecture, cognitive systems, and full feature catalog</summary>

**Detailed docs in this repo:**
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system architecture, memory routing, HRR
- [`docs/SETUP_GEMINI.md`](docs/SETUP_GEMINI.md) — Gemini configuration
- [`docs/self-improving-agent.md`](docs/self-improving-agent.md) — adversarial eval / anti-sycophancy
- [`docs/rfcs/`](docs/rfcs/) — design RFCs
- [`docs/releases/`](docs/releases/) — per-version release notes
- [`CHANGELOG.md`](CHANGELOG.md) — version history (v12.5 Unified Billing, v11.6 Hivemind, v11.5.1 Auto-Scholar, etc.)
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contributor guide

**The original 1933-line README is preserved in git history.** To browse the prior version (full feature catalog, Cognitive Architecture v7.8, Autonomous Cognitive OS v9.0, HRR Zero-Search, Adversarial Evaluation walkthroughs, Universal Import patterns, competitive analysis vs LangMem/MemGPT/Letta/Zep, v12.5 Unified Billing details, v11.6 Hivemind, v11.5.1 Auto-Scholar): `git show HEAD~1:README.md`.

</details>

---

## License

[BUSL-1.1](LICENSE) — Business Source License. Free for non-production use. Production use requires a Synalux subscription or commercial license. After 2 years, converts to MIT.
