import { type Tool } from "@modelcontextprotocol/sdk/types.js";

// ─── Session Save Ledger ─────────────────────────────────────

export const SESSION_SAVE_LEDGER_TOOL: Tool = {
  name: "session_save_ledger",
  description:
    "Save an immutable session log entry to the session ledger. " +
    "Use this at the END of each work session to record what was accomplished. " +
    "The ledger is append-only — entries cannot be updated or deleted. " +
    "This creates a permanent audit trail of all agent work sessions.",
  inputSchema: {
    type: "object",
    properties: {
      project: {
        type: "string",
        description: "Project identifier (e.g. 'bcba-private', 'my-app'). Used to group and filter sessions.",
      },
      conversation_id: {
        type: "string",
        description: "Unique conversation/session identifier.",
      },
      summary: {
        type: "string",
        description: "Brief summary of what was accomplished in this session.",
      },
      todos: {
        type: "array",
        items: { type: "string" },
        description: "Optional list of open TODO items remaining after this session.",
      },
      files_changed: {
        type: "array",
        items: { type: "string" },
        description: "Optional list of files created or modified during this session.",
      },
      decisions: {
        type: "array",
        items: { type: "string" },
        description: "Optional list of key decisions made during this session.",
      },
    },
    required: ["project", "conversation_id", "summary"],
  },
};

// ─── Session Save Handoff ─────────────────────────────────────

export const SESSION_SAVE_HANDOFF_TOOL: Tool = {
  name: "session_save_handoff",
  description:
    "Upsert the latest project handoff state for the next session to consume on boot. " +
    "This is the 'live context' that gets loaded when a new session starts. " +
    "Calling this replaces the previous handoff for the same project (upsert on project).",
  inputSchema: {
    type: "object",
    properties: {
      project: {
        type: "string",
        description: "Project identifier — must match the project used in session_save_ledger.",
      },
      open_todos: {
        type: "array",
        items: { type: "string" },
        description: "Current open TODO items that need attention in the next session.",
      },
      active_branch: {
        type: "string",
        description: "Git branch or context the next session should resume on.",
      },
      last_summary: {
        type: "string",
        description: "Summary of the most recent session — used for quick context recovery.",
      },
      key_context: {
        type: "string",
        description: "Free-form critical context the next session needs to know.",
      },
    },
    required: ["project"],
  },
};

// ─── Session Load Context ─────────────────────────────────────

export const SESSION_LOAD_CONTEXT_TOOL: Tool = {
  name: "session_load_context",
  description:
    "Load session context for a project using progressive context loading. " +
    "Use this at the START of a new session to recover previous work state. " +
    "Three levels available:\n" +
    "- **quick**: Just the latest project state — keywords and open TODOs (~50 tokens)\n" +
    "- **standard**: Project state plus recent session summaries and decisions (~200 tokens, recommended)\n" +
    "- **deep**: Everything — full session history with all files changed, TODOs, and decisions (~1000+ tokens)",
  inputSchema: {
    type: "object",
    properties: {
      project: {
        type: "string",
        description: "Project identifier to load context for.",
      },
      level: {
        type: "string",
        enum: ["quick", "standard", "deep"],
        description: "How much context to load: 'quick' (just TODOs), 'standard' (recommended — includes recent summaries), or 'deep' (full history). Default: standard.",
      },
    },
    required: ["project"],
  },
};

// ─── Knowledge Search ─────────────────────────────────────────

export const KNOWLEDGE_SEARCH_TOOL: Tool = {
  name: "knowledge_search",
  description:
    "Search accumulated knowledge across all sessions by keywords, category, or free text. " +
    "The knowledge base grows automatically as sessions are saved — keywords are extracted " +
    "from every ledger and handoff entry. Use this to find related past work, decisions, " +
    "and context from previous sessions.\n\n" +
    "Categories available: debugging, architecture, deployment, testing, configuration, " +
    "api-integration, data-migration, security, performance, documentation, ai-ml, " +
    "ui-frontend, resume",
  inputSchema: {
    type: "object",
    properties: {
      project: {
        type: "string",
        description: "Optional project filter. If omitted, searches across all projects.",
      },
      query: {
        type: "string",
        description: "Free-text search query. Searched against session summaries using full-text search.",
      },
      category: {
        type: "string",
        description: "Optional category filter (e.g. 'debugging', 'architecture', 'ai-ml'). " +
          "Filters results to sessions in this category.",
      },
      limit: {
        type: "integer",
        description: "Maximum results to return (default: 10, max: 50).",
        default: 10,
      },
    },
  },
};

// ─── Knowledge Forget ─────────────────────────────────────────

export const KNOWLEDGE_FORGET_TOOL: Tool = {
  name: "knowledge_forget",
  description:
    "Selectively forget (delete) accumulated knowledge entries. " +
    "Like a brain pruning bad memories — remove outdated, incorrect, or irrelevant " +
    "session entries to keep the knowledge base clean and relevant.\n\n" +
    "Forget modes:\n" +
    "- **By project**: Clear all knowledge for a specific project\n" +
    "- **By category**: Remove entries matching a category (e.g. 'debugging')\n" +
    "- **By age**: Forget entries older than N days\n" +
    "- **Full reset**: Wipe everything (requires confirm_all=true)\n\n" +
    "⚠️ This permanently deletes ledger entries. Handoff state is preserved unless explicitly cleared.",
  inputSchema: {
    type: "object",
    properties: {
      project: {
        type: "string",
        description: "Project to forget entries for. Required unless using confirm_all.",
      },
      category: {
        type: "string",
        description: "Optional: only forget entries in this category (e.g. 'debugging', 'resume').",
      },
      older_than_days: {
        type: "integer",
        description: "Optional: only forget entries older than this many days.",
      },
      clear_handoff: {
        type: "boolean",
        description: "Also clear the handoff (live state) for this project. Default: false.",
      },
      confirm_all: {
        type: "boolean",
        description: "Set to true to confirm wiping ALL entries for the project (safety flag).",
      },
      dry_run: {
        type: "boolean",
        description: "If true, only count what would be deleted without actually deleting. Default: false.",
      },
    },
  },
};

// ─── Type Guards ──────────────────────────────────────────────

export function isKnowledgeForgetArgs(
  args: unknown
): args is {
  project?: string;
  category?: string;
  older_than_days?: number;
  clear_handoff?: boolean;
  confirm_all?: boolean;
  dry_run?: boolean;
} {
  return typeof args === "object" && args !== null;
}

export function isKnowledgeSearchArgs(
  args: unknown
): args is {
  project?: string;
  query?: string;
  category?: string;
  limit?: number;
} {
  return typeof args === "object" && args !== null;
}

export function isSessionSaveLedgerArgs(
  args: unknown
): args is {
  project: string;
  conversation_id: string;
  summary: string;
  todos?: string[];
  files_changed?: string[];
  decisions?: string[];
} {
  return (
    typeof args === "object" &&
    args !== null &&
    "project" in args &&
    typeof (args as { project: string }).project === "string" &&
    "conversation_id" in args &&
    typeof (args as { conversation_id: string }).conversation_id === "string" &&
    "summary" in args &&
    typeof (args as { summary: string }).summary === "string"
  );
}

export function isSessionSaveHandoffArgs(
  args: unknown
): args is {
  project: string;
  open_todos?: string[];
  active_branch?: string;
  last_summary?: string;
  key_context?: string;
} {
  return (
    typeof args === "object" &&
    args !== null &&
    "project" in args &&
    typeof (args as { project: string }).project === "string"
  );
}

export function isSessionLoadContextArgs(
  args: unknown
): args is { project: string; level?: "quick" | "standard" | "deep" } {
  return (
    typeof args === "object" &&
    args !== null &&
    "project" in args &&
    typeof (args as { project: string }).project === "string"
  );
}
