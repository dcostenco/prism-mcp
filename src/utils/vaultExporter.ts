import { redactSettings } from "../tools/commonHelpers.js";

/**
 * Ensures filenames are safe for all filesystems.
 */
function slugify(text: string): string {
  if (!text) return 'untitled';
  const result = text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return result || 'untitled'; // fallback if all chars were stripped
}

/**
 * Escapes YAML string values safely.
 */
function escapeYaml(text: string): string {
  if (!text) return '""';
  // YAML double-quoted scalar: escape backslashes first, then quotes,
  // then strip control characters (newlines, tabs) to prevent broken frontmatter.
  const safe = text
    .replace(/\\/g, '\\\\')       // \ → \\
    .replace(/"/g, '\\"')        // " → \\"
    .replace(/[\r\n]+/g, ' ')    // newlines → space
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, ''); // strip control chars
  return `"${safe}"`;
}

/**
 * Pre-configured `.obsidian/app.json` — turns on graph view, sets a
 * sensible folder hide list, and configures wikilink resolution. Gives
 * users a working Obsidian vault on first open instead of the default
 * blank state.
 */
const OBSIDIAN_APP_JSON = JSON.stringify({
  attachmentFolderPath: 'attachments',
  newLinkFormat: 'shortest',
  useMarkdownLinks: false,
  alwaysUpdateLinks: true,
  defaultViewMode: 'preview',
  livePreview: true,
}, null, 2);

const OBSIDIAN_GRAPH_JSON = JSON.stringify({
  collapse: true,
  search: '',
  showTags: true,
  showAttachments: false,
  hideUnresolved: true,
  showOrphans: true,
  collapseFilter: false,
  centerStrength: 0.5,
  repelStrength: 12,
  linkStrength: 1,
  lineSizeMultiplier: 1,
  fadeArrow: 0,
  textFadeMultiplier: 0,
  nodeSizeMultiplier: 1,
}, null, 2);

/**
 * Logseq config — minimal but sufficient. Logseq reads `logseq/config.edn`;
 * we ship a no-frills one so the vault renders without warnings.
 */
const LOGSEQ_CONFIG_EDN = `{:meta/version 1
 :preferred-format :markdown
 :feature/enable-block-timestamps? false
 :feature/enable-search-remove-accents? true
 :journal/page-title-format "yyyy-MM-dd"}
`;

export type VaultPkmFlavor = 'plain' | 'obsidian' | 'logseq';

/**
 * Translates a Prism JSON export payload into a flat Map of filepaths to Uint8Array/Buffer.
 * This structure is ready to be consumed by fflate or native fs writers.
 *
 * `pkmFlavor` decides what (if any) PKM sidecar config files to ship
 * alongside the markdown:
 *   'plain'    — markdown only (default; smallest output)
 *   'obsidian' — adds .obsidian/app.json + .obsidian/graph.json
 *   'logseq'   — adds logseq/config.edn
 *
 * The markdown content is identical across flavors — wikilinks +
 * frontmatter work in both. Sidecars just pre-configure the UX.
 */
export function buildVaultDirectory(
  exportData: any,
  pkmFlavor: VaultPkmFlavor = 'plain',
): Record<string, Buffer> {
  const d = exportData?.prism_export;
  if (!d) {
    throw new Error("Invalid or missing Prism memory export data.");
  }

  const vaultFiles: Record<string, Buffer> = {};
  const projectName = d.project || "Unknown_Project";

  // Helper to add files easily
  const addFile = (path: string, content: string) => {
    vaultFiles[path] = Buffer.from(content, "utf-8");
  };

  // 1. Handoff.md (Live Context)
  let handoffMd = `# Live Project State: ${projectName}\n\n`;
  handoffMd += `> Exported: ${d.exported_at || "Unknown Date"} | Version: ${d.version || "Unknown"}\n\n`;
  if (d.handoff) {
    const h = d.handoff;
    if (h.last_summary) handoffMd += `## Last Summary\n${h.last_summary}\n\n`;
    if (h.key_context) handoffMd += `## Key Context\n${h.key_context}\n\n`;
    if (h.active_branch) handoffMd += `**Active Branch:** \`${h.active_branch}\`\n\n`;
    
    if (Array.isArray(h.pending_todo) && h.pending_todo.length > 0) {
      handoffMd += `## Open TODOs\n`;
      h.pending_todo.forEach((t: string) => handoffMd += `- [ ] ${t}\n`);
      handoffMd += `\n`;
    }
  }
  addFile("Handoff.md", handoffMd);

  // 2. Settings/Config.md
  let settingsMd = `# Prism Settings\n\n| Key | Value |\n|-----|-------|\n`;
  for (const [k, v] of Object.entries(redactSettings(d.settings || {}))) {
    // Escape Markdown-table-breaking chars: backslash must come first
    // (otherwise the escape sequence we're about to write gets re-escaped),
    // then pipe and newlines. CodeQL js/incomplete-sanitization flagged
    // the prior single-pass replace as not handling backslash-prefixed
    // input.
    const safeVal = String(v)
      .replace(/\\/g, '\\\\')
      .replace(/\|/g, '\\|')
      .replace(/\r?\n/g, ' ');
    settingsMd += `| \`${k}\` | ${safeVal} |\n`;
  }
  addFile("Settings.md", settingsMd);

  // 3. Visual_Memory/Index.md
  if (Array.isArray(d.visual_memory) && d.visual_memory.length > 0) {
    let visualMd = `# Visual Memory Index\n\n`;
    for (const vm of d.visual_memory) {
      if (!vm) continue;
      const safeId = String(vm.id ?? "").substring(0, 8);
      visualMd += `- **[\`${safeId}\`]** ${vm.description || "No description"}\n`;
      visualMd += `  <small>File: \`${vm.filename || "Unknown"}\` | Date: ${vm.timestamp || "Unknown"}</small>\n`;
    }
    addFile("Visual_Memory/Index.md", visualMd);
  }

  // 4. Ledger/ and Keywords/ processing
  const keywordMentions: Record<string, { sessionName: string, path: string }[]> = {};

  // O(1) filename collision counter: key = "YYYY-MM-DD_slug", value = next suffix number
  const filenameCounters = new Map<string, number>();

  if (Array.isArray(d.ledger)) {
    for (const entry of d.ledger) {
      if (!entry) continue;
      
      const dateStr = typeof entry.created_at === "string" 
        ? entry.created_at.split("T")[0] 
        : "Unknown_Date";
      
      const sessionSlug = slugify(entry.summary ? entry.summary.substring(0, 40) : "Session");
      const baseKey = `${dateStr}_${sessionSlug}`;
      const count = filenameCounters.get(baseKey) ?? 0;
      filenameCounters.set(baseKey, count + 1);
      // First entry gets no suffix; subsequent collisions get -1, -2, …
      const filename = count === 0 ? `${baseKey}.md` : `${baseKey}-${count}.md`;
      const fullPath = `Ledger/${filename}`;

      // Extract raw keywords from entry
      const keywords: string[] = Array.isArray(entry.keywords) ? entry.keywords : [];
      
      // Build YAML Frontmatter
      let content = `---\n`;
      content += `date: ${dateStr}\n`;
      content += `type: prism-session\n`;
      content += `project: ${escapeYaml(projectName)}\n`;
      if (typeof entry.importance === "number") {
        content += `importance: ${entry.importance}\n`;
      } else {
        content += `importance: 1\n`; // default fallback
      }
      
      if (keywords.length > 0) {
        content += `tags: [${keywords.map((k: string) => escapeYaml(k)).join(", ")}]\n`;
      } else {
        content += `tags: []\n`;
      }
      content += `summary: ${escapeYaml(entry.summary || "No summary")}\n`;
      content += `---\n\n`;

      content += `# Session: ${dateStr}\n\n`;
      content += `**Summary:** ${entry.summary || "No summary"}\n\n`;

      if (Array.isArray(entry.todos) && entry.todos.length > 0) {
        content += `## Outstanding TODOs\n`;
        entry.todos.forEach((t: string) => content += `- [ ] ${t}\n`);
        content += `\n`;
      }
      
      if (Array.isArray(entry.decisions) && entry.decisions.length > 0) {
        content += `## Decisions\n`;
        entry.decisions.forEach((dec: string) => content += `- ${dec}\n`);
        content += `\n`;
      }

      if (Array.isArray(entry.files_changed) && entry.files_changed.length > 0) {
        content += `## Files Changed\n`;
        entry.files_changed.forEach((f: string) => content += `- \`${f}\`\n`);
        content += `\n`;
      }

      // Add Wikilinks for related keywords at the bottom
      if (keywords.length > 0) {
        content += `## Indexed Topics\n`;
        keywords.forEach(kw => {
          const kwSlug = slugify(kw);
          // Register for the reverse backlink index
          if (!keywordMentions[kwSlug]) {
            keywordMentions[kwSlug] = [];
          }
          keywordMentions[kwSlug].push({
            sessionName: entry.summary || filename,
            // Vault-relative path (no "../" prefix) — Obsidian [[Wikilinks]] resolve
            // from vault root, not from the current file's directory.
            path: `Ledger/${filename}`
          });

          // Embed wikilink in the ledger file
          // Using vault-relative path for Obsidian/Logseq compatibility
          content += `- [[Keywords/${kwSlug}|${kw}]]\n`;
        });
      }

      addFile(fullPath, content);
    }
  }

  // Generate the Keyword backlink pages
  for (const [kwSlug, mentions] of Object.entries(keywordMentions)) {
    let kwContent = `# Keyword: ${kwSlug}\n\n`;
    kwContent += `## Related Sessions\n`;
    
    // Deduplicate mentions just in case
    const uniqueMentions = Array.from(new Map(mentions.map(m => [m.path, m])).values());
    
    uniqueMentions.forEach(m => {
      // Create a nice human-readable display name for the link
      const displayName = m.sessionName.substring(0, 60).replace(/[\[\]|]/g, '');
      kwContent += `- [[${m.path}|${displayName}]]\n`;
    });

    addFile(`Keywords/${kwSlug}.md`, kwContent);
  }

  // PKM-specific sidecar configs. Identical markdown content across
  // flavors — only the config files differ.
  if (pkmFlavor === 'obsidian') {
    addFile('.obsidian/app.json', OBSIDIAN_APP_JSON);
    addFile('.obsidian/graph.json', OBSIDIAN_GRAPH_JSON);
  } else if (pkmFlavor === 'logseq') {
    addFile('logseq/config.edn', LOGSEQ_CONFIG_EDN);
  }

  return vaultFiles;
}
