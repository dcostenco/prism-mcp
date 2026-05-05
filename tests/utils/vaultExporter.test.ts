import { describe, it, expect } from "vitest";
import { buildVaultDirectory } from "../../src/utils/vaultExporter.js";

describe("vaultExporter", () => {
  describe("slugify edge cases", () => {
    it("handles empty or purely special-character summaries/keywords, falling back to 'untitled'", () => {
      const data = {
        prism_export: {
          project: "test",
          ledger: [
            {
              summary: "",
              keywords: ["", "@@@!!!"]
            }
          ]
        }
      };
      const vault = buildVaultDirectory(data);
      const files = Object.keys(vault);
      
      // Expected empty slugification to fall back to 'untitled'
      expect(files.some(f => f.includes("untitled.md"))).toBe(true);
      expect(files.some(f => f.includes("Keywords/untitled.md"))).toBe(true);
    });

    it("strips special characters and trims dashes", () => {
      const data = {
        prism_export: {
          project: "test",
          ledger: [
            {
              summary: "---Hello_World!!!   ",
              keywords: ["--Special@!#_Chars--"]
            }
          ]
        }
      };
      const vault = buildVaultDirectory(data);
      const files = Object.keys(vault);
      
      expect(files.some(f => f.includes("hello-world.md"))).toBe(true);
      expect(files.some(f => f.includes("Keywords/special-chars.md"))).toBe(true);
    });
  });

  describe("escapeYaml edge cases", () => {
    it("escapes quotes and newlines in project, summary, and keywords safely", () => {
      const data = {
        prism_export: {
          project: 'Project\n"Name"',
          ledger: [
            {
              summary: 'Summary\n"Text";\\',
              keywords: ['Key\n"Word"']
            }
          ]
        }
      };
      
      const vault = buildVaultDirectory(data);
      // We know there's only one ledger file, let's find it
      const ledgerFile = Object.keys(vault).find(f => f.startsWith("Ledger/"));
      expect(ledgerFile).toBeDefined();
      
      const content = vault[ledgerFile!].toString("utf-8");
      
      // Look for the escaped strings in the YAML frontmatter
      expect(content).toContain(`project: "Project \\"Name\\""`);
      expect(content).toContain(`summary: "Summary \\"Text\\";\\\\"`);
      expect(content).toContain(`tags: ["Key \\"Word\\""]`);
    });

    it("handles undefined or null yaml fields", () => {
       const data = {
        prism_export: {
          project: null as any,
          ledger: [
            {
              summary: null as any,
              keywords: null as any
            }
          ]
        }
      };

      const vault = buildVaultDirectory(data);
      const ledgerFile = Object.keys(vault).find(f => f.startsWith("Ledger/"));
      expect(ledgerFile).toBeDefined();

      const content = vault[ledgerFile!].toString("utf-8");
      expect(content).toContain(`project: "Unknown_Project"`);
      expect(content).toContain(`summary: "No summary"`);
      // null keywords is converted to empty array before escaping in the implementation
      expect(content).toContain(`tags: []`);
    });
  });

  describe("PKM flavor sidecars", () => {
    const sample = () => ({
      prism_export: {
        project: 'demo',
        ledger: [{ summary: 'one', keywords: ['alpha'] }],
      },
    });

    it("plain flavor (default) emits no PKM sidecar files", () => {
      const vault = buildVaultDirectory(sample(), 'plain');
      const files = Object.keys(vault);
      expect(files.some(f => f.startsWith('.obsidian/'))).toBe(false);
      expect(files.some(f => f.startsWith('logseq/'))).toBe(false);
    });

    it("default param is 'plain' — no sidecars without explicit flavor", () => {
      const vault = buildVaultDirectory(sample());
      const files = Object.keys(vault);
      expect(files.some(f => f.startsWith('.obsidian/'))).toBe(false);
      expect(files.some(f => f.startsWith('logseq/'))).toBe(false);
    });

    it("obsidian flavor includes .obsidian/app.json + graph.json", () => {
      const vault = buildVaultDirectory(sample(), 'obsidian');
      expect(vault['.obsidian/app.json']).toBeDefined();
      expect(vault['.obsidian/graph.json']).toBeDefined();
      // Validate the JSON parses + has the documented preview-mode key
      const app = JSON.parse(vault['.obsidian/app.json'].toString('utf-8'));
      expect(app.defaultViewMode).toBe('preview');
      expect(app.useMarkdownLinks).toBe(false); // wikilinks
    });

    it("obsidian flavor still ships the markdown content unchanged", () => {
      const plain = buildVaultDirectory(sample(), 'plain');
      const obs = buildVaultDirectory(sample(), 'obsidian');
      // Every plain-flavor markdown file should be byte-identical in obsidian flavor.
      // Only difference: obsidian adds .obsidian/* sidecars.
      const plainMd = Object.keys(plain).filter(f => !f.startsWith('.obsidian/'));
      for (const path of plainMd) {
        expect(obs[path]?.toString('utf-8')).toBe(plain[path]?.toString('utf-8'));
      }
    });

    it("logseq flavor includes logseq/config.edn", () => {
      const vault = buildVaultDirectory(sample(), 'logseq');
      expect(vault['logseq/config.edn']).toBeDefined();
      const edn = vault['logseq/config.edn'].toString('utf-8');
      expect(edn).toContain(':preferred-format :markdown');
    });

    it("logseq flavor does NOT include obsidian sidecars (and vice versa)", () => {
      const obs = buildVaultDirectory(sample(), 'obsidian');
      expect(obs['logseq/config.edn']).toBeUndefined();
      const ls = buildVaultDirectory(sample(), 'logseq');
      expect(ls['.obsidian/app.json']).toBeUndefined();
    });
  });
});
