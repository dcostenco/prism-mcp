/**
 * Image Captioner (v4.5)
 * ─────────────────────────────────────────────────────────────────────────────
 * PURPOSE:
 *   Fire-and-forget background pipeline that auto-captions images saved via
 *   `session_save_image`. Connects the VLM adapter → handoff visual_memory →
 *   session ledger → embedding backfill so images become semantically
 *   searchable without any changes to the tool surface area.
 *
 * PIPELINE (all async, never blocks the MCP response):
 *   1. Read vault file → base64
 *   2. Size check (Anthropic: 5MB hard cap; all others: 20MB soft cap)
 *   3. getLLMProvider().generateImageDescription(base64, mimeType, context)
 *   4. storage.updateImageCaption(project, imageId, caption)   — patches handoff
 *   5. storage.saveLedger(...)                                 — makes caption searchable
 *   6. backfillEmbeddingsAsync(project)                        — vector-indexes caption
 *
 * DESIGN DECISIONS:
 *   - `generateImageDescription` is optional on LLMProvider. If the active
 *     provider doesn't support VLM (e.g. a text-only Ollama model), captioning
 *     is skipped gracefully with a single log line.
 *   - Errors are caught and logged; they never propagate. `session_save_image`
 *     already returned successfully when this runs.
 *   - The Anthropic 5MB limit is checked before calling the API. Gemini and
 *     OpenAI accept up to ~20MB.
 *   - The ledger entry embeds image ID and path in the `summary` string because
 *     the ledger schema has no generic metadata column.
 */

import * as fs from "fs";
import * as nodePath from "path";
import { getLLMProvider } from "./llm/factory.js";
import { getStorage } from "../storage/index.js";
import { debugLog } from "./logger.js";
import { PRISM_USER_ID } from "../config.js";

// ─── Size Caps ────────────────────────────────────────────────────────────────

/** Anthropic Messages API rejects base64 image payloads > 5MB */
const ANTHROPIC_MAX_BYTES = 5 * 1024 * 1024;

/** Gemini / OpenAI accept larger images; 20MB is a conservative safe cap */
const DEFAULT_MAX_BYTES = 20 * 1024 * 1024;

// ─── MIME Type Detection ──────────────────────────────────────────────────────

const EXT_TO_MIME: Record<string, string> = {
  ".png":  "image/png",
  ".jpg":  "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif":  "image/gif",
  ".svg":  "image/svg+xml",
};

function getMimeType(filePath: string): string {
  const ext = nodePath.extname(filePath).toLowerCase();
  return EXT_TO_MIME[ext] ?? "image/png";
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Fire-and-forget wrapper. Call this from `session_save_image` after the
 * file is in the vault and the handoff has been saved. Errors are swallowed.
 *
 * @param project     Project identifier
 * @param imageId     Short UUID assigned by session_save_image (e.g. "a3f1b2c9")
 * @param vaultPath   Absolute path to the copied file in ~/.prism-mcp/media/
 * @param userContext User-provided description (passed as hint to the VLM)
 */
export function fireCaptionAsync(
  project: string,
  imageId: string,
  vaultPath: string,
  userContext: string,
): void {
  captionImageAsync(project, imageId, vaultPath, userContext).catch(err => {
    console.error(`[ImageCaptioner] Failed for [${imageId}]: ${err}`);
  });
}

// ─── Core Pipeline ────────────────────────────────────────────────────────────

async function captionImageAsync(
  project: string,
  imageId: string,
  vaultPath: string,
  userContext: string,
): Promise<void> {

  // ── Step 1: Resolve provider ─────────────────────────────────────────────
  const llm = getLLMProvider();

  if (!llm.generateImageDescription) {
    debugLog(
      `[ImageCaptioner] Active LLM provider does not support VLM — ` +
      `captioning skipped for [${imageId}]. ` +
      `Switch to Gemini, OpenAI (gpt-4o-mini+), or Anthropic to enable.`
    );
    return;
  }

  // ── Step 2: Read file + size check ───────────────────────────────────────
  if (!fs.existsSync(vaultPath)) {
    debugLog(`[ImageCaptioner] Vault file not found: ${vaultPath}`);
    return;
  }

  const fileBuffer = fs.readFileSync(vaultPath);
  const fileSizeBytes = fileBuffer.length;
  const mimeType = getMimeType(vaultPath);

  // Detect active text provider to apply the correct size cap.
  // We import getSettingSync lazily to avoid circular dependency issues.
  const { getSettingSync } = await import("../storage/configStorage.js");
  const textProvider = getSettingSync("text_provider", "gemini");
  const maxBytes = textProvider === "anthropic" ? ANTHROPIC_MAX_BYTES : DEFAULT_MAX_BYTES;

  if (fileSizeBytes > maxBytes) {
    const limitMB = (maxBytes / 1024 / 1024).toFixed(0);
    const actualMB = (fileSizeBytes / 1024 / 1024).toFixed(1);
    console.warn(
      `[ImageCaptioner] Image [${imageId}] is ${actualMB}MB, exceeding ` +
      `the ${textProvider} VLM limit (${limitMB}MB). Captioning skipped. ` +
      (textProvider === "anthropic"
        ? "Switch Embedding Provider to Gemini/OpenAI to caption larger images."
        : "Consider resizing the image.")
    );
    return;
  }

  const imageBase64 = fileBuffer.toString("base64");

  // ── Step 3: Generate caption via VLM ────────────────────────────────────
  debugLog(`[ImageCaptioner] Captioning [${imageId}] via ${textProvider}…`);

  const caption = await llm.generateImageDescription(imageBase64, mimeType, userContext);

  if (!caption || caption.trim().length === 0) {
    debugLog(`[ImageCaptioner] Empty caption returned for [${imageId}] — skipping storage.`);
    return;
  }

  debugLog(`[ImageCaptioner] Caption generated for [${imageId}]: "${caption.slice(0, 80)}…"`);

  // ── Step 4: Patch handoff visual_memory entry ─────────────────────────
  await updateHandoffCaption(project, imageId, caption, vaultPath);

  // ── Step 5: Persist as ledger entry (makes caption semantically searchable)
  // NOTE: The ledger schema has no generic metadata column, so we embed the
  // image context directly in the summary string for LLM-readable references.
  const storage = await getStorage();
  await storage.saveLedger({
    project,
    conversation_id: "vlm-captioner",
    user_id: PRISM_USER_ID,
    event_type: "learning",
    summary:
      `[Visual Memory: ${imageId}]\n` +
      `Path: ${vaultPath}\n` +
      `User description: ${userContext}\n` +
      `VLM Caption: ${caption}`,
    keywords: [`image:${imageId}`, "visual_memory", "image_caption"],
  });

  // ── Step 6: Backfill embeddings (makes caption findable via vector search)
  // ── Step 6: Embed the caption inline ────────────────────────────────
  // We embed the caption directly here rather than calling backfillEmbeddingsHandler
  // to avoid a circular import (imageCaptioner ↔ sessionMemoryHandlers).
  // We already have getLLMProvider() in scope, so the embed cost is near-zero.
  try {
    const embedText =
      `[Visual Memory: ${imageId}] Description: ${userContext}. Caption: ${caption}`;
    const embedding = await llm.generateEmbedding(embedText);

    // Find the ledger entry we just saved and patch its embedding
    const allEntries = await storage.getLedgerEntries({
      project,
      conversation_id: "vlm-captioner",
    }) as Array<{ id?: string; created_at?: string }>;

    // Sort descending and take the most recent (the one we just inserted)
    const latest = allEntries.sort((a, b) =>
      new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime()
    )[0];

    if (latest?.id) {
      await storage.patchLedger(latest.id, { embedding: JSON.stringify(embedding) });
      debugLog(`[ImageCaptioner] Caption embedded for ledger entry [${latest.id}].`);
    }
  } catch (embedErr) {
    // Non-fatal: caption still persists in the ledger as plain text and
    // will be picked up by the next project-wide backfill sweep.
    debugLog(`[ImageCaptioner] Embedding failed (will surface in next backfill): ${embedErr}`);
  }

  debugLog(`[ImageCaptioner] Pipeline complete for [${imageId}].`);
}

// ─── Handoff Patch ────────────────────────────────────────────────────────────

/**
 * Adds `caption` to the matching visual_memory entry inside the handoff JSON.
 * Uses a read-modify-write because visual_memory is embedded in the handoff
 * metadata JSON blob — there's no dedicated column to patch atomically.
 *
 * On version conflict (OCC), retries once with a fresh read. If both fail,
 * logs and returns (the ledger entry still exists as a search fallback).
 */
async function updateHandoffCaption(
  project: string,
  imageId: string,
  caption: string,
  vaultPath: string,
): Promise<void> {
  const storage = await getStorage();

  for (let attempt = 1; attempt <= 2; attempt++) {
    const context = await storage.loadContext(project, "quick", PRISM_USER_ID);
    if (!context) {
      debugLog(`[ImageCaptioner] No handoff context for "${project}" — skipping caption patch.`);
      return;
    }

    const ctx = context as any;
    const meta = ctx.metadata || {};
    const vm: any[] = meta.visual_memory || [];
    const entry = vm.find((e: any) => e.id === imageId);

    if (!entry) {
      debugLog(`[ImageCaptioner] Image [${imageId}] not found in visual_memory — skipping patch.`);
      return;
    }

    // Mutate the entry in-memory, then save back
    entry.caption = caption;
    entry.caption_path = vaultPath;
    entry.caption_at = new Date().toISOString();

    const handoffUpdate = {
      project,
      user_id: PRISM_USER_ID,
      metadata: meta,
      last_summary:       ctx.last_summary ?? null,
      pending_todo:       ctx.pending_todo ?? null,
      active_decisions:   ctx.active_decisions ?? null,
      keywords:           ctx.keywords ?? null,
      key_context:        ctx.key_context ?? null,
      active_branch:      ctx.active_branch ?? null,
    };

    const result = await storage.saveHandoff(handoffUpdate, ctx.version);

    if (result.status !== "conflict") {
      debugLog(`[ImageCaptioner] Handoff patched with caption for [${imageId}] (attempt ${attempt}).`);
      return;
    }

    // OCC conflict — retry once with fresh version
    debugLog(`[ImageCaptioner] OCC conflict patching [${imageId}], attempt ${attempt}. Retrying…`);
  }

  console.warn(
    `[ImageCaptioner] Could not patch handoff for [${imageId}] after 2 attempts. ` +
    `Caption is still saved in the ledger and will surface via semantic search.`
  );
}
