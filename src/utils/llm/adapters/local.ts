import { getSettingSync } from "../../../storage/configStorage.js";
import { debugLog } from "../../logger.js";
import type { LLMProvider } from "../provider.js";

const EMBEDDING_DIMS = 768;
const MAX_EMBEDDING_CHARS = 8000;
const DEFAULT_MODEL = "Xenova/nomic-embed-text-v1.5";
const DEFAULT_REVISION = "main";
const MODEL_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}\/[a-zA-Z0-9._-]{1,128}$/;

export class LocalEmbeddingAdapter implements LLMProvider {
  readonly loadPromise: Promise<void>;
  private pipe: ((text: string, opts: object) => Promise<unknown>) | null = null;
  private loadError: Error | null = null;

  constructor() {
    this.loadPromise = this.initPipeline();
  }

  async generateText(_prompt: string, _systemInstruction?: string): Promise<string> {
    throw new Error(
      "LocalEmbeddingAdapter does not support text generation. " +
      "It is an embedding-only provider. Configure a text provider in the Mind Palace dashboard."
    );
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!text || !text.trim()) {
      throw new Error("[LocalEmbeddingAdapter] generateEmbedding called with empty text");
    }

    let inputText = text;
    if (inputText.length > MAX_EMBEDDING_CHARS) {
      inputText = inputText.substring(0, MAX_EMBEDDING_CHARS);
      const lastSpace = inputText.lastIndexOf(" ");
      if (lastSpace > 0) inputText = inputText.substring(0, lastSpace);
    }

    await this.loadPromise;

    if (this.loadError) throw this.loadError;
    if (!this.pipe) {
      throw new Error("[LocalEmbeddingAdapter] Pipeline not initialized and no load error recorded");
    }

    const result = await this.pipe(`search_document: ${inputText}`, { pooling: "mean", normalize: true });
    const vec = Array.from((result as { data: Float32Array }).data);

    if (vec.length !== EMBEDDING_DIMS) {
      throw new Error(
        `[LocalEmbeddingAdapter] Embedding dimension mismatch: expected ${EMBEDDING_DIMS}, got ${vec.length}. ` +
        `Check the local_embedding_model setting.`
      );
    }

    return vec;
  }

  private async initPipeline(): Promise<void> {
    const model = process.env.LOCAL_EMBEDDING_MODEL ?? getSettingSync("local_embedding_model", DEFAULT_MODEL);

    if (!MODEL_ID_PATTERN.test(model) || model.includes("..")) {
      this.loadError = new Error(
        `[LocalEmbeddingAdapter] Invalid local_embedding_model: "${model}". ` +
        `Must be a HuggingFace model ID in "owner/name" format.`
      );
      return;
    }

    const hfEndpoint = process.env.HF_ENDPOINT;
    if (hfEndpoint && !hfEndpoint.includes("huggingface.co")) {
      console.warn(
        `[LocalEmbeddingAdapter] HF_ENDPOINT is set to "${hfEndpoint}" — model downloads are redirected. ` +
        `Only set if you control and trust this server.`
      );
    }

    let transformers: typeof import("@huggingface/transformers");
    try {
      transformers = await import("@huggingface/transformers");
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      this.loadError = (e as NodeJS.ErrnoException).code === "ERR_MODULE_NOT_FOUND"
        ? new Error(
            "[LocalEmbeddingAdapter] @huggingface/transformers is not installed. " +
            "Run: npm install @huggingface/transformers"
          )
        : e;
      return;
    }

    const quantized = getSettingSync("local_embedding_quantized", "true") !== "false";
    const dtype = quantized ? "q8" : "fp32";
    const revision = getSettingSync("local_embedding_revision", DEFAULT_REVISION);

    try {
      const pipelineInstance = await transformers.pipeline("feature-extraction", model, { dtype, revision } as Parameters<typeof transformers.pipeline>[2]);
      this.pipe = pipelineInstance as (text: string, opts: object) => Promise<unknown>;

      try {
        await this.pipe("warmup text", { pooling: "mean", normalize: true });
        debugLog(`[LocalEmbeddingAdapter] Pipeline ready and warmed up: ${model} (${dtype})`);
      } catch (warmupErr) {
        const we = warmupErr instanceof Error ? warmupErr : new Error(String(warmupErr));
        console.warn(
          `[LocalEmbeddingAdapter] Warmup failed (non-fatal): ${we.message}. ` +
          `First embedding call may be slightly slower.`
        );
      }
    } catch (err) {
      this.loadError = err instanceof Error ? err : new Error(String(err));
      console.error(`[LocalEmbeddingAdapter] Failed to load pipeline: ${this.loadError.message}`);
    }
  }
}
