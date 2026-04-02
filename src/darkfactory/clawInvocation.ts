import { getLLMProvider } from '../utils/llm/factory.js';
import { OpenAIAdapter } from '../utils/llm/adapters/openai.js';
import { PipelineSpec } from './schema.js';
import { PipelineState } from '../storage/interface.js';
import { SafetyController } from './safetyController.js';
import { debugLog } from '../utils/logger.js';

/**
 * JSON output schema instruction injected into EXECUTE step prompts.
 * Forces the LLM to respond with machine-parseable structured output
 * instead of free-form text. The runner validates this shape before
 * any actions are applied.
 */
const EXECUTE_JSON_SCHEMA = `
You MUST respond with ONLY a valid JSON object matching this schema:
{
  "actions": [
    {
      "type": "READ_FILE" | "WRITE_FILE" | "PATCH_FILE" | "RUN_TEST",
      "targetPath": "<relative path within the workspace>",
      "content": "<file content for WRITE_FILE>",
      "patch": "<patch content for PATCH_FILE>",
      "command": "<test command for RUN_TEST>"
    }
  ],
  "notes": "<optional summary of what you did>"
}

RULES:
- type MUST be one of: READ_FILE, WRITE_FILE, PATCH_FILE, RUN_TEST
- targetPath MUST be a relative path within the workspace
- Do NOT include any text outside the JSON object
- Do NOT use markdown code fences
- If you cannot complete the task, return: {"actions": [], "notes": "reason"}
`.trim();

/**
 * Invocation wrapper that routes payload specs to the local Claw agent model (Qwen 2.5),
 * or the active LLM provider as fallback.
 * 
 * Uses SafetyController.generateBoundaryPrompt() for scope injection
 * instead of inline prompt construction — single source of truth for safety rules.
 *
 * v7.3.1: EXECUTE steps request structured JSON output via EXECUTE_JSON_SCHEMA.
 *         Non-EXECUTE steps continue to use free-form text output.
 */
export async function invokeClawAgent(
  spec: PipelineSpec,
  state: PipelineState,
  timeoutMs = 120000 // 2 min default timeout for internal executions
): Promise<{ success: boolean; resultText: string }> {

  // BYOM Override: Provide path to use alternative open-source pipelines 
  // (e.g. through the OpenAI structured adapter which also points to local endpoints like Ollama / vLLM if configured)
  const llm = spec.modelOverride 
    ? new OpenAIAdapter() // Bypasses the factory to route locally
    : getLLMProvider();

  // Scope injection via SafetyController — single source of truth
  const systemPrompt = SafetyController.generateBoundaryPrompt(spec, state);

  // v7.3.1: EXECUTE steps get structured JSON output instructions
  const isExecuteStep = state.current_step === 'EXECUTE';
  const executePrompt = isExecuteStep
    ? `Based on the system instructions, execute the necessary actions for the current step (${state.current_step}).\n\n${EXECUTE_JSON_SCHEMA}`
    : `Based on the system instructions, execute the necessary task for the current step (${state.current_step}). Respond with your actions and observations.`;

  debugLog(`[ClawInvocation] Launching agent on pipeline ${state.id} step=${state.current_step} iter=${state.iteration} with ${timeoutMs}ms limit.${isExecuteStep ? ' (JSON mode)' : ''}`);

  try {
    // Timeout Promise to ensure the runner thread does not block indefinitely
    const timeboundExecution = Promise.race([
      llm.generateText(executePrompt, systemPrompt),
      new Promise<string>((_, reject) => 
        setTimeout(() => reject(new Error('LLM_EXECUTION_TIMEOUT')), timeoutMs)
      )
    ]);

    const result = await timeboundExecution;
    
    return {
      success: true,
      resultText: result
    };
  } catch (error: any) {
    debugLog(`[ClawInvocation] Exception during generation: ${error.message}`);
    return {
      success: false,
      resultText: error.message
    };
  }
}
