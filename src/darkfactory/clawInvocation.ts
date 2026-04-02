import { getLLMProvider } from '../utils/llm/factory.js';
import { OpenAIAdapter } from '../utils/llm/adapters/openai.js';
import { PipelineSpec } from './schema.js';
import { PipelineState } from '../storage/interface.js';
import { SafetyController } from './safetyController.js';
import { debugLog } from '../utils/logger.js';

/**
 * Invocation wrapper that routes payload specs to the local Claw agent model (Qwen 2.5),
 * or the active LLM provider as fallback.
 * 
 * Uses SafetyController.generateBoundaryPrompt() for scope injection
 * instead of inline prompt construction — single source of truth for safety rules.
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

  const executePrompt = `Based on the system instructions, execute the necessary task for the current step (${state.current_step}). Respond with your actions and observations.`;

  debugLog(`[ClawInvocation] Launching agent on pipeline ${state.id} step=${state.current_step} iter=${state.iteration} with ${timeoutMs}ms limit.`);

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
