export type DarkFactoryStep = 'INIT' | 'PLAN' | 'EXECUTE' | 'VERIFY' | 'FINALIZE';

/**
 * Defines the parameters for a Dark Factory run
 */
export interface PipelineSpec {
  /** The core objective for the autonomous pipeline to accomplish */
  objective: string;
  
  /** Maximum number of PLAN->EXECUTE->VERIFY loop iterations */
  maxIterations: number;
  
  /** The context directory where files are allowed to be modified */
  workingDirectory?: string;
  
  /** Optional files to strictly scope context */
  contextFiles?: string[];
  
  /** Optional model override to use for reasoning inside the factory (e.g. qwen3) */
  modelOverride?: string;
}

/**
 * Represents the log/result of a single iteration loop
 */
export interface IterationResult {
  iteration: number;
  step: DarkFactoryStep;
  started_at: string;
  completed_at: string;
  success: boolean;
  notes?: string;
}
