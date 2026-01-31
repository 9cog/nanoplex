/**
 * Recursive Reasoning Engine
 * 
 * This module implements a tensor-based recursive reasoning engine that applies 
 * logical inference (PLN-style) via tensor contractions, enabling chainable, 
 * recursive pattern propagation throughout the cognitive architecture.
 */

import { 
  GgmlTensorKernel, 
  GgmlTensor, 
  NodeTensor, 
  LinkTensor 
} from './GgmlTensorKernel';
import { AtomSpaceTensorEncoder } from './AtomSpaceTensorEncoder';
import { AttentionAllocationEngine } from './AttentionAllocationEngine';

/**
 * Reasoning engine configuration
 */
export interface ReasoningEngineConfig {
  max_reasoning_depth: number;
  confidence_threshold: number;
  tensor_contraction_method: 'dot' | 'einstein' | 'multilinear';
  inference_rules: InferenceRuleConfig[];
  chain_length_limit: number;
  parallel_chains: number;
  gradient_flow_enabled: boolean;
  meta_reasoning_enabled: boolean;
}

/**
 * PLN-style inference rule configuration
 */
export interface InferenceRuleConfig {
  id: string;
  name: string;
  type: 'deduction' | 'induction' | 'abduction' | 'revision' | 'choice';
  premise_patterns: string[];
  conclusion_pattern: string;
  confidence_formula: string;
  tensor_operation: 'contraction' | 'composition' | 'transformation';
  weight_tensor_shape: number[];
}

/**
 * Reasoning chain state
 */
export interface ReasoningChain {
  id: string;
  input_nodes: string[];
  reasoning_steps: ReasoningStep[];
  current_depth: number;
  confidence_score: number;
  convergence_status: 'active' | 'converged' | 'diverged' | 'terminated';
  output_tensors: GgmlTensor[];
}

/**
 * Individual reasoning step
 */
export interface ReasoningStep {
  id: string;
  step_number: number;
  rule_applied: string;
  input_tensors: GgmlTensor[];
  output_tensor: GgmlTensor;
  confidence_change: number;
  tensor_operations: string[];
  metadata: {
    reasoning_time: number;
    memory_usage: number;
    attention_consumed: number;
  };
}

/**
 * Reasoning statistics
 */
export interface ReasoningStats {
  total_chains: number;
  active_chains: number;
  converged_chains: number;
  average_depth: number;
  total_inferences: number;
  average_confidence: number;
  tensor_memory_usage: number;
  reasoning_throughput: number;
}

/**
 * Recursive reasoning engine implementation
 */
export class RecursiveReasoningEngine {
  private kernel: GgmlTensorKernel;
  private encoder: AtomSpaceTensorEncoder;
  private attentionEngine: AttentionAllocationEngine;
  private config: ReasoningEngineConfig;
  private activeChains: Map<string, ReasoningChain>;
  private inferenceRules: Map<string, InferenceRuleConfig>;
  private reasoningHistory: ReasoningStep[];
  private tensorCache: Map<string, GgmlTensor>;

  constructor(
    kernel: GgmlTensorKernel,
    encoder: AtomSpaceTensorEncoder,
    attentionEngine: AttentionAllocationEngine,
    config: ReasoningEngineConfig
  ) {
    this.kernel = kernel;
    this.encoder = encoder;
    this.attentionEngine = attentionEngine;
    this.config = config;
    this.activeChains = new Map();
    this.inferenceRules = new Map();
    this.reasoningHistory = [];
    this.tensorCache = new Map();
    
    this.initializeInferenceRules();
  }

  /**
   * Initialize PLN-style inference rules
   */
  private initializeInferenceRules(): void {
    this.config.inference_rules.forEach(rule => {
      this.inferenceRules.set(rule.id, rule);
    });

    // Add default inference rules if none provided
    if (this.inferenceRules.size === 0) {
      this.addDefaultInferenceRules();
    }
  }

  /**
   * Add default PLN inference rules
   */
  private addDefaultInferenceRules(): void {
    // Deduction rule: A → B, A ⊢ B
    this.inferenceRules.set('deduction', {
      id: 'deduction',
      name: 'Modus Ponens',
      type: 'deduction',
      premise_patterns: ['implication(A,B)', 'A'],
      conclusion_pattern: 'B',
      confidence_formula: 'min(premise_confidences) * 0.9',
      tensor_operation: 'contraction',
      weight_tensor_shape: [2, 1]
    });

    // Induction rule: A → B, B ⊢ A (with lower confidence)
    this.inferenceRules.set('induction', {
      id: 'induction',
      name: 'Inductive Inference',
      type: 'induction',
      premise_patterns: ['implication(A,B)', 'B'],
      conclusion_pattern: 'A',
      confidence_formula: 'min(premise_confidences) * 0.6',
      tensor_operation: 'contraction',
      weight_tensor_shape: [2, 1]
    });

    // Abduction rule: A → B, B ⊢ A (explanation)
    this.inferenceRules.set('abduction', {
      id: 'abduction',
      name: 'Abductive Reasoning',
      type: 'abduction',
      premise_patterns: ['implication(A,B)', 'B'],
      conclusion_pattern: 'A',
      confidence_formula: 'min(premise_confidences) * 0.5',
      tensor_operation: 'composition',
      weight_tensor_shape: [2, 1]
    });

    // Revision rule: A, A' ⊢ A'' (belief revision)
    this.inferenceRules.set('revision', {
      id: 'revision',
      name: 'Belief Revision',
      type: 'revision',
      premise_patterns: ['A', 'A_prime'],
      conclusion_pattern: 'A_revised',
      confidence_formula: 'weighted_average(premise_confidences)',
      tensor_operation: 'transformation',
      weight_tensor_shape: [2, 1]
    });
  }

  /**
   * Start a new reasoning chain
   */
  startReasoningChain(
    inputNodes: NodeTensor[], 
    _?: string
  ): ReasoningChain {
    const chainId = this.generateChainId();
    
    const chain: ReasoningChain = {
      id: chainId,
      input_nodes: inputNodes.map(n => n.atom_id),
      reasoning_steps: [],
      current_depth: 0,
      confidence_score: 1.0,
      convergence_status: 'active',
      output_tensors: []
    };

    // Create initial reasoning tensor
    const initialTensor = this.createInitialReasoningTensor(inputNodes);
    chain.output_tensors.push(initialTensor);

    this.activeChains.set(chainId, chain);
    return chain;
  }

  /**
   * Execute one reasoning step for all active chains
   */
  executeReasoningStep(nodeTensors: NodeTensor[], linkTensors: LinkTensor[]): ReasoningStats {
    const stats = this.initializeStats();
    
    // Process each active chain
    for (const [chainId, chain] of this.activeChains) {
      if (chain.convergence_status === 'active' && chain.current_depth < this.config.max_reasoning_depth) {
        try {
          this.processReasoningChain(chain, nodeTensors, linkTensors);
          this.updateChainStats(chain, stats);
        } catch (error) {
          console.error(`Error processing reasoning chain ${chainId}:`, error);
          chain.convergence_status = 'terminated';
        }
      }
    }

    // Clean up completed chains
    this.cleanupCompletedChains();

    return this.finalizeStats(stats);
  }

  /**
   * Process a single reasoning chain
   */
  private processReasoningChain(
    chain: ReasoningChain,
    nodeTensors: NodeTensor[],
    linkTensors: LinkTensor[]
  ): void {
    // Find applicable inference rules
    const applicableRules = this.findApplicableRules(chain, nodeTensors, linkTensors);
    
    if (applicableRules.length === 0) {
      chain.convergence_status = 'converged';
      return;
    }

    // Select best rule based on attention and confidence
    const selectedRule = this.selectBestRule(applicableRules, chain);
    
    // Apply the selected rule
    const reasoningStep = this.applyInferenceRule(selectedRule, chain, nodeTensors, linkTensors);
    
    // Add step to chain
    chain.reasoning_steps.push(reasoningStep);
    chain.current_depth++;
    chain.confidence_score *= (1 - reasoningStep.confidence_change);

    // Update output tensors
    chain.output_tensors.push(reasoningStep.output_tensor);

    // Check convergence
    if (chain.confidence_score < this.config.confidence_threshold) {
      chain.convergence_status = 'diverged';
    } else if (this.checkConvergence(chain)) {
      chain.convergence_status = 'converged';
    }
  }

  /**
   * Find applicable inference rules for current chain state
   */
  private findApplicableRules(
    chain: ReasoningChain,
    nodeTensors: NodeTensor[],
    linkTensors: LinkTensor[]
  ): InferenceRuleConfig[] {
    const applicableRules: InferenceRuleConfig[] = [];
    
    for (const rule of this.inferenceRules.values()) {
      if (this.isRuleApplicable(rule, chain, nodeTensors, linkTensors)) {
        applicableRules.push(rule);
      }
    }

    return applicableRules;
  }

  /**
   * Check if an inference rule is applicable
   */
  private isRuleApplicable(
    rule: InferenceRuleConfig,
    chain: ReasoningChain,
    _nodes: NodeTensor[],
    _links: LinkTensor[]
  ): boolean {
    // Simple pattern matching for now
    // In a full implementation, this would do sophisticated pattern matching
    
    // Check if we have enough premise tensors
    if (chain.output_tensors.length < rule.premise_patterns.length) {
      return false;
    }

    // Check if rule type matches current reasoning context
    const currentStepType = chain.reasoning_steps.length > 0 ? 
      chain.reasoning_steps[chain.reasoning_steps.length - 1].rule_applied : 'initial';
    
    // Some basic compatibility rules
    if (rule.type === 'deduction' && currentStepType === 'induction') {
      return true; // Good combination
    }
    
    if (rule.type === 'abduction' && currentStepType === 'deduction') {
      return true; // Good combination
    }

    return true; // For now, allow all rules
  }

  /**
   * Select the best inference rule based on attention and confidence
   */
  private selectBestRule(
    applicableRules: InferenceRuleConfig[],
    chain: ReasoningChain
  ): InferenceRuleConfig {
    if (applicableRules.length === 1) {
      return applicableRules[0];
    }

    // Score each rule
    let bestRule = applicableRules[0];
    let bestScore = -Infinity;

    for (const rule of applicableRules) {
      let score = 0;
      
      // Prefer rules that match current confidence level
      if (rule.type === 'deduction' && chain.confidence_score > 0.8) {
        score += 0.3;
      } else if (rule.type === 'induction' && chain.confidence_score > 0.6) {
        score += 0.2;
      } else if (rule.type === 'abduction' && chain.confidence_score > 0.4) {
        score += 0.1;
      }

      // Prefer rules that haven't been used recently
      const recentUse = chain.reasoning_steps.slice(-3).some(step => step.rule_applied === rule.id);
      if (!recentUse) {
        score += 0.1;
      }

      // Add some randomness for exploration
      score += Math.random() * 0.1;

      if (score > bestScore) {
        bestScore = score;
        bestRule = rule;
      }
    }

    return bestRule;
  }

  /**
   * Apply an inference rule to create a reasoning step
   */
  private applyInferenceRule(
    rule: InferenceRuleConfig,
    chain: ReasoningChain,
    _nodes: NodeTensor[],
    _links: LinkTensor[]
  ): ReasoningStep {
    const startTime = Date.now();
    
    // Get input tensors (last N tensors from chain)
    const inputTensors = chain.output_tensors.slice(-rule.premise_patterns.length);
    
    // Apply tensor operation based on rule type
    let outputTensor: GgmlTensor;
    const tensorOperations: string[] = [];

    switch (rule.tensor_operation) {
      case 'contraction':
        outputTensor = this.applyTensorContraction(inputTensors, rule);
        tensorOperations.push('tensor_contraction');
        break;
      case 'composition':
        outputTensor = this.applyTensorComposition(inputTensors, rule);
        tensorOperations.push('tensor_composition');
        break;
      case 'transformation':
        outputTensor = this.applyTensorTransformation(inputTensors, rule);
        tensorOperations.push('tensor_transformation');
        break;
      default:
        throw new Error(`Unknown tensor operation: ${rule.tensor_operation}`);
    }

    // Calculate confidence change
    const confidenceChange = this.calculateConfidenceChange(inputTensors, outputTensor, rule);

    // Create reasoning step
    const step: ReasoningStep = {
      id: `step_${chain.id}_${chain.reasoning_steps.length}`,
      step_number: chain.reasoning_steps.length,
      rule_applied: rule.id,
      input_tensors: inputTensors,
      output_tensor: outputTensor,
      confidence_change: confidenceChange,
      tensor_operations: tensorOperations,
      metadata: {
        reasoning_time: Date.now() - startTime,
        memory_usage: outputTensor.data.length * 4,
        attention_consumed: this.calculateAttentionConsumption(inputTensors)
      }
    };

    // Add to history
    this.reasoningHistory.push(step);
    
    return step;
  }

  /**
   * Apply tensor contraction for inference
   */
  private applyTensorContraction(
    inputTensors: GgmlTensor[],
    _: InferenceRuleConfig
  ): GgmlTensor {
    if (inputTensors.length < 2) {
      throw new Error('Tensor contraction requires at least 2 input tensors');
    }

    const [tensor1, tensor2] = inputTensors;
    
    // Simple contraction - in a full implementation, this would be more sophisticated
    if (tensor1.shape.length === 1 && tensor2.shape.length === 1) {
      return this.kernel.contract(tensor1, tensor2, [[0], [0]]);
    }
    
    // Matrix multiplication for 2D tensors
    if (tensor1.shape.length === 2 && tensor2.shape.length === 2) {
      return this.kernel.matmul(tensor1, tensor2);
    }

    // Default: element-wise product
    return this.kernel.add(tensor1, tensor2);
  }

  /**
   * Apply tensor composition for inference
   */
  private applyTensorComposition(
    inputTensors: GgmlTensor[],
    _: InferenceRuleConfig
  ): GgmlTensor {
    if (inputTensors.length === 0) {
      throw new Error('Tensor composition requires input tensors');
    }

    // Create composition by weighted combination
    let result = this.kernel.clone(inputTensors[0]);
    
    for (let i = 1; i < inputTensors.length; i++) {
      const weight = 1.0 / (i + 1); // Decreasing weights
      const scaledTensor = this.scaleTensor(inputTensors[i], weight);
      result = this.kernel.add(result, scaledTensor);
    }

    return result;
  }

  /**
   * Apply tensor transformation for inference
   */
  private applyTensorTransformation(
    inputTensors: GgmlTensor[],
    _: InferenceRuleConfig
  ): GgmlTensor {
    if (inputTensors.length === 0) {
      throw new Error('Tensor transformation requires input tensors');
    }

    // Create transformation matrix
    const inputSize = inputTensors[0].data.length;
    const transformMatrix = this.kernel.createTensor([inputSize, inputSize], 'f32');
    
    // Initialize as identity matrix with some noise
    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < inputSize; j++) {
        if (i === j) {
          transformMatrix.data[i * inputSize + j] = 1.0 + (Math.random() - 0.5) * 0.1;
        } else {
          transformMatrix.data[i * inputSize + j] = (Math.random() - 0.5) * 0.05;
        }
      }
    }

    // Apply transformation
    const inputReshaped = this.kernel.reshape(inputTensors[0], [inputSize, 1]);
    const transformed = this.kernel.matmul(transformMatrix, inputReshaped);
    
    return this.kernel.reshape(transformed, [inputSize]);
  }

  /**
   * Scale tensor by a factor
   */
  private scaleTensor(tensor: GgmlTensor, scale: number): GgmlTensor {
    const scaled = this.kernel.clone(tensor);
    for (let i = 0; i < scaled.data.length; i++) {
      scaled.data[i] *= scale;
    }
    return scaled;
  }

  /**
   * Calculate confidence change from inference
   */
  private calculateConfidenceChange(
    inputTensors: GgmlTensor[],
    outputTensor: GgmlTensor,
    rule: InferenceRuleConfig
  ): number {
    // Simple confidence calculation
    const inputMagnitude = Math.sqrt(
      inputTensors.reduce((sum, tensor) => 
        sum + tensor.data.reduce((s, val) => s + val * val, 0), 0)
    );
    
    const outputMagnitude = Math.sqrt(
      outputTensor.data.reduce((sum, val) => sum + val * val, 0)
    );

    // Confidence decreases with transformation magnitude
    const transformationRatio = outputMagnitude / (inputMagnitude + 1e-8);
    
    // Rule-specific confidence factors
    const ruleFactor = rule.type === 'deduction' ? 0.9 : 
                      rule.type === 'induction' ? 0.7 : 
                      rule.type === 'abduction' ? 0.5 : 0.6;

    return (1 - transformationRatio) * ruleFactor * 0.1;
  }

  /**
   * Calculate attention consumption for reasoning step
   */
  private calculateAttentionConsumption(inputTensors: GgmlTensor[]): number {
    return inputTensors.reduce((sum, tensor) => 
      sum + tensor.data.reduce((s, val) => s + Math.abs(val), 0), 0
    ) / inputTensors.length;
  }

  /**
   * Check if reasoning chain has converged
   */
  private checkConvergence(chain: ReasoningChain): boolean {
    if (chain.reasoning_steps.length < 3) {
      return false;
    }

    // Check if recent steps have small confidence changes
    const recentSteps = chain.reasoning_steps.slice(-3);
    const avgConfidenceChange = recentSteps.reduce((sum, step) => 
      sum + Math.abs(step.confidence_change), 0) / recentSteps.length;

    return avgConfidenceChange < 0.01;
  }

  /**
   * Create initial reasoning tensor from input nodes
   */
  private createInitialReasoningTensor(inputNodes: NodeTensor[]): GgmlTensor {
    if (inputNodes.length === 0) {
      return this.kernel.createTensor([1], 'f32');
    }

    // Combine input node embeddings
    const embeddingSize = inputNodes[0].embedding.shape[0];
    const combinedTensor = this.kernel.createTensor([embeddingSize], 'f32');
    
    inputNodes.forEach(node => {
      for (let i = 0; i < embeddingSize; i++) {
        combinedTensor.data[i] += node.embedding.data[i];
      }
    });

    // Normalize
    const magnitude = Math.sqrt(combinedTensor.data.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      for (let i = 0; i < combinedTensor.data.length; i++) {
        combinedTensor.data[i] /= magnitude;
      }
    }

    return combinedTensor;
  }

  /**
   * Clean up completed reasoning chains
   */
  private cleanupCompletedChains(): void {
    const completedChains = Array.from(this.activeChains.entries()).filter(
      ([, chain]) => chain.convergence_status !== 'active'
    );

    // Keep only recent completed chains
    const cutoffTime = Date.now() - 300000; // 5 minutes
    completedChains.forEach(([chainId, chain]) => {
      const lastStepTime = chain.reasoning_steps.length > 0 ? 
        chain.reasoning_steps[chain.reasoning_steps.length - 1].metadata.reasoning_time : 0;
      
      if (lastStepTime < cutoffTime) {
        this.activeChains.delete(chainId);
      }
    });
  }

  /**
   * Initialize reasoning statistics
   */
  private initializeStats(): ReasoningStats {
    return {
      total_chains: 0,
      active_chains: 0,
      converged_chains: 0,
      average_depth: 0,
      total_inferences: 0,
      average_confidence: 0,
      tensor_memory_usage: 0,
      reasoning_throughput: 0
    };
  }

  /**
   * Update chain statistics
   */
  private updateChainStats(chain: ReasoningChain, stats: ReasoningStats): void {
    stats.total_chains++;
    
    if (chain.convergence_status === 'active') {
      stats.active_chains++;
    } else if (chain.convergence_status === 'converged') {
      stats.converged_chains++;
    }
    
    stats.average_depth += chain.current_depth;
    stats.total_inferences += chain.reasoning_steps.length;
    stats.average_confidence += chain.confidence_score;
    
    // Calculate memory usage
    chain.output_tensors.forEach(tensor => {
      stats.tensor_memory_usage += tensor.data.length * 4;
    });
  }

  /**
   * Finalize reasoning statistics
   */
  private finalizeStats(stats: ReasoningStats): ReasoningStats {
    if (stats.total_chains > 0) {
      stats.average_depth /= stats.total_chains;
      stats.average_confidence /= stats.total_chains;
    }
    
    stats.reasoning_throughput = stats.total_inferences / (this.reasoningHistory.length || 1);
    
    return stats;
  }

  /**
   * Generate unique chain ID
   */
  private generateChainId(): string {
    return `chain_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get active reasoning chains
   */
  getActiveChains(): ReasoningChain[] {
    return Array.from(this.activeChains.values());
  }

  /**
   * Get reasoning history
   */
  getReasoningHistory(): ReasoningStep[] {
    return [...this.reasoningHistory];
  }

  /**
   * Get inference rules
   */
  getInferenceRules(): InferenceRuleConfig[] {
    return Array.from(this.inferenceRules.values());
  }

  /**
   * Add custom inference rule
   */
  addInferenceRule(rule: InferenceRuleConfig): void {
    this.inferenceRules.set(rule.id, rule);
  }

  /**
   * Remove inference rule
   */
  removeInferenceRule(ruleId: string): void {
    this.inferenceRules.delete(ruleId);
  }

  /**
   * Clear reasoning history
   */
  clearHistory(): void {
    this.reasoningHistory = [];
  }

  /**
   * Reset reasoning engine
   */
  reset(): void {
    this.activeChains.clear();
    this.reasoningHistory = [];
    this.tensorCache.clear();
  }
}