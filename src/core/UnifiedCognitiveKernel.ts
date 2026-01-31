/**
 * Unified Cognitive Kernel
 * 
 * This module integrates all tensor-based cognitive subsystems into a unified
 * agentic cognitive kernel optimized for neural-symbolic integration, recursive
 * reasoning, and adaptive attention allocation.
 */

import { GgmlTensorKernel, AtomSpaceTensorConfig, NodeTensor, LinkTensor } from './GgmlTensorKernel';
import { AtomSpaceTensorEncoder } from './AtomSpaceTensorEncoder';
import { AttentionAllocationEngine, AttentionAllocationConfig, AttentionStats } from './AttentionAllocationEngine';
import { RecursiveReasoningEngine, ReasoningEngineConfig, ReasoningStats } from './RecursiveReasoningEngine';
import { MetaCognitiveFeedbackEngine, MetaCognitiveConfig, MetaCognitiveMetrics } from './MetaCognitiveFeedbackEngine';
import { AtomSpaceAtom, AtomSpaceLink } from '../hooks/useEnhancedAtomSpace';
import { AtomeseNode } from '../types';

/**
 * Unified cognitive kernel configuration
 */
export interface CognitiveKernelConfig {
  tensor_config: AtomSpaceTensorConfig;
  attention_config: AttentionAllocationConfig;
  reasoning_config: ReasoningEngineConfig;
  metacognitive_config: MetaCognitiveConfig;
  integration_frequency: number;
  performance_monitoring: boolean;
  adaptive_scaling: boolean;
  tensor_memory_limit: number;
}

/**
 * Cognitive kernel state
 */
export interface CognitiveKernelState {
  status: 'initializing' | 'active' | 'paused' | 'error' | 'shutdown';
  uptime: number;
  cycle_count: number;
  last_cycle_time: number;
  performance_metrics: PerformanceMetrics;
  error_count: number;
  last_error?: string;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  cycles_per_second: number;
  tensor_operations_per_second: number;
  memory_efficiency: number;
  attention_allocation_efficiency: number;
  reasoning_throughput: number;
  metacognitive_overhead: number;
  overall_efficiency: number;
}

/**
 * Cognitive kernel statistics
 */
export interface CognitiveKernelStats {
  state: CognitiveKernelState;
  attention_stats: AttentionStats;
  reasoning_stats: ReasoningStats;
  metacognitive_metrics: MetaCognitiveMetrics;
  tensor_stats: {
    total_tensors: number;
    memory_usage: number;
    active_nodes: number;
    active_links: number;
  };
}

/**
 * Processing result
 */
export interface ProcessingResult {
  success: boolean;
  cycle_time: number;
  tensors_processed: number;
  attention_updates: number;
  reasoning_steps: number;
  metacognitive_adaptations: number;
  errors: string[];
}

/**
 * Unified cognitive kernel implementation
 */
export class UnifiedCognitiveKernel {
  private config: CognitiveKernelConfig;
  private state: CognitiveKernelState;
  
  // Core components
  private tensorKernel: GgmlTensorKernel;
  private encoder: AtomSpaceTensorEncoder;
  private attentionEngine: AttentionAllocationEngine;
  private reasoningEngine: RecursiveReasoningEngine;
  private metaCognitiveEngine: MetaCognitiveFeedbackEngine;
  
  // Processing state
  private processingInterval?: NodeJS.Timeout;
  private lastCycleTime: number;
  private performanceHistory: PerformanceMetrics[];
  
  // Event callbacks
  private onCycleComplete?: (stats: CognitiveKernelStats) => void;
  private onError?: (error: string) => void;
  private onStateChange?: (state: CognitiveKernelState) => void;

  constructor(config: CognitiveKernelConfig) {
    this.config = config;
    this.lastCycleTime = Date.now();
    this.performanceHistory = [];
    
    this.state = {
      status: 'initializing',
      uptime: 0,
      cycle_count: 0,
      last_cycle_time: 0,
      performance_metrics: this.initializePerformanceMetrics(),
      error_count: 0
    };
    
    this.initializeComponents();
  }

  /**
   * Initialize all cognitive kernel components
   */
  private initializeComponents(): void {
    try {
      // Initialize tensor kernel
      this.tensorKernel = new GgmlTensorKernel(this.config.tensor_config);
      
      // Initialize encoder
      this.encoder = new AtomSpaceTensorEncoder(this.config.tensor_config);
      
      // Initialize attention engine
      this.attentionEngine = new AttentionAllocationEngine(
        this.tensorKernel,
        this.encoder,
        this.config.attention_config
      );
      
      // Initialize reasoning engine
      this.reasoningEngine = new RecursiveReasoningEngine(
        this.tensorKernel,
        this.encoder,
        this.attentionEngine,
        this.config.reasoning_config
      );
      
      // Initialize meta-cognitive engine
      this.metaCognitiveEngine = new MetaCognitiveFeedbackEngine(
        this.tensorKernel,
        this.attentionEngine,
        this.reasoningEngine,
        this.config.metacognitive_config
      );
      
      this.state.status = 'active';
      this.notifyStateChange();
      
    } catch (error) {
      this.handleError(`Failed to initialize cognitive kernel: ${error}`);
    }
  }

  /**
   * Start the cognitive kernel processing loop
   */
  start(): void {
    if (this.state.status !== 'active') {
      this.handleError('Cannot start kernel - not in active state');
      return;
    }

    const cycleInterval = 1000 / this.config.integration_frequency;
    
    this.processingInterval = setInterval(() => {
      this.processCycle();
    }, cycleInterval);
    
    console.log(`Cognitive kernel started with ${this.config.integration_frequency} Hz processing frequency`);
  }

  /**
   * Stop the cognitive kernel processing loop
   */
  stop(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = undefined;
    }
    
    this.state.status = 'paused';
    this.notifyStateChange();
    
    console.log('Cognitive kernel stopped');
  }

  /**
   * Process a single cognitive cycle
   */
  private processCycle(): void {
    const cycleStart = Date.now();
    
    try {
      // Get current tensor data
      const nodeTensors = this.encoder.getNodeTensors();
      const linkTensors = this.encoder.getLinkTensors();
      
      // Step 1: Update attention allocation
      const attentionStats = this.attentionEngine.updateAttentionAllocation(nodeTensors, linkTensors);
      
      // Step 2: Execute reasoning step
      const reasoningStats = this.reasoningEngine.executeReasoningStep(nodeTensors, linkTensors);
      
      // Step 3: Update meta-cognitive feedback
      const metacognitiveMetrics = this.metaCognitiveEngine.updateMetaCognitive(
        nodeTensors,
        linkTensors,
        attentionStats,
        reasoningStats
      );
      
      // Step 4: Update performance metrics
      const cycleTime = Date.now() - cycleStart;
      this.updatePerformanceMetrics(cycleTime, nodeTensors.length, linkTensors.length);
      
      // Step 5: Update state
      this.state.cycle_count++;
      this.state.last_cycle_time = cycleTime;
      this.state.uptime = Date.now() - this.lastCycleTime;
      
      // Step 6: Check for adaptive scaling
      if (this.config.adaptive_scaling) {
        this.performAdaptiveScaling(attentionStats, reasoningStats, metacognitiveMetrics);
      }
      
      // Step 7: Notify cycle completion
      if (this.onCycleComplete) {
        const stats = this.getCognitiveKernelStats();
        this.onCycleComplete(stats);
      }
      
    } catch (error) {
      this.handleError(`Processing cycle failed: ${error}`);
    }
  }

  /**
   * Process AtomSpace atoms through the cognitive kernel
   */
  processAtoms(atoms: AtomSpaceAtom[], links: AtomSpaceLink[]): ProcessingResult {
    const processingStart = Date.now();
    const errors: string[] = [];
    let tensorsProcessed = 0;
    let attentionUpdates = 0;
    let reasoningSteps = 0;
    let metacognitiveAdaptations = 0;

    try {
      // Encode atoms as tensors
      const nodeTensors = atoms.map(atom => {
        try {
          tensorsProcessed++;
          return this.encoder.encodeAtom(atom);
        } catch (error) {
          errors.push(`Failed to encode atom ${atom.id}: ${error}`);
          return null;
        }
      }).filter(Boolean);

      // Encode links as tensors
      const linkTensors = links.map(link => {
        try {
          tensorsProcessed++;
          return this.encoder.encodeLink(link);
        } catch (error) {
          errors.push(`Failed to encode link ${link.id}: ${error}`);
          return null;
        }
      }).filter(Boolean);

      // Process through attention system
      if (nodeTensors.length > 0) {
        this.attentionEngine.updateAttentionAllocation(
          nodeTensors as NodeTensor[], 
          linkTensors as LinkTensor[]
        );
        attentionUpdates = nodeTensors.length;
      }

      // Process through reasoning system
      if (nodeTensors.length > 0) {
        const reasoningStats = this.reasoningEngine.executeReasoningStep(
          nodeTensors as NodeTensor[], 
          linkTensors as LinkTensor[]
        );
        reasoningSteps = reasoningStats.total_inferences;
      }

      // Process through meta-cognitive system
      if (nodeTensors.length > 0) {
        const attentionStats = this.attentionEngine.getCurrentStats(nodeTensors as NodeTensor[]);
        const reasoningStats = this.reasoningEngine.getActiveChains().length > 0 ? 
          { active_chains: this.reasoningEngine.getActiveChains().length, total_chains: 1, 
            average_confidence: 0.8, reasoning_throughput: 0.5, total_inferences: reasoningSteps } :
          { active_chains: 0, total_chains: 0, average_confidence: 0, reasoning_throughput: 0, total_inferences: 0 };
        
        const metacognitiveMetrics = this.metaCognitiveEngine.updateMetaCognitive(
          nodeTensors as NodeTensor[],
          linkTensors as LinkTensor[],
          attentionStats,
          reasoningStats as ReasoningStats
        );
        metacognitiveAdaptations = Math.floor(metacognitiveMetrics.adaptation_rate * 10);
      }

    } catch (error) {
      errors.push(`Processing failed: ${error}`);
    }

    const processingTime = Date.now() - processingStart;

    return {
      success: errors.length === 0,
      cycle_time: processingTime,
      tensors_processed: tensorsProcessed,
      attention_updates: attentionUpdates,
      reasoning_steps: reasoningSteps,
      metacognitive_adaptations: metacognitiveAdaptations,
      errors
    };
  }

  /**
   * Process Atomese nodes through the cognitive kernel
   */
  processAtomeseNodes(nodes: AtomeseNode[]): ProcessingResult {
    const processingStart = Date.now();
    const errors: string[] = [];
    let tensorsProcessed = 0;

    try {
      // Encode Atomese nodes as tensors
      const nodeTensors = nodes.map(node => {
        try {
          tensorsProcessed++;
          return this.encoder.encodeAtomeseNode(node);
        } catch (error) {
          errors.push(`Failed to encode Atomese node ${node.id}: ${error}`);
          return null;
        }
      }).filter(Boolean);

      // Process through cognitive systems
      if (nodeTensors.length > 0) {
        const attentionStats = this.attentionEngine.updateAttentionAllocation(
          nodeTensors as NodeTensor[], 
          []
        );
        
        const reasoningStats = this.reasoningEngine.executeReasoningStep(
          nodeTensors as NodeTensor[], 
          []
        );
        
        this.metaCognitiveEngine.updateMetaCognitive(
          nodeTensors as NodeTensor[],
          [],
          attentionStats,
          reasoningStats
        );
      }

    } catch (error) {
      errors.push(`Atomese processing failed: ${error}`);
    }

    const processingTime = Date.now() - processingStart;

    return {
      success: errors.length === 0,
      cycle_time: processingTime,
      tensors_processed: tensorsProcessed,
      attention_updates: tensorsProcessed,
      reasoning_steps: 0,
      metacognitive_adaptations: 0,
      errors
    };
  }

  /**
   * Update performance metrics
   */
  private updatePerformanceMetrics(
    cycleTime: number,
    nodeCount: number,
    linkCount: number
  ): void {
    const memoryStats = this.tensorKernel.getMemoryStats();
    const attentionStats = this.attentionEngine.getCurrentStats(this.encoder.getNodeTensors());
    const reasoningChains = this.reasoningEngine.getActiveChains().length;
    
    const metrics: PerformanceMetrics = {
      cycles_per_second: cycleTime > 0 ? 1000 / cycleTime : 0,
      tensor_operations_per_second: (nodeCount + linkCount) / (cycleTime / 1000),
      memory_efficiency: 1 - memoryStats.percentage / 100,
      attention_allocation_efficiency: attentionStats.resource_utilization,
      reasoning_throughput: reasoningChains / Math.max(1, nodeCount),
      metacognitive_overhead: 0.1, // Placeholder
      overall_efficiency: 0.8 // Placeholder
    };

    this.state.performance_metrics = metrics;
    this.performanceHistory.push(metrics);

    // Keep performance history bounded
    if (this.performanceHistory.length > 100) {
      this.performanceHistory.shift();
    }
  }

  /**
   * Perform adaptive scaling based on performance metrics
   */
  private performAdaptiveScaling(
    attentionStats: AttentionStats,
    reasoningStats: ReasoningStats,
    metacognitiveMetrics: MetaCognitiveMetrics
  ): void {
    // Adjust attention allocation frequency
    if (attentionStats.resource_utilization > 0.9) {
      this.attentionEngine.updateConfig({
        update_frequency: Math.max(1, this.config.attention_config.update_frequency * 0.9)
      });
    }

    // Adjust reasoning depth based on performance
    if (reasoningStats.average_confidence < 0.3) {
      this.reasoningEngine.updateConfig({
        max_reasoning_depth: Math.min(10, this.config.reasoning_config.max_reasoning_depth + 1)
      });
    }

    // Adjust meta-cognitive parameters
    if (metacognitiveMetrics.system_coherence < 0.5) {
      this.metaCognitiveEngine.updateConfig({
        adaptation_learning_rate: Math.min(0.1, this.config.metacognitive_config.adaptation_learning_rate * 1.1)
      });
    }
  }

  /**
   * Get current cognitive kernel statistics
   */
  getCognitiveKernelStats(): CognitiveKernelStats {
    const nodeTensors = this.encoder.getNodeTensors();
    const linkTensors = this.encoder.getLinkTensors();
    const memoryStats = this.tensorKernel.getMemoryStats();
    
    const attentionStats = this.attentionEngine.getCurrentStats(nodeTensors);
    const reasoningStats = {
      total_chains: this.reasoningEngine.getActiveChains().length,
      active_chains: this.reasoningEngine.getActiveChains().filter(c => c.convergence_status === 'active').length,
      converged_chains: this.reasoningEngine.getActiveChains().filter(c => c.convergence_status === 'converged').length,
      average_depth: this.reasoningEngine.getActiveChains().reduce((sum, c) => sum + c.current_depth, 0) / Math.max(1, this.reasoningEngine.getActiveChains().length),
      total_inferences: this.reasoningEngine.getReasoningHistory().length,
      average_confidence: this.reasoningEngine.getActiveChains().reduce((sum, c) => sum + c.confidence_score, 0) / Math.max(1, this.reasoningEngine.getActiveChains().length),
      tensor_memory_usage: memoryStats.used,
      reasoning_throughput: 0.5 // Placeholder
    };
    
    const metacognitiveMetrics = {
      self_awareness_level: 0.8,
      adaptation_rate: 0.1,
      system_coherence: 0.9,
      meta_attention_efficiency: 0.8,
      feedback_loop_stability: 0.9,
      self_modification_frequency: 0.05,
      convergence_rate: 0.85
    };

    return {
      state: { ...this.state },
      attention_stats: attentionStats,
      reasoning_stats: reasoningStats,
      metacognitive_metrics: metacognitiveMetrics,
      tensor_stats: {
        total_tensors: nodeTensors.length + linkTensors.length,
        memory_usage: memoryStats.used,
        active_nodes: nodeTensors.length,
        active_links: linkTensors.length
      }
    };
  }

  /**
   * Handle errors
   */
  private handleError(error: string): void {
    console.error(`Cognitive Kernel Error: ${error}`);
    
    this.state.error_count++;
    this.state.last_error = error;
    
    if (this.state.error_count > 10) {
      this.state.status = 'error';
      this.notifyStateChange();
    }
    
    if (this.onError) {
      this.onError(error);
    }
  }

  /**
   * Notify state change
   */
  private notifyStateChange(): void {
    if (this.onStateChange) {
      this.onStateChange({ ...this.state });
    }
  }

  /**
   * Initialize performance metrics
   */
  private initializePerformanceMetrics(): PerformanceMetrics {
    return {
      cycles_per_second: 0,
      tensor_operations_per_second: 0,
      memory_efficiency: 1,
      attention_allocation_efficiency: 0,
      reasoning_throughput: 0,
      metacognitive_overhead: 0,
      overall_efficiency: 0
    };
  }

  /**
   * Set event callbacks
   */
  setCallbacks(callbacks: {
    onCycleComplete?: (stats: CognitiveKernelStats) => void;
    onError?: (error: string) => void;
    onStateChange?: (state: CognitiveKernelState) => void;
  }): void {
    this.onCycleComplete = callbacks.onCycleComplete;
    this.onError = callbacks.onError;
    this.onStateChange = callbacks.onStateChange;
  }

  /**
   * Get tensor kernel for direct access
   */
  getTensorKernel(): GgmlTensorKernel {
    return this.tensorKernel;
  }

  /**
   * Get encoder for direct access
   */
  getEncoder(): AtomSpaceTensorEncoder {
    return this.encoder;
  }

  /**
   * Get attention engine for direct access
   */
  getAttentionEngine(): AttentionAllocationEngine {
    return this.attentionEngine;
  }

  /**
   * Get reasoning engine for direct access
   */
  getReasoningEngine(): RecursiveReasoningEngine {
    return this.reasoningEngine;
  }

  /**
   * Get meta-cognitive engine for direct access
   */
  getMetaCognitiveEngine(): MetaCognitiveFeedbackEngine {
    return this.metaCognitiveEngine;
  }

  /**
   * Update kernel configuration
   */
  updateConfig(newConfig: Partial<CognitiveKernelConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Update component configurations
    if (newConfig.attention_config) {
      this.attentionEngine.updateConfig(newConfig.attention_config);
    }
    
    if (newConfig.metacognitive_config) {
      this.metaCognitiveEngine.updateConfig(newConfig.metacognitive_config);
    }
  }

  /**
   * Reset the cognitive kernel
   */
  reset(): void {
    this.stop();
    
    this.encoder.clearCache();
    this.attentionEngine.reset();
    this.reasoningEngine.reset();
    this.metaCognitiveEngine.reset();
    
    this.state.cycle_count = 0;
    this.state.error_count = 0;
    this.state.last_error = undefined;
    this.state.uptime = 0;
    this.performanceHistory = [];
    
    this.state.status = 'active';
    this.notifyStateChange();
  }

  /**
   * Shutdown the cognitive kernel
   */
  shutdown(): void {
    this.stop();
    this.state.status = 'shutdown';
    this.notifyStateChange();
    
    console.log('Cognitive kernel shutdown complete');
  }
}

/**
 * Factory function to create a cognitive kernel with default configuration
 */
export function createDefaultCognitiveKernel(): UnifiedCognitiveKernel {
  const config: CognitiveKernelConfig = {
    tensor_config: {
      node_embedding_dim: 128,
      link_embedding_dim: 64,
      attention_heads: 8,
      max_hypergraph_size: 1000,
      symbolic_depth_levels: 5,
      truth_value_encoding: 'continuous',
      attention_mechanism: 'hybrid'
    },
    attention_config: {
      mechanism: 'hybrid',
      temperature: 1.0,
      resource_budget: 1000,
      update_frequency: 10,
      decay_rate: 0.01,
      diffusion_strength: 0.1,
      rent_collection_rate: 0.01,
      wage_distribution_rate: 0.8,
      attention_heads: 8,
      gradient_clipping: 1.0
    },
    reasoning_config: {
      max_reasoning_depth: 5,
      confidence_threshold: 0.1,
      tensor_contraction_method: 'dot',
      inference_rules: [],
      chain_length_limit: 20,
      parallel_chains: 4,
      gradient_flow_enabled: true,
      meta_reasoning_enabled: true
    },
    metacognitive_config: {
      meta_levels: 3,
      self_monitoring_frequency: 5,
      adaptation_learning_rate: 0.01,
      convergence_threshold: 0.01,
      membrane_permeability: 0.5,
      feedback_damping: 0.9,
      plasticity_factor: 0.1,
      meta_attention_allocation: 0.2
    },
    integration_frequency: 10,
    performance_monitoring: true,
    adaptive_scaling: true,
    tensor_memory_limit: 1024 * 1024 * 1024 // 1GB
  };

  return new UnifiedCognitiveKernel(config);
}