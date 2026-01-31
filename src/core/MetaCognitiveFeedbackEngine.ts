/**
 * Meta-cognitive Feedback System
 * 
 * This module implements meta-tensor structures representing the system's own 
 * state and activation patterns, with feedback mechanisms for self-monitoring 
 * and adaptive self-modification.
 */

import { 
  GgmlTensorKernel, 
  GgmlTensor, 
  MetaCognitiveTensor,
  MembraneTensor,
  NodeTensor, 
  LinkTensor 
} from './GgmlTensorKernel';
import { AttentionAllocationEngine, AttentionStats } from './AttentionAllocationEngine';
import { RecursiveReasoningEngine, ReasoningStats } from './RecursiveReasoningEngine';

/**
 * Meta-cognitive system configuration
 */
export interface MetaCognitiveConfig {
  meta_levels: number;
  self_monitoring_frequency: number;
  adaptation_learning_rate: number;
  convergence_threshold: number;
  membrane_permeability: number;
  feedback_damping: number;
  plasticity_factor: number;
  meta_attention_allocation: number;
}

/**
 * System state representation
 */
export interface SystemState {
  cognitive_load: number;
  attention_distribution: number[];
  reasoning_activity: number;
  memory_usage: number;
  processing_efficiency: number;
  convergence_status: 'stable' | 'adapting' | 'diverging' | 'critical';
  timestamp: number;
}

/**
 * Meta-cognitive feedback loop
 */
export interface FeedbackLoop {
  id: string;
  source_component: 'attention' | 'reasoning' | 'encoding' | 'memory';
  target_component: 'attention' | 'reasoning' | 'encoding' | 'memory';
  feedback_tensor: GgmlTensor;
  feedback_strength: number;
  loop_type: 'positive' | 'negative' | 'oscillatory';
  stability_measure: number;
}

/**
 * Self-modification record
 */
export interface SelfModification {
  id: string;
  timestamp: number;
  modification_type: 'parameter_adjustment' | 'structure_change' | 'rule_addition' | 'rule_removal';
  target_component: string;
  modification_tensor: GgmlTensor;
  effectiveness_score: number;
  reversibility: boolean;
}

/**
 * Meta-cognitive metrics
 */
export interface MetaCognitiveMetrics {
  self_awareness_level: number;
  adaptation_rate: number;
  system_coherence: number;
  meta_attention_efficiency: number;
  feedback_loop_stability: number;
  self_modification_frequency: number;
  convergence_rate: number;
}

/**
 * Meta-cognitive feedback engine
 */
export class MetaCognitiveFeedbackEngine {
  private kernel: GgmlTensorKernel;
  private attentionEngine: AttentionAllocationEngine;
  private reasoningEngine: RecursiveReasoningEngine;
  private config: MetaCognitiveConfig;
  
  // Meta-cognitive state
  private metaTensors: Map<number, MetaCognitiveTensor>;
  private membranes: Map<string, MembraneTensor>;
  private systemStates: SystemState[];
  private feedbackLoops: Map<string, FeedbackLoop>;
  private selfModifications: SelfModification[];
  
  // Monitoring and adaptation
  private lastMonitoringTime: number;
  private adaptationHistory: number[];
  private convergenceMetrics: number[];

  constructor(
    kernel: GgmlTensorKernel,
    attentionEngine: AttentionAllocationEngine,
    reasoningEngine: RecursiveReasoningEngine,
    config: MetaCognitiveConfig
  ) {
    this.kernel = kernel;
    this.attentionEngine = attentionEngine;
    this.reasoningEngine = reasoningEngine;
    this.config = config;
    
    this.metaTensors = new Map();
    this.membranes = new Map();
    this.systemStates = [];
    this.feedbackLoops = new Map();
    this.selfModifications = [];
    
    this.lastMonitoringTime = Date.now();
    this.adaptationHistory = [];
    this.convergenceMetrics = [];
    
    this.initializeMetaCognitiveTensors();
    this.initializeMembranes();
    this.initializeFeedbackLoops();
  }

  /**
   * Initialize meta-cognitive tensor hierarchy
   */
  private initializeMetaCognitiveTensors(): void {
    for (let level = 0; level < this.config.meta_levels; level++) {
      const tensorSize = Math.max(10, 100 - level * 20); // Decreasing size with level
      
      const systemState = this.kernel.createTensor([tensorSize], 'f32', `meta_state_${level}`, true);
      const activationPatterns = this.kernel.createTensor([tensorSize, tensorSize], 'f32', `meta_activation_${level}`, true);
      const feedbackTensor = this.kernel.createTensor([tensorSize], 'f32', `meta_feedback_${level}`, true);
      const selfMonitoring = this.kernel.createTensor([tensorSize], 'f32', `meta_monitoring_${level}`, true);
      const adaptationWeights = this.kernel.createTensor([tensorSize, tensorSize], 'f32', `meta_adaptation_${level}`, true);

      const metaTensor: MetaCognitiveTensor = {
        id: `meta_cognitive_${level}`,
        system_state: systemState,
        activation_patterns: activationPatterns,
        feedback_tensor: feedbackTensor,
        self_monitoring: selfMonitoring,
        adaptation_weights: adaptationWeights,
        meta_level: level,
        convergence_threshold: this.config.convergence_threshold * (1 + level * 0.1)
      };

      this.metaTensors.set(level, metaTensor);
    }
  }

  /**
   * Initialize P-System inspired membranes
   */
  private initializeMembranes(): void {
    const membraneTypes = ['cognitive', 'attention', 'reasoning', 'meta'];
    
    membraneTypes.forEach(type => {
      const boundarySize = 50;
      
      const boundaryTensor = this.kernel.createTensor([boundarySize], 'f32', `membrane_boundary_${type}`, true);
      const permeabilityWeights = this.kernel.createTensor([boundarySize], 'f32', `membrane_permeability_${type}`, true);
      const internalState = this.kernel.createTensor([boundarySize], 'f32', `membrane_internal_${type}`, true);
      const externalInterface = this.kernel.createTensor([boundarySize], 'f32', `membrane_external_${type}`, true);
      const membraneRules = this.kernel.createTensor([boundarySize, boundarySize], 'f32', `membrane_rules_${type}`, true);

      // Initialize permeability based on membrane type
      const basePermeability = type === 'cognitive' ? 0.8 : 
                               type === 'attention' ? 0.6 : 
                               type === 'reasoning' ? 0.4 : 0.2;
      
      for (let i = 0; i < boundarySize; i++) {
        permeabilityWeights.data[i] = basePermeability + (Math.random() - 0.5) * 0.1;
      }

      const membrane: MembraneTensor = {
        id: `membrane_${type}`,
        boundary_tensor: boundaryTensor,
        permeability_weights: permeabilityWeights,
        internal_state: internalState,
        external_interface: externalInterface,
        membrane_type: type as 'cognitive' | 'attention' | 'reasoning' | 'meta',
        membrane_rules: membraneRules
      };

      this.membranes.set(type, membrane);
    });
  }

  /**
   * Initialize feedback loops between components
   */
  private initializeFeedbackLoops(): void {
    const components = ['attention', 'reasoning', 'encoding', 'memory'];
    
    components.forEach(source => {
      components.forEach(target => {
        if (source !== target) {
          const loopId = `${source}_to_${target}`;
          const feedbackTensor = this.kernel.createTensor([20], 'f32', `feedback_${loopId}`, true);
          
          const feedbackLoop: FeedbackLoop = {
            id: loopId,
            source_component: source as 'attention' | 'reasoning' | 'encoding' | 'memory',
            target_component: target as 'attention' | 'reasoning' | 'encoding' | 'memory',
            feedback_tensor: feedbackTensor,
            feedback_strength: 0.1 + Math.random() * 0.3,
            loop_type: Math.random() < 0.7 ? 'positive' : 'negative',
            stability_measure: 0.8 + Math.random() * 0.2
          };

          this.feedbackLoops.set(loopId, feedbackLoop);
        }
      });
    });
  }

  /**
   * Update meta-cognitive state and perform self-monitoring
   */
  updateMetaCognitive(
    nodeTensors: NodeTensor[],
    linkTensors: LinkTensor[],
    attentionStats: AttentionStats,
    reasoningStats: ReasoningStats
  ): MetaCognitiveMetrics {
    const now = Date.now();
    const timeDelta = (now - this.lastMonitoringTime) / 1000.0;
    
    if (timeDelta < 1.0 / this.config.self_monitoring_frequency) {
      return this.calculateMetaCognitiveMetrics();
    }

    // Step 1: Monitor current system state
    const systemState = this.monitorSystemState(nodeTensors, linkTensors, attentionStats, reasoningStats);
    this.systemStates.push(systemState);

    // Step 2: Update meta-cognitive tensors
    this.updateMetaTensors(systemState, nodeTensors, linkTensors);

    // Step 3: Process feedback loops
    this.processFeedbackLoops(systemState);

    // Step 4: Perform self-adaptation
    this.performSelfAdaptation(systemState);

    // Step 5: Update membrane permeability
    this.updateMembranePermeability(systemState);

    // Step 6: Check for self-modification opportunities
    this.checkSelfModificationOpportunities(systemState);

    this.lastMonitoringTime = now;
    return this.calculateMetaCognitiveMetrics();
  }

  /**
   * Monitor current system state
   */
  private monitorSystemState(
    nodeTensors: NodeTensor[],
    linkTensors: LinkTensor[],
    attentionStats: AttentionStats,
    reasoningStats: ReasoningStats
  ): SystemState {
    // Calculate cognitive load
    const totalNodes = nodeTensors.length;
    const totalLinks = linkTensors.length;
    const cognitiveLoad = (totalNodes + totalLinks) / 1000.0; // Normalized

    // Calculate attention distribution entropy
    const attentionDistribution = nodeTensors.map(n => n.attention_weights.data[0]);

    // Calculate reasoning activity
    const reasoningActivity = reasoningStats.active_chains / Math.max(1, reasoningStats.total_chains);

    // Calculate memory usage
    const memoryUsage = this.kernel.getMemoryStats().percentage / 100.0;

    // Calculate processing efficiency
    const processingEfficiency = (attentionStats.resource_utilization + reasoningStats.reasoning_throughput) / 2.0;

    // Determine convergence status
    let convergenceStatus: SystemState['convergence_status'] = 'stable';
    if (attentionStats.gradient_norm > 0.1 || reasoningStats.average_confidence < 0.5) {
      convergenceStatus = 'adapting';
    }
    if (memoryUsage > 0.9 || cognitiveLoad > 0.8) {
      convergenceStatus = 'critical';
    }

    return {
      cognitive_load: cognitiveLoad,
      attention_distribution: attentionDistribution,
      reasoning_activity: reasoningActivity,
      memory_usage: memoryUsage,
      processing_efficiency: processingEfficiency,
      convergence_status: convergenceStatus,
      timestamp: Date.now()
    };
  }

  /**
   * Update meta-cognitive tensors with current state
   */
  private updateMetaTensors(
    systemState: SystemState,
    nodeTensors: NodeTensor[],
    linkTensors: LinkTensor[]
  ): void {
    for (let level = 0; level < this.config.meta_levels; level++) {
      const metaTensor = this.metaTensors.get(level);
      if (!metaTensor) continue;

      // Update system state tensor
      metaTensor.system_state.data[0] = systemState.cognitive_load;
      metaTensor.system_state.data[1] = systemState.reasoning_activity;
      metaTensor.system_state.data[2] = systemState.memory_usage;
      metaTensor.system_state.data[3] = systemState.processing_efficiency;

      // Update activation patterns
      this.updateActivationPatterns(metaTensor, nodeTensors, linkTensors, level);

      // Update self-monitoring
      this.updateSelfMonitoring(metaTensor, systemState, level);

      // Update feedback tensor
      this.updateFeedbackTensor(metaTensor, systemState, level);
    }
  }

  /**
   * Update activation patterns for meta-level
   */
  private updateActivationPatterns(
    metaTensor: MetaCognitiveTensor,
    nodeTensors: NodeTensor[],
    _: LinkTensor[],
    _level: number
  ): void {
    const patternMatrix = metaTensor.activation_patterns;
    const size = patternMatrix.shape[0];
    
    // Sample activation patterns from nodes
    const sampleSize = Math.min(size, nodeTensors.length);
    for (let i = 0; i < sampleSize; i++) {
      const node = nodeTensors[i];
      const attentionValue = node.attention_weights.data[0];
      
      for (let j = 0; j < size; j++) {
        const index = i * size + j;
        if (index < patternMatrix.data.length) {
          patternMatrix.data[index] = patternMatrix.data[index] * 0.9 + attentionValue * 0.1;
        }
      }
    }
  }

  /**
   * Update self-monitoring tensor
   */
  private updateSelfMonitoring(
    metaTensor: MetaCognitiveTensor,
    systemState: SystemState,
    level: number
  ): void {
    const monitoring = metaTensor.self_monitoring;
    
    // Meta-level awareness decreases with level
    const awarenessDecay = Math.pow(0.8, level);
    
    // Update monitoring values
    monitoring.data[0] = systemState.cognitive_load * awarenessDecay;
    monitoring.data[1] = systemState.processing_efficiency * awarenessDecay;
    monitoring.data[2] = (systemState.convergence_status === 'stable' ? 1.0 : 0.0) * awarenessDecay;
    
    // Add some self-reflection patterns
    for (let i = 3; i < Math.min(monitoring.data.length, 10); i++) {
      monitoring.data[i] = monitoring.data[i] * 0.95 + 
                          Math.sin(Date.now() * 0.001 + i) * 0.05 * awarenessDecay;
    }
  }

  /**
   * Update feedback tensor for meta-level
   */
  private updateFeedbackTensor(
    metaTensor: MetaCognitiveTensor,
    systemState: SystemState,
    level: number
  ): void {
    const feedback = metaTensor.feedback_tensor;
    
    // Calculate feedback signals
    const stabilityFeedback = systemState.convergence_status === 'stable' ? 1.0 : -0.5;
    const efficiencyFeedback = systemState.processing_efficiency - 0.5;
    const loadFeedback = 0.5 - systemState.cognitive_load;
    
    // Update feedback values
    feedback.data[0] = stabilityFeedback * (1 - level * 0.1);
    feedback.data[1] = efficiencyFeedback * (1 - level * 0.1);
    feedback.data[2] = loadFeedback * (1 - level * 0.1);
    
    // Apply feedback damping
    for (let i = 0; i < feedback.data.length; i++) {
      feedback.data[i] *= this.config.feedback_damping;
    }
  }

  /**
   * Process feedback loops between components
   */
  private processFeedbackLoops(systemState: SystemState): void {
    for (const [, feedbackLoop] of this.feedbackLoops) {
      const strength = feedbackLoop.feedback_strength;
      const tensor = feedbackLoop.feedback_tensor;
      
      // Calculate feedback based on system state
      let feedbackValue = 0.0;
      
      if (feedbackLoop.source_component === 'attention') {
        feedbackValue = systemState.attention_distribution.reduce((sum, val) => sum + val, 0) / 
                       Math.max(1, systemState.attention_distribution.length);
      } else if (feedbackLoop.source_component === 'reasoning') {
        feedbackValue = systemState.reasoning_activity;
      } else if (feedbackLoop.source_component === 'memory') {
        feedbackValue = 1.0 - systemState.memory_usage;
      }
      
      // Apply feedback type
      if (feedbackLoop.loop_type === 'negative') {
        feedbackValue = -feedbackValue;
      } else if (feedbackLoop.loop_type === 'oscillatory') {
        feedbackValue = feedbackValue * Math.sin(Date.now() * 0.001);
      }
      
      // Update feedback tensor
      for (let i = 0; i < tensor.data.length; i++) {
        tensor.data[i] = tensor.data[i] * 0.9 + feedbackValue * strength * 0.1;
      }
      
      // Update stability measure
      const variation = Math.abs(feedbackValue - tensor.data[0]);
      feedbackLoop.stability_measure = feedbackLoop.stability_measure * 0.99 + 
                                     (1.0 - variation) * 0.01;
    }
  }

  /**
   * Perform self-adaptation based on feedback
   */
  private performSelfAdaptation(systemState: SystemState): void {
    const adaptationRate = this.config.adaptation_learning_rate;
    
    // Adapt based on system state
    if (systemState.convergence_status === 'critical') {
      // Increase adaptation rate for critical states
      this.adaptSystemParameters(adaptationRate * 2.0, systemState);
    } else if (systemState.convergence_status === 'adapting') {
      // Normal adaptation
      this.adaptSystemParameters(adaptationRate, systemState);
    }
    
    // Record adaptation
    this.adaptationHistory.push(adaptationRate);
    
    // Keep adaptation history bounded
    if (this.adaptationHistory.length > 100) {
      this.adaptationHistory.shift();
    }
  }

  /**
   * Adapt system parameters based on feedback
   */
  private adaptSystemParameters(adaptationRate: number, _: SystemState): void {
    // Adapt meta-cognitive parameters
    for (const [, metaTensor] of this.metaTensors) {
      const adaptationWeights = metaTensor.adaptation_weights;
      
      // Apply adaptation based on feedback
      const feedbackMagnitude = Math.sqrt(
        metaTensor.feedback_tensor.data.reduce((sum, val) => sum + val * val, 0)
      );
      
      if (feedbackMagnitude > 0.1) {
        for (let i = 0; i < adaptationWeights.data.length; i++) {
          const adaptationDelta = metaTensor.feedback_tensor.data[i % metaTensor.feedback_tensor.data.length] * 
                                  adaptationRate * this.config.plasticity_factor;
          adaptationWeights.data[i] += adaptationDelta;
          
          // Clamp adaptation weights
          adaptationWeights.data[i] = Math.max(-1.0, Math.min(1.0, adaptationWeights.data[i]));
        }
      }
    }
  }

  /**
   * Update membrane permeability based on system state
   */
  private updateMembranePermeability(systemState: SystemState): void {
    for (const [, membrane] of this.membranes) {
      const permeability = membrane.permeability_weights;
      
      // Adjust permeability based on system state
      let adjustmentFactor = 1.0;
      
      if (systemState.convergence_status === 'critical') {
        adjustmentFactor = 0.5; // Reduce permeability in critical states
      } else if (systemState.processing_efficiency > 0.8) {
        adjustmentFactor = 1.2; // Increase permeability for efficient processing
      }
      
      for (let i = 0; i < permeability.data.length; i++) {
        permeability.data[i] = Math.max(0.1, Math.min(1.0, 
          permeability.data[i] * adjustmentFactor
        ));
      }
      
      // Update internal state based on permeability
      this.updateMembraneInternalState(membrane, systemState);
    }
  }

  /**
   * Update membrane internal state
   */
  private updateMembraneInternalState(membrane: MembraneTensor, _: SystemState): void {
    const internalState = membrane.internal_state;
    const permeability = membrane.permeability_weights;
    
    // Transfer information based on permeability
    for (let i = 0; i < internalState.data.length; i++) {
      const externalValue = membrane.external_interface.data[i];
      const transferRate = permeability.data[i] * 0.1;
      
      internalState.data[i] = internalState.data[i] * (1 - transferRate) + 
                             externalValue * transferRate;
    }
  }

  /**
   * Check for self-modification opportunities
   */
  private checkSelfModificationOpportunities(systemState: SystemState): void {
    // Check if system needs modification
    if (systemState.processing_efficiency < 0.3 || 
        systemState.convergence_status === 'critical') {
      
      this.performSelfModification(systemState);
    }
  }

  /**
   * Perform self-modification
   */
  private performSelfModification(systemState: SystemState): void {
    const modificationId = `modification_${Date.now()}`;
    
    // Create modification tensor
    const modificationTensor = this.kernel.createTensor([10], 'f32', modificationId);
    
    // Determine modification type
    let modificationType: SelfModification['modification_type'] = 'parameter_adjustment';
    
    if (systemState.memory_usage > 0.8) {
      modificationType = 'structure_change';
    } else if (systemState.reasoning_activity < 0.2) {
      modificationType = 'rule_addition';
    }
    
    // Apply modification
    const effectiveness = this.applySelfModification(modificationType, modificationTensor, systemState);
    
    // Record modification
    const modification: SelfModification = {
      id: modificationId,
      timestamp: Date.now(),
      modification_type: modificationType,
      target_component: 'meta_cognitive',
      modification_tensor: modificationTensor,
      effectiveness_score: effectiveness,
      reversibility: effectiveness > 0.5
    };
    
    this.selfModifications.push(modification);
    
    // Keep modification history bounded
    if (this.selfModifications.length > 50) {
      this.selfModifications.shift();
    }
  }

  /**
   * Apply self-modification
   */
  private applySelfModification(
    modificationType: SelfModification['modification_type'],
    modificationTensor: GgmlTensor,
    systemState: SystemState
  ): number {
    let effectiveness = 0.0;
    
    switch (modificationType) {
      case 'parameter_adjustment':
        effectiveness = this.adjustParameters(modificationTensor, systemState);
        break;
      case 'structure_change':
        effectiveness = this.changeStructure(modificationTensor, systemState);
        break;
      case 'rule_addition':
        effectiveness = this.addRule(modificationTensor, systemState);
        break;
      case 'rule_removal':
        effectiveness = this.removeRule(modificationTensor, systemState);
        break;
    }
    
    return effectiveness;
  }

  /**
   * Adjust parameters for self-modification
   */
  private adjustParameters(modificationTensor: GgmlTensor, _: SystemState): number {
    // Adjust learning rates and thresholds
    
    // Simulate parameter adjustment
    for (let i = 0; i < modificationTensor.data.length; i++) {
      modificationTensor.data[i] = (Math.random() - 0.5) * 0.2;
    }
    
    // Return simulated effectiveness
    return 0.7 + Math.random() * 0.3;
  }

  /**
   * Change structure for self-modification
   */
  private changeStructure(_modTensor: GgmlTensor, _state: SystemState): number {
    // Simulate structural changes
    return 0.5 + Math.random() * 0.5;
  }

  /**
   * Add rule for self-modification
   */
  private addRule(_modTensor: GgmlTensor, _state: SystemState): number {
    // Simulate rule addition
    return 0.6 + Math.random() * 0.4;
  }

  /**
   * Remove rule for self-modification
   */
  private removeRule(_modTensor: GgmlTensor, _state: SystemState): number {
    // Simulate rule removal
    return 0.4 + Math.random() * 0.6;
  }

  /**
   * Calculate meta-cognitive metrics
   */
  private calculateMetaCognitiveMetrics(): MetaCognitiveMetrics {
    const recentStates = this.systemStates.slice(-10);
    const recentAdaptations = this.adaptationHistory.slice(-10);
    
    if (recentStates.length === 0) {
      return {
        self_awareness_level: 0,
        adaptation_rate: 0,
        system_coherence: 0,
        meta_attention_efficiency: 0,
        feedback_loop_stability: 0,
        self_modification_frequency: 0,
        convergence_rate: 0
      };
    }
    
    // Calculate self-awareness level
    const selfAwarenessLevel = recentStates.reduce((sum, state) => 
      sum + (state.convergence_status === 'stable' ? 1 : 0), 0) / recentStates.length;
    
    // Calculate adaptation rate
    const adaptationRate = recentAdaptations.reduce((sum, rate) => sum + rate, 0) / 
                          Math.max(1, recentAdaptations.length);
    
    // Calculate system coherence
    const systemCoherence = recentStates.reduce((sum, state) => 
      sum + state.processing_efficiency, 0) / recentStates.length;
    
    // Calculate meta-attention efficiency
    const metaAttentionEfficiency = this.config.meta_attention_allocation;
    
    // Calculate feedback loop stability
    const feedbackLoopStability = Array.from(this.feedbackLoops.values())
      .reduce((sum, loop) => sum + loop.stability_measure, 0) / 
      Math.max(1, this.feedbackLoops.size);
    
    // Calculate self-modification frequency
    const recentModifications = this.selfModifications.filter(mod => 
      Date.now() - mod.timestamp < 60000); // Last minute
    const selfModificationFrequency = recentModifications.length / 60.0; // Per second
    
    // Calculate convergence rate
    const convergenceRate = recentStates.reduce((sum, state) => 
      sum + (state.convergence_status === 'stable' ? 1 : 0), 0) / recentStates.length;
    
    return {
      self_awareness_level: selfAwarenessLevel,
      adaptation_rate: adaptationRate,
      system_coherence: systemCoherence,
      meta_attention_efficiency: metaAttentionEfficiency,
      feedback_loop_stability: feedbackLoopStability,
      self_modification_frequency: selfModificationFrequency,
      convergence_rate: convergenceRate
    };
  }

  /**
   * Get meta-cognitive tensors for specific level
   */
  getMetaTensor(level: number): MetaCognitiveTensor | undefined {
    return this.metaTensors.get(level);
  }

  /**
   * Get membrane by type
   */
  getMembrane(type: string): MembraneTensor | undefined {
    return this.membranes.get(type);
  }

  /**
   * Get system state history
   */
  getSystemStateHistory(): SystemState[] {
    return [...this.systemStates];
  }

  /**
   * Get feedback loops
   */
  getFeedbackLoops(): FeedbackLoop[] {
    return Array.from(this.feedbackLoops.values());
  }

  /**
   * Get self-modifications
   */
  getSelfModifications(): SelfModification[] {
    return [...this.selfModifications];
  }

  /**
   * Reset meta-cognitive system
   */
  reset(): void {
    this.systemStates = [];
    this.adaptationHistory = [];
    this.convergenceMetrics = [];
    this.selfModifications = [];
    this.initializeMetaCognitiveTensors();
    this.initializeMembranes();
    this.initializeFeedbackLoops();
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<MetaCognitiveConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }
}