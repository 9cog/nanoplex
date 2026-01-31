/**
 * Attention Allocation Subsystem
 * 
 * This module integrates dynamic attention mechanisms (softmax and ECAN-inspired) 
 * that assign and update attention weights across the AtomSpace tensor field.
 * The attention allocation is differentiable and can be recursively updated based on 
 * utility gradients and cognitive resource budgets.
 */

import { GgmlTensorKernel, GgmlTensor, AttentionTensor, NodeTensor, LinkTensor } from './GgmlTensorKernel';
import { AtomSpaceTensorEncoder } from './AtomSpaceTensorEncoder';

/**
 * Attention allocation strategy configuration
 */
export interface AttentionAllocationConfig {
  mechanism: 'softmax' | 'ecan' | 'hybrid';
  temperature: number;
  resource_budget: number;
  update_frequency: number;
  decay_rate: number;
  diffusion_strength: number;
  rent_collection_rate: number;
  wage_distribution_rate: number;
  attention_heads: number;
  gradient_clipping: number;
}

/**
 * Attention allocation statistics
 */
export interface AttentionStats {
  total_attention: number;
  average_attention: number;
  attention_entropy: number;
  resource_utilization: number;
  gradient_norm: number;
  convergence_rate: number;
}

/**
 * Attention flow record for monitoring
 */
export interface AttentionFlow {
  timestamp: number;
  source_id: string;
  target_id: string;
  attention_transfer: number;
  reason: 'diffusion' | 'rent' | 'wage' | 'gradient';
}

/**
 * Differentiable attention allocation engine
 */
export class AttentionAllocationEngine {
  private kernel: GgmlTensorKernel;
  private encoder: AtomSpaceTensorEncoder;
  private config: AttentionAllocationConfig;
  private globalAttention: AttentionTensor;
  private attentionHistory: AttentionFlow[];
  private lastUpdate: number;
  private resourceBudget: number;

  constructor(
    kernel: GgmlTensorKernel,
    encoder: AtomSpaceTensorEncoder,
    config: AttentionAllocationConfig
  ) {
    this.kernel = kernel;
    this.encoder = encoder;
    this.config = config;
    this.attentionHistory = [];
    this.lastUpdate = Date.now();
    this.resourceBudget = config.resource_budget;
    
    this.initializeGlobalAttention();
  }

  /**
   * Initialize global attention tensor structure
   */
  private initializeGlobalAttention(): void {
    const maxNodes = 1000; // Configurable maximum nodes
    
    // Global attention distribution
    const globalAttention = this.kernel.createTensor([maxNodes], 'f32', 'global_attention', true);
    
    // Local attention patterns (multi-head)
    const localAttention = this.kernel.createTensor(
      [this.config.attention_heads, maxNodes], 
      'f32', 
      'local_attention', 
      true
    );
    
    // Attention heads for different cognitive processes
    const attentionHeads: GgmlTensor[] = [];
    for (let i = 0; i < this.config.attention_heads; i++) {
      const head = this.kernel.createTensor([maxNodes], 'f32', `attention_head_${i}`, true);
      attentionHeads.push(head);
    }
    
    // Attention mask for active nodes
    const attentionMask = this.kernel.createTensor([maxNodes], 'f32', 'attention_mask', false);
    
    this.globalAttention = {
      id: 'global_attention_system',
      global_attention: globalAttention,
      local_attention: localAttention,
      attention_heads: attentionHeads,
      attention_mask: attentionMask,
      resource_budget: this.resourceBudget,
      update_rate: this.config.update_frequency,
      ecan_parameters: {
        rent_rate: this.config.rent_collection_rate,
        wage_rate: this.config.wage_distribution_rate,
        diffusion_rate: this.config.diffusion_strength
      }
    };
  }

  /**
   * Update attention allocation across all nodes
   */
  updateAttentionAllocation(nodeTensors: NodeTensor[], linkTensors: LinkTensor[]): AttentionStats {
    const now = Date.now();
    const timeDelta = (now - this.lastUpdate) / 1000.0; // Convert to seconds
    
    if (timeDelta < 1.0 / this.config.update_frequency) {
      return this.calculateAttentionStats(nodeTensors);
    }

    // Step 1: Apply attention mechanism
    let stats: AttentionStats;
    switch (this.config.mechanism) {
      case 'softmax':
        stats = this.applySoftmaxAttention(nodeTensors, linkTensors);
        break;
      case 'ecan':
        stats = this.applyECANAttention(nodeTensors, linkTensors);
        break;
      case 'hybrid':
        stats = this.applyHybridAttention(nodeTensors, linkTensors);
        break;
    }

    // Step 2: Apply gradient-based updates
    this.applyGradientUpdates(nodeTensors, linkTensors);

    // Step 3: Enforce resource constraints
    this.enforceResourceConstraints(nodeTensors);

    // Step 4: Record attention flows
    this.recordAttentionFlows(nodeTensors, linkTensors);

    this.lastUpdate = now;
    return stats;
  }

  /**
   * Apply softmax attention mechanism
   */
  private applySoftmaxAttention(nodeTensors: NodeTensor[], linkTensors: LinkTensor[]): AttentionStats {
    if (nodeTensors.length === 0) {
      return this.calculateAttentionStats(nodeTensors);
    }

    // Create attention logits tensor
    const logitsTensor = this.kernel.createTensor([nodeTensors.length], 'f32', 'attention_logits', true);
    
    // Calculate attention logits based on node features
    nodeTensors.forEach((node, i) => {
      let logit = 0.0;
      
      // Base attention from truth value
      logit += node.truth_value_tensor.data[0] * node.truth_value_tensor.data[1]; // strength * confidence
      
      // Boost from current attention
      logit += node.attention_weights.data[0] * 0.5;
      
      // Symbolic depth contribution
      logit += Math.log(node.metadata.symbolic_depth + 1) * 0.1;
      
      // Degrees of freedom contribution
      logit += Math.log(node.metadata.degree_of_freedom + 1) * 0.05;
      
      logitsTensor.data[i] = logit / this.config.temperature;
    });

    // Apply softmax
    const attentionWeights = this.kernel.softmax(logitsTensor);
    
    // Update node attention weights
    nodeTensors.forEach((node, i) => {
      const newAttention = attentionWeights.data[i];
      node.attention_weights.data[0] = newAttention;
      
      // Update STI based on new attention
      node.attention_weights.data[1] = node.attention_weights.data[1] * (1 - this.config.decay_rate) + 
                                     newAttention * this.config.decay_rate;
    });

    // Update link attention based on connected nodes
    this.updateLinkAttention(linkTensors, nodeTensors);

    return this.calculateAttentionStats(nodeTensors);
  }

  /**
   * Apply ECAN (Economic Attention Allocation) mechanism
   */
  private applyECANAttention(nodeTensors: NodeTensor[], linkTensors: LinkTensor[]): AttentionStats {
    // Step 1: Collect rent from all nodes
    const rentRate = this.globalAttention.ecan_parameters.rent_rate;
    let totalRentCollected = 0;
    
    nodeTensors.forEach(node => {
      const currentSTI = node.attention_weights.data[0];
      const rent = currentSTI * rentRate;
      totalRentCollected += rent;
      
      // Deduct rent from STI
      node.attention_weights.data[0] = Math.max(0, currentSTI - rent);
    });

    // Step 2: Distribute wages based on importance and activity
    const wageRate = this.globalAttention.ecan_parameters.wage_rate;
    const totalWages = totalRentCollected * wageRate;
    
    // Calculate importance scores
    const importanceScores = nodeTensors.map(node => {
      const truthValue = node.truth_value_tensor.data[0] * node.truth_value_tensor.data[1];
      const currentAttention = node.attention_weights.data[0];
      const symbolicComplexity = Math.log(node.metadata.symbolic_depth + 1);
      
      return truthValue * 0.4 + currentAttention * 0.3 + symbolicComplexity * 0.3;
    });
    
    const totalImportance = importanceScores.reduce((sum, score) => sum + score, 0);
    
    // Distribute wages proportionally
    nodeTensors.forEach((node, i) => {
      if (totalImportance > 0) {
        const wage = (importanceScores[i] / totalImportance) * totalWages;
        node.attention_weights.data[0] += wage;
      }
    });

    // Step 3: Apply attention diffusion
    this.applyAttentionDiffusion(nodeTensors, linkTensors);

    // Step 4: Update resource budget
    this.resourceBudget = Math.max(0, this.resourceBudget - totalRentCollected + totalWages);

    return this.calculateAttentionStats(nodeTensors);
  }

  /**
   * Apply hybrid attention mechanism (combination of softmax and ECAN)
   */
  private applyHybridAttention(nodeTensors: NodeTensor[], linkTensors: LinkTensor[]): AttentionStats {
    // Apply softmax for fast attention reallocation
    const softmaxStats = this.applySoftmaxAttention(nodeTensors, linkTensors);
    
    // Apply ECAN for economic stability
    const ecanStats = this.applyECANAttention(nodeTensors, linkTensors);
    
    // Combine statistics
    return {
      total_attention: (softmaxStats.total_attention + ecanStats.total_attention) / 2,
      average_attention: (softmaxStats.average_attention + ecanStats.average_attention) / 2,
      attention_entropy: (softmaxStats.attention_entropy + ecanStats.attention_entropy) / 2,
      resource_utilization: ecanStats.resource_utilization,
      gradient_norm: softmaxStats.gradient_norm,
      convergence_rate: (softmaxStats.convergence_rate + ecanStats.convergence_rate) / 2
    };
  }

  /**
   * Apply attention diffusion across connected nodes
   */
  private applyAttentionDiffusion(nodeTensors: NodeTensor[], linkTensors: LinkTensor[]): void {
    const diffusionRate = this.globalAttention.ecan_parameters.diffusion_rate;
    
    linkTensors.forEach(link => {
      const sourceNodes = link.source_nodes;
      const targetNodes = link.target_nodes;
      
      sourceNodes.forEach(sourceId => {
        const sourceNode = nodeTensors.find(n => n.atom_id === sourceId);
        if (!sourceNode) return;
        
        targetNodes.forEach(targetId => {
          const targetNode = nodeTensors.find(n => n.atom_id === targetId);
          if (!targetNode) return;
          
          // Calculate diffusion amount
          const attentionDiff = sourceNode.attention_weights.data[0] - targetNode.attention_weights.data[0];
          const diffusionAmount = attentionDiff * diffusionRate * link.truth_value_tensor.data[0];
          
          // Apply diffusion
          sourceNode.attention_weights.data[0] -= diffusionAmount * 0.5;
          targetNode.attention_weights.data[0] += diffusionAmount * 0.5;
          
          // Record attention flow
          this.attentionHistory.push({
            timestamp: Date.now(),
            source_id: sourceId,
            target_id: targetId,
            attention_transfer: diffusionAmount,
            reason: 'diffusion'
          });
        });
      });
    });
  }

  /**
   * Update link attention based on connected nodes
   */
  private updateLinkAttention(linkTensors: LinkTensor[], nodeTensors: NodeTensor[]): void {
    linkTensors.forEach(link => {
      let totalAttention = 0;
      let nodeCount = 0;
      
      // Aggregate attention from all connected nodes
      [...link.source_nodes, ...link.target_nodes].forEach(nodeId => {
        const node = nodeTensors.find(n => n.atom_id === nodeId);
        if (node) {
          totalAttention += node.attention_weights.data[0];
          nodeCount++;
        }
      });
      
      // Update link attention as average of connected nodes
      if (nodeCount > 0) {
        link.attention_weights.data[0] = totalAttention / nodeCount;
      }
    });
  }

  /**
   * Apply gradient-based updates to attention weights
   */
  private applyGradientUpdates(nodeTensors: NodeTensor[], _: LinkTensor[]): void {
    const learningRate = 0.001;
    
    nodeTensors.forEach(node => {
      if (node.attention_weights.gradient) {
        // Apply gradient clipping
        const gradientNorm = Math.sqrt(
          node.attention_weights.gradient.reduce((sum, g) => sum + g * g, 0)
        );
        
        if (gradientNorm > this.config.gradient_clipping) {
          const scale = this.config.gradient_clipping / gradientNorm;
          for (let i = 0; i < node.attention_weights.gradient.length; i++) {
            node.attention_weights.gradient[i] *= scale;
          }
        }
        
        // Apply gradient update
        for (let i = 0; i < node.attention_weights.data.length; i++) {
          node.attention_weights.data[i] -= learningRate * node.attention_weights.gradient[i];
          node.attention_weights.data[i] = Math.max(0, node.attention_weights.data[i]); // Ensure non-negative
        }
      }
    });
  }

  /**
   * Enforce resource constraints on attention allocation
   */
  private enforceResourceConstraints(nodeTensors: NodeTensor[]): void {
    const totalAttention = nodeTensors.reduce((sum, node) => sum + node.attention_weights.data[0], 0);
    
    if (totalAttention > this.resourceBudget) {
      // Scale down all attention values proportionally
      const scale = this.resourceBudget / totalAttention;
      nodeTensors.forEach(node => {
        node.attention_weights.data[0] *= scale;
      });
    }
  }

  /**
   * Record attention flows for monitoring
   */
  private recordAttentionFlows(_nodes: NodeTensor[], _links: LinkTensor[]): void {
    // Keep only recent history
    const cutoffTime = Date.now() - 60000; // 1 minute
    this.attentionHistory = this.attentionHistory.filter(flow => flow.timestamp > cutoffTime);
  }

  /**
   * Calculate attention statistics
   */
  private calculateAttentionStats(nodeTensors: NodeTensor[]): AttentionStats {
    if (nodeTensors.length === 0) {
      return {
        total_attention: 0,
        average_attention: 0,
        attention_entropy: 0,
        resource_utilization: 0,
        gradient_norm: 0,
        convergence_rate: 0
      };
    }

    const attentionValues = nodeTensors.map(node => node.attention_weights.data[0]);
    const totalAttention = attentionValues.reduce((sum, val) => sum + val, 0);
    const averageAttention = totalAttention / nodeTensors.length;
    
    // Calculate entropy
    let entropy = 0;
    if (totalAttention > 0) {
      attentionValues.forEach(val => {
        const p = val / totalAttention;
        if (p > 0) {
          entropy -= p * Math.log2(p);
        }
      });
    }
    
    // Calculate gradient norm
    let gradientNorm = 0;
    let gradientCount = 0;
    nodeTensors.forEach(node => {
      if (node.attention_weights.gradient) {
        gradientNorm += node.attention_weights.gradient.reduce((sum, g) => sum + g * g, 0);
        gradientCount++;
      }
    });
    gradientNorm = gradientCount > 0 ? Math.sqrt(gradientNorm / gradientCount) : 0;
    
    // Calculate convergence rate (simplified)
    const convergenceRate = Math.exp(-gradientNorm);
    
    return {
      total_attention: totalAttention,
      average_attention: averageAttention,
      attention_entropy: entropy,
      resource_utilization: totalAttention / this.resourceBudget,
      gradient_norm: gradientNorm,
      convergence_rate: convergenceRate
    };
  }

  /**
   * Get attention flow history
   */
  getAttentionFlows(): AttentionFlow[] {
    return [...this.attentionHistory];
  }

  /**
   * Get current attention statistics
   */
  getCurrentStats(nodeTensors: NodeTensor[]): AttentionStats {
    return this.calculateAttentionStats(nodeTensors);
  }

  /**
   * Get global attention tensor
   */
  getGlobalAttention(): AttentionTensor {
    return this.globalAttention;
  }

  /**
   * Reset attention allocation system
   */
  reset(): void {
    this.attentionHistory = [];
    this.resourceBudget = this.config.resource_budget;
    this.initializeGlobalAttention();
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<AttentionAllocationConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.globalAttention.ecan_parameters.rent_rate = this.config.rent_collection_rate;
    this.globalAttention.ecan_parameters.wage_rate = this.config.wage_distribution_rate;
    this.globalAttention.ecan_parameters.diffusion_rate = this.config.diffusion_strength;
  }
}