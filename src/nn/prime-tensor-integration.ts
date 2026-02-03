/**
 * Prime-Shaped Nested Tensor Tuples - Integration Example
 * 
 * Demonstrates how to use quaternion, octonion, and dodecanion modules
 * to build a complete time crystal brain model as described in Chapter 7.
 * 
 * This example shows:
 * 1. Creating sensory processing with quaternions (4D)
 * 2. Pathway fusion with octonions (8D)
 * 3. Cortical manifolds with dodecanions (12D)
 * 4. Decision-making with H3 triplet units
 * 5. Complete brain-inspired neural network architecture
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';
import {
  QuaternionModule,
  OctonionModule,
  DodecanionModule,
  NestedTensorTransformer,
  H3DecisionModule,
  PrimeTensorContainer,
  PRIME_DIMENSIONS,
  FUNDAMENTAL_PRIMES,
  PrimeShapedTensor,
  NestedTensorTuple
} from './PrimeShapedTensors.js';
import { Sequential } from './Container.js';
import { ReLU, Tanh } from './Activation.js';
import { MSECriterion } from './Criterion.js';

/**
 * H3 Triplet Scaling Factors
 * 
 * Based on Chapter 7 Section 7.8: H3 Decision Device
 * The three inputs (primary, secondary, context) are weighted differently
 * to represent their relative importance in decision-making.
 */
const H3_PRIMARY_SCALE = 1.0;    // Primary input (full weight)
const H3_SECONDARY_SCALE = 0.8;  // Secondary input (reduced emphasis)
const H3_CONTEXT_SCALE = 1.2;    // Context input (enhanced influence)

/**
 * Complete Time Crystal Brain Model
 * 
 * Implements the full architecture from Chapter 7:
 * - Sensory input → Quaternion (4D spatial)
 * - Pathway integration → Octonion (8D cross-modal)
 * - Cortical processing → Dodecanion (12D abstract thought)
 * - Decision output → H3 triplet decision unit
 */
export class TimeCrystalBrainModel {
  private nestedTransformer: NestedTensorTransformer;
  private h3Decision: H3DecisionModule;
  private container: PrimeTensorContainer;
  
  /**
   * @param inputSize - Size of sensory input
   * @param outputSize - Size of decision output
   */
  constructor(
    private inputSize: number,
    private outputSize: number = 1
  ) {
    // Create nested transformer for d4D → d8D → d12D
    this.nestedTransformer = new NestedTensorTransformer(inputSize);
    
    // Create H3 decision module
    this.h3Decision = new H3DecisionModule(
      PRIME_DIMENSIONS.DODECANION,
      outputSize
    );
    
    // Create container for organizing tensors
    this.container = new PrimeTensorContainer();
  }
  
  /**
   * Process sensory input through complete brain model
   */
  forward(input: GgmlTensor): {
    decision: GgmlTensor;
    nestedTuple: NestedTensorTuple;
  } {
    // Process through nested transformer
    const nestedTuple = this.nestedTransformer.forwardNested(input);
    
    // Store in container for analysis
    this.container.add('current_sensory', nestedTuple.quaternion);
    this.container.add('current_pathway', nestedTuple.octonion);
    this.container.add('current_cortical', nestedTuple.dodecanion);
    
    // Create H3 input from cortical output (triplet: primary, secondary, context)
    const cortical = nestedTuple.dodecanion.tensor;
    const batchSize = cortical.shape[0];
    
    // Replicate cortical output 3 times for H3 triplet
    const h3Input = this.createH3Input(cortical);
    
    // Make decision
    const decision = this.h3Decision.forward(h3Input);
    
    return {
      decision,
      nestedTuple
    };
  }
  
  /**
   * Backward pass for training
   */
  backward(
    input: GgmlTensor,
    gradDecision: GgmlTensor
  ): GgmlTensor {
    // Get intermediate tensors
    const nestedTuple = this.nestedTransformer.forwardNested(input);
    const cortical = nestedTuple.dodecanion.tensor;
    const h3Input = this.createH3Input(cortical);
    
    // Backward through H3
    const gradH3Input = this.h3Decision.backward(h3Input, gradDecision);
    
    // Extract gradient for cortical (take first 1/3 of H3 gradient)
    const gradCortical = this.extractCorticalGradient(gradH3Input);
    
    // Backward through nested transformer
    const gradInput = this.nestedTransformer.backward(input, gradCortical);
    
    return gradInput;
  }
  
  /**
   * Update all model parameters
   */
  updateParameters(learningRate: number): void {
    this.nestedTransformer.updateParameters(learningRate);
    this.h3Decision.updateParameters(learningRate);
  }
  
  /**
   * Zero all gradients
   */
  zeroGradParameters(): void {
    this.nestedTransformer.zeroGradParameters();
    this.h3Decision.zeroGradParameters();
  }
  
  /**
   * Set training mode
   */
  train(): void {
    this.nestedTransformer.train();
    this.h3Decision.train();
  }
  
  /**
   * Set evaluation mode
   */
  evaluate(): void {
    this.nestedTransformer.evaluate();
    this.h3Decision.evaluate();
  }
  
  /**
   * Get current tensor container
   */
  getTensorContainer(): PrimeTensorContainer {
    return this.container;
  }
  
  /**
   * Get model summary
   */
  summary(): string {
    const containerSummary = this.container.summary();
    return `
Time Crystal Brain Model Summary
================================
Input Size: ${this.inputSize}
Output Size: ${this.outputSize}

Architecture:
- Sensory Layer (Quaternion): 4D with primes [2, 3, 5]
- Pathway Layer (Octonion): 8D with primes [7, 11, 13]
- Cortical Layer (Dodecanion): 12D with primes [29, 31, 37]
- Decision Layer (H3): Triplet decision unit

Tensor Container:
- Total tensors: ${containerSummary.total}
- Quaternions: ${containerSummary.byType.quaternion}
- Octonions: ${containerSummary.byType.octonion}
- Dodecanions: ${containerSummary.byType.dodecanion}

Based on NanoBrain Chapter 7: Time Crystal Brain Model
    `.trim();
  }
  
  // Helper methods
  
  private createH3Input(cortical: GgmlTensor): GgmlTensor {
    const batchSize = cortical.shape[0];
    const corticalSize = cortical.shape[1];
    
    // Create input with 3 copies for H3 triplet
    const h3InputSize = corticalSize * 3;
    const h3Input: GgmlTensor = {
      id: `h3_input_${Math.random()}`,
      shape: [batchSize, h3InputSize],
      data: new Float32Array(batchSize * h3InputSize),
      dtype: 'f32',
      requires_grad: true,
      name: 'h3_input'
    };
    
    // Fill with cortical data (replicated 3 times with scaling)
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < corticalSize; j++) {
        const value = cortical.data[i * corticalSize + j];
        h3Input.data[i * h3InputSize + j] = value * H3_PRIMARY_SCALE;
        h3Input.data[i * h3InputSize + corticalSize + j] = value * H3_SECONDARY_SCALE;
        h3Input.data[i * h3InputSize + 2 * corticalSize + j] = value * H3_CONTEXT_SCALE;
      }
    }
    
    return h3Input;
  }
  
  private extractCorticalGradient(gradH3Input: GgmlTensor): GgmlTensor {
    const batchSize = gradH3Input.shape[0];
    const h3InputSize = gradH3Input.shape[1];
    const corticalSize = Math.floor(h3InputSize / 3);
    
    const gradCortical: GgmlTensor = {
      id: `grad_cortical_${Math.random()}`,
      shape: [batchSize, corticalSize],
      data: new Float32Array(batchSize * corticalSize),
      dtype: 'f32',
      requires_grad: true,
      name: 'grad_cortical'
    };
    
    // Sum gradients from all three H3 inputs (with matching scales)
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < corticalSize; j++) {
        gradCortical.data[i * corticalSize + j] = 
          gradH3Input.data[i * h3InputSize + j] * H3_PRIMARY_SCALE +
          gradH3Input.data[i * h3InputSize + corticalSize + j] * H3_SECONDARY_SCALE +
          gradH3Input.data[i * h3InputSize + 2 * corticalSize + j] * H3_CONTEXT_SCALE;
      }
    }
    
    return gradCortical;
  }
}

/**
 * Example: Cognitive Pattern Classifier using Prime Tensors
 * 
 * Uses quaternion-octonion-dodecanion architecture for pattern recognition
 */
export function createCognitivePatternClassifier(
  inputSize: number,
  hiddenSize: number,
  numClasses: number
): TimeCrystalBrainModel {
  return new TimeCrystalBrainModel(inputSize, numClasses);
}

/**
 * Example: Multi-sensory Integration Model
 * 
 * Demonstrates how different sensory modalities can be processed
 * through prime-shaped tensors
 */
export class MultiSensoryIntegrationModel {
  private visualModule: QuaternionModule;
  private auditoryModule: QuaternionModule;
  private tactileModule: QuaternionModule;
  private fusionModule: OctonionModule;
  private corticalModule: DodecanionModule;
  
  constructor(
    visualInputSize: number,
    auditoryInputSize: number,
    tactileInputSize: number
  ) {
    // Each sensory modality has its own quaternion processor
    this.visualModule = new QuaternionModule(visualInputSize, [2, 3, 5]);
    this.auditoryModule = new QuaternionModule(auditoryInputSize, [3, 5, 7]);
    this.tactileModule = new QuaternionModule(tactileInputSize, [2, 5, 7]);
    
    // Fuse all sensory streams in octonion space
    // 3 quaternions (4D each) = 12D input → 8D octonion
    this.fusionModule = new OctonionModule(12, [7, 11, 13]);
    
    // Final cortical processing
    this.corticalModule = new DodecanionModule(PRIME_DIMENSIONS.OCTONION, [29, 31, 37]);
  }
  
  forward(
    visualInput: GgmlTensor,
    auditoryInput: GgmlTensor,
    tactileInput: GgmlTensor
  ): {
    visualQuaternion: GgmlTensor;
    auditoryQuaternion: GgmlTensor;
    tactileQuaternion: GgmlTensor;
    fusedOctonion: GgmlTensor;
    corticalDodecanion: GgmlTensor;
  } {
    // Process each sensory modality
    const visualQuaternion = this.visualModule.forward(visualInput);
    const auditoryQuaternion = this.auditoryModule.forward(auditoryInput);
    const tactileQuaternion = this.tactileModule.forward(tactileInput);
    
    // Concatenate quaternions for fusion
    const fusionInput = this.concatenateQuaternions(
      visualQuaternion,
      auditoryQuaternion,
      tactileQuaternion
    );
    
    // Fuse in octonion space
    const fusedOctonion = this.fusionModule.forward(fusionInput);
    
    // Process in cortical dodecanion space
    const corticalDodecanion = this.corticalModule.forward(fusedOctonion);
    
    return {
      visualQuaternion,
      auditoryQuaternion,
      tactileQuaternion,
      fusedOctonion,
      corticalDodecanion
    };
  }
  
  private concatenateQuaternions(
    q1: GgmlTensor,
    q2: GgmlTensor,
    q3: GgmlTensor
  ): GgmlTensor {
    const batchSize = q1.shape[0];
    const quatSize = PRIME_DIMENSIONS.QUATERNION;
    
    const concatenated: GgmlTensor = {
      id: `concat_${Math.random()}`,
      shape: [batchSize, quatSize * 3],
      data: new Float32Array(batchSize * quatSize * 3),
      dtype: 'f32',
      requires_grad: true,
      name: 'concatenated_quaternions'
    };
    
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < quatSize; j++) {
        concatenated.data[i * quatSize * 3 + j] = q1.data[i * quatSize + j];
        concatenated.data[i * quatSize * 3 + quatSize + j] = q2.data[i * quatSize + j];
        concatenated.data[i * quatSize * 3 + 2 * quatSize + j] = q3.data[i * quatSize + j];
      }
    }
    
    return concatenated;
  }
}

/**
 * Training example for Time Crystal Brain Model
 */
export async function trainTimeCrystalModel(
  model: TimeCrystalBrainModel,
  trainingData: Array<{ input: GgmlTensor; target: GgmlTensor }>,
  epochs: number = 10,
  learningRate: number = 0.01
): Promise<number[]> {
  const losses: number[] = [];
  const criterion = new MSECriterion();
  
  console.log('Training Time Crystal Brain Model...');
  console.log(model.summary());
  console.log('');
  
  for (let epoch = 0; epoch < epochs; epoch++) {
    let epochLoss = 0;
    
    model.train();
    model.zeroGradParameters();
    
    for (const { input, target } of trainingData) {
      // Forward pass
      const { decision } = model.forward(input);
      
      // Compute loss
      const loss = criterion.forward(decision, target);
      epochLoss += loss;
      
      // Backward pass
      const gradDecision = criterion.backward(decision, target);
      model.backward(input, gradDecision);
    }
    
    // Update parameters
    model.updateParameters(learningRate);
    
    epochLoss /= trainingData.length;
    losses.push(epochLoss);
    
    if (epoch % 2 === 0) {
      console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${epochLoss.toFixed(6)}`);
    }
  }
  
  console.log('\nTraining completed!');
  return losses;
}

/**
 * Demonstration function
 */
const DEMO_SEPARATOR_WIDTH = 70;

export function demonstratePrimeTensors(): void {
  console.log('='.repeat(DEMO_SEPARATOR_WIDTH));
  console.log('Prime-Shaped Nested Tensor Tuples - Integration Demonstration');
  console.log('Based on NanoBrain Chapter 7: Time Crystal Brain Model');
  console.log('='.repeat(DEMO_SEPARATOR_WIDTH));
  console.log('');
  
  // Example 1: Simple cognitive classifier
  console.log('1. Creating Cognitive Pattern Classifier...');
  const classifier = createCognitivePatternClassifier(64, 128, 10);
  console.log(classifier.summary());
  console.log('');
  
  // Example 2: Multi-sensory integration
  console.log('2. Creating Multi-Sensory Integration Model...');
  const multiSensory = new MultiSensoryIntegrationModel(32, 24, 16);
  console.log('   ✓ Visual input: 32D → 4D quaternion');
  console.log('   ✓ Auditory input: 24D → 4D quaternion');
  console.log('   ✓ Tactile input: 16D → 4D quaternion');
  console.log('   ✓ Fusion: 12D → 8D octonion');
  console.log('   ✓ Cortical: 8D → 12D dodecanion');
  console.log('');
  
  // Example 3: Prime constants
  console.log('3. Fundamental Primes (First 15):');
  console.log(`   ${FUNDAMENTAL_PRIMES.join(', ')}`);
  console.log('');
  
  console.log('4. Prime Dimensions:');
  console.log(`   - Quaternion: ${PRIME_DIMENSIONS.QUATERNION}D (sensory)`);
  console.log(`   - Octonion: ${PRIME_DIMENSIONS.OCTONION}D (pathway)`);
  console.log(`   - Dodecanion: ${PRIME_DIMENSIONS.DODECANION}D (cortical)`);
  console.log('');
  
  console.log('='.repeat(DEMO_SEPARATOR_WIDTH));
  console.log('Demonstration completed successfully!');
  console.log('='.repeat(DEMO_SEPARATOR_WIDTH));
}

// Run demonstration if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstratePrimeTensors();
}
