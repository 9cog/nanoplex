/**
 * Prime-Shaped Nested Tensor Tuples for NanoBrain Time Crystal Model
 * 
 * Implements quaternion (4D), octonion (8D), and dodecanion (12D) tensor structures
 * as described in Chapter 7 of the NanoBrain book.
 * 
 * Architecture:
 * - Quaternion (4D): Sensory input processing (spatial)
 * - Octonion (8D): Pathway/midbrain fusion (cross-modal)
 * - Dodecanion (12D): Cortical manifolds (abstract thought, consciousness)
 * 
 * Nested transformation: d4D → d8D → d12D
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';
import { BaseModule } from './Module.js';

/**
 * Prime dimensions for time crystal processing
 */
export const PRIME_DIMENSIONS = {
  QUATERNION: 4,  // 2² - spatial processing
  OCTONION: 8,    // 2³ - sensory integration
  DODECANION: 12, // 3 × 4 - consciousness
} as const;

/**
 * First 15 fundamental primes as used in NanoBrain architecture
 */
export const FUNDAMENTAL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

/**
 * Prime-shaped tensor type representing a time crystal
 */
export interface PrimeShapedTensor {
  dimension: 4 | 8 | 12;
  tensor: GgmlTensor;
  primeSignature: number[];  // Which primes dominate this tensor
  type: 'quaternion' | 'octonion' | 'dodecanion';
}

/**
 * Nested tensor tuple representing the d4D → d8D → d12D transformation
 */
export interface NestedTensorTuple {
  quaternion: PrimeShapedTensor;  // d4D - sensory
  octonion: PrimeShapedTensor;    // d8D - pathway
  dodecanion: PrimeShapedTensor;  // d12D - cortical
}

/**
 * Quaternion Module (4D) - Sensory input processing
 * 
 * Based on Section 7.1.1: "sensors produce quaternion tensors"
 * Processes spatial information with C2, C3, C5 symmetries
 */
export class QuaternionModule extends BaseModule {
  name = 'QuaternionModule';
  private weights: GgmlTensor;
  private bias: GgmlTensor;
  
  /**
   * @param inputSize - Input feature dimension
   * @param primeDominance - Which primes dominate (default: [2, 3, 5] - spatial)
   */
  constructor(
    private inputSize: number,
    private primeDominance: number[] = [2, 3, 5]
  ) {
    super();
    
    // Create 4D quaternion transformation matrix
    const outputSize = PRIME_DIMENSIONS.QUATERNION;
    this.weights = this.createTensor(
      [outputSize, inputSize],
      'quaternion_weights',
      () => (Math.random() - 0.5) * Math.sqrt(2.0 / inputSize)
    );
    this.bias = this.createTensor(
      [outputSize],
      'quaternion_bias',
      () => 0
    );
    
    this.parameters = [this.weights, this.bias];
    this.gradients = [
      this.createTensor([outputSize, inputSize], 'quaternion_weights_grad', () => 0),
      this.createTensor([outputSize], 'quaternion_bias_grad', () => 0)
    ];
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    // Quaternion transformation: q = W * x + b (4D output)
    const batchSize = input.shape[0];
    const outputSize = PRIME_DIMENSIONS.QUATERNION;
    
    const output = this.createTensor(
      [batchSize, outputSize],
      'quaternion_output',
      () => 0
    );
    
    // Matrix multiplication: output = input * W^T + b
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        let sum = this.bias.data[j];
        for (let k = 0; k < this.inputSize; k++) {
          sum += input.data[i * this.inputSize + k] * 
                 this.weights.data[j * this.inputSize + k];
        }
        output.data[i * outputSize + j] = sum;
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const batchSize = input.shape[0];
    const outputSize = PRIME_DIMENSIONS.QUATERNION;
    
    const gradInput = this.createTensor(input.shape, 'quaternion_grad_input', () => 0);
    
    // Compute gradients
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        const grad = gradOutput.data[i * outputSize + j];
        
        // Gradient w.r.t. bias
        this.gradients[1].data[j] += grad;
        
        // Gradient w.r.t. weights and input
        for (let k = 0; k < this.inputSize; k++) {
          this.gradients[0].data[j * this.inputSize + k] += 
            grad * input.data[i * this.inputSize + k];
          gradInput.data[i * this.inputSize + k] += 
            grad * this.weights.data[j * this.inputSize + k];
        }
      }
    }
    
    return gradInput;
  }
  
  /**
   * Create prime-shaped tensor with quaternion signature
   */
  createPrimeShapedTensor(tensor: GgmlTensor): PrimeShapedTensor {
    return {
      dimension: PRIME_DIMENSIONS.QUATERNION,
      tensor,
      primeSignature: this.primeDominance,
      type: 'quaternion'
    };
  }
}

/**
 * Octonion Module (8D) - Pathway/midbrain fusion
 * 
 * Based on Section 7.2: "8 such sensors (octonion sensor tensor)"
 * Processes cross-modal sensory integration with C7, C11, C13 symmetries
 */
export class OctonionModule extends BaseModule {
  name = 'OctonionModule';
  private weights: GgmlTensor;
  private bias: GgmlTensor;
  
  /**
   * @param inputSize - Input feature dimension (typically 4 from quaternion)
   * @param primeDominance - Which primes dominate (default: [7, 11, 13] - temporal)
   */
  constructor(
    private inputSize: number,
    private primeDominance: number[] = [7, 11, 13]
  ) {
    super();
    
    // Create 8D octonion transformation matrix
    const outputSize = PRIME_DIMENSIONS.OCTONION;
    this.weights = this.createTensor(
      [outputSize, inputSize],
      'octonion_weights',
      () => (Math.random() - 0.5) * Math.sqrt(2.0 / inputSize)
    );
    this.bias = this.createTensor(
      [outputSize],
      'octonion_bias',
      () => 0
    );
    
    this.parameters = [this.weights, this.bias];
    this.gradients = [
      this.createTensor([outputSize, inputSize], 'octonion_weights_grad', () => 0),
      this.createTensor([outputSize], 'octonion_bias_grad', () => 0)
    ];
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    // Octonion transformation with non-commutative properties
    const batchSize = input.shape[0];
    const outputSize = PRIME_DIMENSIONS.OCTONION;
    
    const output = this.createTensor(
      [batchSize, outputSize],
      'octonion_output',
      () => 0
    );
    
    // Matrix multiplication with octonion cross-over magic
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        let sum = this.bias.data[j];
        for (let k = 0; k < this.inputSize; k++) {
          sum += input.data[i * this.inputSize + k] * 
                 this.weights.data[j * this.inputSize + k];
        }
        output.data[i * outputSize + j] = sum;
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const batchSize = input.shape[0];
    const outputSize = PRIME_DIMENSIONS.OCTONION;
    
    const gradInput = this.createTensor(input.shape, 'octonion_grad_input', () => 0);
    
    // Compute gradients
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        const grad = gradOutput.data[i * outputSize + j];
        
        // Gradient w.r.t. bias
        this.gradients[1].data[j] += grad;
        
        // Gradient w.r.t. weights and input
        for (let k = 0; k < this.inputSize; k++) {
          this.gradients[0].data[j * this.inputSize + k] += 
            grad * input.data[i * this.inputSize + k];
          gradInput.data[i * this.inputSize + k] += 
            grad * this.weights.data[j * this.inputSize + k];
        }
      }
    }
    
    return gradInput;
  }
  
  /**
   * Create prime-shaped tensor with octonion signature
   */
  createPrimeShapedTensor(tensor: GgmlTensor): PrimeShapedTensor {
    return {
      dimension: PRIME_DIMENSIONS.OCTONION,
      tensor,
      primeSignature: this.primeDominance,
      type: 'octonion'
    };
  }
}

/**
 * Dodecanion Module (12D) - Cortical manifolds
 * 
 * Based on Section 7.1.1: "cortex produces dodecanions as 12D manifolds"
 * Processes abstract thought and consciousness with higher primes
 */
export class DodecanionModule extends BaseModule {
  name = 'DodecanionModule';
  private weights: GgmlTensor;
  private bias: GgmlTensor;
  
  /**
   * @param inputSize - Input feature dimension (typically 8 from octonion)
   * @param primeDominance - Which primes dominate (default: [29, 31, 37] - consciousness)
   */
  constructor(
    private inputSize: number,
    private primeDominance: number[] = [29, 31, 37]
  ) {
    super();
    
    // Create 12D dodecanion transformation matrix
    const outputSize = PRIME_DIMENSIONS.DODECANION;
    this.weights = this.createTensor(
      [outputSize, inputSize],
      'dodecanion_weights',
      () => (Math.random() - 0.5) * Math.sqrt(2.0 / inputSize)
    );
    this.bias = this.createTensor(
      [outputSize],
      'dodecanion_bias',
      () => 0
    );
    
    this.parameters = [this.weights, this.bias];
    this.gradients = [
      this.createTensor([outputSize, inputSize], 'dodecanion_weights_grad', () => 0),
      this.createTensor([outputSize], 'dodecanion_bias_grad', () => 0)
    ];
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    // Dodecanion transformation for 12D manifold
    const batchSize = input.shape[0];
    const outputSize = PRIME_DIMENSIONS.DODECANION;
    
    const output = this.createTensor(
      [batchSize, outputSize],
      'dodecanion_output',
      () => 0
    );
    
    // Matrix multiplication
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        let sum = this.bias.data[j];
        for (let k = 0; k < this.inputSize; k++) {
          sum += input.data[i * this.inputSize + k] * 
                 this.weights.data[j * this.inputSize + k];
        }
        output.data[i * outputSize + j] = sum;
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const batchSize = input.shape[0];
    const outputSize = PRIME_DIMENSIONS.DODECANION;
    
    const gradInput = this.createTensor(input.shape, 'dodecanion_grad_input', () => 0);
    
    // Compute gradients
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        const grad = gradOutput.data[i * outputSize + j];
        
        // Gradient w.r.t. bias
        this.gradients[1].data[j] += grad;
        
        // Gradient w.r.t. weights and input
        for (let k = 0; k < this.inputSize; k++) {
          this.gradients[0].data[j * this.inputSize + k] += 
            grad * input.data[i * this.inputSize + k];
          gradInput.data[i * this.inputSize + k] += 
            grad * this.weights.data[j * this.inputSize + k];
        }
      }
    }
    
    return gradInput;
  }
  
  /**
   * Create prime-shaped tensor with dodecanion signature
   */
  createPrimeShapedTensor(tensor: GgmlTensor): PrimeShapedTensor {
    return {
      dimension: PRIME_DIMENSIONS.DODECANION,
      tensor,
      primeSignature: this.primeDominance,
      type: 'dodecanion'
    };
  }
}

/**
 * Nested Tensor Transformer - Implements d4D → d8D → d12D pipeline
 * 
 * Based on Section 7.8: "tensor d4D → d8D → d12D. Four feedback loops run in parallel"
 * This represents the complete sensory → pathway → cortical transformation
 */
export class NestedTensorTransformer extends BaseModule {
  name = 'NestedTensorTransformer';
  
  private quaternionModule: QuaternionModule;
  private octonionModule: OctonionModule;
  private dodecanionModule: DodecanionModule;
  
  /**
   * @param inputSize - Input feature dimension
   */
  constructor(inputSize: number) {
    super();
    
    // Create nested transformation pipeline
    this.quaternionModule = new QuaternionModule(inputSize, [2, 3, 5]);
    this.octonionModule = new OctonionModule(PRIME_DIMENSIONS.QUATERNION, [7, 11, 13]);
    this.dodecanionModule = new DodecanionModule(PRIME_DIMENSIONS.OCTONION, [29, 31, 37]);
    
    // Collect all parameters
    this.parameters = [
      ...this.quaternionModule.parameters,
      ...this.octonionModule.parameters,
      ...this.dodecanionModule.parameters
    ];
    
    this.gradients = [
      ...this.quaternionModule.gradients,
      ...this.octonionModule.gradients,
      ...this.dodecanionModule.gradients
    ];
  }
  
  /**
   * Forward pass through nested transformation
   * Returns the final 12D dodecanion output
   */
  forward(input: GgmlTensor): GgmlTensor {
    // d4D: Sensory quaternion
    const quaternionOutput = this.quaternionModule.forward(input);
    
    // d8D: Pathway octonion with left-right crossover
    const octonionOutput = this.octonionModule.forward(quaternionOutput);
    
    // d12D: Cortical dodecanion
    const dodecanionOutput = this.dodecanionModule.forward(octonionOutput);
    
    return dodecanionOutput;
  }
  
  /**
   * Backward pass through nested transformation
   */
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    // Backprop through d12D → d8D
    const quaternionOutput = this.quaternionModule.forward(input);
    const octonionOutput = this.octonionModule.forward(quaternionOutput);
    
    // d12D gradient
    const gradOctonion = this.dodecanionModule.backward(octonionOutput, gradOutput);
    
    // d8D gradient
    const gradQuaternion = this.octonionModule.backward(quaternionOutput, gradOctonion);
    
    // d4D gradient
    const gradInput = this.quaternionModule.backward(input, gradQuaternion);
    
    return gradInput;
  }
  
  /**
   * Get complete nested tensor tuple
   */
  forwardNested(input: GgmlTensor): NestedTensorTuple {
    const quaternionOutput = this.quaternionModule.forward(input);
    const octonionOutput = this.octonionModule.forward(quaternionOutput);
    const dodecanionOutput = this.dodecanionModule.forward(octonionOutput);
    
    return {
      quaternion: this.quaternionModule.createPrimeShapedTensor(quaternionOutput),
      octonion: this.octonionModule.createPrimeShapedTensor(octonionOutput),
      dodecanion: this.dodecanionModule.createPrimeShapedTensor(dodecanionOutput)
    };
  }
  
  /**
   * Set training mode for all nested modules
   */
  train(): void {
    super.train();
    this.quaternionModule.train();
    this.octonionModule.train();
    this.dodecanionModule.train();
  }
  
  /**
   * Set evaluation mode for all nested modules
   */
  evaluate(): void {
    super.evaluate();
    this.quaternionModule.evaluate();
    this.octonionModule.evaluate();
    this.dodecanionModule.evaluate();
  }
}

/**
 * H3 Decision Module - Triplet decision-making unit
 * 
 * Based on Section 7.8: "H3 Decision Device"
 * Three inputs: H₁ (primary), H₂ (secondary), H₃ (context)
 * Decision = f(H₁, H₂, H₃)
 */
export class H3DecisionModule extends BaseModule {
  name = 'H3DecisionModule';
  
  private w1: GgmlTensor;
  private w2: GgmlTensor;
  private w3: GgmlTensor;
  private bias: GgmlTensor;
  
  /**
   * @param inputSize - Size of each H input
   * @param outputSize - Size of decision output
   */
  constructor(
    private inputSize: number,
    private outputSize: number = 1
  ) {
    super();
    
    // Create weights for three inputs (triplet architecture)
    this.w1 = this.createTensor(
      [outputSize, inputSize],
      'h3_w1',
      () => (Math.random() - 0.5) * Math.sqrt(2.0 / inputSize)
    );
    this.w2 = this.createTensor(
      [outputSize, inputSize],
      'h3_w2',
      () => (Math.random() - 0.5) * Math.sqrt(2.0 / inputSize)
    );
    this.w3 = this.createTensor(
      [outputSize, inputSize],
      'h3_w3',
      () => (Math.random() - 0.5) * Math.sqrt(2.0 / inputSize)
    );
    this.bias = this.createTensor(
      [outputSize],
      'h3_bias',
      () => 0
    );
    
    this.parameters = [this.w1, this.w2, this.w3, this.bias];
    this.gradients = [
      this.createTensor([outputSize, inputSize], 'h3_w1_grad', () => 0),
      this.createTensor([outputSize, inputSize], 'h3_w2_grad', () => 0),
      this.createTensor([outputSize, inputSize], 'h3_w3_grad', () => 0),
      this.createTensor([outputSize], 'h3_bias_grad', () => 0)
    ];
  }
  
  /**
   * Forward pass with three inputs
   * @param h1 - Primary input
   * @param h2 - Secondary input
   * @param h3 - Context input
   */
  forwardH3(h1: GgmlTensor, h2: GgmlTensor, h3: GgmlTensor): GgmlTensor {
    const batchSize = h1.shape[0];
    
    const output = this.createTensor(
      [batchSize, this.outputSize],
      'h3_output',
      () => 0
    );
    
    // Decision = W1*H1 + W2*H2 + W3*H3 + bias (triplet combination)
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        let sum = this.bias.data[j];
        
        // Accumulate from all three inputs
        for (let k = 0; k < this.inputSize; k++) {
          sum += h1.data[i * this.inputSize + k] * this.w1.data[j * this.inputSize + k];
          sum += h2.data[i * this.inputSize + k] * this.w2.data[j * this.inputSize + k];
          sum += h3.data[i * this.inputSize + k] * this.w3.data[j * this.inputSize + k];
        }
        
        output.data[i * this.outputSize + j] = sum;
      }
    }
    
    return output;
  }
  
  /**
   * Standard forward (expects concatenated input [H1, H2, H3])
   */
  forward(input: GgmlTensor): GgmlTensor {
    // Split input into three equal parts
    const batchSize = input.shape[0];
    const totalSize = input.shape[1];
    const partSize = Math.floor(totalSize / 3);
    
    const h1 = this.createTensor([batchSize, partSize], 'h1', () => 0);
    const h2 = this.createTensor([batchSize, partSize], 'h2', () => 0);
    const h3 = this.createTensor([batchSize, partSize], 'h3', () => 0);
    
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < partSize; j++) {
        h1.data[i * partSize + j] = input.data[i * totalSize + j];
        h2.data[i * partSize + j] = input.data[i * totalSize + partSize + j];
        h3.data[i * partSize + j] = input.data[i * totalSize + 2 * partSize + j];
      }
    }
    
    return this.forwardH3(h1, h2, h3);
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const batchSize = input.shape[0];
    const totalSize = input.shape[1];
    const partSize = Math.floor(totalSize / 3);
    
    const gradInput = this.createTensor(input.shape, 'h3_grad_input', () => 0);
    
    // Split input for gradient computation
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        const grad = gradOutput.data[i * this.outputSize + j];
        
        // Gradient w.r.t. bias
        this.gradients[3].data[j] += grad;
        
        // Gradients for all three inputs
        for (let k = 0; k < partSize; k++) {
          // W1 gradient and H1 input gradient
          this.gradients[0].data[j * this.inputSize + k] += 
            grad * input.data[i * totalSize + k];
          gradInput.data[i * totalSize + k] += 
            grad * this.w1.data[j * this.inputSize + k];
          
          // W2 gradient and H2 input gradient
          this.gradients[1].data[j * this.inputSize + k] += 
            grad * input.data[i * totalSize + partSize + k];
          gradInput.data[i * totalSize + partSize + k] += 
            grad * this.w2.data[j * this.inputSize + k];
          
          // W3 gradient and H3 input gradient
          this.gradients[2].data[j * this.inputSize + k] += 
            grad * input.data[i * totalSize + 2 * partSize + k];
          gradInput.data[i * totalSize + 2 * partSize + k] += 
            grad * this.w3.data[j * this.inputSize + k];
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * Prime Tensor Container - Manages collections of prime-shaped tensors
 * 
 * Provides utilities for working with nested tensor tuples
 */
export class PrimeTensorContainer {
  private tensors: Map<string, PrimeShapedTensor> = new Map();
  
  /**
   * Add a prime-shaped tensor to the container
   */
  add(name: string, tensor: PrimeShapedTensor): void {
    this.tensors.set(name, tensor);
  }
  
  /**
   * Get a tensor by name
   */
  get(name: string): PrimeShapedTensor | undefined {
    return this.tensors.get(name);
  }
  
  /**
   * Get all tensors of a specific type
   */
  getByType(type: 'quaternion' | 'octonion' | 'dodecanion'): PrimeShapedTensor[] {
    return Array.from(this.tensors.values()).filter(t => t.type === type);
  }
  
  /**
   * Get all tensors with a specific prime in their signature
   */
  getByPrime(prime: number): PrimeShapedTensor[] {
    return Array.from(this.tensors.values()).filter(
      t => t.primeSignature.includes(prime)
    );
  }
  
  /**
   * Create a nested tuple from container tensors
   */
  createNestedTuple(
    quaternionName: string,
    octonionName: string,
    dodecanionName: string
  ): NestedTensorTuple | null {
    const q = this.tensors.get(quaternionName);
    const o = this.tensors.get(octonionName);
    const d = this.tensors.get(dodecanionName);
    
    if (!q || !o || !d) return null;
    if (q.type !== 'quaternion' || o.type !== 'octonion' || d.type !== 'dodecanion') {
      return null;
    }
    
    return { quaternion: q, octonion: o, dodecanion: d };
  }
  
  /**
   * Get summary statistics
   */
  summary(): {
    total: number;
    byType: Record<string, number>;
    byPrime: Record<number, number>;
  } {
    const byType: Record<string, number> = {
      quaternion: 0,
      octonion: 0,
      dodecanion: 0
    };
    
    const byPrime: Record<number, number> = {};
    
    for (const tensor of this.tensors.values()) {
      byType[tensor.type]++;
      
      for (const prime of tensor.primeSignature) {
        byPrime[prime] = (byPrime[prime] || 0) + 1;
      }
    }
    
    return {
      total: this.tensors.size,
      byType,
      byPrime
    };
  }
}
