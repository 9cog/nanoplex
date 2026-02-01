/**
 * torch/nn module implementation
 * 
 * A TypeScript implementation of neural network modules inspired by
 * torch/nn (Torch7's neural network library)
 * https://github.com/torch/nn
 * 
 * This module provides:
 * - Module: Base class for all neural network components
 * - Containers: Sequential, Parallel, Concat for combining modules
 * - Linear layers: Linear, Dropout, Embedding, Flatten
 * - Activations: ReLU, Tanh, Sigmoid, Softmax, LogSoftmax, LeakyReLU
 * - Pooling: MaxPool1D, MaxPool2D, AvgPool1D, AvgPool2D
 * - Criterions: MSE, CrossEntropy, NLL, BCE, L1, SmoothL1
 * 
 * @example
 * ```typescript
 * import * as nn from './nn';
 * 
 * // Build a simple neural network
 * const model = new nn.Sequential([
 *   new nn.Linear(784, 128),
 *   new nn.ReLU(),
 *   new nn.Dropout(0.5),
 *   new nn.Linear(128, 10),
 *   new nn.Softmax()
 * ]);
 * 
 * // Forward pass
 * const output = model.forward(input);
 * 
 * // Compute loss
 * const criterion = new nn.MSECriterion();
 * const loss = criterion.forward(output, target);
 * 
 * // Backward pass
 * const gradOutput = criterion.backward(output, target);
 * model.backward(input, gradOutput);
 * 
 * // Update parameters
 * model.updateParameters(0.01);
 * ```
 */

// Core module interface and base class
export { Module, BaseModule } from './Module.js';

// Container modules
export { Sequential, Parallel, Concat } from './Container.js';

// Linear layers
export { Linear, Dropout, Embedding, Flatten } from './Linear.js';

// Activation functions
export {
  ReLU,
  Tanh,
  Sigmoid,
  Softmax,
  LogSoftmax,
  LeakyReLU
} from './Activation.js';

// Pooling layers
export {
  MaxPool1D,
  MaxPool2D,
  AvgPool1D,
  AvgPool2D
} from './Pooling.js';

// Loss functions (Criterions)
export {
  Criterion,
  MSECriterion,
  CrossEntropyCriterion,
  NLLCriterion,
  BCECriterion,
  L1Criterion,
  SmoothL1Criterion
} from './Criterion.js';

/**
 * Initialize module parameters using Xavier/Glorot initialization
 */
export function initXavier(module: Module): void {
  for (const param of module.parameters) {
    if (param.shape.length >= 2) {
      const fanIn = param.shape[1];
      const fanOut = param.shape[0];
      const std = Math.sqrt(2.0 / (fanIn + fanOut));
      
      for (let i = 0; i < param.data.length; i++) {
        param.data[i] = (Math.random() - 0.5) * 2 * std;
      }
    }
  }
}

/**
 * Initialize module parameters using He initialization
 */
export function initHe(module: Module): void {
  for (const param of module.parameters) {
    if (param.shape.length >= 2) {
      const fanIn = param.shape[1];
      const std = Math.sqrt(2.0 / fanIn);
      
      for (let i = 0; i < param.data.length; i++) {
        param.data[i] = (Math.random() - 0.5) * 2 * std;
      }
    }
  }
}

/**
 * Initialize module parameters with zeros
 */
export function initZeros(module: Module): void {
  for (const param of module.parameters) {
    param.data.fill(0);
  }
}

/**
 * Initialize module parameters with ones
 */
export function initOnes(module: Module): void {
  for (const param of module.parameters) {
    param.data.fill(1);
  }
}

/**
 * Initialize module parameters with uniform distribution
 */
export function initUniform(module: Module, a: number = -0.1, b: number = 0.1): void {
  for (const param of module.parameters) {
    for (let i = 0; i < param.data.length; i++) {
      param.data[i] = a + (b - a) * Math.random();
    }
  }
}

/**
 * Initialize module parameters with normal distribution
 */
export function initNormal(module: Module, mean: number = 0, std: number = 0.1): void {
  for (const param of module.parameters) {
    for (let i = 0; i < param.data.length; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      param.data[i] = mean + std * z;
    }
  }
}
