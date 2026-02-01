/**
 * Activation Functions for torch/nn
 * 
 * Common activation functions used in neural networks
 * Based on torch/nn activations
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';
import { BaseModule } from './Module.js';

/**
 * ReLU activation: f(x) = max(0, x)
 */
export class ReLU extends BaseModule {
  name = 'ReLU';
  private inputCache?: GgmlTensor;
  
  forward(input: GgmlTensor): GgmlTensor {
    if (this.training) {
      this.inputCache = input;
    }
    
    const output = this.createTensor(input.shape, 'relu');
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = Math.max(0, input.data[i]);
    }
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const gradInput = this.createTensor(input.shape, 'grad_relu');
    const inputVal = this.inputCache || input;
    
    for (let i = 0; i < gradOutput.data.length; i++) {
      gradInput.data[i] = inputVal.data[i] > 0 ? gradOutput.data[i] : 0;
    }
    return gradInput;
  }
}

/**
 * Tanh activation: f(x) = tanh(x)
 */
export class Tanh extends BaseModule {
  name = 'Tanh';
  private outputCache?: GgmlTensor;
  
  forward(input: GgmlTensor): GgmlTensor {
    const output = this.createTensor(input.shape, 'tanh');
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = Math.tanh(input.data[i]);
    }
    
    if (this.training) {
      this.outputCache = output;
    }
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const gradInput = this.createTensor(input.shape, 'grad_tanh');
    const outputVal = this.outputCache;
    
    if (outputVal) {
      for (let i = 0; i < gradOutput.data.length; i++) {
        const tanhVal = outputVal.data[i];
        gradInput.data[i] = gradOutput.data[i] * (1 - tanhVal * tanhVal);
      }
    } else {
      // Recompute if cache not available
      for (let i = 0; i < gradOutput.data.length; i++) {
        const tanhVal = Math.tanh(input.data[i]);
        gradInput.data[i] = gradOutput.data[i] * (1 - tanhVal * tanhVal);
      }
    }
    
    return gradInput;
  }
}

/**
 * Sigmoid activation: f(x) = 1 / (1 + exp(-x))
 */
export class Sigmoid extends BaseModule {
  name = 'Sigmoid';
  private outputCache?: GgmlTensor;
  
  forward(input: GgmlTensor): GgmlTensor {
    const output = this.createTensor(input.shape, 'sigmoid');
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = 1 / (1 + Math.exp(-input.data[i]));
    }
    
    if (this.training) {
      this.outputCache = output;
    }
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const gradInput = this.createTensor(input.shape, 'grad_sigmoid');
    const outputVal = this.outputCache;
    
    if (outputVal) {
      for (let i = 0; i < gradOutput.data.length; i++) {
        const sigmoidVal = outputVal.data[i];
        gradInput.data[i] = gradOutput.data[i] * sigmoidVal * (1 - sigmoidVal);
      }
    } else {
      // Recompute if cache not available
      for (let i = 0; i < gradOutput.data.length; i++) {
        const sigmoidVal = 1 / (1 + Math.exp(-input.data[i]));
        gradInput.data[i] = gradOutput.data[i] * sigmoidVal * (1 - sigmoidVal);
      }
    }
    
    return gradInput;
  }
}

/**
 * Softmax activation: f(x_i) = exp(x_i) / sum(exp(x_j))
 */
export class Softmax extends BaseModule {
  name = 'Softmax';
  private outputCache?: GgmlTensor;
  private dim: number;
  
  constructor(dim: number = 1) {
    super();
    this.dim = dim;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    const output = this.createTensor(input.shape, 'softmax');
    
    if (input.shape.length === 2 && this.dim === 1) {
      const [batchSize, features] = input.shape;
      
      for (let i = 0; i < batchSize; i++) {
        // Find max for numerical stability
        let maxVal = -Infinity;
        for (let j = 0; j < features; j++) {
          maxVal = Math.max(maxVal, input.data[i * features + j]);
        }
        
        // Compute exp and sum
        let sum = 0;
        for (let j = 0; j < features; j++) {
          const expVal = Math.exp(input.data[i * features + j] - maxVal);
          output.data[i * features + j] = expVal;
          sum += expVal;
        }
        
        // Normalize
        for (let j = 0; j < features; j++) {
          output.data[i * features + j] /= sum;
        }
      }
    }
    
    if (this.training) {
      this.outputCache = output;
    }
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const gradInput = this.createTensor(input.shape, 'grad_softmax');
    const outputVal = this.outputCache;
    
    if (!outputVal) {
      throw new Error('Softmax backward requires cached output');
    }
    
    if (input.shape.length === 2 && this.dim === 1) {
      const [batchSize, features] = input.shape;
      
      for (let i = 0; i < batchSize; i++) {
        // Compute Jacobian-vector product
        for (let j = 0; j < features; j++) {
          let sum = 0;
          for (let k = 0; k < features; k++) {
            const softmaxJ = outputVal.data[i * features + j];
            const softmaxK = outputVal.data[i * features + k];
            const delta = j === k ? 1 : 0;
            sum += gradOutput.data[i * features + k] * softmaxJ * (delta - softmaxK);
          }
          gradInput.data[i * features + j] = sum;
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * LogSoftmax activation: f(x_i) = log(exp(x_i) / sum(exp(x_j)))
 */
export class LogSoftmax extends BaseModule {
  name = 'LogSoftmax';
  private outputCache?: GgmlTensor;
  private dim: number;
  
  constructor(dim: number = 1) {
    super();
    this.dim = dim;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    const output = this.createTensor(input.shape, 'logsoftmax');
    
    if (input.shape.length === 2 && this.dim === 1) {
      const [batchSize, features] = input.shape;
      
      for (let i = 0; i < batchSize; i++) {
        // Find max for numerical stability
        let maxVal = -Infinity;
        for (let j = 0; j < features; j++) {
          maxVal = Math.max(maxVal, input.data[i * features + j]);
        }
        
        // Compute log-sum-exp
        let sumExp = 0;
        for (let j = 0; j < features; j++) {
          sumExp += Math.exp(input.data[i * features + j] - maxVal);
        }
        const logSumExp = Math.log(sumExp) + maxVal;
        
        // Compute log softmax
        for (let j = 0; j < features; j++) {
          output.data[i * features + j] = input.data[i * features + j] - logSumExp;
        }
      }
    }
    
    if (this.training) {
      this.outputCache = output;
    }
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const gradInput = this.createTensor(input.shape, 'grad_logsoftmax');
    const outputVal = this.outputCache;
    
    if (!outputVal) {
      throw new Error('LogSoftmax backward requires cached output');
    }
    
    if (input.shape.length === 2 && this.dim === 1) {
      const [batchSize, features] = input.shape;
      
      for (let i = 0; i < batchSize; i++) {
        // Compute sum of gradOutput
        let sum = 0;
        for (let j = 0; j < features; j++) {
          sum += gradOutput.data[i * features + j];
        }
        
        // Gradient: gradOutput - exp(logsoftmax) * sum(gradOutput)
        for (let j = 0; j < features; j++) {
          const softmaxVal = Math.exp(outputVal.data[i * features + j]);
          gradInput.data[i * features + j] = gradOutput.data[i * features + j] - softmaxVal * sum;
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * LeakyReLU activation: f(x) = max(alpha * x, x)
 */
export class LeakyReLU extends BaseModule {
  name = 'LeakyReLU';
  private alpha: number;
  private inputCache?: GgmlTensor;
  
  constructor(alpha: number = 0.01) {
    super();
    this.alpha = alpha;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    if (this.training) {
      this.inputCache = input;
    }
    
    const output = this.createTensor(input.shape, 'leaky_relu');
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = input.data[i] > 0 ? input.data[i] : this.alpha * input.data[i];
    }
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const gradInput = this.createTensor(input.shape, 'grad_leaky_relu');
    const inputVal = this.inputCache || input;
    
    for (let i = 0; i < gradOutput.data.length; i++) {
      gradInput.data[i] = inputVal.data[i] > 0 ? gradOutput.data[i] : this.alpha * gradOutput.data[i];
    }
    return gradInput;
  }
}
