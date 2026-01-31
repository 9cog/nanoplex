/**
 * Loss Functions (Criterions) for torch/nn
 * 
 * Loss functions for training neural networks
 * Based on torch/nn Criterion modules
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';

/**
 * Base criterion interface
 */
export interface Criterion {
  name: string;
  forward(input: GgmlTensor, target: GgmlTensor): number;
  backward(input: GgmlTensor, target: GgmlTensor): GgmlTensor;
}

/**
 * Mean Squared Error Loss: MSE = mean((input - target)^2)
 */
export class MSECriterion implements Criterion {
  name = 'MSECriterion';
  
  forward(input: GgmlTensor, target: GgmlTensor): number {
    let sum = 0;
    for (let i = 0; i < input.data.length; i++) {
      const diff = input.data[i] - target.data[i];
      sum += diff * diff;
    }
    return sum / input.data.length;
  }
  
  backward(input: GgmlTensor, target: GgmlTensor): GgmlTensor {
    const gradInput: GgmlTensor = {
      id: `grad_mse_${Math.random().toString(36).substr(2, 9)}`,
      shape: input.shape,
      data: new Float32Array(input.data.length),
      dtype: 'f32',
      requires_grad: true,
      name: 'grad_mse'
    };
    
    const scale = 2.0 / input.data.length;
    for (let i = 0; i < input.data.length; i++) {
      gradInput.data[i] = scale * (input.data[i] - target.data[i]);
    }
    
    return gradInput;
  }
}

/**
 * Cross Entropy Loss: -sum(target * log(input))
 * Expects input to be probabilities (after softmax)
 */
export class CrossEntropyCriterion implements Criterion {
  name = 'CrossEntropyCriterion';
  private eps = 1e-8; // For numerical stability
  
  forward(input: GgmlTensor, target: GgmlTensor): number {
    let sum = 0;
    
    if (input.shape.length === 2) {
      // Batch processing
      const [batchSize, numClasses] = input.shape;
      
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < numClasses; j++) {
          const prob = Math.max(input.data[i * numClasses + j], this.eps);
          sum -= target.data[i * numClasses + j] * Math.log(prob);
        }
      }
      
      return sum / batchSize;
    }
    
    return 0;
  }
  
  backward(input: GgmlTensor, target: GgmlTensor): GgmlTensor {
    const gradInput: GgmlTensor = {
      id: `grad_ce_${Math.random().toString(36).substr(2, 9)}`,
      shape: input.shape,
      data: new Float32Array(input.data.length),
      dtype: 'f32',
      requires_grad: true,
      name: 'grad_cross_entropy'
    };
    
    if (input.shape.length === 2) {
      const [batchSize, numClasses] = input.shape;
      const scale = 1.0 / batchSize;
      
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < numClasses; j++) {
          const prob = Math.max(input.data[i * numClasses + j], this.eps);
          gradInput.data[i * numClasses + j] = 
            -scale * target.data[i * numClasses + j] / prob;
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * Negative Log Likelihood Loss: -log(input[target])
 * Expects input to be log probabilities (after log softmax)
 * and target to be class indices
 */
export class NLLCriterion implements Criterion {
  name = 'NLLCriterion';
  
  forward(input: GgmlTensor, target: GgmlTensor): number {
    let sum = 0;
    
    if (input.shape.length === 2) {
      const [batchSize, numClasses] = input.shape;
      
      // Target contains class indices
      for (let i = 0; i < batchSize; i++) {
        const targetClass = Math.floor(target.data[i]);
        if (targetClass >= 0 && targetClass < numClasses) {
          sum -= input.data[i * numClasses + targetClass];
        }
      }
      
      return sum / batchSize;
    }
    
    return 0;
  }
  
  backward(input: GgmlTensor, target: GgmlTensor): GgmlTensor {
    const gradInput: GgmlTensor = {
      id: `grad_nll_${Math.random().toString(36).substr(2, 9)}`,
      shape: input.shape,
      data: new Float32Array(input.data.length),
      dtype: 'f32',
      requires_grad: true,
      name: 'grad_nll'
    };
    
    if (input.shape.length === 2) {
      const [batchSize, numClasses] = input.shape;
      const scale = -1.0 / batchSize;
      
      for (let i = 0; i < batchSize; i++) {
        const targetClass = Math.floor(target.data[i]);
        if (targetClass >= 0 && targetClass < numClasses) {
          gradInput.data[i * numClasses + targetClass] = scale;
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * Binary Cross Entropy Loss: -sum(target * log(input) + (1-target) * log(1-input))
 * For binary classification tasks
 */
export class BCECriterion implements Criterion {
  name = 'BCECriterion';
  private eps = 1e-8;
  
  forward(input: GgmlTensor, target: GgmlTensor): number {
    let sum = 0;
    
    for (let i = 0; i < input.data.length; i++) {
      const prob = Math.max(Math.min(input.data[i], 1 - this.eps), this.eps);
      const t = target.data[i];
      sum -= t * Math.log(prob) + (1 - t) * Math.log(1 - prob);
    }
    
    return sum / input.data.length;
  }
  
  backward(input: GgmlTensor, target: GgmlTensor): GgmlTensor {
    const gradInput: GgmlTensor = {
      id: `grad_bce_${Math.random().toString(36).substr(2, 9)}`,
      shape: input.shape,
      data: new Float32Array(input.data.length),
      dtype: 'f32',
      requires_grad: true,
      name: 'grad_bce'
    };
    
    const scale = 1.0 / input.data.length;
    
    for (let i = 0; i < input.data.length; i++) {
      const prob = Math.max(Math.min(input.data[i], 1 - this.eps), this.eps);
      const t = target.data[i];
      gradInput.data[i] = scale * ((prob - t) / (prob * (1 - prob)));
    }
    
    return gradInput;
  }
}

/**
 * L1 Loss (Mean Absolute Error): MAE = mean(|input - target|)
 */
export class L1Criterion implements Criterion {
  name = 'L1Criterion';
  
  forward(input: GgmlTensor, target: GgmlTensor): number {
    let sum = 0;
    for (let i = 0; i < input.data.length; i++) {
      sum += Math.abs(input.data[i] - target.data[i]);
    }
    return sum / input.data.length;
  }
  
  backward(input: GgmlTensor, target: GgmlTensor): GgmlTensor {
    const gradInput: GgmlTensor = {
      id: `grad_l1_${Math.random().toString(36).substr(2, 9)}`,
      shape: input.shape,
      data: new Float32Array(input.data.length),
      dtype: 'f32',
      requires_grad: true,
      name: 'grad_l1'
    };
    
    const scale = 1.0 / input.data.length;
    for (let i = 0; i < input.data.length; i++) {
      const diff = input.data[i] - target.data[i];
      gradInput.data[i] = scale * (diff > 0 ? 1 : diff < 0 ? -1 : 0);
    }
    
    return gradInput;
  }
}

/**
 * Smooth L1 Loss (Huber Loss): 
 * 0.5 * x^2 if |x| < 1, |x| - 0.5 otherwise, where x = input - target
 */
export class SmoothL1Criterion implements Criterion {
  name = 'SmoothL1Criterion';
  
  forward(input: GgmlTensor, target: GgmlTensor): number {
    let sum = 0;
    for (let i = 0; i < input.data.length; i++) {
      const diff = Math.abs(input.data[i] - target.data[i]);
      sum += diff < 1 ? 0.5 * diff * diff : diff - 0.5;
    }
    return sum / input.data.length;
  }
  
  backward(input: GgmlTensor, target: GgmlTensor): GgmlTensor {
    const gradInput: GgmlTensor = {
      id: `grad_smoothl1_${Math.random().toString(36).substr(2, 9)}`,
      shape: input.shape,
      data: new Float32Array(input.data.length),
      dtype: 'f32',
      requires_grad: true,
      name: 'grad_smooth_l1'
    };
    
    const scale = 1.0 / input.data.length;
    for (let i = 0; i < input.data.length; i++) {
      const diff = input.data[i] - target.data[i];
      const absDiff = Math.abs(diff);
      
      if (absDiff < 1) {
        gradInput.data[i] = scale * diff;
      } else {
        gradInput.data[i] = scale * (diff > 0 ? 1 : -1);
      }
    }
    
    return gradInput;
  }
}
