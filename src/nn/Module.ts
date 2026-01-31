/**
 * Core Module Interface for torch/nn implementation
 * 
 * Based on Torch7's nn.Module architecture
 * https://github.com/torch/nn
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';

/**
 * Base neural network module interface
 * Inspired by nn.Module from torch/nn
 */
export interface Module {
  name: string;
  parameters: GgmlTensor[];
  gradients: GgmlTensor[];
  training: boolean;
  
  /**
   * Forward pass through the module
   */
  forward(input: GgmlTensor): GgmlTensor;
  
  /**
   * Backward pass through the module
   */
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor;
  
  /**
   * Update module parameters with gradients
   */
  updateParameters(learningRate: number): void;
  
  /**
   * Zero out all gradients
   */
  zeroGradParameters(): void;
  
  /**
   * Get all parameters and their gradients
   */
  getParameters(): { weights: GgmlTensor[], gradients: GgmlTensor[] };
  
  /**
   * Set training mode
   */
  train(): void;
  
  /**
   * Set evaluation mode
   */
  evaluate(): void;
}

/**
 * Abstract base class for neural network modules
 */
export abstract class BaseModule implements Module {
  name: string = 'BaseModule';
  parameters: GgmlTensor[] = [];
  gradients: GgmlTensor[] = [];
  training: boolean = true;
  
  abstract forward(input: GgmlTensor): GgmlTensor;
  abstract backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor;
  
  updateParameters(learningRate: number): void {
    for (let i = 0; i < this.parameters.length; i++) {
      const param = this.parameters[i];
      const grad = this.gradients[i];
      for (let j = 0; j < param.data.length; j++) {
        param.data[j] -= learningRate * grad.data[j];
      }
    }
  }
  
  zeroGradParameters(): void {
    for (const grad of this.gradients) {
      grad.data.fill(0);
    }
  }
  
  getParameters(): { weights: GgmlTensor[], gradients: GgmlTensor[] } {
    return {
      weights: this.parameters,
      gradients: this.gradients
    };
  }
  
  train(): void {
    this.training = true;
  }
  
  evaluate(): void {
    this.training = false;
  }
  
  /**
   * Helper method to create tensors
   */
  protected createTensor(
    shape: number[], 
    name: string, 
    initializer?: (index: number) => number
  ): GgmlTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    
    if (initializer) {
      for (let i = 0; i < size; i++) {
        data[i] = initializer(i);
      }
    }
    
    return {
      id: `${name}_${Math.random().toString(36).substr(2, 9)}`,
      shape,
      data,
      dtype: 'f32',
      requires_grad: true,
      name
    };
  }
}
