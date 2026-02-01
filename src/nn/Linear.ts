/**
 * Linear Layers for torch/nn
 * 
 * Basic linear transformation layers
 * Based on torch/nn Linear module
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';
import { BaseModule } from './Module.js';

/**
 * Linear layer: y = Wx + b
 * Applies affine transformation to input
 */
export class Linear extends BaseModule {
  name = 'Linear';
  private weight: GgmlTensor;
  private bias?: GgmlTensor;
  private inputCache?: GgmlTensor;
  private useBias: boolean;
  
  constructor(
    private inputSize: number,
    private outputSize: number,
    bias: boolean = true
  ) {
    super();
    this.useBias = bias;
    
    // Xavier/Glorot initialization
    const stdv = 1.0 / Math.sqrt(inputSize);
    this.weight = this.createTensor(
      [outputSize, inputSize],
      'weight',
      () => (Math.random() - 0.5) * 2 * stdv
    );
    
    if (bias) {
      this.bias = this.createTensor([outputSize], 'bias', () => 0);
      this.parameters = [this.weight, this.bias];
    } else {
      this.parameters = [this.weight];
    }
    
    this.gradients = this.parameters.map(p => ({
      ...p,
      id: `grad_${p.id}`,
      data: new Float32Array(p.data.length)
    }));
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    if (this.training) {
      this.inputCache = input;
    }
    
    const batchSize = input.shape[0];
    const output = this.createTensor([batchSize, this.outputSize], 'linear_output');
    
    // Matrix multiplication: output = input * W^T + b
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        let sum = this.useBias && this.bias ? this.bias.data[j] : 0;
        for (let k = 0; k < this.inputSize; k++) {
          sum += input.data[i * this.inputSize + k] * 
                 this.weight.data[j * this.inputSize + k];
        }
        output.data[i * this.outputSize + j] = sum;
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const batchSize = gradOutput.shape[0];
    const gradInput = this.createTensor(input.shape, 'grad_input');
    
    // Compute gradient w.r.t input: gradInput = gradOutput * W
    for (let i = 0; i < batchSize; i++) {
      for (let k = 0; k < this.inputSize; k++) {
        let sum = 0;
        for (let j = 0; j < this.outputSize; j++) {
          sum += gradOutput.data[i * this.outputSize + j] * 
                 this.weight.data[j * this.inputSize + k];
        }
        gradInput.data[i * this.inputSize + k] = sum;
      }
    }
    
    // Compute gradient w.r.t weight: gradWeight = gradOutput^T * input
    const inputVal = this.inputCache || input;
    const gradWeight = this.gradients[0];
    
    for (let j = 0; j < this.outputSize; j++) {
      for (let k = 0; k < this.inputSize; k++) {
        let sum = 0;
        for (let i = 0; i < batchSize; i++) {
          sum += gradOutput.data[i * this.outputSize + j] * 
                 inputVal.data[i * this.inputSize + k];
        }
        gradWeight.data[j * this.inputSize + k] += sum;
      }
    }
    
    // Compute gradient w.r.t bias: gradBias = sum(gradOutput, dim=0)
    if (this.useBias && this.bias) {
      const gradBias = this.gradients[1];
      for (let j = 0; j < this.outputSize; j++) {
        let sum = 0;
        for (let i = 0; i < batchSize; i++) {
          sum += gradOutput.data[i * this.outputSize + j];
        }
        gradBias.data[j] += sum;
      }
    }
    
    return gradInput;
  }
}

/**
 * Dropout layer: randomly zeros elements with probability p
 */
export class Dropout extends BaseModule {
  name = 'Dropout';
  private p: number;
  private mask?: GgmlTensor;
  
  constructor(p: number = 0.5) {
    super();
    this.p = p;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    if (!this.training) {
      // No dropout during evaluation
      return input;
    }
    
    const output = this.createTensor(input.shape, 'dropout_output');
    const mask = this.createTensor(input.shape, 'dropout_mask');
    
    // Apply dropout and scale
    const scale = 1.0 / (1.0 - this.p);
    for (let i = 0; i < input.data.length; i++) {
      const drop = Math.random() < this.p;
      mask.data[i] = drop ? 0 : 1;
      output.data[i] = drop ? 0 : input.data[i] * scale;
    }
    
    this.mask = mask;
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    if (!this.training || !this.mask) {
      return gradOutput;
    }
    
    const gradInput = this.createTensor(input.shape, 'grad_dropout');
    const scale = 1.0 / (1.0 - this.p);
    
    for (let i = 0; i < gradOutput.data.length; i++) {
      gradInput.data[i] = gradOutput.data[i] * this.mask.data[i] * scale;
    }
    
    return gradInput;
  }
}

/**
 * Embedding layer: looks up embeddings from a table
 */
export class Embedding extends BaseModule {
  name = 'Embedding';
  private weight: GgmlTensor;
  private inputCache?: number[];
  
  constructor(
    private numEmbeddings: number,
    private embeddingDim: number
  ) {
    super();
    
    // Initialize embeddings with normal distribution
    this.weight = this.createTensor(
      [numEmbeddings, embeddingDim],
      'embedding_weight',
      () => (Math.random() - 0.5) * 0.1
    );
    
    this.parameters = [this.weight];
    this.gradients = [this.createTensor(
      this.weight.shape,
      'grad_embedding_weight'
    )];
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    // Input should be integer indices
    const indices = Array.from(input.data).map(x => Math.floor(x));
    
    if (this.training) {
      this.inputCache = indices;
    }
    
    const batchSize = input.shape[0];
    const output = this.createTensor(
      [batchSize, this.embeddingDim],
      'embedding_output'
    );
    
    // Look up embeddings
    for (let i = 0; i < batchSize; i++) {
      const idx = indices[i];
      if (idx >= 0 && idx < this.numEmbeddings) {
        for (let j = 0; j < this.embeddingDim; j++) {
          output.data[i * this.embeddingDim + j] = 
            this.weight.data[idx * this.embeddingDim + j];
        }
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    // Gradients flow back to the embedding table
    const gradWeight = this.gradients[0];
    const indices = this.inputCache || Array.from(input.data).map(x => Math.floor(x));
    
    const batchSize = gradOutput.shape[0];
    
    // Accumulate gradients for each embedding
    for (let i = 0; i < batchSize; i++) {
      const idx = indices[i];
      if (idx >= 0 && idx < this.numEmbeddings) {
        for (let j = 0; j < this.embeddingDim; j++) {
          gradWeight.data[idx * this.embeddingDim + j] += 
            gradOutput.data[i * this.embeddingDim + j];
        }
      }
    }
    
    // No gradient w.r.t input indices
    return this.createTensor(input.shape, 'grad_embedding_input');
  }
}

/**
 * Flatten layer: flattens input starting from dimension
 */
export class Flatten extends BaseModule {
  name = 'Flatten';
  private startDim: number;
  private endDim: number;
  private inputShape?: number[];
  
  constructor(startDim: number = 1, endDim: number = -1) {
    super();
    this.startDim = startDim;
    this.endDim = endDim;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    if (this.training) {
      this.inputShape = input.shape;
    }
    
    const endDim = this.endDim === -1 ? input.shape.length : this.endDim;
    
    // Calculate new shape
    const newShape: number[] = [];
    for (let i = 0; i < this.startDim; i++) {
      newShape.push(input.shape[i]);
    }
    
    let flatSize = 1;
    for (let i = this.startDim; i < endDim; i++) {
      flatSize *= input.shape[i];
    }
    newShape.push(flatSize);
    
    for (let i = endDim; i < input.shape.length; i++) {
      newShape.push(input.shape[i]);
    }
    
    // Data is already flat in memory, just change shape
    const output = { ...input };
    output.shape = newShape;
    output.name = 'flatten_output';
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    // Reshape gradient back to input shape
    const gradInput = { ...gradOutput };
    gradInput.shape = this.inputShape || input.shape;
    gradInput.name = 'grad_flatten';
    
    return gradInput;
  }
}
