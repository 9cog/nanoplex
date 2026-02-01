/**
 * Container Modules for torch/nn
 * 
 * Container modules that hold other modules
 * Based on torch/nn containers like Sequential, Parallel, Concat
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';
import { Module, BaseModule } from './Module.js';

/**
 * Sequential container - chains modules one after another
 * Similar to nn.Sequential from torch/nn
 */
export class Sequential extends BaseModule {
  name = 'Sequential';
  private modules: Module[] = [];
  
  constructor(modules: Module[] = []) {
    super();
    this.modules = modules;
    this.updateParametersList();
  }
  
  add(module: Module): void {
    this.modules.push(module);
    this.updateParametersList();
  }
  
  private updateParametersList(): void {
    this.parameters = [];
    this.gradients = [];
    for (const module of this.modules) {
      this.parameters.push(...module.parameters);
      this.gradients.push(...module.gradients);
    }
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    let output = input;
    for (const module of this.modules) {
      output = module.forward(output);
    }
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    let gradInput = gradOutput;
    
    // Backward through modules in reverse order
    for (let i = this.modules.length - 1; i >= 0; i--) {
      const module = this.modules[i];
      // Get the input to this module from forward pass
      const moduleInput = i === 0 ? input : this.getModuleInput(i, input);
      gradInput = module.backward(moduleInput, gradInput);
    }
    
    return gradInput;
  }
  
  private getModuleInput(moduleIndex: number, originalInput: GgmlTensor): GgmlTensor {
    let output = originalInput;
    for (let i = 0; i < moduleIndex; i++) {
      output = this.modules[i].forward(output);
    }
    return output;
  }
  
  train(): void {
    super.train();
    for (const module of this.modules) {
      module.train();
    }
  }
  
  evaluate(): void {
    super.evaluate();
    for (const module of this.modules) {
      module.evaluate();
    }
  }
}

/**
 * Parallel container - applies modules in parallel and combines outputs
 * Similar to nn.Parallel from torch/nn
 */
export class Parallel extends BaseModule {
  name = 'Parallel';
  private modules: Module[] = [];
  private inputDimension: number;
  private outputDimension: number;
  
  constructor(
    inputDimension: number,
    outputDimension: number,
    modules: Module[] = []
  ) {
    super();
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
    this.modules = modules;
    this.updateParametersList();
  }
  
  add(module: Module): void {
    this.modules.push(module);
    this.updateParametersList();
  }
  
  private updateParametersList(): void {
    this.parameters = [];
    this.gradients = [];
    for (const module of this.modules) {
      this.parameters.push(...module.parameters);
      this.gradients.push(...module.gradients);
    }
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    if (this.modules.length === 0) {
      return input;
    }
    
    // Split input along dimension and apply modules
    const outputs = this.modules.map(module => module.forward(input));
    
    // Concatenate outputs along output dimension
    return this.concatenateTensors(outputs, this.outputDimension);
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    if (this.modules.length === 0) {
      return gradOutput;
    }
    
    // Split gradient along output dimension
    const gradOutputs = this.splitTensor(gradOutput, this.outputDimension, this.modules.length);
    
    // Backward through each module
    const gradInputs = gradOutputs.map((gradOut, i) => 
      this.modules[i].backward(input, gradOut)
    );
    
    // Sum gradients from all modules
    return this.sumTensors(gradInputs);
  }
  
  private concatenateTensors(tensors: GgmlTensor[], dim: number): GgmlTensor {
    if (tensors.length === 0) throw new Error('No tensors to concatenate');
    if (tensors.length === 1) return tensors[0];
    
    const first = tensors[0];
    const shape = [...first.shape];
    shape[dim] = tensors.reduce((sum, t) => sum + t.shape[dim], 0);
    
    const result = this.createTensor(shape, 'parallel_concat');
    
    // Simple concatenation for 2D tensors
    if (dim === 1 && shape.length === 2) {
      const batchSize = shape[0];
      let offset = 0;
      
      for (const tensor of tensors) {
        const cols = tensor.shape[1];
        for (let i = 0; i < batchSize; i++) {
          for (let j = 0; j < cols; j++) {
            result.data[i * shape[1] + offset + j] = tensor.data[i * cols + j];
          }
        }
        offset += cols;
      }
    }
    
    return result;
  }
  
  private splitTensor(tensor: GgmlTensor, dim: number, numSplits: number): GgmlTensor[] {
    const splitSize = Math.floor(tensor.shape[dim] / numSplits);
    const results: GgmlTensor[] = [];
    
    for (let i = 0; i < numSplits; i++) {
      const shape = [...tensor.shape];
      shape[dim] = splitSize;
      const split = this.createTensor(shape, `split_${i}`);
      
      // Simple split for 2D tensors
      if (dim === 1 && shape.length === 2) {
        const batchSize = shape[0];
        const offset = i * splitSize;
        
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < splitSize; j++) {
            split.data[b * splitSize + j] = tensor.data[b * tensor.shape[1] + offset + j];
          }
        }
      }
      
      results.push(split);
    }
    
    return results;
  }
  
  private sumTensors(tensors: GgmlTensor[]): GgmlTensor {
    if (tensors.length === 0) throw new Error('No tensors to sum');
    if (tensors.length === 1) return tensors[0];
    
    const result = this.createTensor(tensors[0].shape, 'sum');
    result.data.fill(0);
    
    for (const tensor of tensors) {
      for (let i = 0; i < result.data.length; i++) {
        result.data[i] += tensor.data[i];
      }
    }
    
    return result;
  }
}

/**
 * Concat container - concatenates outputs of multiple modules
 * Similar to nn.Concat from torch/nn
 */
export class Concat extends BaseModule {
  name = 'Concat';
  private modules: Module[] = [];
  private dimension: number;
  
  constructor(dimension: number, modules: Module[] = []) {
    super();
    this.dimension = dimension;
    this.modules = modules;
    this.updateParametersList();
  }
  
  add(module: Module): void {
    this.modules.push(module);
    this.updateParametersList();
  }
  
  private updateParametersList(): void {
    this.parameters = [];
    this.gradients = [];
    for (const module of this.modules) {
      this.parameters.push(...module.parameters);
      this.gradients.push(...module.gradients);
    }
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    if (this.modules.length === 0) {
      return input;
    }
    
    const outputs = this.modules.map(module => module.forward(input));
    return this.concatenateTensors(outputs, this.dimension);
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    if (this.modules.length === 0) {
      return gradOutput;
    }
    
    const gradOutputs = this.splitTensor(gradOutput, this.dimension, this.modules.length);
    const gradInputs = gradOutputs.map((gradOut, i) => 
      this.modules[i].backward(input, gradOut)
    );
    
    return this.sumTensors(gradInputs);
  }
  
  private concatenateTensors(tensors: GgmlTensor[], dim: number): GgmlTensor {
    if (tensors.length === 0) throw new Error('No tensors to concatenate');
    if (tensors.length === 1) return tensors[0];
    
    const first = tensors[0];
    const shape = [...first.shape];
    shape[dim] = tensors.reduce((sum, t) => sum + t.shape[dim], 0);
    
    const result = this.createTensor(shape, 'concat');
    
    if (dim === 1 && shape.length === 2) {
      const batchSize = shape[0];
      let offset = 0;
      
      for (const tensor of tensors) {
        const cols = tensor.shape[1];
        for (let i = 0; i < batchSize; i++) {
          for (let j = 0; j < cols; j++) {
            result.data[i * shape[1] + offset + j] = tensor.data[i * cols + j];
          }
        }
        offset += cols;
      }
    }
    
    return result;
  }
  
  private splitTensor(tensor: GgmlTensor, dim: number, numSplits: number): GgmlTensor[] {
    const splitSize = Math.floor(tensor.shape[dim] / numSplits);
    const results: GgmlTensor[] = [];
    
    for (let i = 0; i < numSplits; i++) {
      const shape = [...tensor.shape];
      shape[dim] = splitSize;
      const split = this.createTensor(shape, `split_${i}`);
      
      if (dim === 1 && shape.length === 2) {
        const batchSize = shape[0];
        const offset = i * splitSize;
        
        for (let b = 0; b < batchSize; b++) {
          for (let j = 0; j < splitSize; j++) {
            split.data[b * splitSize + j] = tensor.data[b * tensor.shape[1] + offset + j];
          }
        }
      }
      
      results.push(split);
    }
    
    return results;
  }
  
  private sumTensors(tensors: GgmlTensor[]): GgmlTensor {
    if (tensors.length === 0) throw new Error('No tensors to sum');
    if (tensors.length === 1) return tensors[0];
    
    const result = this.createTensor(tensors[0].shape, 'sum');
    result.data.fill(0);
    
    for (const tensor of tensors) {
      for (let i = 0; i < result.data.length; i++) {
        result.data[i] += tensor.data[i];
      }
    }
    
    return result;
  }
}
