/**
 * Pooling Layers for torch/nn
 * 
 * Pooling operations for downsampling feature maps
 * Based on torch/nn pooling modules
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';
import { BaseModule } from './Module.js';

/**
 * Max Pooling 1D
 */
export class MaxPool1D extends BaseModule {
  name = 'MaxPool1D';
  private kernelSize: number;
  private stride: number;
  private padding: number;
  private maxIndices?: Int32Array;
  
  constructor(
    kernelSize: number,
    stride?: number,
    padding: number = 0
  ) {
    super();
    this.kernelSize = kernelSize;
    this.stride = stride || kernelSize;
    this.padding = padding;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    // Input shape: [batch, channels, length]
    const [batch, channels, length] = input.shape;
    const outLength = Math.floor((length + 2 * this.padding - this.kernelSize) / this.stride) + 1;
    
    const output = this.createTensor([batch, channels, outLength], 'maxpool1d_output');
    
    if (this.training) {
      this.maxIndices = new Int32Array(batch * channels * outLength);
    }
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outLength; i++) {
          let maxVal = -Infinity;
          let maxIdx = 0;
          
          for (let k = 0; k < this.kernelSize; k++) {
            const idx = i * this.stride + k - this.padding;
            if (idx >= 0 && idx < length) {
              const val = input.data[b * channels * length + c * length + idx];
              if (val > maxVal) {
                maxVal = val;
                maxIdx = idx;
              }
            }
          }
          
          output.data[b * channels * outLength + c * outLength + i] = maxVal;
          if (this.maxIndices) {
            this.maxIndices[b * channels * outLength + c * outLength + i] = maxIdx;
          }
        }
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const [batch, channels, length] = input.shape;
    const [, , outLength] = gradOutput.shape;
    
    const gradInput = this.createTensor(input.shape, 'grad_maxpool1d');
    gradInput.data.fill(0);
    
    if (!this.maxIndices) {
      return gradInput;
    }
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outLength; i++) {
          const maxIdx = this.maxIndices[b * channels * outLength + c * outLength + i];
          const grad = gradOutput.data[b * channels * outLength + c * outLength + i];
          gradInput.data[b * channels * length + c * length + maxIdx] += grad;
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * Average Pooling 1D
 */
export class AvgPool1D extends BaseModule {
  name = 'AvgPool1D';
  private kernelSize: number;
  private stride: number;
  private padding: number;
  
  constructor(
    kernelSize: number,
    stride?: number,
    padding: number = 0
  ) {
    super();
    this.kernelSize = kernelSize;
    this.stride = stride || kernelSize;
    this.padding = padding;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    const [batch, channels, length] = input.shape;
    const outLength = Math.floor((length + 2 * this.padding - this.kernelSize) / this.stride) + 1;
    
    const output = this.createTensor([batch, channels, outLength], 'avgpool1d_output');
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outLength; i++) {
          let sum = 0;
          let count = 0;
          
          for (let k = 0; k < this.kernelSize; k++) {
            const idx = i * this.stride + k - this.padding;
            if (idx >= 0 && idx < length) {
              sum += input.data[b * channels * length + c * length + idx];
              count++;
            }
          }
          
          output.data[b * channels * outLength + c * outLength + i] = sum / count;
        }
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const [batch, channels, length] = input.shape;
    const [, , outLength] = gradOutput.shape;
    
    const gradInput = this.createTensor(input.shape, 'grad_avgpool1d');
    gradInput.data.fill(0);
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outLength; i++) {
          let count = 0;
          
          for (let k = 0; k < this.kernelSize; k++) {
            const idx = i * this.stride + k - this.padding;
            if (idx >= 0 && idx < length) {
              count++;
            }
          }
          
          const grad = gradOutput.data[b * channels * outLength + c * outLength + i] / count;
          
          for (let k = 0; k < this.kernelSize; k++) {
            const idx = i * this.stride + k - this.padding;
            if (idx >= 0 && idx < length) {
              gradInput.data[b * channels * length + c * length + idx] += grad;
            }
          }
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * Max Pooling 2D
 */
export class MaxPool2D extends BaseModule {
  name = 'MaxPool2D';
  private kernelSize: [number, number];
  private stride: [number, number];
  private padding: [number, number];
  private maxIndices?: Int32Array;
  
  constructor(
    kernelSize: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    this.stride = stride 
      ? (typeof stride === 'number' ? [stride, stride] : stride)
      : this.kernelSize;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    // Input shape: [batch, channels, height, width]
    const [batch, channels, height, width] = input.shape;
    const outH = Math.floor((height + 2 * this.padding[0] - this.kernelSize[0]) / this.stride[0]) + 1;
    const outW = Math.floor((width + 2 * this.padding[1] - this.kernelSize[1]) / this.stride[1]) + 1;
    
    const output = this.createTensor([batch, channels, outH, outW], 'maxpool2d_output');
    
    if (this.training) {
      this.maxIndices = new Int32Array(batch * channels * outH * outW);
    }
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outH; i++) {
          for (let j = 0; j < outW; j++) {
            let maxVal = -Infinity;
            let maxIdxH = 0, maxIdxW = 0;
            
            for (let kh = 0; kh < this.kernelSize[0]; kh++) {
              for (let kw = 0; kw < this.kernelSize[1]; kw++) {
                const h = i * this.stride[0] + kh - this.padding[0];
                const w = j * this.stride[1] + kw - this.padding[1];
                
                if (h >= 0 && h < height && w >= 0 && w < width) {
                  const idx = ((b * channels + c) * height + h) * width + w;
                  const val = input.data[idx];
                  
                  if (val > maxVal) {
                    maxVal = val;
                    maxIdxH = h;
                    maxIdxW = w;
                  }
                }
              }
            }
            
            const outIdx = ((b * channels + c) * outH + i) * outW + j;
            output.data[outIdx] = maxVal;
            
            if (this.maxIndices) {
              this.maxIndices[outIdx] = maxIdxH * width + maxIdxW;
            }
          }
        }
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const [batch, channels, height, width] = input.shape;
    const [, , outH, outW] = gradOutput.shape;
    
    const gradInput = this.createTensor(input.shape, 'grad_maxpool2d');
    gradInput.data.fill(0);
    
    if (!this.maxIndices) {
      return gradInput;
    }
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outH; i++) {
          for (let j = 0; j < outW; j++) {
            const outIdx = ((b * channels + c) * outH + i) * outW + j;
            const maxIdx = this.maxIndices[outIdx];
            const h = Math.floor(maxIdx / width);
            const w = maxIdx % width;
            
            const inIdx = ((b * channels + c) * height + h) * width + w;
            gradInput.data[inIdx] += gradOutput.data[outIdx];
          }
        }
      }
    }
    
    return gradInput;
  }
}

/**
 * Average Pooling 2D
 */
export class AvgPool2D extends BaseModule {
  name = 'AvgPool2D';
  private kernelSize: [number, number];
  private stride: [number, number];
  private padding: [number, number];
  
  constructor(
    kernelSize: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0
  ) {
    super();
    this.kernelSize = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    this.stride = stride 
      ? (typeof stride === 'number' ? [stride, stride] : stride)
      : this.kernelSize;
    this.padding = typeof padding === 'number' ? [padding, padding] : padding;
  }
  
  forward(input: GgmlTensor): GgmlTensor {
    const [batch, channels, height, width] = input.shape;
    const outH = Math.floor((height + 2 * this.padding[0] - this.kernelSize[0]) / this.stride[0]) + 1;
    const outW = Math.floor((width + 2 * this.padding[1] - this.kernelSize[1]) / this.stride[1]) + 1;
    
    const output = this.createTensor([batch, channels, outH, outW], 'avgpool2d_output');
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outH; i++) {
          for (let j = 0; j < outW; j++) {
            let sum = 0;
            let count = 0;
            
            for (let kh = 0; kh < this.kernelSize[0]; kh++) {
              for (let kw = 0; kw < this.kernelSize[1]; kw++) {
                const h = i * this.stride[0] + kh - this.padding[0];
                const w = j * this.stride[1] + kw - this.padding[1];
                
                if (h >= 0 && h < height && w >= 0 && w < width) {
                  const idx = ((b * channels + c) * height + h) * width + w;
                  sum += input.data[idx];
                  count++;
                }
              }
            }
            
            const outIdx = ((b * channels + c) * outH + i) * outW + j;
            output.data[outIdx] = sum / count;
          }
        }
      }
    }
    
    return output;
  }
  
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor {
    const [batch, channels, height, width] = input.shape;
    const [, , outH, outW] = gradOutput.shape;
    
    const gradInput = this.createTensor(input.shape, 'grad_avgpool2d');
    gradInput.data.fill(0);
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let i = 0; i < outH; i++) {
          for (let j = 0; j < outW; j++) {
            let count = 0;
            
            for (let kh = 0; kh < this.kernelSize[0]; kh++) {
              for (let kw = 0; kw < this.kernelSize[1]; kw++) {
                const h = i * this.stride[0] + kh - this.padding[0];
                const w = j * this.stride[1] + kw - this.padding[1];
                
                if (h >= 0 && h < height && w >= 0 && w < width) {
                  count++;
                }
              }
            }
            
            const outIdx = ((b * channels + c) * outH + i) * outW + j;
            const grad = gradOutput.data[outIdx] / count;
            
            for (let kh = 0; kh < this.kernelSize[0]; kh++) {
              for (let kw = 0; kw < this.kernelSize[1]; kw++) {
                const h = i * this.stride[0] + kh - this.padding[0];
                const w = j * this.stride[1] + kw - this.padding[1];
                
                if (h >= 0 && h < height && w >= 0 && w < width) {
                  const inIdx = ((b * channels + c) * height + h) * width + w;
                  gradInput.data[inIdx] += grad;
                }
              }
            }
          }
        }
      }
    }
    
    return gradInput;
  }
}
