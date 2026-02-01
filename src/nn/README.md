# torch/nn Implementation

A TypeScript implementation of neural network modules inspired by [torch/nn](https://github.com/torch/nn) - Torch7's neural network library.

## Overview

This module provides a comprehensive set of neural network building blocks for creating and training deep learning models in TypeScript. It follows the design principles of torch/nn while leveraging TypeScript's type safety and modern JavaScript features.

## Features

### Core Components

- **Module Interface**: Base interface for all neural network components
- **BaseModule**: Abstract base class with common functionality
- **Automatic Differentiation**: Built-in forward and backward propagation

### Container Modules

Modules that contain and organize other modules:

- **Sequential**: Chain modules in a linear pipeline
- **Parallel**: Apply modules in parallel and combine outputs
- **Concat**: Concatenate outputs of multiple modules

### Linear Layers

Basic building blocks for neural networks:

- **Linear**: Fully connected layer (y = Wx + b)
- **Dropout**: Regularization through random dropout
- **Embedding**: Lookup table for embeddings
- **Flatten**: Reshape tensors for linear layers

### Activation Functions

Non-linear transformations:

- **ReLU**: Rectified Linear Unit
- **Tanh**: Hyperbolic tangent
- **Sigmoid**: Logistic sigmoid
- **Softmax**: Softmax normalization
- **LogSoftmax**: Log-softmax (numerically stable)
- **LeakyReLU**: Leaky ReLU with negative slope

### Pooling Layers

Downsampling operations:

- **MaxPool1D**: 1D max pooling
- **MaxPool2D**: 2D max pooling
- **AvgPool1D**: 1D average pooling
- **AvgPool2D**: 2D average pooling

### Loss Functions (Criterions)

Training objectives:

- **MSECriterion**: Mean Squared Error
- **CrossEntropyCriterion**: Cross-entropy loss
- **NLLCriterion**: Negative Log Likelihood
- **BCECriterion**: Binary Cross-Entropy
- **L1Criterion**: L1 (MAE) loss
- **SmoothL1Criterion**: Smooth L1 (Huber) loss

## Usage Examples

### Simple Feed-Forward Network

```typescript
import * as nn from './nn';

// Create a simple 3-layer network
const model = new nn.Sequential([
  new nn.Linear(784, 128),    // Input layer
  new nn.ReLU(),              // Activation
  new nn.Dropout(0.5),        // Regularization
  new nn.Linear(128, 64),     // Hidden layer
  new nn.ReLU(),
  new nn.Linear(64, 10),      // Output layer
  new nn.Softmax()            // Softmax for classification
]);

// Forward pass
const output = model.forward(inputTensor);

// Training
const criterion = new nn.MSECriterion();
const loss = criterion.forward(output, targetTensor);
const gradOutput = criterion.backward(output, targetTensor);
model.backward(inputTensor, gradOutput);
model.updateParameters(0.01); // Learning rate = 0.01
```

### Classification Network

```typescript
import * as nn from './nn';

// Create a classifier
const classifier = new nn.Sequential([
  new nn.Linear(784, 256),
  new nn.ReLU(),
  new nn.Linear(256, 128),
  new nn.ReLU(),
  new nn.Linear(128, 10),
  new nn.LogSoftmax()  // For use with NLL loss
]);

// Use NLL loss for classification
const criterion = new nn.NLLCriterion();

// Training loop
for (let epoch = 0; epoch < 10; epoch++) {
  model.train(); // Set to training mode
  
  const output = classifier.forward(input);
  const loss = criterion.forward(output, labels);
  
  const gradOutput = criterion.backward(output, labels);
  classifier.backward(input, gradOutput);
  classifier.updateParameters(0.01);
  
  console.log(`Epoch ${epoch}: Loss = ${loss}`);
}

// Evaluation
classifier.evaluate(); // Set to evaluation mode
const predictions = classifier.forward(testInput);
```

### Custom Network with Parallel Branches

```typescript
import * as nn from './nn';

// Create parallel branches
const branch1 = new nn.Sequential([
  new nn.Linear(128, 64),
  new nn.ReLU()
]);

const branch2 = new nn.Sequential([
  new nn.Linear(128, 64),
  new nn.Tanh()
]);

// Combine with Concat
const model = new nn.Sequential([
  new nn.Linear(256, 128),
  new nn.Concat(1, [branch1, branch2]), // Concatenate along dimension 1
  new nn.Linear(128, 10),
  new nn.Softmax()
]);
```

### Weight Initialization

```typescript
import * as nn from './nn';

const model = new nn.Linear(100, 50);

// Xavier initialization (default for most cases)
nn.initXavier(model);

// He initialization (good for ReLU networks)
nn.initHe(model);

// Uniform initialization
nn.initUniform(model, -0.1, 0.1);

// Normal initialization
nn.initNormal(model, 0, 0.01);

// Zero initialization
nn.initZeros(model);
```

## API Reference

### Module Interface

```typescript
interface Module {
  name: string;
  parameters: GgmlTensor[];
  gradients: GgmlTensor[];
  training: boolean;
  
  forward(input: GgmlTensor): GgmlTensor;
  backward(input: GgmlTensor, gradOutput: GgmlTensor): GgmlTensor;
  updateParameters(learningRate: number): void;
  zeroGradParameters(): void;
  getParameters(): { weights: GgmlTensor[], gradients: GgmlTensor[] };
  train(): void;
  evaluate(): void;
}
```

### Linear Layer

```typescript
new nn.Linear(inputSize: number, outputSize: number, bias?: boolean)
```

- `inputSize`: Number of input features
- `outputSize`: Number of output features
- `bias`: Whether to include bias term (default: true)

### Sequential Container

```typescript
new nn.Sequential(modules?: Module[])
```

- `modules`: Array of modules to chain together
- `add(module)`: Add a module to the sequence

### Dropout

```typescript
new nn.Dropout(p?: number)
```

- `p`: Dropout probability (default: 0.5)

### Activation Functions

```typescript
new nn.ReLU()
new nn.Tanh()
new nn.Sigmoid()
new nn.Softmax(dim?: number)
new nn.LogSoftmax(dim?: number)
new nn.LeakyReLU(alpha?: number)
```

### Pooling Layers

```typescript
new nn.MaxPool1D(kernelSize: number, stride?: number, padding?: number)
new nn.MaxPool2D(kernelSize: number | [number, number], stride?, padding?)
new nn.AvgPool1D(kernelSize: number, stride?: number, padding?: number)
new nn.AvgPool2D(kernelSize: number | [number, number], stride?, padding?)
```

### Loss Functions

```typescript
const criterion = new nn.MSECriterion();
const loss = criterion.forward(prediction, target);
const gradient = criterion.backward(prediction, target);
```

## Comparison with torch/nn

This implementation maintains compatibility with torch/nn's core concepts:

| torch/nn (Lua) | This Implementation |
|----------------|---------------------|
| nn.Module | Module interface + BaseModule |
| nn.Sequential | Sequential |
| nn.Parallel | Parallel |
| nn.Concat | Concat |
| nn.Linear | Linear |
| nn.ReLU | ReLU |
| nn.Tanh | Tanh |
| nn.Sigmoid | Sigmoid |
| nn.SoftMax | Softmax |
| nn.LogSoftMax | LogSoftmax |
| nn.Dropout | Dropout |
| nn.MSECriterion | MSECriterion |
| nn.ClassNLLCriterion | NLLCriterion |
| nn.BCECriterion | BCECriterion |

## Integration with Existing Code

The nn module is designed to work seamlessly with the existing NanoBrain cognitive architecture:

```typescript
// Use with LearnabilityEmbeddings
import * as nn from './nn';
import { GgmlTensor } from '../core/GgmlTensorKernel';

// Create a cognitive model
const cognitiveModel = new nn.Sequential([
  new nn.Embedding(10000, 128),  // Vocabulary embeddings
  new nn.Linear(128, 64),
  new nn.ReLU(),
  new nn.Linear(64, 32),
  new nn.Tanh()
]);
```

## Performance Considerations

- Uses Float32Array for efficient memory usage
- Implements caching for backward pass optimization
- Supports batch processing for efficiency
- Compatible with WebGPU acceleration (when available)

## Future Enhancements

Planned additions to match full torch/nn functionality:

- Convolutional layers (Conv1D, Conv2D)
- Recurrent layers (LSTM, GRU)
- Batch normalization
- More pooling variants (AdaptivePooling)
- Additional activation functions (ELU, GELU, Swish)
- More loss functions
- Parameter initialization schemes

## Contributing

Contributions are welcome! Please ensure new modules:
- Implement the Module interface
- Include both forward and backward methods
- Provide comprehensive tests
- Follow TypeScript best practices

## License

MIT License - Same as the parent NanoBrain project

## References

- [torch/nn GitHub Repository](https://github.com/torch/nn)
- [Torch7 Documentation](http://torch.ch/)
- [NanoBrain Cognitive Architecture](../README.md)
