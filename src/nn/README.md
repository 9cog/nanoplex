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

## Prime-Shaped Nested Tensor Tuples (Chapter 7)

### Overview

This implementation includes specialized modules for processing prime-shaped nested tensor tuples as described in **NanoBrain Chapter 7: Time Crystal Brain Model**. These modules implement the quaternion-octonion-dodecanion architecture that mirrors the brain's hierarchical processing.

### Key Concepts

The brain processes information through three nested levels of prime-shaped tensors:

1. **Quaternion (4D)**: Sensory input processing with spatial primes [2, 3, 5]
2. **Octonion (8D)**: Pathway/midbrain fusion with temporal primes [7, 11, 13]
3. **Dodecanion (12D)**: Cortical manifolds with consciousness primes [29, 31, 37]

The complete transformation pipeline: **d4D → d8D → d12D**

### Prime-Shaped Modules

#### QuaternionModule (4D)

Processes sensory input into 4D quaternion space:

```typescript
import { QuaternionModule } from './nn';

const quatModule = new QuaternionModule(
  64,           // input size
  [2, 3, 5]     // spatial primes
);

const quaternionOutput = quatModule.forward(sensorInput);
```

#### OctonionModule (8D)

Fuses information into 8D octonion space with non-commutative properties:

```typescript
import { OctonionModule } from './nn';

const octModule = new OctonionModule(
  4,            // input size (from quaternion)
  [7, 11, 13]   // temporal primes
);

const octonionOutput = octModule.forward(quaternionOutput);
```

#### DodecanionModule (12D)

Processes abstract thought in 12D dodecanion manifolds:

```typescript
import { DodecanionModule } from './nn';

const dodModule = new DodecanionModule(
  8,            // input size (from octonion)
  [29, 31, 37]  // consciousness primes
);

const dodecanionOutput = dodModule.forward(octonionOutput);
```

### NestedTensorTransformer

Complete pipeline for d4D → d8D → d12D transformation:

```typescript
import { NestedTensorTransformer } from './nn';

const transformer = new NestedTensorTransformer(64); // input size

// Get final 12D output
const output = transformer.forward(input);

// Get all intermediate tensors
const nestedTuple = transformer.forwardNested(input);
console.log(nestedTuple.quaternion);  // 4D sensory
console.log(nestedTuple.octonion);    // 8D pathway
console.log(nestedTuple.dodecanion);  // 12D cortical
```

### H3DecisionModule

Implements the triplet decision-making unit (Section 7.8):

```typescript
import { H3DecisionModule } from './nn';

const h3 = new H3DecisionModule(
  12,  // input size (from dodecanion)
  1    // output size (decision)
);

// Three inputs: H₁ (primary), H₂ (secondary), H₃ (context)
const decision = h3.forwardH3(h1, h2, h3);
```

### Complete Time Crystal Brain Model

Full integration example:

```typescript
import { 
  TimeCrystalBrainModel,
  createCognitivePatternClassifier 
} from './nn/prime-tensor-integration';

// Create complete brain model
const brainModel = new TimeCrystalBrainModel(
  64,  // input size
  10   // output classes
);

// Forward pass
const { decision, nestedTuple } = brainModel.forward(input);

// Training
brainModel.train();
brainModel.zeroGradParameters();
const gradInput = brainModel.backward(input, gradDecision);
brainModel.updateParameters(0.01);

// Summary
console.log(brainModel.summary());
```

### Multi-Sensory Integration

Process multiple sensory modalities:

```typescript
import { MultiSensoryIntegrationModel } from './nn/prime-tensor-integration';

const multiSensory = new MultiSensoryIntegrationModel(
  32,  // visual input size
  24,  // auditory input size
  16   // tactile input size
);

const {
  visualQuaternion,
  auditoryQuaternion,
  tactileQuaternion,
  fusedOctonion,
  corticalDodecanion
} = multiSensory.forward(visualInput, auditoryInput, tactileInput);
```

### PrimeTensorContainer

Organize and manage prime-shaped tensors:

```typescript
import { PrimeTensorContainer } from './nn';

const container = new PrimeTensorContainer();

// Add tensors
container.add('sensory', primeShapedQuaternion);
container.add('pathway', primeShapedOctonion);
container.add('cortical', primeShapedDodecanion);

// Query tensors
const quaternions = container.getByType('quaternion');
const spatialTensors = container.getByPrime(2); // tensors with prime 2

// Get statistics
const summary = container.summary();
console.log(summary.byType);   // { quaternion: 1, octonion: 1, dodecanion: 1 }
console.log(summary.byPrime);  // { 2: 1, 3: 1, 5: 1, ... }
```

### Constants

```typescript
import { PRIME_DIMENSIONS, FUNDAMENTAL_PRIMES } from './nn';

console.log(PRIME_DIMENSIONS.QUATERNION);   // 4
console.log(PRIME_DIMENSIONS.OCTONION);     // 8
console.log(PRIME_DIMENSIONS.DODECANION);   // 12

console.log(FUNDAMENTAL_PRIMES);
// [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```

### Testing

Run the prime tensor test suite:

```bash
npm run test:prime-tensors
# or
node src/nn/test-prime-tensors.ts
```

### Theoretical Foundation

These modules implement the **Time Crystal Brain Model** from NanoBrain Chapter 7, which describes how:

- The brain uses prime number patterns at all organizational scales
- Information flows through nested dimensional transformations (4D → 8D → 12D)
- Consciousness emerges from 12D dodecanion manifolds
- Decision-making uses triplet H3 units at all neural scales
- Multi-dimensional algebras (quaternions, octonions, dodecanions) enable efficient cognitive processing

### References

- NanoBrain Chapter 7: Complete Time Crystal Brain Model
- Section 7.1.1: Four, Eight, and Twelve Imaginary Worlds
- Section 7.7: Brain's Wheel of Primes (Octonion)
- Section 7.8: H3 Decision Device
- Section 7.11: Hexagonal Prime Lattice

## License

MIT License - Same as the parent NanoBrain project

## References

- [torch/nn GitHub Repository](https://github.com/torch/nn)
- [Torch7 Documentation](http://torch.ch/)
- [NanoBrain Cognitive Architecture](../README.md)
- [NanoBrain Chapter 7: Time Crystal Brain Model](../../docs/CHAPTER_7_SUMMARY.md)
