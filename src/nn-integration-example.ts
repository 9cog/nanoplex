/**
 * Integration Example: Using torch/nn with NanoBrain Cognitive Architecture
 * 
 * This example demonstrates how the new torch/nn module integrates
 * with the existing cognitive architecture components.
 */

import { Module } from './nn/Module.js';
import { Linear, Dropout, Embedding } from './nn/Linear.js';
import { ReLU, Tanh, LogSoftmax } from './nn/Activation.js';
import { Sequential, Concat } from './nn/Container.js';
import { Criterion, MSECriterion } from './nn/Criterion.js';
import { GgmlTensor } from './core/GgmlTensorKernel.js';

// Weight initialization functions (from index.ts)
function initXavier(module: Module): void {
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

function initHe(module: Module): void {
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
 * Example 1: Simple Cognitive Pattern Classifier
 * Uses the nn module to build a neural network for pattern classification
 */
export function createCognitivePatternClassifier(
  inputDim: number,
  hiddenDim: number,
  numClasses: number
): Module {
  const model = new Sequential([
    new Linear(inputDim, hiddenDim),
    new ReLU(),
    new Dropout(0.5),
    new Linear(hiddenDim, Math.floor(hiddenDim / 2)),
    new ReLU(),
    new Linear(Math.floor(hiddenDim / 2), numClasses),
    new LogSoftmax(1)
  ]);
  
  // Initialize with Xavier
  initXavier(model);
  
  return model;
}

/**
 * Example 2: Embedding-based Cognitive Memory
 * Creates a model for learning embeddings of cognitive states
 */
export function createCognitiveMemoryEmbedding(
  vocabularySize: number,
  embeddingDim: number,
  outputDim: number
): Module {
  return new Sequential([
    new Embedding(vocabularySize, embeddingDim),
    new Linear(embeddingDim, embeddingDim * 2),
    new Tanh(),
    new Linear(embeddingDim * 2, outputDim)
  ]);
}

/**
 * Example 3: Multi-Branch Cognitive Processor
 * Demonstrates parallel processing of cognitive inputs
 */
export function createMultiBranchCognitiveProcessor(
  inputDim: number,
  branchDims: number[]
): Module {
  const branches = branchDims.map(dim => 
    new Sequential([
      new Linear(inputDim, dim),
      new ReLU(),
      new Linear(dim, Math.floor(dim / 2))
    ])
  );
  
  const totalOutput = branchDims.reduce((sum, dim) => sum + Math.floor(dim / 2), 0);
  
  return new Sequential([
    new Concat(1, branches),
    new Linear(totalOutput, Math.floor(totalOutput / 2)),
    new ReLU()
  ]);
}

/**
 * Example 4: Training Loop for Cognitive Models
 */
export class CognitiveModelTrainer {
  private model: Module;
  private criterion: Criterion;
  private learningRate: number;
  private history: { epoch: number; loss: number }[] = [];
  
  constructor(
    model: Module,
    criterion: Criterion = new MSECriterion(),
    learningRate: number = 0.01
  ) {
    this.model = model;
    this.criterion = criterion;
    this.learningRate = learningRate;
  }
  
  /**
   * Train for one epoch
   */
  train(
    inputs: GgmlTensor[],
    targets: GgmlTensor[]
  ): number {
    this.model.train();
    let totalLoss = 0;
    
    for (let i = 0; i < inputs.length; i++) {
      this.model.zeroGradParameters();
      
      // Forward pass
      const output = this.model.forward(inputs[i]);
      const loss = this.criterion.forward(output, targets[i]);
      
      // Backward pass
      const gradOutput = this.criterion.backward(output, targets[i]);
      this.model.backward(inputs[i], gradOutput);
      
      // Update parameters
      this.model.updateParameters(this.learningRate);
      
      totalLoss += loss;
    }
    
    return totalLoss / inputs.length;
  }
  
  /**
   * Evaluate on validation data
   */
  evaluate(
    inputs: GgmlTensor[],
    targets: GgmlTensor[]
  ): number {
    this.model.evaluate();
    let totalLoss = 0;
    
    for (let i = 0; i < inputs.length; i++) {
      const output = this.model.forward(inputs[i]);
      const loss = this.criterion.forward(output, targets[i]);
      totalLoss += loss;
    }
    
    return totalLoss / inputs.length;
  }
  
  /**
   * Train for multiple epochs
   */
  trainEpochs(
    trainInputs: GgmlTensor[],
    trainTargets: GgmlTensor[],
    validInputs: GgmlTensor[],
    validTargets: GgmlTensor[],
    numEpochs: number
  ): void {
    for (let epoch = 0; epoch < numEpochs; epoch++) {
      const trainLoss = this.train(trainInputs, trainTargets);
      const validLoss = this.evaluate(validInputs, validTargets);
      
      this.history.push({ epoch, loss: trainLoss });
      
      console.log(
        `Epoch ${epoch + 1}/${numEpochs} - ` +
        `Train Loss: ${trainLoss.toFixed(6)}, ` +
        `Valid Loss: ${validLoss.toFixed(6)}`
      );
    }
  }
  
  /**
   * Get training history
   */
  getHistory(): { epoch: number; loss: number }[] {
    return this.history;
  }
}

/**
 * Example 5: Cognitive State Predictor
 * A complete example showing integration with cognitive architecture
 */
export class CognitiveStatePredictor {
  private model: Module;
  private trainer: CognitiveModelTrainer;
  
  constructor(stateDim: number, predictedDim: number) {
    // Build model architecture
    this.model = new Sequential([
      new Linear(stateDim, 128),
      new ReLU(),
      new Dropout(0.3),
      new Linear(128, 64),
      new ReLU(),
      new Linear(64, predictedDim),
      new Tanh() // Output in [-1, 1]
    ]);
    
    // Initialize model
    initHe(this.model);
    
    // Create trainer
    this.trainer = new CognitiveModelTrainer(
      this.model,
      new MSECriterion(),
      0.001
    );
  }
  
  /**
   * Train the predictor
   */
  train(
    states: Float32Array[],
    predictions: Float32Array[],
    epochs: number = 100
  ): void {
    // Convert to tensors
    const stateTensors = states.map(s => this.arrayToTensor(s, 'state'));
    const predTensors = predictions.map(p => this.arrayToTensor(p, 'pred'));
    
    // Split into train/validation
    const splitIdx = Math.floor(states.length * 0.8);
    const trainStates = stateTensors.slice(0, splitIdx);
    const trainPreds = predTensors.slice(0, splitIdx);
    const validStates = stateTensors.slice(splitIdx);
    const validPreds = predTensors.slice(splitIdx);
    
    // Train
    this.trainer.trainEpochs(
      trainStates,
      trainPreds,
      validStates,
      validPreds,
      epochs
    );
  }
  
  /**
   * Predict future state
   */
  predict(state: Float32Array): Float32Array {
    this.model.evaluate();
    const input = this.arrayToTensor(state, 'input');
    const output = this.model.forward(input);
    return output.data;
  }
  
  /**
   * Helper to convert array to tensor
   */
  private arrayToTensor(arr: Float32Array, name: string): GgmlTensor {
    return {
      id: `${name}_${Math.random().toString(36).substr(2, 9)}`,
      shape: [1, arr.length],
      data: new Float32Array(arr),
      dtype: 'f32',
      requires_grad: true,
      name
    };
  }
}

/**
 * Example 6: Demo Usage
 */
export function demoNNIntegration(): void {
  console.log('=== torch/nn Integration Demo ===\n');
  
  // Create a pattern classifier
  console.log('1. Creating Cognitive Pattern Classifier...');
  const classifier = createCognitivePatternClassifier(64, 128, 10);
  console.log(`   Created model with ${classifier.parameters.length} parameter tensors\n`);
  
  // Create cognitive memory embedding
  console.log('2. Creating Cognitive Memory Embedding...');
  const memory = createCognitiveMemoryEmbedding(1000, 128, 64);
  console.log(`   Created embedding model with ${memory.parameters.length} parameter tensors\n`);
  
  // Create multi-branch processor
  console.log('3. Creating Multi-Branch Cognitive Processor...');
  const processor = createMultiBranchCognitiveProcessor(128, [64, 32, 16]);
  console.log(`   Created parallel processor with ${processor.parameters.length} parameter tensors\n`);
  
  // Create and train a state predictor
  console.log('4. Creating and Training Cognitive State Predictor...');
  const predictor = new CognitiveStatePredictor(32, 16);
  
  // Generate synthetic training data
  const numSamples = 100;
  const states: Float32Array[] = [];
  const predictions: Float32Array[] = [];
  
  for (let i = 0; i < numSamples; i++) {
    const state = new Float32Array(32);
    const pred = new Float32Array(16);
    
    for (let j = 0; j < 32; j++) {
      state[j] = Math.random() * 2 - 1;
    }
    for (let j = 0; j < 16; j++) {
      pred[j] = Math.sin(state[j % 32] * Math.PI);
    }
    
    states.push(state);
    predictions.push(pred);
  }
  
  console.log(`   Training on ${numSamples} samples...`);
  predictor.train(states, predictions, 10);
  
  // Make a prediction
  const testState = new Float32Array(32);
  for (let i = 0; i < 32; i++) {
    testState[i] = Math.random() * 2 - 1;
  }
  const prediction = predictor.predict(testState);
  console.log(`   Prediction shape: [${prediction.length}]`);
  console.log(`   Sample values: [${Array.from(prediction.slice(0, 3)).map(x => x.toFixed(4)).join(', ')}...]\n`);
  
  console.log('=== Demo Complete ===');
}

// Run demo if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  demoNNIntegration();
}
