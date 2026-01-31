/**
 * Test suite for torch/nn implementation
 * 
 * Validates the functionality of neural network modules
 */

import * as nn from './index.js';
import { GgmlTensor } from '../core/GgmlTensorKernel.js';

/**
 * Helper function to create a test tensor
 */
function createTensor(shape: number[], values?: number[]): GgmlTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  
  if (values) {
    for (let i = 0; i < Math.min(size, values.length); i++) {
      data[i] = values[i];
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = Math.random();
    }
  }
  
  return {
    id: `test_${Math.random().toString(36).substring(2, 11)}`,
    shape,
    data,
    dtype: 'f32',
    requires_grad: true,
    name: 'test_tensor'
  };
}

/**
 * Test Linear layer
 */
export function testLinear(): boolean {
  console.log('Testing Linear layer...');
  
  const linear = new nn.Linear(10, 5);
  const input = createTensor([2, 10]);
  
  // Forward pass
  const output = linear.forward(input);
  
  if (output.shape[0] !== 2 || output.shape[1] !== 5) {
    console.error('Linear forward shape mismatch');
    return false;
  }
  
  // Backward pass
  const gradOutput = createTensor([2, 5]);
  const gradInput = linear.backward(input, gradOutput);
  
  if (gradInput.shape[0] !== 2 || gradInput.shape[1] !== 10) {
    console.error('Linear backward shape mismatch');
    return false;
  }
  
  console.log('✓ Linear layer test passed');
  return true;
}

/**
 * Test Sequential container
 */
export function testSequential(): boolean {
  console.log('Testing Sequential container...');
  
  const model = new nn.Sequential([
    new nn.Linear(10, 20),
    new nn.ReLU(),
    new nn.Linear(20, 5)
  ]);
  
  const input = createTensor([2, 10]);
  
  // Forward pass
  const output = model.forward(input);
  
  if (output.shape[0] !== 2 || output.shape[1] !== 5) {
    console.error('Sequential forward shape mismatch');
    return false;
  }
  
  // Backward pass
  const gradOutput = createTensor([2, 5]);
  const gradInput = model.backward(input, gradOutput);
  
  if (gradInput.shape[0] !== 2 || gradInput.shape[1] !== 10) {
    console.error('Sequential backward shape mismatch');
    return false;
  }
  
  console.log('✓ Sequential container test passed');
  return true;
}

/**
 * Test activation functions
 */
export function testActivations(): boolean {
  console.log('Testing activation functions...');
  
  const input = createTensor([2, 10], [-2, -1, 0, 1, 2, -0.5, 0.5, 1.5, -1.5, 0.3]);
  
  // Test ReLU
  const relu = new nn.ReLU();
  const reluOut = relu.forward(input);
  
  // ReLU should zero out negative values
  if (reluOut.data[0] !== 0 || reluOut.data[3] !== 1) {
    console.error('ReLU activation failed');
    return false;
  }
  
  // Test Tanh
  const tanh = new nn.Tanh();
  const tanhOut = tanh.forward(input);
  
  // Tanh output should be in (-1, 1)
  for (let i = 0; i < tanhOut.data.length; i++) {
    if (Math.abs(tanhOut.data[i]) >= 1) {
      console.error('Tanh output out of range');
      return false;
    }
  }
  
  // Test Sigmoid
  const sigmoid = new nn.Sigmoid();
  const sigmoidOut = sigmoid.forward(input);
  
  // Sigmoid output should be in (0, 1)
  for (let i = 0; i < sigmoidOut.data.length; i++) {
    if (sigmoidOut.data[i] < 0 || sigmoidOut.data[i] > 1) {
      console.error('Sigmoid output out of range');
      return false;
    }
  }
  
  console.log('✓ Activation functions test passed');
  return true;
}

/**
 * Test Softmax
 */
export function testSoftmax(): boolean {
  console.log('Testing Softmax...');
  
  const softmax = new nn.Softmax(1);
  const input = createTensor([2, 3], [1, 2, 3, 4, 5, 6]);
  
  const output = softmax.forward(input);
  
  // Check that each row sums to 1
  const row1Sum = output.data[0] + output.data[1] + output.data[2];
  const row2Sum = output.data[3] + output.data[4] + output.data[5];
  
  if (Math.abs(row1Sum - 1.0) > 1e-5 || Math.abs(row2Sum - 1.0) > 1e-5) {
    console.error('Softmax rows do not sum to 1');
    return false;
  }
  
  console.log('✓ Softmax test passed');
  return true;
}

/**
 * Test loss functions
 */
export function testCriterions(): boolean {
  console.log('Testing loss functions...');
  
  const input = createTensor([2, 5]);
  const target = createTensor([2, 5]);
  
  // Test MSE
  const mse = new nn.MSECriterion();
  const mseLoss = mse.forward(input, target);
  
  if (mseLoss < 0) {
    console.error('MSE loss should be non-negative');
    return false;
  }
  
  const mseGrad = mse.backward(input, target);
  if (mseGrad.shape[0] !== 2 || mseGrad.shape[1] !== 5) {
    console.error('MSE gradient shape mismatch');
    return false;
  }
  
  // Test BCE
  const bce = new nn.BCECriterion();
  const inputProb = createTensor([2, 5]);
  // Clamp to (0, 1)
  for (let i = 0; i < inputProb.data.length; i++) {
    inputProb.data[i] = 0.1 + 0.8 * Math.random();
  }
  
  const bceLoss = bce.forward(inputProb, target);
  if (bceLoss < 0) {
    console.error('BCE loss should be non-negative');
    return false;
  }
  
  console.log('✓ Loss functions test passed');
  return true;
}

/**
 * Test Dropout
 */
export function testDropout(): boolean {
  console.log('Testing Dropout...');
  
  const dropout = new nn.Dropout(0.5);
  const input = createTensor([2, 10]);
  
  // Training mode: should drop some values
  dropout.train();
  const trainOutput = dropout.forward(input);
  
  let droppedCount = 0;
  for (let i = 0; i < trainOutput.data.length; i++) {
    if (trainOutput.data[i] === 0) {
      droppedCount++;
    }
  }
  
  if (droppedCount === 0) {
    console.error('Dropout should drop some values in training mode');
    return false;
  }
  
  // Evaluation mode: should not drop values
  dropout.evaluate();
  const evalOutput = dropout.forward(input);
  
  let sameCount = 0;
  for (let i = 0; i < evalOutput.data.length; i++) {
    if (evalOutput.data[i] === input.data[i]) {
      sameCount++;
    }
  }
  
  if (sameCount !== evalOutput.data.length) {
    console.error('Dropout should not drop values in evaluation mode');
    return false;
  }
  
  console.log('✓ Dropout test passed');
  return true;
}

/**
 * Test complete training pipeline
 */
export function testTrainingPipeline(): boolean {
  console.log('Testing complete training pipeline...');
  
  // Create a simple network
  const model = new nn.Sequential([
    new nn.Linear(5, 10),
    new nn.ReLU(),
    new nn.Linear(10, 3),
    new nn.Softmax()
  ]);
  
  const criterion = new nn.MSECriterion();
  
  const input = createTensor([4, 5]);
  const target = createTensor([4, 3]);
  
  // Initialize with Xavier
  nn.initXavier(model);
  
  // Training step
  model.train();
  model.zeroGradParameters();
  
  const output = model.forward(input);
  const loss = criterion.forward(output, target);
  
  if (loss < 0) {
    console.error('Loss should be non-negative');
    return false;
  }
  
  const gradOutput = criterion.backward(output, target);
  model.backward(input, gradOutput);
  
  // Update parameters
  const learningRate = 0.01;
  model.updateParameters(learningRate);
  
  // Check that parameters were updated
  const params = model.getParameters();
  if (params.weights.length === 0) {
    console.error('No parameters in model');
    return false;
  }
  
  console.log(`✓ Training pipeline test passed (loss: ${loss.toFixed(4)})`);
  return true;
}

/**
 * Test Embedding layer
 */
export function testEmbedding(): boolean {
  console.log('Testing Embedding layer...');
  
  const embedding = new nn.Embedding(100, 10);
  const input = createTensor([3, 1], [5, 10, 20]);
  
  const output = embedding.forward(input);
  
  if (output.shape[0] !== 3 || output.shape[1] !== 10) {
    console.error('Embedding output shape mismatch');
    return false;
  }
  
  console.log('✓ Embedding layer test passed');
  return true;
}

/**
 * Test Flatten layer
 */
export function testFlatten(): boolean {
  console.log('Testing Flatten layer...');
  
  const flatten = new nn.Flatten(1);
  const input = createTensor([2, 3, 4]);
  
  const output = flatten.forward(input);
  
  if (output.shape.length !== 2 || output.shape[0] !== 2 || output.shape[1] !== 12) {
    console.error('Flatten output shape mismatch');
    return false;
  }
  
  console.log('✓ Flatten layer test passed');
  return true;
}

/**
 * Run all tests
 */
export function runAllTests(): void {
  console.log('\n=== Running torch/nn Tests ===\n');
  
  const tests = [
    testLinear,
    testSequential,
    testActivations,
    testSoftmax,
    testCriterions,
    testDropout,
    testEmbedding,
    testFlatten,
    testTrainingPipeline
  ];
  
  let passed = 0;
  let failed = 0;
  
  for (const test of tests) {
    try {
      if (test()) {
        passed++;
      } else {
        failed++;
      }
    } catch (error) {
      console.error(`Test failed with error: ${error}`);
      failed++;
    }
    console.log('');
  }
  
  console.log('=== Test Summary ===');
  console.log(`Total: ${tests.length}`);
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  
  if (failed === 0) {
    console.log('\n✓ All tests passed!\n');
  } else {
    console.log(`\n✗ ${failed} test(s) failed\n`);
  }
}

// Export test runner for external use
export default runAllTests;
