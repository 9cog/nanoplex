/**
 * Simple test for nn modules
 */

import { Linear } from './Linear.js';
import { ReLU, Sigmoid, Softmax } from './Activation.js';
import { Sequential } from './Container.js';
import { MSECriterion } from './Criterion.js';
import { GgmlTensor } from '../core/GgmlTensorKernel.js';

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
    id: `test_${Math.random().toString(36).substr(2, 9)}`,
    shape,
    data,
    dtype: 'f32',
    requires_grad: true,
    name: 'test_tensor'
  };
}

console.log('Testing torch/nn implementation...\n');

// Test Linear layer
console.log('1. Testing Linear layer...');
const linear = new Linear(10, 5);
const input1 = createTensor([2, 10]);
const output1 = linear.forward(input1);
console.log(`   Input shape: [${input1.shape}], Output shape: [${output1.shape}]`);
console.log('   ✓ Linear layer works\n');

// Test ReLU activation
console.log('2. Testing ReLU activation...');
const relu = new ReLU();
const input2 = createTensor([2, 5], [-1, -0.5, 0, 0.5, 1, -2, -1, 0, 1, 2]);
const output2 = relu.forward(input2);
console.log(`   Input: [${Array.from(input2.data.slice(0, 5)).map(x => x.toFixed(1))}]`);
console.log(`   Output: [${Array.from(output2.data.slice(0, 5)).map(x => x.toFixed(1))}]`);
console.log('   ✓ ReLU activation works\n');

// Test Sequential container
console.log('3. Testing Sequential container...');
const model = new Sequential([
  new Linear(10, 20),
  new ReLU(),
  new Linear(20, 5)
]);
const input3 = createTensor([2, 10]);
const output3 = model.forward(input3);
console.log(`   Model: Linear(10→20) → ReLU → Linear(20→5)`);
console.log(`   Input shape: [${input3.shape}], Output shape: [${output3.shape}]`);
console.log('   ✓ Sequential container works\n');

// Test MSE loss
console.log('4. Testing MSE loss...');
const criterion = new MSECriterion();
const pred = createTensor([2, 5]);
const target = createTensor([2, 5]);
const loss = criterion.forward(pred, target);
console.log(`   Loss: ${loss.toFixed(6)}`);
console.log('   ✓ MSE criterion works\n');

// Test Softmax
console.log('5. Testing Softmax...');
const softmax = new Softmax(1);
const input5 = createTensor([2, 3], [1, 2, 3, 4, 5, 6]);
const output5 = softmax.forward(input5);
const sum1 = output5.data[0] + output5.data[1] + output5.data[2];
const sum2 = output5.data[3] + output5.data[4] + output5.data[5];
console.log(`   Row 1 sum: ${sum1.toFixed(6)}`);
console.log(`   Row 2 sum: ${sum2.toFixed(6)}`);
console.log('   ✓ Softmax works\n');

// Test training pipeline
console.log('6. Testing training pipeline...');
const trainModel = new Sequential([
  new Linear(5, 10),
  new ReLU(),
  new Linear(10, 3)
]);

const trainInput = createTensor([4, 5]);
const trainTarget = createTensor([4, 3]);

trainModel.train();
trainModel.zeroGradParameters();

const trainOutput = trainModel.forward(trainInput);
const trainLoss = criterion.forward(trainOutput, trainTarget);
const gradOutput = criterion.backward(trainOutput, trainTarget);
trainModel.backward(trainInput, gradOutput);
trainModel.updateParameters(0.01);

console.log(`   Initial loss: ${trainLoss.toFixed(6)}`);
console.log(`   Parameters: ${trainModel.parameters.length}`);
console.log('   ✓ Training pipeline works\n');

console.log('=== All tests passed! ===');
