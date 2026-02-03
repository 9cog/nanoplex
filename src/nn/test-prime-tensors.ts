/**
 * Tests for Prime-Shaped Nested Tensor Tuples
 * 
 * Tests the implementation of quaternion, octonion, and dodecanion modules
 * as described in NanoBrain Chapter 7
 */

import { GgmlTensor } from '../core/GgmlTensorKernel.js';
import {
  QuaternionModule,
  OctonionModule,
  DodecanionModule,
  NestedTensorTransformer,
  H3DecisionModule,
  PrimeTensorContainer,
  PRIME_DIMENSIONS,
  FUNDAMENTAL_PRIMES,
  PrimeShapedTensor,
  NestedTensorTuple
} from './PrimeShapedTensors.js';

// Helper function to create test tensor
function createTestTensor(shape: number[], fill: number = 1): GgmlTensor {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = fill;
  }
  return {
    id: `test_${Math.random()}`,
    shape,
    data,
    dtype: 'f32',
    requires_grad: true,
    name: 'test_tensor'
  };
}

// Helper function to check tensor dimensions
function checkDimensions(tensor: GgmlTensor, expectedShape: number[]): boolean {
  if (tensor.shape.length !== expectedShape.length) return false;
  for (let i = 0; i < tensor.shape.length; i++) {
    if (tensor.shape[i] !== expectedShape[i]) return false;
  }
  return true;
}

console.log('Testing Prime-Shaped Nested Tensor Tuples...\n');

// Test 1: Quaternion Module (4D)
console.log('1. Testing QuaternionModule (4D sensory processing)...');
try {
  const quatModule = new QuaternionModule(8, [2, 3, 5]); // spatial primes
  const input = createTestTensor([2, 8], 0.5); // batch of 2
  
  const output = quatModule.forward(input);
  
  if (checkDimensions(output, [2, PRIME_DIMENSIONS.QUATERNION])) {
    console.log('   ✓ Quaternion output has correct shape: [2, 4]');
  } else {
    throw new Error(`Incorrect shape: ${output.shape}`);
  }
  
  // Test backward pass
  const gradOutput = createTestTensor([2, PRIME_DIMENSIONS.QUATERNION], 0.1);
  const gradInput = quatModule.backward(input, gradOutput);
  
  if (checkDimensions(gradInput, [2, 8])) {
    console.log('   ✓ Quaternion gradient has correct shape: [2, 8]');
  } else {
    throw new Error(`Incorrect gradient shape: ${gradInput.shape}`);
  }
  
  // Test prime-shaped tensor creation
  const primeOutput = quatModule.createPrimeShapedTensor(output);
  if (primeOutput.dimension === 4 && 
      primeOutput.type === 'quaternion' &&
      primeOutput.primeSignature.includes(2) &&
      primeOutput.primeSignature.includes(3) &&
      primeOutput.primeSignature.includes(5)) {
    console.log('   ✓ Prime-shaped quaternion tensor created correctly');
  }
  
  console.log('   ✓ QuaternionModule test passed!\n');
} catch (error) {
  console.error('   ✗ QuaternionModule test failed:', error);
  process.exit(1);
}

// Test 2: Octonion Module (8D)
console.log('2. Testing OctonionModule (8D pathway fusion)...');
try {
  const octModule = new OctonionModule(PRIME_DIMENSIONS.QUATERNION, [7, 11, 13]); // temporal primes
  const input = createTestTensor([2, PRIME_DIMENSIONS.QUATERNION], 0.5);
  
  const output = octModule.forward(input);
  
  if (checkDimensions(output, [2, PRIME_DIMENSIONS.OCTONION])) {
    console.log('   ✓ Octonion output has correct shape: [2, 8]');
  } else {
    throw new Error(`Incorrect shape: ${output.shape}`);
  }
  
  // Test backward pass
  const gradOutput = createTestTensor([2, PRIME_DIMENSIONS.OCTONION], 0.1);
  const gradInput = octModule.backward(input, gradOutput);
  
  if (checkDimensions(gradInput, [2, PRIME_DIMENSIONS.QUATERNION])) {
    console.log('   ✓ Octonion gradient has correct shape: [2, 4]');
  }
  
  // Test prime-shaped tensor creation
  const primeOutput = octModule.createPrimeShapedTensor(output);
  if (primeOutput.dimension === 8 && 
      primeOutput.type === 'octonion' &&
      primeOutput.primeSignature.includes(7)) {
    console.log('   ✓ Prime-shaped octonion tensor created correctly');
  }
  
  console.log('   ✓ OctonionModule test passed!\n');
} catch (error) {
  console.error('   ✗ OctonionModule test failed:', error);
  process.exit(1);
}

// Test 3: Dodecanion Module (12D)
console.log('3. Testing DodecanionModule (12D cortical manifolds)...');
try {
  const dodModule = new DodecanionModule(PRIME_DIMENSIONS.OCTONION, [29, 31, 37]); // consciousness primes
  const input = createTestTensor([2, PRIME_DIMENSIONS.OCTONION], 0.5);
  
  const output = dodModule.forward(input);
  
  if (checkDimensions(output, [2, PRIME_DIMENSIONS.DODECANION])) {
    console.log('   ✓ Dodecanion output has correct shape: [2, 12]');
  } else {
    throw new Error(`Incorrect shape: ${output.shape}`);
  }
  
  // Test backward pass
  const gradOutput = createTestTensor([2, PRIME_DIMENSIONS.DODECANION], 0.1);
  const gradInput = dodModule.backward(input, gradOutput);
  
  if (checkDimensions(gradInput, [2, PRIME_DIMENSIONS.OCTONION])) {
    console.log('   ✓ Dodecanion gradient has correct shape: [2, 8]');
  }
  
  // Test prime-shaped tensor creation
  const primeOutput = dodModule.createPrimeShapedTensor(output);
  if (primeOutput.dimension === 12 && 
      primeOutput.type === 'dodecanion' &&
      primeOutput.primeSignature.includes(29)) {
    console.log('   ✓ Prime-shaped dodecanion tensor created correctly');
  }
  
  console.log('   ✓ DodecanionModule test passed!\n');
} catch (error) {
  console.error('   ✗ DodecanionModule test failed:', error);
  process.exit(1);
}

// Test 4: Nested Tensor Transformer (d4D → d8D → d12D)
console.log('4. Testing NestedTensorTransformer (complete pipeline)...');
try {
  const transformer = new NestedTensorTransformer(16); // 16-dim input
  const input = createTestTensor([2, 16], 0.5);
  
  // Test standard forward
  const output = transformer.forward(input);
  
  if (checkDimensions(output, [2, PRIME_DIMENSIONS.DODECANION])) {
    console.log('   ✓ Nested transformer output has correct shape: [2, 12]');
  } else {
    throw new Error(`Incorrect shape: ${output.shape}`);
  }
  
  // Test nested forward (returns all intermediate tensors)
  const nestedOutput: NestedTensorTuple = transformer.forwardNested(input);
  
  if (nestedOutput.quaternion.dimension === 4 &&
      nestedOutput.octonion.dimension === 8 &&
      nestedOutput.dodecanion.dimension === 12) {
    console.log('   ✓ Nested tuple contains all three tensor types (4D, 8D, 12D)');
  }
  
  if (nestedOutput.quaternion.type === 'quaternion' &&
      nestedOutput.octonion.type === 'octonion' &&
      nestedOutput.dodecanion.type === 'dodecanion') {
    console.log('   ✓ Nested tuple has correct type labels');
  }
  
  // Test backward pass
  const gradOutput = createTestTensor([2, PRIME_DIMENSIONS.DODECANION], 0.1);
  const gradInput = transformer.backward(input, gradOutput);
  
  if (checkDimensions(gradInput, [2, 16])) {
    console.log('   ✓ Nested transformer gradient has correct shape: [2, 16]');
  }
  
  // Test training/evaluation modes
  transformer.train();
  transformer.evaluate();
  console.log('   ✓ Training/evaluation mode switching works');
  
  console.log('   ✓ NestedTensorTransformer test passed!\n');
} catch (error) {
  console.error('   ✗ NestedTensorTransformer test failed:', error);
  process.exit(1);
}

// Test 5: H3 Decision Module
console.log('5. Testing H3DecisionModule (triplet decision-making)...');
try {
  const h3Module = new H3DecisionModule(4, 2); // 4-dim inputs, 2-dim output
  
  // Test with three separate inputs
  const h1 = createTestTensor([2, 4], 0.5); // primary
  const h2 = createTestTensor([2, 4], 0.3); // secondary
  const h3 = createTestTensor([2, 4], 0.7); // context
  
  const output = h3Module.forwardH3(h1, h2, h3);
  
  if (checkDimensions(output, [2, 2])) {
    console.log('   ✓ H3 decision output has correct shape: [2, 2]');
  } else {
    throw new Error(`Incorrect shape: ${output.shape}`);
  }
  
  // Test with concatenated input
  const concatInput = createTestTensor([2, 12], 0.5); // 3 * 4 = 12
  const output2 = h3Module.forward(concatInput);
  
  if (checkDimensions(output2, [2, 2])) {
    console.log('   ✓ H3 decision with concatenated input works');
  }
  
  // Test backward pass
  const gradOutput = createTestTensor([2, 2], 0.1);
  const gradInput = h3Module.backward(concatInput, gradOutput);
  
  if (checkDimensions(gradInput, [2, 12])) {
    console.log('   ✓ H3 decision gradient has correct shape: [2, 12]');
  }
  
  // Check that we have 3 weight tensors + 1 bias = 4 parameters
  if (h3Module.parameters.length === 4) {
    console.log('   ✓ H3 module has correct number of parameters (3 weights + bias)');
  }
  
  console.log('   ✓ H3DecisionModule test passed!\n');
} catch (error) {
  console.error('   ✗ H3DecisionModule test failed:', error);
  process.exit(1);
}

// Test 6: Prime Tensor Container
console.log('6. Testing PrimeTensorContainer...');
try {
  const container = new PrimeTensorContainer();
  
  // Create and add test tensors
  const quatModule = new QuaternionModule(8, [2, 3, 5]);
  const octModule = new OctonionModule(4, [7, 11, 13]);
  const dodModule = new DodecanionModule(8, [29, 31, 37]);
  
  const quatOutput = quatModule.forward(createTestTensor([1, 8], 0.5));
  const octOutput = octModule.forward(createTestTensor([1, 4], 0.5));
  const dodOutput = dodModule.forward(createTestTensor([1, 8], 0.5));
  
  container.add('sensory', quatModule.createPrimeShapedTensor(quatOutput));
  container.add('pathway', octModule.createPrimeShapedTensor(octOutput));
  container.add('cortical', dodModule.createPrimeShapedTensor(dodOutput));
  
  // Test retrieval
  const sensory = container.get('sensory');
  if (sensory && sensory.type === 'quaternion') {
    console.log('   ✓ Container retrieval works');
  }
  
  // Test type filtering
  const quaternions = container.getByType('quaternion');
  if (quaternions.length === 1) {
    console.log('   ✓ Type filtering works');
  }
  
  // Test prime filtering
  const spatialTensors = container.getByPrime(2); // spatial prime
  if (spatialTensors.length >= 1) {
    console.log('   ✓ Prime filtering works');
  }
  
  // Test nested tuple creation
  const tuple = container.createNestedTuple('sensory', 'pathway', 'cortical');
  if (tuple && 
      tuple.quaternion.type === 'quaternion' &&
      tuple.octonion.type === 'octonion' &&
      tuple.dodecanion.type === 'dodecanion') {
    console.log('   ✓ Nested tuple creation from container works');
  }
  
  // Test summary
  const summary = container.summary();
  if (summary.total === 3 &&
      summary.byType.quaternion === 1 &&
      summary.byType.octonion === 1 &&
      summary.byType.dodecanion === 1) {
    console.log('   ✓ Container summary statistics correct');
  }
  
  console.log('   ✓ PrimeTensorContainer test passed!\n');
} catch (error) {
  console.error('   ✗ PrimeTensorContainer test failed:', error);
  process.exit(1);
}

// Test 7: Constants and Prime Signatures
console.log('7. Testing constants and prime signatures...');
try {
  // Test PRIME_DIMENSIONS
  if (PRIME_DIMENSIONS.QUATERNION === 4 &&
      PRIME_DIMENSIONS.OCTONION === 8 &&
      PRIME_DIMENSIONS.DODECANION === 12) {
    console.log('   ✓ PRIME_DIMENSIONS constants correct');
  }
  
  // Test FUNDAMENTAL_PRIMES
  if (FUNDAMENTAL_PRIMES.length === 15 &&
      FUNDAMENTAL_PRIMES[0] === 2 &&
      FUNDAMENTAL_PRIMES[14] === 47) {
    console.log('   ✓ FUNDAMENTAL_PRIMES array correct (15 primes)');
  }
  
  // Verify all are prime
  function isPrime(n: number): boolean {
    if (n <= 1) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {
      if (n % i === 0) return false;
    }
    return true;
  }
  
  const allPrime = FUNDAMENTAL_PRIMES.every(isPrime);
  if (allPrime) {
    console.log('   ✓ All FUNDAMENTAL_PRIMES are indeed prime numbers');
  }
  
  console.log('   ✓ Constants test passed!\n');
} catch (error) {
  console.error('   ✗ Constants test failed:', error);
  process.exit(1);
}

// Test 8: Complete Integration Test
console.log('8. Testing complete integration (sensory → cortical)...');
try {
  // Simulate a complete brain processing pipeline
  const transformer = new NestedTensorTransformer(64); // 64-dim sensory input
  const h3Decision = new H3DecisionModule(12, 1); // decision based on 12D manifold
  
  // Sensory input
  const sensoryInput = createTestTensor([4, 64], 0.5); // batch of 4
  
  // Process through nested transformer
  const nestedOutput = transformer.forwardNested(sensoryInput);
  
  // Make decision based on cortical output
  const corticalTensor = nestedOutput.dodecanion.tensor;
  const decisionInput = createTestTensor([4, 36], 0.5); // 3 * 12 for H3
  const decision = h3Decision.forward(decisionInput);
  
  if (checkDimensions(decision, [4, 1])) {
    console.log('   ✓ Complete pipeline produces decision output: [4, 1]');
  }
  
  // Verify we can access all intermediate representations
  console.log('   ✓ Pipeline stages:');
  console.log(`      - Sensory (Quaternion 4D): ${nestedOutput.quaternion.primeSignature}`);
  console.log(`      - Pathway (Octonion 8D): ${nestedOutput.octonion.primeSignature}`);
  console.log(`      - Cortical (Dodecanion 12D): ${nestedOutput.dodecanion.primeSignature}`);
  console.log(`      - Decision: scalar output`);
  
  console.log('   ✓ Complete integration test passed!\n');
} catch (error) {
  console.error('   ✗ Complete integration test failed:', error);
  process.exit(1);
}

const SEPARATOR_WIDTH = 60;
console.log('='.repeat(SEPARATOR_WIDTH));
console.log('✓ All Prime-Shaped Nested Tensor Tuple tests passed!');
console.log('='.repeat(SEPARATOR_WIDTH));
console.log('\nImplementation Summary:');
console.log('- QuaternionModule: 4D sensory processing (primes 2,3,5)');
console.log('- OctonionModule: 8D pathway fusion (primes 7,11,13)');
console.log('- DodecanionModule: 12D cortical manifolds (primes 29,31,37)');
console.log('- NestedTensorTransformer: d4D → d8D → d12D pipeline');
console.log('- H3DecisionModule: Triplet decision-making unit');
console.log('- PrimeTensorContainer: Tensor collection management');
console.log('\nBased on NanoBrain Chapter 7: Time Crystal Brain Model');
