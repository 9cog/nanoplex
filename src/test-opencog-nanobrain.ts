/**
 * Basic test runner for OpenCog NanoBrain system
 */

import { OpenCogNanoBrainKernel } from './core/OpenCogNanoBrainKernel';
import { OpenCogNanoBrainTestSuite } from './core/OpenCogNanoBrainTestSuite';

async function testOpenCogNanoBrain() {
  console.log('🧠 Testing OpenCog NanoBrain Time Crystal Architecture...\n');

  // Test 1: Basic kernel initialization
  console.log('1. Testing kernel initialization...');
  try {
    const kernel = new OpenCogNanoBrainKernel();
    await kernel.start();
    
    const metrics = kernel.getMetrics();
    console.log('✓ Kernel initialized successfully');
    console.log(`   - Atoms: ${metrics.totalAtoms}`);
    console.log(`   - Quantum coherence: ${(metrics.quantumCoherence * 100).toFixed(1)}%`);
    console.log(`   - Consciousness emergence: ${(metrics.consciousnessEmergence * 100).toFixed(1)}%`);
    
    await kernel.stop();
  } catch (error) {
    console.error('✗ Kernel initialization failed:', error);
  }

  // Test 2: Run comprehensive test suite
  console.log('\n2. Running comprehensive test suite...');
  try {
    const testSuite = new OpenCogNanoBrainTestSuite();
    const results = await testSuite.runAllTests();
    
    const summary = testSuite.getTestSummary();
    console.log(`✓ Test suite completed: ${summary.passed}/${summary.total} tests passed`);
    console.log(`   - Duration: ${(summary.duration / 1000).toFixed(2)}s`);
    
    if (summary.failed > 0) {
      console.log('\n❌ Failed tests:');
      results.filter(r => !r.passed).forEach(result => {
        console.log(`   - ${result.name}: ${result.message}`);
      });
    }
    
    if (summary.passed === summary.total) {
      console.log('🎉 All tests passed! OpenCog NanoBrain system is working correctly.');
    }
    
  } catch (error) {
    console.error('✗ Test suite failed:', error);
  }
}

// Run if executed directly
if (typeof window === 'undefined') {
  testOpenCogNanoBrain().catch(console.error);
}

export { testOpenCogNanoBrain };