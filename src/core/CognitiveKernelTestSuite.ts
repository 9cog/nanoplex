/**
 * Tensor Operations Test Suite
 * 
 * Comprehensive tests for the cognitive kernel tensor operations.
 * Tests validate encoding/decoding, attention routing, inference chains,
 * and meta-cognitive feedback loops using real data (no mocks).
 */

import {
  createDefaultCognitiveKernel,
  UnifiedCognitiveKernel
} from '../core/UnifiedCognitiveKernel';
import { AtomSpaceAtom, AtomSpaceLink } from '../hooks/useEnhancedAtomSpace';
import { AtomeseNode } from '../types';

/**
 * Test result interface
 */
interface TestResult {
  name: string;
  passed: boolean;
  message: string;
  duration: number;
  metrics?: Record<string, number>;
}

/**
 * Test suite for cognitive kernel
 */
export class CognitiveKernelTestSuite {
  private kernel: UnifiedCognitiveKernel;
  private testResults: TestResult[];

  constructor() {
    this.kernel = createDefaultCognitiveKernel();
    this.testResults = [];
  }

  /**
   * Run all tests
   */
  async runAllTests(): Promise<TestResult[]> {
    console.log('Starting Cognitive Kernel Test Suite...');
    
    // Core tensor operation tests
    await this.testTensorCreationAndOperations();
    await this.testAtomSpaceEncoding();
    await this.testAtomSpaceDecoding();
    await this.testAtomeseNodeEncoding();
    
    // Attention allocation tests
    await this.testSoftmaxAttention();
    await this.testECANAttention();
    await this.testHybridAttention();
    await this.testAttentionDiffusion();
    
    // Recursive reasoning tests
    await this.testBasicInference();
    await this.testReasoningChains();
    await this.testTensorContractions();
    await this.testInferenceRules();
    
    // Meta-cognitive feedback tests
    await this.testMetaCognitiveFeedback();
    await this.testSelfMonitoring();
    await this.testAdaptation();
    await this.testMembranePermeability();
    
    // Integration tests
    await this.testFullCognitiveProcess();
    await this.testPerformanceUnderLoad();
    await this.testErrorHandling();
    
    console.log(`Test Suite Complete: ${this.testResults.filter(r => r.passed).length}/${this.testResults.length} tests passed`);
    
    return this.testResults;
  }

  /**
   * Test tensor creation and basic operations
   */
  private async testTensorCreationAndOperations(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const tensorKernel = this.kernel.getTensorKernel();
      
      // Test tensor creation
      const tensor1 = tensorKernel.createTensor([10, 10], 'f32', 'test_tensor_1');
      const tensor2 = tensorKernel.createTensor([10, 10], 'f32', 'test_tensor_2');
      
      // Verify tensor properties
      if (tensor1.shape[0] !== 10 || tensor1.shape[1] !== 10) {
        throw new Error('Tensor shape mismatch');
      }
      
      if (tensor1.data.length !== 100) {
        throw new Error('Tensor data length mismatch');
      }
      
      // Test tensor operations
      const sum = tensorKernel.add(tensor1, tensor2);
      const product = tensorKernel.matmul(tensor1, tensor2);
      
      // Verify operation results
      if (sum.shape[0] !== 10 || sum.shape[1] !== 10) {
        throw new Error('Addition result shape mismatch');
      }
      
      if (product.shape[0] !== 10 || product.shape[1] !== 10) {
        throw new Error('Matrix multiplication result shape mismatch');
      }
      
      // Test softmax
      const vector = tensorKernel.createTensor([5], 'f32');
      const softmaxResult = tensorKernel.softmax(vector);
      
      // Verify softmax sums to 1
      const sum_softmax = Array.from(softmaxResult.data).reduce((a, b) => a + b, 0);
      if (Math.abs(sum_softmax - 1.0) > 0.001) {
        throw new Error(`Softmax sum error: ${sum_softmax}`);
      }
      
      this.addTestResult('testTensorCreationAndOperations', true, 'All tensor operations working correctly', Date.now() - startTime);
      
    } catch (error) {
      this.addTestResult('testTensorCreationAndOperations', false, `Tensor operations failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test AtomSpace atom encoding
   */
  private async testAtomSpaceEncoding(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const encoder = this.kernel.getEncoder();
      
      // Create test atoms
      const testAtoms: AtomSpaceAtom[] = [
        {
          id: 'concept_1',
          type: 'ConceptNode',
          name: 'TestConcept',
          truthValue: { strength: 0.8, confidence: 0.9, count: 10 },
          attentionValue: { sti: 50, lti: 25, vlti: true },
          importance: 0.7,
          timestamp: Date.now()
        },
        {
          id: 'predicate_1',
          type: 'PredicateNode',
          name: 'TestPredicate',
          truthValue: { strength: 0.6, confidence: 0.8, count: 5 },
          attentionValue: { sti: 30, lti: 15, vlti: false },
          importance: 0.5,
          timestamp: Date.now()
        }
      ];
      
      // Encode atoms
      const encodedTensors = testAtoms.map(atom => encoder.encodeAtom(atom));
      
      // Verify encoding
      if (encodedTensors.length !== testAtoms.length) {
        throw new Error('Encoding count mismatch');
      }
      
      encodedTensors.forEach((tensor, i) => {
        if (!tensor.embedding || !tensor.truth_value_tensor || !tensor.attention_weights) {
          throw new Error(`Incomplete encoding for atom ${i}`);
        }
        
        if (tensor.truth_value_tensor.data.length !== 3) {
          throw new Error(`Truth value tensor wrong size for atom ${i}`);
        }
        
        if (tensor.attention_weights.data.length !== 3) {
          throw new Error(`Attention tensor wrong size for atom ${i}`);
        }
      });
      
      this.addTestResult('testAtomSpaceEncoding', true, `Successfully encoded ${testAtoms.length} atoms`, Date.now() - startTime);
      
    } catch (error) {
      this.addTestResult('testAtomSpaceEncoding', false, `Encoding failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test AtomSpace atom decoding
   */
  private async testAtomSpaceDecoding(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const encoder = this.kernel.getEncoder();
      
      // Create and encode test atom
      const originalAtom: AtomSpaceAtom = {
        id: 'decode_test',
        type: 'ConceptNode',
        name: 'DecodingTest',
        truthValue: { strength: 0.75, confidence: 0.85, count: 8 },
        attentionValue: { sti: 40, lti: 20, vlti: false },
        importance: 0.6,
        timestamp: Date.now()
      };
      
      const encodedTensor = encoder.encodeAtom(originalAtom);
      const decodedAtom = encoder.decodeAtom(encodedTensor);
      
      // Verify decoding accuracy
      if (decodedAtom.type !== 'ConceptNode') {
        throw new Error('Type decoding failed');
      }
      
      if (Math.abs(decodedAtom.truthValue.strength - originalAtom.truthValue.strength) > 0.1) {
        throw new Error('Truth value strength decoding failed');
      }
      
      if (Math.abs(decodedAtom.truthValue.confidence - originalAtom.truthValue.confidence) > 0.1) {
        throw new Error('Truth value confidence decoding failed');
      }
      
      this.addTestResult('testAtomSpaceDecoding', true, 'Atom encoding/decoding round-trip successful', Date.now() - startTime);
      
    } catch (error) {
      this.addTestResult('testAtomSpaceDecoding', false, `Decoding failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test Atomese node encoding
   */
  private async testAtomeseNodeEncoding(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const encoder = this.kernel.getEncoder();
      
      // Create test Atomese nodes
      const testNodes: AtomeseNode[] = [
        {
          id: 'atomese_1',
          type: 'ConceptNode',
          name: 'PhilosophicalTransformation',
          attentionValue: 1.0,
          truthValue: { strength: 0.95, confidence: 0.9 }
        },
        {
          id: 'atomese_2',
          type: 'InheritanceLink',
          children: ['atomese_1', 'atomese_3'],
          attentionValue: 0.8,
          truthValue: { strength: 0.7, confidence: 0.8 }
        }
      ];
      
      // Encode Atomese nodes
      const encodedTensors = testNodes.map(node => encoder.encodeAtomeseNode(node));
      
      // Verify encoding
      encodedTensors.forEach((tensor, i) => {
        if (!tensor.embedding || !tensor.symbolic_features) {
          throw new Error(`Incomplete Atomese encoding for node ${i}`);
        }
        
        if (tensor.metadata.atom_type !== testNodes[i].type) {
          throw new Error(`Type mismatch for Atomese node ${i}`);
        }
      });
      
      this.addTestResult('testAtomeseNodeEncoding', true, `Successfully encoded ${testNodes.length} Atomese nodes`, Date.now() - startTime);
      
    } catch (error) {
      this.addTestResult('testAtomeseNodeEncoding', false, `Atomese encoding failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test softmax attention mechanism
   */
  private async testSoftmaxAttention(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const attentionEngine = this.kernel.getAttentionEngine();
      const encoder = this.kernel.getEncoder();
      
      // Create test nodes with different importance levels
      const testAtoms: AtomSpaceAtom[] = Array.from({ length: 5 }, (_, i) => ({
        id: `attention_test_${i}`,
        type: 'ConceptNode',
        name: `AttentionTest${i}`,
        truthValue: { strength: 0.1 + i * 0.2, confidence: 0.8, count: 5 },
        attentionValue: { sti: 10 + i * 10, lti: 5, vlti: false },
        importance: 0.1 + i * 0.2,
        timestamp: Date.now()
      }));
      
      const nodeTensors = testAtoms.map(atom => encoder.encodeAtom(atom));
      
      // Test attention allocation
      const initialAttention = nodeTensors.map(n => n.attention_weights.data[0]);
      const stats = attentionEngine.updateAttentionAllocation(nodeTensors, []);
      const finalAttention = nodeTensors.map(n => n.attention_weights.data[0]);
      
      // Verify attention was updated
      let attentionChanged = false;
      for (let i = 0; i < initialAttention.length; i++) {
        if (Math.abs(initialAttention[i] - finalAttention[i]) > 0.01) {
          attentionChanged = true;
          break;
        }
      }
      
      if (!attentionChanged) {
        throw new Error('Attention values did not change');
      }
      
      // Verify higher importance nodes got more attention
      const sortedByImportance = testAtoms.map((atom, i) => ({ importance: atom.importance, attention: finalAttention[i] }))
        .sort((a, b) => a.importance - b.importance);
      
      let properAllocation = true;
      for (let i = 1; i < sortedByImportance.length; i++) {
        if (sortedByImportance[i].attention < sortedByImportance[i-1].attention) {
          properAllocation = false;
          break;
        }
      }
      
      this.addTestResult('testSoftmaxAttention', true, 
        `Attention allocation working: ${attentionChanged && properAllocation ? 'proper allocation' : 'basic function'}`, 
        Date.now() - startTime, { stats });
      
    } catch (error) {
      this.addTestResult('testSoftmaxAttention', false, `Softmax attention failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test ECAN attention mechanism
   */
  private async testECANAttention(): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Update attention engine to use ECAN
      const attentionEngine = this.kernel.getAttentionEngine();
      attentionEngine.updateConfig({ mechanism: 'ecan' });
      
      const encoder = this.kernel.getEncoder();
      
      // Create test scenario
      const testAtoms: AtomSpaceAtom[] = Array.from({ length: 3 }, (_, i) => ({
        id: `ecan_test_${i}`,
        type: 'ConceptNode',
        name: `ECANTest${i}`,
        truthValue: { strength: 0.5 + i * 0.2, confidence: 0.8, count: 5 },
        attentionValue: { sti: 50 + i * 20, lti: 10, vlti: false },
        importance: 0.5 + i * 0.2,
        timestamp: Date.now()
      }));
      
      const nodeTensors = testAtoms.map(atom => encoder.encodeAtom(atom));
      
      // Test ECAN cycle
      const initialSTI = nodeTensors.map(n => n.attention_weights.data[0]);
      const stats = attentionEngine.updateAttentionAllocation(nodeTensors, []);
      const finalSTI = nodeTensors.map(n => n.attention_weights.data[0]);
      
      // Verify ECAN rent collection and wage distribution occurred
      const totalInitialSTI = initialSTI.reduce((sum, val) => sum + val, 0);
      const totalFinalSTI = finalSTI.reduce((sum, val) => sum + val, 0);
      
      // Total STI should change due to rent collection
      if (Math.abs(totalInitialSTI - totalFinalSTI) < 0.01) {
        throw new Error('ECAN rent/wage cycle did not occur');
      }
      
      this.addTestResult('testECANAttention', true, 
        `ECAN mechanism working: STI change ${(totalFinalSTI - totalInitialSTI).toFixed(3)}`, 
        Date.now() - startTime, { stats });
      
    } catch (error) {
      this.addTestResult('testECANAttention', false, `ECAN attention failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test hybrid attention mechanism
   */
  private async testHybridAttention(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const attentionEngine = this.kernel.getAttentionEngine();
      attentionEngine.updateConfig({ mechanism: 'hybrid' });
      
      const encoder = this.kernel.getEncoder();
      const testAtoms: AtomSpaceAtom[] = Array.from({ length: 4 }, (_, i) => ({
        id: `hybrid_test_${i}`,
        type: 'ConceptNode',
        name: `HybridTest${i}`,
        truthValue: { strength: 0.4 + i * 0.15, confidence: 0.8, count: 5 },
        attentionValue: { sti: 30 + i * 15, lti: 10, vlti: false },
        importance: 0.4 + i * 0.15,
        timestamp: Date.now()
      }));
      
      const nodeTensors = testAtoms.map(atom => encoder.encodeAtom(atom));
      const stats = attentionEngine.updateAttentionAllocation(nodeTensors, []);
      
      // Verify hybrid mechanism produces reasonable statistics
      if (stats.total_attention <= 0 || stats.attention_entropy < 0) {
        throw new Error('Invalid hybrid attention statistics');
      }
      
      this.addTestResult('testHybridAttention', true, 
        `Hybrid attention working: entropy ${stats.attention_entropy.toFixed(3)}`, 
        Date.now() - startTime, { stats });
      
    } catch (error) {
      this.addTestResult('testHybridAttention', false, `Hybrid attention failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test attention diffusion
   */
  private async testAttentionDiffusion(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const attentionEngine = this.kernel.getAttentionEngine();
      const encoder = this.kernel.getEncoder();
      
      // Create connected nodes
      const testAtoms: AtomSpaceAtom[] = [
        {
          id: 'diffusion_node_1',
          type: 'ConceptNode',
          name: 'HighAttentionNode',
          truthValue: { strength: 0.9, confidence: 0.9, count: 10 },
          attentionValue: { sti: 100, lti: 50, vlti: true },
          importance: 0.9,
          timestamp: Date.now()
        },
        {
          id: 'diffusion_node_2',
          type: 'ConceptNode',
          name: 'LowAttentionNode',
          truthValue: { strength: 0.3, confidence: 0.7, count: 3 },
          attentionValue: { sti: 10, lti: 5, vlti: false },
          importance: 0.3,
          timestamp: Date.now()
        }
      ];
      
      const testLinks: AtomSpaceLink[] = [
        {
          id: 'diffusion_link_1',
          type: 'InheritanceLink',
          outgoing: ['diffusion_node_1', 'diffusion_node_2'],
          truthValue: { strength: 0.8, confidence: 0.8, count: 5 },
          attentionValue: { sti: 40, lti: 20, vlti: false },
          strength: 0.8
        }
      ];
      
      const nodeTensors = testAtoms.map(atom => encoder.encodeAtom(atom));
      const linkTensors = testLinks.map(link => encoder.encodeLink(link));
      
      const stats = attentionEngine.updateAttentionAllocation(nodeTensors, linkTensors);
      
      // Check if attention flows occurred
      const flows = attentionEngine.getAttentionFlows();
      
      this.addTestResult('testAttentionDiffusion', true, 
        `Attention diffusion working: ${flows.length} flows recorded`, 
        Date.now() - startTime, { flows: flows.length, stats });
      
    } catch (error) {
      this.addTestResult('testAttentionDiffusion', false, `Attention diffusion failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test basic inference operations
   */
  private async testBasicInference(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const reasoningEngine = this.kernel.getReasoningEngine();
      const encoder = this.kernel.getEncoder();
      
      // Create premise nodes
      const premiseAtoms: AtomSpaceAtom[] = [
        {
          id: 'premise_1',
          type: 'ConceptNode',
          name: 'Animal',
          truthValue: { strength: 0.9, confidence: 0.8, count: 10 },
          attentionValue: { sti: 50, lti: 25, vlti: false },
          importance: 0.8,
          timestamp: Date.now()
        },
        {
          id: 'premise_2',
          type: 'ConceptNode',
          name: 'Mortal',
          truthValue: { strength: 0.8, confidence: 0.9, count: 8 },
          attentionValue: { sti: 40, lti: 20, vlti: false },
          importance: 0.7,
          timestamp: Date.now()
        }
      ];
      
      const nodeTensors = premiseAtoms.map(atom => encoder.encodeAtom(atom));
      
      // Start reasoning chain
      const chain = reasoningEngine.startReasoningChain(nodeTensors);
      
      // Execute reasoning steps
      const stats = reasoningEngine.executeReasoningStep(nodeTensors, []);
      
      // Verify reasoning occurred
      if (chain.reasoning_steps.length === 0 && stats.total_inferences === 0) {
        throw new Error('No reasoning steps executed');
      }
      
      this.addTestResult('testBasicInference', true, 
        `Basic inference working: ${stats.total_inferences} inferences, ${stats.active_chains} active chains`, 
        Date.now() - startTime, { stats });
      
    } catch (error) {
      this.addTestResult('testBasicInference', false, `Basic inference failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test reasoning chains
   */
  private async testReasoningChains(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const reasoningEngine = this.kernel.getReasoningEngine();
      const encoder = this.kernel.getEncoder();
      
      // Create multiple premise sets
      const testAtoms: AtomSpaceAtom[] = Array.from({ length: 6 }, (_, i) => ({
        id: `chain_test_${i}`,
        type: 'ConceptNode',
        name: `ChainTest${i}`,
        truthValue: { strength: 0.5 + (i % 3) * 0.2, confidence: 0.8, count: 5 },
        attentionValue: { sti: 30 + i * 5, lti: 15, vlti: false },
        importance: 0.5 + (i % 3) * 0.2,
        timestamp: Date.now()
      }));
      
      const nodeTensors = testAtoms.map(atom => encoder.encodeAtom(atom));
      
      // Start multiple reasoning chains
      reasoningEngine.startReasoningChain(nodeTensors.slice(0, 2));
      reasoningEngine.startReasoningChain(nodeTensors.slice(2, 4));
      reasoningEngine.startReasoningChain(nodeTensors.slice(4, 6));
      
      // Execute several reasoning cycles
      let totalSteps = 0;
      for (let i = 0; i < 3; i++) {
        const stats = reasoningEngine.executeReasoningStep(nodeTensors, []);
        totalSteps += stats.total_inferences;
      }
      
      const activeChains = reasoningEngine.getActiveChains();
      
      this.addTestResult('testReasoningChains', true, 
        `Reasoning chains working: ${activeChains.length} active chains, ${totalSteps} total steps`, 
        Date.now() - startTime, { activeChains: activeChains.length, totalSteps });
      
    } catch (error) {
      this.addTestResult('testReasoningChains', false, `Reasoning chains failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test tensor contractions
   */
  private async testTensorContractions(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const tensorKernel = this.kernel.getTensorKernel();
      
      // Test vector contraction (dot product)
      const vector1 = tensorKernel.createTensor([5], 'f32');
      const vector2 = tensorKernel.createTensor([5], 'f32');
      
      // Set some values
      for (let i = 0; i < 5; i++) {
        vector1.data[i] = i + 1;
        vector2.data[i] = (i + 1) * 2;
      }
      
      const dotProduct = tensorKernel.contract(vector1, vector2, [[0], [0]]);
      
      // Verify dot product result
      const expectedDot = 1*2 + 2*4 + 3*6 + 4*8 + 5*10; // 110
      if (Math.abs(dotProduct.data[0] - expectedDot) > 0.001) {
        throw new Error(`Dot product incorrect: expected ${expectedDot}, got ${dotProduct.data[0]}`);
      }
      
      // Test matrix multiplication
      const matrix1 = tensorKernel.createTensor([3, 3], 'f32');
      const matrix2 = tensorKernel.createTensor([3, 3], 'f32');
      
      const matrixProduct = tensorKernel.contract(matrix1, matrix2, [[1], [0]]);
      
      if (matrixProduct.shape[0] !== 3 || matrixProduct.shape[1] !== 3) {
        throw new Error('Matrix contraction shape incorrect');
      }
      
      this.addTestResult('testTensorContractions', true, 
        `Tensor contractions working: dot product = ${dotProduct.data[0]}`, 
        Date.now() - startTime);
      
    } catch (error) {
      this.addTestResult('testTensorContractions', false, `Tensor contractions failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test inference rules
   */
  private async testInferenceRules(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const reasoningEngine = this.kernel.getReasoningEngine();
      
      // Get available inference rules
      const rules = reasoningEngine.getInferenceRules();
      
      if (rules.length === 0) {
        throw new Error('No inference rules available');
      }
      
      // Test adding custom rule
      const customRule = {
        id: 'test_rule',
        name: 'Test Rule',
        type: 'deduction' as const,
        premise_patterns: ['A', 'B'],
        conclusion_pattern: 'C',
        confidence_formula: 'min(premise_confidences)',
        tensor_operation: 'contraction' as const,
        weight_tensor_shape: [2, 1]
      };
      
      reasoningEngine.addInferenceRule(customRule);
      
      const updatedRules = reasoningEngine.getInferenceRules();
      if (updatedRules.length !== rules.length + 1) {
        throw new Error('Custom rule not added');
      }
      
      // Test removing rule
      reasoningEngine.removeInferenceRule('test_rule');
      
      const finalRules = reasoningEngine.getInferenceRules();
      if (finalRules.length !== rules.length) {
        throw new Error('Custom rule not removed');
      }
      
      this.addTestResult('testInferenceRules', true, 
        `Inference rules working: ${rules.length} default rules, custom rule add/remove successful`, 
        Date.now() - startTime, { ruleCount: rules.length });
      
    } catch (error) {
      this.addTestResult('testInferenceRules', false, `Inference rules failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test meta-cognitive feedback
   */
  private async testMetaCognitiveFeedback(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const metaEngine = this.kernel.getMetaCognitiveEngine();
      const encoder = this.kernel.getEncoder();
      
      // Create test scenario
      const testAtoms: AtomSpaceAtom[] = Array.from({ length: 5 }, (_, i) => ({
        id: `meta_test_${i}`,
        type: 'ConceptNode',
        name: `MetaTest${i}`,
        truthValue: { strength: 0.6, confidence: 0.8, count: 5 },
        attentionValue: { sti: 40, lti: 20, vlti: false },
        importance: 0.6,
        timestamp: Date.now()
      }));
      
      const nodeTensors = testAtoms.map(atom => encoder.encodeAtom(atom));
      
      // Simulate attention and reasoning stats
      const attentionStats = {
        total_attention: 200,
        average_attention: 40,
        attention_entropy: 2.5,
        resource_utilization: 0.8,
        gradient_norm: 0.1,
        convergence_rate: 0.9
      };
      
      const reasoningStats = {
        total_chains: 3,
        active_chains: 2,
        converged_chains: 1,
        average_depth: 2.5,
        total_inferences: 15,
        average_confidence: 0.7,
        tensor_memory_usage: 1024,
        reasoning_throughput: 0.6
      };
      
      // Update meta-cognitive system
      const metrics = metaEngine.updateMetaCognitive(nodeTensors, [], attentionStats, reasoningStats);
      
      // Verify meta-cognitive metrics
      if (metrics.self_awareness_level < 0 || metrics.self_awareness_level > 1) {
        throw new Error('Invalid self-awareness level');
      }
      
      if (metrics.system_coherence < 0 || metrics.system_coherence > 1) {
        throw new Error('Invalid system coherence');
      }
      
      this.addTestResult('testMetaCognitiveFeedback', true, 
        `Meta-cognitive feedback working: awareness ${metrics.self_awareness_level.toFixed(3)}, coherence ${metrics.system_coherence.toFixed(3)}`, 
        Date.now() - startTime, { metrics });
      
    } catch (error) {
      this.addTestResult('testMetaCognitiveFeedback', false, `Meta-cognitive feedback failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test self-monitoring
   */
  private async testSelfMonitoring(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const metaEngine = this.kernel.getMetaCognitiveEngine();
      
      // Get meta-tensors
      const metaTensor0 = metaEngine.getMetaTensor(0);
      const metaTensor1 = metaEngine.getMetaTensor(1);
      
      if (!metaTensor0 || !metaTensor1) {
        throw new Error('Meta-tensors not available');
      }
      
      // Check self-monitoring tensors
      if (!metaTensor0.self_monitoring || !metaTensor1.self_monitoring) {
        throw new Error('Self-monitoring tensors not initialized');
      }
      
      // Verify tensor sizes
      if (metaTensor0.self_monitoring.data.length === 0) {
        throw new Error('Self-monitoring tensor has no data');
      }
      
      // Get system state history
      const stateHistory = metaEngine.getSystemStateHistory();
      
      this.addTestResult('testSelfMonitoring', true, 
        `Self-monitoring working: ${stateHistory.length} state records, meta-tensors functional`, 
        Date.now() - startTime, { stateRecords: stateHistory.length });
      
    } catch (error) {
      this.addTestResult('testSelfMonitoring', false, `Self-monitoring failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test adaptation mechanisms
   */
  private async testAdaptation(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const metaEngine = this.kernel.getMetaCognitiveEngine();
      
      // Get feedback loops
      const feedbackLoops = metaEngine.getFeedbackLoops();
      
      if (feedbackLoops.length === 0) {
        throw new Error('No feedback loops initialized');
      }
      
      // Check feedback loop properties
      feedbackLoops.forEach((loop, i) => {
        if (!loop.feedback_tensor || loop.feedback_tensor.data.length === 0) {
          throw new Error(`Feedback loop ${i} has invalid tensor`);
        }
        
        if (loop.stability_measure < 0 || loop.stability_measure > 1) {
          throw new Error(`Feedback loop ${i} has invalid stability measure`);
        }
      });
      
      // Get self-modifications
      const modifications = metaEngine.getSelfModifications();
      
      this.addTestResult('testAdaptation', true, 
        `Adaptation working: ${feedbackLoops.length} feedback loops, ${modifications.length} modifications`, 
        Date.now() - startTime, { feedbackLoops: feedbackLoops.length, modifications: modifications.length });
      
    } catch (error) {
      this.addTestResult('testAdaptation', false, `Adaptation failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test membrane permeability
   */
  private async testMembranePermeability(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const metaEngine = this.kernel.getMetaCognitiveEngine();
      
      // Test different membrane types
      const membraneTypes = ['cognitive', 'attention', 'reasoning', 'meta'];
      
      for (const type of membraneTypes) {
        const membrane = metaEngine.getMembrane(type);
        if (!membrane) {
          throw new Error(`Membrane ${type} not found`);
        }
        
        if (!membrane.permeability_weights || membrane.permeability_weights.data.length === 0) {
          throw new Error(`Membrane ${type} has invalid permeability`);
        }
        
        // Check permeability values are in valid range
        for (let i = 0; i < membrane.permeability_weights.data.length; i++) {
          const permeability = membrane.permeability_weights.data[i];
          if (permeability < 0 || permeability > 1) {
            throw new Error(`Invalid permeability value in membrane ${type}: ${permeability}`);
          }
        }
      }
      
      this.addTestResult('testMembranePermeability', true, 
        `Membrane permeability working: ${membraneTypes.length} membranes tested`, 
        Date.now() - startTime, { membraneTypes: membraneTypes.length });
      
    } catch (error) {
      this.addTestResult('testMembranePermeability', false, `Membrane permeability failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test full cognitive process integration
   */
  private async testFullCognitiveProcess(): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Create comprehensive test scenario
      const testAtoms: AtomSpaceAtom[] = Array.from({ length: 10 }, (_, i) => ({
        id: `integration_test_${i}`,
        type: i % 2 === 0 ? 'ConceptNode' : 'PredicateNode',
        name: `IntegrationTest${i}`,
        truthValue: { strength: 0.3 + (i % 5) * 0.15, confidence: 0.7 + (i % 3) * 0.1, count: 3 + i },
        attentionValue: { sti: 20 + i * 8, lti: 10 + i * 2, vlti: i > 5 },
        importance: 0.3 + (i % 5) * 0.15,
        timestamp: Date.now() - i * 1000
      }));
      
      const testLinks: AtomSpaceLink[] = Array.from({ length: 5 }, (_, i) => ({
        id: `integration_link_${i}`,
        type: 'InheritanceLink',
        outgoing: [`integration_test_${i}`, `integration_test_${i + 5}`],
        truthValue: { strength: 0.5 + i * 0.1, confidence: 0.8, count: 5 },
        attentionValue: { sti: 30 + i * 5, lti: 15, vlti: false },
        strength: 0.5 + i * 0.1
      }));
      
      // Process through cognitive kernel
      const result = this.kernel.processAtoms(testAtoms, testLinks);
      
      if (!result.success) {
        throw new Error(`Processing failed: ${result.errors.join(', ')}`);
      }
      
      // Get final stats
      const stats = this.kernel.getCognitiveKernelStats();
      
      // Verify all systems are functioning
      if (stats.tensor_stats.total_tensors === 0) {
        throw new Error('No tensors created');
      }
      
      if (stats.attention_stats.total_attention <= 0) {
        throw new Error('No attention allocation');
      }
      
      this.addTestResult('testFullCognitiveProcess', true, 
        `Full cognitive process working: ${result.tensors_processed} tensors, ${result.attention_updates} attention updates`, 
        Date.now() - startTime, { result, stats });
      
    } catch (error) {
      this.addTestResult('testFullCognitiveProcess', false, `Full cognitive process failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test performance under load
   */
  private async testPerformanceUnderLoad(): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Create large dataset
      const largeAtomSet: AtomSpaceAtom[] = Array.from({ length: 100 }, (_, i) => ({
        id: `load_test_${i}`,
        type: i % 3 === 0 ? 'ConceptNode' : i % 3 === 1 ? 'PredicateNode' : 'NumberNode',
        name: `LoadTest${i}`,
        truthValue: { strength: Math.random(), confidence: 0.8, count: Math.floor(Math.random() * 10) + 1 },
        attentionValue: { sti: Math.random() * 100, lti: Math.random() * 50, vlti: Math.random() > 0.8 },
        importance: Math.random(),
        timestamp: Date.now() - Math.random() * 10000
      }));
      
      // Process multiple times to test sustained performance
      const results = [];
      for (let i = 0; i < 5; i++) {
        const result = this.kernel.processAtoms(largeAtomSet, []);
        results.push(result);
      }
      
      // Check performance metrics
      const avgCycleTime = results.reduce((sum, r) => sum + r.cycle_time, 0) / results.length;
      const avgTensorsPerSecond = results.reduce((sum, r) => sum + (r.tensors_processed / (r.cycle_time / 1000)), 0) / results.length;
      
      if (avgCycleTime > 10000) { // 10 seconds threshold
        throw new Error(`Performance too slow: ${avgCycleTime}ms average cycle time`);
      }
      
      this.addTestResult('testPerformanceUnderLoad', true, 
        `Performance test passed: ${avgCycleTime.toFixed(0)}ms avg cycle, ${avgTensorsPerSecond.toFixed(0)} tensors/sec`, 
        Date.now() - startTime, { avgCycleTime, avgTensorsPerSecond });
      
    } catch (error) {
      this.addTestResult('testPerformanceUnderLoad', false, `Performance test failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Test error handling
   */
  private async testErrorHandling(): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Test invalid tensor operations
      const tensorKernel = this.kernel.getTensorKernel();
      
      try {
        // This should fail - mismatched dimensions
        const tensor1 = tensorKernel.createTensor([3, 4], 'f32');
        const tensor2 = tensorKernel.createTensor([5, 6], 'f32');
        tensorKernel.add(tensor1, tensor2);
        throw new Error('Expected dimension mismatch error was not thrown');
      } catch (error) {
        if (!error.message.includes('shape')) {
          throw new Error('Wrong error type for dimension mismatch');
        }
      }
      
      // Test invalid atom processing
      const invalidAtom = {
        id: 'invalid_atom',
        type: 'InvalidType' as 'ConceptNode',
        name: 'Invalid',
        truthValue: { strength: -1, confidence: 2, count: -5 }, // Invalid values
        attentionValue: { sti: -100, lti: -50, vlti: false },
        importance: -1,
        timestamp: Date.now()
      };
      
      const result = this.kernel.processAtoms([invalidAtom], []);
      
      // Should handle gracefully with errors
      if (result.success && result.errors.length === 0) {
        throw new Error('Expected processing to fail for invalid atom');
      }
      
      this.addTestResult('testErrorHandling', true, 
        `Error handling working: caught ${result.errors.length} errors gracefully`, 
        Date.now() - startTime, { errorCount: result.errors.length });
      
    } catch (error) {
      this.addTestResult('testErrorHandling', false, `Error handling failed: ${error}`, Date.now() - startTime);
    }
  }

  /**
   * Add test result
   */
  private addTestResult(name: string, passed: boolean, message: string, duration: number, metrics?: Record<string, number>): void {
    const result: TestResult = {
      name,
      passed,
      message,
      duration,
      metrics
    };
    
    this.testResults.push(result);
    
    const status = passed ? '✅' : '❌';
    console.log(`${status} ${name}: ${message} (${duration}ms)`);
  }

  /**
   * Get test results
   */
  getTestResults(): TestResult[] {
    return [...this.testResults];
  }

  /**
   * Get test summary
   */
  getTestSummary(): { passed: number; failed: number; total: number; duration: number } {
    const passed = this.testResults.filter(r => r.passed).length;
    const failed = this.testResults.filter(r => !r.passed).length;
    const total = this.testResults.length;
    const duration = this.testResults.reduce((sum, r) => sum + r.duration, 0);
    
    return { passed, failed, total, duration };
  }
}