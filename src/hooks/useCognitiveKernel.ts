/**
 * Cognitive Kernel Integration Hook
 * 
 * This hook integrates the unified cognitive kernel with the existing 
 * cognitive architecture, providing seamless access to tensor-based
 * neural-symbolic processing capabilities.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  createDefaultCognitiveKernel,
  ProcessingResult,
  CognitiveKernelConfig
} from '../core/UnifiedCognitiveKernel';
import { CognitiveKernelTestSuite } from '../core/CognitiveKernelTestSuite';
import { AtomSpaceAtom, AtomSpaceLink } from './useEnhancedAtomSpace';
import { AtomeseNode, CognitiveNode } from '../types';

/**
 * Hook for cognitive kernel integration
 */
export function useCognitiveKernel() {
  const [state, setState] = useState<CognitiveKernelHookState>({
    kernel: null,
    isInitialized: false,
    isRunning: false,
    stats: null,
    error: null,
    testResults: null
  });

  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const testSuiteRef = useRef<CognitiveKernelTestSuite | null>(null);

  /**
   * Initialize the cognitive kernel
   */
  const initializeKernel = useCallback(() => {
    try {
      const kernel = createDefaultCognitiveKernel();
      
      // Set up event callbacks
      kernel.setCallbacks({
        onCycleComplete: (stats) => {
          setState(prev => ({ ...prev, stats }));
        },
        onError: (error) => {
          setState(prev => ({ ...prev, error }));
        },
        onStateChange: (kernelState) => {
          setState(prev => ({ 
            ...prev, 
            isRunning: kernelState.status === 'active',
            error: kernelState.last_error || null
          }));
        }
      });

      setState(prev => ({
        ...prev,
        kernel,
        isInitialized: true,
        error: null
      }));

      console.log('Cognitive kernel initialized successfully');
      
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: `Failed to initialize cognitive kernel: ${error}`,
        isInitialized: false
      }));
    }
  }, []);

  /**
   * Start the cognitive kernel
   */
  const startKernel = useCallback(() => {
    if (!state.kernel) {
      setState(prev => ({ ...prev, error: 'Kernel not initialized' }));
      return;
    }

    try {
      state.kernel.start();
      setState(prev => ({ ...prev, isRunning: true, error: null }));
    } catch (error) {
      setState(prev => ({ ...prev, error: `Failed to start kernel: ${error}` }));
    }
  }, [state.kernel]);

  /**
   * Stop the cognitive kernel
   */
  const stopKernel = useCallback(() => {
    if (!state.kernel) return;

    try {
      state.kernel.stop();
      setState(prev => ({ ...prev, isRunning: false }));
    } catch (error) {
      setState(prev => ({ ...prev, error: `Failed to stop kernel: ${error}` }));
    }
  }, [state.kernel]);

  /**
   * Process AtomSpace atoms through the cognitive kernel
   */
  const processAtoms = useCallback((
    atoms: AtomSpaceAtom[], 
    links: AtomSpaceLink[] = [],
    _: ProcessingOptions = {}
  ): ProcessingResult | null => {
    if (!state.kernel || !state.isInitialized) {
      console.warn('Cognitive kernel not ready for processing');
      return null;
    }

    try {
      const result = state.kernel.processAtoms(atoms, links);
      
      // Update stats after processing
      const stats = state.kernel.getCognitiveKernelStats();
      setState(prev => ({ ...prev, stats }));
      
      return result;
    } catch (error) {
      setState(prev => ({ ...prev, error: `Processing failed: ${error}` }));
      return null;
    }
  }, [state.kernel, state.isInitialized]);

  /**
   * Process Atomese nodes through the cognitive kernel
   */
  const processAtomeseNodes = useCallback((
    nodes: AtomeseNode[],
    _: ProcessingOptions = {}
  ): ProcessingResult | null => {
    if (!state.kernel || !state.isInitialized) {
      console.warn('Cognitive kernel not ready for Atomese processing');
      return null;
    }

    try {
      const result = state.kernel.processAtomeseNodes(nodes);
      
      // Update stats after processing
      const stats = state.kernel.getCognitiveKernelStats();
      setState(prev => ({ ...prev, stats }));
      
      return result;
    } catch (error) {
      setState(prev => ({ ...prev, error: `Atomese processing failed: ${error}` }));
      return null;
    }
  }, [state.kernel, state.isInitialized]);

  /**
   * Run cognitive kernel test suite
   */
  const runTestSuite = useCallback(async () => {
    if (!testSuiteRef.current) {
      testSuiteRef.current = new CognitiveKernelTestSuite();
    }

    try {
      setState(prev => ({ ...prev, error: null }));
      
      const results = await testSuiteRef.current.runAllTests();
      
      setState(prev => ({ ...prev, testResults: results }));
      
      return results;
    } catch (error) {
      setState(prev => ({ ...prev, error: `Test suite failed: ${error}` }));
      return null;
    }
  }, []);

  /**
   * Get current tensor statistics
   */
  const getTensorStats = useCallback(() => {
    if (!state.kernel) return null;

    try {
      const stats = state.kernel.getCognitiveKernelStats();
      return {
        tensorStats: stats.tensor_stats,
        memoryStats: state.kernel.getTensorKernel().getMemoryStats(),
        vocabularyStats: state.kernel.getEncoder().getVocabularyStats(),
        attentionFlows: state.kernel.getAttentionEngine().getAttentionFlows(),
        reasoningChains: state.kernel.getReasoningEngine().getActiveChains(),
        systemStateHistory: state.kernel.getMetaCognitiveEngine().getSystemStateHistory()
      };
    } catch (error) {
      console.error('Failed to get tensor stats:', error);
      return null;
    }
  }, [state.kernel]);

  /**
   * Update kernel configuration
   */
  const updateKernelConfig = useCallback((config: Partial<CognitiveKernelConfig>) => {
    if (!state.kernel) return;

    try {
      state.kernel.updateConfig(config);
      setState(prev => ({ ...prev, error: null }));
    } catch (error) {
      setState(prev => ({ ...prev, error: `Config update failed: ${error}` }));
    }
  }, [state.kernel]);

  /**
   * Reset the cognitive kernel
   */
  const resetKernel = useCallback(() => {
    if (!state.kernel) return;

    try {
      state.kernel.reset();
      setState(prev => ({ ...prev, stats: null, error: null }));
    } catch (error) {
      setState(prev => ({ ...prev, error: `Reset failed: ${error}` }));
    }
  }, [state.kernel]);

  /**
   * Get detailed component access
   */
  const getKernelComponents = useCallback(() => {
    if (!state.kernel) return null;

    return {
      tensorKernel: state.kernel.getTensorKernel(),
      encoder: state.kernel.getEncoder(),
      attentionEngine: state.kernel.getAttentionEngine(),
      reasoningEngine: state.kernel.getReasoningEngine(),
      metaCognitiveEngine: state.kernel.getMetaCognitiveEngine()
    };
  }, [state.kernel]);

  /**
   * Integration with existing cognitive nodes
   */
  const integrateCognitiveNodes = useCallback((cognitiveNodes: CognitiveNode[]) => {
    if (!state.kernel || cognitiveNodes.length === 0) return;

    try {
      // Convert cognitive nodes to AtomSpace format
      const atoms: AtomSpaceAtom[] = cognitiveNodes.map(node => ({
        id: node.id,
        type: node.type === 'prime' ? 'NumberNode' : 'ConceptNode',
        name: node.type === 'prime' ? node.value.toString() : node.id,
        truthValue: { 
          strength: node.activation, 
          confidence: node.confidence || 0.8, 
          count: 1 
        },
        attentionValue: {
          sti: (node.attentionValue || 0) * 100,
          lti: node.activation * 50,
          vlti: node.activation > 0.9
        },
        importance: node.activation * (node.confidence || 0.5),
        timestamp: node.timestamp || Date.now()
      }));

      // Create links from node connections
      const links: AtomSpaceLink[] = [];
      cognitiveNodes.forEach(node => {
        if (node.connections && node.connections.length > 0) {
          node.connections.forEach(connId => {
            const targetNode = cognitiveNodes.find(n => n.id === connId);
            if (targetNode) {
              links.push({
                id: `link-${node.id}-${connId}`,
                type: 'InheritanceLink',
                outgoing: [node.id, connId],
                truthValue: {
                  strength: (node.activation + targetNode.activation) / 2,
                  confidence: 0.7,
                  count: 1
                },
                attentionValue: {
                  sti: ((node.attentionValue || 0) + (targetNode.attentionValue || 0)) * 50,
                  lti: 25,
                  vlti: false
                },
                strength: (node.activation + targetNode.activation) / 2
              });
            }
          });
        }
      });

      return processAtoms(atoms, links);
    } catch (error) {
      setState(prev => ({ ...prev, error: `Integration failed: ${error}` }));
      return null;
    }
  }, [state.kernel, processAtoms]);

  /**
   * Auto-initialize on mount
   */
  useEffect(() => {
    if (!state.isInitialized && !state.kernel) {
      initializeKernel();
    }
  }, [state.isInitialized, state.kernel, initializeKernel]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
      }
      
      if (state.kernel) {
        state.kernel.shutdown();
      }
    };
  }, [state.kernel]);

  /**
   * Periodic stats update
   */
  useEffect(() => {
    if (state.isRunning && state.kernel) {
      const interval = setInterval(() => {
        try {
          const stats = state.kernel!.getCognitiveKernelStats();
          setState(prev => ({ ...prev, stats }));
        } catch (error) {
          console.error('Failed to update stats:', error);
        }
      }, 1000); // Update every second

      processingIntervalRef.current = interval;

      return () => clearInterval(interval);
    }
  }, [state.isRunning, state.kernel]);

  return {
    // State
    isInitialized: state.isInitialized,
    isRunning: state.isRunning,
    stats: state.stats,
    error: state.error,
    testResults: state.testResults,

    // Core operations
    initializeKernel,
    startKernel,
    stopKernel,
    resetKernel,

    // Processing functions
    processAtoms,
    processAtomeseNodes,
    integrateCognitiveNodes,

    // Configuration and monitoring
    updateKernelConfig,
    getTensorStats,
    getKernelComponents,

    // Testing
    runTestSuite,

    // Computed values
    isReady: state.isInitialized && !state.error,
    tensorCount: state.stats?.tensor_stats.total_tensors || 0,
    memoryUsage: state.stats?.tensor_stats.memory_usage || 0,
    attentionEfficiency: state.stats?.attention_stats.resource_utilization || 0,
    reasoningActivity: state.stats?.reasoning_stats.reasoning_throughput || 0,
    metaCognitiveCoherence: state.stats?.metacognitive_metrics.system_coherence || 0,
    overallPerformance: state.stats?.state.performance_metrics.overall_efficiency || 0
  };
}