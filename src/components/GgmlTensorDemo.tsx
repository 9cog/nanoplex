/**
 * GGML Tensor Demo Component
 * 
 * Demonstrates the cognitive kernel tensor operations with real-time visualization
 * of encoding, attention allocation, reasoning, and meta-cognitive feedback.
 */

import React, { useState, useEffect } from 'react';
import { Brain, Cpu, Activity, Zap, Play, Pause, RotateCcw } from 'lucide-react';
import { useCognitiveKernel } from '../hooks/useCognitiveKernel';
import { AtomSpaceAtom } from '../hooks/useEnhancedAtomSpace';
import { AtomeseNode, DetailedStats, TestResult } from '../types';
import { ProcessingResult } from '../core/UnifiedCognitiveKernel';

export function GgmlTensorDemo() {
  const {
    isReady,
    isRunning,
    stats,
    error,
    startKernel,
    stopKernel,
    resetKernel,
    processAtoms,
    processAtomeseNodes,
    getTensorStats,
    runTestSuite,
    testResults,
    tensorCount,
    memoryUsage,
    attentionEfficiency,
    reasoningActivity,
    metaCognitiveCoherence
  } = useCognitiveKernel();

  const [demoAtoms, setDemoAtoms] = useState<AtomSpaceAtom[]>([]);
  const [demoAtomese, setDemoAtomese] = useState<AtomeseNode[]>([]);
  const [processingResults, setProcessingResults] = useState<ProcessingResult | null>(null);
  const [detailedStats, setDetailedStats] = useState<DetailedStats | null>(null);

  // Initialize demo data
  useEffect(() => {
    const atoms: AtomSpaceAtom[] = [
      {
        id: 'demo_concept_1',
        type: 'ConceptNode',
        name: 'Intelligence',
        truthValue: { strength: 0.9, confidence: 0.8, count: 10 },
        attentionValue: { sti: 80, lti: 40, vlti: true },
        importance: 0.9,
        timestamp: Date.now()
      },
      {
        id: 'demo_concept_2',
        type: 'ConceptNode',
        name: 'Consciousness',
        truthValue: { strength: 0.7, confidence: 0.6, count: 8 },
        attentionValue: { sti: 60, lti: 30, vlti: false },
        importance: 0.7,
        timestamp: Date.now()
      },
      {
        id: 'demo_predicate_1',
        type: 'PredicateNode',
        name: 'HasProperty',
        truthValue: { strength: 0.8, confidence: 0.9, count: 15 },
        attentionValue: { sti: 70, lti: 35, vlti: false },
        importance: 0.8,
        timestamp: Date.now()
      }
    ];

    const atomese: AtomeseNode[] = [
      {
        id: 'demo_atomese_1',
        type: 'ConceptNode',
        name: 'PhilosophicalTransformation',
        attentionValue: 1.0,
        truthValue: { strength: 0.95, confidence: 0.9 }
      },
      {
        id: 'demo_atomese_2',
        type: 'InheritanceLink',
        children: ['demo_atomese_1', 'demo_concept_1'],
        attentionValue: 0.8,
        truthValue: { strength: 0.7, confidence: 0.8 }
      }
    ];

    setDemoAtoms(atoms);
    setDemoAtomese(atomese);
  }, []);

  // Process demo atoms
  const handleProcessAtoms = () => {
    if (!isReady) return;

    const result = processAtoms(demoAtoms);
    setProcessingResults(result);

    // Get detailed stats
    const tensorStats = getTensorStats();
    setDetailedStats(tensorStats);
  };

  // Process demo Atomese
  const handleProcessAtomese = () => {
    if (!isReady) return;

    const result = processAtomeseNodes(demoAtomese);
    setProcessingResults(result);

    const tensorStats = getTensorStats();
    setDetailedStats(tensorStats);
  };

  // Run test suite
  const handleRunTests = async () => {
    if (!isReady) return;
    await runTestSuite();
  };

  return (
    <div className="space-y-6 p-6 bg-gray-900 rounded-lg">
      <div className="flex items-center space-x-3 mb-6">
        <Brain className="text-cyan-400" size={32} />
        <h2 className="text-2xl font-bold text-white">GGML Tensor Cognitive Kernel</h2>
      </div>

      {/* Status Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <Cpu className="text-blue-400" size={20} />
            <span className="text-gray-300 font-medium">Kernel Status</span>
          </div>
          <div className="space-y-1">
            <div className={`text-sm ${isReady ? 'text-green-400' : 'text-red-400'}`}>
              {isReady ? '● Ready' : '● Not Ready'}
            </div>
            <div className={`text-sm ${isRunning ? 'text-green-400' : 'text-gray-400'}`}>
              {isRunning ? '▶ Running' : '⏸ Stopped'}
            </div>
            {error && (
              <div className="text-xs text-red-400 mt-1">{error}</div>
            )}
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="text-green-400" size={20} />
            <span className="text-gray-300 font-medium">Tensor Stats</span>
          </div>
          <div className="space-y-1 text-sm text-gray-300">
            <div>Tensors: {tensorCount}</div>
            <div>Memory: {(memoryUsage / (1024 * 1024)).toFixed(1)}MB</div>
            <div>Attention: {(attentionEfficiency * 100).toFixed(1)}%</div>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <Zap className="text-purple-400" size={20} />
            <span className="text-gray-300 font-medium">Cognitive Activity</span>
          </div>
          <div className="space-y-1 text-sm text-gray-300">
            <div>Reasoning: {(reasoningActivity * 100).toFixed(1)}%</div>
            <div>Meta-Cog: {(metaCognitiveCoherence * 100).toFixed(1)}%</div>
            <div>Cycles: {stats?.state.cycle_count || 0}</div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={isRunning ? stopKernel : startKernel}
          disabled={!isReady}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
            isRunning
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-green-600 hover:bg-green-700 text-white'
          } disabled:bg-gray-600 disabled:cursor-not-allowed`}
        >
          {isRunning ? <Pause size={16} /> : <Play size={16} />}
          <span>{isRunning ? 'Stop' : 'Start'} Kernel</span>
        </button>

        <button
          onClick={resetKernel}
          disabled={!isReady}
          className="flex items-center space-x-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          <RotateCcw size={16} />
          <span>Reset</span>
        </button>

        <button
          onClick={handleProcessAtoms}
          disabled={!isReady}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          Process AtomSpace
        </button>

        <button
          onClick={handleProcessAtomese}
          disabled={!isReady}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          Process Atomese
        </button>

        <button
          onClick={handleRunTests}
          disabled={!isReady}
          className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          Run Test Suite
        </button>
      </div>

      {/* Processing Results */}
      {processingResults && (
        <div className="bg-gray-800 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-3">Processing Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Success:</span>
              <span className={`ml-2 ${processingResults.success ? 'text-green-400' : 'text-red-400'}`}>
                {processingResults.success ? '✓' : '✗'}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Cycle Time:</span>
              <span className="ml-2 text-cyan-400">{processingResults.cycle_time}ms</span>
            </div>
            <div>
              <span className="text-gray-400">Tensors:</span>
              <span className="ml-2 text-green-400">{processingResults.tensors_processed}</span>
            </div>
            <div>
              <span className="text-gray-400">Attention Updates:</span>
              <span className="ml-2 text-blue-400">{processingResults.attention_updates}</span>
            </div>
          </div>
          {processingResults.errors && processingResults.errors.length > 0 && (
            <div className="mt-3 p-2 bg-red-900 rounded text-red-300 text-xs">
              <strong>Errors:</strong>
              <ul className="mt-1 list-disc list-inside">
                {processingResults.errors.map((error: string, i: number) => (
                  <li key={i}>{error}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Detailed Stats */}
      {detailedStats && (
        <div className="bg-gray-800 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-3">Detailed Tensor Statistics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-md font-medium text-cyan-400 mb-2">Memory & Performance</h4>
              <div className="space-y-1 text-sm text-gray-300">
                <div>Total Tensors: {detailedStats.tensorStats?.total_tensors || 0}</div>
                <div>Memory Used: {((detailedStats.memoryStats?.used || 0) / (1024 * 1024)).toFixed(1)}MB</div>
                <div>Memory %: {(detailedStats.memoryStats?.percentage || 0).toFixed(1)}%</div>
                <div>Vocabulary Size: {detailedStats.vocabularyStats?.size || 0}</div>
              </div>
            </div>
            <div>
              <h4 className="text-md font-medium text-purple-400 mb-2">Cognitive Activity</h4>
              <div className="space-y-1 text-sm text-gray-300">
                <div>Attention Flows: {detailedStats.attentionFlows?.length || 0}</div>
                <div>Reasoning Chains: {detailedStats.reasoningChains?.length || 0}</div>
                <div>System States: {detailedStats.systemStateHistory?.length || 0}</div>
                <div>Active Nodes: {detailedStats.tensorStats?.active_nodes || 0}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Results */}
      {testResults && (
        <div className="bg-gray-800 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-3">Test Suite Results</h3>
          <div className="mb-3">
            <span className="text-green-400">Passed: {testResults.filter((r: TestResult) => r.passed).length}</span>
            <span className="text-gray-400 mx-2">/</span>
            <span className="text-gray-300">Total: {testResults.length}</span>
          </div>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {testResults.map((result: TestResult, i: number) => (
              <div key={i} className="flex items-start space-x-2 text-sm">
                <span className={result.passed ? 'text-green-400' : 'text-red-400'}>
                  {result.passed ? '✓' : '✗'}
                </span>
                <div className="flex-1">
                  <div className="text-gray-300 font-medium">{result.name}</div>
                  <div className="text-gray-400 text-xs">{result.message}</div>
                  <div className="text-gray-500 text-xs">{result.duration}ms</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Demo Data */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-800 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-3">Demo AtomSpace Data</h3>
          <div className="space-y-2 text-sm">
            {demoAtoms.map((atom, i) => (
              <div key={i} className="border border-gray-700 p-2 rounded">
                <div className="text-cyan-400 font-medium">{atom.type}: {atom.name}</div>
                <div className="text-gray-400 text-xs">
                  TV: [{atom.truthValue.strength.toFixed(2)}, {atom.truthValue.confidence.toFixed(2)}]
                  STI: {atom.attentionValue.sti}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-3">Demo Atomese Data</h3>
          <div className="space-y-2 text-sm">
            {demoAtomese.map((node, i) => (
              <div key={i} className="border border-gray-700 p-2 rounded">
                <div className="text-purple-400 font-medium">{node.type}: {node.name}</div>
                <div className="text-gray-400 text-xs">
                  Attention: {node.attentionValue?.toFixed(2)}
                  {node.children && ` | Children: ${node.children.length}`}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}