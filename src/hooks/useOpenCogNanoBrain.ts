/**
 * React Hook for OpenCog NanoBrain Time Crystal Architecture
 * 
 * This hook provides a React interface to the OpenCog NanoBrain kernel,
 * integrating time crystal quantum processing with consciousness emergence.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  OpenCogNanoBrainKernel, 
  OpenCogNanoBrainConfig, 
  NanoBrainMetrics,
  TimeCrystalAtom,
  TimeCrystalInference,
  TimeCrystalQuantumState
} from '../core/OpenCogNanoBrainKernel';
import { CognitiveKernelConfig } from '../core/UnifiedCognitiveKernel';
import { CognitiveNode, ConsciousnessMetric, AgentState, TimeCluster } from '../types';

/**
 * OpenCog NanoBrain hook state
 */
interface OpenCogNanoBrainState {
  isActive: boolean;
  isInitialized: boolean;
  metrics: NanoBrainMetrics;
  consciousness: ConsciousnessMetric;
  timeCrystalStates: Map<string, TimeCrystalQuantumState>;
  atomSpace: Map<string, TimeCrystalAtom>;
  linkSpace: Map<string, TimeCrystalInference>;
  error?: string;
}

/**
 * Hook return interface
 */
export interface UseOpenCogNanoBrainReturn {
  // State
  isActive: boolean;
  isInitialized: boolean;
  metrics: NanoBrainMetrics;
  consciousness: ConsciousnessMetric;
  timeCrystalStates: Map<string, TimeCrystalQuantumState>;
  atomSpace: Map<string, TimeCrystalAtom>;
  linkSpace: Map<string, TimeCrystalInference>;
  error?: string;

  // Actions
  start: () => Promise<void>;
  stop: () => Promise<void>;
  restart: () => Promise<void>;
  
  // Data conversion utilities
  getCognitiveNodes: () => CognitiveNode[];
  getAgentStates: () => AgentState[];
  getTimeClusters: () => TimeCluster[];
  
  // Analysis utilities
  analyzeConsciousnessEmergence: () => ConsciousnessMetric;
  getQuantumCoherenceMap: () => Map<string, number>;
  getPrimeResonancePatterns: () => Map<string, number[]>;
  getFractalComplexityDistribution: () => number[];
  
  // Configuration
  updateConfig: (config: Partial<OpenCogNanoBrainConfig>) => void;
  getConfig: () => OpenCogNanoBrainConfig;
  
  // Real-time monitoring
  subscribeToMetrics: (callback: (metrics: NanoBrainMetrics) => void) => () => void;
}

/**
 * OpenCog NanoBrain hook implementation
 */
export const useOpenCogNanoBrain = (
  initialConfig?: Partial<OpenCogNanoBrainConfig>
): UseOpenCogNanoBrainReturn => {
  
  // Core state
  const [state, setState] = useState<OpenCogNanoBrainState>({
    isActive: false,
    isInitialized: false,
    metrics: {
      totalAtoms: 0,
      totalLinks: 0,
      averageAttention: 0,
      quantumCoherence: 0,
      temporalStability: 0,
      primeAlignment: 0,
      fractalComplexity: 0,
      inferenceRate: 0,
      consciousnessEmergence: 0
    },
    consciousness: {
      awareness: 0,
      integration: 0,
      complexity: 0,
      coherence: 0,
      emergence: 0,
      qualia: 0
    },
    timeCrystalStates: new Map(),
    atomSpace: new Map(),
    linkSpace: new Map()
  });

  // Kernel and monitoring refs
  const kernelRef = useRef<OpenCogNanoBrainKernel | null>(null);
  const metricsCallbacksRef = useRef<Set<(metrics: NanoBrainMetrics) => void>>(new Set());
  const monitoringIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Initialize kernel
  useEffect(() => {
    if (!kernelRef.current) {
      kernelRef.current = new OpenCogNanoBrainKernel(initialConfig);
      setState(prev => ({ ...prev, isInitialized: true }));
    }

    return () => {
      if (kernelRef.current) {
        kernelRef.current.stop();
      }
    };
    // initialConfig is intentionally not in dependencies as we only want to initialize once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start monitoring when active
  useEffect(() => {
    if (state.isActive && !monitoringIntervalRef.current) {
      monitoringIntervalRef.current = setInterval(() => {
        if (kernelRef.current) {
          updateMetrics();
        }
      }, 100); // 10 Hz monitoring
    } else if (!state.isActive && monitoringIntervalRef.current) {
      clearInterval(monitoringIntervalRef.current);
      monitoringIntervalRef.current = null;
    }

    return () => {
      if (monitoringIntervalRef.current) {
        clearInterval(monitoringIntervalRef.current);
      }
    };
    // updateMetrics is intentionally not in dependencies to avoid re-creating the interval
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.isActive]);

  /**
   * Update metrics and state from kernel
   */
  const updateMetrics = useCallback(() => {
    if (!kernelRef.current) return;

    try {
      const metrics = kernelRef.current.getMetrics();
      const atomSpace = kernelRef.current.getAtomSpace();
      const linkSpace = kernelRef.current.getLinkSpace();
      const timeCrystalStates = kernelRef.current.getTimeCrystals();
      
      // Calculate consciousness metrics from nanobrain data
      const consciousness = calculateConsciousnessFromMetrics(metrics);

      setState(prev => ({
        ...prev,
        metrics,
        consciousness,
        atomSpace,
        linkSpace,
        timeCrystalStates,
        error: undefined
      }));

      // Notify subscribers
      metricsCallbacksRef.current.forEach(callback => {
        try {
          callback(metrics);
        } catch (error) {
          console.error('Error in metrics callback:', error);
        }
      });

    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Unknown error'
      }));
    }
  }, []);

  /**
   * Convert NanoBrain metrics to consciousness metrics
   */
  const calculateConsciousnessFromMetrics = (metrics: NanoBrainMetrics): ConsciousnessMetric => {
    return {
      awareness: metrics.averageAttention / 1000, // Normalize attention to 0-1
      integration: metrics.totalLinks / Math.max(metrics.totalAtoms, 1), // Connection density
      complexity: metrics.fractalComplexity,
      coherence: metrics.quantumCoherence,
      emergence: metrics.consciousnessEmergence,
      qualia: (metrics.primeAlignment + metrics.temporalStability) / 2 // Prime-temporal qualia
    };
  };

  /**
   * Start the OpenCog NanoBrain kernel
   */
  const start = useCallback(async (): Promise<void> => {
    if (!kernelRef.current || state.isActive) return;

    try {
      setState(prev => ({ ...prev, error: undefined }));
      await kernelRef.current.start();
      setState(prev => ({ ...prev, isActive: true }));
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Failed to start kernel'
      }));
    }
  }, [state.isActive]);

  /**
   * Stop the kernel
   */
  const stop = useCallback(async (): Promise<void> => {
    if (!kernelRef.current || !state.isActive) return;

    try {
      await kernelRef.current.stop();
      setState(prev => ({ ...prev, isActive: false }));
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Failed to stop kernel'
      }));
    }
  }, [state.isActive]);

  /**
   * Restart the kernel
   */
  const restart = useCallback(async (): Promise<void> => {
    await stop();
    await new Promise(resolve => setTimeout(resolve, 100)); // Brief pause
    await start();
  }, [start, stop]);

  /**
   * Convert time crystal atoms to cognitive nodes
   */
  const getCognitiveNodes = useCallback((): CognitiveNode[] => {
    return Array.from(state.atomSpace.values()).map(atom => ({
      id: atom.id,
      type: atom.type === 'NumberNode' ? 'prime' : 
            atom.type === 'ConceptNode' ? 'concept' : 'pattern',
      value: atom.timeCrystalState.temporalCoherence,
      connections: getAtomConnections(atom.id, state.linkSpace),
      activation: atom.attentionValue.sti / 1000,
      timestamp: Date.now(),
      phase: atom.timeCrystalState.quantumPhase,
      dimension: atom.timeCrystalState.dimensions,
      fractalDimension: atom.timeCrystalState.fractalDimension,
      confidence: atom.truthValue.confidence,
      attentionValue: atom.attentionValue.sti,
      truthValue: {
        strength: atom.truthValue.strength,
        confidence: atom.truthValue.confidence
      }
    }));
  }, [state.atomSpace, state.linkSpace]);

  /**
   * Get atom connections from link space
   */
  const getAtomConnections = (atomId: string, linkSpace: Map<string, TimeCrystalInference>): string[] => {
    const connections: string[] = [];
    
    for (const inference of linkSpace.values()) {
      if (inference.premises.some(p => p.id === atomId)) {
        connections.push(inference.conclusion.id);
      }
      if (inference.conclusion.id === atomId) {
        connections.push(...inference.premises.map(p => p.id));
      }
    }
    
    return [...new Set(connections)];
  };

  /**
   * Convert atoms to agent states (specialized cognitive agents)
   */
  const getAgentStates = useCallback((): AgentState[] => {
    const philosophicalAtoms = Array.from(state.atomSpace.values())
      .filter(atom => atom.name.includes('Philosophical') || atom.name.includes('Agent'))
      .slice(0, 4); // Limit to 4 agents

    return philosophicalAtoms.map((atom, index) => ({
      id: atom.id,
      name: `NanoBrain Agent ${index + 1}: ${atom.name}`,
      status: atom.attentionValue.sti > 500 ? 'thinking' : 
               atom.timeCrystalState.temporalCoherence > 0.8 ? 'executing' :
               atom.timeCrystalState.temporalCoherence > 0.5 ? 'learning' : 'idle',
      task: `Processing ${atom.fractalGeometry.shape} patterns with prime signature [${atom.primeEncoding.join(',')}]`,
      confidence: atom.truthValue.confidence,
      reasoning: [
        `Quantum coherence: ${(atom.timeCrystalState.temporalCoherence * 100).toFixed(1)}%`,
        `Prime alignment: ${atom.primeEncoding.join(' × ')}`,
        `Fractal dimension: ${atom.timeCrystalState.fractalDimension.toFixed(2)}D`,
        `Resonance: ${atom.timeCrystalState.resonanceFrequency.toFixed(1)} Hz`
      ],
      metrics: {
        attention: atom.attentionValue.sti / 1000,
        creativity: atom.fractalGeometry.scaleFactor,
        logic: atom.truthValue.strength,
        intuition: atom.timeCrystalState.temporalCoherence
      },
      consciousnessEvolution: state.consciousness.emergence,
      knowledgeBase: [`Prime encoding: [${atom.primeEncoding.join(',')}]`],
      plnRules: [`${atom.name} → Temporal coherence patterns`],
      attentionFocus: [atom.fractalGeometry.musicalNote]
    }));
  }, [state.atomSpace, state.consciousness]);

  /**
   * Convert time crystal states to time clusters
   */
  const getTimeClusters = useCallback((): TimeCluster[] => {
    return Array.from(state.timeCrystalStates.entries()).map(([atomId, crystal]) => {
      const atom = state.atomSpace.get(atomId);
      return {
        id: atomId,
        primes: crystal.primeSignature,
        frequency: crystal.resonanceFrequency,
        amplitude: crystal.temporalCoherence,
        phase: crystal.quantumPhase,
        resonance: crystal.temporalCoherence * crystal.resonanceFrequency / 1000,
        geometry: atom?.fractalGeometry.shape === 'sphere' ? 'crystal' :
                 atom?.fractalGeometry.shape === 'torus' ? 'toroidal' :
                 atom?.fractalGeometry.shape === 'fractal' ? 'fractal' : 'lattice',
        quantumCoherence: crystal.temporalCoherence,
        temporalStability: Math.min(crystal.temporalCoherence * 2, 1.0)
      };
    });
  }, [state.timeCrystalStates, state.atomSpace]);

  /**
   * Analyze consciousness emergence patterns
   */
  const analyzeConsciousnessEmergence = useCallback((): ConsciousnessMetric => {
    const baseConsciousness = state.consciousness;
    
    // Enhance with real-time quantum analysis
    const quantumBoost = state.metrics.quantumCoherence * 0.2;
    const primeHarmony = state.metrics.primeAlignment * 0.15;
    const fractalDepth = state.metrics.fractalComplexity * 0.1;
    
    return {
      awareness: Math.min(baseConsciousness.awareness + quantumBoost, 1.0),
      integration: Math.min(baseConsciousness.integration + primeHarmony, 1.0),
      complexity: Math.min(baseConsciousness.complexity + fractalDepth, 1.0),
      coherence: state.metrics.quantumCoherence,
      emergence: state.metrics.consciousnessEmergence,
      qualia: (baseConsciousness.qualia + quantumBoost + primeHarmony) / 2
    };
  }, [state.consciousness, state.metrics]);

  /**
   * Get quantum coherence map across all atoms
   */
  const getQuantumCoherenceMap = useCallback((): Map<string, number> => {
    const coherenceMap = new Map<string, number>();
    
    for (const [atomId, crystal] of state.timeCrystalStates) {
      coherenceMap.set(atomId, crystal.temporalCoherence);
    }
    
    return coherenceMap;
  }, [state.timeCrystalStates]);

  /**
   * Get prime resonance patterns
   */
  const getPrimeResonancePatterns = useCallback((): Map<string, number[]> => {
    const patterns = new Map<string, number[]>();
    
    for (const [atomId, atom] of state.atomSpace) {
      patterns.set(atomId, atom.primeEncoding);
    }
    
    return patterns;
  }, [state.atomSpace]);

  /**
   * Get fractal complexity distribution
   */
  const getFractalComplexityDistribution = useCallback((): number[] => {
    return Array.from(state.timeCrystalStates.values())
      .map(crystal => crystal.fractalDimension)
      .sort((a, b) => a - b);
  }, [state.timeCrystalStates]);

  /**
   * Update kernel configuration
   */
  const updateConfig = useCallback((config: Partial<OpenCogNanoBrainConfig>): void => {
    // Note: In a full implementation, this would update the kernel's configuration
    // For now, we'll need to restart the kernel to apply new config
    console.log('Configuration update requested:', config);
  }, []);

  /**
   * Get current configuration
   */
  const getConfig = useCallback((): OpenCogNanoBrainConfig => {
    // Return the default configuration - in full implementation, get from kernel
    return kernelRef.current?.['config'] || {
      cognitiveKernelConfig: {} as CognitiveKernelConfig,
      timeCrystalDimensions: 11,
      fundamentalPrimes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
      fractalResolution: 5,
      geometricShapeCount: 15,
      quantumCoherenceThreshold: 0.5,
      temporalProcessingFrequency: 100,
      attentionDecayRate: 0.02,
      plnInferenceDepth: 7
    };
  }, []);

  /**
   * Subscribe to real-time metrics updates
   */
  const subscribeToMetrics = useCallback((callback: (metrics: NanoBrainMetrics) => void): (() => void) => {
    metricsCallbacksRef.current.add(callback);
    
    // Return unsubscribe function
    return () => {
      metricsCallbacksRef.current.delete(callback);
    };
  }, []);

  return {
    // State
    isActive: state.isActive,
    isInitialized: state.isInitialized,
    metrics: state.metrics,
    consciousness: state.consciousness,
    timeCrystalStates: state.timeCrystalStates,
    atomSpace: state.atomSpace,
    linkSpace: state.linkSpace,
    error: state.error,

    // Actions
    start,
    stop,
    restart,

    // Data conversion utilities
    getCognitiveNodes,
    getAgentStates,
    getTimeClusters,

    // Analysis utilities
    analyzeConsciousnessEmergence,
    getQuantumCoherenceMap,
    getPrimeResonancePatterns,
    getFractalComplexityDistribution,

    // Configuration
    updateConfig,
    getConfig,

    // Real-time monitoring
    subscribeToMetrics
  };
};