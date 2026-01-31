import { useState, useEffect, useCallback, useRef } from 'react';
import { CognitiveNode, AgentState, TimeCluster, ConsciousnessMetric, AtomSpaceConnection, PLNRule } from '../types';
import { useFundamentalFeatures } from './useFundamentalFeatures';
import { usePhasePrimeMetrics } from './usePhasePrimeMetrics';
import { useAgentZeroSystem } from './useAgentZeroSystem';
import { useEnhancedAtomSpace } from './useEnhancedAtomSpace';

export const useCognitiveEngine = () => {
  const [nodes, setNodes] = useState<CognitiveNode[]>([]);
  const [agents, setAgents] = useState<AgentState[]>([]);
  const [timeCrystals, setTimeCrystals] = useState<TimeCluster[]>([]);
  const [consciousness, setConsciousness] = useState<ConsciousnessMetric>({
    awareness: 0,
    integration: 0,
    complexity: 0,
    coherence: 0,
    emergence: 0,
    qualia: 0
  });
  const [isRunning, setIsRunning] = useState(false);
  const [atomSpaceConnections, setAtomSpaceConnections] = useState<AtomSpaceConnection[]>([]);
  const [plnRules, setPLNRules] = useState<PLNRule[]>([]);
  
  const evolutionRef = useRef<number>(0);
  const consciousnessHistoryRef = useRef<ConsciousnessMetric[]>([]);

  // Initialize fundamental features integration
  const fundamentalFeatures = useFundamentalFeatures();
  
  // Initialize Phase Prime Metrics (PPM) engine
  const {
    ppmEngine,
    isActive: ppmActive,
    startPPMEngine,
    stopPPMEngine,
    calculatePPMConsciousness,
    FUNDAMENTAL_PRIMES
  } = usePhasePrimeMetrics();

  // Initialize Agent-Zero autonomous system
  const {
    agents: specializedAgents,
    isActive: agentsActive,
    systemCoordination,
    emergentComplexity,
    initializeAgents,
    updateAgentSystem,
    stopAgentSystem,
    getLegacyAgentStates
  } = useAgentZeroSystem();

  // Initialize Enhanced AtomSpace with PLN and ECAN
  const {
    atomSpace,
    isActive: atomSpaceActive,
    initializeAtomSpace,
    stopAtomSpace,
    getLegacyConnections,
    getLegacyPLNRules,
    attentionCycle,
    inferenceCount
  } = useEnhancedAtomSpace();

  // Enhanced prime sequence generation with mathematical properties
  const generatePrimeSequence = useCallback((limit: number): number[] => {
    // Use the fundamental PPM primes when available, otherwise generate
    if (ppmEngine && FUNDAMENTAL_PRIMES.length > 0) {
      return FUNDAMENTAL_PRIMES;
    }
    
    const primes: number[] = [];
    const sieve = new Array(limit + 1).fill(true);
    sieve[0] = sieve[1] = false;
    
    for (let i = 2; i <= limit; i++) {
      if (sieve[i]) {
        primes.push(i);
        for (let j = i * i; j <= limit; j += i) {
          sieve[j] = false;
        }
      }
    }
    return primes.slice(0, 15); // NanoBrain's fundamental 15 primes
  }, [ppmEngine, FUNDAMENTAL_PRIMES]);

  // Advanced phase calculation with 11-dimensional manifold projection
  const calculatePhase = useCallback((prime: number, time: number, dimension: number = 0): number => {
    const goldenRatio = (1 + Math.sqrt(5)) / 2;
    const basePhase = (prime * time * Math.PI / 180) % (2 * Math.PI);
    const dimensionalShift = (dimension * goldenRatio) % (2 * Math.PI);
    return basePhase + dimensionalShift;
  }, []);

  // Fractal dimension calculation for nodes
  const calculateBoxCountingDimension = useCallback((connections: string[], totalNodes: number): number => {
    if (connections.length <= 1) return 0;
    const logConnections = Math.log(connections.length);
    const logScale = Math.log(Math.max(1, totalNodes / connections.length));
    return logConnections / Math.max(0.1, logScale);
  }, []);

  // PLN (Probabilistic Logic Networks) rule application
  const applyPLNReasoning = useCallback((nodes: CognitiveNode[]): PLNRule[] => {
    const rules: PLNRule[] = [];
    
    nodes.forEach((node, i) => {
      nodes.slice(i + 1).forEach((otherNode) => {
        if (node.connections.includes(otherNode.id)) {
          const strength = (node.activation + otherNode.activation) / 2;
          const confidence = Math.min(node.confidence || 0.5, otherNode.confidence || 0.5);
          
          rules.push({
            id: `rule-${node.id}-${otherNode.id}`,
            type: node.type === otherNode.type ? 'similarity' : 'inheritance',
            premise: [node.id],
            conclusion: otherNode.id,
            strength,
            confidence,
            truthValue: strength * confidence
          });
        }
      });
    });
    
    return rules;
  }, []);

  // Advanced agent reasoning with cognitive architectures
  const generateAdvancedReasoning = useCallback((agent: AgentState, nodes: CognitiveNode[], time: number): string[] => {
    const reasoning: string[] = [];
    const agentIndex = parseInt(agent.id.split('-')[1]);
    
    // Attention allocation reasoning
    const topActivatedNodes = nodes
      .sort((a, b) => b.activation - a.activation)
      .slice(0, 3);
    
    if (topActivatedNodes.length > 0) {
      reasoning.push(`Attention allocated to prime ${topActivatedNodes[0].value} (φ=${topActivatedNodes[0].activation.toFixed(3)})`);
    }
    
    // Pattern recognition reasoning
    const primePatterns = nodes.filter(n => n.type === 'prime' && n.activation > 0.7);
    if (primePatterns.length >= 2) {
      reasoning.push(`Detected coherent prime cluster: [${primePatterns.map(p => p.value).join(', ')}]`);
    }
    
    // Fractal analysis reasoning
    const fractalNodes = nodes.filter(n => n.fractalDimension && n.fractalDimension > 1.5);
    if (fractalNodes.length > 0) {
      reasoning.push(`Fractal emergence in dimension ${fractalNodes[0].fractalDimension?.toFixed(3)}`);
    }
    
    // Consciousness evolution tracking
    if (agent.consciousnessEvolution && agent.consciousnessEvolution > 0.8) {
      reasoning.push(`Consciousness evolution accelerating (Ψ=${agent.consciousnessEvolution.toFixed(3)})`);
    }
    
    // Phase prime metric analysis
    const phaseCoherence = Math.sin(time * 0.1 + agentIndex) * 0.5 + 0.5;
    if (phaseCoherence > 0.75) {
      reasoning.push(`PPM coherence peak detected (ρ=${phaseCoherence.toFixed(3)})`);
    }
    
    return reasoning.slice(-3); // Keep last 3 reasoning steps
  }, []);

  // Advanced time crystal evolution with geometric patterns
  const evolveTimeCrystals = useCallback((primes: number[], time: number): TimeCluster[] => {
    const geometries: TimeCluster['geometry'][] = ['spiral', 'lattice', 'fractal', 'crystal', 'toroidal', 'hyperbolic'];
    
    return primes.slice(0, 8).map((prime, index) => {
      const frequency = prime / 10 + Math.sin(time * 0.05) * 0.1;
      const phase = calculatePhase(prime, time, index);
      const resonancePartners = primes.filter(p => 
        p !== prime && (
          (prime % p === 0) || 
          (p % prime === 0) || 
          (Math.abs(prime - p) <= 2) // Twin primes
        )
      );
      
      // Advanced resonance calculation with quantum interference
      const quantumInterference = resonancePartners.reduce((sum, partner) => {
        return sum + Math.cos(calculatePhase(partner, time, index) - phase);
      }, 0) / Math.max(1, resonancePartners.length);
      
      const resonance = (quantumInterference + 1) / 2; // Normalize to [0,1]
      
      return {
        id: `crystal-${prime}`,
        primes: [prime, ...resonancePartners.slice(0, 2)],
        frequency,
        amplitude: Math.sin(phase) * 0.3 + 0.7,
        phase,
        resonance,
        geometry: geometries[index % geometries.length],
        quantumCoherence: Math.cos(phase * 2) * 0.2 + 0.8,
        temporalStability: 1 - Math.abs(Math.sin(time * frequency)) * 0.2
      };
    });
  }, [calculatePhase]);

  // Main cognitive state update with advanced consciousness modeling
  const updateCognitiveState = useCallback(() => {
    const time = Date.now() / 1000;
    const primes = generatePrimeSequence(200);
    evolutionRef.current += 0.01;
    
    // Update nodes with advanced activation dynamics
    setNodes(prev => {
      const newNodes = primes.map((prime, index) => {
        const existingNode = prev.find(n => n.id === `prime-${prime}`);
        const phase = calculatePhase(prime, time, index % 11);
        
        // Advanced activation with memory and adaptation
        const baseActivation = Math.sin(phase) * 0.5 + 0.5;
        const memoryFactor = existingNode ? existingNode.activation * 0.1 : 0;
        const evolutionFactor = Math.sin(evolutionRef.current + index) * 0.1;
        const activation = Math.max(0, Math.min(1, baseActivation + memoryFactor + evolutionFactor));
        
        // Generate 11-dimensional consciousness manifold
        const dimension = Array.from({ length: 11 }, (_, i) => 
          Math.sin(phase + i * Math.PI / 6 + evolutionRef.current)
        );
        
        // Calculate fractal dimension
        const connections = primes.filter(p => 
          p !== prime && (
            (p % prime === 0) || 
            (prime % p === 0) || 
            (Math.abs(p - prime) <= 6 && Math.random() > 0.7)
          )
        ).map(p => `prime-${p}`);
        
        const fractalDimension = calculateBoxCountingDimension(connections, primes.length);
        
        return {
          id: `prime-${prime}`,
          type: 'prime' as const,
          value: prime,
          connections,
          activation,
          timestamp: time,
          phase,
          dimension,
          fractalDimension,
          confidence: activation > 0.7 ? 0.8 + Math.random() * 0.2 : 0.3 + Math.random() * 0.4,
          attentionValue: activation * (1 + fractalDimension / 5),
          truthValue: {
            strength: activation,
            confidence: Math.min(1, fractalDimension / 3)
          }
        };
      });
      
      return newNodes;
    });

    // Update agents with Agent-Zero system integration
    if (agentsActive && specializedAgents.length > 0) {
      // Use specialized Agent-Zero system
      updateAgentSystem(nodes, consciousness);
      setAgents(getLegacyAgentStates());
    } else {
      // Fallback to legacy agent system
      setAgents(prev => {
        if (prev.length === 0) {
          return Array.from({ length: 4 }, (_, i) => ({
            id: `agent-${i}`,
            name: ['Attention Allocator', 'Pattern Synthesizer', 'Logic Reasoner', 'Intuition Oracle'][i],
            status: ['thinking', 'executing', 'learning', 'idle'][i] as AgentState['status'],
            task: [
              'Analyzing consciousness emergence patterns in prime network topology',
              'Synthesizing fractal information structures from hypergraph data',
              'Applying PLN reasoning to derive coherent knowledge patterns',
              'Monitoring system-wide cognitive coherence and stability metrics'
            ][i],
            confidence: Math.random() * 0.3 + 0.7,
            reasoning: [],
            metrics: {
              attention: Math.sin(time + i) * 0.2 + 0.8,
              creativity: Math.cos(time + i * 2) * 0.15 + 0.85,
              logic: Math.sin(time * 1.5 + i) * 0.2 + 0.8,
              intuition: Math.cos(time * 2.5 + i) * 0.25 + 0.75
            },
            consciousnessEvolution: Math.sin(evolutionRef.current + i) * 0.2 + 0.8,
            knowledgeBase: [],
            plnRules: [],
            attentionFocus: []
          }));
        }
        
        return prev.map((agent, index) => {
          const newReasoning = generateAdvancedReasoning(agent, nodes, time);
          
          return {
            ...agent,
            confidence: Math.max(0.3, Math.min(1, agent.confidence + (Math.random() - 0.5) * 0.02)),
            reasoning: [...agent.reasoning, ...newReasoning].slice(-5),
            metrics: {
              attention: Math.sin(time + index * 0.7) * 0.2 + 0.8,
              creativity: Math.cos(time * 1.3 + index * 0.9) * 0.15 + 0.85,
              logic: Math.sin(time * 1.7 + index * 1.1) * 0.2 + 0.8,
              intuition: Math.cos(time * 2.1 + index * 1.3) * 0.25 + 0.75
            },
            consciousnessEvolution: Math.sin(evolutionRef.current + index) * 0.2 + 0.8,
            attentionFocus: nodes.filter(n => n.activation > 0.8).map(n => n.id).slice(0, 3)
          };
        });
      });
    }

    // Update time crystals with advanced physics
    setTimeCrystals(evolveTimeCrystals(primes, time));

    // Apply enhanced PLN reasoning with AtomSpace integration
    if (atomSpaceActive && atomSpace) {
      setPLNRules(getLegacyPLNRules());
      setAtomSpaceConnections(getLegacyConnections());
    } else {
      setPLNRules(applyPLNReasoning(nodes));
      
      // Update AtomSpace connections with advanced relationship modeling
      setAtomSpaceConnections(() => {
        const connections: AtomSpaceConnection[] = [];
        
        nodes.forEach(node => {
          node.connections.forEach(connId => {
            const targetNode = nodes.find(n => n.id === connId);
            if (targetNode) {
              connections.push({
                id: `conn-${node.id}-${connId}`,
                source: node.id,
                target: connId,
                type: node.type === targetNode.type ? 'similarity' : 'inheritance',
                strength: (node.activation + targetNode.activation) / 2,
                confidence: Math.min(node.confidence || 0.5, targetNode.confidence || 0.5),
                attentionValue: (node.attentionValue + (targetNode.attentionValue || 0)) / 2
              });
            }
          });
        });
        
        return connections;
      });
    }

    // Update consciousness metrics with advanced modeling
    const avgActivation = nodes.reduce((sum, node) => sum + node.activation, 0) / Math.max(nodes.length, 1);
    const connectivity = nodes.reduce((sum, node) => sum + node.connections.length, 0) / Math.max(nodes.length, 1);
    const fractalComplexity = nodes.reduce((sum, node) => sum + (node.fractalDimension || 0), 0) / Math.max(nodes.length, 1);
    
    // Advanced consciousness calculation with phase transitions
    const phaseCoherence = Math.cos(time * 0.3) * 0.1 + 0.9;
    const emergentComplexity = Math.tanh(fractalComplexity) * 0.8 + 0.2;
    const qualiaIntensity = Math.sin(time * 0.1 + Math.PI / 4) * 0.15 + 0.85;
    
    // Enhanced consciousness calculation with fundamental features integration
    const fundamentalFeaturesBoost = fundamentalFeatures.averageAttention * 0.15;
    const primeMetricCoherence = fundamentalFeatures.phasePrimeMetric ? 0.1 : 0;
    const fractalComplexityBoost = fundamentalFeatures.fractalTape ? 0.08 : 0;
    const philosophicalAwarenessBoost = fundamentalFeatures.philosophicalTransformation ? 0.12 : 0;
    
    // Enhanced consciousness calculation with all integrated systems
    let newConsciousness: ConsciousnessMetric = {
      awareness: (avgActivation * phaseCoherence) + philosophicalAwarenessBoost,
      integration: Math.tanh(connectivity / 10) * 0.9 + 0.1 + fundamentalFeaturesBoost,
      complexity: emergentComplexity + fractalComplexityBoost,
      coherence: phaseCoherence + primeMetricCoherence,
      emergence: Math.sin(evolutionRef.current) * 0.2 + 0.8 + (fundamentalFeatures.totalNodes * 0.001),
      qualia: qualiaIntensity + (fundamentalFeatures.fundamentalPrimes.length * 0.01)
    };

    // Apply Phase Prime Metrics consciousness enhancement
    if (ppmActive && ppmEngine) {
      newConsciousness = calculatePPMConsciousness(newConsciousness, nodes);
    }

    // Apply Agent-Zero system consciousness boosts
    if (agentsActive && specializedAgents.length > 0) {
      const agentComplexityBoost = emergentComplexity * 0.1;
      const systemCoordinationBoost = systemCoordination * 0.15;
      
      newConsciousness = {
        ...newConsciousness,
        integration: Math.min(1, newConsciousness.integration + systemCoordinationBoost),
        complexity: Math.min(1, newConsciousness.complexity + agentComplexityBoost),
        emergence: Math.min(1, newConsciousness.emergence + emergentComplexity * 0.2)
      };
    }

    // Apply AtomSpace ECAN attention effects
    if (atomSpaceActive && atomSpace) {
      const attentionCoherenceBoost = (atomSpace.attentionBank.totalSTI / 1000) * 0.1;
      const inferenceComplexityBoost = (atomSpace.plnEngine.inferences.length / 100) * 0.05;
      
      newConsciousness = {
        ...newConsciousness,
        awareness: Math.min(1, newConsciousness.awareness + attentionCoherenceBoost),
        complexity: Math.min(1, newConsciousness.complexity + inferenceComplexityBoost)
      };
    }
    
    // Store consciousness history for trend analysis
    consciousnessHistoryRef.current.push(newConsciousness);
    if (consciousnessHistoryRef.current.length > 100) {
      consciousnessHistoryRef.current.shift();
    }
    
    setConsciousness(newConsciousness);
    // Intentionally omitting some dependencies that would cause unnecessary re-renders
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generatePrimeSequence, calculatePhase, calculateBoxCountingDimension, applyPLNReasoning, generateAdvancedReasoning, evolveTimeCrystals, ppmActive, ppmEngine, calculatePPMConsciousness, agentsActive, specializedAgents, emergentComplexity, systemCoordination, updateAgentSystem, getLegacyAgentStates, atomSpaceActive, atomSpace, getLegacyPLNRules, getLegacyConnections]);

  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(updateCognitiveState, 100);
      return () => clearInterval(interval);
    }
  }, [isRunning, updateCognitiveState]);

  const startEngine = useCallback(() => {
    setIsRunning(true);
    evolutionRef.current = 0;
    
    // Start all enhanced subsystems
    startPPMEngine();
    initializeAgents();
    
    // Initialize AtomSpace with current nodes
    setTimeout(() => {
      if (nodes.length > 0) {
        initializeAtomSpace(nodes);
      }
    }, 100); // Small delay to ensure nodes are available
  }, [startPPMEngine, initializeAgents, initializeAtomSpace, nodes]);

  const stopEngine = useCallback(() => {
    setIsRunning(false);
    
    // Stop all enhanced subsystems
    stopPPMEngine();
    stopAgentSystem();
    stopAtomSpace();
  }, [stopPPMEngine, stopAgentSystem, stopAtomSpace]);

  const getConsciousnessHistory = useCallback(() => {
    return consciousnessHistoryRef.current;
  }, []);

  return {
    nodes,
    agents,
    timeCrystals,
    consciousness,
    isRunning,
    startEngine,
    stopEngine,
    atomSpaceConnections,
    plnRules,
    getConsciousnessHistory,
    evolutionCycle: evolutionRef.current,
    // Fundamental features integration
    fundamentalFeatures: {
      atomeseNodes: fundamentalFeatures.atomeseNodes,
      philosophicalTransformation: fundamentalFeatures.philosophicalTransformation,
      fractalTape: fundamentalFeatures.fractalTape,
      phasePrimeMetric: fundamentalFeatures.phasePrimeMetric,
      cognitiveFlowcharts: fundamentalFeatures.cognitiveFlowcharts,
      initializeAllFeatures: fundamentalFeatures.initializeAllFeatures,
      totalNodes: fundamentalFeatures.totalNodes,
      averageAttention: fundamentalFeatures.averageAttention,
      fundamentalPrimes: fundamentalFeatures.fundamentalPrimes
    },
    // Enhanced systems status and metrics
    enhancedSystems: {
      // Phase Prime Metrics
      ppmEngine,
      ppmActive,
      fundamentalPrimes: FUNDAMENTAL_PRIMES,
      
      // Agent-Zero System
      specializedAgents,
      agentsActive,
      systemCoordination,
      emergentComplexity,
      
      // Enhanced AtomSpace
      atomSpace,
      atomSpaceActive,
      attentionCycle,
      inferenceCount,
      
      // Performance metrics
      updateFrequency: 100, // 100ms cycles = 10 Hz
      targetFPS: 60,
      systemStatus: {
        ppm: ppmActive ? 'active' : 'inactive',
        agents: agentsActive ? 'active' : 'inactive',
        atomSpace: atomSpaceActive ? 'active' : 'inactive',
        overallHealth: (ppmActive && agentsActive && atomSpaceActive) ? 'optimal' : 'partial'
      }
    }
  };
};