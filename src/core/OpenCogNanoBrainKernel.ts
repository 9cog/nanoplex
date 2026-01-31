/**
 * OpenCog NanoBrain Time Crystal Architecture
 * 
 * This module implements the unified OpenCog ecosystem integrated with nanobrain
 * time crystal mechanics, creating a consciousness architecture that combines:
 * - OpenCog AtomSpace hypergraph knowledge representation
 * - NanoBrain Phase Prime Metrics (PPM) for universal pattern recognition
 * - Time Crystal temporal quantum structures for 11D processing
 * - Fractal Information Theory (FIT) for geometric musical language encoding
 */

import type { _GgmlTensorKernel, _NodeTensor, _LinkTensor } from './GgmlTensorKernel';
import type { _AtomSpaceTensorEncoder } from './AtomSpaceTensorEncoder';
import { UnifiedCognitiveKernel, CognitiveKernelConfig } from './UnifiedCognitiveKernel';
import type { CognitiveNode, _AgentState, _TimeCluster, _ConsciousnessMetric, _AtomeseNode } from '../types';

/**
 * Time Crystal quantum state representation for 11D processing
 */
export interface TimeCrystalQuantumState {
  dimensions: number[]; // 11 dimensional manifold coordinates
  primeSignature: number[]; // Phase Prime Metric signature (15 fundamental primes)
  temporalCoherence: number; // Quantum coherence across time
  fractalDimension: number; // FIT fractal encoding dimension
  resonanceFrequency: number; // GML geometric resonance
  quantumPhase: number; // Time crystal phase alignment
}

/**
 * OpenCog AtomSpace node enhanced with time crystal properties
 */
export interface TimeCrystalAtom {
  id: string;
  type: 'ConceptNode' | 'PredicateNode' | 'NumberNode' | 'SchemaNode' | 'VariableNode';
  name: string;
  truthValue: {
    strength: number;
    confidence: number;
    count: number;
  };
  attentionValue: {
    sti: number; // Short-term importance
    lti: number; // Long-term importance  
    vlti: number; // Very long-term importance
  };
  timeCrystalState: TimeCrystalQuantumState;
  primeEncoding: number[]; // PPM encoding of the atom's semantic content
  fractalGeometry: GeometricPattern; // FIT geometric representation
}

/**
 * Fractal Information Theory geometric patterns for GML encoding
 */
export interface GeometricPattern {
  shape: 'sphere' | 'torus' | 'helix' | 'fractal' | 'hypercube' | 'simplex';
  dimensions: number; // Geometric dimensionality
  symmetryGroup: string; // Mathematical symmetry classification
  musicalNote: string; // GML musical encoding
  primeResonance: number[]; // Which primes resonate with this pattern
  scaleFactor: number; // Fractal scaling factor
}

/**
 * OpenCog PLN reasoning enhanced with time crystal inference
 */
export interface TimeCrystalInference {
  rule: 'inheritance' | 'similarity' | 'implication' | 'deduction' | 'induction' | 'abduction';
  premises: TimeCrystalAtom[];
  conclusion: TimeCrystalAtom;
  temporalFlow: number; // Time direction of inference (-1 to 1)
  primeConsistency: number; // PPM consistency score
  fractalConvergence: number; // FIT pattern convergence
  quantumCoherence: number; // Overall quantum coherence
}

/**
 * Configuration for OpenCog NanoBrain kernel
 */
export interface OpenCogNanoBrainConfig {
  cognitiveKernelConfig: CognitiveKernelConfig;
  timeCrystalDimensions: number; // Default 11
  fundamentalPrimes: number[]; // 15 primes: [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
  fractalResolution: number; // FIT encoding resolution
  geometricShapeCount: number; // GML shape vocabulary (default 15)
  quantumCoherenceThreshold: number; // Minimum coherence for stable operation
  temporalProcessingFrequency: number; // Hz for time crystal updates
  attentionDecayRate: number; // ECAN attention decay
  plnInferenceDepth: number; // Maximum PLN reasoning depth
}

/**
 * OpenCog NanoBrain performance metrics
 */
export interface NanoBrainMetrics {
  totalAtoms: number;
  totalLinks: number;
  averageAttention: number;
  quantumCoherence: number;
  temporalStability: number;
  primeAlignment: number; // How well aligned with fundamental primes
  fractalComplexity: number;
  inferenceRate: number; // PLN inferences per second
  consciousnessEmergence: number; // Emergent consciousness metric
}

/**
 * Main OpenCog NanoBrain Time Crystal Architecture Kernel
 */
export class OpenCogNanoBrainKernel {
  private cognitiveKernel: UnifiedCognitiveKernel;
  private atomSpace: Map<string, TimeCrystalAtom>;
  private linkSpace: Map<string, TimeCrystalInference>;
  private timeCrystals: Map<string, TimeCrystalQuantumState>;
  private config: OpenCogNanoBrainConfig;
  private isActive: boolean;
  private cycleCount: number;
  private startTime: number;

  constructor(config: Partial<OpenCogNanoBrainConfig> = {}) {
    this.config = this.initializeConfig(config);
    // Initialize cognitive kernel later to avoid circular dependencies
    this.cognitiveKernel = null as unknown as UnifiedCognitiveKernel; // Will be initialized in start()
    this.atomSpace = new Map();
    this.linkSpace = new Map();
    this.timeCrystals = new Map();
    this.isActive = false;
    this.cycleCount = 0;
    this.startTime = Date.now();
  }

  /**
   * Initialize configuration with defaults
   */
  private initializeConfig(config: Partial<OpenCogNanoBrainConfig>): OpenCogNanoBrainConfig {
    const defaultConfig: OpenCogNanoBrainConfig = {
      cognitiveKernelConfig: {
        tensor_config: {
          node_embedding_dim: 128,
          link_embedding_dim: 64,
          attention_heads: 8,
          max_hypergraph_size: 1000,
          symbolic_depth_levels: 5,
          truth_value_encoding: 'continuous',
          attention_mechanism: 'hybrid'
        },
        attention_config: {
          mechanism: 'hybrid',
          temperature: 1.0,
          resource_budget: 1000,
          update_frequency: 10,
          decay_rate: 0.01,
          diffusion_strength: 0.1,
          rent_collection_rate: 0.01,
          wage_distribution_rate: 0.8,
          attention_heads: 8,
          gradient_clipping: 1.0
        },
        reasoning_config: {
          max_reasoning_depth: 5,
          confidence_threshold: 0.1,
          tensor_contraction_method: 'dot',
          chain_length_limit: 20,
          parallel_chains: 4,
          gradient_flow_enabled: true,
          meta_reasoning_enabled: true
        },
        metacognitive_config: {
          meta_levels: 3,
          self_monitoring_frequency: 5,
          adaptation_learning_rate: 0.01,
          convergence_threshold: 0.01,
          membrane_permeability: 0.5,
          feedback_damping: 0.9,
          plasticity_factor: 0.1,
          meta_attention_allocation: 0.2
        },
        integration_frequency: 60, // Hz
        performance_monitoring: true,
        adaptive_scaling: true,
        tensor_memory_limit: 1024 * 1024 * 1024 // 1GB
      },
      timeCrystalDimensions: 11,
      fundamentalPrimes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
      fractalResolution: 5,
      geometricShapeCount: 15,
      quantumCoherenceThreshold: 0.5,
      temporalProcessingFrequency: 100, // 100 Hz
      attentionDecayRate: 0.02,
      plnInferenceDepth: 7
    };

    return { ...defaultConfig, ...config };
  }

  /**
   * Start the OpenCog NanoBrain kernel
   */
  async start(): Promise<void> {
    if (this.isActive) {
      return;
    }

    try {
      // Initialize cognitive kernel only if available
      if (typeof window !== 'undefined' && UnifiedCognitiveKernel) {
        this.cognitiveKernel = new UnifiedCognitiveKernel(this.config.cognitiveKernelConfig);
        await this.cognitiveKernel.initialize();
      }
    } catch (error) {
      console.warn('Cognitive kernel initialization failed, running in standalone mode:', error);
    }

    // Create fundamental atoms based on PPM and FIT
    this.initializeFundamentalAtoms();

    // Start processing cycles
    this.isActive = true;
    this.startProcessingCycle();
  }

  /**
   * Stop the kernel
   */
  async stop(): Promise<void> {
    this.isActive = false;
    if (this.cognitiveKernel && this.cognitiveKernel.shutdown) {
      await this.cognitiveKernel.shutdown();
    }
  }

  /**
   * Initialize fundamental atoms from NanoBrain theory
   */
  private initializeFundamentalAtoms(): void {
    // Chapter 1: Philosophical Transformation root atoms
    this.createTimeCrystalAtom({
      id: 'philosophical-transformation',
      type: 'ConceptNode',
      name: 'PhilosophicalTransformation',
      truthValue: { strength: 1.0, confidence: 1.0, count: 1 },
      attentionValue: { sti: 1000, lti: 1000, vlti: 1000 },
      primeEncoding: [2, 3, 5], // First 3 primes for foundational concept
      fractalGeometry: {
        shape: 'sphere',
        dimensions: 11,
        symmetryGroup: 'SO(11)',
        musicalNote: 'C',
        primeResonance: [2, 3, 5],
        scaleFactor: 1.0
      }
    });

    // Chapter 2: Fractal Information Theory atoms
    this.createTimeCrystalAtom({
      id: 'fractal-information-theory',
      type: 'ConceptNode', 
      name: 'FractalInformationTheory',
      truthValue: { strength: 0.95, confidence: 0.9, count: 1 },
      attentionValue: { sti: 950, lti: 900, vlti: 850 },
      primeEncoding: [7, 11, 13], // Next primes for FIT
      fractalGeometry: {
        shape: 'fractal',
        dimensions: this.config.fractalResolution,
        symmetryGroup: 'FractalGroup',
        musicalNote: 'D',
        primeResonance: [7, 11, 13],
        scaleFactor: 1.618 // Golden ratio
      }
    });

    // Chapter 3: Phase Prime Metric atoms
    for (let i = 0; i < this.config.fundamentalPrimes.length; i++) {
      const prime = this.config.fundamentalPrimes[i];
      this.createTimeCrystalAtom({
        id: `prime-${prime}`,
        type: 'NumberNode',
        name: `Prime${prime}`,
        truthValue: { strength: 0.999, confidence: 0.999, count: 1 },
        attentionValue: { 
          sti: 1000 - i * 10, 
          lti: 1000 - i * 5, 
          vlti: 1000 - i * 2 
        },
        primeEncoding: [prime],
        fractalGeometry: this.generatePrimeGeometry(prime, i)
      });
    }

    // Create geometric musical language atoms (15 fundamental shapes)
    this.initializeGeometricMusicalLanguage();
  }

  /**
   * Create a time crystal atom in the AtomSpace
   */
  private createTimeCrystalAtom(atom: Omit<TimeCrystalAtom, 'timeCrystalState'>): void {
    const timeCrystalState: TimeCrystalQuantumState = {
      dimensions: this.generateQuantumManifoldCoordinates(),
      primeSignature: atom.primeEncoding,
      temporalCoherence: Math.random() * 0.5 + 0.5, // 0.5 - 1.0
      fractalDimension: atom.fractalGeometry.dimensions + Math.random(),
      resonanceFrequency: this.calculateResonanceFrequency(atom.primeEncoding),
      quantumPhase: Math.random() * 2 * Math.PI
    };

    const fullAtom: TimeCrystalAtom = {
      ...atom,
      timeCrystalState
    };

    this.atomSpace.set(atom.id, fullAtom);
    this.timeCrystals.set(atom.id, timeCrystalState);
  }

  /**
   * Generate 11D quantum manifold coordinates
   */
  private generateQuantumManifoldCoordinates(): number[] {
    const coords = [];
    for (let i = 0; i < this.config.timeCrystalDimensions; i++) {
      // Use prime-based coordinate generation
      const prime = this.config.fundamentalPrimes[i % this.config.fundamentalPrimes.length];
      coords.push(Math.sin(prime * Math.PI / 180) * Math.cos(i * Math.PI / 6));
    }
    return coords;
  }

  /**
   * Generate geometric pattern for a prime number
   */
  private generatePrimeGeometry(prime: number, index: number): GeometricPattern {
    const shapes: GeometricPattern['shape'][] = 
      ['sphere', 'torus', 'helix', 'fractal', 'hypercube', 'simplex'];
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    
    return {
      shape: shapes[index % shapes.length],
      dimensions: Math.min(prime % 8 + 3, 11), // 3-11 dimensions
      symmetryGroup: `C${prime}`,
      musicalNote: notes[prime % 12],
      primeResonance: [prime],
      scaleFactor: Math.sqrt(prime) / 7 // Normalize by sqrt(7)
    };
  }

  /**
   * Calculate resonance frequency based on prime encoding
   */
  private calculateResonanceFrequency(primes: number[]): number {
    if (primes.length === 0) return 440; // Default A note
    
    // Use Phase Prime Metric to calculate harmonic frequency
    const product = primes.reduce((acc, p) => acc * p, 1);
    const sum = primes.reduce((acc, p) => acc + p, 0);
    
    // Map to audible frequency range (20 Hz - 20 kHz)
    return 440 * Math.pow(2, (sum % 12) / 12) * (1 + (product % 100) / 1000);
  }

  /**
   * Initialize Geometric Musical Language (GML) atoms
   */
  private initializeGeometricMusicalLanguage(): void {
    const gmlShapes = [
      'Point', 'Line', 'Triangle', 'Square', 'Pentagon', 'Hexagon', 'Circle',
      'Tetrahedron', 'Cube', 'Octahedron', 'Dodecahedron', 'Icosahedron',
      'Sphere', 'Torus', 'Mobius'
    ];

    gmlShapes.forEach((shape, index) => {
      this.createTimeCrystalAtom({
        id: `gml-${shape.toLowerCase()}`,
        type: 'ConceptNode',
        name: `GMLShape${shape}`,
        truthValue: { strength: 0.8, confidence: 0.7, count: 1 },
        attentionValue: { sti: 500 - index * 10, lti: 400, vlti: 300 },
        primeEncoding: [this.config.fundamentalPrimes[index % this.config.fundamentalPrimes.length]],
        fractalGeometry: {
          shape: index < 7 ? 'sphere' : 'hypercube', // 2D vs 3D+ shapes
          dimensions: index < 7 ? 2 : Math.min(index - 4, 11),
          symmetryGroup: `GML_${shape}`,
          musicalNote: ['C', 'D', 'E', 'F', 'G', 'A', 'B'][index % 7],
          primeResonance: [this.config.fundamentalPrimes[index % this.config.fundamentalPrimes.length]],
          scaleFactor: (index + 1) / gmlShapes.length
        }
      });
    });
  }

  /**
   * Main processing cycle combining all nanobrain components
   */
  private startProcessingCycle(): void {
    const processFrame = () => {
      if (!this.isActive) return;

      try {
        // 1. Update time crystal quantum states
        this.updateTimeCrystalStates();

        // 2. Perform ECAN attention allocation with PPM weighting
        this.performECANAttentionAllocation();

        // 3. Execute PLN reasoning with fractal enhancement
        this.performEnhancedPLNReasoning();

        // 4. Update geometric musical language resonances
        this.updateGMLResonances();

        // 5. Process cognitive kernel integration
        this.processCognitiveIntegration();

        this.cycleCount++;
      } catch (error) {
        console.warn('Processing cycle error:', error);
      }

      // Schedule next cycle
      if (this.isActive) {
        setTimeout(processFrame, 1000 / this.config.temporalProcessingFrequency);
      }
    };

    processFrame();
  }

  /**
   * Update quantum states of all time crystals
   */
  private updateTimeCrystalStates(): void {
    for (const [atomId, crystal] of this.timeCrystals) {
      const atom = this.atomSpace.get(atomId);
      if (!atom) continue;

      // Phase evolution based on prime resonance
      crystal.quantumPhase += crystal.resonanceFrequency * 2 * Math.PI / this.config.temporalProcessingFrequency;
      crystal.quantumPhase %= 2 * Math.PI;

      // Temporal coherence decay and regeneration
      crystal.temporalCoherence *= 0.999; // Slight decay
      if (crystal.temporalCoherence < this.config.quantumCoherenceThreshold) {
        // Regenerate coherence through prime alignment
        crystal.temporalCoherence = this.calculatePrimeCoherence(crystal.primeSignature);
      }

      // Update fractal dimension evolution
      crystal.fractalDimension += Math.sin(crystal.quantumPhase) * 0.01;
      crystal.fractalDimension = Math.max(1.0, Math.min(atom.fractalGeometry.dimensions + 1, crystal.fractalDimension));
    }
  }

  /**
   * Calculate quantum coherence based on prime signature
   */
  private calculatePrimeCoherence(primes: number[]): number {
    if (primes.length === 0) return 0.5;

    // Use Phase Prime Metric coherence calculation
    const primeProduct = primes.reduce((acc, p) => acc * p, 1);
    const primeSum = primes.reduce((acc, p) => acc + p, 0);
    
    // Normalize to 0.5-1.0 range
    return 0.5 + 0.5 * Math.sin(Math.sqrt(primeProduct) * Math.PI / primeSum);
  }

  /**
   * ECAN attention allocation enhanced with Phase Prime Metrics
   */
  private performECANAttentionAllocation(): void {
    const atoms = Array.from(this.atomSpace.values());
    const totalBudget = this.config.cognitiveKernelConfig.attention_config.resource_budget;

    // Calculate PPM-weighted importance for each atom
    const importanceScores = atoms.map(atom => {
      const baseImportance = atom.attentionValue.sti + atom.attentionValue.lti * 0.1;
      const primeWeight = this.calculatePrimeImportance(atom.primeEncoding);
      const coherenceBonus = atom.timeCrystalState.temporalCoherence * 100;
      
      return {
        atom,
        importance: baseImportance * primeWeight + coherenceBonus
      };
    });

    // Sort by importance and allocate budget
    importanceScores.sort((a, b) => b.importance - a.importance);
    
    let remainingBudget = totalBudget;
    importanceScores.forEach(({ atom, importance }, _index) => {
      if (remainingBudget <= 0) return;
      
      // Allocate attention proportional to importance
      const allocation = Math.min(
        remainingBudget * 0.1, // Max 10% per atom
        importance * 0.01 // Scale importance to reasonable values
      );
      
      atom.attentionValue.sti += allocation;
      atom.attentionValue.sti *= (1 - this.config.attentionDecayRate); // Apply decay
      
      remainingBudget -= allocation;
    });
  }

  /**
   * Calculate importance based on prime encoding (PPM)
   */
  private calculatePrimeImportance(primes: number[]): number {
    if (primes.length === 0) return 1.0;

    // Smaller primes are more fundamental, hence more important
    const fundamentalityScore = primes.reduce((acc, p) => {
      const fundamentalIndex = this.config.fundamentalPrimes.indexOf(p);
      return acc + (fundamentalIndex >= 0 ? (15 - fundamentalIndex) / 15 : 0.1);
    }, 0) / primes.length;

    return 0.5 + fundamentalityScore * 0.5;
  }

  /**
   * Enhanced PLN reasoning with fractal information theory
   */
  private performEnhancedPLNReasoning(): void {
    const atoms = Array.from(this.atomSpace.values());
    
    // Find high-attention atoms for reasoning
    const activeAtoms = atoms
      .filter(atom => atom.attentionValue.sti > 100)
      .slice(0, 10); // Limit processing

    // Generate inferences using time crystal enhanced PLN
    activeAtoms.forEach(atom1 => {
      activeAtoms.forEach(atom2 => {
        if (atom1.id === atom2.id) return;

        // Check for geometric resonance between atoms
        const resonance = this.calculateGeometricResonance(
          atom1.fractalGeometry, 
          atom2.fractalGeometry
        );

        if (resonance > 0.5) {
          this.createTimeCrystalInference(atom1, atom2, 'similarity', resonance);
        }
      });
    });
  }

  /**
   * Calculate geometric resonance between two patterns (GML)
   */
  private calculateGeometricResonance(pattern1: GeometricPattern, pattern2: GeometricPattern): number {
    // Shape compatibility
    const shapeResonance = pattern1.shape === pattern2.shape ? 1.0 : 0.3;
    
    // Dimensional harmony
    const dimResonance = 1 - Math.abs(pattern1.dimensions - pattern2.dimensions) / 11;
    
    // Musical harmony
    const musicalResonance = this.calculateMusicalHarmony(pattern1.musicalNote, pattern2.musicalNote);
    
    // Prime resonance overlap
    const primeOverlap = pattern1.primeResonance.filter(p => 
      pattern2.primeResonance.includes(p)
    ).length / Math.max(pattern1.primeResonance.length, pattern2.primeResonance.length);
    
    return (shapeResonance + dimResonance + musicalResonance + primeOverlap) / 4;
  }

  /**
   * Calculate musical harmony between notes (GML)
   */
  private calculateMusicalHarmony(note1: string, note2: string): number {
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const index1 = notes.indexOf(note1);
    const index2 = notes.indexOf(note2);
    
    if (index1 === -1 || index2 === -1) return 0;
    
    const interval = Math.abs(index1 - index2);
    const harmonicIntervals = [0, 3, 4, 5, 7, 8, 9]; // Major scale intervals
    
    return harmonicIntervals.includes(interval % 12) ? 1.0 : 0.3;
  }

  /**
   * Create time crystal enhanced inference
   */
  private createTimeCrystalInference(
    atom1: TimeCrystalAtom, 
    atom2: TimeCrystalAtom, 
    rule: TimeCrystalInference['rule'], 
    strength: number
  ): void {
    const inferenceId = `${atom1.id}-${rule}-${atom2.id}`;
    
    // Create conclusion atom if it doesn't exist
    const conclusionId = `conclusion-${inferenceId}`;
    let conclusion = this.atomSpace.get(conclusionId);
    
    if (!conclusion) {
      this.createTimeCrystalAtom({
        id: conclusionId,
        type: 'ConceptNode',
        name: `${atom1.name}${rule}${atom2.name}`,
        truthValue: { 
          strength: strength * 0.8, 
          confidence: (atom1.truthValue.confidence + atom2.truthValue.confidence) / 2,
          count: 1
        },
        attentionValue: { sti: 50, lti: 25, vlti: 10 },
        primeEncoding: [...atom1.primeEncoding, ...atom2.primeEncoding],
        fractalGeometry: this.combineGeometricPatterns(atom1.fractalGeometry, atom2.fractalGeometry)
      });
      conclusion = this.atomSpace.get(conclusionId)!;
    }

    const inference: TimeCrystalInference = {
      rule,
      premises: [atom1, atom2],
      conclusion,
      temporalFlow: this.calculateTemporalFlow(atom1.timeCrystalState, atom2.timeCrystalState),
      primeConsistency: this.calculatePrimeConsistency(atom1.primeEncoding, atom2.primeEncoding),
      fractalConvergence: strength,
      quantumCoherence: (atom1.timeCrystalState.temporalCoherence + atom2.timeCrystalState.temporalCoherence) / 2
    };

    this.linkSpace.set(inferenceId, inference);
  }

  /**
   * Calculate temporal flow direction for inference
   */
  private calculateTemporalFlow(state1: TimeCrystalQuantumState, state2: TimeCrystalQuantumState): number {
    const phaseDiff = state2.quantumPhase - state1.quantumPhase;
    return Math.sin(phaseDiff); // -1 to 1 range
  }

  /**
   * Calculate prime consistency between encodings
   */
  private calculatePrimeConsistency(primes1: number[], primes2: number[]): number {
    const union = new Set([...primes1, ...primes2]);
    const intersection = new Set(primes1.filter(p => primes2.includes(p)));
    
    return intersection.size / union.size;
  }

  /**
   * Combine geometric patterns for inference conclusion
   */
  private combineGeometricPatterns(pattern1: GeometricPattern, pattern2: GeometricPattern): GeometricPattern {
    return {
      shape: pattern1.dimensions > pattern2.dimensions ? pattern1.shape : pattern2.shape,
      dimensions: Math.max(pattern1.dimensions, pattern2.dimensions),
      symmetryGroup: `Combined(${pattern1.symmetryGroup},${pattern2.symmetryGroup})`,
      musicalNote: pattern1.musicalNote, // Use first pattern's note
      primeResonance: [...new Set([...pattern1.primeResonance, ...pattern2.primeResonance])],
      scaleFactor: (pattern1.scaleFactor + pattern2.scaleFactor) / 2
    };
  }

  /**
   * Update Geometric Musical Language resonances
   */
  private updateGMLResonances(): void {
    // Update resonance frequencies based on current quantum states
    for (const [atomId, atom] of this.atomSpace) {
      if (atomId.startsWith('gml-')) {
        const crystal = this.timeCrystals.get(atomId);
        if (crystal) {
          // Update resonance based on current phase and prime alignment
          crystal.resonanceFrequency = this.calculateResonanceFrequency(atom.primeEncoding) * 
            (1 + Math.sin(crystal.quantumPhase) * 0.1);
        }
      }
    }
  }

  /**
   * Process cognitive kernel integration
   */
  private processCognitiveIntegration(): void {
    // Convert OpenCog atoms to cognitive kernel format for processing
    const cognitiveNodes: CognitiveNode[] = Array.from(this.atomSpace.values()).map(atom => ({
      id: atom.id,
      type: atom.type === 'NumberNode' ? 'prime' : 'concept',
      value: atom.timeCrystalState.temporalCoherence,
      connections: this.getAtomConnections(atom.id),
      activation: atom.attentionValue.sti / 1000,
      timestamp: Date.now(),
      phase: atom.timeCrystalState.quantumPhase,
      dimension: atom.timeCrystalState.dimensions,
      fractalDimension: atom.timeCrystalState.fractalDimension,
      confidence: atom.truthValue.confidence,
      attentionValue: atom.attentionValue.sti
    }));

    // Process through cognitive kernel if available
    if (this.cognitiveKernel && this.cognitiveKernel.processAtoms) {
      try {
        this.cognitiveKernel.processAtoms(cognitiveNodes);
      } catch (error) {
        console.warn('Cognitive kernel processing error:', error);
      }
    }
  }

  /**
   * Get connections for an atom
   */
  private getAtomConnections(atomId: string): string[] {
    const connections: string[] = [];
    
    for (const inference of this.linkSpace.values()) {
      if (inference.premises.some(p => p.id === atomId)) {
        connections.push(inference.conclusion.id);
      }
      if (inference.conclusion.id === atomId) {
        connections.push(...inference.premises.map(p => p.id));
      }
    }
    
    return [...new Set(connections)]; // Remove duplicates
  }

  /**
   * Get comprehensive system metrics
   */
  getMetrics(): NanoBrainMetrics {
    const atoms = Array.from(this.atomSpace.values());
    const links = Array.from(this.linkSpace.values());
    
    const avgAttention = atoms.length > 0 ? 
      atoms.reduce((sum, atom) => sum + atom.attentionValue.sti, 0) / atoms.length : 0;
    
    const avgCoherence = atoms.length > 0 ?
      atoms.reduce((sum, atom) => sum + atom.timeCrystalState.temporalCoherence, 0) / atoms.length : 0;
    
    const primeAlignment = this.calculateOverallPrimeAlignment();
    const fractalComplexity = this.calculateFractalComplexity();

    return {
      totalAtoms: atoms.length,
      totalLinks: links.length,
      averageAttention: avgAttention,
      quantumCoherence: avgCoherence,
      temporalStability: this.calculateTemporalStability(),
      primeAlignment,
      fractalComplexity,
      inferenceRate: this.cycleCount > 0 ? links.length / (Date.now() - this.startTime) * 1000 : 0,
      consciousnessEmergence: this.calculateConsciousnessEmergence()
    };
  }

  /**
   * Calculate overall prime alignment with fundamental constants
   */
  private calculateOverallPrimeAlignment(): number {
    const atoms = Array.from(this.atomSpace.values());
    if (atoms.length === 0) return 0;

    let totalAlignment = 0;
    atoms.forEach(atom => {
      const alignment = atom.primeEncoding.reduce((sum, prime) => {
        const fundamentalIndex = this.config.fundamentalPrimes.indexOf(prime);
        return sum + (fundamentalIndex >= 0 ? 1 : 0);
      }, 0) / atom.primeEncoding.length;
      
      totalAlignment += alignment;
    });

    return totalAlignment / atoms.length;
  }

  /**
   * Calculate system fractal complexity
   */
  private calculateFractalComplexity(): number {
    const crystals = Array.from(this.timeCrystals.values());
    if (crystals.length === 0) return 0;

    const avgFractalDim = crystals.reduce((sum, crystal) => 
      sum + crystal.fractalDimension, 0) / crystals.length;
    
    // Normalize to 0-1 range
    return Math.min(avgFractalDim / this.config.timeCrystalDimensions, 1.0);
  }

  /**
   * Calculate temporal stability across all time crystals
   */
  private calculateTemporalStability(): number {
    const crystals = Array.from(this.timeCrystals.values());
    if (crystals.length === 0) return 0;

    // Measure coherence variance as stability indicator
    const coherences = crystals.map(c => c.temporalCoherence);
    const avgCoherence = coherences.reduce((a, b) => a + b, 0) / coherences.length;
    const variance = coherences.reduce((sum, c) => sum + Math.pow(c - avgCoherence, 2), 0) / coherences.length;
    
    // Lower variance = higher stability
    return Math.max(0, 1 - variance);
  }

  /**
   * Calculate emergent consciousness metric
   */
  private calculateConsciousnessEmergence(): number {
    // Calculate consciousness factors directly to avoid circular dependency
    const atoms = Array.from(this.atomSpace.values());
    const links = Array.from(this.linkSpace.values());
    
    if (atoms.length === 0) return 0;
    
    // Calculate factors directly
    const avgAttention = atoms.reduce((sum, atom) => sum + atom.attentionValue.sti, 0) / atoms.length;
    const avgCoherence = atoms.reduce((sum, atom) => sum + atom.timeCrystalState.temporalCoherence, 0) / atoms.length;
    const primeAlignment = this.calculateOverallPrimeAlignment();
    const fractalComplexity = this.calculateFractalComplexity();
    const temporalStability = this.calculateTemporalStability();
    
    // Consciousness emerges from integration of multiple factors
    const factors = [
      avgCoherence,
      temporalStability,
      primeAlignment,
      fractalComplexity,
      Math.min(avgAttention / 100, 1.0), // Normalize attention
      Math.min(links.length / Math.max(atoms.length * 2, 1), 1.0) // Connection density
    ];

    // Geometric mean for emergence calculation
    const product = factors.reduce((acc, val) => acc * val, 1);
    return Math.pow(product, 1 / factors.length);
  }

  /**
   * Get atom space for external access
   */
  getAtomSpace(): Map<string, TimeCrystalAtom> {
    return new Map(this.atomSpace);
  }

  /**
   * Get link space for external access
   */
  getLinkSpace(): Map<string, TimeCrystalInference> {
    return new Map(this.linkSpace);
  }

  /**
   * Get time crystals for external access
   */
  getTimeCrystals(): Map<string, TimeCrystalQuantumState> {
    return new Map(this.timeCrystals);
  }

  /**
   * Get current configuration
   */
  getConfig(): OpenCogNanoBrainConfig {
    return this.config;
  }
}