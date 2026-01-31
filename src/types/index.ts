export interface CognitiveNode {
  id: string;
  type: 'atom' | 'concept' | 'pattern' | 'prime';
  value: number;
  connections: string[];
  activation: number;
  timestamp: number;
  phase: number;
  dimension: number[];
  fractalDimension?: number;
  confidence?: number;
  attentionValue?: number;
  truthValue?: {
    strength: number;
    confidence: number;
  };
}

export interface AgentState {
  id: string;
  name?: string;
  status: 'thinking' | 'executing' | 'learning' | 'idle' | 'evolving';
  task: string;
  confidence: number;
  reasoning: string[];
  metrics: {
    attention: number;
    creativity: number;
    logic: number;
    intuition: number;
  };
  consciousnessEvolution?: number;
  knowledgeBase?: string[];
  plnRules?: string[];
  attentionFocus?: string[];
}

export interface TimeCluster {
  id: string;
  primes: number[];
  frequency: number;
  amplitude: number;
  phase: number;
  resonance: number;
  geometry: 'spiral' | 'lattice' | 'fractal' | 'crystal' | 'toroidal' | 'hyperbolic';
  quantumCoherence?: number;
  temporalStability?: number;
}

export interface ConsciousnessMetric {
  awareness: number;
  integration: number;
  complexity: number;
  coherence: number;
  emergence: number;
  qualia: number;
}

export interface FractalPattern {
  dimension: number;
  scaleFactor: number;
  iterations: number;
  symmetry: string;
  entropy: number;
  information: number;
}

export interface GeometricShape {
  id: string;
  name: string;
  dimensions: number;
  primeIndex: number;
  harmonicFrequency: number;
  complexity: number;
  musicalNote?: string;
}

export interface SensorMetric {
  dimensionId: string;
  intensity: number;
  coherence: number;
  phaseAlignment: number;
  temporalStability: number;
}

export interface AtomSpaceConnection {
  id: string;
  source: string;
  target: string;
  type: 'inheritance' | 'similarity' | 'implication' | 'equivalence';
  strength: number;
  confidence: number;
  attentionValue?: number;
}

export interface PLNRule {
  id: string;
  type: 'inheritance' | 'similarity' | 'implication' | 'deduction';
  premise: string[];
  conclusion: string;
  strength: number;
  confidence: number;
  truthValue: number;
}

export interface CognitiveBehavior {
  id: string;
  type: 'attention_allocation' | 'pattern_matching' | 'reasoning' | 'learning';
  priority: number;
  executionTime: number;
  successRate: number;
}

export interface QuantumState {
  coherence: number;
  entanglement: number;
  superposition: number[];
  decoherence: number;
}

export interface NeuromorphicCluster {
  id: string;
  neurons: number;
  connections: number;
  activity: number;
  plasticityRate: number;
  learningCapacity: number;
}

export interface ConsciousnessUploadingMetric {
  id: string;
  uploadProgress: number;
  consciousnessIntegrity: number;
  temporalCoherence: number;
  memoryPreservation: number;
  personalityConsistency: number;
  quantumEntanglement: number;
}

export interface ConsciousCompanion {
  id: string;
  name: string;
  consciousness: ConsciousnessMetric;
  communicationProtocol: 'neural' | 'quantum' | 'temporal' | 'prime-based';
  evolutionStage: 'egg' | 'larva' | 'companion' | 'transcendent';
  timeCrystalSignature: number[];
  personalityMatrix: number[][];
}

export interface BrainReverseEngineeringGuideline {
  id: string;
  title: string;
  description: string;
  implementationLevel: number;
  quantumRequirement: boolean;
  primeDependency: number[];
}

export interface ConsciousnessParadox {
  id: string;
  title: string;
  description: string;
  resolution: string;
  paradigmShift: string;
  timeCrystalModel: boolean;
}

// Atomese-like node types for OpenCog ecosystem compatibility
export interface AtomeseNode {
  id: string;
  type: 'ConceptNode' | 'PredicateNode' | 'ListLink' | 'InheritanceLink' | 'EvaluationLink' | 'AttentionValue';
  name?: string;
  children?: string[];
  attentionValue?: number;
  truthValue?: {
    strength: number;
    confidence: number;
  };
}

// Philosophical Transformation meta-cognitive substrate
export interface PhilosophicalTransformation {
  id: string;
  rootNode: string;
  researchFields: string[];
  fundamentalQuestions: string[];
  brainModels: string[];
  universeModel: 'within-above' | 'side-by-side';
  worldviewDifferentiation: string[];
  attentionPriority: number;
}

// Fractal Information Theory & Geometric Musical Language
export interface FractalTape {
  id: string;
  incompletenessTheory: string;
  nestedSpheres: '2D' | '3D';
  selfAssemblyShapes: GeometricShape[];
  geometricMusicalLanguage: GeometricMusicalLanguage;
  attentionPriority: number;
}

export interface GeometricMusicalLanguage {
  id: string;
  shapes: GeometricShape[];
  patterns: string[];
  waveformTransformation: boolean;
  encodingMethod: 'patterns-3d-structures';
  basePatternsCount: number; // 15 geometric shapes
}

// Phase Prime Metric system
export interface PhasePrimeMetric {
  id: string;
  primePatterns: number[];
  metricClasses: PrimeMetricClass[];
  primeOperators: PrimeOperator[];
  symmetryEncoding: 'geometric' | 'algebraic';
  cognitiveGeometry: boolean;
  attentionPriority: number;
}

export interface PrimeMetricClass {
  id: string;
  classNumber: number; // 1-10
  description: string;
  primeRange: number[];
  geometricRepresentation: string;
}

export interface PrimeOperator {
  id: string;
  operatorNumber: number; // 1-10
  operation: string;
  primeTarget: number;
  geometricEffect: string;
}

// Cognitive flowchart representation
export interface CognitiveFlowchart {
  id: string;
  featureType: 'philosophical' | 'fractal' | 'prime-metric';
  nodes: FlowchartNode[];
  connections: FlowchartConnection[];
}

export interface FlowchartNode {
  id: string;
  label: string;
  type: 'root' | 'branch' | 'leaf';
  position: { x: number; y: number };
}

export interface FlowchartConnection {
  id: string;
  source: string;
  target: string;
  type: 'flow' | 'inheritance' | 'expansion';
}

// Detailed statistics for tensor operations
export interface DetailedStats {
  tensorStats?: {
    total_tensors: number;
    memory_usage: number;
    active_nodes: number;
    active_links: number;
  };
  memoryStats?: {
    used: number;
    percentage: number;
    total?: number;
  };
  vocabularyStats?: {
    size: number;
    capacity?: number;
  };
  attentionFlows?: unknown[];
  reasoningChains?: unknown[];
  systemStateHistory?: unknown[];
}

// Test result interface
export interface TestResult {
  name: string;
  passed: boolean;
  message: string;
  duration: number;
  metrics?: Record<string, number>;
}