import { useState, useCallback, useEffect, useMemo } from 'react';
import { 
  AtomeseNode, 
  PhilosophicalTransformation, 
  FractalTape, 
  PhasePrimeMetric, 
  GeometricShape,
  CognitiveFlowchart,
  PrimeMetricClass,
  PrimeOperator,
  GeometricMusicalLanguage
} from '../types';
import {
  generateCompleteAtomeseKnowledgeBase,
  generateSchemeOutput,
  AtomeseExpression
} from '../core/AtomeseRecursiveInstantiation';

export const useFundamentalFeatures = () => {
  const [atomeseNodes, setAtomeseNodes] = useState<AtomeseNode[]>([]);
  const [philosophicalTransformation, setPhilosophicalTransformation] = useState<PhilosophicalTransformation | null>(null);
  const [fractalTape, setFractalTape] = useState<FractalTape | null>(null);
  const [phasePrimeMetric, setPhasePrimeMetric] = useState<PhasePrimeMetric | null>(null);
  const [cognitiveFlowcharts, setCognitiveFlowcharts] = useState<CognitiveFlowchart[]>([]);
  const [recursiveAtomeseKnowledgeBase, setRecursiveAtomeseKnowledgeBase] = useState<AtomeseExpression[]>([]);
  const [schemeCode, setSchemeCode] = useState<string>('');

  // Initialize the 15 fundamental primes for 99.99% universe coverage
  const fundamentalPrimes = useMemo(() => [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], []);

  // Convert AtomeseExpression to AtomeseNode format for compatibility
  const convertAtomeseExpressionToNode = useCallback((expr: AtomeseExpression, id: string): AtomeseNode => {
    return {
      id,
      type: expr.type,
      name: expr.name,
      children: expr.children?.map((_, index) => `${id}-child-${index}`) || [],
      attentionValue: expr.attentionValue,
      truthValue: expr.truthValue
    };
  }, []);

  // Generate complete recursive Atomese knowledge base
  const initializeRecursiveAtomeseKnowledgeBase = useCallback(() => {
    const knowledgeBase = generateCompleteAtomeseKnowledgeBase();
    const schemeOutput = generateSchemeOutput();
    
    setRecursiveAtomeseKnowledgeBase(knowledgeBase);
    setSchemeCode(schemeOutput);
    
    // Convert to AtomeseNode format and add to main nodes collection
    const convertedNodes = knowledgeBase.map((expr, index) => 
      convertAtomeseExpressionToNode(expr, `atomese-kb-${index}`)
    );
    
    setAtomeseNodes(prev => [...prev, ...convertedNodes]);
    
    return { knowledgeBase, schemeOutput };
  }, [convertAtomeseExpressionToNode]);

  // 1. PHILOSOPHICAL TRANSFORMATION - Meta-Cognitive Substrate
  const initializePhilosophicalTransformation = useCallback(() => {
    // Create Atomese nodes for philosophical transformation
    const nodes: AtomeseNode[] = [
      // Root concept node
      {
        id: 'philosophical-transformation-root',
        type: 'ConceptNode',
        name: 'Philosophical-Transformation',
        attentionValue: 1.0,
        truthValue: { strength: 0.95, confidence: 0.9 }
      },
      // Differentiation from traditional worldview
      {
        id: 'worldview-diff-eval',
        type: 'EvaluationLink',
        children: ['differs-from-pred', 'worldview-diff-list'],
        attentionValue: 0.95
      },
      {
        id: 'differs-from-pred',
        type: 'PredicateNode',
        name: 'Differs-From',
        attentionValue: 0.8
      },
      {
        id: 'worldview-diff-list',
        type: 'ListLink',
        children: ['philosophical-transformation-root', 'traditional-turing-worldview'],
        attentionValue: 0.85
      },
      {
        id: 'traditional-turing-worldview',
        type: 'ConceptNode',
        name: 'Traditional-Turing-Worldview',
        attentionValue: 0.7
      }
    ];

    // Add research fields as subnodes (1-10)
    for (let i = 1; i <= 10; i++) {
      nodes.push({
        id: `research-field-${i}`,
        type: 'ConceptNode',
        name: `Research-Field-${i}`,
        attentionValue: 0.9 - (i * 0.02) // Decreasing attention by depth
      });
      
      // Inheritance links to root
      nodes.push({
        id: `research-field-${i}-inheritance`,
        type: 'InheritanceLink',
        children: [`research-field-${i}`, 'philosophical-transformation-root'],
        attentionValue: 0.85 - (i * 0.02)
      });
    }

    // Universe modeling
    nodes.push({
      id: 'universe-model-eval',
      type: 'EvaluationLink',
      children: ['models-universe-pred', 'universe-model-list'],
      attentionValue: 0.92
    });

    const transformation: PhilosophicalTransformation = {
      id: 'philosophical-transformation-1',
      rootNode: 'philosophical-transformation-root',
      researchFields: Array.from({ length: 10 }, (_, i) => `research-field-${i + 1}`),
      fundamentalQuestions: [
        'How does information look like in nature?',
        'Why do two individuals understand each other or the universe?',
        'What are the terminologies of life that computers do not support?',
        'How does consciousness emerge from geometric patterns?'
      ],
      brainModels: [
        'Neural Network Models', 'Connectome Mapping', 'Bayesian Brain',
        'Global Workspace Theory', 'Integrated Information Theory',
        'Orchestrated Objective Reduction', 'Predictive Processing',
        'Default Mode Network', 'Embodied Cognition', 'Quantum Mind Theories'
      ],
      universeModel: 'within-above',
      worldviewDifferentiation: [
        'Time crystal consciousness vs neuron-centric models',
        'Geometric musical language vs binary computation',
        'Prime-based symmetries vs classical algorithms'
      ],
      attentionPriority: 1.0
    };

    setAtomeseNodes(prev => [...prev, ...nodes]);
    setPhilosophicalTransformation(transformation);

    return { nodes, transformation };
  }, []);

  // 2. FRACTAL INFORMATION THEORY & GEOMETRIC MUSICAL LANGUAGE
  const initializeFractalTape = useCallback(() => {
    const nodes: AtomeseNode[] = [
      // Fractal tape root
      {
        id: 'fractal-tape-root',
        type: 'ConceptNode',
        name: 'Fractal-Tape',
        attentionValue: 0.98,
        truthValue: { strength: 0.92, confidence: 0.88 }
      },
      // FIT substrate relationship
      {
        id: 'fit-substrate-eval',
        type: 'EvaluationLink',
        children: ['substrate-for-pred', 'fit-substrate-list'],
        attentionValue: 0.95
      },
      {
        id: 'substrate-for-pred',
        type: 'PredicateNode',
        name: 'Substrate-For',
        attentionValue: 0.9
      },
      // Geometric Musical Language root
      {
        id: 'gml-root',
        type: 'ConceptNode',
        name: 'Geometric-Musical-Language',
        attentionValue: 0.98,
        truthValue: { strength: 0.94, confidence: 0.87 }
      },
      // GML information encoding
      {
        id: 'gml-encoding-eval',
        type: 'EvaluationLink',
        children: ['encodes-information-pred', 'gml-encoding-list'],
        attentionValue: 0.93
      }
    ];

    // 15 geometric shapes as basis for all patterns
    const geometricShapes: GeometricShape[] = [];
    for (let i = 1; i <= 15; i++) {
      const shape: GeometricShape = {
        id: `gml-shape-${i}`,
        name: `GML-Shape-${i}`,
        dimensions: Math.ceil(i / 5) + 2, // 3D, 4D, 5D progression
        primeIndex: fundamentalPrimes[i - 1],
        harmonicFrequency: 440 * Math.pow(2, (i - 1) / 12), // Musical scale
        complexity: i * 0.067, // Normalized complexity
        musicalNote: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][(i - 1) % 12]
      };
      geometricShapes.push(shape);

      // Add inheritance links to GML
      nodes.push({
        id: `gml-shape-${i}-inheritance`,
        type: 'InheritanceLink',
        children: [`gml-shape-${i}`, 'gml-root'],
        attentionValue: 0.96 - (i * 0.01)
      });
    }

    const gml: GeometricMusicalLanguage = {
      id: 'gml-1',
      shapes: geometricShapes,
      patterns: [
        '2D/3D Nested Spheres',
        'Self-Assembly Geometric Shapes',
        'Waveform Transformation Patterns',
        'Fractal Information Encoding'
      ],
      waveformTransformation: true,
      encodingMethod: 'patterns-3d-structures',
      basePatternsCount: 15
    };

    const fractalTapeSystem: FractalTape = {
      id: 'fractal-tape-1',
      incompletenessTheory: 'FIT-Incompleteness-Beyond-Godel',
      nestedSpheres: '3D',
      selfAssemblyShapes: geometricShapes,
      geometricMusicalLanguage: gml,
      attentionPriority: 0.98
    };

    setAtomeseNodes(prev => [...prev, ...nodes]);
    setFractalTape(fractalTapeSystem);

    return { nodes, fractalTapeSystem };
  }, [fundamentalPrimes]);

  // 3. PHASE PRIME METRIC - Prime-Based Symmetry Engine
  const initializePhasePrimeMetric = useCallback(() => {
    const nodes: AtomeseNode[] = [
      // Phase Prime Metric root
      {
        id: 'ppm-root',
        type: 'ConceptNode',
        name: 'Phase-Prime-Metric',
        attentionValue: 0.97,
        truthValue: { strength: 0.96, confidence: 0.91 }
      },
      // Symmetry encoding relationship
      {
        id: 'ppm-symmetry-eval',
        type: 'EvaluationLink',
        children: ['encodes-symmetry-pred', 'ppm-symmetry-list'],
        attentionValue: 0.94
      },
      {
        id: 'encodes-symmetry-pred',
        type: 'PredicateNode',
        name: 'Encodes-Symmetry',
        attentionValue: 0.9
      }
    ];

    // Ten metric classes
    const metricClasses: PrimeMetricClass[] = [];
    for (let i = 1; i <= 10; i++) {
      const metricClass: PrimeMetricClass = {
        id: `ppm-metric-${i}`,
        classNumber: i,
        description: `Metric Class ${i}: ${getMetricClassDescription(i)}`,
        primeRange: fundamentalPrimes.slice((i - 1) * 1.5, i * 1.5),
        geometricRepresentation: getGeometricRepresentation(i)
      };
      metricClasses.push(metricClass);

      // Add inheritance links to PPM root
      nodes.push({
        id: `ppm-metric-${i}-inheritance`,
        type: 'InheritanceLink',
        children: [`ppm-metric-${i}`, 'ppm-root'],
        attentionValue: 0.95 - (i * 0.02)
      });
    }

    // Ten prime operators
    const primeOperators: PrimeOperator[] = [];
    for (let i = 1; i <= 10; i++) {
      const operator: PrimeOperator = {
        id: `prime-operator-${i}`,
        operatorNumber: i,
        operation: getPrimeOperation(i),
        primeTarget: fundamentalPrimes[i - 1] || fundamentalPrimes[i % fundamentalPrimes.length],
        geometricEffect: getGeometricEffect(i)
      };
      primeOperators.push(operator);

      nodes.push({
        id: `prime-operator-${i}-inheritance`,
        type: 'InheritanceLink',
        children: [`prime-operator-${i}`, 'ppm-root'],
        attentionValue: 0.93 - (i * 0.015)
      });
    }

    const ppmSystem: PhasePrimeMetric = {
      id: 'ppm-1',
      primePatterns: fundamentalPrimes,
      metricClasses,
      primeOperators,
      symmetryEncoding: 'geometric',
      cognitiveGeometry: true,
      attentionPriority: 0.97
    };

    setAtomeseNodes(prev => [...prev, ...nodes]);
    setPhasePrimeMetric(ppmSystem);

    return { nodes, ppmSystem };
  }, [fundamentalPrimes]);

  // Helper functions for PPM system
  const getMetricClassDescription = (classNumber: number): string => {
    const descriptions = [
      'Integer geometric shape replacement',
      'Ordered factor metric from prime products',
      'Phase-limited integer paths (360°)',
      'Domain-limited prime utilization',
      'Ordered factor magnitude analysis',
      'Phase plot divisor hole detection',
      'Statistical prime dominance',
      'Normalized ripple periodicity',
      'Prime lattice group relationships',
      'Multilayer imaginary operations'
    ];
    return descriptions[classNumber - 1] || `Metric Class ${classNumber}`;
  };

  const getGeometricRepresentation = (classNumber: number): string => {
    const representations = [
      'Polygon', 'Spiral', 'Torus', 'Hypercube', 'Dodecahedron',
      'Icosahedron', 'Tetrahedron', 'Octahedron', 'Hexagon', 'Fractal'
    ];
    return representations[classNumber - 1] || 'Complex';
  };

  const getPrimeOperation = (operatorNumber: number): string => {
    const operations = [
      'Phase Rotation', 'Amplitude Modulation', 'Frequency Shift',
      'Geometric Transform', 'Symmetry Breaking', 'Pattern Completion',
      'Resonance Coupling', 'Quantum Entanglement', 'Dimensional Folding',
      'Temporal Synchronization'
    ];
    return operations[operatorNumber - 1] || `Operation ${operatorNumber}`;
  };

  const getGeometricEffect = (operatorNumber: number): string => {
    const effects = [
      'Spiral Rotation', 'Radial Expansion', 'Angular Shift',
      'Dimensional Projection', 'Symmetry Inversion', 'Pattern Emergence',
      'Harmonic Resonance', 'Quantum Coherence', 'Space Curvature',
      'Time Dilation'
    ];
    return effects[operatorNumber - 1] || `Effect ${operatorNumber}`;
  };

  // Generate cognitive flowcharts for visualization
  const generateCognitiveFlowcharts = useCallback(() => {
    const flowcharts: CognitiveFlowchart[] = [
      // Philosophical Transformation flowchart
      {
        id: 'philosophical-flowchart',
        featureType: 'philosophical',
        nodes: [
          { id: 'pt-root', label: 'Philosophical Transformation Node', type: 'root', position: { x: 0, y: 0 } },
          { id: 'pt-diff', label: 'Worldview Differentiation', type: 'branch', position: { x: -100, y: 100 } },
          { id: 'pt-research', label: 'Research Fields Node', type: 'branch', position: { x: 100, y: 100 } },
          { id: 'pt-universe', label: 'Universe Within/Above', type: 'leaf', position: { x: -150, y: 200 } },
          { id: 'pt-questions', label: 'Fundamental Questions', type: 'leaf', position: { x: -50, y: 200 } },
          { id: 'pt-brain', label: 'Brain Model Comparison', type: 'leaf', position: { x: 50, y: 200 } },
          { id: 'pt-understanding', label: 'Human Understanding Node', type: 'leaf', position: { x: 150, y: 200 } }
        ],
        connections: [
          { id: 'pt-c1', source: 'pt-root', target: 'pt-diff', type: 'flow' },
          { id: 'pt-c2', source: 'pt-root', target: 'pt-research', type: 'flow' },
          { id: 'pt-c3', source: 'pt-diff', target: 'pt-universe', type: 'expansion' },
          { id: 'pt-c4', source: 'pt-diff', target: 'pt-questions', type: 'expansion' },
          { id: 'pt-c5', source: 'pt-research', target: 'pt-brain', type: 'expansion' },
          { id: 'pt-c6', source: 'pt-research', target: 'pt-understanding', type: 'expansion' }
        ]
      },
      // Fractal Information Theory flowchart
      {
        id: 'fractal-flowchart',
        featureType: 'fractal',
        nodes: [
          { id: 'ft-root', label: 'Fractal Tape Node', type: 'root', position: { x: 0, y: 0 } },
          { id: 'ft-incompleteness', label: 'FIT Incompleteness', type: 'branch', position: { x: -100, y: 100 } },
          { id: 'ft-spheres', label: '2D/3D Nested Spheres', type: 'branch', position: { x: 100, y: 100 } },
          { id: 'ft-assembly', label: 'Self-Assembly: Geometric Shapes', type: 'leaf', position: { x: -150, y: 200 } },
          { id: 'ft-gml', label: 'Geometric Musical Language Node', type: 'leaf', position: { x: 0, y: 200 } },
          { id: 'ft-patterns', label: 'GML Shapes/Patterns', type: 'leaf', position: { x: 100, y: 300 } },
          { id: 'ft-waveform', label: 'Waveform Transformation', type: 'leaf', position: { x: 200, y: 300 } }
        ],
        connections: [
          { id: 'ft-c1', source: 'ft-root', target: 'ft-incompleteness', type: 'flow' },
          { id: 'ft-c2', source: 'ft-root', target: 'ft-spheres', type: 'flow' },
          { id: 'ft-c3', source: 'ft-root', target: 'ft-assembly', type: 'flow' },
          { id: 'ft-c4', source: 'ft-root', target: 'ft-gml', type: 'flow' },
          { id: 'ft-c5', source: 'ft-gml', target: 'ft-patterns', type: 'expansion' },
          { id: 'ft-c6', source: 'ft-patterns', target: 'ft-waveform', type: 'expansion' }
        ]
      },
      // Phase Prime Metric flowchart
      {
        id: 'ppm-flowchart',
        featureType: 'prime-metric',
        nodes: [
          { id: 'ppm-root', label: 'Phase Prime Metric Node', type: 'root', position: { x: 0, y: 0 } },
          { id: 'ppm-patterns', label: 'Prime Patterns', type: 'branch', position: { x: -100, y: 100 } },
          { id: 'ppm-metrics', label: 'Metric Classes 1–10', type: 'branch', position: { x: 100, y: 100 } },
          { id: 'ppm-encoding', label: 'Geometric Encoding', type: 'leaf', position: { x: -150, y: 200 } },
          { id: 'ppm-operators', label: 'Prime Operators', type: 'leaf', position: { x: 50, y: 200 } },
          { id: 'ppm-symmetry', label: 'Symmetry/Emergence', type: 'leaf', position: { x: -50, y: 300 } },
          { id: 'ppm-geometry', label: 'Cognitive Geometry', type: 'leaf', position: { x: 150, y: 300 } }
        ],
        connections: [
          { id: 'ppm-c1', source: 'ppm-root', target: 'ppm-patterns', type: 'flow' },
          { id: 'ppm-c2', source: 'ppm-root', target: 'ppm-metrics', type: 'flow' },
          { id: 'ppm-c3', source: 'ppm-patterns', target: 'ppm-encoding', type: 'expansion' },
          { id: 'ppm-c4', source: 'ppm-metrics', target: 'ppm-operators', type: 'expansion' },
          { id: 'ppm-c5', source: 'ppm-encoding', target: 'ppm-symmetry', type: 'inheritance' },
          { id: 'ppm-c6', source: 'ppm-operators', target: 'ppm-geometry', type: 'inheritance' }
        ]
      }
    ];

    setCognitiveFlowcharts(flowcharts);
    return flowcharts;
  }, []);

  // Enhanced adaptive attention allocation with dynamic propagation
  const allocateAdaptiveAttention = useCallback(() => {
    setAtomeseNodes(prev => prev.map(node => {
      let newAttentionValue = node.attentionValue || 0.5;
      
      // Root nodes maintain highest attention (1.0)
      if (node.name?.includes('Philosophical-Transformation')) {
        newAttentionValue = Math.min(1.0, newAttentionValue + 0.02);
      } else if (node.name?.includes('Fractal-Tape') || 
                 node.name?.includes('Phase-Prime-Metric')) {
        newAttentionValue = Math.min(0.98, newAttentionValue + 0.015);
      }
      
      // Recursive subnodes inherit attention based on structural proximity
      if (node.type === 'InheritanceLink') {
        newAttentionValue = Math.max(0.1, newAttentionValue - 0.02);
      } else if (node.type === 'ImplicationLink') {
        // ImplicationLinks propagate attention dynamically
        newAttentionValue = Math.max(0.2, newAttentionValue - 0.01);
      } else if (node.type === 'EvaluationLink') {
        // EvaluationLinks maintain intermediate attention
        newAttentionValue = Math.max(0.15, newAttentionValue - 0.015);
      }
      
      // Dynamic propagation based on cognitive demands and truth values
      if (node.truthValue && node.truthValue.strength > 0.9) {
        newAttentionValue = Math.min(1.0, newAttentionValue + 0.01);
      }
      
      // Attention decay for low-truth value nodes
      if (node.truthValue && node.truthValue.strength < 0.5) {
        newAttentionValue = Math.max(0.05, newAttentionValue - 0.005);
      }
      
      // Boost attention for nodes related to neural-symbolic integration
      if (node.name?.includes('Neural-Symbolic') || 
          node.name?.includes('Integration-Hub') ||
          node.name?.includes('Pattern-Mining')) {
        newAttentionValue = Math.min(0.97, newAttentionValue + 0.008);
      }

      return {
        ...node,
        attentionValue: newAttentionValue
      };
    }));
  }, []);

  // Initialize all features with enhanced recursive instantiation
  const initializeAllFeatures = useCallback(() => {
    const pt = initializePhilosophicalTransformation();
    const ft = initializeFractalTape();
    const ppm = initializePhasePrimeMetric();
    const flowcharts = generateCognitiveFlowcharts();
    const recursiveKB = initializeRecursiveAtomeseKnowledgeBase();
    
    return {
      philosophicalTransformation: pt.transformation,
      fractalTape: ft.fractalTapeSystem,
      phasePrimeMetric: ppm.ppmSystem,
      cognitiveFlowcharts: flowcharts,
      recursiveAtomeseKnowledgeBase: recursiveKB.knowledgeBase,
      schemeCode: recursiveKB.schemeOutput,
      totalAtomeseNodes: pt.nodes.length + ft.nodes.length + ppm.nodes.length + recursiveKB.knowledgeBase.length
    };
  }, [initializePhilosophicalTransformation, initializeFractalTape, initializePhasePrimeMetric, generateCognitiveFlowcharts, initializeRecursiveAtomeseKnowledgeBase]);

  // Auto-allocate attention every 2 seconds
  useEffect(() => {
    const interval = setInterval(allocateAdaptiveAttention, 2000);
    return () => clearInterval(interval);
  }, [allocateAdaptiveAttention]);

  return {
    // State
    atomeseNodes,
    philosophicalTransformation,
    fractalTape,
    phasePrimeMetric,
    cognitiveFlowcharts,
    recursiveAtomeseKnowledgeBase,
    schemeCode,
    
    // Actions
    initializeAllFeatures,
    initializePhilosophicalTransformation,
    initializeFractalTape,
    initializePhasePrimeMetric,
    generateCognitiveFlowcharts,
    allocateAdaptiveAttention,
    initializeRecursiveAtomeseKnowledgeBase,
    
    // Computed values
    totalNodes: atomeseNodes.length,
    averageAttention: atomeseNodes.reduce((sum, node) => sum + (node.attentionValue || 0), 0) / Math.max(atomeseNodes.length, 1),
    fundamentalPrimes,
    
    // Cognitive architecture metrics
    rootNodeAttention: atomeseNodes.filter(n => 
      n.name?.includes('Philosophical-Transformation') || 
      n.name?.includes('Fractal-Tape') || 
      n.name?.includes('Phase-Prime-Metric')
    ).reduce((sum, node) => sum + (node.attentionValue || 0), 0) / 3,
    
    recursiveDepth: Math.max(...atomeseNodes.map(n => n.children?.length || 0)),
    neuralSymbolicBridgeStrength: atomeseNodes.filter(n => 
      n.name?.includes('Neural-Symbolic')
    ).reduce((sum, node) => sum + (node.truthValue?.strength || 0), 0)
  };
};