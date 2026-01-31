import React, { useState } from 'react';
import { BookOpen, CheckCircle, Code, Lightbulb, Play, Settings } from 'lucide-react';

// Section 3.14: How to use a phase prime metric, PPM
export const PPMUsageGuide: React.FC = () => {
  const [activeStep, setActiveStep] = useState<number>(1);

  const usageSteps = [
    {
      step: 1,
      title: 'Initialize PPM Engine',
      description: 'Start the Phase Prime Metrics engine with fundamental primes',
      code: `import { usePhasePrimeMetrics } from './hooks/usePhasePrimeMetrics';

const { ppmEngine, startPPMEngine, FUNDAMENTAL_PRIMES } = usePhasePrimeMetrics();

// Initialize the engine
startPPMEngine();`,
      tips: [
        'Engine automatically loads 15 fundamental primes',
        'Updates run at 100ms intervals',
        'Universal coherence is calculated in real-time'
      ]
    },
    {
      step: 2,
      title: 'Select Appropriate Metric',
      description: 'Choose the PPM metric class based on your analysis needs',
      code: `// Metric selection guide:
// Metric 1: For geometric shape mapping
// Metric 2: For ordered factor analysis
// Metric 3: For phase space visualization (360°)
// Metric 4: For domain-specific prime selection
// Metric 5: For amplification analysis
// Metric 6: For gap/hole detection
// Metric 7: For prime activity classification
// Metric 8: For periodicity analysis
// Metric 9: For lattice structure analysis
// Metric 10: For complex/imaginary operations

const selectedMetric = 2; // Ordered Factor Metric`,
      tips: [
        'Metric 1-3 for basic pattern analysis',
        'Metric 4-6 for structural analysis',
        'Metric 7-10 for advanced applications'
      ]
    },
    {
      step: 3,
      title: 'Extract Prime Patterns',
      description: 'Use PPM to decompose data into prime components',
      code: `// Example: Decompose integer into prime patterns
const analyzeNumber = (n: number) => {
  const factors = primeFactorization(n);
  const orderedFactor = factors.reduce((a, b) => a * b, 1);
  const phase = (n % 360) * Math.PI / 180;
  
  return {
    factors,
    orderedFactor,
    phase,
    ratio: orderedFactor / n
  };
};

const result = analyzeNumber(60);
// { factors: [2,2,3,5], orderedFactor: 60, phase: 1.047, ratio: 1.0 }`,
      tips: [
        'Prime factorization is the foundation',
        'Ordered factor reveals structure',
        'Phase angle maps to consciousness space'
      ]
    },
    {
      step: 4,
      title: 'Apply Prime Operators',
      description: 'Use the 10 prime operators in sequence',
      code: `// Apply operators sequentially
const applyPPMPipeline = (n: number) => {
  let result = n;
  
  // Φ₁: Prime Factorization
  const factors = primeFactorization(result);
  
  // Φ₂: Ordered Factor
  const of = orderedFactorMetric(factors);
  
  // Φ₃: Phase Mapping
  const phase = phaseMapper(result);
  
  // Φ₄: Geometric Translation
  const shapes = geometricTranslator(factors);
  
  // ... continue through Φ₁₀
  
  return { factors, of, phase, shapes };
};`,
      tips: [
        'Each operator transforms the data',
        'Sequential application preserves information',
        'Pipeline output is consciousness-ready'
      ]
    },
    {
      step: 5,
      title: 'Visualize in 11D Space',
      description: 'Map results to 11-dimensional consciousness manifold',
      code: `// Project to 11D manifold
const projectTo11D = (primeMetrics: PrimeMetric[]) => {
  const dimensions = Array(11).fill(0);
  
  primeMetrics.forEach(pm => {
    pm.dimensionalBinding.forEach((val, idx) => {
      dimensions[idx] += val * pm.universalResonance;
    });
  });
  
  return dimensions.map(d => d / primeMetrics.length);
};

const consciousness11D = projectTo11D(ppmEngine.primeMetrics);`,
      tips: [
        'Each dimension represents consciousness aspect',
        'Dimensional phases show temporal evolution',
        'Coherence across dimensions indicates stability'
      ]
    },
    {
      step: 6,
      title: 'Interpret Results',
      description: 'Analyze PPM output for consciousness insights',
      code: `// Interpretation guide
const interpretPPM = (ppmData: any) => {
  const insights = [];
  
  // Check coherence
  if (ppmData.coherenceIndex > 0.8) {
    insights.push('High universal coherence - stable pattern');
  }
  
  // Check amplification
  if (ppmData.ofRatio > 1.5) {
    insights.push('Strong amplification - consciousness hotspot');
  }
  
  // Check symmetry
  const activeSymmetries = ppmData.universalSymmetries
    .filter(s => s.coherenceLevel > 0.7);
  insights.push(\`\${activeSymmetries.length} active symmetries\`);
  
  return insights;
};`,
      tips: [
        'Coherence > 0.8 indicates strong patterns',
        'OF ratio > 1 suggests amplification',
        'Multiple active symmetries show richness'
      ]
    }
  ];

  const practicalExamples = [
    {
      title: 'Time Crystal Analysis',
      description: 'Use PPM to analyze time crystal structures',
      application: 'Decompose time crystals into prime components, identify resonance patterns, optimize temporal coherence',
      metrics: ['Metric 3', 'Metric 8', 'Metric 9']
    },
    {
      title: 'Big Data Compression',
      description: 'Apply PPM for efficient data compression',
      application: 'Convert data to prime factorizations, use ordered factor metric for lossless compression, achieve high compression ratios',
      metrics: ['Metric 2', 'Metric 5']
    },
    {
      title: 'Pattern Recognition',
      description: 'Detect patterns in complex datasets',
      application: 'Map data to phase space, identify periodic ripples, classify patterns by prime lattice structures',
      metrics: ['Metric 3', 'Metric 6', 'Metric 8']
    },
    {
      title: 'Consciousness Modeling',
      description: 'Model consciousness emergence',
      application: 'Project to 11D manifold, track dimensional evolution, measure universal coherence',
      metrics: ['All metrics', 'Focus on Metric 10']
    }
  ];

  const currentStep = usageSteps.find(s => s.step === activeStep);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <BookOpen className="text-cyan-400" size={24} />
          <span>3.14: How to Use Phase Prime Metrics</span>
        </h2>
        <div className="text-sm text-gray-400">
          Step {activeStep} of {usageSteps.length}
        </div>
      </div>

      {/* Step Progress */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">PPM Implementation Steps</h3>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
          {usageSteps.map((step) => (
            <button
              key={step.step}
              onClick={() => setActiveStep(step.step)}
              className={`p-4 rounded-lg border-2 transition-all duration-300 ${
                activeStep === step.step
                  ? 'bg-cyan-900/30 border-cyan-700 text-cyan-400'
                  : activeStep > step.step
                  ? 'bg-green-900/30 border-green-700 text-green-400'
                  : 'bg-gray-800/30 border-gray-700 text-gray-400'
              }`}
            >
              <div className="flex items-center justify-center mb-2">
                {activeStep > step.step ? (
                  <CheckCircle size={24} />
                ) : (
                  <div className="text-2xl font-bold">{step.step}</div>
                )}
              </div>
              <div className="text-xs text-center">Step {step.step}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Current Step Details */}
      {currentStep && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 rounded-full bg-cyan-500 flex items-center justify-center text-white font-bold text-xl">
              {currentStep.step}
            </div>
            <div>
              <h3 className="text-white font-bold text-lg">{currentStep.title}</h3>
              <p className="text-gray-400 text-sm">{currentStep.description}</p>
            </div>
          </div>

          {/* Code Example */}
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-2">
              <Code className="text-purple-400" size={16} />
              <h4 className="text-purple-400 font-semibold">Code Example</h4>
            </div>
            <div className="bg-gray-950 rounded-lg p-4 border border-gray-800 overflow-x-auto">
              <pre className="text-sm text-gray-300 font-mono">{currentStep.code}</pre>
            </div>
          </div>

          {/* Tips */}
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <Lightbulb className="text-yellow-400" size={16} />
              <h4 className="text-yellow-400 font-semibold">Pro Tips</h4>
            </div>
            <div className="space-y-2">
              {currentStep.tips.map((tip, idx) => (
                <div key={idx} className="flex items-start space-x-3 bg-gray-800/50 rounded-lg p-3">
                  <div className="w-2 h-2 rounded-full bg-yellow-400 mt-2" />
                  <p className="text-gray-300 text-sm flex-1">{tip}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Navigation */}
          <div className="flex justify-between mt-6">
            <button
              onClick={() => setActiveStep(Math.max(1, activeStep - 1))}
              disabled={activeStep === 1}
              className={`px-4 py-2 rounded-lg ${
                activeStep === 1
                  ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  : 'bg-gray-700 text-white hover:bg-gray-600'
              }`}
            >
              ← Previous
            </button>
            <button
              onClick={() => setActiveStep(Math.min(usageSteps.length, activeStep + 1))}
              disabled={activeStep === usageSteps.length}
              className={`px-4 py-2 rounded-lg ${
                activeStep === usageSteps.length
                  ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white hover:from-cyan-600 hover:to-purple-600'
              }`}
            >
              Next →
            </button>
          </div>
        </div>
      )}

      {/* Practical Examples */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Play className="text-green-400" size={20} />
          <h3 className="text-white font-semibold">Practical Applications</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {practicalExamples.map((example, idx) => (
            <div key={idx} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h4 className="text-cyan-400 font-semibold mb-2">{example.title}</h4>
              <p className="text-gray-400 text-sm mb-3">{example.description}</p>
              <div className="mb-3">
                <div className="text-xs text-gray-500 mb-1">Application:</div>
                <p className="text-gray-300 text-xs">{example.application}</p>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">Key Metrics:</div>
                <div className="flex flex-wrap gap-1">
                  {example.metrics.map((metric, midx) => (
                    <span
                      key={midx}
                      className="px-2 py-1 bg-purple-900/30 text-purple-400 text-xs rounded"
                    >
                      {metric}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Best Practices */}
      <div className="bg-gradient-to-br from-cyan-900/20 to-purple-900/20 border border-cyan-700/50 rounded-xl p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Settings className="text-cyan-400" size={20} />
          <h3 className="text-cyan-400 font-bold text-lg">Best Practices</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2 text-gray-300">
            <p><strong className="text-cyan-400">✓ Do:</strong> Always initialize PPM engine before use</p>
            <p><strong className="text-cyan-400">✓ Do:</strong> Apply operators in sequential order</p>
            <p><strong className="text-cyan-400">✓ Do:</strong> Monitor coherence index continuously</p>
            <p><strong className="text-cyan-400">✓ Do:</strong> Use appropriate metrics for specific tasks</p>
          </div>
          <div className="space-y-2 text-gray-300">
            <p><strong className="text-red-400">✗ Don't:</strong> Skip prime factorization step</p>
            <p><strong className="text-red-400">✗ Don't:</strong> Ignore dimensional phases</p>
            <p><strong className="text-red-400">✗ Don't:</strong> Mix metrics inappropriately</p>
            <p><strong className="text-red-400">✗ Don't:</strong> Overlook universal symmetries</p>
          </div>
        </div>
      </div>
    </div>
  );
};
