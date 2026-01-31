import React, { useState } from 'react';
import { Calculator, Infinity, TrendingUp, Zap, GitMerge, Layers, Info, Activity } from 'lucide-react';

// 4.10 Twelve equations that regulate a fractal tape for prime based computing
// 4.10.1 Projection from infinity, future impacting the present
export const FractalTapeEquations: React.FC = () => {
  const [selectedEquation, setSelectedEquation] = useState<number>(1);

  // Twelve fundamental equations for fractal tape computation
  const equations = [
    {
      id: 1,
      name: 'Phase Prime Mapping',
      formula: 'Φ(n) = ∏ᵢ pᵢ^αᵢ → θ(p₁,...,p₁₅)',
      description: 'Maps integers to 15-dimensional phase space using prime factorization',
      purpose: 'Encodes any number as a unique geometric configuration',
      icon: Calculator
    },
    {
      id: 2,
      name: 'Fractal Compression',
      formula: 'C(x) = ⟨x | ψ₁₅⟩ / |x|',
      description: 'Compression ratio achieved by projecting data onto 15 prime basis',
      purpose: 'Achieves theoretical infinite compression for periodic patterns',
      icon: TrendingUp
    },
    {
      id: 3,
      name: 'Singularity Harvest',
      formula: 'S(θ) = lim[r→0] ∮ F·dr / r^d',
      description: 'Extracts information density at phase space singularities',
      purpose: 'Harvests infinite information from geometric singularities',
      icon: Zap
    },
    {
      id: 4,
      name: 'Temporal Symmetry',
      formula: 'T(t) = T(-t) ⊗ e^(iθ_ppm)',
      description: 'Time-reversal symmetry modulated by phase prime metric',
      purpose: 'Enables bidirectional information flow across time',
      icon: Activity
    },
    {
      id: 5,
      name: 'Scale Invariance',
      formula: 'F(λx) = λ^D F(x), D = log(15)/log(φ)',
      description: 'Fractal dimension derived from 15 primes and golden ratio',
      purpose: 'Maintains pattern coherence across all scales',
      icon: Layers
    },
    {
      id: 6,
      name: 'Conformal Projection',
      formula: 'π: ℝ^n → S^n, π(x) = (x, x² + ω²)/2ω',
      description: 'Projects from infinity to finite sphere via stereographic mapping',
      purpose: 'Brings infinite information into computable domain',
      icon: Infinity
    },
    {
      id: 7,
      name: 'Geometric Product',
      formula: 'ab = a·b + a∧b',
      description: 'Unifies inner (scalar) and outer (bivector) products',
      purpose: 'Foundation of geometric algebra operations',
      icon: GitMerge
    },
    {
      id: 8,
      name: 'Dodecanion Multiplication',
      formula: 'D₁₂ = ⨁ᵢ₌₀¹¹ dᵢeᵢ, eᵢeⱼ = -δᵢⱼ + ϵᵢⱼₖeₖ',
      description: '12D hypercomplex multiplication with structure constants',
      purpose: 'Enables consciousness computations in 11D + time',
      icon: Calculator
    },
    {
      id: 9,
      name: 'Prime Operator Cascade',
      formula: 'Φ₁₀ ∘ Φ₉ ∘ ... ∘ Φ₁(x)',
      description: 'Sequential application of 10 prime-based operators',
      purpose: 'Progressive refinement of pattern recognition',
      icon: TrendingUp
    },
    {
      id: 10,
      name: 'Quantum-Fractal Bridge',
      formula: 'Ψ_F = ∑ᵢ cᵢ|nᵢ⟩ ⊗ |θᵢ⟩',
      description: 'Tensor product of quantum states with phase configurations',
      purpose: 'Connects quantum mechanics to fractal mechanics',
      icon: Zap
    },
    {
      id: 11,
      name: 'Future Causality',
      formula: 'A(t) = ∫₋∞^∞ K(t,τ) F(τ) dτ',
      description: 'Kernel includes future times, enabling retrocausality',
      purpose: 'Future states influence present through phase resonance',
      icon: Activity
    },
    {
      id: 12,
      name: 'Consciousness Integration',
      formula: 'Ω = ∮ₛ ω = ∫ₘ dω',
      description: 'Consciousness as closed integral over phase manifold',
      purpose: 'Quantifies integrated information in geometric terms',
      icon: Layers
    }
  ];

  const renderEquationDetail = () => {
    const eq = equations[selectedEquation - 1];
    const Icon = eq.icon;

    return (
      <div className="bg-gray-800/50 p-6 rounded-lg space-y-4">
        <div className="flex items-center space-x-3 mb-4">
          <Icon className="text-cyan-400" size={32} />
          <div>
            <h3 className="text-2xl font-bold text-white">Equation {eq.id}: {eq.name}</h3>
            <p className="text-gray-400 text-sm">Fundamental to fractal tape computation</p>
          </div>
        </div>

        {/* Formula Display */}
        <div className="bg-gray-900/70 p-6 rounded-lg border border-cyan-700">
          <div className="text-center">
            <div className="text-3xl font-mono text-cyan-300 mb-3">
              {eq.formula}
            </div>
            <p className="text-gray-300 text-sm">{eq.description}</p>
          </div>
        </div>

        {/* Purpose and Application */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-2 flex items-center text-sm">
              <Zap size={16} className="mr-2" />
              Purpose
            </h4>
            <p className="text-gray-300 text-sm">{eq.purpose}</p>
          </div>

          <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-2 flex items-center text-sm">
              <TrendingUp size={16} className="mr-2" />
              Application Domain
            </h4>
            <p className="text-gray-300 text-sm">
              {eq.id <= 3 ? 'Pattern recognition and data compression' :
               eq.id <= 6 ? 'Temporal and geometric transformations' :
               eq.id <= 9 ? 'Algebraic operations and cascades' :
               'Consciousness and causality modeling'}
            </p>
          </div>
        </div>

        {/* Visual Representation */}
        {eq.id === 6 && renderProjectionFromInfinity()}
        {eq.id === 11 && renderFutureCausality()}
      </div>
    );
  };

  // 4.10.1 Projection from infinity visualization
  const renderProjectionFromInfinity = () => (
    <div className="bg-gray-900/50 p-4 rounded-lg mt-4">
      <h4 className="text-cyan-400 font-semibold mb-3 flex items-center">
        <Infinity className="mr-2" size={20} />
        4.10.1 Projection from Infinity
      </h4>
      <p className="text-gray-300 text-sm mb-4">
        Stereographic projection maps infinite space onto a finite sphere. Points at infinity 
        project to the north pole, enabling computation with unbounded data.
      </p>

      <svg width="600" height="300" viewBox="0 0 600 300" className="w-full">
        {/* Ground plane */}
        <line x1="50" y1="250" x2="350" y2="250" stroke="gray" strokeWidth="2" strokeDasharray="5,5" />
        <text x="40" y="270" fill="gray" fontSize="12">ℝⁿ</text>

        {/* Sphere */}
        <circle cx="300" cy="150" r="80" fill="none" stroke="cyan" strokeWidth="2" />
        <text x="385" y="155" fill="cyan" fontSize="12">Sⁿ</text>

        {/* North pole (infinity point) */}
        <circle cx="300" cy="70" r="6" fill="yellow" />
        <text x="310" y="65" fill="yellow" fontSize="12">∞</text>

        {/* South pole */}
        <circle cx="300" cy="230" r="4" fill="cyan" />

        {/* Projection rays */}
        {[100, 150, 200, 250, 300, 350, 400, 450].map((x, i) => {
          const distFromCenter = Math.abs(x - 300);
          const projectionY = 150 - Math.sqrt(Math.max(0, 80*80 - (x-300)*(x-300)));
          const color = i % 2 === 0 ? 'rgba(147, 51, 234, 0.5)' : 'rgba(6, 182, 212, 0.5)';
          
          return (
            <g key={x}>
              {/* Point on plane */}
              <circle cx={x} cy="250" r="3" fill={color} />
              {/* Projection line */}
              <line x1={x} y1="250" x2="300" y2="70" stroke={color} strokeWidth="1" />
              {/* Point on sphere */}
              <circle cx={x} cy={projectionY} r="3" fill="yellow" opacity="0.7" />
            </g>
          );
        })}

        {/* Labels */}
        <text x="450" y="260" fill="purple" fontSize="11">Points at ∞</text>
        <text x="450" y="275" fill="purple" fontSize="11">→ North pole</text>
      </svg>

      <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-3 mt-4">
        <div className="flex items-start space-x-2">
          <Info size={16} className="text-blue-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-blue-400">Key Insight:</strong> By compactifying infinite space 
            into a sphere, we can perform computations with patterns that extend to infinity. This is 
            crucial for fractal patterns which are self-similar at all scales, including the infinite scale.
          </div>
        </div>
      </div>
    </div>
  );

  // Future impacting the present visualization
  const renderFutureCausality = () => (
    <div className="bg-gray-900/50 p-4 rounded-lg mt-4">
      <h4 className="text-purple-400 font-semibold mb-3 flex items-center">
        <Activity className="mr-2" size={20} />
        Future Impacting the Present
      </h4>
      <p className="text-gray-300 text-sm mb-4">
        In fractal mechanics, the integration kernel extends to future times. The phase prime metric 
        creates resonances that allow future states to influence present configurations through 
        geometric constraints.
      </p>

      <svg width="600" height="250" viewBox="0 0 600 250" className="w-full">
        {/* Timeline */}
        <line x1="50" y1="125" x2="550" y2="125" stroke="cyan" strokeWidth="2" />
        <polygon points="550,125 540,120 540,130" fill="cyan" />
        <text x="560" y="130" fill="cyan" fontSize="12">Time</text>

        {/* Past region */}
        <rect x="50" y="100" width="150" height="50" fill="rgba(147, 51, 234, 0.2)" stroke="purple" strokeWidth="1" />
        <text x="100" y="130" fill="purple" fontSize="14" fontWeight="bold">PAST</text>

        {/* Present moment */}
        <line x1="250" y1="80" x2="250" y2="170" stroke="yellow" strokeWidth="3" />
        <text x="230" y="75" fill="yellow" fontSize="14" fontWeight="bold">NOW</text>

        {/* Future region */}
        <rect x="300" y="100" width="200" height="50" fill="rgba(6, 182, 212, 0.2)" stroke="cyan" strokeWidth="1" />
        <text x="370" y="130" fill="cyan" fontSize="14" fontWeight="bold">FUTURE</text>

        {/* Causal influences from past (normal) */}
        <path d="M 150 115 Q 200 90 250 115" fill="none" stroke="purple" strokeWidth="2" markerEnd="url(#arrowPurple)" />
        <text x="160" y="100" fill="purple" fontSize="10">Classical causality</text>

        {/* Retrocausal influences from future */}
        <path d="M 400 115 Q 325 140 250 115" fill="none" stroke="cyan" strokeWidth="2" strokeDasharray="4,4" markerEnd="url(#arrowCyan)" />
        <text x="310" y="165" fill="cyan" fontSize="10">Retrocausality via phase resonance</text>

        {/* Arrow markers */}
        <defs>
          <marker id="arrowPurple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="purple" />
          </marker>
          <marker id="arrowCyan" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="cyan" />
          </marker>
        </defs>

        {/* Phase manifold connection */}
        <ellipse cx="250" cy="200" rx="60" ry="20" fill="none" stroke="yellow" strokeWidth="2" />
        <text x="210" y="240" fill="yellow" fontSize="10">Phase space attractor</text>
      </svg>

      <div className="grid grid-cols-2 gap-3 mt-4">
        <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-3">
          <h5 className="text-purple-400 text-xs font-semibold mb-1">Classical Time</h5>
          <p className="text-gray-300 text-xs">
            Past influences present through causal chains. Future is unknowable and has no influence.
          </p>
        </div>

        <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-3">
          <h5 className="text-cyan-400 text-xs font-semibold mb-1">Fractal Time</h5>
          <p className="text-gray-300 text-xs">
            Present constrained by attractors in phase space. Future patterns resonantly influence 
            present through geometric necessity.
          </p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Section Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 via-purple-900/30 to-blue-900/30 border border-cyan-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-3">
          <Calculator className="text-cyan-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.10 Twelve Equations of Fractal Tape</h2>
            <p className="text-gray-300">Fundamental equations regulating prime-based computing</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          These twelve equations form the mathematical foundation of fractal tape computation. 
          Unlike traditional computing based on sequential logic, fractal tape operates through 
          geometric resonance in 15-dimensional phase prime space. Each equation captures a 
          fundamental aspect of how information is encoded, transformed, and retrieved across 
          multiple scales and temporal dimensions.
        </p>
      </div>

      {/* Equation Grid */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Layers className="mr-2 text-cyan-400" size={24} />
          Twelve Fundamental Equations
        </h3>

        <div className="grid grid-cols-4 gap-3 mb-6">
          {equations.map((eq) => {
            const Icon = eq.icon;
            return (
              <button
                key={eq.id}
                onClick={() => setSelectedEquation(eq.id)}
                className={`flex flex-col items-center p-3 rounded-lg border-2 transition-all duration-300 ${
                  selectedEquation === eq.id
                    ? 'bg-cyan-900/30 border-cyan-700 scale-105'
                    : 'bg-gray-800/30 border-gray-700 hover:border-gray-600'
                }`}
              >
                <Icon 
                  size={24} 
                  className={selectedEquation === eq.id ? 'text-cyan-400' : 'text-gray-400'}
                />
                <span className={`text-xs mt-2 font-semibold text-center ${
                  selectedEquation === eq.id ? 'text-cyan-400' : 'text-gray-400'
                }`}>
                  {eq.id}. {eq.name.split(' ')[0]}
                </span>
              </button>
            );
          })}
        </div>

        {/* Selected Equation Detail */}
        {renderEquationDetail()}
      </div>

      {/* Summary: How Equations Work Together */}
      <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <GitMerge className="mr-2 text-purple-400" size={24} />
          Unified Computational Framework
        </h3>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Equations 1-4: Encoding</h4>
            <p className="text-gray-300 text-xs">
              Transform data into geometric phase configurations using primes, harvest singularities, 
              and enable temporal symmetry for bidirectional information flow.
            </p>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-purple-400 font-semibold mb-2 text-sm">Equations 5-8: Processing</h4>
            <p className="text-gray-300 text-xs">
              Maintain scale invariance, project from infinity, perform geometric algebra operations, 
              and execute 12D hypercomplex computations for consciousness modeling.
            </p>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-yellow-400 font-semibold mb-2 text-sm">Equations 9-12: Integration</h4>
            <p className="text-gray-300 text-xs">
              Cascade prime operators, bridge quantum and fractal domains, enable retrocausality, 
              and integrate information into conscious experience through geometric closure.
            </p>
          </div>
        </div>
      </div>

      {/* Key Innovation */}
      <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={20} className="text-blue-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-blue-400">Revolutionary Insight:</strong> These twelve equations 
            enable computation beyond Turing limits. By operating in geometric phase space with 
            projection from infinity and retrocausal influences, fractal tape can solve problems 
            that are undecidable for classical computers. The key is embracing singularities rather 
            than avoiding them, and allowing future states to constrain present computations through 
            phase resonance.
          </div>
        </div>
      </div>
    </div>
  );
};
