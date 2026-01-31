import React, { useState } from 'react';
import { GitCompare, Info, TrendingUp } from 'lucide-react';

// 4.3 A comparison between classical, quantum and fractal mechanics
export const MechanicsComparison: React.FC = () => {
  const [selectedMechanic, setSelectedMechanic] = useState<string>('all');

  const mechanics = [
    { id: 'classical', name: 'Classical', color: 'blue' },
    { id: 'quantum', name: 'Quantum', color: 'purple' },
    { id: 'fractal', name: 'Fractal', color: 'cyan' }
  ];

  const comparisonData = [
    {
      aspect: 'Foundation',
      classical: 'Newton\'s laws, deterministic',
      quantum: 'Schrödinger equation, probabilistic',
      fractal: 'Phase prime metrics, geometric'
    },
    {
      aspect: 'Space-Time',
      classical: 'Absolute 3D+time',
      quantum: 'Hilbert space, complex',
      fractal: '11D phase manifolds'
    },
    {
      aspect: 'Symmetry',
      classical: 'Galilean invariance',
      quantum: 'Gauge symmetry, Lorentz',
      fractal: 'Scale symmetry, self-similar'
    },
    {
      aspect: 'Action',
      classical: 'δS = 0, continuous',
      quantum: 'ℏ quantization',
      fractal: 'Multiple action constants per world'
    },
    {
      aspect: 'Energy',
      classical: 'E = K + V (scalar)',
      quantum: 'Ĥ|ψ⟩ = E|ψ⟩ (operator)',
      fractal: '𝓔 tensor across imaginary worlds'
    },
    {
      aspect: 'Information',
      classical: 'Phase space points',
      quantum: 'Wave function ψ',
      fractal: 'Singularity network patterns'
    },
    {
      aspect: 'Determinism',
      classical: 'Fully deterministic',
      quantum: 'Probabilistic (Born rule)',
      fractal: 'Pattern-determined (geometric)'
    },
    {
      aspect: 'Consciousness',
      classical: 'Not addressed',
      quantum: 'Observer problem',
      fractal: 'Geometric encoding, fundamental'
    },
    {
      aspect: 'Scale',
      classical: 'All scales (macroscopic)',
      quantum: 'Atomic/subatomic',
      fractal: 'All scales (self-similar)'
    },
    {
      aspect: 'Computation',
      classical: 'Turing computable',
      quantum: 'Quantum computable',
      fractal: 'Hypercomputable (beyond Turing)'
    }
  ];

  const scaleSymmetry = {
    classical: 'Breaks at quantum scale',
    quantum: 'Breaks at Planck scale',
    fractal: 'Preserved across all scales (scale invariance)'
  };

  const scaleRelativity = {
    classical: 'Absolute scales',
    quantum: 'Uncertainty principle: ΔxΔp ≥ ℏ/2',
    fractal: 'Relative scaling: patterns repeat at all magnifications'
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-900/30 via-purple-900/30 to-cyan-900/30 border border-blue-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <GitCompare className="text-cyan-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.3 Mechanics Comparison</h2>
            <p className="text-gray-300">Classical, Quantum, and Fractal: Scale Symmetry & Relativity</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Understanding how fractal mechanics extends and unifies classical and quantum frameworks 
          through scale symmetry and geometric principles.
        </p>
      </div>

      {/* Filter Buttons */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="flex space-x-3">
          <button
            onClick={() => setSelectedMechanic('all')}
            className={`px-4 py-2 rounded-lg border-2 transition-all duration-300 ${
              selectedMechanic === 'all'
                ? 'bg-gray-700 border-gray-500 text-white'
                : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
            }`}
          >
            All
          </button>
          {mechanics.map(mech => {
            const isActive = selectedMechanic === mech.id;
            const colorClasses = {
              blue: 'bg-blue-900/30 border-blue-700 text-blue-400',
              purple: 'bg-purple-900/30 border-purple-700 text-purple-400',
              cyan: 'bg-cyan-900/30 border-cyan-700 text-cyan-400'
            };
            return (
              <button
                key={mech.id}
                onClick={() => setSelectedMechanic(mech.id)}
                className={`px-4 py-2 rounded-lg border-2 transition-all duration-300 ${
                  isActive
                    ? colorClasses[mech.color as keyof typeof colorClasses]
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                {mech.name}
              </button>
            );
          })}
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Comprehensive Comparison</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-gray-700">
                <th className="text-left p-3 text-gray-400 font-semibold">Aspect</th>
                {(selectedMechanic === 'all' || selectedMechanic === 'classical') && (
                  <th className="text-left p-3 text-blue-400 font-semibold">Classical Mechanics</th>
                )}
                {(selectedMechanic === 'all' || selectedMechanic === 'quantum') && (
                  <th className="text-left p-3 text-purple-400 font-semibold">Quantum Mechanics</th>
                )}
                {(selectedMechanic === 'all' || selectedMechanic === 'fractal') && (
                  <th className="text-left p-3 text-cyan-400 font-semibold">Fractal Mechanics</th>
                )}
              </tr>
            </thead>
            <tbody>
              {comparisonData.map((row, idx) => (
                <tr key={idx} className="border-b border-gray-800 hover:bg-gray-800/30">
                  <td className="p-3 font-semibold text-gray-300">{row.aspect}</td>
                  {(selectedMechanic === 'all' || selectedMechanic === 'classical') && (
                    <td className="p-3 text-gray-300">{row.classical}</td>
                  )}
                  {(selectedMechanic === 'all' || selectedMechanic === 'quantum') && (
                    <td className="p-3 text-gray-300">{row.quantum}</td>
                  )}
                  {(selectedMechanic === 'all' || selectedMechanic === 'fractal') && (
                    <td className="p-3 text-gray-300">{row.fractal}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scale Symmetry */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <TrendingUp className="mr-2 text-cyan-400" size={24} />
          Scale Symmetry
        </h3>
        
        <p className="text-gray-300 text-sm mb-4 leading-relaxed">
          Scale symmetry refers to how physical laws behave under changes of scale. Fractal mechanics 
          uniquely maintains perfect scale symmetry through self-similar geometric patterns.
        </p>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-3">Classical</h4>
            <p className="text-gray-300 text-xs leading-relaxed">{scaleSymmetry.classical}</p>
            <div className="mt-3">
              <svg width="200" height="100" viewBox="0 0 200 100">
                <line x1="20" y1="50" x2="180" y2="50" stroke="blue" strokeWidth="2" />
                <line x1="150" y1="30" x2="150" y2="70" stroke="red" strokeWidth="3" />
                <text x="155" y="25" fill="red" fontSize="10">Breaks</text>
                <text x="10" y="70" fill="blue" fontSize="9">Macro</text>
                <text x="120" y="70" fill="blue" fontSize="9">Micro</text>
              </svg>
            </div>
          </div>

          <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-3">Quantum</h4>
            <p className="text-gray-300 text-xs leading-relaxed">{scaleSymmetry.quantum}</p>
            <div className="mt-3">
              <svg width="200" height="100" viewBox="0 0 200 100">
                <line x1="20" y1="50" x2="180" y2="50" stroke="purple" strokeWidth="2" />
                <line x1="40" y1="30" x2="40" y2="70" stroke="red" strokeWidth="3" />
                <text x="45" y="25" fill="red" fontSize="10">Planck</text>
                <text x="60" y="70" fill="purple" fontSize="9">Quantum</text>
                <text x="10" y="70" fill="gray" fontSize="9">?</text>
              </svg>
            </div>
          </div>

          <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-3">Fractal</h4>
            <p className="text-gray-300 text-xs leading-relaxed">{scaleSymmetry.fractal}</p>
            <div className="mt-3">
              <svg width="200" height="100" viewBox="0 0 200 100">
                <line x1="20" y1="50" x2="180" y2="50" stroke="cyan" strokeWidth="2" />
                {[40, 80, 120, 160].map(x => (
                  <g key={x}>
                    <circle cx={x} cy="50" r="3" fill="cyan" />
                    <path d={`M ${x-8} 35 L ${x} 50 L ${x+8} 35`} fill="none" stroke="cyan" strokeWidth="1" />
                  </g>
                ))}
                <text x="60" y="70" fill="cyan" fontSize="9">All scales</text>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Scale Relativity */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Scale Relativity</h3>
        
        <p className="text-gray-300 text-sm mb-4 leading-relaxed">
          Scale relativity examines how measurements and physical laws depend on the observational scale. 
          In fractal mechanics, patterns are scale-relative but self-similar.
        </p>

        <div className="space-y-4">
          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-2 text-sm">Classical: {scaleRelativity.classical}</h4>
            <p className="text-gray-300 text-xs">
              Measurements are absolute. A meter is a meter regardless of context. Time flows uniformly.
            </p>
          </div>

          <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-2 text-sm">Quantum: {scaleRelativity.quantum}</h4>
            <p className="text-gray-300 text-xs">
              Position and momentum cannot both be precisely measured. There's a fundamental limit to 
              how much we can know about a system at small scales.
            </p>
            <div className="mt-2 bg-gray-900/50 p-2 rounded">
              <p className="font-mono text-xs text-purple-300">ΔxΔp ≥ ℏ/2 ≈ 5.3 × 10⁻³⁵ J·s</p>
            </div>
          </div>

          <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Fractal: {scaleRelativity.fractal}</h4>
            <p className="text-gray-300 text-xs mb-2">
              Patterns repeat at every scale with geometric relationships preserved. What you observe 
              depends on your scale of observation, but the underlying geometric structure remains invariant.
            </p>
            <div className="flex justify-center">
              <svg width="400" height="150" viewBox="0 0 400 150">
                {/* Draw fractal-like pattern at 3 scales */}
                {[0, 1, 2].map(scale => {
                  const size = 100 / Math.pow(2, scale);
                  const x = 50 + scale * 130;
                  return (
                    <g key={scale}>
                      <circle cx={x} cy="75" r={size/2} fill="none" stroke="cyan" strokeWidth="1.5" />
                      <circle cx={x-size/4} cy="75" r={size/4} fill="none" stroke="cyan" strokeWidth="1" />
                      <circle cx={x+size/4} cy="75" r={size/4} fill="none" stroke="cyan" strokeWidth="1" />
                      <text x={x} y="130" textAnchor="middle" fill="cyan" fontSize="10">
                        Scale {scale + 1}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Key Insights */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
          <div className="flex items-start space-x-2">
            <Info size={18} className="text-cyan-400 mt-1 flex-shrink-0" />
            <div className="text-sm text-gray-300">
              <strong className="text-cyan-400">Unification:</strong> Fractal mechanics doesn't replace 
              classical or quantum mechanics but provides a geometric framework that encompasses both. 
              Classical mechanics emerges at large scales, quantum at small scales, and fractal mechanics 
              describes the transition between them.
            </div>
          </div>
        </div>

        <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
          <div className="flex items-start space-x-2">
            <Info size={18} className="text-purple-400 mt-1 flex-shrink-0" />
            <div className="text-sm text-gray-300">
              <strong className="text-purple-400">Consciousness Bridge:</strong> The scale-invariant 
              nature of fractal mechanics makes it ideal for modeling consciousness, which operates 
              coherently across vastly different scales - from neural oscillations (ms) to cognitive 
              processes (seconds) to personality traits (lifetime).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
