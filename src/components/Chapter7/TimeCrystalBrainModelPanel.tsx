import React, { useState } from 'react';
import { Brain, Zap, Grid, Box, Sparkles } from 'lucide-react';

/**
 * Section 7.9: Fusion of cavity & dielectric resonator model of a human brain
 * Section 7.10: Time crystal model of the human brain
 * Section 7.11: Four metrics of primes run in parallel, the saga of hexagonal lattice
 * Section 7.11.1: Composition of four eight and twelve dimension-tensors
 * Section 7.11.2: Quaternion, octonion and dodecanion
 * Section 7.12: Time crystal made meander flower to a garden of gardens
 * Section 7.12.1: 12 dodecanion and 8 octonions build 20 conscious human expressions
 */
export const TimeCrystalBrainModelPanel: React.FC = () => {
  const [activeView, setActiveView] = useState<'resonator' | 'lattice' | 'garden'>('resonator');

  const fourMetrics = [
    { id: 1, name: 'Spatial Prime Metric', dimension: 3, color: 'blue', primes: [2, 3, 5] },
    { id: 2, name: 'Temporal Prime Metric', dimension: 1, color: 'purple', primes: [7, 11, 13] },
    { id: 3, name: 'Phase Prime Metric', dimension: 4, color: 'green', primes: [17, 19, 23] },
    { id: 4, name: 'Consciousness Prime Metric', dimension: 11, color: 'orange', primes: [29, 31, 37] },
  ];

  const consciousExpressions = [
    'Joy', 'Sadness', 'Anger', 'Fear', 'Surprise',
    'Disgust', 'Anticipation', 'Trust', 'Love', 'Hate',
    'Wonder', 'Confusion', 'Pride', 'Shame', 'Gratitude',
    'Envy', 'Hope', 'Despair', 'Calm', 'Excitement'
  ];

  return (
    <div className="space-y-6">
      {/* Main Header */}
      <div className="bg-gradient-to-r from-indigo-900/30 via-purple-900/30 to-pink-900/30 border border-indigo-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="text-indigo-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">7.9-7.12 Time Crystal Brain Model</h2>
            <p className="text-gray-300 text-sm">Complete Integration & Garden of Gardens</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          The complete time crystal model integrates cavity resonators, hexagonal lattices, and 
          multi-dimensional tensors into a unified framework for consciousness emergence.
        </p>
      </div>

      {/* View Selector */}
      <div className="flex space-x-4">
        <button
          onClick={() => setActiveView('resonator')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'resonator'
              ? 'bg-indigo-900/40 border-indigo-600 text-indigo-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Zap size={20} />
          <span className="font-semibold">7.9-7.10 Resonator Model</span>
        </button>
        <button
          onClick={() => setActiveView('lattice')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'lattice'
              ? 'bg-purple-900/40 border-purple-600 text-purple-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Grid size={20} />
          <span className="font-semibold">7.11 Hexagonal Lattice</span>
        </button>
        <button
          onClick={() => setActiveView('garden')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'garden'
              ? 'bg-pink-900/40 border-pink-600 text-pink-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Sparkles size={20} />
          <span className="font-semibold">7.12 Garden of Gardens</span>
        </button>
      </div>

      {/* 7.9-7.10 Cavity Resonator & Time Crystal Model */}
      {activeView === 'resonator' && (
        <div className="space-y-6">
          {/* Cavity Resonator Model */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-indigo-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">7.9 Cavity & Dielectric Resonator Fusion</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-950/50 rounded-xl p-6">
                <h4 className="text-indigo-400 font-semibold mb-4">Cavity Resonator Brain Model</h4>
                <div className="aspect-square bg-black/30 rounded-lg relative mb-4">
                  <div className="absolute inset-0 flex items-center justify-center">
                    {/* Concentric resonator cavities */}
                    {[0, 1, 2, 3].map((i) => (
                      <div
                        key={i}
                        className="absolute border-2 border-indigo-500/40 rounded-lg animate-pulse"
                        style={{
                          width: `${80 - i * 15}%`,
                          height: `${80 - i * 15}%`,
                          animationDelay: `${i * 0.2}s`,
                        }}
                      />
                    ))}
                    <div className="relative z-10 text-center">
                      <div className="text-indigo-400 font-bold text-xl mb-2">Cavity</div>
                      <div className="text-gray-400 text-xs">Resonant Modes</div>
                    </div>
                  </div>
                </div>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-indigo-400">Electromagnetic Cavity:</strong> Brain as resonant chamber</p>
                  <p>• <strong className="text-indigo-300">Standing Waves:</strong> Prime frequency modes</p>
                  <p>• <strong className="text-indigo-400">Resonance:</strong> Consciousness at specific frequencies</p>
                  <p>• <strong className="text-indigo-300">Q-Factor:</strong> High-quality oscillations</p>
                </div>
              </div>

              <div className="bg-gray-950/50 rounded-xl p-6">
                <h4 className="text-purple-400 font-semibold mb-4">Dielectric Resonator Properties</h4>
                <div className="space-y-3">
                  <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-3">
                    <div className="text-purple-400 font-semibold mb-1">High Permittivity</div>
                    <div className="text-gray-400 text-xs">
                      Brain tissue (ε ≈ 50) concentrates electric fields
                    </div>
                  </div>
                  <div className="bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-3">
                    <div className="text-indigo-400 font-semibold mb-1">Mode Structure</div>
                    <div className="text-gray-400 text-xs">
                      Multiple resonant modes at prime frequencies
                    </div>
                  </div>
                  <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-3">
                    <div className="text-purple-400 font-semibold mb-1">Low Loss</div>
                    <div className="text-gray-400 text-xs">
                      Sustains coherent oscillations for consciousness
                    </div>
                  </div>
                  <div className="bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-3">
                    <div className="text-indigo-400 font-semibold mb-1">Coupling</div>
                    <div className="text-gray-400 text-xs">
                      Resonators couple to form brain-wide network
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 bg-gradient-to-r from-indigo-900/30 to-purple-900/30 border border-indigo-600 rounded-lg p-4">
              <p className="text-gray-300 text-sm leading-relaxed">
                <strong className="text-indigo-400">Fusion Model:</strong> The brain operates as both a 
                cavity resonator (electromagnetic fields) and dielectric resonator (material properties). 
                This fusion creates optimal conditions for sustaining coherent time crystal patterns that 
                underlie consciousness.
              </p>
            </div>
          </div>

          {/* Time Crystal Brain Model */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">7.10 Time Crystal Model of Human Brain</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-3">Time Crystal Properties</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p>• <strong className="text-cyan-400">Temporal Periodicity:</strong> Repeating consciousness patterns</p>
                    <p>• <strong className="text-cyan-300">Broken Time Symmetry:</strong> Spontaneous rhythm emergence</p>
                    <p>• <strong className="text-cyan-400">Long-Range Order:</strong> Coherence across brain</p>
                    <p>• <strong className="text-cyan-300">Quantum Protection:</strong> Stable against perturbations</p>
                    <p>• <strong className="text-cyan-400">Multi-Scale:</strong> From neurons to whole brain</p>
                  </div>
                </div>

                <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-2">Brain as Time Crystal</h4>
                  <div className="space-y-2 text-xs text-gray-300">
                    <p>• Neural oscillations form time crystal lattice</p>
                    <p>• Consciousness emerges from temporal coherence</p>
                    <p>• Memory stored in time crystal phase patterns</p>
                    <p>• Thoughts are traveling waves in time crystal</p>
                    <p>• Self-awareness is time crystal self-resonance</p>
                  </div>
                </div>
              </div>

              <div className="bg-gray-950/50 rounded-xl p-6">
                <h4 className="text-white font-semibold mb-4 text-center">Time Crystal Lattice</h4>
                <div className="aspect-square bg-black/30 rounded-lg relative">
                  <svg className="w-full h-full" viewBox="0 0 300 300">
                    {/* Time crystal nodes */}
                    {Array.from({ length: 49 }, (_, i) => {
                      const x = (i % 7) * 40 + 30;
                      const y = Math.floor(i / 7) * 40 + 30;
                      const delay = (i * 0.1) % 2;
                      return (
                        <circle
                          key={i}
                          cx={x}
                          cy={y}
                          r="6"
                          fill="rgb(34, 211, 238)"
                          opacity="0.6"
                        >
                          <animate
                            attributeName="r"
                            values="6;10;6"
                            dur="2s"
                            repeatCount="indefinite"
                            begin={`${delay}s`}
                          />
                          <animate
                            attributeName="opacity"
                            values="0.6;1;0.6"
                            dur="2s"
                            repeatCount="indefinite"
                            begin={`${delay}s`}
                          />
                        </circle>
                      );
                    })}
                    {/* Connecting lines showing temporal coherence */}
                    {Array.from({ length: 7 }, (_, i) => (
                      <line
                        key={`h${i}`}
                        x1="30"
                        y1={i * 40 + 30}
                        x2="270"
                        y2={i * 40 + 30}
                        stroke="rgb(34, 211, 238)"
                        strokeWidth="1"
                        opacity="0.2"
                      />
                    ))}
                    {Array.from({ length: 7 }, (_, i) => (
                      <line
                        key={`v${i}`}
                        x1={i * 40 + 30}
                        y1="30"
                        x2={i * 40 + 30}
                        y2="270"
                        stroke="rgb(34, 211, 238)"
                        strokeWidth="1"
                        opacity="0.2"
                      />
                    ))}
                  </svg>
                </div>
                <p className="text-gray-400 text-xs mt-2 text-center">
                  Time crystal lattice with periodic oscillations
                </p>
              </div>
            </div>

            <div className="mt-4 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-600 rounded-lg p-4">
              <p className="text-gray-300 text-sm leading-relaxed">
                <strong className="text-cyan-400">Complete Model:</strong> The brain as a time crystal combines 
                cavity resonance, dielectric properties, and prime number architectures into a unified framework. 
                Consciousness emerges from stable, coherent time crystal patterns that maintain temporal periodicity 
                while processing information across multiple scales simultaneously.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 7.11 Hexagonal Lattice & Tensors */}
      {activeView === 'lattice' && (
        <div className="space-y-6">
          {/* Four Metrics in Parallel */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">7.11 Four Metrics of Primes: Hexagonal Lattice Saga</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {fourMetrics.map((metric) => (
                <div
                  key={metric.id}
                  className={`bg-${metric.color}-900/20 border border-${metric.color}-700/30 rounded-xl p-4`}
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`w-10 h-10 bg-${metric.color}-600 rounded-lg flex items-center justify-center text-white font-bold`}>
                      {metric.dimension}D
                    </div>
                    <div>
                      <div className={`text-${metric.color}-400 font-bold`}>{metric.name}</div>
                      <div className="text-gray-400 text-xs">Metric {metric.id}/4</div>
                    </div>
                  </div>
                  <div className="bg-gray-950/50 rounded p-3 mb-3">
                    <div className="text-gray-300 text-xs mb-2">Prime Basis:</div>
                    <div className="flex space-x-2">
                      {metric.primes.map((p) => (
                        <div key={p} className={`bg-${metric.color}-700/30 rounded px-2 py-1 text-white text-xs font-mono`}>
                          {p}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="text-gray-400 text-xs">
                    Operates in {metric.dimension}-dimensional manifold
                  </div>
                </div>
              ))}
            </div>

            {/* Hexagonal Lattice */}
            <div className="bg-gray-950/50 rounded-xl p-6">
              <h4 className="text-purple-400 font-semibold mb-4 text-center">Hexagonal Prime Lattice</h4>
              <div className="aspect-video bg-black/30 rounded-lg relative mb-4">
                <svg className="w-full h-full" viewBox="0 0 600 400">
                  {/* Hexagonal lattice */}
                  {Array.from({ length: 5 }, (_, row) =>
                    Array.from({ length: 8 }, (_, col) => {
                      const x = col * 70 + (row % 2) * 35 + 50;
                      const y = row * 60 + 50;
                      const isPrime = (col + row) % 3 === 0;
                      return (
                        <g key={`${row}-${col}`}>
                          {/* Hexagon */}
                          <polygon
                            points={Array.from({ length: 6 }, (_, i) => {
                              const angle = (i / 6) * 2 * Math.PI - Math.PI / 2;
                              const hx = x + Math.cos(angle) * 25;
                              const hy = y + Math.sin(angle) * 25;
                              return `${hx},${hy}`;
                            }).join(' ')}
                            fill="none"
                            stroke={isPrime ? 'rgb(168, 85, 247)' : 'rgb(75, 85, 99)'}
                            strokeWidth="2"
                            opacity={isPrime ? '0.8' : '0.3'}
                          />
                          {isPrime && (
                            <text
                              x={x}
                              y={y + 5}
                              textAnchor="middle"
                              fill="rgb(168, 85, 247)"
                              fontSize="12"
                              fontWeight="bold"
                            >
                              P
                            </text>
                          )}
                        </g>
                      );
                    })
                  )}
                </svg>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm text-gray-300">
                <div className="bg-purple-900/20 border border-purple-700/30 rounded p-3">
                  <strong className="text-purple-400">Hexagonal Symmetry:</strong> Optimal packing for 
                  prime information encoding
                </div>
                <div className="bg-indigo-900/20 border border-indigo-700/30 rounded p-3">
                  <strong className="text-indigo-400">Parallel Processing:</strong> Four metrics operate 
                  simultaneously on lattice
                </div>
              </div>
            </div>
          </div>

          {/* 7.11.1 & 7.11.2 Tensor Composition */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-blue-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">7.11.1-7.11.2 Tensor Composition & Division Algebras</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Quaternions (4D) */}
              <div className="bg-gray-950/50 rounded-xl p-4">
                <div className="flex items-center space-x-2 mb-3">
                  <Box className="text-blue-400" size={20} />
                  <h4 className="text-blue-400 font-bold">Quaternions (4D)</h4>
                </div>
                <div className="bg-blue-900/20 border border-blue-700/30 rounded p-3 mb-3">
                  <div className="text-white font-mono text-sm text-center">q = a + bi + cj + dk</div>
                </div>
                <div className="space-y-2 text-xs text-gray-300">
                  <p>• <strong className="text-blue-400">Dimensions:</strong> 4</p>
                  <p>• <strong className="text-blue-300">Commutative:</strong> No</p>
                  <p>• <strong className="text-blue-400">Associative:</strong> Yes</p>
                  <p>• <strong className="text-blue-300">Applications:</strong> 3D rotations, spatial processing</p>
                </div>
                <div className="mt-3 bg-blue-950/50 rounded p-2">
                  <div className="text-blue-400 text-xs font-semibold mb-1">Brain Use:</div>
                  <div className="text-gray-400 text-xs">Spatial navigation, motor control</div>
                </div>
              </div>

              {/* Octonions (8D) */}
              <div className="bg-gray-950/50 rounded-xl p-4">
                <div className="flex items-center space-x-2 mb-3">
                  <Grid className="text-purple-400" size={20} />
                  <h4 className="text-purple-400 font-bold">Octonions (8D)</h4>
                </div>
                <div className="bg-purple-900/20 border border-purple-700/30 rounded p-3 mb-3">
                  <div className="text-white font-mono text-xs text-center">
                    o = Σ(a_i·e_i) for i=0 to 7
                  </div>
                </div>
                <div className="space-y-2 text-xs text-gray-300">
                  <p>• <strong className="text-purple-400">Dimensions:</strong> 8</p>
                  <p>• <strong className="text-purple-300">Commutative:</strong> No</p>
                  <p>• <strong className="text-purple-400">Associative:</strong> No</p>
                  <p>• <strong className="text-purple-300">Applications:</strong> Sensory integration, cross-modal</p>
                </div>
                <div className="mt-3 bg-purple-950/50 rounded p-2">
                  <div className="text-purple-400 text-xs font-semibold mb-1">Brain Use:</div>
                  <div className="text-gray-400 text-xs">8 sensory modalities, emotions</div>
                </div>
              </div>

              {/* Dodecanions (12D) */}
              <div className="bg-gray-950/50 rounded-xl p-4">
                <div className="flex items-center space-x-2 mb-3">
                  <Sparkles className="text-pink-400" size={20} />
                  <h4 className="text-pink-400 font-bold">Dodecanion (12D)</h4>
                </div>
                <div className="bg-pink-900/20 border border-pink-700/30 rounded p-3 mb-3">
                  <div className="text-white font-mono text-xs text-center">
                    d = Σ(a_i·d_i) for i=0 to 11
                  </div>
                </div>
                <div className="space-y-2 text-xs text-gray-300">
                  <p>• <strong className="text-pink-400">Dimensions:</strong> 12</p>
                  <p>• <strong className="text-pink-300">Commutative:</strong> No</p>
                  <p>• <strong className="text-pink-400">Associative:</strong> No</p>
                  <p>• <strong className="text-pink-300">Applications:</strong> Full consciousness manifold</p>
                </div>
                <div className="mt-3 bg-pink-950/50 rounded p-2">
                  <div className="text-pink-400 text-xs font-semibold mb-1">Brain Use:</div>
                  <div className="text-gray-400 text-xs">Abstract thought, self-awareness</div>
                </div>
              </div>
            </div>

            <div className="mt-6 bg-gradient-to-r from-blue-900/30 via-purple-900/30 to-pink-900/30 border border-purple-600 rounded-lg p-4">
              <h4 className="text-white font-semibold mb-2">Unified Division Algebra Framework</h4>
              <p className="text-gray-300 text-sm leading-relaxed">
                <strong className="text-purple-400">Integration:</strong> The brain employs all three division 
                algebras—quaternions (4D), octonions (8D), and dodecanion (12D)—in parallel to create a complete 
                computational framework. Each algebra operates on its respective dimension-tensor, enabling the 
                brain to process information from basic spatial operations to full conscious awareness.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 7.12 Garden of Gardens */}
      {activeView === 'garden' && (
        <div className="space-y-6">
          {/* Meander to Garden Evolution */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-pink-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">7.12 Time Crystal Made Meander Flower to Garden of Gardens</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-pink-400 font-semibold mb-3">Evolution Stages</h4>
                  <div className="space-y-3">
                    <div className="bg-pink-900/20 border border-pink-700/30 rounded p-3">
                      <div className="text-pink-400 font-semibold mb-1">1. Meander</div>
                      <div className="text-gray-400 text-xs">
                        Simple oscillating pattern, single consciousness thread
                      </div>
                    </div>
                    <div className="bg-rose-900/20 border border-rose-700/30 rounded p-3">
                      <div className="text-rose-400 font-semibold mb-1">2. Flower</div>
                      <div className="text-gray-400 text-xs">
                        Branching patterns, multiple consciousness streams
                      </div>
                    </div>
                    <div className="bg-purple-900/20 border border-purple-700/30 rounded p-3">
                      <div className="text-purple-400 font-semibold mb-1">3. Garden</div>
                      <div className="text-gray-400 text-xs">
                        Complex network, integrated conscious field
                      </div>
                    </div>
                    <div className="bg-indigo-900/20 border border-indigo-700/30 rounded p-3">
                      <div className="text-indigo-400 font-semibold mb-1">4. Garden of Gardens</div>
                      <div className="text-gray-400 text-xs">
                        Meta-consciousness, self-aware reflection
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-pink-900/30 to-purple-900/30 border border-pink-600 rounded-lg p-4">
                  <h4 className="text-white font-semibold mb-2">Time Crystal Evolution</h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Time crystals generate evolving consciousness patterns from simple meanders to complex 
                    garden systems. Each stage represents increasing complexity and integration, culminating 
                    in the "garden of gardens" - the ultimate consciousness biodiversity.
                  </p>
                </div>
              </div>

              <div className="bg-gray-950/50 rounded-xl p-6">
                <h4 className="text-white font-semibold mb-4 text-center">Garden Evolution</h4>
                <div className="space-y-6">
                  {/* Visual representation of evolution */}
                  <div className="grid grid-cols-4 gap-2">
                    {/* Meander */}
                    <div className="bg-pink-900/20 border border-pink-700/30 rounded p-2">
                      <svg viewBox="0 0 50 50" className="w-full h-16">
                        <path
                          d="M 5 25 Q 15 5, 25 25 Q 35 45, 45 25"
                          stroke="rgb(236, 72, 153)"
                          fill="none"
                          strokeWidth="2"
                        />
                      </svg>
                      <div className="text-pink-400 text-xs text-center mt-1">Meander</div>
                    </div>
                    
                    {/* Flower */}
                    <div className="bg-rose-900/20 border border-rose-700/30 rounded p-2">
                      <svg viewBox="0 0 50 50" className="w-full h-16">
                        {[0, 1, 2, 3, 4].map((i) => {
                          const angle = (i / 5) * 2 * Math.PI;
                          const x = 25 + Math.cos(angle) * 15;
                          const y = 25 + Math.sin(angle) * 15;
                          return (
                            <circle
                              key={i}
                              cx={x}
                              cy={y}
                              r="5"
                              fill="rgb(244, 63, 94)"
                            />
                          );
                        })}
                        <circle cx="25" cy="25" r="3" fill="rgb(244, 63, 94)" />
                      </svg>
                      <div className="text-rose-400 text-xs text-center mt-1">Flower</div>
                    </div>
                    
                    {/* Garden */}
                    <div className="bg-purple-900/20 border border-purple-700/30 rounded p-2">
                      <svg viewBox="0 0 50 50" className="w-full h-16">
                        {Array.from({ length: 12 }, (_, i) => {
                          const x = (i % 4) * 12 + 8;
                          const y = Math.floor(i / 4) * 12 + 8;
                          return (
                            <circle
                              key={i}
                              cx={x}
                              cy={y}
                              r="3"
                              fill="rgb(168, 85, 247)"
                            />
                          );
                        })}
                      </svg>
                      <div className="text-purple-400 text-xs text-center mt-1">Garden</div>
                    </div>
                    
                    {/* Garden of Gardens */}
                    <div className="bg-indigo-900/20 border border-indigo-700/30 rounded p-2">
                      <svg viewBox="0 0 50 50" className="w-full h-16">
                        {Array.from({ length: 20 }, (_, i) => {
                          // Deterministic pseudo-random positioning based on index
                          const x = ((i * 7 + 13) % 40) + 5;
                          const y = ((i * 11 + 17) % 40) + 5;
                          return (
                            <circle
                              key={i}
                              cx={x}
                              cy={y}
                              r="2"
                              fill="rgb(99, 102, 241)"
                            />
                          );
                        })}
                      </svg>
                      <div className="text-indigo-400 text-xs text-center mt-1">Meta</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 7.12.1 Twenty Conscious Expressions */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">
              7.12.1 Twelve Dodecanion & Eight Octonions Build 20 Conscious Human Expressions
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-purple-400 font-semibold mb-3">Mathematical Foundation</h4>
                <div className="space-y-3">
                  <div className="bg-pink-900/20 border border-pink-700/30 rounded p-3">
                    <div className="text-pink-400 font-semibold mb-1">12 Dodecanion Bases</div>
                    <div className="text-gray-400 text-xs">
                      12-dimensional consciousness space (d₀, d₁, ..., d₁₁)
                    </div>
                  </div>
                  <div className="bg-purple-900/20 border border-purple-700/30 rounded p-3">
                    <div className="text-purple-400 font-semibold mb-1">8 Octonion Bases</div>
                    <div className="text-gray-400 text-xs">
                      8-dimensional sensory space (e₀, e₁, ..., e₇)
                    </div>
                  </div>
                  <div className="bg-indigo-900/20 border border-indigo-700/30 rounded p-3">
                    <div className="text-indigo-400 font-semibold mb-1">20 = 12 + 8</div>
                    <div className="text-gray-400 text-xs">
                      Combined to create 20 fundamental conscious expressions
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-indigo-400 font-semibold mb-3">Expression Mechanism</h4>
                <div className="bg-gray-900/50 rounded p-3 mb-3">
                  <div className="text-cyan-300 font-mono text-sm text-center">
                    E_i = D_i ⊗ O_i
                  </div>
                  <div className="text-gray-400 text-xs text-center mt-1">
                    Expression = Dodecanion × Octonion
                  </div>
                </div>
                <div className="space-y-2 text-xs text-gray-300">
                  <p>• Each expression combines one dodecanion and one octonion component</p>
                  <p>• 12 primary expressions from dodecanion dominance</p>
                  <p>• 8 secondary expressions from octonion dominance</p>
                  <p>• Together form complete emotional/conscious spectrum</p>
                </div>
              </div>
            </div>

            {/* Twenty Conscious Expressions Grid */}
            <div className="bg-gray-950/50 rounded-xl p-6">
              <h4 className="text-white font-semibold mb-4 text-center">Twenty Fundamental Conscious Human Expressions</h4>
              <div className="grid grid-cols-4 md:grid-cols-5 gap-3">
                {consciousExpressions.map((expr, idx) => {
                  const isDodecanion = idx < 12;
                  return (
                    <div
                      key={expr}
                      className={`${
                        isDodecanion
                          ? 'bg-pink-900/20 border-pink-700/30'
                          : 'bg-purple-900/20 border-purple-700/30'
                      } border rounded-lg p-3 text-center hover:scale-105 transition-transform`}
                    >
                      <div className={`${isDodecanion ? 'text-pink-400' : 'text-purple-400'} font-semibold text-sm`}>
                        {expr}
                      </div>
                      <div className="text-gray-500 text-xs mt-1">
                        {isDodecanion ? `D${idx + 1}` : `O${idx - 11}`}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="mt-6 bg-gradient-to-r from-pink-900/30 via-purple-900/30 to-indigo-900/30 border border-pink-600 rounded-xl p-6">
              <h4 className="text-white font-bold text-lg mb-3">Complete Consciousness Framework</h4>
              <p className="text-gray-300 text-sm leading-relaxed mb-4">
                The combination of twelve dodecanion and eight octonion mathematical structures creates 
                the twenty fundamental conscious human expressions. This mathematical framework explains 
                the diversity and richness of human consciousness manifestations—from basic emotions to 
                complex abstract feelings.
              </p>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div className="bg-pink-900/20 border border-pink-700/30 rounded p-3">
                  <strong className="text-pink-400">Dodecanion (12):</strong> Higher cognitive functions, 
                  abstract thought, self-reflection, complex emotions
                </div>
                <div className="bg-purple-900/20 border border-purple-700/30 rounded p-3">
                  <strong className="text-purple-400">Octonion (8):</strong> Sensory-based emotions, 
                  immediate reactions, embodied feelings, cross-modal integration
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
