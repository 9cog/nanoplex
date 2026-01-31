import React, { useState } from 'react';
import { Circle, Hexagon, Zap } from 'lucide-react';

/**
 * Section 7.7: Brain's wheel of primes
 * Section 7.7.1: Eight sensors hold cross-over magic of octonion
 * Section 7.7.2: Eight mathematical operators run two orthogonal math engines
 * Section 7.8: The basic device for decision making in the brain
 * Section 7.8.1: H3 device brain uses everywhere
 */
export const BrainWheelDecisionPanel: React.FC = () => {
  const [activeView, setActiveView] = useState<'wheel' | 'operators' | 'decision'>('wheel');

  const eightSensors = [
    { id: 1, name: 'Visual', angle: 0, color: 'blue', prime: 2 },
    { id: 2, name: 'Auditory', angle: 45, color: 'purple', prime: 3 },
    { id: 3, name: 'Tactile', angle: 90, color: 'green', prime: 5 },
    { id: 4, name: 'Olfactory', angle: 135, color: 'pink', prime: 7 },
    { id: 5, name: 'Gustatory', angle: 180, color: 'orange', prime: 11 },
    { id: 6, name: 'Vestibular', angle: 225, color: 'cyan', prime: 13 },
    { id: 7, name: 'Proprioceptive', angle: 270, color: 'indigo', prime: 17 },
    { id: 8, name: 'Interoceptive', angle: 315, color: 'rose', prime: 19 },
  ];

  const eightOperators = [
    { id: 1, name: 'Addition (+)', engine: 'A', color: 'blue' },
    { id: 2, name: 'Subtraction (−)', engine: 'A', color: 'purple' },
    { id: 3, name: 'Multiplication (×)', engine: 'A', color: 'green' },
    { id: 4, name: 'Division (÷)', engine: 'A', color: 'cyan' },
    { id: 5, name: 'Integration (∫)', engine: 'B', color: 'orange' },
    { id: 6, name: 'Differentiation (d/dx)', engine: 'B', color: 'pink' },
    { id: 7, name: 'Convolution (⊗)', engine: 'B', color: 'indigo' },
    { id: 8, name: 'Transformation (T)', engine: 'B', color: 'rose' },
  ];

  return (
    <div className="space-y-6">
      {/* Main Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 via-blue-900/30 to-purple-900/30 border border-cyan-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Circle className="text-cyan-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">7.7-7.8 Brain Wheel & Decision Systems</h2>
            <p className="text-gray-300 text-sm">Octonion Architecture & H3 Decision Device</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          The brain's wheel of primes integrates eight sensory modalities through octonion mathematics, 
          operating two orthogonal processing engines for optimal decision making via H3 devices.
        </p>
      </div>

      {/* View Selector */}
      <div className="flex space-x-4">
        <button
          onClick={() => setActiveView('wheel')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'wheel'
              ? 'bg-cyan-900/40 border-cyan-600 text-cyan-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Circle size={20} />
          <span className="font-semibold">7.7 & 7.7.1 Wheel & Sensors</span>
        </button>
        <button
          onClick={() => setActiveView('operators')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'operators'
              ? 'bg-purple-900/40 border-purple-600 text-purple-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Zap size={20} />
          <span className="font-semibold">7.7.2 Math Operators</span>
        </button>
        <button
          onClick={() => setActiveView('decision')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'decision'
              ? 'bg-orange-900/40 border-orange-600 text-orange-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Hexagon size={20} />
          <span className="font-semibold">7.8 & 7.8.1 H3 Decision</span>
        </button>
      </div>

      {/* 7.7 & 7.7.1 Brain's Wheel and Eight Sensors */}
      {activeView === 'wheel' && (
        <div className="space-y-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Brain's Wheel of Primes: Eight Sensors</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Wheel Visualization */}
              <div className="bg-gray-950/50 rounded-xl p-6">
                <h4 className="text-cyan-400 font-semibold mb-4 text-center">Octonion Sensor Wheel</h4>
                <div className="aspect-square bg-black/30 rounded-lg relative overflow-hidden">
                  <svg className="w-full h-full" viewBox="0 0 400 400">
                    {/* Center circle */}
                    <circle cx="200" cy="200" r="60" fill="none" stroke="rgb(34, 211, 238)" strokeWidth="2" />
                    <text x="200" y="205" textAnchor="middle" fill="rgb(34, 211, 238)" fontSize="16" fontWeight="bold">
                      Brain Core
                    </text>
                    
                    {/* Outer circle */}
                    <circle cx="200" cy="200" r="150" fill="none" stroke="rgb(34, 211, 238)" strokeWidth="2" opacity="0.3" />
                    
                    {/* Eight sensors positioned on the wheel */}
                    {eightSensors.map((sensor) => {
                      const angleRad = (sensor.angle * Math.PI) / 180;
                      const x = 200 + Math.cos(angleRad) * 120;
                      const y = 200 + Math.sin(angleRad) * 120;
                      const textX = 200 + Math.cos(angleRad) * 150;
                      const textY = 200 + Math.sin(angleRad) * 150;
                      
                      return (
                        <g key={sensor.id}>
                          {/* Line from center */}
                          <line
                            x1="200"
                            y1="200"
                            x2={x}
                            y2={y}
                            stroke="rgb(34, 211, 238)"
                            strokeWidth="1"
                            opacity="0.3"
                            strokeDasharray="5,5"
                          />
                          {/* Sensor node */}
                          <circle
                            cx={x}
                            cy={y}
                            r="20"
                            fill={`rgb(${sensor.color === 'blue' ? '59, 130, 246' : 
                                         sensor.color === 'purple' ? '168, 85, 247' : 
                                         sensor.color === 'green' ? '34, 197, 94' : 
                                         sensor.color === 'pink' ? '236, 72, 153' : 
                                         sensor.color === 'orange' ? '249, 115, 22' : 
                                         sensor.color === 'cyan' ? '6, 182, 212' : 
                                         sensor.color === 'indigo' ? '99, 102, 241' : '244, 63, 94'})`}
                            opacity="0.6"
                          >
                            <animate
                              attributeName="r"
                              values="20;25;20"
                              dur="2s"
                              repeatCount="indefinite"
                            />
                          </circle>
                          {/* Prime number */}
                          <text x={x} y={y + 5} textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                            {sensor.prime}
                          </text>
                          {/* Label */}
                          <text
                            x={textX}
                            y={textY}
                            textAnchor="middle"
                            fill="rgb(156, 163, 175)"
                            fontSize="12"
                          >
                            {sensor.name}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
                <p className="text-gray-400 text-xs mt-4 text-center">
                  Eight sensory modalities positioned in octonion geometry
                </p>
              </div>

              {/* Sensor Details */}
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-3">Octonion Cross-Over Magic</h4>
                  <p className="text-gray-300 text-sm mb-3">
                    Eight sensors form an octonion algebra structure, enabling non-commutative 
                    integration of sensory information. Each sensor maps to one octonion basis element.
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    {eightSensors.map((sensor) => (
                      <div
                        key={sensor.id}
                        className={`bg-${sensor.color}-900/20 border border-${sensor.color}-700/30 rounded p-2`}
                      >
                        <div className="flex items-center justify-between">
                          <span className={`text-${sensor.color}-400 text-sm font-semibold`}>
                            e{sensor.id - 1}
                          </span>
                          <span className="text-gray-400 text-xs">{sensor.name}</span>
                        </div>
                        <div className="text-gray-500 text-xs mt-1">Prime: {sensor.prime}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-3">Cross-Over Properties</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p>• <strong className="text-cyan-400">Non-Commutative:</strong> e₁ × e₂ ≠ e₂ × e₁</p>
                    <p>• <strong className="text-cyan-300">Non-Associative:</strong> (e₁ × e₂) × e₃ ≠ e₁ × (e₂ × e₃)</p>
                    <p>• <strong className="text-cyan-400">Alternative:</strong> (x × x) × y = x × (x × y)</p>
                    <p>• <strong className="text-cyan-300">Normed Division:</strong> |x × y| = |x| × |y|</p>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-600 rounded-lg p-4">
                  <h4 className="text-white font-semibold mb-2">Integration Magic</h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Octonion algebra enables the brain to integrate eight independent sensory streams 
                    while preserving their non-commutative relationships. This "cross-over magic" allows 
                    for rich multisensory experiences where order and context matter.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 7.7.2 Eight Mathematical Operators */}
      {activeView === 'operators' && (
        <div className="space-y-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Eight Mathematical Operators: Two Orthogonal Engines</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Engine A: Algebraic */}
              <div className="bg-gray-950/50 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
                    A
                  </div>
                  <div>
                    <h4 className="text-blue-400 font-bold text-lg">Engine A: Algebraic</h4>
                    <p className="text-gray-400 text-xs">Discrete Operations</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  {eightOperators.slice(0, 4).map((op, idx) => (
                    <div
                      key={op.id}
                      className={`bg-${op.color}-900/20 border border-${op.color}-700/30 rounded-lg p-3`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className={`text-${op.color}-400 font-semibold`}>{idx + 1}. {op.name}</span>
                        <span className="text-gray-500 text-xs">Engine {op.engine}</span>
                      </div>
                      <div className="text-gray-400 text-xs">
                        {idx === 0 && 'Combines signals: S₁ + S₂ → combined activation'}
                        {idx === 1 && 'Computes differences: S₁ − S₂ → contrast detection'}
                        {idx === 2 && 'Scales signals: S₁ × k → gain modulation'}
                        {idx === 3 && 'Normalizes: S₁ ÷ Σ(S) → relative activation'}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
                  <p className="text-gray-300 text-xs">
                    <strong className="text-blue-400">Function:</strong> Engine A handles discrete, 
                    instantaneous computations for rapid sensory processing and immediate responses.
                  </p>
                </div>
              </div>

              {/* Engine B: Analytical */}
              <div className="bg-gray-950/50 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-12 h-12 bg-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
                    B
                  </div>
                  <div>
                    <h4 className="text-purple-400 font-bold text-lg">Engine B: Analytical</h4>
                    <p className="text-gray-400 text-xs">Continuous Operations</p>
                  </div>
                </div>
                
                <div className="space-y-3">
                  {eightOperators.slice(4, 8).map((op, idx) => (
                    <div
                      key={op.id}
                      className={`bg-${op.color}-900/20 border border-${op.color}-700/30 rounded-lg p-3`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className={`text-${op.color}-400 font-semibold`}>{idx + 5}. {op.name}</span>
                        <span className="text-gray-500 text-xs">Engine {op.engine}</span>
                      </div>
                      <div className="text-gray-400 text-xs">
                        {idx === 0 && 'Accumulates over time: ∫S(t)dt → temporal integration'}
                        {idx === 1 && 'Computes change: dS/dt → velocity of state change'}
                        {idx === 2 && 'Combines patterns: S₁ ⊗ S₂ → feature binding'}
                        {idx === 3 && 'Coordinate transforms: T(S) → reference frame shifts'}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 bg-purple-900/20 border border-purple-700/30 rounded-lg p-3">
                  <p className="text-gray-300 text-xs">
                    <strong className="text-purple-400">Function:</strong> Engine B handles continuous, 
                    temporal computations for pattern recognition and predictive processing.
                  </p>
                </div>
              </div>
            </div>

            {/* Orthogonal Processing */}
            <div className="mt-6 bg-gradient-to-r from-blue-900/30 via-purple-900/30 to-indigo-900/30 border border-purple-600 rounded-xl p-6">
              <h4 className="text-white font-bold text-lg mb-3">Orthogonal Math Engines</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
                  <div className="text-blue-400 font-semibold mb-2">Engine A Properties</div>
                  <div className="space-y-1 text-xs text-gray-300">
                    <p>• Operates in discrete time steps</p>
                    <p>• Local, instantaneous operations</p>
                    <p>• Fast, reflexive processing</p>
                    <p>• Parallel across spatial locations</p>
                  </div>
                </div>
                <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-3">
                  <div className="text-purple-400 font-semibold mb-2">Engine B Properties</div>
                  <div className="space-y-1 text-xs text-gray-300">
                    <p>• Operates in continuous time</p>
                    <p>• Global, temporal operations</p>
                    <p>• Slow, deliberate processing</p>
                    <p>• Sequential across time points</p>
                  </div>
                </div>
              </div>
              <p className="text-gray-300 text-sm leading-relaxed">
                <strong className="text-cyan-400">Orthogonality:</strong> The two engines operate in orthogonal 
                computational spaces—Engine A in discrete space-parallel mode, Engine B in continuous time-serial 
                mode. This orthogonal architecture enables the brain to process information simultaneously across 
                both spatial and temporal dimensions, creating a complete computational framework for consciousness.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 7.8 & 7.8.1 H3 Decision Device */}
      {activeView === 'decision' && (
        <div className="space-y-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-orange-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">H3 Device: Universal Decision Making Unit</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* H3 Structure */}
              <div className="bg-gray-950/50 rounded-xl p-6">
                <h4 className="text-orange-400 font-semibold mb-4 text-center">H3 Triplet Architecture</h4>
                <div className="aspect-square bg-black/30 rounded-lg relative flex items-center justify-center">
                  <div className="relative w-64 h-64">
                    {/* Three H units in triangular formation */}
                    <div className="absolute top-0 left-1/2 transform -translate-x-1/2">
                      <div className="w-20 h-20 bg-gradient-to-br from-orange-600 to-red-600 rounded-lg flex items-center justify-center">
                        <div className="text-white font-bold text-2xl">H₁</div>
                      </div>
                      <div className="text-orange-400 text-xs text-center mt-2">Input A</div>
                    </div>
                    <div className="absolute bottom-0 left-0">
                      <div className="w-20 h-20 bg-gradient-to-br from-orange-600 to-amber-600 rounded-lg flex items-center justify-center">
                        <div className="text-white font-bold text-2xl">H₂</div>
                      </div>
                      <div className="text-orange-400 text-xs text-center mt-2">Input B</div>
                    </div>
                    <div className="absolute bottom-0 right-0">
                      <div className="w-20 h-20 bg-gradient-to-br from-orange-600 to-yellow-600 rounded-lg flex items-center justify-center">
                        <div className="text-white font-bold text-2xl">H₃</div>
                      </div>
                      <div className="text-orange-400 text-xs text-center mt-2">Context</div>
                    </div>
                    
                    {/* Connecting lines */}
                    <svg className="absolute inset-0" viewBox="0 0 256 256">
                      <line x1="128" y1="40" x2="40" y2="216" stroke="rgb(249, 115, 22)" strokeWidth="2" opacity="0.5" />
                      <line x1="128" y1="40" x2="216" y2="216" stroke="rgb(249, 115, 22)" strokeWidth="2" opacity="0.5" />
                      <line x1="40" y1="216" x2="216" y2="216" stroke="rgb(249, 115, 22)" strokeWidth="2" opacity="0.5" />
                    </svg>
                    
                    {/* Center decision point */}
                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                      <div className="w-16 h-16 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-full flex items-center justify-center animate-pulse">
                        <div className="text-white font-bold">D</div>
                      </div>
                      <div className="text-yellow-400 text-xs text-center mt-1">Decision</div>
                    </div>
                  </div>
                </div>
                <p className="text-gray-400 text-xs mt-4 text-center">
                  Three H units form decision triangle with central decision point
                </p>
              </div>

              {/* H3 Properties */}
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-orange-400 font-semibold mb-3">H3 Components</h4>
                  <div className="space-y-3">
                    <div className="bg-red-900/20 border border-red-700/30 rounded p-3">
                      <div className="text-red-400 font-semibold mb-1">H₁ (Input A)</div>
                      <div className="text-gray-400 text-xs">
                        Primary input stream: sensory data, current state information
                      </div>
                    </div>
                    <div className="bg-amber-900/20 border border-amber-700/30 rounded p-3">
                      <div className="text-amber-400 font-semibold mb-1">H₂ (Input B)</div>
                      <div className="text-gray-400 text-xs">
                        Secondary input stream: memory, predictions, alternatives
                      </div>
                    </div>
                    <div className="bg-yellow-900/20 border border-yellow-700/30 rounded p-3">
                      <div className="text-yellow-400 font-semibold mb-1">H₃ (Context)</div>
                      <div className="text-gray-400 text-xs">
                        Contextual modulation: goals, emotions, constraints
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-orange-400 font-semibold mb-3">Decision Function</h4>
                  <div className="bg-gray-900/50 rounded p-3 mb-3">
                    <div className="text-yellow-300 font-mono text-sm text-center">
                      D = f(H₁, H₂, H₃)
                    </div>
                  </div>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p>• <strong className="text-orange-400">Triplet Integration:</strong> Combines three input streams</p>
                    <p>• <strong className="text-amber-400">Non-linear Mixing:</strong> Complex interaction dynamics</p>
                    <p>• <strong className="text-yellow-400">Context Sensitivity:</strong> H₃ modulates H₁-H₂ interaction</p>
                    <p>• <strong className="text-orange-300">Prime Structure:</strong> 3-element decision (prime)</p>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 border border-orange-600 rounded-lg p-4">
                  <h4 className="text-white font-semibold mb-2">Universal Application</h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    The H3 device serves as the brain's fundamental decision-making unit, used 
                    everywhere from single neurons to entire brain regions. Its triplet structure 
                    enables robust, context-dependent decisions with minimal computational overhead.
                  </p>
                </div>
              </div>
            </div>

            {/* H3 Applications */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
                <h5 className="text-blue-400 font-semibold mb-2">Neuronal Level</h5>
                <div className="text-gray-300 text-xs space-y-1">
                  <p>• H₁: Excitatory input</p>
                  <p>• H₂: Inhibitory input</p>
                  <p>• H₃: Neuromodulation</p>
                  <p>• D: Fire or not fire</p>
                </div>
              </div>
              <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-4">
                <h5 className="text-purple-400 font-semibold mb-2">Circuit Level</h5>
                <div className="text-gray-300 text-xs space-y-1">
                  <p>• H₁: Sensory input</p>
                  <p>• H₂: Memory recall</p>
                  <p>• H₃: Attention</p>
                  <p>• D: Action selection</p>
                </div>
              </div>
              <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-4">
                <h5 className="text-pink-400 font-semibold mb-2">System Level</h5>
                <div className="text-gray-300 text-xs space-y-1">
                  <p>• H₁: Current state</p>
                  <p>• H₂: Desired state</p>
                  <p>• H₃: Constraints</p>
                  <p>• D: Policy decision</p>
                </div>
              </div>
            </div>

            {/* Mathematical Framework */}
            <div className="mt-6 bg-gray-950/50 rounded-xl p-6">
              <h4 className="text-orange-400 font-semibold mb-4">Mathematical Framework</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="bg-gray-900/50 rounded p-3">
                    <div className="text-orange-400 text-xs font-semibold mb-1">Linear Case</div>
                    <div className="text-gray-300 font-mono text-xs">
                      D = w₁·H₁ + w₂·H₂ + w₃·H₃
                    </div>
                  </div>
                  <div className="bg-gray-900/50 rounded p-3">
                    <div className="text-amber-400 text-xs font-semibold mb-1">Multiplicative Case</div>
                    <div className="text-gray-300 font-mono text-xs">
                      D = H₁ · H₂ · f(H₃)
                    </div>
                  </div>
                  <div className="bg-gray-900/50 rounded p-3">
                    <div className="text-yellow-400 text-xs font-semibold mb-1">Gating Case</div>
                    <div className="text-gray-300 font-mono text-xs">
                      D = H₃ · (H₁ − H₂)
                    </div>
                  </div>
                </div>
                <div className="bg-orange-900/20 border border-orange-700/30 rounded-lg p-4">
                  <h5 className="text-orange-400 font-semibold mb-2">Key Properties</h5>
                  <div className="space-y-1 text-xs text-gray-300">
                    <p>• <strong>Flexibility:</strong> Adapts to different decision types</p>
                    <p>• <strong>Robustness:</strong> Three inputs provide redundancy</p>
                    <p>• <strong>Context:</strong> H₃ enables situation-dependent decisions</p>
                    <p>• <strong>Prime:</strong> 3-element structure (prime number)</p>
                    <p>• <strong>Universal:</strong> Scales from micro to macro</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
