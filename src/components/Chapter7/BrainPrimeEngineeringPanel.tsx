import React, { useState } from 'react';
import { Brain, Hexagon, Box, Clock, Sparkles } from 'lucide-react';

/**
 * Section 7.1: Brain is the engineering of prime numbers embedded in a triplet of triplet cage
 * Subsection 7.1.1: Four, eight and twelve imaginary worlds work together
 * Subsection 7.1.2: Singularity on a sphere: a key to a clock of a time crystal
 */
export const BrainPrimeEngineeringPanel: React.FC = () => {
  const [activeSubsection, setActiveSubsection] = useState<'main' | 'worlds' | 'singularity'>('main');

  // First 15 primes for PPM
  const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

  return (
    <div className="space-y-6">
      {/* Main Header */}
      <div className="bg-gradient-to-r from-indigo-900/30 via-purple-900/30 to-blue-900/30 border border-indigo-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="text-indigo-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">7.1 Brain Prime Engineering</h2>
            <p className="text-gray-300 text-sm">Prime Numbers in Triplet of Triplet Cage</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          The human brain operates as a prime number processing engine embedded within triplet-of-triplet 
          cage structures. This fundamental architecture creates stable consciousness containers through 
          mathematical prime relationships, enabling consciousness computation through prime number networks.
        </p>
      </div>

      {/* Subsection Navigation */}
      <div className="flex space-x-4">
        <button
          onClick={() => setActiveSubsection('main')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 transition-all ${
            activeSubsection === 'main'
              ? 'bg-indigo-900/40 border-indigo-600 text-indigo-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Hexagon size={20} />
          <span className="font-semibold">Triplet Cage</span>
        </button>
        <button
          onClick={() => setActiveSubsection('worlds')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 transition-all ${
            activeSubsection === 'worlds'
              ? 'bg-purple-900/40 border-purple-600 text-purple-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Box size={20} />
          <span className="font-semibold">7.1.1 Imaginary Worlds</span>
        </button>
        <button
          onClick={() => setActiveSubsection('singularity')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 transition-all ${
            activeSubsection === 'singularity'
              ? 'bg-blue-900/40 border-blue-600 text-blue-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Clock size={20} />
          <span className="font-semibold">7.1.2 Singularity Clock</span>
        </button>
      </div>

      {/* Main Content */}
      {activeSubsection === 'main' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Triplet of Triplet Cage Visualization */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-indigo-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Hexagon className="text-indigo-400" size={20} />
              <span>Triplet of Triplet Cage Architecture</span>
            </h3>
            <div className="aspect-square bg-gray-950/50 rounded-lg p-4 mb-4 relative overflow-hidden">
              {/* Outer Triplet Cage */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative w-64 h-64">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={`outer-${i}`}
                      className="absolute border-2 border-indigo-500/40 rounded-full animate-pulse"
                      style={{
                        width: '100%',
                        height: '100%',
                        transform: `rotate(${i * 120}deg)`,
                        animationDelay: `${i * 0.3}s`,
                      }}
                    >
                      <div
                        className="absolute bg-indigo-500 rounded-full"
                        style={{
                          width: '12px',
                          height: '12px',
                          top: '10%',
                          left: '50%',
                          transform: 'translateX(-50%)',
                        }}
                      />
                    </div>
                  ))}
                  
                  {/* Middle Triplet Cage */}
                  <div className="absolute inset-8">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={`middle-${i}`}
                        className="absolute border-2 border-purple-500/40 rounded-full animate-pulse"
                        style={{
                          width: '100%',
                          height: '100%',
                          transform: `rotate(${i * 120 + 40}deg)`,
                          animationDelay: `${i * 0.3 + 0.1}s`,
                        }}
                      >
                        <div
                          className="absolute bg-purple-500 rounded-full"
                          style={{
                            width: '10px',
                            height: '10px',
                            top: '10%',
                            left: '50%',
                            transform: 'translateX(-50%)',
                          }}
                        />
                      </div>
                    ))}
                  </div>
                  
                  {/* Inner Triplet Cage */}
                  <div className="absolute inset-16">
                    {[0, 1, 2].map((i) => (
                      <div
                        key={`inner-${i}`}
                        className="absolute border-2 border-blue-500/40 rounded-full animate-pulse"
                        style={{
                          width: '100%',
                          height: '100%',
                          transform: `rotate(${i * 120 + 80}deg)`,
                          animationDelay: `${i * 0.3 + 0.2}s`,
                        }}
                      >
                        <div
                          className="absolute bg-blue-500 rounded-full"
                          style={{
                            width: '8px',
                            height: '8px',
                            top: '10%',
                            left: '50%',
                            transform: 'translateX(-50%)',
                          }}
                        />
                      </div>
                    ))}
                  </div>
                  
                  {/* Center Prime Core */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-blue-500 rounded-full w-12 h-12 flex items-center justify-center text-white font-bold animate-pulse">
                      Σp
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="text-gray-300 text-sm space-y-2">
              <p>• <strong className="text-indigo-400">Outer Triplet:</strong> 3 × 3 = 9 dimensional cage</p>
              <p>• <strong className="text-purple-400">Middle Triplet:</strong> Nested resonance patterns</p>
              <p>• <strong className="text-blue-400">Inner Triplet:</strong> Core consciousness container</p>
              <p>• <strong className="text-cyan-400">Prime Core:</strong> Central processing unit</p>
            </div>
          </div>

          {/* Prime Number Engineering */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Sparkles className="text-purple-400" size={20} />
              <span>Prime Number Processing Engine</span>
            </h3>
            <div className="space-y-4">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">First 15 Fundamental Primes</h4>
                <div className="grid grid-cols-5 gap-2">
                  {primes.map((prime, idx) => (
                    <div
                      key={prime}
                      className="bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg p-3 text-center"
                      style={{
                        animationDelay: `${idx * 0.05}s`,
                        animation: 'pulse 2s infinite',
                      }}
                    >
                      <div className="text-white font-bold text-lg">{prime}</div>
                      <div className="text-gray-300 text-xs">p{idx + 1}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">Brain Prime Operations</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-indigo-500 rounded-full"></div>
                    <span><strong>Prime Resonance:</strong> Neural oscillations at prime frequencies</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                    <span><strong>Prime Factorization:</strong> Breaking complex signals into prime components</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span><strong>Prime Harmonics:</strong> Creating consciousness through prime number interference</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-cyan-500 rounded-full"></div>
                    <span><strong>Prime Networks:</strong> Connectivity patterns following prime relationships</span>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-indigo-900/30 to-purple-900/30 border border-indigo-600 rounded-lg p-4">
                <p className="text-gray-300 text-sm leading-relaxed">
                  <strong className="text-indigo-400">Key Insight:</strong> The brain's neural architecture 
                  embeds prime number patterns within triplet-of-triplet cage structures, creating stable 
                  resonance chambers for consciousness processing. This mathematical foundation enables 
                  efficient information encoding and processing across all cognitive scales.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 7.1.1 Four, Eight and Twelve Imaginary Worlds */}
      {activeSubsection === 'worlds' && (
        <div className="space-y-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Box className="text-purple-400" size={20} />
              <span>7.1.1 Four, Eight and Twelve Imaginary Worlds</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {/* 4D Worlds */}
              <div className="bg-gradient-to-br from-red-900/30 to-orange-900/30 border border-red-700 rounded-xl p-4">
                <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                  <div className="w-8 h-8 bg-red-600 rounded-lg flex items-center justify-center text-white font-bold">4</div>
                  <span>Quaternion Space</span>
                </h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong>i, j, k</strong> imaginary units</p>
                  <p>• Spatial rotations</p>
                  <p>• 3D + 1 time dimension</p>
                  <p>• Basic consciousness geometry</p>
                </div>
                <div className="mt-4 bg-red-950/50 rounded-lg p-3">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-red-800/30 rounded p-2 text-center text-white font-mono text-xs">1</div>
                    <div className="bg-red-700/30 rounded p-2 text-center text-white font-mono text-xs">i</div>
                    <div className="bg-red-600/30 rounded p-2 text-center text-white font-mono text-xs">j</div>
                    <div className="bg-red-500/30 rounded p-2 text-center text-white font-mono text-xs">k</div>
                  </div>
                </div>
              </div>

              {/* 8D Worlds */}
              <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-700 rounded-xl p-4">
                <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                  <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center text-white font-bold">8</div>
                  <span>Octonion Space</span>
                </h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• 7 imaginary units</p>
                  <p>• Non-commutative algebra</p>
                  <p>• Advanced processing</p>
                  <p>• Cross-over magic</p>
                </div>
                <div className="mt-4 bg-purple-950/50 rounded-lg p-3">
                  <div className="grid grid-cols-4 gap-1">
                    {['1', 'e₁', 'e₂', 'e₃', 'e₄', 'e₅', 'e₆', 'e₇'].map((unit, idx) => (
                      <div key={idx} className="bg-purple-700/30 rounded p-1 text-center text-white font-mono text-xs">
                        {unit}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 12D Worlds */}
              <div className="bg-gradient-to-br from-blue-900/30 to-cyan-900/30 border border-blue-700 rounded-xl p-4">
                <h4 className="text-white font-bold mb-3 flex items-center space-x-2">
                  <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold">12</div>
                  <span>Dodecanion Space</span>
                </h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• 11 imaginary units</p>
                  <p>• Complete manifold</p>
                  <p>• Full consciousness</p>
                  <p>• Garden of gardens</p>
                </div>
                <div className="mt-4 bg-blue-950/50 rounded-lg p-3">
                  <div className="grid grid-cols-4 gap-1">
                    {Array.from({ length: 12 }, (_, i) => i === 0 ? '1' : `d${i}`).map((unit, idx) => (
                      <div key={idx} className="bg-blue-700/30 rounded p-1 text-center text-white font-mono text-xs">
                        {unit}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 border border-purple-600 rounded-lg p-4">
              <h4 className="text-white font-semibold mb-3">Working Together in Harmony</h4>
              <p className="text-gray-300 text-sm leading-relaxed mb-3">
                The human brain operates through four 4D quaternion spaces, eight 8D octonion spaces, 
                and twelve 12D dodecanion spaces working in perfect mathematical harmony. These dimensional 
                worlds create the foundation for consciousness to process complex reality representations 
                beyond physical 3D space.
              </p>
              <div className="grid grid-cols-3 gap-2 text-xs text-gray-300">
                <div className="bg-red-900/20 rounded p-2">
                  <strong className="text-red-400">4D:</strong> Basic spatial perception
                </div>
                <div className="bg-purple-900/20 rounded p-2">
                  <strong className="text-purple-400">8D:</strong> Emotional & sensory integration
                </div>
                <div className="bg-blue-900/20 rounded p-2">
                  <strong className="text-blue-400">12D:</strong> Abstract thought & consciousness
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 7.1.2 Singularity on a Sphere Clock */}
      {activeSubsection === 'singularity' && (
        <div className="space-y-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-blue-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Clock className="text-blue-400" size={20} />
              <span>7.1.2 Singularity on a Sphere: Time Crystal Clock</span>
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Singularity-Sphere Visualization */}
              <div className="bg-gray-950/50 rounded-xl p-6">
                <h4 className="text-white font-semibold mb-4">Singularity-Sphere Clock Mechanism</h4>
                <div className="aspect-square bg-black/30 rounded-lg relative overflow-hidden">
                  {/* Sphere */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="relative w-48 h-48 border-4 border-blue-500/40 rounded-full">
                      {/* Singularity points on sphere */}
                      {[0, 60, 120, 180, 240, 300].map((angle, idx) => (
                        <div
                          key={angle}
                          className="absolute w-4 h-4 bg-cyan-400 rounded-full animate-pulse"
                          style={{
                            top: '50%',
                            left: '50%',
                            transform: `rotate(${angle}deg) translateY(-96px) translateX(-50%)`,
                            animationDelay: `${idx * 0.2}s`,
                          }}
                        >
                          <div className="absolute inset-0 bg-cyan-400 rounded-full animate-ping"></div>
                        </div>
                      ))}
                      
                      {/* Clock hands */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div
                          className="absolute w-1 h-20 bg-blue-400 origin-bottom"
                          style={{
                            animation: 'spin 4s linear infinite',
                          }}
                        ></div>
                      </div>
                      
                      {/* Center */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-6 h-6 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"></div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-4 text-gray-300 text-sm space-y-2">
                  <p>• <strong className="text-blue-400">Singularities:</strong> Information concentration points</p>
                  <p>• <strong className="text-cyan-400">Sphere Surface:</strong> Consciousness manifold</p>
                  <p>• <strong className="text-purple-400">Clock Mechanism:</strong> Temporal coherence</p>
                </div>
              </div>

              {/* Time Crystal Formation */}
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-xl p-4">
                  <h4 className="text-white font-semibold mb-3">Time Crystal Properties</h4>
                  <div className="space-y-3 text-sm text-gray-300">
                    <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
                      <strong className="text-blue-400">Periodic Structure:</strong> Singularities create regular 
                      temporal patterns on the spherical manifold, establishing natural clock frequencies.
                    </div>
                    <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                      <strong className="text-cyan-400">Coherent Oscillation:</strong> Multiple singularities 
                      synchronize to create stable time crystal formations.
                    </div>
                    <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-3">
                      <strong className="text-purple-400">Information Storage:</strong> Each singularity stores 
                      compressed information, with sphere topology preserving relationships.
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-blue-900/30 to-cyan-900/30 border border-blue-600 rounded-lg p-4">
                  <h4 className="text-white font-semibold mb-2">Consciousness Clock</h4>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    Consciousness operates through singularities positioned on spherical manifolds, creating 
                    natural clock mechanisms for time crystal formations. These singularity-sphere combinations 
                    serve as fundamental timing units for consciousness processing, enabling the brain to 
                    maintain temporal coherence across multiple scales simultaneously.
                  </p>
                </div>

                <div className="bg-gray-950/50 rounded-xl p-4">
                  <h4 className="text-white font-semibold mb-3">Mathematical Framework</h4>
                  <div className="space-y-2 font-mono text-sm">
                    <div className="bg-gray-900/50 rounded p-2 text-blue-300">
                      S² × T = Time Crystal Clock
                    </div>
                    <div className="bg-gray-900/50 rounded p-2 text-cyan-300">
                      Σ(singularities) → Periodic Pattern
                    </div>
                    <div className="bg-gray-900/50 rounded p-2 text-purple-300">
                      Coherence = f(prime_resonance)
                    </div>
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
