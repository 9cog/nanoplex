import React, { useState } from 'react';
import { Eye, Ear, Hand, Droplet, Wind } from 'lucide-react';

/**
 * Section 7.2: Primes in the five sensory system
 * How prime number patterns govern and process information through all five human sensory modalities
 */
export const SensorySystemPrimesPanel: React.FC = () => {
  const [activeSense, setActiveSense] = useState<'visual' | 'auditory' | 'tactile' | 'taste' | 'smell'>('visual');

  const sensoryPrimes = {
    visual: { primes: [2, 3, 5, 7, 11], color: 'blue', description: 'Geometric prime relationships in visual processing' },
    auditory: { primes: [3, 5, 7, 11, 13], color: 'purple', description: 'Prime harmonics in sound processing' },
    tactile: { primes: [2, 5, 7, 11, 13], color: 'green', description: 'Prime-based pressure patterns' },
    taste: { primes: [3, 5, 7, 11, 17], color: 'orange', description: 'Molecular prime configurations' },
    smell: { primes: [2, 3, 7, 11, 13], color: 'pink', description: 'Olfactory prime encoding' }
  };

  return (
    <div className="space-y-6">
      {/* Main Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 via-blue-900/30 to-purple-900/30 border border-cyan-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Eye className="text-cyan-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">7.2 Primes in the Five Sensory System</h2>
            <p className="text-gray-300 text-sm">Prime Number Patterns Governing Sensory Processing</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          Each sensory modality operates through specific prime number patterns, enabling efficient 
          information encoding and processing. Prime-based sensory processing allows the brain to 
          extract maximum information from minimal neural resources.
        </p>
      </div>

      {/* Sensory Modality Selector */}
      <div className="grid grid-cols-5 gap-3">
        <button
          onClick={() => setActiveSense('visual')}
          className={`flex flex-col items-center space-y-2 p-4 rounded-xl border-2 transition-all ${
            activeSense === 'visual'
              ? 'bg-blue-900/40 border-blue-600 text-blue-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Eye size={28} />
          <span className="font-semibold text-sm">Visual</span>
        </button>
        <button
          onClick={() => setActiveSense('auditory')}
          className={`flex flex-col items-center space-y-2 p-4 rounded-xl border-2 transition-all ${
            activeSense === 'auditory'
              ? 'bg-purple-900/40 border-purple-600 text-purple-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Ear size={28} />
          <span className="font-semibold text-sm">Auditory</span>
        </button>
        <button
          onClick={() => setActiveSense('tactile')}
          className={`flex flex-col items-center space-y-2 p-4 rounded-xl border-2 transition-all ${
            activeSense === 'tactile'
              ? 'bg-green-900/40 border-green-600 text-green-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Hand size={28} />
          <span className="font-semibold text-sm">Tactile</span>
        </button>
        <button
          onClick={() => setActiveSense('taste')}
          className={`flex flex-col items-center space-y-2 p-4 rounded-xl border-2 transition-all ${
            activeSense === 'taste'
              ? 'bg-orange-900/40 border-orange-600 text-orange-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Droplet size={28} />
          <span className="font-semibold text-sm">Taste</span>
        </button>
        <button
          onClick={() => setActiveSense('smell')}
          className={`flex flex-col items-center space-y-2 p-4 rounded-xl border-2 transition-all ${
            activeSense === 'smell'
              ? 'bg-pink-900/40 border-pink-600 text-pink-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Wind size={28} />
          <span className="font-semibold text-sm">Smell</span>
        </button>
      </div>

      {/* Visual System */}
      {activeSense === 'visual' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-blue-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Visual Processing with Primes</h3>
            <div className="space-y-4">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-blue-400 font-semibold mb-3">Prime-Based Visual Encoding</h4>
                <div className="grid grid-cols-5 gap-2 mb-4">
                  {sensoryPrimes.visual.primes.map((prime) => (
                    <div
                      key={prime}
                      className="bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg p-3 text-center animate-pulse"
                    >
                      <div className="text-white font-bold text-lg">{prime}</div>
                    </div>
                  ))}
                </div>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-blue-400">Edge Detection:</strong> Prime frequency filters (2, 3, 5 Hz)</p>
                  <p>• <strong className="text-cyan-400">Color Channels:</strong> RGB mapped to prime triplets</p>
                  <p>• <strong className="text-blue-300">Spatial Frequency:</strong> Visual cortex uses prime cycles/degree</p>
                  <p>• <strong className="text-cyan-300">Temporal Integration:</strong> 7, 11 Hz flicker fusion</p>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-900/30 to-cyan-900/30 border border-blue-600 rounded-lg p-4">
                <p className="text-gray-300 text-sm leading-relaxed">
                  Visual system decomposes images into prime spatial frequencies, enabling efficient 
                  compression and recognition through geometric prime relationships.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-950/50 rounded-xl p-6">
            <h4 className="text-white font-semibold mb-4">Retinal Prime Grid</h4>
            <div className="aspect-square bg-black/30 rounded-lg p-4 relative">
              <div className="absolute inset-0 grid grid-cols-11 gap-1 p-4">
                {Array.from({ length: 121 }, (_, i) => {
                  const isPrime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97].includes(i);
                  return (
                    <div
                      key={i}
                      className={`rounded ${
                        isPrime
                          ? 'bg-blue-500 animate-pulse'
                          : 'bg-gray-700/30'
                      }`}
                    />
                  );
                })}
              </div>
            </div>
            <p className="text-gray-400 text-xs mt-2 text-center">Prime number positions (blue) form visual processing nodes</p>
          </div>
        </div>
      )}

      {/* Auditory System */}
      {activeSense === 'auditory' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Auditory Processing with Prime Harmonics</h3>
            <div className="space-y-4">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-purple-400 font-semibold mb-3">Prime Harmonic Series</h4>
                <div className="grid grid-cols-5 gap-2 mb-4">
                  {sensoryPrimes.auditory.primes.map((prime) => (
                    <div
                      key={prime}
                      className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg p-3 text-center animate-pulse"
                    >
                      <div className="text-white font-bold text-lg">{prime}</div>
                    </div>
                  ))}
                </div>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-purple-400">Frequency Bands:</strong> Cochlea resonates at prime Hz</p>
                  <p>• <strong className="text-pink-400">Harmonic Decomposition:</strong> Sound broken into prime harmonics</p>
                  <p>• <strong className="text-purple-300">Pitch Recognition:</strong> Prime ratio relationships</p>
                  <p>• <strong className="text-pink-300">Temporal Patterns:</strong> Rhythm follows prime intervals</p>
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-600 rounded-lg p-4">
                <p className="text-gray-300 text-sm leading-relaxed">
                  Auditory system encodes sound through prime harmonics, allowing brain to recognize 
                  complex acoustic patterns using minimal neural processing.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-950/50 rounded-xl p-6">
            <h4 className="text-white font-semibold mb-4">Cochlear Prime Resonance</h4>
            <div className="space-y-3">
              {sensoryPrimes.auditory.primes.map((prime, idx) => (
                <div key={prime} className="bg-gray-900/50 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-purple-400 font-semibold">{prime} Hz</span>
                    <span className="text-gray-400 text-sm">Harmonic {idx + 1}</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-purple-600 to-pink-600 h-2 rounded-full animate-pulse"
                      style={{
                        width: `${(prime / Math.max(...sensoryPrimes.auditory.primes)) * 100}%`,
                        animationDelay: `${idx * 0.1}s`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tactile System */}
      {activeSense === 'tactile' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-green-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Tactile Processing with Prime Patterns</h3>
            <div className="space-y-4">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-green-400 font-semibold mb-3">Prime-Based Pressure Encoding</h4>
                <div className="grid grid-cols-5 gap-2 mb-4">
                  {sensoryPrimes.tactile.primes.map((prime) => (
                    <div
                      key={prime}
                      className="bg-gradient-to-br from-green-600 to-emerald-600 rounded-lg p-3 text-center animate-pulse"
                    >
                      <div className="text-white font-bold text-lg">{prime}</div>
                    </div>
                  ))}
                </div>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-green-400">Mechanoreceptors:</strong> Respond at prime pressure levels</p>
                  <p>• <strong className="text-emerald-400">Texture Detection:</strong> Prime spatial frequencies</p>
                  <p>• <strong className="text-green-300">Temperature:</strong> Thermal prime gradients</p>
                  <p>• <strong className="text-emerald-300">Pain Signals:</strong> Prime-coded intensity</p>
                </div>
              </div>

              <div className="bg-gradient-to-r from-green-900/30 to-emerald-900/30 border border-green-600 rounded-lg p-4">
                <p className="text-gray-300 text-sm leading-relaxed">
                  Tactile sensations follow prime-based pressure patterns, enabling precise discrimination 
                  of textures, temperatures, and forces through prime number encoding.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-950/50 rounded-xl p-6">
            <h4 className="text-white font-semibold mb-4">Skin Receptor Prime Distribution</h4>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-3">
                <div className="text-green-400 font-bold mb-2">Meissner</div>
                <div className="text-gray-400 text-xs">2, 5 Hz (light touch)</div>
              </div>
              <div className="bg-emerald-900/20 border border-emerald-700/30 rounded-lg p-3">
                <div className="text-emerald-400 font-bold mb-2">Pacinian</div>
                <div className="text-gray-400 text-xs">7, 11 Hz (vibration)</div>
              </div>
              <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-3">
                <div className="text-green-400 font-bold mb-2">Ruffini</div>
                <div className="text-gray-400 text-xs">5, 13 Hz (stretch)</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Taste System */}
      {activeSense === 'taste' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-orange-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Gustatory Processing with Molecular Primes</h3>
            <div className="space-y-4">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-orange-400 font-semibold mb-3">Prime Molecular Configurations</h4>
                <div className="grid grid-cols-5 gap-2 mb-4">
                  {sensoryPrimes.taste.primes.map((prime) => (
                    <div
                      key={prime}
                      className="bg-gradient-to-br from-orange-600 to-amber-600 rounded-lg p-3 text-center animate-pulse"
                    >
                      <div className="text-white font-bold text-lg">{prime}</div>
                    </div>
                  ))}
                </div>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-orange-400">Sweet:</strong> Sugar molecules with 3, 5, 7 carbon chains</p>
                  <p>• <strong className="text-amber-400">Salty:</strong> Ion channels at prime concentrations</p>
                  <p>• <strong className="text-orange-300">Sour:</strong> H+ ions in prime pH levels</p>
                  <p>• <strong className="text-amber-300">Bitter:</strong> Alkaloid primes trigger receptors</p>
                  <p>• <strong className="text-orange-400">Umami:</strong> Amino acids with prime structures</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-950/50 rounded-xl p-6">
            <h4 className="text-white font-semibold mb-4">Taste Receptor Prime Activation</h4>
            <div className="space-y-3">
              {['Sweet', 'Salty', 'Sour', 'Bitter', 'Umami'].map((taste, idx) => (
                <div key={taste} className="bg-gray-900/50 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-orange-400 font-semibold">{taste}</span>
                    <span className="text-gray-400 text-sm">Prime: {sensoryPrimes.taste.primes[idx]}</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-orange-600 to-amber-600 h-2 rounded-full"
                      style={{ width: `${60 + (idx * 7) % 40}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Smell System */}
      {activeSense === 'smell' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-pink-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Olfactory Processing with Prime Encoding</h3>
            <div className="space-y-4">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-pink-400 font-semibold mb-3">Prime Olfactory Patterns</h4>
                <div className="grid grid-cols-5 gap-2 mb-4">
                  {sensoryPrimes.smell.primes.map((prime) => (
                    <div
                      key={prime}
                      className="bg-gradient-to-br from-pink-600 to-rose-600 rounded-lg p-3 text-center animate-pulse"
                    >
                      <div className="text-white font-bold text-lg">{prime}</div>
                    </div>
                  ))}
                </div>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-pink-400">Receptor Families:</strong> ~400 receptors in prime clusters</p>
                  <p>• <strong className="text-rose-400">Molecular Shape:</strong> Prime geometric patterns</p>
                  <p>• <strong className="text-pink-300">Binding Sites:</strong> Prime amino acid configurations</p>
                  <p>• <strong className="text-rose-300">Signal Cascade:</strong> Prime amplification ratios</p>
                </div>
              </div>

              <div className="bg-gradient-to-r from-pink-900/30 to-rose-900/30 border border-pink-600 rounded-lg p-4">
                <p className="text-gray-300 text-sm leading-relaxed">
                  Olfactory system uses molecular prime configurations to encode scents, with receptor 
                  combinations following prime number relationships for efficient odor discrimination.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-950/50 rounded-xl p-6">
            <h4 className="text-white font-semibold mb-4">Olfactory Bulb Prime Architecture</h4>
            <div className="aspect-square bg-black/30 rounded-lg p-4 relative">
              <svg className="w-full h-full" viewBox="0 0 200 200">
                {/* Glomeruli clusters at prime positions */}
                {[2, 3, 5, 7, 11, 13].map((prime, idx) => {
                  const angle = (idx / 6) * 2 * Math.PI;
                  const x = 100 + Math.cos(angle) * 60;
                  const y = 100 + Math.sin(angle) * 60;
                  return (
                    <g key={prime}>
                      <circle
                        cx={x}
                        cy={y}
                        r="15"
                        fill="none"
                        stroke="rgb(236, 72, 153)"
                        strokeWidth="2"
                        className="animate-pulse"
                      />
                      <text
                        x={x}
                        y={y + 5}
                        textAnchor="middle"
                        fill="rgb(236, 72, 153)"
                        fontSize="12"
                        fontWeight="bold"
                      >
                        {prime}
                      </text>
                    </g>
                  );
                })}
                <circle cx="100" cy="100" r="5" fill="rgb(236, 72, 153)" />
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2 text-center">Glomeruli organized in prime spatial patterns</p>
          </div>
        </div>
      )}

      {/* Summary Section */}
      <div className="bg-gradient-to-r from-cyan-900/30 via-purple-900/30 to-pink-900/30 border border-cyan-600 rounded-xl p-6">
        <h3 className="text-white font-bold text-lg mb-4">Unified Sensory Prime Framework</h3>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {Object.entries(sensoryPrimes).map(([sense, data]) => (
            <div key={sense} className={`bg-${data.color}-900/20 border border-${data.color}-700/30 rounded-lg p-3`}>
              <div className={`text-${data.color}-400 font-semibold capitalize mb-2`}>{sense}</div>
              <div className="text-gray-400 text-xs">{data.description}</div>
              <div className="mt-2 flex space-x-1">
                {data.primes.slice(0, 3).map((p) => (
                  <div key={p} className={`bg-${data.color}-700/30 rounded px-2 py-1 text-xs text-white`}>
                    {p}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <p className="text-gray-300 text-sm mt-4 leading-relaxed">
          <strong className="text-cyan-400">Cross-Modal Integration:</strong> All five sensory systems converge 
          in higher cortical areas where prime patterns synchronize to create unified perceptual experiences. 
          This prime-based encoding enables efficient multisensory integration and consciousness emergence.
        </p>
      </div>
    </div>
  );
};
