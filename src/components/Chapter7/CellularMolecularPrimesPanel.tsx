import React, { useState } from 'react';
import { Dna, Hexagon, Activity, Clock } from 'lucide-react';

/**
 * Section 7.5: Primes in the neuron, glia, dendro-astrocytes, microtubule, proteins DNA
 * Section 7.6: Twelve ways to memorizing and twelve carriers operating rhythms/clocks
 */
export const CellularMolecularPrimesPanel: React.FC = () => {
  const [activeView, setActiveView] = useState<'cellular' | 'memory'>('cellular');

  const cellularComponents = {
    neuron: {
      primes: [2, 3, 5, 7, 11],
      features: ['Axon hillock (prime threshold)', 'Dendritic spines (~2-7 per μm)', 'Action potential (~5-7 ms)'],
      color: 'blue',
    },
    glia: {
      primes: [3, 5, 7, 11, 13],
      features: ['Astrocyte processes (3-5 major)', 'Oligodendrocyte wraps (5-7 layers)', 'Microglia branches (prime)'],
      color: 'green',
    },
    microtubule: {
      primes: [2, 3, 5, 7, 11, 13],
      features: ['13 protofilaments (prime!)', '8 nm diameter', 'Tubulin dimers at prime positions'],
      color: 'purple',
    },
  };

  const twelveMemoryWays = [
    { id: 1, name: 'Episodic', carrier: 'Hippocampal theta (5-7 Hz)', color: 'blue' },
    { id: 2, name: 'Semantic', carrier: 'Cortical gamma (40-100 Hz)', color: 'purple' },
    { id: 3, name: 'Procedural', carrier: 'Cerebellar oscillations', color: 'green' },
    { id: 4, name: 'Working', carrier: 'Prefrontal beta (13-30 Hz)', color: 'cyan' },
    { id: 5, name: 'Spatial', carrier: 'Grid cell firing (prime)', color: 'orange' },
    { id: 6, name: 'Emotional', carrier: 'Amygdala rhythms', color: 'red' },
    { id: 7, name: 'Sensory', carrier: 'Cortical alpha (8-13 Hz)', color: 'pink' },
    { id: 8, name: 'Motor', carrier: 'Mu rhythm (8-13 Hz)', color: 'indigo' },
    { id: 9, name: 'Implicit', carrier: 'Basal ganglia oscillations', color: 'teal' },
    { id: 10, name: 'Explicit', carrier: 'Medial temporal rhythms', color: 'amber' },
    { id: 11, name: 'Prospective', carrier: 'Frontal delta (2-4 Hz)', color: 'lime' },
    { id: 12, name: 'Retrospective', carrier: 'Parietal alpha', color: 'rose' },
  ];

  return (
    <div className="space-y-6">
      {/* Main Header */}
      <div className="bg-gradient-to-r from-indigo-900/30 via-purple-900/30 to-pink-900/30 border border-indigo-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Dna className="text-indigo-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">7.5-7.6 Cellular & Memory Systems</h2>
            <p className="text-gray-300 text-sm">Prime Patterns from Molecules to Memory</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          Prime number patterns extend from molecular structures through cellular organization to 
          memory systems, creating a unified prime-based architecture across all scales of brain function.
        </p>
      </div>

      {/* View Selector */}
      <div className="flex space-x-4">
        <button
          onClick={() => setActiveView('cellular')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'cellular'
              ? 'bg-indigo-900/40 border-indigo-600 text-indigo-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Hexagon size={20} />
          <span className="font-semibold">7.5 Cellular & Molecular</span>
        </button>
        <button
          onClick={() => setActiveView('memory')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'memory'
              ? 'bg-purple-900/40 border-purple-600 text-purple-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Clock size={20} />
          <span className="font-semibold">7.6 Twelve Memory Systems</span>
        </button>
      </div>

      {/* 7.5 Cellular & Molecular */}
      {activeView === 'cellular' && (
        <div className="space-y-6">
          {/* Neurons */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-blue-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Activity className="text-blue-400" size={24} />
              <span>Neurons: Prime Signal Processing</span>
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-blue-400 font-semibold mb-3">Neuronal Prime Parameters</h4>
                  <div className="grid grid-cols-5 gap-2 mb-4">
                    {cellularComponents.neuron.primes.map((prime) => (
                      <div
                        key={prime}
                        className="bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg p-3 text-center animate-pulse"
                      >
                        <div className="text-white font-bold">{prime}</div>
                      </div>
                    ))}
                  </div>
                  <div className="space-y-2 text-sm text-gray-300">
                    {cellularComponents.neuron.features.map((feature) => (
                      <div key={feature} className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
                  <h4 className="text-blue-400 font-semibold mb-2">Action Potential Primes</h4>
                  <div className="space-y-2 text-xs text-gray-300">
                    <p>• <strong>Threshold:</strong> ~-55 mV (prime relationship to resting)</p>
                    <p>• <strong>Duration:</strong> 1-2 ms rising, 3 ms falling (primes)</p>
                    <p>• <strong>Refractory:</strong> 2 ms absolute, 5 ms relative</p>
                    <p>• <strong>Propagation:</strong> 1-100 m/s (prime factors)</p>
                  </div>
                </div>
              </div>
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">Synaptic Prime Architecture</h4>
                <div className="space-y-3">
                  <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
                    <div className="text-blue-400 font-semibold mb-1">Vesicle Release</div>
                    <div className="text-gray-400 text-xs">
                      Quantal release: 2-3 vesicles per AP (prime)
                    </div>
                  </div>
                  <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                    <div className="text-cyan-400 font-semibold mb-1">Synaptic Cleft</div>
                    <div className="text-gray-400 text-xs">
                      Width: ~20 nm (near prime multiples)
                    </div>
                  </div>
                  <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
                    <div className="text-blue-400 font-semibold mb-1">Dendritic Spines</div>
                    <div className="text-gray-400 text-xs">
                      Density: 2-7 per μm (prime range)
                    </div>
                  </div>
                  <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                    <div className="text-cyan-400 font-semibold mb-1">LTP/LTD</div>
                    <div className="text-gray-400 text-xs">
                      Induction: 5-7 Hz theta, 11-13 Hz beta
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Glia & Astrocytes */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-green-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Hexagon className="text-green-400" size={24} />
              <span>Glia & Dendro-Astrocytes: Support Prime Networks</span>
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-green-400 font-semibold mb-3">Glial Prime Parameters</h4>
                  <div className="grid grid-cols-5 gap-2 mb-4">
                    {cellularComponents.glia.primes.map((prime) => (
                      <div
                        key={prime}
                        className="bg-gradient-to-br from-green-600 to-emerald-600 rounded-lg p-3 text-center animate-pulse"
                      >
                        <div className="text-white font-bold">{prime}</div>
                      </div>
                    ))}
                  </div>
                  <div className="space-y-2 text-sm text-gray-300">
                    {cellularComponents.glia.features.map((feature) => (
                      <div key={feature} className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-4">
                  <h4 className="text-green-400 font-semibold mb-2">Astrocyte Domains</h4>
                  <div className="space-y-2 text-xs text-gray-300">
                    <p>• <strong>Territorial:</strong> Each contacts ~100,000 synapses</p>
                    <p>• <strong>Processes:</strong> 3-5 major branches (prime)</p>
                    <p>• <strong>Ca²⁺ Waves:</strong> Propagate at ~5-7 μm/s</p>
                    <p>• <strong>Gliotransmitters:</strong> Released in prime quanta</p>
                  </div>
                </div>
              </div>
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">Glial Prime Functions</h4>
                <div className="space-y-3">
                  <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-3">
                    <div className="text-green-400 font-semibold mb-1">Metabolic Support</div>
                    <div className="text-gray-400 text-xs">
                      Lactate transfer at prime-optimized rates
                    </div>
                  </div>
                  <div className="bg-emerald-900/20 border border-emerald-700/30 rounded-lg p-3">
                    <div className="text-emerald-400 font-semibold mb-1">Ion Homeostasis</div>
                    <div className="text-gray-400 text-xs">
                      K⁺ buffering with prime efficiency
                    </div>
                  </div>
                  <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-3">
                    <div className="text-green-400 font-semibold mb-1">Neurotransmitter Recycling</div>
                    <div className="text-gray-400 text-xs">
                      Glutamate uptake at prime kinetics
                    </div>
                  </div>
                  <div className="bg-emerald-900/20 border border-emerald-700/30 rounded-lg p-3">
                    <div className="text-emerald-400 font-semibold mb-1">Tripartite Synapse</div>
                    <div className="text-gray-400 text-xs">
                      3-way communication (prime structure)
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Microtubules */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Hexagon className="text-purple-400" size={24} />
              <span>Microtubules: 13-Protofilament Prime Structure</span>
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-purple-400 font-semibold mb-3">13 Protofilaments (Prime!)</h4>
                <div className="aspect-square bg-black/30 rounded-lg p-4 relative">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="relative w-48 h-48">
                      {Array.from({ length: 13 }, (_, i) => {
                        const angle = (i / 13) * 2 * Math.PI - Math.PI / 2;
                        const x = 50 + Math.cos(angle) * 40;
                        const y = 50 + Math.sin(angle) * 40;
                        return (
                          <div
                            key={i}
                            className="absolute w-4 h-4 bg-purple-500 rounded-full"
                            style={{
                              left: `${x}%`,
                              top: `${y}%`,
                              transform: 'translate(-50%, -50%)',
                            }}
                          >
                            <div className="absolute inset-0 bg-purple-500 rounded-full animate-ping"></div>
                          </div>
                        );
                      })}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-purple-400 font-bold text-2xl">13</div>
                      </div>
                    </div>
                  </div>
                </div>
                <p className="text-gray-400 text-xs mt-2 text-center">
                  13 protofilaments form hollow cylinder - a prime number structure
                </p>
              </div>
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-purple-400 font-semibold mb-3">Tubulin Prime Properties</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p>• <strong className="text-purple-400">Dimer Length:</strong> 8 nm (2³)</p>
                    <p>• <strong className="text-purple-300">Diameter:</strong> 25 nm (near 23, prime)</p>
                    <p>• <strong className="text-purple-400">Growth Rate:</strong> ~2-5 μm/min (primes)</p>
                    <p>• <strong className="text-purple-300">GTP Hydrolysis:</strong> Powers growth</p>
                    <p>• <strong className="text-purple-400">Dynamic Instability:</strong> Rapid assembly/disassembly</p>
                  </div>
                </div>
                <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-4">
                  <h4 className="text-purple-400 font-semibold mb-2">Quantum Processing</h4>
                  <div className="space-y-2 text-xs text-gray-300">
                    <p>• Tubulin dimers may support quantum states</p>
                    <p>• 13-protofilament structure enables quantum coherence</p>
                    <p>• Prime structure optimizes information processing</p>
                    <p>• Orchestrated Objective Reduction (Orch OR) theory</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Proteins & DNA */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-pink-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Dna className="text-pink-400" size={24} />
              <span>Proteins & DNA: Molecular Prime Encoding</span>
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-pink-400 font-semibold mb-3">DNA Prime Structure</h4>
                <div className="space-y-3">
                  <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-3">
                    <div className="text-pink-400 font-semibold mb-1">Double Helix</div>
                    <div className="text-gray-400 text-xs">
                      2-strand structure (prime), 10 bp per turn
                    </div>
                  </div>
                  <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-3">
                    <div className="text-pink-400 font-semibold mb-1">Base Pairing</div>
                    <div className="text-gray-400 text-xs">
                      2 purines, 2 pyrimidines (prime pairs)
                    </div>
                  </div>
                  <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-3">
                    <div className="text-pink-400 font-semibold mb-1">Codons</div>
                    <div className="text-gray-400 text-xs">
                      3-base sequences (prime), 64 total (2⁶)
                    </div>
                  </div>
                  <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-3">
                    <div className="text-pink-400 font-semibold mb-1">Amino Acids</div>
                    <div className="text-gray-400 text-xs">
                      20 standard (4×5, uses primes 2 and 5)
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">Protein Prime Folding</h4>
                <div className="space-y-3">
                  <div className="bg-rose-900/20 border border-rose-700/30 rounded-lg p-3">
                    <div className="text-rose-400 font-semibold mb-1">Alpha Helix</div>
                    <div className="text-gray-400 text-xs">
                      3.6 residues/turn, pitch ~5.4 Å (near primes)
                    </div>
                  </div>
                  <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-3">
                    <div className="text-pink-400 font-semibold mb-1">Beta Sheet</div>
                    <div className="text-gray-400 text-xs">
                      2-strand minimum (prime), extended structure
                    </div>
                  </div>
                  <div className="bg-rose-900/20 border border-rose-700/30 rounded-lg p-3">
                    <div className="text-rose-400 font-semibold mb-1">Domain Structure</div>
                    <div className="text-gray-400 text-xs">
                      Typical protein: 2-5 domains (prime range)
                    </div>
                  </div>
                  <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-3">
                    <div className="text-pink-400 font-semibold mb-1">Ion Channels</div>
                    <div className="text-gray-400 text-xs">
                      Often 2, 3, 5, or 7 subunits (all primes)
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 7.6 Twelve Memory Ways */}
      {activeView === 'memory' && (
        <div className="space-y-6">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <Clock className="text-purple-400" size={24} />
              <span>Twelve Ways to Memorizing & Twelve Carriers Operating Rhythms</span>
            </h3>
            <p className="text-gray-300 text-sm mb-6">
              The brain employs twelve distinct memory systems, each operating through specific 
              carrier waves and rhythmic patterns. These twelve systems work in concert to create 
              the full spectrum of human memory and consciousness.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {twelveMemoryWays.map((memory) => (
                <div
                  key={memory.id}
                  className={`bg-${memory.color}-900/20 border border-${memory.color}-700/30 rounded-xl p-4 hover:border-${memory.color}-600 transition-all`}
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`w-10 h-10 bg-${memory.color}-600 rounded-lg flex items-center justify-center text-white font-bold`}>
                      {memory.id}
                    </div>
                    <div>
                      <div className={`text-${memory.color}-400 font-bold`}>{memory.name}</div>
                      <div className="text-gray-400 text-xs">Memory Type {memory.id}/12</div>
                    </div>
                  </div>
                  <div className="bg-gray-950/50 rounded-lg p-3">
                    <div className="text-gray-300 text-xs mb-2">
                      <strong className={`text-${memory.color}-400`}>Carrier Wave:</strong>
                    </div>
                    <div className="text-gray-400 text-xs">{memory.carrier}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Twelve Carriers Operating Rhythms */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-indigo-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Twelve Carriers: Oscillatory Clocks</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-indigo-400 font-semibold mb-3">Slow Oscillations (0.5-8 Hz)</h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-indigo-900/20 border border-indigo-700/30 rounded p-2">
                      <strong className="text-indigo-400">1. Delta (0.5-4 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Deep sleep, memory consolidation</span>
                    </div>
                    <div className="bg-purple-900/20 border border-purple-700/30 rounded p-2">
                      <strong className="text-purple-400">2. Theta (4-8 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Hippocampal memory encoding</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-green-400 font-semibold mb-3">Medium Oscillations (8-30 Hz)</h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-green-900/20 border border-green-700/30 rounded p-2">
                      <strong className="text-green-400">3. Alpha (8-13 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Relaxed awareness, attention</span>
                    </div>
                    <div className="bg-emerald-900/20 border border-emerald-700/30 rounded p-2">
                      <strong className="text-emerald-400">4. Mu (8-13 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Motor cortex idle state</span>
                    </div>
                    <div className="bg-cyan-900/20 border border-cyan-700/30 rounded p-2">
                      <strong className="text-cyan-400">5. Beta (13-30 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Active thinking, working memory</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-orange-400 font-semibold mb-3">Fast Oscillations (30-200 Hz)</h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-orange-900/20 border border-orange-700/30 rounded p-2">
                      <strong className="text-orange-400">6. Low Gamma (30-50 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Sensory binding</span>
                    </div>
                    <div className="bg-amber-900/20 border border-amber-700/30 rounded p-2">
                      <strong className="text-amber-400">7. Mid Gamma (50-80 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Cognitive integration</span>
                    </div>
                    <div className="bg-red-900/20 border border-red-700/30 rounded p-2">
                      <strong className="text-red-400">8. High Gamma (80-200 Hz):</strong>
                      <span className="text-gray-400 text-xs ml-2">Local processing</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-pink-400 font-semibold mb-3">Cross-Frequency Coupling</h4>
                  <div className="space-y-2 text-sm">
                    <div className="bg-pink-900/20 border border-pink-700/30 rounded p-2">
                      <strong className="text-pink-400">9. Theta-Gamma:</strong>
                      <span className="text-gray-400 text-xs ml-2">Memory encoding/retrieval</span>
                    </div>
                    <div className="bg-rose-900/20 border border-rose-700/30 rounded p-2">
                      <strong className="text-rose-400">10. Alpha-Beta:</strong>
                      <span className="text-gray-400 text-xs ml-2">Attention control</span>
                    </div>
                    <div className="bg-purple-900/20 border border-purple-700/30 rounded p-2">
                      <strong className="text-purple-400">11. Delta-Ripple:</strong>
                      <span className="text-gray-400 text-xs ml-2">Sleep consolidation</span>
                    </div>
                    <div className="bg-indigo-900/20 border border-indigo-700/30 rounded p-2">
                      <strong className="text-indigo-400">12. Beta-Gamma:</strong>
                      <span className="text-gray-400 text-xs ml-2">Active cognition</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 border border-purple-600 rounded-lg p-4">
                  <h4 className="text-white font-semibold mb-2">Harmonic Integration</h4>
                  <p className="text-gray-300 text-sm leading-relaxed mb-3">
                    These twelve oscillatory carriers operate in prime-related frequency bands, 
                    creating a multi-scale temporal framework for consciousness. Cross-frequency 
                    coupling enables information flow between different timescales.
                  </p>
                  <div className="bg-purple-950/50 rounded p-3">
                    <div className="text-purple-400 text-xs font-mono">
                      Memory = Σ(carrier_i × content_i) for i = 1 to 12
                    </div>
                  </div>
                </div>

                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-2">Prime Timing</h4>
                  <div className="text-gray-300 text-xs space-y-1">
                    <p>• Many carriers operate in prime Hz ranges (5-7, 11-13, etc.)</p>
                    <p>• Cross-frequency coupling follows prime ratios</p>
                    <p>• Twelve systems match dodecanion structure (12D)</p>
                    <p>• Optimal information capacity through prime harmonics</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className="bg-gradient-to-r from-indigo-900/30 via-purple-900/30 to-pink-900/30 border border-indigo-600 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Unified Memory-Carrier Framework</h3>
            <p className="text-gray-300 text-sm leading-relaxed">
              <strong className="text-indigo-400">Integration:</strong> The twelve memory systems operate 
              through twelve carrier wave patterns, creating a complete temporal framework for consciousness. 
              Each memory type has its dedicated oscillatory carrier, yet all systems interact through 
              cross-frequency coupling. This twelve-fold architecture mirrors the dodecanion mathematical 
              structure, enabling the brain to process information across multiple temporal scales 
              simultaneously. Prime frequency bands ensure maximal information capacity and minimal 
              interference between carriers, creating an optimal substrate for conscious experience.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
