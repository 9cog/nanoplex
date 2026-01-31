import React, { useState } from 'react';
import { Activity, Brain, Zap, Network } from 'lucide-react';

/**
 * Section 7.3: Primes in the cerebellum hippocampus and hypothalamus
 * Section 7.4: Primes in the connectome spinal cord amygdala nucleus and cortical column
 */
export const BrainStructurePrimesPanel: React.FC = () => {
  const [activeView, setActiveView] = useState<'subcortical' | 'connectome'>('subcortical');

  const brainRegions = {
    cerebellum: {
      primes: [2, 3, 5, 7, 11],
      functions: ['Motor control', 'Balance', 'Coordination', 'Learning'],
      color: 'blue',
    },
    hippocampus: {
      primes: [3, 5, 7, 11, 13],
      functions: ['Memory formation', 'Spatial navigation', 'Consolidation'],
      color: 'purple',
    },
    hypothalamus: {
      primes: [2, 5, 7, 11, 13],
      functions: ['Homeostasis', 'Hormone regulation', 'Circadian rhythms'],
      color: 'green',
    },
  };

  return (
    <div className="space-y-6">
      {/* Main Header */}
      <div className="bg-gradient-to-r from-purple-900/30 via-blue-900/30 to-green-900/30 border border-purple-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Activity className="text-purple-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">7.3-7.4 Brain Structures & Connectome Primes</h2>
            <p className="text-gray-300 text-sm">Prime Operations in Major Brain Structures</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          Prime number operations within major brain structures enable memory, learning, regulation, and 
          complex connectivity patterns throughout the nervous system.
        </p>
      </div>

      {/* View Selector */}
      <div className="flex space-x-4">
        <button
          onClick={() => setActiveView('subcortical')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'subcortical'
              ? 'bg-purple-900/40 border-purple-600 text-purple-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Brain size={20} />
          <span className="font-semibold">7.3 Subcortical Structures</span>
        </button>
        <button
          onClick={() => setActiveView('connectome')}
          className={`flex items-center space-x-2 px-6 py-3 rounded-lg border-2 transition-all ${
            activeView === 'connectome'
              ? 'bg-cyan-900/40 border-cyan-600 text-cyan-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Network size={20} />
          <span className="font-semibold">7.4 Connectome & Networks</span>
        </button>
      </div>

      {/* 7.3 Subcortical Structures */}
      {activeView === 'subcortical' && (
        <div className="space-y-6">
          {/* Cerebellum */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-blue-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Activity size={20} className="text-white" />
              </div>
              <span>Cerebellum: Motor Prime Processing</span>
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-blue-400 font-semibold mb-3">Prime Frequencies</h4>
                  <div className="grid grid-cols-5 gap-2">
                    {brainRegions.cerebellum.primes.map((prime) => (
                      <div
                        key={prime}
                        className="bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg p-3 text-center animate-pulse"
                      >
                        <div className="text-white font-bold">{prime}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
                  <h4 className="text-blue-400 font-semibold mb-2">Functions</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    {brainRegions.cerebellum.functions.map((func) => (
                      <div key={func} className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <span>{func}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">Cerebellar Architecture</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-blue-400">Purkinje Cells:</strong> Fire at prime intervals (2-5 Hz)</p>
                  <p>• <strong className="text-cyan-400">Granule Cells:</strong> ~50 billion (near 47th prime)</p>
                  <p>• <strong className="text-blue-300">Climbing Fibers:</strong> 1:1 ratio (prime unity)</p>
                  <p>• <strong className="text-cyan-300">Parallel Fibers:</strong> Prime branching patterns</p>
                  <p>• <strong className="text-blue-400">Motor Learning:</strong> Long-term depression at 7-11 Hz</p>
                </div>
                <div className="mt-4 bg-blue-900/20 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">
                    The cerebellum contains more neurons than the rest of the brain combined, 
                    organized in prime-based layers for ultra-efficient motor coordination.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Hippocampus */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center">
                <Brain size={20} className="text-white" />
              </div>
              <span>Hippocampus: Memory Prime Encoding</span>
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-purple-400 font-semibold mb-3">Prime Frequencies</h4>
                  <div className="grid grid-cols-5 gap-2">
                    {brainRegions.hippocampus.primes.map((prime) => (
                      <div
                        key={prime}
                        className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg p-3 text-center animate-pulse"
                      >
                        <div className="text-white font-bold">{prime}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-4">
                  <h4 className="text-purple-400 font-semibold mb-2">Functions</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    {brainRegions.hippocampus.functions.map((func) => (
                      <div key={func} className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                        <span>{func}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">Hippocampal Rhythms</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-purple-400">Theta Oscillations:</strong> 5-7 Hz (prime range)</p>
                  <p>• <strong className="text-pink-400">Place Cells:</strong> Fire in prime spatial patterns</p>
                  <p>• <strong className="text-purple-300">Grid Cells:</strong> Hexagonal prime lattice</p>
                  <p>• <strong className="text-pink-300">Sharp Wave Ripples:</strong> 11-13 Hz bursts</p>
                  <p>• <strong className="text-purple-400">Memory Consolidation:</strong> Prime sequence replay</p>
                </div>
                <div className="mt-4 bg-purple-900/20 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">
                    Hippocampus uses theta oscillations in prime frequency bands to encode spatial 
                    and episodic memories with maximal efficiency.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Hypothalamus */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-green-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center">
                <Zap size={20} className="text-white" />
              </div>
              <span>Hypothalamus: Regulatory Prime Control</span>
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-green-400 font-semibold mb-3">Prime Frequencies</h4>
                  <div className="grid grid-cols-5 gap-2">
                    {brainRegions.hypothalamus.primes.map((prime) => (
                      <div
                        key={prime}
                        className="bg-gradient-to-br from-green-600 to-emerald-600 rounded-lg p-3 text-center animate-pulse"
                      >
                        <div className="text-white font-bold">{prime}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-4">
                  <h4 className="text-green-400 font-semibold mb-2">Functions</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    {brainRegions.hypothalamus.functions.map((func) => (
                      <div key={func} className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span>{func}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-3">Prime Regulatory Cycles</h4>
                <div className="space-y-2 text-sm text-gray-300">
                  <p>• <strong className="text-green-400">Circadian Rhythm:</strong> 24 hrs (near 23rd prime hour)</p>
                  <p>• <strong className="text-emerald-400">Ultradian Cycles:</strong> 2, 5, 7 hour oscillations</p>
                  <p>• <strong className="text-green-300">Hormone Pulses:</strong> Prime interval release</p>
                  <p>• <strong className="text-emerald-300">Temperature:</strong> Prime feedback loops</p>
                  <p>• <strong className="text-green-400">Hunger/Satiety:</strong> 3-5 hour prime cycles</p>
                </div>
                <div className="mt-4 bg-green-900/20 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">
                    Hypothalamus orchestrates body regulation through prime-based rhythms and 
                    hormone release patterns for optimal homeostasis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 7.4 Connectome & Networks */}
      {activeView === 'connectome' && (
        <div className="space-y-6">
          {/* Spinal Cord */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Spinal Cord: Prime Segmentation</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-cyan-400 font-semibold mb-3">31 Spinal Segments (Prime Number)</h4>
                <div className="space-y-3">
                  <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                    <div className="text-cyan-400 font-semibold mb-2">Cervical: 8 segments</div>
                    <div className="text-gray-400 text-xs">Controls neck, arms, diaphragm</div>
                  </div>
                  <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                    <div className="text-cyan-400 font-semibold mb-2">Thoracic: 12 segments</div>
                    <div className="text-gray-400 text-xs">Controls trunk and chest muscles</div>
                  </div>
                  <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                    <div className="text-cyan-400 font-semibold mb-2">Lumbar: 5 segments (prime)</div>
                    <div className="text-gray-400 text-xs">Controls legs and hips</div>
                  </div>
                  <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-3">
                    <div className="text-cyan-400 font-semibold mb-2">Sacral: 5 segments (prime)</div>
                    <div className="text-gray-400 text-xs">Controls bowel, bladder, sexual function</div>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-3">Prime Connectivity</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p>• <strong className="text-cyan-400">Reflexes:</strong> 2-synapse prime pathways</p>
                    <p>• <strong className="text-cyan-300">Central Pattern Generators:</strong> Prime oscillations</p>
                    <p>• <strong className="text-cyan-400">Ascending Tracts:</strong> Prime relay stations</p>
                    <p>• <strong className="text-cyan-300">Descending Control:</strong> Prime modulation</p>
                  </div>
                </div>
                <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-600 rounded-lg p-4">
                  <p className="text-gray-300 text-sm">
                    The 31 spinal segments (a prime number) organize motor and sensory information 
                    flow with prime-based efficiency.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Amygdala */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-orange-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Amygdala: Emotional Prime Processing</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-orange-400 font-semibold mb-3">13 Nuclei (Prime Number)</h4>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    'Lateral', 'Basal', 'Accessory Basal', 'Central',
                    'Medial', 'Cortical', 'Intercalated', 'Anterior',
                    'Posterior', 'Basomedial', 'Basolateral', 'Cortico-medial', 'Extended'
                  ].map((nucleus) => (
                    <div key={nucleus} className="bg-orange-900/20 border border-orange-700/30 rounded p-2">
                      <div className="text-orange-400 text-xs font-semibold">{nucleus}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-orange-400 font-semibold mb-3">Prime Emotional Patterns</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p>• <strong className="text-orange-400">Fear Response:</strong> 5-7 Hz theta coupling</p>
                    <p>• <strong className="text-orange-300">Memory Modulation:</strong> Prime enhancement</p>
                    <p>• <strong className="text-orange-400">Social Processing:</strong> Prime network integration</p>
                    <p>• <strong className="text-orange-300">Threat Detection:</strong> Prime vigilance cycles</p>
                  </div>
                </div>
                <div className="bg-gradient-to-r from-orange-900/30 to-red-900/30 border border-orange-600 rounded-lg p-4">
                  <p className="text-gray-300 text-sm">
                    13 amygdala nuclei process emotions through prime frequency patterns, 
                    creating efficient threat detection and emotional memory formation.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Cortical Column */}
          <div className="bg-gray-900/60 backdrop-blur-sm border border-pink-700 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Cortical Column: 6-Layer Prime Architecture</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-950/50 rounded-lg p-4">
                <h4 className="text-pink-400 font-semibold mb-3">Six Cortical Layers</h4>
                <div className="space-y-2">
                  {[
                    { layer: 'I', name: 'Molecular', cells: 'Few neurons, dendrites' },
                    { layer: 'II', name: 'External Granular', cells: 'Stellate cells (prime)' },
                    { layer: 'III', name: 'External Pyramidal', cells: 'Small pyramidal (prime)' },
                    { layer: 'IV', name: 'Internal Granular', cells: 'Stellate, sensory input' },
                    { layer: 'V', name: 'Internal Pyramidal', cells: 'Large pyramidal (prime)' },
                    { layer: 'VI', name: 'Multiform', cells: 'Diverse, thalamic' },
                  ].map((layer) => (
                    <div key={layer.layer} className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-2">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-pink-600 rounded flex items-center justify-center text-white font-bold text-sm">
                          {layer.layer}
                        </div>
                        <div>
                          <div className="text-pink-400 text-sm font-semibold">{layer.name}</div>
                          <div className="text-gray-400 text-xs">{layer.cells}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="space-y-4">
                <div className="bg-gray-950/50 rounded-lg p-4">
                  <h4 className="text-pink-400 font-semibold mb-3">Prime Connectivity Patterns</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p>• <strong className="text-pink-400">Vertical Integration:</strong> 6 layers prime structure</p>
                    <p>• <strong className="text-pink-300">Horizontal Connections:</strong> Prime spacing (~500μm)</p>
                    <p>• <strong className="text-pink-400">Minicolumns:</strong> ~80-100 neurons (near primes)</p>
                    <p>• <strong className="text-pink-300">Hypercolumns:</strong> Prime organization units</p>
                    <p>• <strong className="text-pink-400">Feedback/Feedforward:</strong> Prime layer targeting</p>
                  </div>
                </div>
                <div className="bg-gradient-to-r from-pink-900/30 to-purple-900/30 border border-pink-600 rounded-lg p-4">
                  <p className="text-gray-300 text-sm">
                    Cortical columns organize in 6-layer prime architecture with ~100 neurons per 
                    minicolumn, creating fundamental computational units of cortex.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Connectome Summary */}
          <div className="bg-gradient-to-r from-cyan-900/30 via-orange-900/30 to-pink-900/30 border border-cyan-600 rounded-xl p-6">
            <h3 className="text-white font-bold text-lg mb-4">Unified Connectome Prime Framework</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-4">
                <div className="text-cyan-400 font-semibold mb-2">Spinal Cord</div>
                <div className="text-gray-300 text-sm">31 segments with prime reflex patterns</div>
              </div>
              <div className="bg-orange-900/20 border border-orange-700/30 rounded-lg p-4">
                <div className="text-orange-400 font-semibold mb-2">Amygdala</div>
                <div className="text-gray-300 text-sm">13 nuclei with prime emotional encoding</div>
              </div>
              <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-4">
                <div className="text-pink-400 font-semibold mb-2">Cortical Column</div>
                <div className="text-gray-300 text-sm">6 layers with prime connectivity</div>
              </div>
            </div>
            <p className="text-gray-300 text-sm mt-4">
              <strong className="text-cyan-400">Connectome Integration:</strong> These prime-structured 
              components form an integrated network where information flows through prime pathways, 
              enabling efficient brain-wide computation and consciousness emergence.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
