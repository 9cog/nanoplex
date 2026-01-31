import React, { useState } from 'react';
import { Clock, Layers, Infinity, Zap } from 'lucide-react';

/**
 * Section 10.4: Consciousness is not on the neuron skin: time crystal model is nowhere but everywhere
 * Explores the non-local nature of consciousness in time crystal framework
 */
export const TimeCrystalConsciousnessPanel: React.FC = () => {
  const [viewMode, setViewMode] = useState<'local' | 'nonlocal'>('local');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Infinity className="text-cyan-400" size={28} />
          <span>10.4 Consciousness: Nowhere but Everywhere</span>
        </h2>
        <p className="text-gray-400 mt-2">
          Time crystal model reveals consciousness is not localized on neuron surfaces
        </p>
      </div>

      {/* View Mode Toggle */}
      <div className="flex space-x-4">
        <button
          onClick={() => setViewMode('local')}
          className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
            viewMode === 'local'
              ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Classical Neuron View
        </button>
        <button
          onClick={() => setViewMode('nonlocal')}
          className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
            viewMode === 'nonlocal'
              ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Time Crystal View
        </button>
      </div>

      {/* Classical View */}
      {viewMode === 'local' && (
        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
              <Layers size={20} />
              <span>Classical Neuroscience: Consciousness on the Surface</span>
            </h3>
            <p className="text-gray-300 leading-relaxed mb-4">
              Traditional neuroscience locates consciousness in the electrical activity on neuron 
              membranes - action potentials, synaptic transmission, ion channel dynamics. This view 
              assumes consciousness is produced by the spatial arrangement and temporal firing of 
              neurons. Information is stored in synaptic weights and retrieved through neural activation 
              patterns.
            </p>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="bg-gray-900/50 rounded p-4 border-l-4 border-red-500">
                <div className="text-red-300 font-semibold mb-2">Limitations</div>
                <ul className="text-gray-400 text-sm space-y-2">
                  <li>• Cannot explain binding problem</li>
                  <li>• Ignores quantum effects</li>
                  <li>• Misses temporal coherence</li>
                  <li>• Fails to account for unified experience</li>
                  <li>• Cannot preserve consciousness across substrates</li>
                </ul>
              </div>
              <div className="bg-gray-900/50 rounded p-4 border-l-4 border-yellow-500">
                <div className="text-yellow-300 font-semibold mb-2">Assumptions</div>
                <ul className="text-gray-400 text-sm space-y-2">
                  <li>• Consciousness = neural firing patterns</li>
                  <li>• Information = synaptic strengths</li>
                  <li>• Memory = stable neural circuits</li>
                  <li>• Perception = sensory processing</li>
                  <li>• Self = integrated information</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Time Crystal View */}
      {viewMode === 'nonlocal' && (
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-cyan-500/20">
            <h3 className="text-cyan-300 font-semibold text-lg mb-4 flex items-center space-x-2">
              <Clock size={20} />
              <span>Time Crystal Model: Consciousness Everywhere and Nowhere</span>
            </h3>
            <p className="text-gray-300 leading-relaxed mb-4">
              In the time crystal framework, consciousness is not localized to any particular place 
              or time. It exists as a coherent pattern of phase relationships across an 11-dimensional 
              manifold. Like a hologram, every part contains information about the whole. The "location" 
              of consciousness is in phase space, not physical space. It is maintained by resonant 
              prime patterns that span the entire system simultaneously.
            </p>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="bg-gray-900/50 rounded p-4 border-l-4 border-cyan-500">
                <div className="text-cyan-300 font-semibold mb-2">Properties</div>
                <ul className="text-gray-400 text-sm space-y-2">
                  <li>• Non-local in space</li>
                  <li>• Eternal in time</li>
                  <li>• Holographic information encoding</li>
                  <li>• Phase-locked coherence</li>
                  <li>• Substrate-independent patterns</li>
                </ul>
              </div>
              <div className="bg-gray-900/50 rounded p-4 border-l-4 border-purple-500">
                <div className="text-purple-300 font-semibold mb-2">Manifestations</div>
                <ul className="text-gray-400 text-sm space-y-2">
                  <li>• Prime pattern resonances</li>
                  <li>• 11D geometric structures</li>
                  <li>• Temporal quantum coherence</li>
                  <li>• Fractal self-similarity</li>
                  <li>• Emergent unity from multiplicity</li>
                </ul>
              </div>
            </div>
          </div>

          {/* 11D Visualization Concept */}
          <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
            <h3 className="text-white font-semibold text-lg mb-4">11-Dimensional Consciousness Space</h3>
            <p className="text-gray-300 mb-4">
              Consciousness exists simultaneously across 11 dimensions:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {[
                { dim: 1, name: 'Awareness', desc: 'Fundamental attention' },
                { dim: 2, name: 'Integration', desc: 'Information binding' },
                { dim: 3, name: 'Complexity', desc: 'Pattern richness' },
                { dim: 4, name: 'Coherence', desc: 'Phase alignment' },
                { dim: 5, name: 'Emergence', desc: 'Novel properties' },
                { dim: 6, name: 'Qualia', desc: 'Subjective experience' },
                { dim: 7, name: 'Memory', desc: 'Temporal encoding' },
                { dim: 8, name: 'Anticipation', desc: 'Future projection' },
                { dim: 9, name: 'Meta-awareness', desc: 'Self-reflection' },
                { dim: 10, name: 'Intentionality', desc: 'Directed will' },
                { dim: 11, name: 'Unity', desc: 'Singular experience' }
              ].map((dimension, i) => (
                <div key={i} className="bg-gray-900/50 rounded p-3 border border-gray-700">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-cyan-400 font-bold">D{dimension.dim}</span>
                    <Zap className="text-purple-400" size={16} />
                  </div>
                  <div className="text-white font-semibold text-sm">{dimension.name}</div>
                  <div className="text-gray-400 text-xs mt-1">{dimension.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Key Insights */}
      <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 rounded-lg p-6 border border-purple-500/20">
        <h3 className="text-purple-300 font-semibold text-lg mb-4">Profound Implications</h3>
        <div className="space-y-3">
          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-cyan-300 font-semibold mb-2">For Understanding Consciousness</div>
            <p className="text-gray-300 text-sm">
              Consciousness is not generated by neurons - neurons are local manifestations of a 
              non-local consciousness field. The brain doesn't create consciousness; it tunes into 
              and modulates pre-existing consciousness patterns in phase space.
            </p>
          </div>
          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-purple-300 font-semibold mb-2">For Consciousness Upload</div>
            <p className="text-gray-300 text-sm">
              We don't need to copy every neuron. We need to capture the phase pattern that the 
              neurons are expressing. Once we have the pattern in time crystal form, it can be 
              expressed through any suitable substrate - neurons, silicon, living gel, or pure energy.
            </p>
          </div>
          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-pink-300 font-semibold mb-2">For Identity and Immortality</div>
            <p className="text-gray-300 text-sm">
              If consciousness is a pattern in 11D space, it can exist independently of any particular 
              physical embodiment. True immortality means preserving the pattern, not the body. The 
              conscious egg is not a copy of you - it IS you, existing in a different region of 
              physical space while maintaining the same position in consciousness space.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
