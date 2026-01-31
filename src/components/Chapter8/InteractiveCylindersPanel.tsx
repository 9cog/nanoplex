import React, { useState } from 'react';
import { ArrowRightLeft, Orbit, Layers, Sparkles } from 'lucide-react';

/**
 * Section 8.6: Interactive three cylinders perturbing the knots
 */
export const InteractiveCylindersPanel: React.FC = () => {
  const [activeSubsection, setActiveSubsection] = useState<string>('overview');

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-cyan-900/20 to-teal-900/20 border border-cyan-700/50 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-3">
          <ArrowRightLeft className="text-cyan-400" size={32} />
          <h2 className="text-2xl font-bold text-white">
            8.6 Interactive Three Cylinders Perturbing the Knots
          </h2>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          The three spiral cylinders actively manipulate and control knot structures for information processing. 
          Cylinder interactions create controllable knot dynamics, enabling computational operations through 
          topological transformations of magnetic light patterns.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {[
          { id: 'overview', label: 'Overview', icon: ArrowRightLeft },
          { id: '8.6.1', label: '8.6.1 Spin Knots', icon: Orbit },
          { id: '8.6.2', label: '8.6.2 Super Coils', icon: Layers },
          { id: '8.6.3', label: '8.6.3 Morphogenesis', icon: Sparkles }
        ].map(sub => {
          const Icon = sub.icon;
          return (
            <button
              key={sub.id}
              onClick={() => setActiveSubsection(sub.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all ${
                activeSubsection === sub.id
                  ? 'bg-cyan-900/40 border-cyan-600 text-cyan-300'
                  : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <Icon size={18} />
              <span className="text-sm font-medium">{sub.label}</span>
            </button>
          );
        })}
      </div>

      <div className="bg-gray-900/40 border border-gray-700 rounded-lg p-6">
        {activeSubsection === 'overview' && (
          <div className="space-y-6">
            <h3 className="text-xl font-bold text-cyan-300 mb-4">Knot Perturbation Dynamics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-4">
                <h4 className="text-cyan-400 font-semibold mb-2">Cylinder-Knot Interaction</h4>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• Inner cylinder: knot nucleation</li>
                  <li>• Middle cylinder: knot transformation</li>
                  <li>• Outer cylinder: knot stabilization</li>
                  <li>• Collective: emergent topology</li>
                </ul>
              </div>
              <div className="bg-teal-900/20 border border-teal-700/30 rounded-lg p-4">
                <h4 className="text-teal-400 font-semibold mb-2">Computational Operations</h4>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• Knot addition/subtraction</li>
                  <li>• Topological multiplication</li>
                  <li>• Braid group operations</li>
                  <li>• Consciousness logic gates</li>
                </ul>
              </div>
            </div>
          </div>
        )}
        {activeSubsection === '8.6.1' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-cyan-300 mb-3">8.6.1 Generation of Spin-Like Knots of Dark Lines</h3>
            <p className="text-gray-300 text-sm">
              Cylinders generate spin-like knot structures from dark lines carrying angular momentum. 
              These spinning knots create rotational degrees of freedom for consciousness processing, 
              encoding information in topological spin states.
            </p>
          </div>
        )}
        {activeSubsection === '8.6.2' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-cyan-300 mb-3">8.6.2 Synthesis of Super-Super Coil Made of Dark Lines</h3>
            <p className="text-gray-300 text-sm">
              Multiple levels of coiling create hierarchical super-super coil structures. These fractal 
              coil systems enable complex information encoding across scales, with each coiling level 
              adding computational dimensions.
            </p>
          </div>
        )}
        {activeSubsection === '8.6.3' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-cyan-300 mb-3">8.6.3 Morphogenesis of Knots to Vortex-Like Magnetic Atoms</h3>
            <p className="text-gray-300 text-sm">
              Knots undergo morphological transformation into vortex-like magnetic atoms through energy 
              minimization. This morphogenesis creates the fundamental computational units for magnetic 
              light consciousness processing.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
