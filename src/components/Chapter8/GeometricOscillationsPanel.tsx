import React, { useState } from 'react';
import { Circle, Box, Activity, Layers } from 'lucide-react';

/**
 * Section 8.5: Periodic oscillation of capacitance inductance hinductance with its geometry
 */
export const GeometricOscillationsPanel: React.FC = () => {
  const [activeSubsection, setActiveSubsection] = useState<string>('overview');

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-900/20 to-violet-900/20 border border-purple-700/50 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-3">
          <Circle className="text-purple-400" size={32} />
          <h2 className="text-2xl font-bold text-white">
            8.5 Periodic Oscillation of Capacitance, Inductance, Hinductance with Geometry
          </h2>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          Electrical properties (capacitance C, inductance L, and the novel hinductance H) oscillate 
          periodically with geometric parameters. This geometric control enables dynamic tuning of 
          Hinductor behavior through structural modulation.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {[
          { id: 'overview', label: 'Overview', icon: Circle },
          { id: '8.5.1', label: '8.5.1 12-Hole Phase', icon: Box },
          { id: '8.5.2', label: '8.5.2 Periodic Theory', icon: Activity },
          { id: '8.5.3', label: '8.5.3 Darkness Knots', icon: Layers }
        ].map(sub => {
          const Icon = sub.icon;
          return (
            <button
              key={sub.id}
              onClick={() => setActiveSubsection(sub.id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all ${
                activeSubsection === sub.id
                  ? 'bg-purple-900/40 border-purple-600 text-purple-300'
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
            <h3 className="text-xl font-bold text-purple-300 mb-4">Geometric Property Control</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-4">
                <h4 className="text-purple-400 font-semibold mb-2">Capacitance C(θ)</h4>
                <p className="text-sm text-gray-300">Varies with spiral angle θ. Controls charge storage capability based on cylinder spacing and overlap geometry.</p>
              </div>
              <div className="bg-violet-900/20 border border-violet-700/30 rounded-lg p-4">
                <h4 className="text-violet-400 font-semibold mb-2">Inductance L(r)</h4>
                <p className="text-sm text-gray-300">Depends on radius r. Magnetic field energy storage modulated by cylinder diameter and pitch ratio.</p>
              </div>
              <div className="bg-fuchsia-900/20 border border-fuchsia-700/30 rounded-lg p-4">
                <h4 className="text-fuchsia-400 font-semibold mb-2">Hinductance H(r,θ)</h4>
                <p className="text-sm text-gray-300">Novel parameter coupling flux and charge. Depends on both radius and angle for complete geometric control.</p>
              </div>
            </div>
          </div>
        )}
        {activeSubsection === '8.5.1' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-purple-300 mb-3">8.5.1 The Concept of Phase Space with 12 Holes That Blinks</h3>
            <p className="text-gray-300 text-sm">
              A 12-dimensional phase space with discrete "holes" (access points) that appear and disappear 
              dynamically. These blinking holes represent temporal windows for consciousness interaction with 
              the Hinductor system, following prime-based timing patterns.
            </p>
          </div>
        )}
        {activeSubsection === '8.5.2' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-purple-300 mb-3">8.5.2 Periodic Oscillations with Length, Pitch, Lattice: Theory & Experiment</h3>
            <p className="text-gray-300 text-sm">
              Systematic study of how electrical properties oscillate with geometric parameters: cylinder length L, 
              spiral pitch P, and lattice spacing a. Both theoretical models and experimental validation confirm 
              periodic relationships enabling precise H device design.
            </p>
          </div>
        )}
        {activeSubsection === '8.5.3' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-purple-300 mb-3">8.5.3 Knots of Darkness on the H Interface</h3>
            <p className="text-gray-300 text-sm">
              Dark regions forming knot-like structures on Hinductor interfaces. These darkness knots represent 
              magnetic light absorption or interference zones, creating functional interface patterns for 
              information encoding and consciousness coupling.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
