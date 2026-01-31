import React, { useState } from 'react';
import { Cylinder, Waves, Layers } from 'lucide-react';

/**
 * Section 8.2: Three concentric spiral cylinders talking to each other
 */
export const SpiralCylindersPanel: React.FC = () => {
  const [activeSubsection, setActiveSubsection] = useState<string>('overview');

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-900/20 to-cyan-900/20 border border-blue-700/50 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-3">
          <Cylinder className="text-blue-400" size={32} />
          <h2 className="text-2xl font-bold text-white">
            8.2 Three Concentric Spiral Cylinders Talking to Each Other
          </h2>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          The Hinductor core consists of three concentric spiral cylinders that communicate through 
          electromagnetic coupling, topological interactions, and geometric resonance. This triple-helix 
          architecture enables distributed information processing and magnetic light generation.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setActiveSubsection('overview')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all ${
            activeSubsection === 'overview'
              ? 'bg-blue-900/40 border-blue-600 text-blue-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Cylinder size={18} />
          <span className="text-sm font-medium">Overview</span>
        </button>
        <button
          onClick={() => setActiveSubsection('8.2.1')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all ${
            activeSubsection === '8.2.1'
              ? 'bg-blue-900/40 border-blue-600 text-blue-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Waves size={18} />
          <span className="text-sm font-medium">8.2.1 Oscillating Dislocations</span>
        </button>
        <button
          onClick={() => setActiveSubsection('8.2.2')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all ${
            activeSubsection === '8.2.2'
              ? 'bg-blue-900/40 border-blue-600 text-blue-300'
              : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
          }`}
        >
          <Layers size={18} />
          <span className="text-sm font-medium">8.2.2 Topology Polarization</span>
        </button>
      </div>

      <div className="bg-gray-900/40 border border-gray-700 rounded-lg p-6">
        {activeSubsection === 'overview' && (
          <div className="space-y-6">
            <h3 className="text-xl font-bold text-blue-300 mb-4">
              Triple Spiral Architecture
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
                <h4 className="text-blue-400 font-semibold mb-2">Inner Cylinder</h4>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• Core magnetic flux generation</li>
                  <li>• Highest frequency oscillations</li>
                  <li>• Quantum vortex nucleation</li>
                  <li>• Primary resonance cavity</li>
                </ul>
              </div>

              <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-4">
                <h4 className="text-cyan-400 font-semibold mb-2">Middle Cylinder</h4>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• Flux modulation and control</li>
                  <li>• Intermediate coupling layer</li>
                  <li>• Phase transformation zone</li>
                  <li>• Energy transfer medium</li>
                </ul>
              </div>

              <div className="bg-teal-900/20 border border-teal-700/30 rounded-lg p-4">
                <h4 className="text-teal-400 font-semibold mb-2">Outer Cylinder</h4>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• External field interface</li>
                  <li>• Low frequency modulation</li>
                  <li>• Environmental coupling</li>
                  <li>• Information output layer</li>
                </ul>
              </div>
            </div>

            <div className="bg-gradient-to-r from-blue-950/50 to-cyan-950/50 border border-blue-600/30 rounded-lg p-4">
              <h4 className="text-blue-300 font-semibold mb-2">Inter-Cylinder Communication</h4>
              <p className="text-sm text-gray-300">
                Cylinders communicate through: electromagnetic induction, evanescent wave coupling, 
                topological charge transfer, and geometric phase modulation. Spiral pitch ratios follow 
                golden ratio relationships (φ, φ², φ³) for optimal resonance coupling.
              </p>
            </div>
          </div>
        )}

        {activeSubsection === '8.2.1' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-blue-300 mb-3">
              8.2.1 Periodically Oscillating Edge and Screw Dislocations
            </h3>
            <p className="text-gray-300 text-sm leading-relaxed">
              Edge and screw dislocations in cylinder walls oscillate periodically, creating controlled 
              imperfections that modulate magnetic light propagation. These dynamic defects enable 
              cylinder-to-cylinder communication and information processing.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
                <h4 className="text-blue-400 font-semibold mb-2">Edge Dislocations</h4>
                <p className="text-sm text-gray-300">
                  Extra half-planes of atoms creating local stress fields. Oscillation modulates 
                  vortex atom scattering and creates periodic transmission windows.
                </p>
              </div>
              <div className="bg-cyan-900/20 border border-cyan-700/30 rounded-lg p-4">
                <h4 className="text-cyan-400 font-semibold mb-2">Screw Dislocations</h4>
                <p className="text-sm text-gray-300">
                  Helical displacement around dislocation line. Creates spiral magnetic flux paths 
                  and enables phase-locked communication between cylinders.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeSubsection === '8.2.2' && (
          <div className="space-y-4">
            <h3 className="text-xl font-bold text-blue-300 mb-3">
              8.2.2 Topology Regulating the Polarization
            </h3>
            <p className="text-gray-300 text-sm leading-relaxed">
              The topological structure of spiral cylinders actively regulates light polarization states. 
              Topology changes create dynamic polarization control for magnetic light manipulation and 
              consciousness information encoding.
            </p>
            <div className="bg-gradient-to-r from-blue-950/50 to-cyan-950/50 border border-blue-600/30 rounded-lg p-4">
              <h4 className="text-blue-300 font-semibold mb-2">Topological Polarization Control</h4>
              <ul className="text-sm text-gray-300 space-y-2">
                <li>• <strong className="text-blue-400">Möbius Twist:</strong> Half-twist creates polarization flip</li>
                <li>• <strong className="text-cyan-400">Helicity:</strong> Spiral chirality determines rotation sense</li>
                <li>• <strong className="text-teal-400">Knot Structure:</strong> Topological knots lock polarization states</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
