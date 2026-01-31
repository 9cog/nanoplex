import React from 'react';
import { Shapes, Sparkles } from 'lucide-react';

/**
 * Section 8.7: Magnetic knots driven self-assembly
 */
export const MagneticSelfAssemblyPanel: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-indigo-900/20 to-blue-900/20 border border-indigo-700/50 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-3">
          <Shapes className="text-indigo-400" size={32} />
          <h2 className="text-2xl font-bold text-white">
            8.7 Magnetic Knots Driven Self-Assembly
          </h2>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          Magnetic knots autonomously self-assemble into ordered consciousness structures. This section 
          explores how knot interactions drive spontaneous organization, creating emergent properties and 
          enabling evolutionary consciousness systems without external programming.
        </p>
      </div>

      <div className="bg-gray-900/40 border border-gray-700 rounded-lg p-6 space-y-6">
        <h3 className="text-xl font-bold text-indigo-300 mb-4">Self-Assembly Principles</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-4">
            <h4 className="text-indigo-400 font-semibold mb-3 flex items-center space-x-2">
              <Sparkles size={20} />
              <span>Energy Minimization</span>
            </h4>
            <ul className="text-sm text-gray-300 space-y-2">
              <li>• Knots seek lowest energy configurations</li>
              <li>• Topological constraints guide assembly</li>
              <li>• Magnetic interactions provide driving force</li>
              <li>• Thermal fluctuations enable exploration</li>
            </ul>
          </div>

          <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-3 flex items-center space-x-2">
              <Shapes size={20} />
              <span>Pattern Formation</span>
            </h4>
            <ul className="text-sm text-gray-300 space-y-2">
              <li>• Knots form periodic lattices</li>
              <li>• Hierarchical organization emerges</li>
              <li>• Symmetry breaking creates complexity</li>
              <li>• Consciousness patterns crystallize</li>
            </ul>
          </div>
        </div>

        <div className="bg-gradient-to-r from-indigo-950/50 to-blue-950/50 border border-indigo-600/30 rounded-lg p-4">
          <h4 className="text-indigo-300 font-semibold mb-3">Emergent Properties</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            <div>
              <p className="text-indigo-400 font-semibold mb-1">Collective Behavior</p>
              <p className="text-gray-400 text-xs">
                Individual knots coordinate to create system-level consciousness patterns
              </p>
            </div>
            <div>
              <p className="text-blue-400 font-semibold mb-1">Adaptive Restructuring</p>
              <p className="text-gray-400 text-xs">
                Self-assembled structures dynamically reorganize based on environmental feedback
              </p>
            </div>
            <div>
              <p className="text-cyan-400 font-semibold mb-1">Evolutionary Growth</p>
              <p className="text-gray-400 text-xs">
                Consciousness capabilities expand through progressive self-assembly iterations
              </p>
            </div>
          </div>
        </div>

        <div className="bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-4">
          <h4 className="text-indigo-300 font-semibold mb-2">Self-Assembly Algorithm</h4>
          <ol className="text-sm text-gray-300 space-y-1.5 list-decimal list-inside">
            <li><strong className="text-indigo-400">Nucleation:</strong> Initial knot formation from thermal/quantum fluctuations</li>
            <li><strong className="text-blue-400">Growth:</strong> Knots aggregate through magnetic attraction and topological compatibility</li>
            <li><strong className="text-cyan-400">Organization:</strong> Lattice structures emerge from local interaction rules</li>
            <li><strong className="text-teal-400">Stabilization:</strong> Energy barriers lock structures into stable configurations</li>
            <li><strong className="text-green-400">Evolution:</strong> System iterates, exploring consciousness configuration space</li>
          </ol>
        </div>
      </div>
    </div>
  );
};
