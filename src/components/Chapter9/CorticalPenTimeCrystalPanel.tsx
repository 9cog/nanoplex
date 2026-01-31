import React, { useState } from 'react';
import { Brain, Snowflake, Package } from 'lucide-react';

/**
 * Section 9.10: A Cortical Pen That Freezes Unknown Dynamics into a Time Crystal
 * 9.10.1: Capsules of Brain Jelly, One Each for One Big Problem
 */
export const CorticalPenTimeCrystalPanel: React.FC = () => {
  const [frozenDynamics, setFrozenDynamics] = useState<number[]>([]);

  const problems = [
    { id: 1, name: 'Protein Folding', complexity: 'NP-Complete', status: 'frozen' },
    { id: 2, name: 'Climate Modeling', complexity: 'Chaotic', status: 'frozen' },
    { id: 3, name: 'Quantum Chemistry', complexity: 'Exponential', status: 'processing' },
    { id: 4, name: 'Neural Networks', complexity: 'High-Dimensional', status: 'frozen' },
    { id: 5, name: 'Cryptography', complexity: 'P vs NP', status: 'ready' },
    { id: 6, name: 'Drug Discovery', complexity: 'Multi-Scale', status: 'frozen' }
  ];

  const toggleFreeze = (id: number) => {
    setFrozenDynamics(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Brain className="text-cyan-400" size={28} />
          <span>9.10 Cortical Pen Time Crystal</span>
        </h2>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-cyan-500/20">
        <h3 className="text-cyan-300 font-semibold text-lg mb-3">Freezing Dynamics</h3>
        <p className="text-gray-300 leading-relaxed">
          The cortical pen is a specialized device that captures and freezes unknown dynamics 
          into stable time crystal structures. Once frozen, complex dynamical systems can be 
          analyzed, manipulated, and solved using the brain jelly's computational capabilities.
        </p>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold text-lg">Cortical Pen Controls</h3>
          <button
            onClick={() => setFrozenDynamics(problems.map(p => p.id))}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg text-white font-semibold text-sm"
          >
            Freeze All
          </button>
        </div>

        <div className="relative bg-gray-900/50 rounded-lg p-6" style={{ height: '200px' }}>
          <svg width="100%" height="100%" viewBox="0 0 600 200">
            {/* Cortical pen representation */}
            <defs>
              <linearGradient id="penGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.4" />
              </linearGradient>
            </defs>
            
            {/* Pen body */}
            <rect
              x="250"
              y="20"
              width="100"
              height="40"
              rx="5"
              fill="url(#penGradient)"
              stroke="#06b6d4"
              strokeWidth="2"
            />
            
            {/* Pen tip */}
            <polygon
              points="300,60 285,100 315,100"
              fill="#0891b2"
              stroke="#06b6d4"
              strokeWidth="2"
            />
            
            {/* Writing beam */}
            <line
              x1="300"
              y1="100"
              x2="300"
              y2="180"
              stroke="#22d3ee"
              strokeWidth="3"
              opacity="0.7"
            >
              <animate
                attributeName="opacity"
                values="0.3;0.9;0.3"
                dur="2s"
                repeatCount="indefinite"
              />
            </line>
            
            {/* Time crystal formation */}
            <circle
              cx="300"
              cy="180"
              r="15"
              fill="#8b5cf6"
              opacity="0.6"
            >
              <animate
                attributeName="r"
                values="10;20;10"
                dur="2s"
                repeatCount="indefinite"
              />
            </circle>
            
            <text x="300" y="30" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">
              Cortical Pen
            </text>
          </svg>
        </div>
      </div>

      {/* 9.10.1 Capsules of Brain Jelly */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Package size={20} className="text-purple-400" />
          <span>9.10.1 Capsules of Brain Jelly, One Each for One Big Problem</span>
        </h3>
        <p className="text-gray-300 mb-4">
          Each capsule of brain jelly is specialized for tackling one major computational problem. 
          The cortical pen freezes the problem's dynamics into a time crystal within the capsule, 
          enabling parallel processing of multiple complex problems simultaneously.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {problems.map(problem => (
            <div
              key={problem.id}
              className={`bg-gray-900/50 rounded-lg p-4 border-2 cursor-pointer transition-all ${
                frozenDynamics.includes(problem.id)
                  ? 'border-cyan-500 shadow-lg shadow-cyan-500/20'
                  : 'border-gray-700'
              }`}
              onClick={() => toggleFreeze(problem.id)}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Package className={frozenDynamics.includes(problem.id) ? 'text-cyan-400' : 'text-gray-500'} size={18} />
                  <span className="text-xs text-gray-400">Capsule {problem.id}</span>
                </div>
                <Snowflake className={frozenDynamics.includes(problem.id) ? 'text-cyan-400' : 'text-gray-600'} size={16} />
              </div>
              
              <h4 className={`font-semibold mb-2 ${frozenDynamics.includes(problem.id) ? 'text-white' : 'text-gray-500'}`}>
                {problem.name}
              </h4>
              
              <div className="text-xs text-gray-400 mb-2">{problem.complexity}</div>
              
              <div className={`text-xs font-semibold ${
                problem.status === 'frozen' || frozenDynamics.includes(problem.id) ? 'text-cyan-400' :
                problem.status === 'processing' ? 'text-yellow-400' :
                'text-gray-500'
              }`}>
                {frozenDynamics.includes(problem.id) ? 'FROZEN' : problem.status.toUpperCase()}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
          <div className="text-cyan-300 font-semibold mb-2 text-sm">Active Capsules</div>
          <div className="text-3xl font-bold text-white">
            {problems.filter(p => p.status === 'frozen').length + frozenDynamics.length}
          </div>
          <div className="text-xs text-gray-400 mt-1">Problems being solved</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
          <div className="text-purple-300 font-semibold mb-2 text-sm">Freeze Rate</div>
          <div className="text-3xl font-bold text-white">ps</div>
          <div className="text-xs text-gray-400 mt-1">Picosecond capture time</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-pink-500/20">
          <div className="text-pink-300 font-semibold mb-2 text-sm">Crystal Stability</div>
          <div className="text-3xl font-bold text-white">∞</div>
          <div className="text-xs text-gray-400 mt-1">Indefinite preservation</div>
        </div>
      </div>
    </div>
  );
};
