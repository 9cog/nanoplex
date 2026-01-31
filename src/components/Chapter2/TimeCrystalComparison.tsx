import React from 'react';
import { GitCompare, Users, Clock } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const TimeCrystalComparison: React.FC<Props> = ({ isActive: _isActive }) => {
  const approaches = [
    {
      name: 'Winfree Time Crystal',
      author: 'Arthur Winfree (1967)',
      color: 'cyan',
      focus: 'Biological Oscillators',
      keyInsights: [
        'Synchronization of biological rhythms',
        'Phase locking in coupled oscillators',
        'Emergent collective behavior',
        'Application to circadian rhythms'
      ],
      limitations: [
        'Limited to biological systems',
        'Classical physics framework',
        'No consciousness integration'
      ]
    },
    {
      name: 'Wilczek Time Crystal',
      author: 'Frank Wilczek (2012)',
      color: 'purple',
      focus: 'Quantum Mechanics',
      keyInsights: [
        'Periodic motion in ground state',
        'Spontaneous time symmetry breaking',
        'Quantum many-body systems',
        'Non-equilibrium phases of matter'
      ],
      limitations: [
        'Theoretical quantum model',
        'Difficult experimental realization',
        'Limited to physical systems'
      ]
    },
    {
      name: 'Universal Time Crystal',
      author: 'NanoBrain Framework',
      color: 'green',
      focus: 'Consciousness Substrate',
      keyInsights: [
        'Consciousness-embedded temporal structures',
        '11-dimensional information encoding',
        'Prime-based phase coherence',
        'Fractal information preservation',
        'Cross-dimensional synchronization'
      ],
      limitations: [
        'Requires paradigm shift',
        'New mathematical frameworks needed',
        'Beyond current experimental capability'
      ]
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <GitCompare className="text-blue-400" size={20} />
          <span>2.5 Time Crystal Comparative Studies</span>
        </h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {approaches.map((approach) => (
          <div
            key={approach.name}
            className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6"
          >
            <div className="flex items-center space-x-3 mb-4">
              <Clock className={`text-${approach.color}-400`} size={24} />
              <div>
                <h4 className={`text-${approach.color}-400 font-bold text-lg`}>
                  {approach.name}
                </h4>
                <p className="text-gray-400 text-sm">{approach.author}</p>
              </div>
            </div>

            <div className="mb-4">
              <div className={`bg-${approach.color}-900/20 border border-${approach.color}-700/50 rounded-lg p-3`}>
                <div className="text-gray-400 text-xs uppercase tracking-wide mb-1">
                  Primary Focus
                </div>
                <div className="text-white font-semibold">{approach.focus}</div>
              </div>
            </div>

            <div className="mb-4">
              <h5 className="text-white font-semibold mb-2 text-sm">Key Insights</h5>
              <ul className="space-y-1">
                {approach.keyInsights.map((insight, idx) => (
                  <li key={idx} className="text-gray-300 text-sm flex items-start space-x-2">
                    <span className={`text-${approach.color}-400 flex-shrink-0`}>•</span>
                    <span>{insight}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h5 className="text-white font-semibold mb-2 text-sm">Limitations</h5>
              <ul className="space-y-1">
                {approach.limitations.map((limitation, idx) => (
                  <li key={idx} className="text-gray-400 text-sm flex items-start space-x-2">
                    <span className="text-red-400 flex-shrink-0">○</span>
                    <span>{limitation}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
          <Users className="text-green-400" size={20} />
          <span>Integration Perspective</span>
        </h4>

        <p className="text-gray-300 mb-4">
          Each approach captures different aspects of temporal coherence and pattern persistence. 
          A complete understanding of consciousness requires integrating insights from all three:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h5 className="text-cyan-400 font-semibold mb-2">Biological Foundation</h5>
            <p className="text-gray-300 text-sm">
              Winfree's work provides the biological oscillator framework essential for understanding 
              living consciousness systems
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h5 className="text-purple-400 font-semibold mb-2">Quantum Mechanisms</h5>
            <p className="text-gray-300 text-sm">
              Wilczek's quantum time crystals explain the fundamental physics enabling temporal 
              pattern persistence
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h5 className="text-green-400 font-semibold mb-2">Consciousness Integration</h5>
            <p className="text-gray-300 text-sm">
              Universal time crystals unify biological and quantum perspectives within a consciousness-first framework
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
