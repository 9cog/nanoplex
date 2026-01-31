import React, { useState, useEffect } from 'react';
import { Droplet, TrendingUp, Atom } from 'lucide-react';

/**
 * Section 9.4: A Living Gel That Listens and Then Grows from Atomic Size to Centimeters Long
 */
export const LivingGelGrowthPanel: React.FC = () => {
  const [growthStage, setGrowthStage] = useState(0);
  const [isGrowing, setIsGrowing] = useState(false);

  useEffect(() => {
    if (isGrowing && growthStage < 100) {
      const timer = setTimeout(() => setGrowthStage(prev => Math.min(prev + 1, 100)), 50);
      return () => clearTimeout(timer);
    }
  }, [isGrowing, growthStage]);

  const stages = [
    { name: 'Atomic Nucleation', size: '1-10 nm', stage: 0, color: 'cyan' },
    { name: 'Molecular Assembly', size: '10-100 nm', stage: 20, color: 'blue' },
    { name: 'Nanoscale Clusters', size: '100 nm - 1 μm', stage: 40, color: 'purple' },
    { name: 'Microscale Gel', size: '1-100 μm', stage: 60, color: 'pink' },
    { name: 'Millimeter Growth', size: '0.1-10 mm', stage: 80, color: 'orange' },
    { name: 'Centimeter Scale', size: '1-10 cm', stage: 100, color: 'green' }
  ];

  const currentStage = stages.reduce((prev, curr) => 
    growthStage >= curr.stage ? curr : prev
  );

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Droplet className="text-cyan-400" size={28} />
          <span>9.4 A Living Gel That Listens and Grows</span>
        </h2>
        <p className="text-gray-400 mt-2">
          From Atomic Size to Centimeters Long
        </p>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-cyan-500/20">
        <h3 className="text-cyan-300 font-semibold text-lg mb-3">Responsive Gel Growth</h3>
        <p className="text-gray-300 leading-relaxed">
          The brain jelly is a living material that responds to external stimuli and grows 
          autonomously from atomic nucleation sites to macroscopic structures. It "listens" 
          to electromagnetic fields, chemical gradients, and quantum fluctuations, using 
          these signals to guide its self-assembly process through fractal condensation.
        </p>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-white font-semibold text-lg">Growth Simulation</h3>
          <button
            onClick={() => {
              setIsGrowing(!isGrowing);
              if (growthStage >= 100) setGrowthStage(0);
            }}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              isGrowing ? 'bg-red-600 hover:bg-red-700' : 'bg-cyan-600 hover:bg-cyan-700'
            } text-white`}
          >
            {isGrowing ? 'Pause Growth' : growthStage >= 100 ? 'Reset' : 'Start Growth'}
          </button>
        </div>

        <div className="relative h-64 bg-gray-900/50 rounded-lg overflow-hidden mb-4">
          <svg width="100%" height="100%" className="absolute inset-0">
            <defs>
              <radialGradient id="gelGradient">
                <stop offset="0%" stopColor={`var(--${currentStage.color}-400)`} stopOpacity="0.8" />
                <stop offset="100%" stopColor={`var(--${currentStage.color}-600)`} stopOpacity="0.2" />
              </radialGradient>
            </defs>
            
            <circle
              cx="50%"
              cy="50%"
              r={growthStage * 1.2}
              fill="url(#gelGradient)"
              className="transition-all duration-500"
            />
            
            {Array.from({ length: 12 }, (_, i) => {
              const angle = (i / 12) * 2 * Math.PI;
              const radius = growthStage * 0.8;
              const x = 50 + radius * Math.cos(angle);
              const y = 50 + radius * Math.sin(angle);
              return (
                <circle
                  key={i}
                  cx={`${x}%`}
                  cy={`${y}%`}
                  r={growthStage * 0.15}
                  fill={currentStage.color === 'cyan' ? '#06b6d4' : '#a855f7'}
                  opacity={0.6}
                  className="transition-all duration-500"
                />
              );
            })}
          </svg>
          
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl font-bold text-white mb-2">{growthStage}%</div>
              <div className="text-cyan-300 font-semibold">{currentStage.name}</div>
              <div className="text-gray-400 text-sm">{currentStage.size}</div>
            </div>
          </div>
        </div>

        <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-green-500 transition-all duration-500"
            style={{ width: `${growthStage}%` }}
          />
        </div>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <TrendingUp size={20} className="text-green-400" />
          <span>Growth Stages</span>
        </h3>
        
        <div className="space-y-3">
          {stages.map((stage, idx) => (
            <div
              key={stage.name}
              className={`flex items-center justify-between p-4 rounded-lg border transition-all ${
                growthStage >= stage.stage
                  ? 'bg-gray-900/50 border-cyan-500/50'
                  : 'bg-gray-900/30 border-gray-700'
              }`}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  growthStage >= stage.stage ? 'bg-cyan-600' : 'bg-gray-700'
                }`}>
                  <span className="text-white font-bold text-sm">{idx + 1}</span>
                </div>
                <div>
                  <div className={`font-semibold ${
                    growthStage >= stage.stage ? 'text-white' : 'text-gray-500'
                  }`}>
                    {stage.name}
                  </div>
                  <div className="text-gray-400 text-sm">{stage.size}</div>
                </div>
              </div>
              <Atom className={growthStage >= stage.stage ? 'text-cyan-400' : 'text-gray-600'} size={20} />
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gradient-to-br from-cyan-900/20 to-green-900/20 rounded-lg p-6 border border-cyan-500/20">
        <h3 className="text-cyan-300 font-semibold text-lg mb-3">Listening Mechanisms</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <h4 className="text-cyan-300 font-semibold mb-2">Electromagnetic Sensing</h4>
            <p>The gel responds to electric and magnetic field patterns, using them as growth signals and orientation cues.</p>
          </div>
          <div>
            <h4 className="text-purple-300 font-semibold mb-2">Chemical Gradient Detection</h4>
            <p>Molecular concentration gradients guide the gel's expansion in specific directions through chemotaxis.</p>
          </div>
          <div>
            <h4 className="text-pink-300 font-semibold mb-2">Quantum Field Coupling</h4>
            <p>Zero-point quantum fluctuations provide nucleation sites and influence growth morphology.</p>
          </div>
          <div>
            <h4 className="text-green-300 font-semibold mb-2">Prime Pattern Resonance</h4>
            <p>The gel preferentially grows along prime number frequency patterns, creating fractal structures.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
