import React, { useState } from 'react';
import { Atom, Beaker, Zap } from 'lucide-react';

/**
 * Section 9.6: Fractal Reaction Kinetics - Many Syntheses in One Beaker at a Time
 */
export const FractalReactionKineticsPanel: React.FC = () => {
  const [activeReactions, setActiveReactions] = useState<Set<number>>(new Set([0, 2, 4]));

  const reactions = [
    { id: 0, name: 'Hinductor Formation', rate: 95, product: 'H-Circuit Elements', color: 'cyan' },
    { id: 1, name: 'Time Crystal Growth', rate: 87, product: '11D Crystal Structures', color: 'purple' },
    { id: 2, name: 'Megamer Polymerization', rate: 92, product: 'Resonance Chains', color: 'pink' },
    { id: 3, name: 'Nanowire Assembly', rate: 88, product: 'Spiral Structures', color: 'blue' },
    { id: 4, name: 'Gel Matrix Formation', rate: 96, product: 'Brain Jelly Base', color: 'green' },
    { id: 5, name: 'Sensor Integration', rate: 84, product: 'Responsive Nodes', color: 'yellow' },
    { id: 6, name: 'Quantum Coupling', rate: 79, product: 'Coherent States', color: 'orange' }
  ];

  const toggleReaction = (id: number) => {
    setActiveReactions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) newSet.delete(id);
      else newSet.add(id);
      return newSet;
    });
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Beaker className="text-green-400" size={28} />
          <span>9.6 Fractal Reaction Kinetics</span>
        </h2>
        <p className="text-gray-400 mt-2">
          Many Syntheses in One Beaker at a Time
        </p>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-green-500/20">
        <h3 className="text-green-300 font-semibold text-lg mb-3">Parallel Chemical Synthesis</h3>
        <p className="text-gray-300 leading-relaxed">
          Fractal reaction kinetics enables multiple chemical syntheses to occur simultaneously 
          in a single reaction vessel without interference. Each reaction follows its own prime 
          pattern frequency channel, allowing independent control and monitoring. This represents 
          a revolutionary approach to materials synthesis where complexity emerges from parallel 
          processing at the molecular level.
        </p>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">Reaction Beaker Simulation</h3>
        
        <div className="relative bg-gray-900/50 rounded-lg p-6 mb-4" style={{ height: '300px' }}>
          <svg width="100%" height="100%" viewBox="0 0 400 300">
            {/* Beaker outline */}
            <path
              d="M 100 50 L 100 220 Q 100 260 140 260 L 260 260 Q 300 260 300 220 L 300 50 Z"
              fill="rgba(17, 24, 39, 0.8)"
              stroke="#4b5563"
              strokeWidth="3"
            />
            
            {/* Liquid level */}
            <path
              d="M 100 100 L 100 220 Q 100 260 140 260 L 260 260 Q 300 260 300 220 L 300 100 Z"
              fill="rgba(16, 185, 129, 0.2)"
              stroke="#10b981"
              strokeWidth="2"
            />
            
            {/* Reaction bubbles/particles */}
            {reactions.map((reaction, idx) => {
              if (!activeReactions.has(reaction.id)) return null;
              
              const particles = Array.from({ length: 8 }, (_, i) => {
                const angle = (i / 8) * 2 * Math.PI + idx * 0.5;
                const radius = 60 + idx * 15;
                const x = 200 + radius * Math.cos(angle);
                const y = 180 + radius * Math.sin(angle) * 0.7;
                return { x, y };
              });
              
              return (
                <g key={reaction.id}>
                  {particles.map((particle, i) => (
                    <circle
                      key={`${reaction.id}-${i}`}
                      cx={particle.x}
                      cy={particle.y}
                      r="4"
                      fill={`hsl(${idx * 50}, 70%, 60%)`}
                      opacity={0.7}
                    >
                      <animate
                        attributeName="r"
                        values="3;5;3"
                        dur="2s"
                        repeatCount="indefinite"
                        begin={`${i * 0.2}s`}
                      />
                    </circle>
                  ))}
                </g>
              );
            })}
            
            {/* Beaker label */}
            <text x="200" y="30" textAnchor="middle" fill="#9ca3af" fontSize="14" fontWeight="bold">
              Multi-Synthesis Beaker
            </text>
          </svg>
        </div>
        
        <div className="text-xs text-gray-400 text-center mb-4">
          Active reactions shown as colored particle systems within the beaker
        </div>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">Reaction Control Panel</h3>
        
        <div className="space-y-3">
          {reactions.map((reaction) => {
            const isActive = activeReactions.has(reaction.id);
            
            return (
              <div
                key={reaction.id}
                className={`flex items-center justify-between p-4 rounded-lg border transition-all cursor-pointer ${
                  isActive
                    ? 'bg-gray-900/50 border-green-500/50'
                    : 'bg-gray-900/30 border-gray-700'
                }`}
                onClick={() => toggleReaction(reaction.id)}
              >
                <div className="flex items-center space-x-3 flex-1">
                  <div className={`w-4 h-4 rounded-full ${
                    isActive ? 'bg-green-400 animate-pulse' : 'bg-gray-600'
                  }`} />
                  <div className="flex-1">
                    <div className={`font-semibold ${isActive ? 'text-white' : 'text-gray-500'}`}>
                      {reaction.name}
                    </div>
                    <div className="text-xs text-gray-400">{reaction.product}</div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-sm text-gray-400">Yield</div>
                    <div className={`text-lg font-bold ${isActive ? 'text-green-400' : 'text-gray-600'}`}>
                      {reaction.rate}%
                    </div>
                  </div>
                  <Atom className={isActive ? 'text-green-400' : 'text-gray-600'} size={20} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 border border-green-500/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-green-300 font-semibold text-sm">Active Reactions</span>
            <Beaker className="text-green-400" size={16} />
          </div>
          <div className="text-3xl font-bold text-white">{activeReactions.size}</div>
          <div className="text-xs text-gray-400 mt-1">of {reactions.length} running</div>
        </div>
        
        <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-cyan-300 font-semibold text-sm">Avg Yield</span>
            <Zap className="text-cyan-400" size={16} />
          </div>
          <div className="text-3xl font-bold text-white">
            {(reactions
              .filter(r => activeReactions.has(r.id))
              .reduce((sum, r) => sum + r.rate, 0) / (activeReactions.size || 1)
            ).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400 mt-1">Combined efficiency</div>
        </div>
        
        <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-purple-300 font-semibold text-sm">Interference</span>
            <Atom className="text-purple-400" size={16} />
          </div>
          <div className="text-3xl font-bold text-white">0%</div>
          <div className="text-xs text-gray-400 mt-1">Prime frequency isolation</div>
        </div>
      </div>

      <div className="bg-gradient-to-br from-green-900/20 to-cyan-900/20 rounded-lg p-6 border border-green-500/20">
        <h3 className="text-green-300 font-semibold text-lg mb-3">Fractal Kinetics Principles</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <h4 className="text-green-300 font-semibold mb-2">Frequency Channel Separation</h4>
            <p>
              Each reaction operates on a unique prime frequency, preventing cross-talk 
              and enabling true parallel processing in a single vessel.
            </p>
          </div>
          <div>
            <h4 className="text-cyan-300 font-semibold mb-2">Independent Control</h4>
            <p>
              Reactions can be individually started, stopped, or modulated by tuning 
              their corresponding prime frequency channels.
            </p>
          </div>
          <div>
            <h4 className="text-purple-300 font-semibold mb-2">Self-Organization</h4>
            <p>
              Products spontaneously organize into hierarchical structures following 
              fractal patterns encoded in the prime metrics.
            </p>
          </div>
          <div>
            <h4 className="text-yellow-300 font-semibold mb-2">Cascade Synthesis</h4>
            <p>
              Products from one reaction can serve as reactants for others, creating 
              complex synthesis cascades in a single operation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
