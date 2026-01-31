import React from 'react';
import { Zap, TrendingUp, Database } from 'lucide-react';

/**
 * Section 9.9: Entropy Drives the Synthesis of a Pattern of Primes
 * 9.9.1: Hesse Diagram for Entropy Builds Material Analog of Integers
 */
export const EntropyDrivenSynthesisPanel: React.FC = () => {
  const integers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Zap className="text-yellow-400" size={28} />
          <span>9.9 Entropy-Driven Synthesis</span>
        </h2>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-yellow-500/20">
        <h3 className="text-yellow-300 font-semibold text-lg mb-3">Entropy as Organizing Force</h3>
        <p className="text-gray-300 leading-relaxed">
          Contrary to intuition, entropy drives the spontaneous organization of prime patterns 
          in the brain jelly. The system naturally evolves toward maximum entropy states, which 
          correspond to prime number distributions - the most informationally dense configurations.
        </p>
      </div>

      {/* 9.9.1 Hesse Diagram */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Database size={20} className="text-purple-400" />
          <span>9.9.1 Hesse Diagram for Entropy</span>
        </h3>
        <p className="text-gray-300 mb-4">
          The Hesse diagram provides a lattice structure showing how entropy builds a material 
          analog of integers. Each node represents an entropy state corresponding to an integer, 
          with prime positions showing maximum stability.
        </p>

        <div className="relative bg-gray-900/50 rounded-lg p-6" style={{ height: '350px' }}>
          <svg width="100%" height="100%" viewBox="0 0 500 350">
            {/* Draw lattice structure */}
            {integers.map((num, idx) => {
              const row = Math.floor(idx / 4);
              const col = idx % 4;
              const x = 80 + col * 110;
              const y = 60 + row * 90;
              const isPrime = [2, 3, 5, 7, 11].includes(num);
              
              // Draw connections
              if (num > 1) {
                const factors = [];
                for (let i = 2; i < num; i++) {
                  if (num % i === 0) factors.push(i);
                }
                factors.forEach(factor => {
                  const factorIdx = integers.indexOf(factor);
                  if (factorIdx >= 0) {
                    const fRow = Math.floor(factorIdx / 4);
                    const fCol = factorIdx % 4;
                    const fx = 80 + fCol * 110;
                    const fy = 60 + fRow * 90;
                    
                    return (
                      <line
                        key={`${num}-${factor}`}
                        x1={x}
                        y1={y}
                        x2={fx}
                        y2={fy}
                        stroke="#4b5563"
                        strokeWidth="1"
                        opacity="0.3"
                      />
                    );
                  }
                });
              }
              
              return (
                <g key={num}>
                  {/* Node circle */}
                  <circle
                    cx={x}
                    cy={y}
                    r={isPrime ? 25 : 20}
                    fill={isPrime ? 'rgba(251, 191, 36, 0.3)' : 'rgba(75, 85, 99, 0.3)'}
                    stroke={isPrime ? '#fbbf24' : '#6b7280'}
                    strokeWidth={isPrime ? '3' : '2'}
                  />
                  
                  {/* Number label */}
                  <text
                    x={x}
                    y={y + 5}
                    textAnchor="middle"
                    fill={isPrime ? '#fbbf24' : '#9ca3af'}
                    fontSize="16"
                    fontWeight="bold"
                  >
                    {num}
                  </text>
                  
                  {/* Prime indicator */}
                  {isPrime && (
                    <circle
                      cx={x}
                      cy={y}
                      r="28"
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth="1"
                      opacity="0.5"
                    >
                      <animate
                        attributeName="r"
                        values="28;32;28"
                        dur="2s"
                        repeatCount="indefinite"
                      />
                    </circle>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
        
        <div className="text-xs text-gray-400 text-center mt-4">
          Hesse diagram showing entropy-based integer lattice. Primes (highlighted) represent maximum stability points.
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 border border-yellow-500/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-yellow-300 font-semibold text-sm">Entropy Production</span>
            <TrendingUp className="text-yellow-400" size={16} />
          </div>
          <div className="text-3xl font-bold text-white">k_B ln(Ω)</div>
          <div className="text-xs text-gray-400 mt-1">Boltzmann entropy</div>
        </div>
        
        <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-purple-300 font-semibold text-sm">Prime Stability</span>
            <Zap className="text-purple-400" size={16} />
          </div>
          <div className="text-3xl font-bold text-white">Max</div>
          <div className="text-xs text-gray-400 mt-1">Energy minima at primes</div>
        </div>
        
        <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-cyan-300 font-semibold text-sm">Information Density</span>
            <Database className="text-cyan-400" size={16} />
          </div>
          <div className="text-3xl font-bold text-white">99.9%</div>
          <div className="text-xs text-gray-400 mt-1">Prime pattern coverage</div>
        </div>
      </div>

      <div className="bg-gradient-to-br from-yellow-900/20 to-purple-900/20 rounded-lg p-6 border border-yellow-500/20">
        <h3 className="text-yellow-300 font-semibold text-lg mb-3">Entropy-Driven Organization</h3>
        <div className="space-y-2 text-sm text-gray-300">
          <p>
            <span className="text-yellow-300 font-semibold">Maximum Entropy Principle:</span> The 
            system spontaneously evolves to maximize entropy, which corresponds to prime pattern formation.
          </p>
          <p>
            <span className="text-purple-300 font-semibold">Material Integer Analog:</span> Physical 
            structures in the gel create a material representation of integer relationships.
          </p>
          <p>
            <span className="text-cyan-300 font-semibold">Self-Organization:</span> No external 
            instruction needed - entropy gradient naturally guides synthesis toward prime patterns.
          </p>
        </div>
      </div>
    </div>
  );
};
