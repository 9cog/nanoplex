import React, { useState, useEffect } from 'react';
import { Sparkles, Zap, Activity } from 'lucide-react';

/**
 * Section 9.5: Fractal Condensation - Condensing Everywhere at Once
 * 9.5.1: Pattern of Primes Need Not to Be Instructed
 */
export const FractalCondensationPanel: React.FC = () => {
  const [condensationPhase, setCondensationPhase] = useState(0);
  const [activeNodes, setActiveNodes] = useState<Set<number>>(new Set());

  useEffect(() => {
    const interval = setInterval(() => {
      setCondensationPhase(prev => (prev + 0.05) % 1);
      
      // Simulate simultaneous condensation at prime-pattern locations
      const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
      const newActive = new Set<number>();
      primes.forEach(prime => {
        if (Math.random() > 0.3) {
          newActive.add(prime);
        }
      });
      setActiveNodes(newActive);
    }, 200);
    
    return () => clearInterval(interval);
  }, []);

  const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Sparkles className="text-yellow-400" size={28} />
          <span>9.5 Fractal Condensation</span>
        </h2>
        <p className="text-gray-400 mt-2">
          Condensing Everywhere at Once
        </p>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-yellow-500/20">
        <h3 className="text-yellow-300 font-semibold text-lg mb-3">Simultaneous Phase Transitions</h3>
        <p className="text-gray-300 leading-relaxed">
          Fractal condensation is an ultrafast phenomenon where phase transitions occur simultaneously 
          throughout the entire material volume. Unlike traditional condensation that nucleates at 
          specific points and spreads, fractal condensation happens everywhere at once, driven by 
          the underlying prime pattern structure that permeates the entire gel.
        </p>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">Real-time Condensation Pattern</h3>
        
        <div className="relative bg-gray-900/50 rounded-lg p-8" style={{ height: '400px' }}>
          <svg width="100%" height="100%" viewBox="0 0 400 400">
            <defs>
              <radialGradient id="condensationGlow">
                <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
              </radialGradient>
            </defs>
            
            {/* Background grid */}
            {Array.from({ length: 20 }, (_, i) => (
              <React.Fragment key={`grid-${i}`}>
                <line
                  x1={i * 20}
                  y1="0"
                  x2={i * 20}
                  y2="400"
                  stroke="#374151"
                  strokeWidth="0.5"
                  opacity="0.3"
                />
                <line
                  x1="0"
                  y1={i * 20}
                  x2="400"
                  y2={i * 20}
                  stroke="#374151"
                  strokeWidth="0.5"
                  opacity="0.3"
                />
              </React.Fragment>
            ))}
            
            {/* Condensation nodes at prime positions */}
            {primes.map((prime, idx) => {
              const angle = (idx / primes.length) * 2 * Math.PI;
              const radius = 150;
              const x = 200 + radius * Math.cos(angle);
              const y = 200 + radius * Math.sin(angle);
              const isActive = activeNodes.has(prime);
              const pulseSize = isActive ? 15 + Math.sin(condensationPhase * 2 * Math.PI) * 5 : 10;
              
              return (
                <g key={prime}>
                  {/* Condensation glow */}
                  {isActive && (
                    <circle
                      cx={x}
                      cy={y}
                      r={pulseSize * 2}
                      fill="url(#condensationGlow)"
                      opacity={0.6}
                    />
                  )}
                  
                  {/* Node circle */}
                  <circle
                    cx={x}
                    cy={y}
                    r={pulseSize}
                    fill={isActive ? '#fbbf24' : '#4b5563'}
                    stroke={isActive ? '#f59e0b' : '#6b7280'}
                    strokeWidth="2"
                    className="transition-all duration-200"
                  />
                  
                  {/* Prime number label */}
                  <text
                    x={x}
                    y={y + 5}
                    textAnchor="middle"
                    fill="white"
                    fontSize="12"
                    fontWeight="bold"
                  >
                    {prime}
                  </text>
                  
                  {/* Connection lines to center */}
                  {isActive && (
                    <line
                      x1={x}
                      y1={y}
                      x2="200"
                      y2="200"
                      stroke="#fbbf24"
                      strokeWidth="2"
                      opacity={0.4}
                      className="transition-opacity duration-200"
                    />
                  )}
                </g>
              );
            })}
            
            {/* Center crystallization point */}
            <circle
              cx="200"
              cy="200"
              r="20"
              fill="#8b5cf6"
              stroke="#a78bfa"
              strokeWidth="3"
            />
            <text
              x="200"
              y="205"
              textAnchor="middle"
              fill="white"
              fontSize="14"
              fontWeight="bold"
            >
              Core
            </text>
          </svg>
        </div>
        
        <div className="mt-4 text-xs text-gray-400 text-center">
          Yellow nodes show active condensation sites at prime number locations
        </div>
      </div>

      {/* 9.5.1 Pattern of Primes Need Not to Be Instructed */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Activity size={20} className="text-purple-400" />
          <span>9.5.1 Pattern of Primes Need Not to Be Instructed</span>
        </h3>
        <p className="text-gray-300 mb-4">
          The remarkable feature of fractal condensation is that prime patterns emerge spontaneously 
          without external instruction. The gel material naturally organizes around prime number 
          frequencies because these represent the most stable energy configurations. It's not programmed - 
          it's fundamental to the physics of the system.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-900/50 rounded-lg p-4 border border-yellow-500/20">
            <div className="flex items-center justify-between mb-2">
              <span className="text-yellow-300 font-semibold text-sm">Active Sites</span>
              <Zap className="text-yellow-400" size={16} />
            </div>
            <div className="text-3xl font-bold text-white">{activeNodes.size}</div>
            <div className="text-xs text-gray-400 mt-1">of {primes.length} prime locations</div>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
            <div className="flex items-center justify-between mb-2">
              <span className="text-purple-300 font-semibold text-sm">Synchronicity</span>
              <Sparkles className="text-purple-400" size={16} />
            </div>
            <div className="text-3xl font-bold text-white">98.7%</div>
            <div className="text-xs text-gray-400 mt-1">Simultaneous condensation</div>
          </div>
          
          <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
            <div className="flex items-center justify-between mb-2">
              <span className="text-cyan-300 font-semibold text-sm">Emergence Speed</span>
              <Activity className="text-cyan-400" size={16} />
            </div>
            <div className="text-3xl font-bold text-white">ps</div>
            <div className="text-xs text-gray-400 mt-1">Picosecond timescale</div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-br from-yellow-900/20 to-purple-900/20 rounded-lg p-6 border border-yellow-500/20">
        <h3 className="text-yellow-300 font-semibold text-lg mb-3">Fractal Condensation Properties</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <h4 className="text-yellow-300 font-semibold mb-2">Ultrafast Transition</h4>
            <p>
              Phase transitions occur on picosecond timescales across the entire material 
              volume, bypassing traditional nucleation-and-growth kinetics.
            </p>
          </div>
          <div>
            <h4 className="text-purple-300 font-semibold mb-2">Self-Organizing</h4>
            <p>
              Prime patterns emerge spontaneously from the energy landscape without 
              external programming or instruction.
            </p>
          </div>
          <div>
            <h4 className="text-cyan-300 font-semibold mb-2">Coherent Phases</h4>
            <p>
              All condensation sites maintain quantum coherence, creating a unified 
              field across macroscopic distances.
            </p>
          </div>
          <div>
            <h4 className="text-pink-300 font-semibold mb-2">Energy Optimization</h4>
            <p>
              The system naturally finds minimum energy configurations that correspond 
              to prime number resonances.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
