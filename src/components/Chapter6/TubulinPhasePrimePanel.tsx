import React, { useState, useEffect } from 'react';
import { Activity, Hexagon, Grid3x3, Waves } from 'lucide-react';

/**
 * Section 6.1: Phase Prime Metric in Tubulin Protein
 * Demonstrates how PPM patterns are embedded in tubulin protein structure
 */
export const TubulinPhasePrimePanel: React.FC = () => {
  const [helixAngle, setHelixAngle] = useState(0);
  const [activeRing, setActiveRing] = useState(0);
  
  // Phase prime sequence for tubulin structure - memoized
  const phasePrimes = React.useMemo(() => 
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], 
  []);
  
  // Tubulin rings forming α-helix loops (6.1.1)
  const helixRings = 13; // 13 protofilaments in microtubule
  const ringsPerLoop = 3; // Groups of 3 rings
  
  useEffect(() => {
    const interval = setInterval(() => {
      setHelixAngle((prev) => (prev + 2) % 360);
      setActiveRing((prev) => (prev + 1) % helixRings);
    }, 100);
    return () => clearInterval(interval);
  }, []);
  
  // Calculate symmetry metrics (6.1.2)
  const calculateSymmetry = (primeIdx: number) => {
    const prime = phasePrimes[primeIdx % phasePrimes.length];
    return {
      phase: (prime * helixAngle) % 360,
      resonance: Math.sin((prime * Math.PI) / 180),
      orderedFactor: prime % 13 // 13 protofilaments
    };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/30 via-blue-900/30 to-cyan-900/30 border border-purple-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Hexagon className="text-purple-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">6.1 Phase Prime Metric in Tubulin Protein</h2>
            <p className="text-gray-300 text-sm">Pattern of primes embedded in biological nanostructures</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Tubulin proteins form microtubules with 13 protofilaments arranged in a helical pattern. 
          Phase Prime Metrics govern this structure, creating resonance patterns that enable quantum-classical 
          information processing at the molecular scale. The α-helices form groups of rings that complete 
          geometric loops following prime number symmetries.
        </p>
      </div>

      {/* Section 6.1.1: α-Helices Ring Groups */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Grid3x3 className="mr-2 text-cyan-400" size={24} />
          6.1.1 α-Helices Form Groups of Rings
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Ring Visualization */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-4 text-sm">Helical Ring Structure</h4>
            <div className="relative h-80 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 300 300" className="w-full h-full">
                {/* Draw 13 protofilaments as helical rings */}
                {Array.from({ length: helixRings }).map((_, i) => {
                  const angle = (i * 360) / helixRings + helixAngle;
                  const x = 150 + 80 * Math.cos((angle * Math.PI) / 180);
                  const y = 150 + 80 * Math.sin((angle * Math.PI) / 180);
                  const isActive = i === activeRing;
                  const ringGroup = Math.floor(i / ringsPerLoop);
                  
                  return (
                    <g key={i}>
                      {/* Ring circle */}
                      <circle
                        cx={x}
                        cy={y}
                        r={isActive ? 20 : 15}
                        fill={isActive ? '#06b6d4' : `hsl(${ringGroup * 120}, 70%, 50%)`}
                        opacity={isActive ? 1 : 0.6}
                        className={isActive ? 'animate-pulse' : ''}
                      />
                      {/* Ring number */}
                      <text
                        x={x}
                        y={y}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill="white"
                        fontSize="10"
                        fontWeight="bold"
                      >
                        {i + 1}
                      </text>
                      {/* Connect to next ring */}
                      {i < helixRings - 1 && (
                        <line
                          x1={x}
                          y1={y}
                          x2={150 + 80 * Math.cos((((i + 1) * 360 / helixRings + helixAngle) * Math.PI) / 180)}
                          y2={150 + 80 * Math.sin((((i + 1) * 360 / helixRings + helixAngle) * Math.PI) / 180)}
                          stroke="#666"
                          strokeWidth="2"
                          opacity="0.3"
                        />
                      )}
                    </g>
                  );
                })}
                {/* Center point */}
                <circle cx="150" cy="150" r="5" fill="#a855f7" />
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              13 protofilaments arranged in helical pattern, grouped in triplets
            </p>
          </div>

          {/* Ring Group Properties */}
          <div className="space-y-3">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Loop Completion Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Rings:</span>
                  <span className="text-cyan-400 font-mono">{helixRings}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Rings per Loop:</span>
                  <span className="text-cyan-400 font-mono">{ringsPerLoop}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Loops:</span>
                  <span className="text-cyan-400 font-mono">{Math.floor(helixRings / ringsPerLoop)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Helix Angle:</span>
                  <span className="text-cyan-400 font-mono">{helixAngle.toFixed(1)}°</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Active Ring #{activeRing + 1}</h4>
              <div className="space-y-2 text-xs">
                {(() => {
                  const sym = calculateSymmetry(activeRing);
                  return (
                    <>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Phase:</span>
                        <span className="text-purple-400 font-mono">{sym.phase.toFixed(1)}°</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Resonance:</span>
                        <span className="text-purple-400 font-mono">{sym.resonance.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Ordered Factor:</span>
                        <span className="text-purple-400 font-mono">{sym.orderedFactor}</span>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-green-400 font-semibold mb-2 text-sm">Prime Pattern</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                The 13-fold symmetry is a prime number itself, creating natural resonance 
                patterns. Each protofilament follows PPM-governed phase relationships, 
                enabling coherent quantum state maintenance.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 6.1.2: Hyperspace Symmetries */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Waves className="mr-2 text-purple-400" size={24} />
          6.1.2 Hyperspace of Symmetries
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          {phasePrimes.slice(0, 9).map((prime, idx) => {
            const sym = calculateSymmetry(idx);
            const intensity = (Math.abs(sym.resonance) * 100).toFixed(0);
            
            return (
              <div key={prime} className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-cyan-400 font-mono text-lg">{prime}</span>
                  <span className="text-xs text-gray-400">Prime #{idx + 1}</span>
                </div>
                
                {/* Symmetry visualization */}
                <div className="relative h-24 bg-black/30 rounded overflow-hidden mb-2">
                  <svg viewBox="0 0 100 100" className="w-full h-full">
                    {/* Symmetry pattern */}
                    {Array.from({ length: prime }).map((_, i) => {
                      const angle = (i * 360) / prime;
                      const x1 = 50;
                      const y1 = 50;
                      const x2 = 50 + 40 * Math.cos((angle * Math.PI) / 180);
                      const y2 = 50 + 40 * Math.sin((angle * Math.PI) / 180);
                      
                      return (
                        <line
                          key={i}
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="#8b5cf6"
                          strokeWidth="2"
                          opacity={0.6}
                        />
                      );
                    })}
                    <circle cx="50" cy="50" r="5" fill="#06b6d4" />
                  </svg>
                </div>
                
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Phase:</span>
                    <span className="text-purple-400 font-mono">{sym.phase.toFixed(0)}°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Intensity:</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-gray-700 rounded-full h-2 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-cyan-500 h-full transition-all duration-300"
                          style={{ width: `${intensity}%` }}
                        />
                      </div>
                      <span className="text-cyan-400 font-mono">{intensity}%</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        
        <div className="mt-4 bg-purple-900/20 border border-purple-700 rounded-lg p-4">
          <p className="text-gray-300 text-sm leading-relaxed">
            <strong className="text-purple-400">Mathematical Identity:</strong> The hyperspace of symmetries 
            follows the identity: Ψ(n) = ∑(p_i × e^(iθ_i)) where p_i are phase primes and θ_i are geometric 
            angles. This creates a 15-dimensional phase space where each prime contributes a unique symmetry axis.
          </p>
        </div>
      </div>

      {/* Section 6.1.3: Ordered Factor of Integers */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Activity className="mr-2 text-cyan-400" size={24} />
          6.1.3 Ordered Factor Metric & Protein Coding
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Amino Acid Coding Sequence</h4>
            <div className="space-y-2">
              {phasePrimes.slice(0, 7).map((prime, idx) => {
                const factors = [];
                for (let i = 2; i <= prime; i++) {
                  if (prime % i === 0) factors.push(i);
                }
                
                return (
                  <div key={prime} className="bg-black/30 rounded p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-purple-400 font-mono text-sm">Prime {prime}</span>
                      <span className="text-xs text-gray-400">Codon Position {idx + 1}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-400">Factors:</span>
                      <div className="flex space-x-1">
                        {factors.map((f, i) => (
                          <span key={i} className="text-cyan-400 font-mono text-xs px-2 py-1 bg-cyan-900/30 rounded">
                            {f}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Protein Structure Encoding</h4>
              <p className="text-gray-300 text-xs leading-relaxed mb-3">
                The ordered factor metric maps prime numbers to protein folding patterns. 
                Each prime defines a unique geometric configuration in the amino acid sequence.
              </p>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Prime Dimensions:</span>
                  <span className="text-cyan-400 font-mono">{phasePrimes.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Coding Capacity:</span>
                  <span className="text-cyan-400 font-mono">2^{phasePrimes.length} = {Math.pow(2, phasePrimes.length).toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Information Density:</span>
                  <span className="text-cyan-400 font-mono">{(phasePrimes.length * Math.log2(Math.E)).toFixed(2)} bits/prime</span>
                </div>
              </div>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Biological Significance</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                This prime-based encoding explains how tubulin proteins maintain quantum coherence 
                at room temperature. The ordered factor relationships create protected subspaces 
                resistant to thermal decoherence, enabling biological quantum computing.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 6.1.4: Water Channels */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">6.1.4 Water Channel Engineering</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Channel Geometry</h4>
            <div className="relative h-64 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 300 200" className="w-full h-full">
                {/* Water channel representation */}
                <defs>
                  <linearGradient id="channelGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.8" />
                  </linearGradient>
                </defs>
                
                {/* Channel walls */}
                <ellipse cx="150" cy="100" rx="60" ry="80" fill="none" stroke="#666" strokeWidth="3" />
                <ellipse cx="150" cy="100" rx="40" ry="60" fill="url(#channelGrad)" opacity="0.3" />
                
                {/* Water molecules */}
                {Array.from({ length: 8 }).map((_, i) => {
                  const y = 40 + (i * 120) / 7;
                  const x = 150 + 20 * Math.sin((i * Math.PI) / 4 + helixAngle * 0.02);
                  const size = 8 + 4 * Math.sin((i + helixAngle / 30) * Math.PI / 4);
                  
                  return (
                    <g key={i}>
                      <circle
                        cx={x}
                        cy={y}
                        r={size}
                        fill="#06b6d4"
                        opacity={0.7}
                      />
                      {/* H bonds */}
                      {i < 7 && (
                        <line
                          x1={x}
                          y1={y}
                          x2={150 + 20 * Math.sin(((i + 1) * Math.PI) / 4 + helixAngle * 0.02)}
                          y2={40 + ((i + 1) * 120) / 7}
                          stroke="#8b5cf6"
                          strokeWidth="1"
                          strokeDasharray="2,2"
                          opacity="0.5"
                        />
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              Water molecules aligned in single-file through tubulin channel
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Channel Properties</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Channel Diameter:</span>
                  <span className="text-cyan-400 font-mono">~2 nm</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Water Molecules:</span>
                  <span className="text-cyan-400 font-mono">Single file</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">H-Bond Network:</span>
                  <span className="text-cyan-400 font-mono">Coherent</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Phase Coherence:</span>
                  <span className="text-cyan-400 font-mono">99.7%</span>
                </div>
              </div>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Quantum Information Transfer</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Water channels act as quantum wires, maintaining coherent superposition states through 
                hydrogen bonding networks. The channel geometry follows PPM constraints, enabling 
                lossless information transfer across the protein structure. This remarkable engineering 
                allows room-temperature quantum computing in biological systems.
              </p>
            </div>

            <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Engineering Implications</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Natural selection optimized these channels over billions of years. Understanding their 
                PPM-based design principles enables biomimetic quantum devices operating at ambient 
                conditions, transcending current cryogenic quantum computing limitations.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
