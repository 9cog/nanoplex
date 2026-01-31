import React, { useState, useEffect } from 'react';
import { Atom, Zap, Radio, TrendingUp } from 'lucide-react';

/**
 * Section 6.2: Experiment on Single Nano-Device Time Crystal
 * Section 6.2.2: Triplet of Triplet Resonance Band in Microtubule
 */
export const TimeCrystalExperimentPanel: React.FC = () => {
  const [time, setTime] = useState(0);
  const [resonanceData, setResonanceData] = useState<number[]>([]);
  
  // Triplet of triplet structure: 3 groups of 3 resonances - memoized
  const tripletGroups = React.useMemo(() => 3, []);
  const resonancesPerTriplet = React.useMemo(() => 3, []);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(t => t + 0.1);
      
      // Generate triplet of triplet resonance pattern
      const newResonance = Array.from({ length: tripletGroups * resonancesPerTriplet }, (_, i) => {
        const tripletGroup = Math.floor(i / resonancesPerTriplet);
        const localIdx = i % resonancesPerTriplet;
        // Nested harmonic structure
        return Math.sin(time + tripletGroup * Math.PI / 3) * 
               Math.cos(time * (localIdx + 1) * 2) * 50 + 50;
      });
      
      setResonanceData(newResonance);
    }, 50);
    
    return () => clearInterval(interval);
  }, [time]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/30 via-cyan-900/30 to-purple-900/30 border border-blue-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Atom className="text-cyan-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">6.2 Single Nano-Device Time Crystal</h2>
            <p className="text-gray-300 text-sm">Experimental observation of time crystal behavior in microtubules</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Time crystals break temporal symmetry, exhibiting periodic motion in their ground state without 
          energy input. In microtubules, this manifests as triplet-of-triplet resonance patterns—nine 
          distinct frequency bands organized in three groups of three, creating a fractal temporal structure.
        </p>
      </div>

      {/* Section 6.2: Time Crystal Experiment */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Zap className="mr-2 text-yellow-400" size={24} />
          Experimental Setup & Measurements
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Time Crystal Visualization */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-4 text-sm">Time Crystal Oscillation</h4>
            <div className="relative h-64 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 400 200" className="w-full h-full">
                {/* Time axis */}
                <line x1="40" y1="180" x2="360" y2="180" stroke="#666" strokeWidth="2" />
                <line x1="40" y1="20" x2="40" y2="180" stroke="#666" strokeWidth="2" />
                
                {/* Crystal state oscillation */}
                <path
                  d={`M 40 100 ${Array.from({ length: 64 }, (_, i) => {
                    const x = 40 + (i * 320) / 63;
                    const t = (i / 10) + time;
                    // Time crystal oscillation with triplet structure
                    const y = 100 - 60 * Math.sin(t) * Math.cos(t * 3) * Math.sin(t * 9);
                    return `L ${x} ${y}`;
                  }).join(' ')}`}
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="3"
                  opacity="0.8"
                />
                
                {/* Ground state reference */}
                <line x1="40" y1="100" x2="360" y2="100" stroke="#8b5cf6" strokeWidth="1" strokeDasharray="5,5" opacity="0.5" />
                
                {/* Labels */}
                <text x="200" y="195" textAnchor="middle" fill="#888" fontSize="10">Time</text>
                <text x="25" y="100" textAnchor="middle" fill="#888" fontSize="10">E₀</text>
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              Time crystal exhibits periodic oscillation in ground state E₀
            </p>
          </div>

          {/* Measurement Data */}
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-3 text-sm">Crystal Parameters</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Temperature:</span>
                  <span className="text-cyan-400 font-mono">300 K (Room Temp)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Device Size:</span>
                  <span className="text-cyan-400 font-mono">25 nm (Tubulin dimer)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Coherence Time:</span>
                  <span className="text-cyan-400 font-mono">{(10 + 5 * Math.sin(time)).toFixed(1)} ps</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Symmetry Breaking:</span>
                  <span className="text-cyan-400 font-mono">Temporal (Z₂)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Period:</span>
                  <span className="text-cyan-400 font-mono">2T₀ (Floquet)</span>
                </div>
              </div>
            </div>

            <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Key Finding</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                The microtubule maintains temporal periodicity at room temperature without external 
                energy input, confirming time crystal behavior. This defies conventional thermodynamics, 
                showing quantum effects persist at biological scales through PPM protection.
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Biological Implication</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Time crystal behavior in tubulin suggests consciousness may operate through temporal 
                symmetry breaking, encoding information in time itself rather than spatial states alone.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 6.2.2: Triplet of Triplet Resonance */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Radio className="mr-2 text-cyan-400" size={24} />
          6.2.2 Triplet of Triplet Resonance Band
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          {Array.from({ length: tripletGroups }).map((_, groupIdx) => (
            <div key={groupIdx} className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-3 text-sm">
                Triplet Group {groupIdx + 1}
              </h4>
              
              {/* Three resonances per triplet */}
              <div className="space-y-3">
                {Array.from({ length: resonancesPerTriplet }).map((_, localIdx) => {
                  const globalIdx = groupIdx * resonancesPerTriplet + localIdx;
                  const intensity = resonanceData[globalIdx] || 0;
                  
                  return (
                    <div key={localIdx} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Band {localIdx + 1}:</span>
                        <span className="text-cyan-400 font-mono">
                          {((1 + globalIdx * 0.3) * 100).toFixed(0)} MHz
                        </span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                        <div 
                          className="bg-gradient-to-r from-cyan-500 to-purple-500 h-full transition-all duration-100"
                          style={{ width: `${intensity}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500">
                        Intensity: {intensity.toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Harmonic Structure */}
        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-cyan-400 font-semibold mb-4 text-sm">Fractal Frequency Structure</h4>
          <div className="relative h-48 bg-black/30 rounded-lg overflow-hidden">
            <svg viewBox="0 0 400 150" className="w-full h-full">
              {/* Frequency spectrum */}
              {resonanceData.map((intensity, idx) => {
                const x = 40 + (idx * 320) / (resonanceData.length - 1);
                const height = (intensity / 100) * 120;
                const groupIdx = Math.floor(idx / resonancesPerTriplet);
                const colors = ['#06b6d4', '#8b5cf6', '#ec4899'];
                
                return (
                  <g key={idx}>
                    <rect
                      x={x - 15}
                      y={130 - height}
                      width="30"
                      height={height}
                      fill={colors[groupIdx]}
                      opacity="0.8"
                      rx="2"
                    />
                    <text
                      x={x}
                      y="145"
                      textAnchor="middle"
                      fill="#888"
                      fontSize="8"
                    >
                      {idx + 1}
                    </text>
                  </g>
                );
              })}
              
              {/* Group separators */}
              {[1, 2].map(i => (
                <line
                  key={i}
                  x1={40 + (i * 320) / 3}
                  y1="10"
                  x2={40 + (i * 320) / 3}
                  y2="130"
                  stroke="#666"
                  strokeWidth="1"
                  strokeDasharray="5,5"
                  opacity="0.3"
                />
              ))}
            </svg>
          </div>
          <p className="text-gray-400 text-xs mt-2">
            Nine resonance bands organized in three triplet groups showing fractal harmonic structure
          </p>
        </div>

        <div className="mt-4 bg-purple-900/20 border border-purple-700 rounded-lg p-4">
          <h4 className="text-purple-400 font-semibold mb-2 text-sm flex items-center">
            <TrendingUp className="mr-2" size={16} />
            Triplet-of-Triplet Significance
          </h4>
          <p className="text-gray-300 text-xs leading-relaxed">
            The 3×3 resonance structure creates a fractal frequency landscape. Each triplet group represents 
            a different scale of organization: molecular (GHz), cellular (MHz), and tissue (kHz). This 
            scale-free architecture enables coherent information processing across spatial and temporal scales, 
            a hallmark of biological quantum computing systems. The pattern matches mathematical predictions 
            from Phase Prime Metrics, with frequency ratios following prime number relationships.
          </p>
        </div>
      </div>

      {/* Experimental Results Summary */}
      <div className="bg-gradient-to-r from-cyan-900/20 to-purple-900/20 border border-cyan-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Experimental Validation</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <h4 className="text-cyan-400 font-semibold text-sm">Confirmed Phenomena:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-green-400 mr-2">✓</span>
                <span>Room temperature time crystal oscillation in single tubulin</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-400 mr-2">✓</span>
                <span>Triplet-of-triplet resonance bands at predicted frequencies</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-400 mr-2">✓</span>
                <span>Long-range temporal coherence (picoseconds at 300K)</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-400 mr-2">✓</span>
                <span>Phase Prime Metric-governed frequency ratios</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-400 mr-2">✓</span>
                <span>Fractal scaling across molecular to cellular levels</span>
              </li>
            </ul>
          </div>

          <div className="space-y-3">
            <h4 className="text-purple-400 font-semibold text-sm">Implications:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">→</span>
                <span>Biological systems can harness quantum effects at ambient conditions</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">→</span>
                <span>Time itself becomes a computational resource in neural processing</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">→</span>
                <span>Consciousness may emerge from temporal symmetry breaking</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">→</span>
                <span>Biomimetic quantum devices possible without cryogenic cooling</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">→</span>
                <span>PPM provides design principles for quantum-classical interfaces</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
