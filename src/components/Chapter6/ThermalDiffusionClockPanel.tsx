import React, { useState, useEffect } from 'react';
import { Thermometer, Clock, TrendingUp, Activity } from 'lucide-react';

/**
 * Section 6.3: Single Molecule's Thermal Diffusion Emerges into Natural Clock
 * Demonstrates how thermal motion creates temporal markers
 */
export const ThermalDiffusionClockPanel: React.FC = () => {
  const [molecules, setMolecules] = useState<Array<{x: number, y: number, vx: number, vy: number}>>([]);
  const [clockSignal, setClockSignal] = useState<number[]>([]);
  
  useEffect(() => {
    // Initialize molecules with random positions and velocities
    const initMolecules = Array.from({ length: 50 }, () => ({
      x: Math.random() * 300,
      y: Math.random() * 200,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2
    }));
    setMolecules(initMolecules);
  }, []);
  
  useEffect(() => {
    const interval = setInterval(() => {
      // Update molecule positions (Brownian motion)
      setMolecules(prev => {
        const updated = prev.map(m => {
          let newX = m.x + m.vx;
          let newY = m.y + m.vy;
          let newVx = m.vx + (Math.random() - 0.5) * 0.5;
          let newVy = m.vy + (Math.random() - 0.5) * 0.5;
          
          // Bounce off walls
          if (newX < 0 || newX > 300) { newVx = -newVx; newX = Math.max(0, Math.min(300, newX)); }
          if (newY < 0 || newY > 200) { newVy = -newVy; newY = Math.max(0, Math.min(200, newY)); }
          
          // Damping
          newVx *= 0.99;
          newVy *= 0.99;
          
          return { x: newX, y: newY, vx: newVx, vy: newVy };
        });
        
        // Calculate collective motion metric (clock signal)
        setClockSignal(prevSignal => {
          const avgVelocity = updated.reduce((sum, m) => 
            sum + Math.sqrt(m.vx * m.vx + m.vy * m.vy), 0) / Math.max(updated.length, 1);
          const newSignal = [...prevSignal, avgVelocity * 10];
          return newSignal.slice(-60); // Keep last 60 samples
        });
        
        return updated;
      });
    }, 50);
    
    return () => clearInterval(interval);
  }, []);

  // Calculate clock metrics
  const avgClockSignal = clockSignal.length > 0 
    ? clockSignal.reduce((a, b) => a + b, 0) / clockSignal.length 
    : 0;
  
  const clockVariance = clockSignal.length > 0
    ? Math.sqrt(clockSignal.reduce((sum, val) => sum + Math.pow(val - avgClockSignal, 2), 0) / clockSignal.length)
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-900/30 via-red-900/30 to-purple-900/30 border border-orange-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Thermometer className="text-orange-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">6.3 Thermal Diffusion Natural Clock</h2>
            <p className="text-gray-300 text-sm">How random molecular motion creates precise temporal markers</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          At the molecular scale, thermal diffusion appears random. However, when observed collectively 
          through Phase Prime Metrics, thermal motion creates coherent clock signals. Each molecule's 
          Brownian motion contributes to an emergent temporal pattern, transforming noise into information—a 
          fundamental principle of biological timekeeping and consciousness.
        </p>
      </div>

      {/* Brownian Motion Visualization */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Activity className="mr-2 text-orange-400" size={24} />
          Molecular Thermal Motion
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Motion Visualization */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-orange-400 font-semibold mb-4 text-sm">Brownian Motion Field</h4>
            <div className="relative h-96 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 300 200" className="w-full h-full">
                {/* Container box */}
                <rect x="0" y="0" width="300" height="200" fill="none" stroke="#444" strokeWidth="2" />
                
                {/* Molecules */}
                {molecules.map((m, i) => {
                  const speed = Math.sqrt(m.vx * m.vx + m.vy * m.vy);
                  const hue = (speed * 60) % 360;
                  
                  return (
                    <g key={i}>
                      {/* Velocity trail */}
                      <line
                        x1={m.x}
                        y1={m.y}
                        x2={m.x - m.vx * 5}
                        y2={m.y - m.vy * 5}
                        stroke={`hsl(${hue}, 70%, 50%)`}
                        strokeWidth="1"
                        opacity="0.3"
                      />
                      {/* Molecule */}
                      <circle
                        cx={m.x}
                        cy={m.y}
                        r={3 + speed * 0.5}
                        fill={`hsl(${hue}, 70%, 60%)`}
                        opacity="0.8"
                      />
                    </g>
                  );
                })}
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              {molecules.length} molecules undergoing thermal diffusion at 300K
            </p>
          </div>

          {/* Clock Signal Emergence */}
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-4 text-sm">Emergent Clock Signal</h4>
              <div className="relative h-48 bg-black/30 rounded-lg overflow-hidden">
                <svg viewBox="0 0 300 100" className="w-full h-full">
                  {/* Baseline */}
                  <line x1="0" y1="50" x2="300" y2="50" stroke="#444" strokeWidth="1" />
                  
                  {/* Clock signal */}
                  {clockSignal.length > 1 && (
                    <path
                      d={`M 0 ${50 - (clockSignal[0] - avgClockSignal) * 2} ${clockSignal.slice(1).map((val, i) => {
                        const x = ((i + 1) * 300) / (clockSignal.length - 1);
                        const y = 50 - (val - avgClockSignal) * 2;
                        return `L ${x} ${y}`;
                      }).join(' ')}`}
                      fill="none"
                      stroke="#f97316"
                      strokeWidth="2"
                      opacity="0.9"
                    />
                  )}
                  
                  {/* Average line */}
                  <line x1="0" y1="50" x2="300" y2="50" stroke="#06b6d4" strokeWidth="1" strokeDasharray="5,5" opacity="0.5" />
                </svg>
              </div>
              <p className="text-gray-400 text-xs mt-2">
                Collective motion metric reveals periodic temporal structure
              </p>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-3 text-sm">Clock Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Temperature:</span>
                  <span className="text-orange-400 font-mono">300 K</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Avg Signal:</span>
                  <span className="text-orange-400 font-mono">{avgClockSignal.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Variance:</span>
                  <span className="text-orange-400 font-mono">{clockVariance.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Coherence:</span>
                  <span className="text-orange-400 font-mono">
                    {((1 - clockVariance / Math.max(avgClockSignal, 0.1)) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Molecules:</span>
                  <span className="text-orange-400 font-mono">{molecules.length}</span>
                </div>
              </div>
            </div>

            <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-4">
              <h4 className="text-orange-400 font-semibold mb-2 text-sm">Noise → Information</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Individual molecular motion is unpredictable, but collective behavior exhibits 
                temporal coherence. PPM extracts clock signals from thermal noise, converting 
                randomness into usable information—the basis of biological sensing and timing.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Clock Formation Mechanism */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Clock className="mr-2 text-cyan-400" size={24} />
          Natural Clock Formation
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Stage 1: Random Motion</h4>
            <div className="relative h-32 bg-black/30 rounded-lg mb-3 flex items-center justify-center">
              <div className="text-4xl">🔀</div>
            </div>
            <p className="text-gray-300 text-xs leading-relaxed">
              Thermal energy creates random molecular motion. At T=300K, molecules move with 
              velocities following Maxwell-Boltzmann distribution.
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-3 text-sm">Stage 2: PPM Filtering</h4>
            <div className="relative h-32 bg-black/30 rounded-lg mb-3 flex items-center justify-center">
              <div className="text-4xl">🔍</div>
            </div>
            <p className="text-gray-300 text-xs leading-relaxed">
              Phase Prime Metrics extract coherent patterns from noise. Prime-based 
              filters resonate with natural frequencies, amplifying signal.
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-orange-400 font-semibold mb-3 text-sm">Stage 3: Clock Signal</h4>
            <div className="relative h-32 bg-black/30 rounded-lg mb-3 flex items-center justify-center">
              <div className="text-4xl">⏰</div>
            </div>
            <p className="text-gray-300 text-xs leading-relaxed">
              Emergent temporal pattern provides precise timing reference. Clock operates 
              without external energy, driven by thermal fluctuations.
            </p>
          </div>
        </div>

        <div className="mt-6 bg-purple-900/20 border border-purple-700 rounded-lg p-4">
          <h4 className="text-purple-400 font-semibold mb-2 text-sm flex items-center">
            <TrendingUp className="mr-2" size={16} />
            Mathematical Foundation
          </h4>
          <p className="text-gray-300 text-xs leading-relaxed mb-3">
            The clock signal emerges from the autocorrelation function of molecular positions:
          </p>
          <div className="bg-black/30 rounded p-3 font-mono text-xs text-cyan-400">
            C(τ) = ⟨v(t) · v(t+τ)⟩ = ∑ᵢ pᵢ · cos(ωᵢτ + φᵢ)
          </div>
          <p className="text-gray-300 text-xs leading-relaxed mt-3">
            where pᵢ are phase primes, ωᵢ are resonant frequencies, and φᵢ are geometric phases. 
            This transforms thermal noise into a deterministic clock through PPM phase relationships.
          </p>
        </div>
      </div>

      {/* Applications */}
      <div className="bg-gradient-to-r from-orange-900/20 to-cyan-900/20 border border-orange-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Biological & Technological Applications</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <h4 className="text-orange-400 font-semibold text-sm">Biological Systems:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-orange-400 mr-2">•</span>
                <span><strong>Circadian Rhythms:</strong> 24-hour cycles emerge from molecular thermal clocks</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-2">•</span>
                <span><strong>Neural Timing:</strong> Action potential timing uses thermal diffusion clocks</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-2">•</span>
                <span><strong>Metabolic Cycles:</strong> Cellular processes synchronized by thermal signals</span>
              </li>
              <li className="flex items-start">
                <span className="text-orange-400 mr-2">•</span>
                <span><strong>Consciousness:</strong> Temporal binding through coherent thermal patterns</span>
              </li>
            </ul>
          </div>

          <div className="space-y-3">
            <h4 className="text-cyan-400 font-semibold text-sm">Technology Applications:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">•</span>
                <span><strong>Thermal Sensors:</strong> Ultra-precise temperature measurement devices</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">•</span>
                <span><strong>Energy Harvesting:</strong> Convert ambient thermal noise to clock signals</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">•</span>
                <span><strong>Quantum Timing:</strong> Reference for quantum computing operations</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">•</span>
                <span><strong>Nanoscale Clocks:</strong> Self-powered timing devices without batteries</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
