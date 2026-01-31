import React, { useState, useEffect } from 'react';
import { Brain, Image, Layers, Zap } from 'lucide-react';

/**
 * Section 6.4: Brain's fMRI Images Created on Molecular Surface
 * Section 6.5: Massively Parallel Computing by Cellular Automaton
 * Section 6.6: One-to-Many Orbital Computing in Nano Wheel
 */
export const MolecularComputingPanel: React.FC = () => {
  const [time, setTime] = useState(0);
  const [caState, setCaState] = useState<number[][]>([]);
  const [orbitalState, setOrbitalState] = useState<number[]>([]);
  
  // Cellular automaton grid size
  const gridSize = 20;
  
  useEffect(() => {
    // Initialize cellular automaton
    const initCA = Array.from({ length: gridSize }, () =>
      Array.from({ length: gridSize }, () => Math.random() > 0.7 ? 1 : 0)
    );
    setCaState(initCA);
    
    // Initialize orbital states (8 orbitals in nano wheel)
    setOrbitalState(Array.from({ length: 8 }, (_, i) => i % 2));
  }, []);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(t => t + 1);
      
      // Update cellular automaton (Conway-like rules)
      setCaState(prev => {
        if (prev.length === 0) return prev;
        
        return prev.map((row, i) => row.map((cell, j) => {
          // Count neighbors
          let neighbors = 0;
          for (let di = -1; di <= 1; di++) {
            for (let dj = -1; dj <= 1; dj++) {
              if (di === 0 && dj === 0) continue;
              const ni = (i + di + gridSize) % gridSize;
              const nj = (j + dj + gridSize) % gridSize;
              neighbors += prev[ni][nj];
            }
          }
          
          // Apply rules
          if (cell === 1) {
            return neighbors === 2 || neighbors === 3 ? 1 : 0;
          } else {
            return neighbors === 3 ? 1 : 0;
          }
        }));
      });
      
      // Update orbital states (one-to-many mapping)
      setOrbitalState(prev => {
        const newState = [...prev];
        const inputBit = time % 2;
        // Rotate and transform
        newState.unshift(newState.pop() || 0);
        newState[0] = inputBit;
        newState[1] = inputBit ^ newState[2];
        return newState;
      });
    }, 200);
    
    return () => clearInterval(interval);
  }, [time]);
  
  const activeCells = caState.flat().filter(c => c === 1).length;
  const totalCells = gridSize * gridSize;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-900/30 via-blue-900/30 to-purple-900/30 border border-green-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="text-green-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">6.4-6.6 Molecular Computing Paradigms</h2>
            <p className="text-gray-300 text-sm">fMRI imaging, cellular automata, and orbital computing at nanoscale</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Biological molecules implement massively parallel computing through multiple mechanisms: 
          molecular surface patterns that mirror brain activity (fMRI-like imaging), cellular automaton 
          dynamics for distributed processing, and orbital computing that maps one input to many outputs 
          simultaneously—all governed by Phase Prime Metric principles.
        </p>
      </div>

      {/* Section 6.4: Molecular fMRI */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Image className="mr-2 text-green-400" size={24} />
          6.4 Brain's fMRI on Molecular Surface
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-green-400 font-semibold mb-4 text-sm">Molecular Activity Pattern</h4>
            <div className="relative h-80 bg-black/30 rounded-lg overflow-hidden">
              {/* Simulated molecular surface activity */}
              <svg viewBox="0 0 300 300" className="w-full h-full">
                {Array.from({ length: 30 }).map((_, i) => 
                  Array.from({ length: 30 }).map((_, j) => {
                    const activity = Math.sin((i + time * 0.1) * 0.3) * 
                                   Math.cos((j + time * 0.1) * 0.3) * 0.5 + 0.5;
                    const hue = activity * 120 + 200; // Blue to red
                    
                    return (
                      <rect
                        key={`${i}-${j}`}
                        x={i * 10}
                        y={j * 10}
                        width="10"
                        height="10"
                        fill={`hsl(${hue}, 70%, ${50 + activity * 30}%)`}
                        opacity="0.8"
                      />
                    );
                  })
                )}
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              Molecular surface activity pattern resembling brain fMRI imaging
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Activity Metrics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Resolution:</span>
                  <span className="text-green-400 font-mono">1 nm (molecular scale)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Time Resolution:</span>
                  <span className="text-green-400 font-mono">1 ps</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Signal Type:</span>
                  <span className="text-green-400 font-mono">Electron density</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">SNR:</span>
                  <span className="text-green-400 font-mono">98.5 dB</span>
                </div>
              </div>
            </div>

            <div className="bg-green-900/20 border border-green-700 rounded-lg p-4">
              <h4 className="text-green-400 font-semibold mb-2 text-sm">Molecular fMRI Principle</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Just as fMRI tracks blood oxygen levels to map brain activity, molecular surfaces 
                exhibit electron density patterns that correspond to computational states. These 
                patterns can be read optically or through AFM, providing real-time imaging of 
                molecular information processing at unprecedented resolution—millions of times 
                better than conventional fMRI.
              </p>
            </div>

            <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Scale Correspondence</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Remarkably, molecular activity patterns exhibit fractal similarity to macroscopic 
                brain fMRI images. The same PPM-governed dynamics operate at both scales, suggesting 
                consciousness emerges from identical computational principles expressed across spatial 
                hierarchies.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 6.5: Cellular Automaton */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Layers className="mr-2 text-purple-400" size={24} />
          6.5 Massively Parallel Cellular Automaton
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-4 text-sm">CA State Evolution</h4>
            <div className="relative h-80 bg-black/30 rounded-lg overflow-hidden p-2">
              <div className="grid gap-1" style={{ 
                gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
                height: '100%'
              }}>
                {caState.flat().map((cell, idx) => (
                  <div
                    key={idx}
                    className={`rounded-sm transition-all duration-200 ${
                      cell === 1 
                        ? 'bg-gradient-to-br from-purple-500 to-cyan-500' 
                        : 'bg-gray-800'
                    }`}
                    style={{ aspectRatio: '1/1' }}
                  />
                ))}
              </div>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              {gridSize}×{gridSize} cellular automaton on molecular surface
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-3 text-sm">CA Statistics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Grid Size:</span>
                  <span className="text-purple-400 font-mono">{gridSize}×{gridSize}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Active Cells:</span>
                  <span className="text-purple-400 font-mono">{activeCells}/{totalCells}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Density:</span>
                  <span className="text-purple-400 font-mono">
                    {((activeCells / totalCells) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Generation:</span>
                  <span className="text-purple-400 font-mono">{time}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Update Rate:</span>
                  <span className="text-purple-400 font-mono">5 Hz</span>
                </div>
              </div>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Parallel Processing Power</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Each cell updates simultaneously based on neighbor states—truly parallel computation. 
                In microtubules, ~10⁸ tubulin dimers act as CA cells, providing 100 million parallel 
                processors per neuron. This massive parallelism enables the brain's remarkable 
                computational capacity.
              </p>
            </div>

            <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">CA Rules & PPM</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                CA update rules are governed by Phase Prime Metrics. Each cell's state depends on 
                prime-number neighbor counts, creating natural resonance patterns that maintain 
                computational coherence even in presence of thermal noise.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 6.6: Orbital Computing */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Zap className="mr-2 text-yellow-400" size={24} />
          6.6 One-to-Many Orbital Computing
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-yellow-400 font-semibold mb-4 text-sm">Nano Wheel Orbitals</h4>
            <div className="relative h-80 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 300 300" className="w-full h-full">
                {/* Central hub */}
                <circle cx="150" cy="150" r="30" fill="#1e293b" stroke="#475569" strokeWidth="2" />
                <text x="150" y="155" textAnchor="middle" fill="#fff" fontSize="12">Input</text>
                
                {/* 8 orbitals around wheel */}
                {orbitalState.map((state, i) => {
                  const angle = (i * 360) / 8 - 90;
                  const rad = (angle * Math.PI) / 180;
                  const x = 150 + 100 * Math.cos(rad);
                  const y = 150 + 100 * Math.sin(rad);
                  
                  return (
                    <g key={i}>
                      {/* Connection line */}
                      <line
                        x1="150"
                        y1="150"
                        x2={x}
                        y2={y}
                        stroke={state === 1 ? '#facc15' : '#374151'}
                        strokeWidth="3"
                        opacity="0.6"
                      />
                      
                      {/* Orbital node */}
                      <circle
                        cx={x}
                        cy={y}
                        r="25"
                        fill={state === 1 ? '#facc15' : '#1f2937'}
                        stroke={state === 1 ? '#fef08a' : '#4b5563'}
                        strokeWidth="3"
                        className={state === 1 ? 'animate-pulse' : ''}
                      />
                      
                      {/* Orbital label */}
                      <text
                        x={x}
                        y={y + 5}
                        textAnchor="middle"
                        fill={state === 1 ? '#000' : '#9ca3af'}
                        fontSize="14"
                        fontWeight="bold"
                      >
                        {i + 1}
                      </text>
                      
                      {/* State indicator */}
                      <text
                        x={x}
                        y={y - 35}
                        textAnchor="middle"
                        fill="#9ca3af"
                        fontSize="10"
                      >
                        {state}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              8 output orbitals driven by single input—exponential fan-out
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Orbital States</h4>
              <div className="space-y-2">
                {orbitalState.map((state, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-gray-400">Orbital {i + 1}:</span>
                    <div className="flex items-center space-x-2">
                      <div className={`w-16 h-2 rounded-full ${
                        state === 1 ? 'bg-yellow-400' : 'bg-gray-600'
                      }`} />
                      <span className={`font-mono ${
                        state === 1 ? 'text-yellow-400' : 'text-gray-500'
                      }`}>
                        {state}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-4">
              <h4 className="text-yellow-400 font-semibold mb-2 text-sm">One-to-Many Mapping</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                A single input electron can simultaneously influence multiple orbital states through 
                quantum superposition and PPM-governed resonances. This creates exponential 
                computational fan-out: one input generates 2⁸ = 256 possible output configurations. 
                In biological systems, this enables rapid parallel search and pattern matching.
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Many-to-One Integration</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                The reverse process—many inputs converging to one output—implements weighted 
                summation and decision-making. Combined with one-to-many distribution, nano wheels 
                form complete neural network building blocks operating at molecular scale with 
                femtosecond switching times.
              </p>
            </div>

            <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Computational Advantage</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Orbital computing transcends Boolean logic. Instead of sequential bit operations, 
                it performs continuous analog computation in phase space. This enables solving NP-hard 
                problems efficiently by exploring solution space in parallel through quantum orbital 
                interference patterns.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Integration Summary */}
      <div className="bg-gradient-to-r from-green-900/20 to-purple-900/20 border border-green-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Unified Molecular Computing Architecture</h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-green-400 font-semibold mb-2 text-sm">Molecular fMRI</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Real-time observation of molecular computation through surface electron density patterns
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-2 text-sm">Cellular Automaton</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Massively parallel distributed processing with 10⁸ processors per microtubule
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-yellow-400 font-semibold mb-2 text-sm">Orbital Computing</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Exponential fan-out and integration through quantum orbital resonances
            </p>
          </div>
        </div>

        <div className="mt-4 bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
          <p className="text-gray-300 text-xs leading-relaxed">
            <strong className="text-cyan-400">Combined Power:</strong> These three mechanisms work 
            synergistically in biological systems. Orbital computing generates possibilities, cellular 
            automata process them in parallel, and molecular fMRI patterns enable higher-level systems 
            to read and integrate results. All coordinated by Phase Prime Metrics, creating computational 
            capacity far exceeding conventional computing paradigms.
          </p>
        </div>
      </div>
    </div>
  );
};
