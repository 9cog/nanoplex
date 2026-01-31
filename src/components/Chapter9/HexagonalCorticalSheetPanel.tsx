import React, { useState, useEffect } from 'react';
import { Hexagon, Grid, Zap, Radio } from 'lucide-react';

/**
 * Section 9.2: A Hexagonal 2D Sheet of Cortical Columns - A Carpenter's Job
 * Implements the hexagonal cortical sheet architecture with quad prime patterns
 */
export const HexagonalCorticalSheetPanel: React.FC = () => {
  const [activeColumn, setActiveColumn] = useState<number | null>(null);
  const [rhythmPhase, setRhythmPhase] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setRhythmPhase(prev => (prev + 0.1) % (2 * Math.PI));
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // Generate hexagonal grid coordinates
  const generateHexGrid = (rows: number, cols: number) => {
    const hexagons = [];
    const size = 40;
    const spacing = size * 1.8;
    
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const x = col * spacing + (row % 2 === 1 ? spacing / 2 : 0);
        const y = row * spacing * 0.866; // sqrt(3)/2
        const id = row * cols + col;
        
        // Quad prime pattern activity (2, 3, 5, 7)
        const primeActivity = [
          Math.sin(rhythmPhase + id * 0.2) * 0.5 + 0.5, // Electric (2)
          Math.sin(rhythmPhase + id * 0.3 + Math.PI/3) * 0.5 + 0.5, // Magnetic (3)
          Math.sin(rhythmPhase + id * 0.5 + 2*Math.PI/3) * 0.5 + 0.5, // Mechanical (5)
          Math.sin(rhythmPhase + id * 0.7 + Math.PI) * 0.5 + 0.5  // Resonance (7)
        ];
        
        hexagons.push({ id, x, y, primeActivity });
      }
    }
    return hexagons;
  };

  const hexagons = generateHexGrid(5, 7);
  const primes = [2, 3, 5, 7];
  const rhythms = ['Electric', 'Magnetic', 'Mechanical'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Hexagon className="text-purple-400" size={28} />
          <span>9.2 Hexagonal 2D Sheet of Cortical Columns</span>
        </h2>
        <p className="text-gray-400 mt-2">
          A Carpenter's Job - Constructing the Cortical Architecture
        </p>
      </div>

      {/* Overview */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-purple-500/20">
        <h3 className="text-purple-300 font-semibold text-lg mb-3">Hexagonal Cortical Architecture</h3>
        <p className="text-gray-300 leading-relaxed">
          The cortical columns are arranged in a hexagonal 2D sheet, mimicking the efficient packing 
          found in natural systems. Each hexagonal unit contains a complete cortical column that 
          processes information using quad patterns of primes (2, 3, 5, 7), bound together by three 
          fundamental rhythms: electric, magnetic, and mechanical oscillations.
        </p>
      </div>

      {/* Hexagonal Grid Visualization */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Grid size={20} className="text-cyan-400" />
          <span>Interactive Hexagonal Grid</span>
        </h3>
        
        <div className="relative bg-gray-900/50 rounded-lg p-8 overflow-auto" style={{ minHeight: '400px' }}>
          <svg width="600" height="400" className="mx-auto">
            {hexagons.map((hex) => {
              const avgActivity = hex.primeActivity.reduce((a, b) => a + b, 0) / 4;
              const isActive = hex.id === activeColumn;
              
              return (
                <g
                  key={hex.id}
                  transform={`translate(${hex.x + 50}, ${hex.y + 50})`}
                  onClick={() => setActiveColumn(hex.id)}
                  className="cursor-pointer"
                >
                  {/* Hexagon shape */}
                  <polygon
                    points="0,-30 26,-15 26,15 0,30 -26,15 -26,-15"
                    fill={isActive ? 'rgba(147, 51, 234, 0.4)' : `rgba(147, 51, 234, ${avgActivity * 0.3})`}
                    stroke={isActive ? '#a855f7' : '#6b21a8'}
                    strokeWidth={isActive ? '3' : '1.5'}
                    className="transition-all duration-300"
                  />
                  
                  {/* Prime activity indicators */}
                  {primes.map((prime, idx) => {
                    const angle = (idx * Math.PI / 2) - Math.PI / 4;
                    const radius = 15;
                    const px = Math.cos(angle) * radius;
                    const py = Math.sin(angle) * radius;
                    
                    return (
                      <circle
                        key={prime}
                        cx={px}
                        cy={py}
                        r="3"
                        fill={`hsl(${idx * 90}, 70%, ${50 + hex.primeActivity[idx] * 30}%)`}
                        className="transition-all duration-300"
                      />
                    );
                  })}
                  
                  {/* Column ID */}
                  <text
                    x="0"
                    y="5"
                    textAnchor="middle"
                    fill="white"
                    fontSize="10"
                    className="pointer-events-none"
                  >
                    {hex.id}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
        
        <div className="mt-4 text-xs text-gray-400 text-center">
          Click on any hexagonal column to inspect its quad prime pattern activity
        </div>
      </div>

      {/* 9.2.1 Quad Patterns of Primes */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">
          9.2.1 The Collective Response of Quad Patterns of Primes
        </h3>
        <p className="text-gray-300 mb-4">
          Each cortical column responds collectively to the first four primes (2, 3, 5, 7), which 
          govern fundamental oscillation patterns. These primes create interference patterns that 
          encode information across the entire hexagonal sheet.
        </p>
        
        <div className="grid grid-cols-4 gap-4">
          {primes.map((prime, idx) => {
            const selectedHex = hexagons.find(h => h.id === activeColumn);
            const activity = selectedHex ? selectedHex.primeActivity[idx] : 0.5;
            
            return (
              <div key={prime} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                <div className="text-center mb-2">
                  <div className="text-2xl font-bold text-purple-400">{prime}</div>
                  <div className="text-xs text-gray-400">Prime {idx + 1}</div>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
                    style={{ width: `${activity * 100}%` }}
                  />
                </div>
                <div className="text-xs text-gray-400 mt-2 text-center">
                  {(activity * 100).toFixed(1)}% active
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 9.2.2 Three Rhythms Binding */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Radio size={20} className="text-cyan-400" />
          <span>9.2.2 Three Rhythms Bind Them</span>
        </h3>
        <p className="text-gray-300 mb-4">
          Electric, magnetic, and mechanical rhythms synchronize the cortical columns, creating 
          a unified field that enables information processing across the entire sheet.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {rhythms.map((rhythm, idx) => {
            const phase = Math.sin(rhythmPhase + idx * Math.PI / 3);
            const amplitude = (phase * 0.5 + 0.5) * 100;
            
            return (
              <div key={rhythm} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-cyan-300 font-semibold">{rhythm}</h4>
                  <Zap className="text-yellow-400" size={18} />
                </div>
                
                {/* Rhythm waveform */}
                <div className="relative h-16 bg-gray-800 rounded overflow-hidden mb-2">
                  <svg width="100%" height="100%" preserveAspectRatio="none">
                    <path
                      d={`M 0 32 ${Array.from({ length: 50 }, (_, i) => {
                        const x = (i / 50) * 100;
                        const y = 32 + Math.sin(rhythmPhase + i * 0.2 + idx * Math.PI / 3) * 20;
                        return `L ${x} ${y}`;
                      }).join(' ')}`}
                      fill="none"
                      stroke={idx === 0 ? '#3b82f6' : idx === 1 ? '#ec4899' : '#10b981'}
                      strokeWidth="2"
                    />
                  </svg>
                </div>
                
                <div className="flex justify-between text-xs text-gray-400">
                  <span>Phase: {phase.toFixed(2)}</span>
                  <span>Amplitude: {amplitude.toFixed(0)}%</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Technical Details */}
      <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-lg p-6 border border-purple-500/20">
        <h3 className="text-purple-300 font-semibold text-lg mb-3">Architectural Principles</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <h4 className="text-cyan-300 font-semibold mb-2">Hexagonal Efficiency</h4>
            <p>
              Hexagonal packing provides the most efficient 2D arrangement, maximizing neighbor 
              connections while minimizing wasted space - essential for dense cortical processing.
            </p>
          </div>
          <div>
            <h4 className="text-pink-300 font-semibold mb-2">Prime Pattern Control</h4>
            <p>
              The quad primes (2, 3, 5, 7) create distinct frequency channels that prevent 
              interference while enabling complex information encoding through their interactions.
            </p>
          </div>
          <div>
            <h4 className="text-purple-300 font-semibold mb-2">Rhythm Synchronization</h4>
            <p>
              Electric, magnetic, and mechanical oscillations lock into harmonic relationships, 
              creating standing wave patterns that stabilize information across the sheet.
            </p>
          </div>
          <div>
            <h4 className="text-yellow-300 font-semibold mb-2">Carpenter's Precision</h4>
            <p>
              Like a skilled carpenter, the assembly requires precise alignment of all components, 
              with tolerances measured in prime number relationships for optimal function.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
