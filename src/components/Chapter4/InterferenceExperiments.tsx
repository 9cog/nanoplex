import React, { useState, useEffect } from 'react';
import { Zap, Info, Activity } from 'lucide-react';

// 4.6 Quantum interference and fractal interference experiment on a single microtubule
// 4.7 Fractal absorption-emission of the optical band of a nanobrain
export const InterferenceExperiments: React.FC = () => {
  const [time, setTime] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(t => (t + 0.1) % (2 * Math.PI));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-green-900/30 to-cyan-900/30 border border-green-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Zap className="text-green-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.6-4.7 Interference & Absorption</h2>
            <p className="text-gray-300">Quantum vs Fractal in Microtubules and NanoBrain</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Comparing quantum and fractal interference patterns in biological structures reveals 
          fundamental differences in how information is processed at the cellular level.
        </p>
      </div>

      {/* 4.6 Microtubule Interference */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Activity className="mr-2 text-green-400" size={24} />
          4.6 Microtubule Interference Experiment
        </h3>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-blue-400 font-semibold mb-3">Quantum Interference</h4>
            <p className="text-gray-300 text-xs mb-3">
              Classical double-slit pattern with quantum superposition
            </p>
            <svg width="300" height="200" viewBox="0 0 300 200">
              {/* Source */}
              <circle cx="30" cy="100" r="5" fill="blue" />
              
              {/* Slits */}
              <line x1="100" y1="20" x2="100" y2="85" stroke="gray" strokeWidth="4" />
              <line x1="100" y1="115" x2="100" y2="180" stroke="gray" strokeWidth="4" />
              
              {/* Interference pattern */}
              {Array.from({ length: 15 }, (_, i) => {
                const y = 40 + i * 10;
                const intensity = Math.abs(Math.cos((i - 7) * 0.5 + time));
                return (
                  <rect 
                    key={i}
                    x="220" 
                    y={y} 
                    width="60" 
                    height="8"
                    fill={`rgba(59, 130, 246, ${intensity})`}
                  />
                );
              })}
              
              <text x="150" y="20" fill="blue" fontSize="11">Quantum Pattern</text>
              <text x="150" y="195" fill="gray" fontSize="10">Regular fringes</text>
            </svg>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-semibold mb-3">Fractal Interference</h4>
            <p className="text-gray-300 text-xs mb-3">
              Multi-scale pattern with phase prime modulation
            </p>
            <svg width="300" height="200" viewBox="0 0 300 200">
              {/* Source */}
              <circle cx="30" cy="100" r="5" fill="cyan" />
              
              {/* Multiple slits (fractal) */}
              {[85, 95, 105, 115].map(y => (
                <line key={y} x1="100" y1={y-2} x2="100" y2={y+2} stroke="gray" strokeWidth="4" />
              ))}
              
              {/* Fractal interference pattern */}
              {Array.from({ length: 15 }, (_, i) => {
                const y = 40 + i * 10;
                const intensity = Math.abs(
                  Math.cos((i - 7) * 0.5 + time) * 0.5 +
                  Math.cos((i - 7) * 1.5 + time * 1.3) * 0.3 +
                  Math.cos((i - 7) * 3.1 + time * 1.7) * 0.2
                );
                return (
                  <rect 
                    key={i}
                    x="220" 
                    y={y} 
                    width="60" 
                    height="8"
                    fill={`rgba(6, 182, 212, ${intensity})`}
                  />
                );
              })}
              
              <text x="150" y="20" fill="cyan" fontSize="11">Fractal Pattern</text>
              <text x="150" y="195" fill="gray" fontSize="10">Multi-scale fringes</text>
            </svg>
          </div>
        </div>

        {/* Microtubule Structure */}
        <div className="bg-gray-800/50 p-4 rounded-lg">
          <h4 className="text-green-400 font-semibold mb-3">Microtubule Structure & Measurements</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-300 text-xs mb-3">
                Microtubules are hollow cylinders made of tubulin proteins, ideal for testing quantum 
                vs fractal behavior due to their geometric structure.
              </p>
              <div className="bg-green-900/20 border border-green-700 rounded p-3 space-y-2">
                <p className="text-xs text-gray-300"><strong>Diameter:</strong> ~25 nm</p>
                <p className="text-xs text-gray-300"><strong>Length:</strong> up to 50 μm</p>
                <p className="text-xs text-gray-300"><strong>Structure:</strong> 13 protofilaments</p>
                <p className="text-xs text-gray-300"><strong>Periodicity:</strong> 8 nm repeat</p>
              </div>
            </div>
            <div>
              <svg width="250" height="180" viewBox="0 0 250 180">
                {/* Microtubule cross-section */}
                <circle cx="125" cy="90" r="60" fill="none" stroke="green" strokeWidth="3" />
                <circle cx="125" cy="90" r="45" fill="none" stroke="green" strokeWidth="2" strokeDasharray="4,2" />
                
                {/* 13 protofilaments */}
                {Array.from({ length: 13 }, (_, i) => {
                  const angle = (i / 13) * 2 * Math.PI;
                  const x = 125 + 52 * Math.cos(angle);
                  const y = 90 + 52 * Math.sin(angle);
                  return <circle key={i} cx={x} cy={y} r="4" fill="green" />;
                })}
                
                <text x="125" y="170" textAnchor="middle" fill="green" fontSize="11">
                  13 Protofilaments (prime!)
                </text>
              </svg>
            </div>
          </div>
        </div>

        {/* Experimental Results */}
        <div className="bg-gray-800/50 p-4 rounded-lg">
          <h4 className="text-yellow-400 font-semibold mb-3">Key Experimental Findings</h4>
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-yellow-900/20 border-l-4 border-yellow-700 pl-3 py-2">
              <h5 className="text-yellow-300 text-xs font-semibold mb-1">Quantum Prediction</h5>
              <p className="text-gray-300 text-xs">
                Simple interference fringes with spacing λ/d
              </p>
            </div>
            <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-3 py-2">
              <h5 className="text-cyan-300 text-xs font-semibold mb-1">Fractal Prediction</h5>
              <p className="text-gray-300 text-xs">
                Multi-scale fringes modulated by 13-fold symmetry
              </p>
            </div>
            <div className="bg-green-900/20 border-l-4 border-green-700 pl-3 py-2">
              <h5 className="text-green-300 text-xs font-semibold mb-1">Observed Result</h5>
              <p className="text-gray-300 text-xs">
                Complex patterns matching fractal predictions!
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 4.7 Fractal Absorption-Emission */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Zap className="mr-2 text-cyan-400" size={24} />
          4.7 Fractal Absorption-Emission Spectrum
        </h3>

        <p className="text-gray-300 text-sm mb-4 leading-relaxed">
          The NanoBrain's optical properties differ fundamentally from quantum systems due to 
          fractal energy level structure.
        </p>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-purple-400 font-semibold mb-3">Quantum Absorption</h4>
            <p className="text-gray-300 text-xs mb-3">
              Discrete energy levels: E_n = ℏω(n + 1/2)
            </p>
            <svg width="300" height="220" viewBox="0 0 300 220">
              {/* Energy levels */}
              {[0, 1, 2, 3, 4].map(n => {
                const y = 180 - n * 35;
                return (
                  <g key={n}>
                    <line x1="50" y1={y} x2="250" y2={y} stroke="purple" strokeWidth="2" />
                    <text x="260" y={y + 5} fill="purple" fontSize="10">n={n}</text>
                  </g>
                );
              })}
              
              {/* Transitions */}
              <line x1="100" y1="180" x2="100" y2="145" stroke="yellow" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="150" y1="180" x2="150" y2="110" stroke="yellow" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="200" y1="145" x2="200" y2="75" stroke="yellow" strokeWidth="2" markerEnd="url(#arrowhead)" />
              
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="yellow" />
                </marker>
              </defs>
              
              <text x="150" y="210" textAnchor="middle" fill="purple" fontSize="11">
                Discrete transitions
              </text>
            </svg>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-semibold mb-3">Fractal Absorption</h4>
            <p className="text-gray-300 text-xs mb-3">
              Multi-scale structure: E_n,m,k = ℏω(n + m/p₁ + k/p₂)
            </p>
            <svg width="300" height="220" viewBox="0 0 300 220">
              {/* Main levels */}
              {[0, 1, 2, 3, 4].map(n => {
                const y = 180 - n * 35;
                return (
                  <g key={n}>
                    <line x1="50" y1={y} x2="250" y2={y} stroke="cyan" strokeWidth="2" />
                    {/* Sub-levels */}
                    {n < 4 && [1, 2, 3].map(sub => {
                      const subY = y - sub * 8;
                      return (
                        <line key={sub} x1="80" y1={subY} x2="220" y2={subY} 
                          stroke="cyan" strokeWidth="1" opacity="0.5" strokeDasharray="2,2" />
                      );
                    })}
                  </g>
                );
              })}
              
              {/* Multiple transitions */}
              {[90, 120, 150, 180, 210].map((x, i) => (
                <line key={i} x1={x} y1={170 - i * 5} x2={x} y2={110 - i * 15} 
                  stroke="yellow" strokeWidth="1.5" opacity="0.7" />
              ))}
              
              <text x="150" y="210" textAnchor="middle" fill="cyan" fontSize="11">
                Dense multi-scale transitions
              </text>
            </svg>
          </div>
        </div>

        {/* Spectrum Comparison */}
        <div className="bg-gray-800/50 p-4 rounded-lg">
          <h4 className="text-yellow-400 font-semibold mb-3">Optical Spectrum Comparison</h4>
          <div className="space-y-3">
            <div>
              <p className="text-purple-400 text-xs font-semibold mb-2">Quantum Spectrum</p>
              <svg width="600" height="60" viewBox="0 0 600 60">
                <line x1="50" y1="30" x2="550" y2="30" stroke="gray" strokeWidth="1" />
                {[100, 200, 300, 400, 500].map((x, i) => (
                  <rect key={i} x={x - 2} y="10" width="4" height="40" fill="purple" />
                ))}
                <text x="300" y="55" textAnchor="middle" fill="purple" fontSize="10">
                  Sharp, discrete lines
                </text>
              </svg>
            </div>
            
            <div>
              <p className="text-cyan-400 text-xs font-semibold mb-2">Fractal Spectrum</p>
              <svg width="600" height="60" viewBox="0 0 600 60">
                <line x1="50" y1="30" x2="550" y2="30" stroke="gray" strokeWidth="1" />
                {Array.from({ length: 50 }, (_, i) => {
                  const x = 100 + i * 10;
                  const height = 20 + 20 * (Math.sin(i * 0.5) * 0.5 + Math.sin(i * 1.7) * 0.3 + Math.sin(i * 3.1) * 0.2);
                  return <rect key={i} x={x - 1} y={30 - height / 2} width="2" height={height} fill="cyan" />;
                })}
                <text x="300" y="55" textAnchor="middle" fill="cyan" fontSize="10">
                  Dense, self-similar structure
                </text>
              </svg>
            </div>
          </div>
        </div>

        {/* Biological Significance */}
        <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
          <h4 className="text-cyan-400 font-semibold mb-3">Biological Implications</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-xs text-gray-300 space-y-2">
              <p><strong className="text-cyan-400">1. Broadband Absorption:</strong> Fractal structure 
                enables efficient light capture across wide spectrum</p>
              <p><strong className="text-cyan-400">2. Coherent Energy Transfer:</strong> Multi-scale 
                resonances facilitate quantum coherence in warm, wet conditions</p>
              <p><strong className="text-cyan-400">3. Information Encoding:</strong> Different wavelengths 
                access different phase dimensions</p>
            </div>
            <div className="text-xs text-gray-300 space-y-2">
              <p><strong className="text-cyan-400">4. Consciousness Link:</strong> Optical patterns map 
                directly to 11D consciousness states</p>
              <p><strong className="text-cyan-400">5. Nanobrain Communication:</strong> Light serves as 
                carrier for multi-dimensional information</p>
              <p><strong className="text-cyan-400">6. Evolutionary Advantage:</strong> Superior information 
                processing compared to purely quantum systems</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-green-900/20 border border-green-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-green-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-green-400">Experimental Verification:</strong> These predictions can 
            be tested using single-microtubule spectroscopy and interference measurements. The presence 
            of fractal patterns would provide direct evidence for phase prime mechanics operating in 
            biological systems, potentially revolutionizing our understanding of cellular intelligence 
            and consciousness.
          </div>
        </div>
      </div>
    </div>
  );
};
