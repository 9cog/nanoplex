import React, { useState, useEffect } from 'react';
import { Clock, Waves, Zap, GitMerge, Activity, Info } from 'lucide-react';

// 4.1 Revisiting basic concepts of quantum mechanics using clocks
// 4.1.1 Wave-particle duality, beating, interference, entanglement, harmonic oscillator
export const QuantumClockBasics: React.FC = () => {
  const [time, setTime] = useState(0);
  const [activeDemo, setActiveDemo] = useState<string>('wave-particle');

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(t => (t + 0.05) % (2 * Math.PI));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  const demos = [
    { id: 'wave-particle', label: 'Wave-Particle Duality', icon: Waves },
    { id: 'beating', label: 'Beating Phenomenon', icon: Activity },
    { id: 'interference', label: 'Interference', icon: GitMerge },
    { id: 'entanglement', label: 'Entanglement', icon: Zap },
  ];

  const renderWaveParticleDuality = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-4 rounded-lg">
        <h4 className="text-cyan-400 font-semibold mb-2 flex items-center">
          <Waves className="mr-2" size={18} />
          Clock Representation of Wave-Particle Duality
        </h4>
        <p className="text-gray-300 text-sm mb-4">
          A clock hand rotating represents both wave (circular motion) and particle (hand position)
        </p>
        <div className="flex justify-center items-center space-x-8">
          {/* Circular Clock */}
          <div className="relative">
            <svg width="120" height="120" viewBox="-60 -60 120 120">
              <circle cx="0" cy="0" r="50" fill="none" stroke="cyan" strokeWidth="2" />
              <line 
                x1="0" y1="0" 
                x2={Math.cos(time) * 45} 
                y2={Math.sin(time) * 45}
                stroke="yellow" 
                strokeWidth="3" 
              />
              <circle cx={Math.cos(time) * 45} cy={Math.sin(time) * 45} r="5" fill="yellow" />
            </svg>
            <p className="text-center text-xs text-gray-400 mt-2">Particle View</p>
          </div>
          
          {/* Wave Representation */}
          <div>
            <svg width="200" height="120" viewBox="0 0 200 120">
              <path
                d={`M 0 60 ${Array.from({ length: 50 }, (_, i) => {
                  const x = i * 4;
                  const y = 60 + 30 * Math.sin(time - i * 0.2);
                  return `L ${x} ${y}`;
                }).join(' ')}`}
                fill="none"
                stroke="cyan"
                strokeWidth="2"
              />
              <circle cx={time * 30} cy={60 + 30 * Math.sin(time)} r="5" fill="yellow" />
            </svg>
            <p className="text-center text-xs text-gray-400 mt-2">Wave View</p>
          </div>
        </div>
      </div>

      <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-blue-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-blue-400">Key Insight:</strong> The clock hand's rotation creates both 
            a circular trajectory (wave) and a definite position at any instant (particle). This duality 
            is fundamental to quantum mechanics and fractal mechanics alike.
          </div>
        </div>
      </div>
    </div>
  );

  const renderBeating = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-4 rounded-lg">
        <h4 className="text-purple-400 font-semibold mb-2 flex items-center">
          <Activity className="mr-2" size={18} />
          Beating: Two Clocks at Different Frequencies
        </h4>
        <p className="text-gray-300 text-sm mb-4">
          When two clocks rotate at slightly different speeds, their interference creates beating patterns
        </p>
        <div className="flex justify-center items-center space-x-8">
          {/* Clock 1 */}
          <div>
            <svg width="100" height="100" viewBox="-50 -50 100 100">
              <circle cx="0" cy="0" r="40" fill="none" stroke="cyan" strokeWidth="2" />
              <line 
                x1="0" y1="0" 
                x2={Math.cos(time * 1.0) * 35} 
                y2={Math.sin(time * 1.0) * 35}
                stroke="cyan" 
                strokeWidth="2" 
              />
            </svg>
            <p className="text-center text-xs text-gray-400">ω₁ = 1.0</p>
          </div>
          
          {/* Clock 2 */}
          <div>
            <svg width="100" height="100" viewBox="-50 -50 100 100">
              <circle cx="0" cy="0" r="40" fill="none" stroke="purple" strokeWidth="2" />
              <line 
                x1="0" y1="0" 
                x2={Math.cos(time * 1.2) * 35} 
                y2={Math.sin(time * 1.2) * 35}
                stroke="purple" 
                strokeWidth="2" 
              />
            </svg>
            <p className="text-center text-xs text-gray-400">ω₂ = 1.2</p>
          </div>
          
          {/* Combined Beat */}
          <div>
            <svg width="100" height="100" viewBox="-50 -50 100 100">
              <circle cx="0" cy="0" r="40" fill="none" stroke="yellow" strokeWidth="2" strokeDasharray="4,2" />
              <line 
                x1="0" y1="0" 
                x2={(Math.cos(time * 1.0) + Math.cos(time * 1.2)) * 17} 
                y2={(Math.sin(time * 1.0) + Math.sin(time * 1.2)) * 17}
                stroke="yellow" 
                strokeWidth="3" 
              />
              <circle 
                cx={(Math.cos(time * 1.0) + Math.cos(time * 1.2)) * 17}
                cy={(Math.sin(time * 1.0) + Math.sin(time * 1.2)) * 17}
                r="5" 
                fill="yellow" 
              />
            </svg>
            <p className="text-center text-xs text-gray-400">Beat Pattern</p>
          </div>
        </div>
        
        {/* Beat Envelope */}
        <div className="mt-4">
          <svg width="300" height="80" viewBox="0 0 300 80" className="mx-auto">
            <path
              d={`M 0 40 ${Array.from({ length: 75 }, (_, i) => {
                const x = i * 4;
                const beat = Math.cos(time * 1.0 - i * 0.1) + Math.cos(time * 1.2 - i * 0.1);
                const y = 40 - beat * 15;
                return `L ${x} ${y}`;
              }).join(' ')}`}
              fill="none"
              stroke="yellow"
              strokeWidth="2"
            />
          </svg>
          <p className="text-center text-xs text-gray-400 mt-1">Beat Waveform: |ω₁ - ω₂|</p>
        </div>
      </div>

      <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-purple-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-purple-400">Beating Formula:</strong> When two frequencies ω₁ and ω₂ 
            combine, the result oscillates at the average frequency (ω₁+ω₂)/2 with an amplitude modulated 
            at the beat frequency |ω₁-ω₂|/2. This is fundamental to understanding interference patterns.
          </div>
        </div>
      </div>
    </div>
  );

  const renderInterference = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-4 rounded-lg">
        <h4 className="text-green-400 font-semibold mb-2 flex items-center">
          <GitMerge className="mr-2" size={18} />
          Clock-Based Interference Patterns
        </h4>
        <p className="text-gray-300 text-sm mb-4">
          Multiple clocks create constructive and destructive interference
        </p>
        
        {/* Interference Pattern */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h5 className="text-green-300 text-sm mb-2">Source Clocks</h5>
            <div className="flex space-x-4 justify-center">
              {[0, 1, 2].map(i => (
                <svg key={i} width="60" height="60" viewBox="-30 -30 60 60">
                  <circle cx="0" cy="0" r="25" fill="none" stroke="green" strokeWidth="1.5" />
                  <line 
                    x1="0" y1="0" 
                    x2={Math.cos(time + i * 0.5) * 20} 
                    y2={Math.sin(time + i * 0.5) * 20}
                    stroke="green" 
                    strokeWidth="2" 
                  />
                </svg>
              ))}
            </div>
          </div>
          
          <div>
            <h5 className="text-green-300 text-sm mb-2">Interference Result</h5>
            <svg width="200" height="100" viewBox="0 0 200 100" className="mx-auto">
              {Array.from({ length: 40 }, (_, i) => {
                const x = i * 5;
                const interference = Math.cos(time - i * 0.15) + 
                                   Math.cos(time + 0.5 - i * 0.15) + 
                                   Math.cos(time + 1.0 - i * 0.15);
                const intensity = Math.abs(interference) / 3;
                return (
                  <rect 
                    key={i}
                    x={x} 
                    y={50 - intensity * 40} 
                    width="4" 
                    height={intensity * 80}
                    fill={`rgba(74, 222, 128, ${intensity})`}
                  />
                );
              })}
            </svg>
            <p className="text-center text-xs text-gray-400 mt-1">Interference Pattern</p>
          </div>
        </div>
      </div>

      <div className="bg-green-900/20 border border-green-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-green-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-green-400">Interference Principle:</strong> When multiple clock phases 
            align (in phase), they constructively interfere creating bright fringes. When out of phase by π, 
            they destructively interfere creating dark fringes. This explains double-slit experiments.
          </div>
        </div>
      </div>
    </div>
  );

  const renderEntanglement = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-4 rounded-lg">
        <h4 className="text-yellow-400 font-semibold mb-2 flex items-center">
          <Zap className="mr-2" size={18} />
          Quantum Entanglement via Synchronized Clocks
        </h4>
        <p className="text-gray-300 text-sm mb-4">
          Entangled particles behave like synchronized clocks maintaining correlated phases
        </p>
        
        <div className="flex justify-center items-center space-x-12">
          {/* Clock A */}
          <div className="text-center">
            <svg width="100" height="100" viewBox="-50 -50 100 100">
              <circle cx="0" cy="0" r="40" fill="none" stroke="yellow" strokeWidth="2" />
              <line 
                x1="0" y1="0" 
                x2={Math.cos(time) * 35} 
                y2={Math.sin(time) * 35}
                stroke="yellow" 
                strokeWidth="3" 
              />
              <circle cx={Math.cos(time) * 35} cy={Math.sin(time) * 35} r="5" fill="yellow" />
            </svg>
            <p className="text-xs text-gray-400 mt-2">Particle A</p>
            <p className="text-xs text-yellow-400">θ = {(time * 180 / Math.PI).toFixed(0)}°</p>
          </div>
          
          {/* Entanglement Link */}
          <div className="text-center">
            <Zap className="text-yellow-400 animate-pulse" size={32} />
            <p className="text-xs text-gray-400 mt-1">Entangled</p>
          </div>
          
          {/* Clock B - Correlated */}
          <div className="text-center">
            <svg width="100" height="100" viewBox="-50 -50 100 100">
              <circle cx="0" cy="0" r="40" fill="none" stroke="orange" strokeWidth="2" />
              <line 
                x1="0" y1="0" 
                x2={Math.cos(time + Math.PI) * 35} 
                y2={Math.sin(time + Math.PI) * 35}
                stroke="orange" 
                strokeWidth="3" 
              />
              <circle cx={Math.cos(time + Math.PI) * 35} cy={Math.sin(time + Math.PI) * 35} r="5" fill="orange" />
            </svg>
            <p className="text-xs text-gray-400 mt-2">Particle B</p>
            <p className="text-xs text-orange-400">θ = {((time + Math.PI) * 180 / Math.PI).toFixed(0)}°</p>
          </div>
        </div>
        
        <div className="mt-4 text-center text-sm text-gray-300">
          <p>Phase Correlation: θ_B = θ_A + π (Anti-correlated spins)</p>
        </div>
      </div>

      <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-yellow-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-yellow-400">Entanglement Insight:</strong> Entangled particles maintain 
            phase coherence regardless of distance. Measuring one clock's position instantly determines the 
            other's phase. This non-local correlation is central to quantum mechanics and consciousness theories.
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Clock className="text-cyan-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.1 Quantum Mechanics Using Clocks</h2>
            <p className="text-gray-300">Wave-particle duality, beating, interference, entanglement</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Understanding quantum phenomena through clock rotations provides an intuitive bridge between 
          classical wave mechanics and quantum behavior. Each rotating clock represents a phasor that 
          can exhibit both wave-like and particle-like properties.
        </p>
      </div>

      {/* Demo Selection */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="grid grid-cols-4 gap-3">
          {demos.map(demo => {
            const Icon = demo.icon;
            return (
              <button
                key={demo.id}
                onClick={() => setActiveDemo(demo.id)}
                className={`flex flex-col items-center p-3 rounded-lg border-2 transition-all duration-300 ${
                  activeDemo === demo.id
                    ? 'bg-cyan-900/30 border-cyan-700 text-cyan-400'
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                <Icon size={24} />
                <span className="text-xs font-semibold mt-2 text-center">{demo.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Demo */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        {activeDemo === 'wave-particle' && renderWaveParticleDuality()}
        {activeDemo === 'beating' && renderBeating()}
        {activeDemo === 'interference' && renderInterference()}
        {activeDemo === 'entanglement' && renderEntanglement()}
      </div>

      {/* Harmonic Oscillator Section */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Activity className="mr-2 text-cyan-400" size={24} />
          Harmonic Oscillator
        </h3>
        
        <div className="space-y-4">
          <p className="text-gray-300 text-sm">
            A harmonic oscillator is perfectly represented by a rotating clock, where the projection 
            onto any axis gives simple harmonic motion.
          </p>
          
          <div className="flex justify-center items-center space-x-8">
            <div>
              <svg width="150" height="150" viewBox="-75 -75 150 150">
                <circle cx="0" cy="0" r="60" fill="none" stroke="cyan" strokeWidth="2" />
                <line x1="-70" y1="0" x2="70" y2="0" stroke="gray" strokeWidth="1" strokeDasharray="4,2" />
                <line x1="0" y1="-70" x2="0" y2="70" stroke="gray" strokeWidth="1" strokeDasharray="4,2" />
                <line 
                  x1="0" y1="0" 
                  x2={Math.cos(time) * 55} 
                  y2={Math.sin(time) * 55}
                  stroke="cyan" 
                  strokeWidth="3" 
                />
                <circle cx={Math.cos(time) * 55} cy={Math.sin(time) * 55} r="6" fill="cyan" />
                <line 
                  x1={Math.cos(time) * 55} 
                  y1={Math.sin(time) * 55}
                  x2={Math.cos(time) * 55}
                  y2="0"
                  stroke="yellow"
                  strokeWidth="2"
                  strokeDasharray="4,2"
                />
                <circle cx={Math.cos(time) * 55} cy="0" r="5" fill="yellow" />
              </svg>
            </div>
            
            <div>
              <svg width="200" height="150" viewBox="0 0 200 150">
                <line x1="0" y1="75" x2="200" y2="75" stroke="gray" strokeWidth="1" />
                <path
                  d={`M 0 75 ${Array.from({ length: 50 }, (_, i) => {
                    const x = i * 4;
                    const y = 75 - Math.cos(time - i * 0.15) * 50;
                    return `L ${x} ${y}`;
                  }).join(' ')}`}
                  fill="none"
                  stroke="yellow"
                  strokeWidth="2"
                />
              </svg>
              <p className="text-center text-xs text-gray-400 mt-2">x(t) = A·cos(ωt)</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
