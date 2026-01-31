import React, { useState, useEffect } from 'react';
import { Zap, Radio, TrendingUp, Layers } from 'lucide-react';

/**
 * Section 6.7: Hidden Communication in Filaments Before Nerve Spike
 * Section 6.7.1: Triplet of Triplet Resonance Band in Axon
 * Section 6.7.2: Scale-Free Triplet Band Across Scales
 */
export const HiddenCommunicationPanel: React.FC = () => {
  const [time, setTime] = useState(0);
  const [axonSignal, setAxonSignal] = useState<number[]>([]);
  const [preSpike, setPreSpike] = useState(false);
  
  // Triplet of triplet across three scales - defined outside to avoid deps warning
  const scales = React.useMemo(() => [
    { name: 'Tubulin', scale: '25 nm', freq: 'GHz' },
    { name: 'Microtubule', scale: '2 μm', freq: 'MHz' },
    { name: 'Neuron', scale: '100 μm', freq: 'kHz' }
  ], []);
  
  const [scaleResonances, setScaleResonances] = useState<number[][]>([[], [], []]);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(t => t + 0.1);
      
      // Generate axon hidden signal (before spike)
      const hiddenSignal = Math.sin(time * 0.5) * 30 + 50;
      const spikeThreshold = 75;
      
      setAxonSignal(prev => {
        const newSignal = [...prev, hiddenSignal];
        return newSignal.slice(-100);
      });
      
      setPreSpike(hiddenSignal > spikeThreshold - 10 && hiddenSignal < spikeThreshold);
      
      // Generate triplet of triplet resonances at each scale
      setScaleResonances(scales.map((_, scaleIdx) => {
        return Array.from({ length: 9 }, (_, i) => {
          const tripletGroup = Math.floor(i / 3);
          const localIdx = i % 3;
          const scaleFreq = 1 / Math.pow(10, scaleIdx);
          
          return Math.abs(
            Math.sin(time * scaleFreq + tripletGroup * Math.PI / 3) * 
            Math.cos(time * scaleFreq * (localIdx + 1) * 2) * 50 + 50
          );
        });
      }));
    }, 50);
    
    return () => clearInterval(interval);
  }, [time, scales]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/30 via-pink-900/30 to-blue-900/30 border border-purple-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Zap className="text-purple-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">6.7 Hidden Communication in Filaments</h2>
            <p className="text-gray-300 text-sm">Pre-synaptic information transfer before nerve spikes</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Neural computation doesn't begin with action potentials—it starts much earlier in molecular 
          filaments. Microtubules and neurofilaments carry signals milliseconds before nerve spikes, 
          implementing a hidden communication layer that processes and prepares information. This 
          pre-conscious activity operates through triplet-of-triplet resonance patterns spanning from 
          molecular to cellular scales.
        </p>
      </div>

      {/* Section 6.7: Hidden Pre-Spike Communication */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Radio className="mr-2 text-pink-400" size={24} />
          Hidden Filament Signals Before Action Potential
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Signal Timeline */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-pink-400 font-semibold mb-4 text-sm">Temporal Signal Sequence</h4>
            <div className="relative h-64 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 400 200" className="w-full h-full">
                {/* Time axis */}
                <line x1="40" y1="180" x2="360" y2="180" stroke="#666" strokeWidth="2" />
                
                {/* Hidden signal (purple) */}
                {axonSignal.length > 1 && (
                  <path
                    d={`M ${40} ${180 - axonSignal[0] * 1.5} ${axonSignal.slice(1).map((val, i) => {
                      const x = 40 + ((i + 1) * 320) / (axonSignal.length - 1);
                      const y = 180 - val * 1.5;
                      return `L ${x} ${y}`;
                    }).join(' ')}`}
                    fill="none"
                    stroke="#d946ef"
                    strokeWidth="3"
                    opacity="0.8"
                  />
                )}
                
                {/* Spike threshold */}
                <line x1="40" y1="65" x2="360" y2="65" stroke="#ef4444" strokeWidth="2" strokeDasharray="5,5" opacity="0.7" />
                <text x="365" y="70" fill="#ef4444" fontSize="10">Spike</text>
                
                {/* Pre-spike warning zone */}
                <rect x="40" y="50" width="320" height="30" fill="#ef4444" opacity="0.1" />
                
                {/* Current position indicator */}
                <circle
                  cx="360"
                  cy={180 - (axonSignal[axonSignal.length - 1] || 0) * 1.5}
                  r="5"
                  fill={preSpike ? '#ef4444' : '#d946ef'}
                  className={preSpike ? 'animate-pulse' : ''}
                />
                
                {/* Labels */}
                <text x="200" y="195" textAnchor="middle" fill="#888" fontSize="10">Time (ms)</text>
                <text x="25" y="180" textAnchor="middle" fill="#888" fontSize="10">0</text>
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              Hidden signal in filaments precedes action potential spike
            </p>
          </div>

          {/* Signal Properties */}
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Signal Characteristics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Current Level:</span>
                  <span className={`font-mono ${preSpike ? 'text-red-400 font-bold' : 'text-pink-400'}`}>
                    {(axonSignal[axonSignal.length - 1] || 0).toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Pre-Spike:</span>
                  <span className={`font-mono ${preSpike ? 'text-red-400' : 'text-gray-500'}`}>
                    {preSpike ? 'ACTIVE' : 'Normal'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Lead Time:</span>
                  <span className="text-pink-400 font-mono">2-5 ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Propagation:</span>
                  <span className="text-pink-400 font-mono">10⁴ m/s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Mechanism:</span>
                  <span className="text-pink-400 font-mono">Coherent states</span>
                </div>
              </div>
            </div>

            <div className="bg-pink-900/20 border border-pink-700 rounded-lg p-4">
              <h4 className="text-pink-400 font-semibold mb-2 text-sm">Hidden Layer Discovery</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Conventional neuroscience focuses on action potentials (nerve spikes), but these are 
                just the output. The real computation happens earlier in cytoskeletal filaments. 
                Microtubules process information and make "decisions" 2-5 milliseconds before neurons 
                fire. This hidden layer operates via quantum coherent states maintained by PPM structures.
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Computational Significance</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Hidden filament communication provides 100× more computational capacity than action 
                potentials alone. It enables predictive processing, error correction, and pattern 
                completion before conscious awareness. This may explain phenomena like intuition, 
                priming effects, and subliminal perception.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Section 6.7.1: Triplet of Triplet in Axon */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <TrendingUp className="mr-2 text-cyan-400" size={24} />
          6.7.1 Triplet of Triplet Resonance in Axon
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          {['Tubulin', 'Microtubule', 'Axon'].map((level, levelIdx) => (
            <div key={level} className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-3 text-sm">{level} Level</h4>
              
              {/* Frequency bands */}
              <div className="space-y-2 mb-4">
                {Array.from({ length: 3 }).map((_, groupIdx) => (
                  <div key={groupIdx} className="space-y-1">
                    <div className="text-xs text-gray-400">Triplet {groupIdx + 1}</div>
                    {Array.from({ length: 3 }).map((_, localIdx) => {
                      const idx = groupIdx * 3 + localIdx;
                      const intensity = scaleResonances[levelIdx]?.[idx] || 0;
                      
                      return (
                        <div key={localIdx} className="flex items-center space-x-2">
                          <div className="w-full bg-gray-700 rounded-full h-1.5 overflow-hidden">
                            <div 
                              className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all duration-100"
                              style={{ width: `${intensity}%` }}
                            />
                          </div>
                          <span className="text-xs text-cyan-400 font-mono w-8">
                            {idx + 1}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
              
              <div className="text-xs text-gray-400 border-t border-gray-700 pt-2">
                Scale: {scales[levelIdx].scale}<br/>
                Freq: {scales[levelIdx].freq}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 bg-purple-900/20 border border-purple-700 rounded-lg p-4">
          <p className="text-gray-300 text-xs leading-relaxed">
            <strong className="text-purple-400">Axonal Resonance:</strong> The axon contains ~1000 
            microtubules, each with 13 protofilaments, each with ~1000 tubulin dimers. This creates 
            a three-level hierarchy where triplet-of-triplet patterns at each scale couple to form 
            a unified resonance structure. Information encoded at the tubulin level (GHz) propagates 
            coherently through microtubules (MHz) to axonal transmission (kHz)—all maintaining the 
            3×3 fractal pattern.
          </p>
        </div>
      </div>

      {/* Section 6.7.2: Scale-Free Triplet Bands */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Layers className="mr-2 text-cyan-400" size={24} />
          6.7.2 Scale-Free Triplet of Triplet Band
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Scale Hierarchy */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-4 text-sm">Fractal Scale Hierarchy</h4>
            <div className="space-y-4">
              {scales.map((scale, idx) => (
                <div key={scale.name} className="relative">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-purple-400 font-semibold text-sm">{scale.name}</span>
                    <span className="text-xs text-gray-400">{scale.scale}</span>
                  </div>
                  
                  {/* Visual scale representation */}
                  <div className="relative h-16 bg-black/30 rounded overflow-hidden">
                    <svg viewBox="0 0 300 60" className="w-full h-full">
                      {/* Triplet groups */}
                      {Array.from({ length: 3 }).map((_, g) => (
                        <g key={g}>
                          {/* Group boundary */}
                          <rect
                            x={g * 100 + 5}
                            y="5"
                            width="90"
                            height="50"
                            fill="none"
                            stroke="#374151"
                            strokeWidth="1"
                            rx="4"
                          />
                          
                          {/* Three elements per group */}
                          {Array.from({ length: 3 }).map((_, i) => {
                            const resonance = scaleResonances[idx]?.[g * 3 + i] || 0;
                            return (
                              <rect
                                key={i}
                                x={g * 100 + 10 + i * 27}
                                y="10"
                                width="22"
                                height="40"
                                fill={`hsl(${280 + idx * 30}, 70%, ${30 + resonance * 0.4}%)`}
                                rx="2"
                              />
                            );
                          })}
                        </g>
                      ))}
                    </svg>
                  </div>
                  
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Frequency: {scale.freq}</span>
                    <span>9 bands = 3×3</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Scale-Free Properties */}
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-3 text-sm">Scale Invariance</h4>
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Pattern Type:</span>
                  <span className="text-cyan-400 font-mono">Fractal 3×3</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Scale Ratio:</span>
                  <span className="text-cyan-400 font-mono">~1000×</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Frequency Ratio:</span>
                  <span className="text-cyan-400 font-mono">~1000×</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Phase Coupling:</span>
                  <span className="text-cyan-400 font-mono">Coherent</span>
                </div>
              </div>
            </div>

            <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Scale-Free Principle</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                The triplet-of-triplet pattern repeats identically across spatial scales differing by 
                1000×. A computation at the tubulin level (nanoseconds, nanometers) mirrors identical 
                logic at the microtubule level (microseconds, micrometers) and again at the neuronal 
                level (milliseconds, millimeters). This scale invariance enables hierarchical information 
                processing where results at one scale immediately propagate to all other scales through 
                phase coherence.
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">PPM Governs Coupling</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Phase Prime Metrics determine the coupling strength between scales. Prime number 
                frequency ratios create constructive interference across hierarchies, while non-prime 
                ratios destructively interfere. This natural filtering ensures only coherent, 
                PPM-consistent patterns propagate through the scale hierarchy, providing robustness 
                against noise and decoherence.
              </p>
            </div>

            <div className="bg-pink-900/20 border border-pink-700 rounded-lg p-4">
              <h4 className="text-pink-400 font-semibold mb-2 text-sm">Consciousness Implication</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                The scale-free architecture may explain unified conscious experience. A single thought 
                simultaneously exists at molecular, cellular, and systems levels—not as separate 
                representations but as the same information expressed at different scales through PPM 
                phase relationships. This provides a physical mechanism for the unity of consciousness.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-purple-900/20 to-pink-900/20 border border-purple-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Hidden Communication: Key Insights</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <h4 className="text-purple-400 font-semibold text-sm">Discoveries:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">✓</span>
                <span>Filament signals precede nerve spikes by 2-5 milliseconds</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">✓</span>
                <span>Triplet-of-triplet resonance bands span tubulin to neuron scales</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">✓</span>
                <span>Scale-free architecture maintains 3×3 pattern across 1000× size ranges</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-400 mr-2">✓</span>
                <span>PPM phase relationships couple hierarchical levels coherently</span>
              </li>
            </ul>
          </div>

          <div className="space-y-3">
            <h4 className="text-pink-400 font-semibold text-sm">Implications:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-pink-400 mr-2">→</span>
                <span>Conscious processing begins before neural spikes (pre-conscious layer)</span>
              </li>
              <li className="flex items-start">
                <span className="text-pink-400 mr-2">→</span>
                <span>100× more computational capacity than action potential models suggest</span>
              </li>
              <li className="flex items-start">
                <span className="text-pink-400 mr-2">→</span>
                <span>Unified experience emerges from scale-invariant information structure</span>
              </li>
              <li className="flex items-start">
                <span className="text-pink-400 mr-2">→</span>
                <span>Hidden layer explains intuition, priming, and subliminal perception</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
