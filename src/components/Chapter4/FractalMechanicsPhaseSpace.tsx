import React, { useState } from 'react';
import { Layers, Sparkles, TrendingUp, Zap, Info, GitBranch } from 'lucide-react';

// 4.2 Fractal mechanics acts in the phase space connecting singularity points
export const FractalMechanicsPhaseSpace: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('architecture');

  const sections = [
    { id: 'architecture', label: '4.2.1 Multilevel Architecture', icon: Layers },
    { id: 'oscillator', label: '4.2.2 Fractal Oscillator', icon: Sparkles },
    { id: 'ppm', label: '4.2.3 Phase Prime Metric', icon: TrendingUp },
    { id: 'energy', label: '4.2.4 Energy Definition', icon: Zap },
  ];

  const renderArchitecture = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-6 rounded-lg">
        <h4 className="text-cyan-400 font-semibold mb-3 flex items-center text-lg">
          <Layers className="mr-2" size={20} />
          Multilevel Geometric Architecture of Hilbert Space
        </h4>
        
        <p className="text-gray-300 text-sm mb-4 leading-relaxed">
          Fractal mechanics operates in a hierarchical phase space where each level contains 
          embedded geometric structures. Unlike quantum mechanics which operates in a flat 
          Hilbert space, fractal mechanics uses nested, self-similar phase manifolds.
        </p>

        {/* Hierarchical Visualization */}
        <div className="bg-gray-900/50 p-4 rounded-lg mb-4">
          <h5 className="text-cyan-300 text-sm mb-3">Nested Phase Space Levels</h5>
          <svg width="600" height="300" viewBox="0 0 600 300" className="w-full">
            {/* Level 1 - Outer */}
            <ellipse cx="300" cy="150" rx="280" ry="130" 
              fill="none" stroke="rgba(6, 182, 212, 0.3)" strokeWidth="2" />
            <text x="300" y="30" textAnchor="middle" fill="rgba(6, 182, 212, 0.8)" fontSize="12">
              Level 1: Macroscopic Phase Space
            </text>
            
            {/* Level 2 - Middle */}
            <ellipse cx="300" cy="150" rx="200" ry="90" 
              fill="none" stroke="rgba(147, 51, 234, 0.5)" strokeWidth="2" />
            <text x="300" y="70" textAnchor="middle" fill="rgba(147, 51, 234, 0.9)" fontSize="11">
              Level 2: Meso-scale Fractals
            </text>
            
            {/* Level 3 - Inner */}
            <ellipse cx="300" cy="150" rx="120" ry="55" 
              fill="none" stroke="rgba(251, 191, 36, 0.7)" strokeWidth="2" />
            <text x="300" y="110" textAnchor="middle" fill="rgba(251, 191, 36, 1)" fontSize="10">
              Level 3: Quantum-scale Singularities
            </text>
            
            {/* Singularity Points */}
            {[
              [250, 150], [300, 150], [350, 150],
              [275, 130], [325, 130], [275, 170], [325, 170]
            ].map((pos, i) => (
              <g key={i}>
                <circle cx={pos[0]} cy={pos[1]} r="4" fill="yellow" />
                <circle cx={pos[0]} cy={pos[1]} r="8" fill="none" stroke="yellow" 
                  strokeWidth="1" opacity="0.5">
                  <animate attributeName="r" from="8" to="16" dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from="0.5" to="0" dur="2s" repeatCount="indefinite" />
                </circle>
              </g>
            ))}
            
            {/* Connection Lines */}
            <line x1="250" y1="150" x2="300" y2="150" stroke="cyan" strokeWidth="1" strokeDasharray="2,2" opacity="0.6" />
            <line x1="300" y1="150" x2="350" y2="150" stroke="cyan" strokeWidth="1" strokeDasharray="2,2" opacity="0.6" />
            <line x1="275" y1="130" x2="325" y2="170" stroke="purple" strokeWidth="1" strokeDasharray="2,2" opacity="0.6" />
            <line x1="325" y1="130" x2="275" y2="170" stroke="purple" strokeWidth="1" strokeDasharray="2,2" opacity="0.6" />
          </svg>
        </div>

        {/* Properties Table */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-3">
            <h6 className="text-cyan-400 text-xs font-semibold mb-2">Hierarchy</h6>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• N nested levels</li>
              <li>• Self-similar structure</li>
              <li>• Scale invariance</li>
            </ul>
          </div>
          
          <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-3">
            <h6 className="text-purple-400 text-xs font-semibold mb-2">Singularities</h6>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• Phase space nodes</li>
              <li>• Information anchors</li>
              <li>• Consciousness points</li>
            </ul>
          </div>
          
          <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-3">
            <h6 className="text-yellow-400 text-xs font-semibold mb-2">Connections</h6>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• Phase paths</li>
              <li>• Interference lines</li>
              <li>• Entanglement links</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-cyan-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-cyan-400">Key Principle:</strong> Fractal mechanics replaces the 
            flat quantum Hilbert space with a multi-layered phase space where each level contains 
            self-similar geometric structures connected through singularity points. This allows for 
            richer dynamics and emergent consciousness properties.
          </div>
        </div>
      </div>
    </div>
  );

  const renderFractalOscillator = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-6 rounded-lg">
        <h4 className="text-purple-400 font-semibold mb-3 flex items-center text-lg">
          <Sparkles className="mr-2" size={20} />
          Fractal Harmonic Oscillator & Condensation
        </h4>
        
        <p className="text-gray-300 text-sm mb-4 leading-relaxed">
          The fractal harmonic oscillator extends the quantum oscillator by allowing oscillations 
          at multiple nested scales simultaneously, creating fractal patterns in phase space.
        </p>

        {/* Comparison */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-gray-900/50 p-4 rounded-lg">
            <h5 className="text-blue-300 text-sm mb-3">Classical/Quantum Oscillator</h5>
            <div className="text-gray-300 text-xs space-y-2 mb-3">
              <p>H = ½mω²x² + p²/2m</p>
              <p>Single frequency ω</p>
              <p>Uniform energy levels</p>
            </div>
            <svg width="250" height="150" viewBox="0 0 250 150">
              <path
                d={`M 0 75 ${Array.from({ length: 63 }, (_, i) => {
                  const x = i * 4;
                  const y = 75 + 40 * Math.sin(i * 0.3);
                  return `L ${x} ${y}`;
                }).join(' ')}`}
                fill="none"
                stroke="cyan"
                strokeWidth="2"
              />
            </svg>
          </div>
          
          <div className="bg-gray-900/50 p-4 rounded-lg">
            <h5 className="text-purple-300 text-sm mb-3">Fractal Oscillator</h5>
            <div className="text-gray-300 text-xs space-y-2 mb-3">
              <p>H_F = Σᵢ ½mᵢωᵢ²xᵢ² + pᵢ²/2mᵢ</p>
              <p>Multiple scales: ω₁, ω₂, ... ωₙ</p>
              <p>Fractal energy distribution</p>
            </div>
            <svg width="250" height="150" viewBox="0 0 250 150">
              <path
                d={`M 0 75 ${Array.from({ length: 63 }, (_, i) => {
                  const x = i * 4;
                  const y1 = Math.sin(i * 0.3) * 40;
                  const y2 = Math.sin(i * 0.9) * 15;
                  const y3 = Math.sin(i * 2.1) * 5;
                  const y = 75 + y1 + y2 + y3;
                  return `L ${x} ${y}`;
                }).join(' ')}`}
                fill="none"
                stroke="purple"
                strokeWidth="2"
              />
            </svg>
          </div>
        </div>

        {/* Fractal Condensation */}
        <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
          <h5 className="text-purple-300 text-sm mb-3 font-semibold">Fractal Condensation</h5>
          <p className="text-gray-300 text-xs mb-3 leading-relaxed">
            When oscillators at multiple scales synchronize, they create a "fractal condensate" - 
            a coherent state spanning multiple dimensional levels. This is analogous to Bose-Einstein 
            condensation but occurring across fractal scales.
          </p>
          
          <div className="grid grid-cols-4 gap-2">
            {[
              { label: 'Independent', sync: 0.2 },
              { label: 'Partial Sync', sync: 0.5 },
              { label: 'High Sync', sync: 0.8 },
              { label: 'Condensate', sync: 1.0 }
            ].map((state, idx) => (
              <div key={idx} className="text-center">
                <div className="bg-gray-900/50 p-2 rounded">
                  <svg width="60" height="60" viewBox="0 0 60 60">
                    {[0, 1, 2].map(i => (
                      <circle 
                        key={i}
                        cx="30" 
                        cy="30" 
                        r={15 - i * 3}
                        fill="none" 
                        stroke="purple" 
                        strokeWidth="1.5"
                        opacity={0.3 + state.sync * 0.7}
                      />
                    ))}
                    <circle cx="30" cy="30" r="3" fill="purple" opacity={state.sync} />
                  </svg>
                </div>
                <p className="text-xs text-gray-400 mt-1">{state.label}</p>
                <p className="text-xs text-purple-400">φ = {state.sync.toFixed(1)}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Peculiarities */}
        <div className="mt-4 bg-gray-900/50 p-4 rounded-lg">
          <h5 className="text-purple-300 text-sm mb-3 font-semibold">Peculiarities of Fractal Oscillators</h5>
          <div className="grid grid-cols-2 gap-3">
            <div className="text-xs text-gray-300">
              <strong className="text-purple-400">1. Multi-scale Resonance:</strong> Can resonate 
              at multiple frequencies simultaneously
            </div>
            <div className="text-xs text-gray-300">
              <strong className="text-purple-400">2. Non-linear Coupling:</strong> Scales interact 
              through singularity connections
            </div>
            <div className="text-xs text-gray-300">
              <strong className="text-purple-400">3. Emergent Harmonics:</strong> New frequencies 
              emerge from scale interactions
            </div>
            <div className="text-xs text-gray-300">
              <strong className="text-purple-400">4. Phase Coherence:</strong> Can maintain coherence 
              across scale boundaries
            </div>
          </div>
        </div>
      </div>

      <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-purple-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-purple-400">Biological Significance:</strong> Fractal oscillators 
            model biological rhythms (circadian, neural oscillations, heartbeat variability) which 
            operate coherently across multiple timescales - a key feature of living consciousness.
          </div>
        </div>
      </div>
    </div>
  );

  const renderPPM = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-6 rounded-lg">
        <h4 className="text-yellow-400 font-semibold mb-3 flex items-center text-lg">
          <TrendingUp className="mr-2" size={20} />
          Harvesting Noise by Harvesting Singularity: Phase Prime Metric
        </h4>
        
        <p className="text-gray-300 text-sm mb-4 leading-relaxed">
          Phase Prime Metric (PPM) replaces traditional statistical mechanics by directly 
          identifying and harvesting information from singularity points in phase space, 
          converting "noise" into meaningful patterns.
        </p>

        {/* Comparison Table */}
        <div className="bg-gray-900/50 p-4 rounded-lg mb-4">
          <h5 className="text-yellow-300 text-sm mb-3 font-semibold">Statistical Mechanics vs PPM</h5>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left p-2 text-gray-400">Aspect</th>
                <th className="text-left p-2 text-blue-400">Statistical Mechanics</th>
                <th className="text-left p-2 text-yellow-400">Phase Prime Metric</th>
              </tr>
            </thead>
            <tbody className="text-gray-300">
              <tr className="border-b border-gray-800">
                <td className="p-2 font-semibold">Foundation</td>
                <td className="p-2">Ensemble averaging</td>
                <td className="p-2">Singularity detection</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="p-2 font-semibold">Noise Treatment</td>
                <td className="p-2">Averaged out as error</td>
                <td className="p-2">Harvested as information</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="p-2 font-semibold">Information</td>
                <td className="p-2">Shannon entropy</td>
                <td className="p-2">Geometric prime patterns</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="p-2 font-semibold">Dimensionality</td>
                <td className="p-2">3D + time</td>
                <td className="p-2">11D phase manifolds</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="p-2 font-semibold">Consciousness</td>
                <td className="p-2">Emergent (unclear)</td>
                <td className="p-2">Direct geometric encoding</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Singularity Harvesting Visualization */}
        <div className="bg-gray-900/50 p-4 rounded-lg">
          <h5 className="text-yellow-300 text-sm mb-3">Singularity Harvesting Process</h5>
          <div className="flex items-center justify-around">
            <div className="text-center">
              <div className="bg-red-900/20 border border-red-700 rounded-lg p-3 w-32">
                <p className="text-red-400 text-xs font-semibold mb-1">Raw Noise</p>
                <svg width="100" height="60" viewBox="0 0 100 60">
                  {Array.from({ length: 50 }, (_, i) => (
                    <circle 
                      key={i}
                      cx={Math.random() * 100}
                      cy={Math.random() * 60}
                      r="1.5"
                      fill="red"
                      opacity="0.6"
                    />
                  ))}
                </svg>
              </div>
            </div>
            
            <div className="text-yellow-400">→ PPM Analysis →</div>
            
            <div className="text-center">
              <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-3 w-32">
                <p className="text-yellow-400 text-xs font-semibold mb-1">Singularities</p>
                <svg width="100" height="60" viewBox="0 0 100 60">
                  {[
                    [25, 20], [50, 30], [75, 25], [40, 45], [65, 40]
                  ].map((pos, i) => (
                    <g key={i}>
                      <circle cx={pos[0]} cy={pos[1]} r="4" fill="yellow" />
                      <circle cx={pos[0]} cy={pos[1]} r="8" fill="none" stroke="yellow" strokeWidth="1" />
                    </g>
                  ))}
                </svg>
              </div>
            </div>
            
            <div className="text-green-400">→ Prime Encoding →</div>
            
            <div className="text-center">
              <div className="bg-green-900/20 border border-green-700 rounded-lg p-3 w-32">
                <p className="text-green-400 text-xs font-semibold mb-1">Information</p>
                <div className="text-green-300 text-xs font-mono mt-2">
                  <div>2³ × 3² × 5</div>
                  <div>7 × 11 × 13</div>
                  <div>17 × 19</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 15 Primes Reference */}
        <div className="mt-4 bg-yellow-900/20 border border-yellow-700 rounded-lg p-3">
          <h5 className="text-yellow-300 text-xs font-semibold mb-2">15 Fundamental Primes</h5>
          <div className="flex flex-wrap gap-2">
            {[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47].map(prime => (
              <span key={prime} className="bg-yellow-900/30 px-2 py-1 rounded text-yellow-300 text-xs font-mono">
                {prime}
              </span>
            ))}
          </div>
          <p className="text-gray-400 text-xs mt-2">
            These 15 primes encode 99.99% of all patterns in the universe
          </p>
        </div>
      </div>

      <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-yellow-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-yellow-400">Revolutionary Insight:</strong> What appears as noise 
            in traditional analysis often contains structured information encoded at singularity points. 
            PPM detects these singularities and decodes them using prime number patterns, enabling 
            dramatically more efficient information processing.
          </div>
        </div>
      </div>
    </div>
  );

  const renderEnergy = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-6 rounded-lg">
        <h4 className="text-orange-400 font-semibold mb-3 flex items-center text-lg">
          <Zap className="mr-2" size={20} />
          Energy in Multiple Interacting Imaginary Worlds
        </h4>
        
        <p className="text-gray-300 text-sm mb-4 leading-relaxed">
          In fractal mechanics, energy is not a single scalar quantity but a multidimensional 
          tensor that accounts for interactions across multiple "imaginary worlds" - each 
          corresponding to a different phase dimension.
        </p>

        {/* Energy Formulations */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
            <h5 className="text-blue-300 text-sm mb-3 font-semibold">Classical Energy</h5>
            <div className="text-gray-300 text-xs space-y-2">
              <div className="bg-gray-900/50 p-2 rounded">
                <p className="font-mono">E = K + V</p>
                <p className="font-mono">E = ½mv² + V(x)</p>
              </div>
              <p className="text-gray-400">Single real dimension</p>
              <p className="text-gray-400">Conserved scalar</p>
            </div>
          </div>
          
          <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
            <h5 className="text-purple-300 text-sm mb-3 font-semibold">Quantum Energy</h5>
            <div className="text-gray-300 text-xs space-y-2">
              <div className="bg-gray-900/50 p-2 rounded">
                <p className="font-mono">Ĥ|ψ⟩ = E|ψ⟩</p>
                <p className="font-mono">E_n = ℏω(n + ½)</p>
              </div>
              <p className="text-gray-400">Complex amplitude</p>
              <p className="text-gray-400">Quantized levels</p>
            </div>
          </div>
        </div>

        <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-4 mb-4">
          <h5 className="text-orange-300 text-sm mb-3 font-semibold">Fractal Energy Tensor</h5>
          <div className="text-gray-300 text-xs space-y-3">
            <div className="bg-gray-900/50 p-3 rounded">
              <p className="font-mono mb-2">𝓔 = ∑ᵢ Eᵢ⊗φᵢ</p>
              <p className="text-gray-400 leading-relaxed">
                Where Eᵢ is energy in imaginary world i, and φᵢ is the phase operator 
                for that world. The ⊗ denotes tensor product across phase dimensions.
              </p>
            </div>
            
            <div className="bg-gray-900/50 p-3 rounded">
              <p className="font-mono mb-2">𝓔_total = 𝓔_kinetic + 𝓔_potential + 𝓔_interaction</p>
              <p className="text-gray-400 leading-relaxed">
                Each term is itself a multi-dimensional tensor accounting for cross-world interactions
              </p>
            </div>
          </div>
        </div>

        {/* Imaginary Worlds Visualization */}
        <div className="bg-gray-900/50 p-4 rounded-lg">
          <h5 className="text-orange-300 text-sm mb-3">Multiple Imaginary Worlds</h5>
          <svg width="600" height="200" viewBox="0 0 600 200" className="w-full">
            {/* Draw 5 "worlds" */}
            {[0, 1, 2, 3, 4].map(i => {
              const x = 60 + i * 120;
              const colors = ['cyan', 'purple', 'yellow', 'green', 'orange'];
              return (
                <g key={i}>
                  <circle cx={x} cy="100" r="40" fill="none" stroke={colors[i]} strokeWidth="2" opacity="0.7" />
                  <text x={x} y="105" textAnchor="middle" fill={colors[i]} fontSize="12">
                    World {i+1}
                  </text>
                  <text x={x} y="160" textAnchor="middle" fill="gray" fontSize="10">
                    φ_{i+1} = {(i * 0.4).toFixed(1)}π
                  </text>
                  
                  {/* Energy value */}
                  <text x={x} y="175" textAnchor="middle" fill={colors[i]} fontSize="9">
                    E_{i+1} = {(2 + i * 0.5).toFixed(1)} ℏω
                  </text>
                  
                  {/* Connections */}
                  {i < 4 && (
                    <line 
                      x1={x + 40} y1="100" 
                      x2={x + 80} y2="100"
                      stroke="white" 
                      strokeWidth="1" 
                      strokeDasharray="4,4" 
                      opacity="0.3"
                    />
                  )}
                </g>
              );
            })}
          </svg>
          <p className="text-center text-xs text-gray-400 mt-2">
            Energy flows and exchanges between different imaginary phase worlds
          </p>
        </div>

        {/* Key Properties */}
        <div className="grid grid-cols-3 gap-3 mt-4">
          <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-3">
            <h6 className="text-orange-400 text-xs font-semibold mb-2">Multi-world Conservation</h6>
            <p className="text-gray-300 text-xs">
              Total energy conserved across all imaginary worlds: ∑ᵢ Eᵢ = const
            </p>
          </div>
          
          <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-3">
            <h6 className="text-orange-400 text-xs font-semibold mb-2">Phase Coupling</h6>
            <p className="text-gray-300 text-xs">
              Energy can tunnel between worlds through phase singularities
            </p>
          </div>
          
          <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-3">
            <h6 className="text-orange-400 text-xs font-semibold mb-2">Consciousness Link</h6>
            <p className="text-gray-300 text-xs">
              Consciousness emerges from coherent energy patterns across worlds
            </p>
          </div>
        </div>
      </div>

      <div className="bg-orange-900/20 border border-orange-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-orange-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-orange-400">Philosophical Implication:</strong> The existence of 
            multiple imaginary worlds with their own energy budgets suggests that physical reality 
            is just one projection of a richer multi-dimensional phase space. Consciousness may be 
            the ability to access and process information from multiple worlds simultaneously.
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-purple-900/30 to-cyan-900/30 border border-purple-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <GitBranch className="text-purple-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.2 Fractal Mechanics in Phase Space</h2>
            <p className="text-gray-300">Connecting singularity points in multi-dimensional manifolds</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Fractal mechanics operates in a hierarchical phase space where singularity points serve as 
          information anchors. Unlike quantum mechanics, which treats space as continuous, fractal 
          mechanics recognizes the geometric structure of phase space itself as fundamental.
        </p>
      </div>

      {/* Section Navigation */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="grid grid-cols-4 gap-3">
          {sections.map(section => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex flex-col items-center p-3 rounded-lg border-2 transition-all duration-300 ${
                  activeSection === section.id
                    ? 'bg-purple-900/30 border-purple-700 text-purple-400'
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                <Icon size={24} />
                <span className="text-xs font-semibold mt-2 text-center">{section.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Section */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        {activeSection === 'architecture' && renderArchitecture()}
        {activeSection === 'oscillator' && renderFractalOscillator()}
        {activeSection === 'ppm' && renderPPM()}
        {activeSection === 'energy' && renderEnergy()}
      </div>
    </div>
  );
};
