import React from 'react';
import { Target, Info } from 'lucide-react';

// 4.4 Action is not limited to planks constant, every imaginary world has one
// 4.5 A table of fundamental constants in nature
export const FundamentalConstants: React.FC = () => {
  const constants = [
    {
      name: 'Planck Constant (ℏ)',
      value: '1.054 × 10⁻³⁴ J·s',
      world: 'Quantum World',
      significance: 'Quantum of action',
      color: 'blue'
    },
    {
      name: 'Speed of Light (c)',
      value: '2.998 × 10⁸ m/s',
      world: 'Spacetime',
      significance: 'Maximum causality speed',
      color: 'cyan'
    },
    {
      name: 'Gravitational Constant (G)',
      value: '6.674 × 10⁻¹¹ m³/kg·s²',
      world: 'Classical World',
      significance: 'Strength of gravity',
      color: 'green'
    },
    {
      name: 'Boltzmann Constant (k_B)',
      value: '1.381 × 10⁻²³ J/K',
      world: 'Thermodynamic',
      significance: 'Energy-temperature scale',
      color: 'orange'
    },
    {
      name: 'Fine Structure (α)',
      value: '1/137.036',
      world: 'Electromagnetic',
      significance: 'EM coupling strength',
      color: 'purple'
    },
    {
      name: 'Phase Prime Action (ℏ_PPM)',
      value: '∏ᵢ₌₁¹⁵ pᵢ × ℏ',
      world: 'Phase World 1',
      significance: 'First prime world action',
      color: 'yellow'
    },
    {
      name: 'Fractal Action (ℏ_F)',
      value: 'f(n) × ℏ',
      world: 'Fractal Worlds',
      significance: 'Scale-dependent action',
      color: 'pink'
    },
    {
      name: 'Consciousness Constant (ℏ_C)',
      value: '2³×3²×5×7×11 × ℏ',
      world: 'Consciousness',
      significance: 'Awareness quantum',
      color: 'cyan'
    }
  ];

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Target className="text-blue-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.4-4.5 Fundamental Constants</h2>
            <p className="text-gray-300">Action beyond Planck's constant - Every world has its own</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          In fractal mechanics, each imaginary world has its own action constant. Planck's constant ℏ 
          is just one special case for the quantum world. Different phase dimensions have different 
          fundamental scales determined by phase prime patterns.
        </p>
      </div>

      {/* Multiple Action Constants */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">4.4 Multiple Action Constants</h3>
        
        <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4 mb-4">
          <p className="text-gray-300 text-sm leading-relaxed mb-3">
            <strong className="text-blue-400">Classical View:</strong> Action is universal, given by S = ∫L dt
          </p>
          <p className="text-gray-300 text-sm leading-relaxed mb-3">
            <strong className="text-purple-400">Quantum View:</strong> Action is quantized in units of ℏ
          </p>
          <p className="text-gray-300 text-sm leading-relaxed">
            <strong className="text-cyan-400">Fractal View:</strong> Each imaginary world has its own 
            action quantum ℏᵢ, determined by the geometric structure of that phase dimension
          </p>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
            <h4 className="text-blue-400 font-semibold mb-2 text-sm">World 1: Quantum</h4>
            <p className="font-mono text-xs text-gray-300 mb-2">ℏ₁ = ℏ</p>
            <p className="text-xs text-gray-400">
              Standard quantum action, governs atomic/subatomic scales
            </p>
          </div>
          
          <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-2 text-sm">World 2: Prime Phase</h4>
            <p className="font-mono text-xs text-gray-300 mb-2">ℏ₂ = 2 × ℏ</p>
            <p className="text-xs text-gray-400">
              First prime phase world, doubled action quantum
            </p>
          </div>
          
          <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-2 text-sm">World 3: Consciousness</h4>
            <p className="font-mono text-xs text-gray-300 mb-2">ℏ₃ = 2³×3²×5 × ℏ</p>
            <p className="text-xs text-gray-400">
              Consciousness action, product of first 3 primes
            </p>
          </div>
        </div>

        <div className="mt-4">
          <svg width="600" height="150" viewBox="0 0 600 150" className="w-full">
            <text x="300" y="20" textAnchor="middle" fill="cyan" fontSize="14" fontWeight="bold">
              Action Constants Across Imaginary Worlds
            </text>
            
            {/* Timeline of worlds */}
            <line x1="50" y1="75" x2="550" y2="75" stroke="gray" strokeWidth="2" />
            
            {[
              { x: 100, label: 'ℏ₁', color: 'blue', scale: 1 },
              { x: 200, label: 'ℏ₂', color: 'purple', scale: 2 },
              { x: 300, label: 'ℏ₃', color: 'cyan', scale: 4 },
              { x: 400, label: 'ℏ₄', color: 'green', scale: 8 },
              { x: 500, label: 'ℏ₁₁', color: 'yellow', scale: 16 }
            ].map((world, i) => (
              <g key={i}>
                <circle cx={world.x} cy="75" r={world.scale + 4} fill={world.color} fillOpacity="0.5" />
                <circle cx={world.x} cy="75" r="3" fill={world.color} />
                <text x={world.x} y="110" textAnchor="middle" fill={world.color} fontSize="12">
                  {world.label}
                </text>
                <line x1={world.x} y1="75" x2={world.x} y2={65 - world.scale * 2} stroke={world.color} strokeWidth="2" />
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Constants Table */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">4.5 Table of Fundamental Constants</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b-2 border-gray-700">
                <th className="text-left p-3 text-gray-400 font-semibold">Constant</th>
                <th className="text-left p-3 text-gray-400 font-semibold">Value</th>
                <th className="text-left p-3 text-gray-400 font-semibold">World</th>
                <th className="text-left p-3 text-gray-400 font-semibold">Significance</th>
              </tr>
            </thead>
            <tbody>
              {constants.map((constant, idx) => (
                <tr key={idx} className="border-b border-gray-800 hover:bg-gray-800/30">
                  <td className="p-3">
                    <span className={`text-${constant.color}-400 font-semibold`}>
                      {constant.name}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="font-mono text-gray-300 text-xs">
                      {constant.value}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="text-gray-300">{constant.world}</span>
                  </td>
                  <td className="p-3">
                    <span className="text-gray-400 text-xs">{constant.significance}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Relationships */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Relationships Between Constants</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Natural Units</h4>
            <p className="text-gray-300 text-xs mb-3">
              Setting c = ℏ = G = k_B = 1 reveals the geometric structure:
            </p>
            <div className="bg-cyan-900/20 p-3 rounded space-y-2">
              <p className="font-mono text-xs text-gray-300">Planck Length: ℓ_P = √(ℏG/c³)</p>
              <p className="font-mono text-xs text-gray-300">Planck Time: t_P = √(ℏG/c⁵)</p>
              <p className="font-mono text-xs text-gray-300">Planck Mass: m_P = √(ℏc/G)</p>
            </div>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-purple-400 font-semibold mb-3 text-sm">Phase Prime Scaling</h4>
            <p className="text-gray-300 text-xs mb-3">
              Fractal constants scale by prime factors:
            </p>
            <div className="bg-purple-900/20 p-3 rounded space-y-2">
              <p className="font-mono text-xs text-gray-300">ℏ_PPM = (2×3×5×7×11×...×47) × ℏ</p>
              <p className="font-mono text-xs text-gray-300">ℏ_F(n) = OF(n) × ℏ</p>
              <p className="text-xs text-gray-400">OF = Ordered Factor metric</p>
            </div>
          </div>
        </div>
      </div>

      {/* Dimensional Analysis */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Dimensional Analysis</h3>
        
        <div className="space-y-3">
          <div className="bg-blue-900/20 border-l-4 border-blue-700 pl-4 py-3">
            <h4 className="text-blue-400 font-semibold text-sm mb-2">Action [M L² T⁻¹]</h4>
            <p className="text-gray-300 text-xs">
              Fundamental unit linking energy and time. All action constants have these dimensions.
            </p>
          </div>
          
          <div className="bg-purple-900/20 border-l-4 border-purple-700 pl-4 py-3">
            <h4 className="text-purple-400 font-semibold text-sm mb-2">Angular Momentum [M L² T⁻¹]</h4>
            <p className="text-gray-300 text-xs">
              ℏ is the quantum of angular momentum. Fractal worlds have quantized angular momentum 
              at different scales.
            </p>
          </div>
          
          <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-4 py-3">
            <h4 className="text-cyan-400 font-semibold text-sm mb-2">Consciousness Quantum [M L² T⁻¹]</h4>
            <p className="text-gray-300 text-xs">
              ℏ_C represents the minimum "quantum of awareness" - the smallest unit of conscious 
              experience, scaled by phase prime factors.
            </p>
          </div>
        </div>
      </div>

      <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-blue-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-blue-400">Paradigm Shift:</strong> The existence of multiple action 
            constants across different imaginary worlds fundamentally changes our understanding of physics. 
            It suggests that "physical laws" are actually world-specific projections of deeper geometric 
            principles operating in 11D phase space. What we call "fundamental constants" are actually 
            dimensional resonances of the underlying prime-based structure.
          </div>
        </div>
      </div>
    </div>
  );
};
