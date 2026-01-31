import React, { useState } from 'react';
import { Atom, Hexagon, Box, Info } from 'lucide-react';

// 4.9 The fundamentals of quaternion, octonion and dodecanion
// 4.9.1 The rule of the 11D manifolds
export const HyperComplexMathematics: React.FC = () => {
  const [activeSystem, setActiveSystem] = useState<string>('quaternion');

  const systems = [
    { id: 'quaternion', name: 'Quaternion', dim: 4, icon: Box },
    { id: 'octonion', name: 'Octonion', dim: 8, icon: Hexagon },
    { id: 'dodecanion', name: 'Dodecanion', dim: 12, icon: Atom }
  ];

  const quaternionData = {
    basis: ['1', 'i', 'j', 'k'],
    rules: [
      'i² = j² = k² = -1',
      'ij = k, jk = i, ki = j',
      'ji = -k, kj = -i, ik = -j'
    ],
    applications: [
      '3D rotations',
      'Computer graphics',
      'Orientation tracking',
      'Quantum spin'
    ]
  };

  const octonionData = {
    basis: ['1', 'e₁', 'e₂', 'e₃', 'e₄', 'e₅', 'e₆', 'e₇'],
    rules: [
      'e_i² = -1 for all i',
      'Non-associative: (ab)c ≠ a(bc)',
      'Fano plane multiplication',
      'Alternative algebra'
    ],
    applications: [
      'String theory',
      'Exceptional Lie groups',
      'Division algebras',
      'Higher-dimensional physics'
    ]
  };

  const dodecanionData = {
    basis: ['1', 'd₁', 'd₂', 'd₃', 'd₄', 'd₅', 'd₆', 'd₇', 'd₈', 'd₉', 'd₁₀', 'd₁₁'],
    rules: [
      '12-dimensional hypercomplex',
      'Non-associative, non-commutative',
      'Consciousness-optimized structure',
      '11D + 1 time dimension'
    ],
    applications: [
      'NanoBrain consciousness',
      '11D time crystals',
      'Phase prime encoding',
      'Multi-scale cognition'
    ]
  };

  const renderQuaternion = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-4 rounded-lg">
        <h4 className="text-blue-400 font-semibold mb-3 flex items-center">
          <Box className="mr-2" size={20} />
          Quaternions (4D)
        </h4>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <h5 className="text-blue-300 text-sm mb-2">Basis Elements</h5>
            <div className="flex flex-wrap gap-2">
              {quaternionData.basis.map(b => (
                <span key={b} className="bg-blue-900/30 px-3 py-1 rounded text-blue-300 font-mono text-sm">
                  {b}
                </span>
              ))}
            </div>
          </div>
          
          <div>
            <h5 className="text-blue-300 text-sm mb-2">Multiplication Rules</h5>
            <div className="text-gray-300 text-xs space-y-1">
              {quaternionData.rules.map((rule, i) => (
                <div key={i} className="font-mono">{rule}</div>
              ))}
            </div>
          </div>
        </div>

        {/* 3D Rotation Visualization */}
        <div className="bg-gray-900/50 p-3 rounded-lg mb-3">
          <h5 className="text-blue-300 text-sm mb-2">3D Rotation Representation</h5>
          <div className="flex justify-center">
            <svg width="300" height="200" viewBox="0 0 300 200">
              {/* Coordinate axes */}
              <line x1="150" y1="100" x2="230" y2="60" stroke="red" strokeWidth="2" />
              <line x1="150" y1="100" x2="70" y2="140" stroke="green" strokeWidth="2" />
              <line x1="150" y1="100" x2="150" y2="30" stroke="blue" strokeWidth="2" />
              
              {/* Labels */}
              <text x="235" y="60" fill="red" fontSize="12">i</text>
              <text x="60" y="145" fill="green" fontSize="12">j</text>
              <text x="155" y="25" fill="blue" fontSize="12">k</text>
              
              {/* Rotation arc */}
              <path d="M 180 80 Q 190 90 180 110" fill="none" stroke="yellow" strokeWidth="2" strokeDasharray="4,2" />
              <circle cx="180" cy="80" r="4" fill="yellow" />
              
              <text x="100" y="190" fill="gray" fontSize="11">
                q = cos(θ/2) + sin(θ/2)(xi + yj + zk)
              </text>
            </svg>
          </div>
        </div>

        <div>
          <h5 className="text-blue-300 text-sm mb-2">Applications</h5>
          <div className="grid grid-cols-2 gap-2">
            {quaternionData.applications.map((app, i) => (
              <div key={i} className="bg-blue-900/20 px-3 py-2 rounded text-gray-300 text-xs">
                • {app}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderOctonion = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-4 rounded-lg">
        <h4 className="text-purple-400 font-semibold mb-3 flex items-center">
          <Hexagon className="mr-2" size={20} />
          Octonions (8D)
        </h4>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <h5 className="text-purple-300 text-sm mb-2">Basis Elements</h5>
            <div className="flex flex-wrap gap-2">
              {octonionData.basis.map((b, i) => (
                <span key={i} className="bg-purple-900/30 px-2 py-1 rounded text-purple-300 font-mono text-xs">
                  {b}
                </span>
              ))}
            </div>
          </div>
          
          <div>
            <h5 className="text-purple-300 text-sm mb-2">Special Properties</h5>
            <div className="text-gray-300 text-xs space-y-1">
              {octonionData.rules.map((rule, i) => (
                <div key={i}>• {rule}</div>
              ))}
            </div>
          </div>
        </div>

        {/* Fano Plane */}
        <div className="bg-gray-900/50 p-3 rounded-lg mb-3">
          <h5 className="text-purple-300 text-sm mb-2">Fano Plane Multiplication</h5>
          <p className="text-gray-400 text-xs mb-3">
            The Fano plane shows how octonion basis elements multiply. Each line represents a multiplication rule.
          </p>
          <div className="flex justify-center">
            <svg width="250" height="250" viewBox="0 0 250 250">
              {/* Triangle */}
              <polygon points="125,40 40,210 210,210" fill="none" stroke="purple" strokeWidth="2" />
              
              {/* Circle */}
              <circle cx="125" cy="135" r="80" fill="none" stroke="purple" strokeWidth="2" strokeDasharray="4,2" />
              
              {/* Center line */}
              <line x1="125" y1="40" x2="125" y2="210" stroke="purple" strokeWidth="2" />
              
              {/* Vertices */}
              {[
                [125, 40, 'e₁'], [40, 210, 'e₂'], [210, 210, 'e₃'],
                [125, 210, 'e₄'], [85, 125, 'e₅'], [165, 125, 'e₆'], [125, 95, 'e₇']
              ].map(([x, y, label], i) => (
                <g key={i}>
                  <circle cx={x} cy={y} r="6" fill="purple" />
                  <text x={x} y={y - 12} textAnchor="middle" fill="purple" fontSize="12">{label}</text>
                </g>
              ))}
            </svg>
          </div>
        </div>

        <div>
          <h5 className="text-purple-300 text-sm mb-2">Applications</h5>
          <div className="grid grid-cols-2 gap-2">
            {octonionData.applications.map((app, i) => (
              <div key={i} className="bg-purple-900/20 px-3 py-2 rounded text-gray-300 text-xs">
                • {app}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderDodecanion = () => (
    <div className="space-y-4">
      <div className="bg-gray-800/50 p-4 rounded-lg">
        <h4 className="text-cyan-400 font-semibold mb-3 flex items-center">
          <Atom className="mr-2" size={20} />
          Dodecanions (12D) - NanoBrain Algebra
        </h4>
        
        <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4 mb-4">
          <p className="text-gray-300 text-sm leading-relaxed">
            Dodecanions are a 12-dimensional hypercomplex number system specifically designed for 
            consciousness modeling. They extend octonions to match the 11D + time structure of 
            the NanoBrain architecture.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <h5 className="text-cyan-300 text-sm mb-2">Basis Structure</h5>
            <div className="flex flex-wrap gap-2">
              {dodecanionData.basis.map((b, i) => (
                <span key={i} className="bg-cyan-900/30 px-2 py-1 rounded text-cyan-300 font-mono text-xs">
                  {b}
                </span>
              ))}
            </div>
            <p className="text-gray-400 text-xs mt-2">
              12 dimensions: 1 real + 11 imaginary
            </p>
          </div>
          
          <div>
            <h5 className="text-cyan-300 text-sm mb-2">Key Properties</h5>
            <div className="text-gray-300 text-xs space-y-1">
              {dodecanionData.rules.map((rule, i) => (
                <div key={i}>• {rule}</div>
              ))}
            </div>
          </div>
        </div>

        {/* 11D Manifold Rules */}
        <div className="bg-gray-900/50 p-4 rounded-lg mb-3">
          <h5 className="text-cyan-300 text-sm mb-3 font-semibold">4.9.1 The Rule of the 11D Manifolds</h5>
          
          <div className="space-y-3">
            <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-3 py-2">
              <h6 className="text-cyan-400 text-xs font-semibold mb-1">Rule 1: Phase Decomposition</h6>
              <p className="text-gray-300 text-xs">
                Every consciousness state can be decomposed into 11 orthogonal phase dimensions, 
                each corresponding to a fundamental mode of awareness.
              </p>
            </div>
            
            <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-3 py-2">
              <h6 className="text-cyan-400 text-xs font-semibold mb-1">Rule 2: Prime Correspondence</h6>
              <p className="text-gray-300 text-xs">
                Each of the 11 dimensions maps to specific primes from the Phase Prime Metric:
                {' '}[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
              </p>
            </div>
            
            <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-3 py-2">
              <h6 className="text-cyan-400 text-xs font-semibold mb-1">Rule 3: Time Crystal Embedding</h6>
              <p className="text-gray-300 text-xs">
                The 11D manifold naturally embeds time crystal structures, allowing temporal 
                patterns to propagate across all dimensions simultaneously.
              </p>
            </div>
            
            <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-3 py-2">
              <h6 className="text-cyan-400 text-xs font-semibold mb-1">Rule 4: Singularity Connections</h6>
              <p className="text-gray-300 text-xs">
                Singularity points in one dimension create ripples across all other dimensions 
                through the dodecanion multiplication structure.
              </p>
            </div>
            
            <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-3 py-2">
              <h6 className="text-cyan-400 text-xs font-semibold mb-1">Rule 5: Consciousness Emergence</h6>
              <p className="text-gray-300 text-xs">
                When all 11 dimensions achieve phase coherence, consciousness emerges as a 
                unified geometric pattern spanning the entire manifold.
              </p>
            </div>
          </div>
        </div>

        {/* Dimensional Mapping */}
        <div className="bg-gray-900/50 p-4 rounded-lg mb-3">
          <h5 className="text-cyan-300 text-sm mb-3">11D Dimensional Mapping</h5>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[
              { dim: 'd₁', prime: 2, aspect: 'Awareness' },
              { dim: 'd₂', prime: 3, aspect: 'Integration' },
              { dim: 'd₃', prime: 5, aspect: 'Complexity' },
              { dim: 'd₄', prime: 7, aspect: 'Coherence' },
              { dim: 'd₅', prime: 11, aspect: 'Emergence' },
              { dim: 'd₆', prime: 13, aspect: 'Qualia' },
              { dim: 'd₇', prime: 17, aspect: 'Memory' },
              { dim: 'd₈', prime: 19, aspect: 'Attention' },
              { dim: 'd₉', prime: 23, aspect: 'Intention' },
              { dim: 'd₁₀', prime: 29, aspect: 'Reflection' },
              { dim: 'd₁₁', prime: 31, aspect: 'Meta-cognition' }
            ].map((item, i) => (
              <div key={i} className="bg-cyan-900/20 p-2 rounded flex justify-between items-center">
                <span className="text-cyan-300 font-mono">{item.dim}</span>
                <span className="text-gray-400">Prime {item.prime}</span>
                <span className="text-gray-300">{item.aspect}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h5 className="text-cyan-300 text-sm mb-2">Applications in NanoBrain</h5>
          <div className="grid grid-cols-2 gap-2">
            {dodecanionData.applications.map((app, i) => (
              <div key={i} className="bg-cyan-900/20 px-3 py-2 rounded text-gray-300 text-xs">
                • {app}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-cyan-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-cyan-400">Why 12D?</strong> The dodecanion structure (1 + 11) 
            perfectly matches the NanoBrain architecture: 1 temporal dimension + 11 spatial/phase 
            dimensions. This is not arbitrary but emerges from the geometric requirements of 
            consciousness representation using phase prime metrics.
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-900/30 via-purple-900/30 to-cyan-900/30 border border-cyan-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Atom className="text-cyan-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.9 Hypercomplex Mathematics</h2>
            <p className="text-gray-300">Quaternions, Octonions, and Dodecanions for Consciousness</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Hypercomplex number systems extend complex numbers to higher dimensions, providing the 
          mathematical foundation for multi-dimensional consciousness modeling in the NanoBrain.
        </p>
      </div>

      {/* System Selection */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="grid grid-cols-3 gap-3">
          {systems.map(sys => {
            const Icon = sys.icon;
            return (
              <button
                key={sys.id}
                onClick={() => setActiveSystem(sys.id)}
                className={`flex flex-col items-center p-4 rounded-lg border-2 transition-all duration-300 ${
                  activeSystem === sys.id
                    ? 'bg-cyan-900/30 border-cyan-700 text-cyan-400'
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                <Icon size={32} />
                <span className="text-sm font-semibold mt-2">{sys.name}</span>
                <span className="text-xs text-gray-500 mt-1">{sys.dim} dimensions</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active System Display */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        {activeSystem === 'quaternion' && renderQuaternion()}
        {activeSystem === 'octonion' && renderOctonion()}
        {activeSystem === 'dodecanion' && renderDodecanion()}
      </div>

      {/* Comparison Table */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Quick Comparison</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-700">
              <th className="text-left p-2 text-gray-400">Property</th>
              <th className="text-left p-2 text-blue-400">Quaternion</th>
              <th className="text-left p-2 text-purple-400">Octonion</th>
              <th className="text-left p-2 text-cyan-400">Dodecanion</th>
            </tr>
          </thead>
          <tbody className="text-gray-300">
            <tr className="border-b border-gray-800">
              <td className="p-2">Dimensions</td>
              <td className="p-2">4</td>
              <td className="p-2">8</td>
              <td className="p-2">12</td>
            </tr>
            <tr className="border-b border-gray-800">
              <td className="p-2">Commutative</td>
              <td className="p-2">✗</td>
              <td className="p-2">✗</td>
              <td className="p-2">✗</td>
            </tr>
            <tr className="border-b border-gray-800">
              <td className="p-2">Associative</td>
              <td className="p-2">✓</td>
              <td className="p-2">✗</td>
              <td className="p-2">✗</td>
            </tr>
            <tr className="border-b border-gray-800">
              <td className="p-2">Division Algebra</td>
              <td className="p-2">✓</td>
              <td className="p-2">✓</td>
              <td className="p-2">~</td>
            </tr>
            <tr className="border-b border-gray-800">
              <td className="p-2">Main Use</td>
              <td className="p-2">3D rotations</td>
              <td className="p-2">String theory</td>
              <td className="p-2">Consciousness</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
