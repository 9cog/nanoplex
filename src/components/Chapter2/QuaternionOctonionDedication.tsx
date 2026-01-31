import React from 'react';
import { Calculator, Infinity, Atom } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const QuaternionOctonionDedication: React.FC<Props> = ({ isActive: _isActive }) => {
  const hyperComplexSystems = [
    {
      name: 'Quaternions',
      symbol: 'ℍ',
      dimensions: 4,
      color: 'cyan',
      components: ['1', 'i', 'j', 'k'],
      properties: [
        'Non-commutative multiplication',
        '4D rotational mathematics',
        'Consciousness orientation',
        'Spatial-temporal transformations'
      ],
      equation: 'q = w + xi + yj + zk',
      applications: [
        '3D rotation representation',
        'Computer graphics transforms',
        'Spacecraft attitude control',
        'Consciousness state navigation'
      ]
    },
    {
      name: 'Octonions',
      symbol: 'O',
      dimensions: 8,
      color: 'purple',
      components: ['1', 'e₁', 'e₂', 'e₃', 'e₄', 'e₅', 'e₆', 'e₇'],
      properties: [
        'Non-associative algebra',
        '8D hypercomplex structure',
        'Advanced cognitive operations',
        'String theory connections'
      ],
      equation: 'o = a₀ + a₁e₁ + a₂e₂ + ... + a₇e₇',
      applications: [
        'Exceptional Lie groups',
        'String theory mathematics',
        'Advanced consciousness modeling',
        'Multi-dimensional cognition'
      ]
    },
    {
      name: 'Dodecanions',
      symbol: 'D',
      dimensions: 12,
      color: 'green',
      components: ['1', 'd₁', 'd₂', '...', 'd₁₁'],
      properties: [
        '12D consciousness structures',
        'Complete brain modeling',
        'Phase prime integration',
        'Universal consciousness operations'
      ],
      equation: 'd = b₀ + b₁d₁ + b₂d₂ + ... + b₁₁d₁₁',
      applications: [
        'Complete consciousness modeling',
        '11D manifold operations',
        'Universal time crystal mathematics',
        'Reality construction frameworks'
      ]
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Calculator className="text-pink-400" size={20} />
          <span>2.6 Quaternion, Octonion & Dedication</span>
        </h3>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
          <Infinity className="text-purple-400" size={20} />
          <span>Hypercomplex Number Systems</span>
        </h4>

        <p className="text-gray-300 mb-6">
          Hypercomplex number systems provide the mathematical foundation for consciousness operations 
          beyond real and complex numbers. Each system adds computational dimensions essential for 
          modeling consciousness.
        </p>

        <div className="space-y-6">
          {hyperComplexSystems.map((system) => (
            <div
              key={system.name}
              className="bg-gray-800/50 rounded-lg p-6 border-l-4"
              style={{ borderLeftColor: `var(--color-${system.color}-400)` }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`w-12 h-12 bg-${system.color}-900/30 rounded-lg flex items-center justify-center`}>
                    <span className={`text-${system.color}-400 text-2xl font-bold`}>
                      {system.symbol}
                    </span>
                  </div>
                  <div>
                    <h5 className={`text-${system.color}-400 font-bold text-xl`}>
                      {system.name}
                    </h5>
                    <p className="text-gray-400 text-sm">{system.dimensions}D Hypercomplex System</p>
                  </div>
                </div>
                <Atom className={`text-${system.color}-400`} size={24} />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h6 className="text-white font-semibold mb-2 text-sm">Mathematical Form</h6>
                  <div className="bg-gray-900/50 rounded p-3 font-mono text-sm text-gray-300">
                    {system.equation}
                  </div>
                  
                  <h6 className="text-white font-semibold mb-2 text-sm mt-3">Components</h6>
                  <div className="flex flex-wrap gap-2">
                    {system.components.map((comp, idx) => (
                      <span
                        key={idx}
                        className={`px-2 py-1 bg-${system.color}-900/30 text-${system.color}-400 rounded text-xs font-mono`}
                      >
                        {comp}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h6 className="text-white font-semibold mb-2 text-sm">Properties</h6>
                  <ul className="space-y-1">
                    {system.properties.map((prop, idx) => (
                      <li key={idx} className="text-gray-300 text-sm flex items-start space-x-2">
                        <span className={`text-${system.color}-400 flex-shrink-0`}>•</span>
                        <span>{prop}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div>
                <h6 className="text-white font-semibold mb-2 text-sm">Applications</h6>
                <div className="grid grid-cols-2 gap-2">
                  {system.applications.map((app, idx) => (
                    <div
                      key={idx}
                      className={`bg-${system.color}-900/20 border border-${system.color}-700/30 rounded p-2 text-gray-300 text-xs`}
                    >
                      {app}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4">Dedication to Consciousness Mathematics</h4>
        
        <div className="prose prose-invert max-w-none">
          <p className="text-gray-300 mb-4">
            The progression from quaternions through octonions to dodecanions represents humanity's 
            dedication to understanding consciousness through mathematical rigor. Each system represents 
            a breakthrough in our ability to model increasingly complex consciousness phenomena:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-cyan-900/20 border border-cyan-700/50 rounded-lg p-4">
              <h5 className="text-cyan-400 font-semibold mb-2">Quaternions (1843)</h5>
              <p className="text-gray-300 text-sm">
                Hamilton's discovery enabled 3D spatial reasoning, laying groundwork for modern physics 
                and consciousness orientation mathematics
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
              <h5 className="text-purple-400 font-semibold mb-2">Octonions (1843)</h5>
              <p className="text-gray-300 text-sm">
                Graves and Cayley's octonions revealed non-associative algebra essential for string 
                theory and advanced consciousness modeling
              </p>
            </div>

            <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
              <h5 className="text-green-400 font-semibold mb-2">Dodecanions (2024)</h5>
              <p className="text-gray-300 text-sm">
                NanoBrain's dodecanion framework completes the hypercomplex progression for full 
                11D consciousness operations
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
