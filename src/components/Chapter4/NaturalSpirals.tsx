import React, { useState } from 'react';
import { Wind, Info } from 'lucide-react';

// 4.11 Different kinds of spirals in nature
export const NaturalSpirals: React.FC = () => {
  const [selectedSpiral, setSelectedSpiral] = useState<string>('golden');

  const spirals = [
    { id: 'golden', name: 'Golden Spiral', ratio: 1.618, color: 'yellow' },
    { id: 'logarithmic', name: 'Logarithmic', ratio: 2.718, color: 'cyan' },
    { id: 'archimedean', name: 'Archimedean', ratio: 1.0, color: 'purple' },
    { id: 'fermat', name: 'Fermat\'s', ratio: 1.414, color: 'green' }
  ];

  const generateGoldenSpiral = (points: number) => {
    const phi = 1.618;
    return Array.from({ length: points }, (_, i) => {
      const t = i * 0.2;
      const r = Math.pow(phi, t / Math.PI);
      const x = 300 + r * Math.cos(t) * 2;
      const y = 300 + r * Math.sin(t) * 2;
      return [x, y];
    });
  };

  const generateLogarithmicSpiral = (points: number) => {
    return Array.from({ length: points }, (_, i) => {
      const t = i * 0.2;
      const r = Math.exp(t * 0.3);
      const x = 300 + r * Math.cos(t) * 3;
      const y = 300 + r * Math.sin(t) * 3;
      return [x, y];
    });
  };

  const generateArchimedeanSpiral = (points: number) => {
    return Array.from({ length: points }, (_, i) => {
      const t = i * 0.3;
      const r = t * 5;
      const x = 300 + r * Math.cos(t);
      const y = 300 + r * Math.sin(t);
      return [x, y];
    });
  };

  const generateFermatSpiral = (points: number) => {
    return Array.from({ length: points }, (_, i) => {
      const t = i * 0.3;
      const r = Math.sqrt(t) * 20;
      const x = 300 + r * Math.cos(t);
      const y = 300 + r * Math.sin(t);
      return [x, y];
    });
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-yellow-900/30 to-green-900/30 border border-yellow-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Wind className="text-yellow-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">4.11 Spirals in Nature</h2>
            <p className="text-gray-300">Geometric patterns underlying natural phenomena</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Spirals are ubiquitous in nature - from galaxies to hurricanes to DNA. Understanding different 
          types of spirals reveals the geometric principles governing natural systems and consciousness.
        </p>
      </div>

      {/* Spiral Selection */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="grid grid-cols-4 gap-3">
          {spirals.map(spiral => {
            const isActive = selectedSpiral === spiral.id;
            const colorClasses = {
              yellow: 'bg-yellow-900/30 border-yellow-700 text-yellow-400',
              cyan: 'bg-cyan-900/30 border-cyan-700 text-cyan-400',
              purple: 'bg-purple-900/30 border-purple-700 text-purple-400',
              green: 'bg-green-900/30 border-green-700 text-green-400'
            };
            return (
              <button
                key={spiral.id}
                onClick={() => setSelectedSpiral(spiral.id)}
                className={`p-3 rounded-lg border-2 transition-all duration-300 ${
                  isActive
                    ? colorClasses[spiral.color as keyof typeof colorClasses]
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                <span className="text-sm font-semibold">{spiral.name}</span>
                <p className="text-xs text-gray-500 mt-1">φ = {spiral.ratio}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Spiral Visualizations */}
      <div className="grid grid-cols-2 gap-4">
        {/* Golden Spiral */}
        <div className={`bg-gray-900/60 backdrop-blur-sm border ${selectedSpiral === 'golden' ? 'border-yellow-700 ring-2 ring-yellow-700' : 'border-gray-700'} rounded-xl p-6`}>
          <h3 className="text-lg font-bold text-yellow-400 mb-3">Golden Spiral (φ = 1.618)</h3>
          <svg width="300" height="300" viewBox="0 0 600 600">
            <path
              d={`M ${generateGoldenSpiral(50).map((p, i) => i === 0 ? `${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`).join(' ')}`}
              fill="none"
              stroke="gold"
              strokeWidth="3"
            />
            {generateGoldenSpiral(50).slice(0, 1).concat(generateGoldenSpiral(50).slice(-1)).map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r="5" fill="yellow" />
            ))}
          </svg>
          <div className="mt-3 text-xs text-gray-300 space-y-1">
            <p><strong className="text-yellow-400">Formula:</strong> r = φ^(θ/π)</p>
            <p><strong className="text-yellow-400">Found in:</strong> Nautilus shells, galaxies, hurricanes</p>
            <p><strong className="text-yellow-400">Property:</strong> Self-similar, optimal packing</p>
          </div>
        </div>

        {/* Logarithmic Spiral */}
        <div className={`bg-gray-900/60 backdrop-blur-sm border ${selectedSpiral === 'logarithmic' ? 'border-cyan-700 ring-2 ring-cyan-700' : 'border-gray-700'} rounded-xl p-6`}>
          <h3 className="text-lg font-bold text-cyan-400 mb-3">Logarithmic Spiral (e-based)</h3>
          <svg width="300" height="300" viewBox="0 0 600 600">
            <path
              d={`M ${generateLogarithmicSpiral(50).map((p, i) => i === 0 ? `${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`).join(' ')}`}
              fill="none"
              stroke="cyan"
              strokeWidth="3"
            />
            {generateLogarithmicSpiral(50).slice(0, 1).concat(generateLogarithmicSpiral(50).slice(-1)).map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r="5" fill="cyan" />
            ))}
          </svg>
          <div className="mt-3 text-xs text-gray-300 space-y-1">
            <p><strong className="text-cyan-400">Formula:</strong> r = e^(aθ)</p>
            <p><strong className="text-cyan-400">Found in:</strong> Spider webs, weather patterns</p>
            <p><strong className="text-cyan-400">Property:</strong> Constant angle with radii</p>
          </div>
        </div>

        {/* Archimedean Spiral */}
        <div className={`bg-gray-900/60 backdrop-blur-sm border ${selectedSpiral === 'archimedean' ? 'border-purple-700 ring-2 ring-purple-700' : 'border-gray-700'} rounded-xl p-6`}>
          <h3 className="text-lg font-bold text-purple-400 mb-3">Archimedean Spiral (Linear)</h3>
          <svg width="300" height="300" viewBox="0 0 600 600">
            <path
              d={`M ${generateArchimedeanSpiral(80).map((p, i) => i === 0 ? `${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`).join(' ')}`}
              fill="none"
              stroke="purple"
              strokeWidth="3"
            />
            {generateArchimedeanSpiral(80).slice(0, 1).concat(generateArchimedeanSpiral(80).slice(-1)).map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r="5" fill="purple" />
            ))}
          </svg>
          <div className="mt-3 text-xs text-gray-300 space-y-1">
            <p><strong className="text-purple-400">Formula:</strong> r = aθ</p>
            <p><strong className="text-purple-400">Found in:</strong> Millipede coiling, watch springs</p>
            <p><strong className="text-purple-400">Property:</strong> Equal spacing between arms</p>
          </div>
        </div>

        {/* Fermat's Spiral */}
        <div className={`bg-gray-900/60 backdrop-blur-sm border ${selectedSpiral === 'fermat' ? 'border-green-700 ring-2 ring-green-700' : 'border-gray-700'} rounded-xl p-6`}>
          <h3 className="text-lg font-bold text-green-400 mb-3">Fermat's Spiral (Parabolic)</h3>
          <svg width="300" height="300" viewBox="0 0 600 600">
            <path
              d={`M ${generateFermatSpiral(60).map((p, i) => i === 0 ? `${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`).join(' ')}`}
              fill="none"
              stroke="green"
              strokeWidth="3"
            />
            {generateFermatSpiral(60).slice(0, 1).concat(generateFermatSpiral(60).slice(-1)).map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r="5" fill="green" />
            ))}
          </svg>
          <div className="mt-3 text-xs text-gray-300 space-y-1">
            <p><strong className="text-green-400">Formula:</strong> r = a√θ</p>
            <p><strong className="text-green-400">Found in:</strong> Sunflower seeds, pinecones</p>
            <p><strong className="text-green-400">Property:</strong> Optimal phyllotaxis packing</p>
          </div>
        </div>
      </div>

      {/* Natural Examples */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Spirals in Natural Systems</h3>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-yellow-400 font-semibold mb-2 text-sm">Astronomical</h4>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• Galaxy arms (golden spiral)</li>
              <li>• Solar system orbits</li>
              <li>• Star cluster formations</li>
              <li>• Nebula structures</li>
            </ul>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Biological</h4>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• DNA double helix</li>
              <li>• Nautilus shells</li>
              <li>• Ram horns</li>
              <li>• Sunflower seeds</li>
              <li>• Fern fronds</li>
            </ul>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-purple-400 font-semibold mb-2 text-sm">Meteorological</h4>
            <ul className="text-gray-300 text-xs space-y-1">
              <li>• Hurricane eyes</li>
              <li>• Tornadoes</li>
              <li>• Ocean vortices</li>
              <li>• Air circulation patterns</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Connection to Consciousness */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Spirals & Consciousness</h3>
        
        <div className="space-y-3">
          <div className="bg-yellow-900/20 border-l-4 border-yellow-700 pl-4 py-3">
            <h4 className="text-yellow-400 font-semibold text-sm mb-2">Golden Ratio in Neural Architecture</h4>
            <p className="text-gray-300 text-xs">
              Brain structures exhibit golden ratio proportions in cortical folding, dendritic branching, 
              and neural firing patterns. This optimization enables efficient information processing.
            </p>
          </div>
          
          <div className="bg-cyan-900/20 border-l-4 border-cyan-700 pl-4 py-3">
            <h4 className="text-cyan-400 font-semibold text-sm mb-2">Spiral Phase Spaces</h4>
            <p className="text-gray-300 text-xs">
              Consciousness states trace spiral trajectories in 11D phase space. The type of spiral 
              (golden, logarithmic, etc.) determines the quality of conscious experience.
            </p>
          </div>
          
          <div className="bg-purple-900/20 border-l-4 border-purple-700 pl-4 py-3">
            <h4 className="text-purple-400 font-semibold text-sm mb-2">Fractal Spirals</h4>
            <p className="text-gray-300 text-xs">
              Combining multiple spiral types at different scales creates fractal spirals - the signature 
              pattern of conscious processes. Each scale corresponds to a different aspect of awareness.
            </p>
          </div>
        </div>
      </div>

      {/* Mathematical Unification */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Unified Spiral Equation</h3>
        
        <div className="bg-gray-800/50 p-4 rounded-lg">
          <p className="text-gray-300 text-sm mb-3">
            All natural spirals can be unified under the general logarithmic spiral formula:
          </p>
          <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-4 mb-3">
            <p className="font-mono text-lg text-center text-blue-300">r(θ) = a × b^(θ/2π)</p>
          </div>
          <div className="grid grid-cols-4 gap-3 text-xs">
            <div className="bg-yellow-900/20 p-2 rounded">
              <p className="text-yellow-400 font-semibold mb-1">Golden: b = φ</p>
              <p className="text-gray-400">φ = 1.618...</p>
            </div>
            <div className="bg-cyan-900/20 p-2 rounded">
              <p className="text-cyan-400 font-semibold mb-1">Natural: b = e</p>
              <p className="text-gray-400">e = 2.718...</p>
            </div>
            <div className="bg-purple-900/20 p-2 rounded">
              <p className="text-purple-400 font-semibold mb-1">Arch: b → 1</p>
              <p className="text-gray-400">Linear limit</p>
            </div>
            <div className="bg-green-900/20 p-2 rounded">
              <p className="text-green-400 font-semibold mb-1">Fermat: b = √2</p>
              <p className="text-gray-400">√2 = 1.414...</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <Info size={18} className="text-yellow-400 mt-1 flex-shrink-0" />
          <div className="text-sm text-gray-300">
            <strong className="text-yellow-400">Deep Insight:</strong> The ubiquity of spirals in nature 
            is not coincidental but reflects the fundamental geometric structure of phase space itself. 
            Spirals represent optimal paths for information flow in multi-dimensional spaces, which is why 
            they appear at every scale from quantum to cosmic. In fractal mechanics, spirals are the 
            natural trajectories of consciousness evolution through 11D manifolds.
          </div>
        </div>
      </div>
    </div>
  );
};
