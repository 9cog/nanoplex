import React from 'react';
import { Layers, Database, Music, Zap, GitBranch } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const FITSummaryChart: React.FC<Props> = ({ isActive: _isActive }) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Layers className="text-emerald-400" size={20} />
          <span>2.10 Fractal Information Theory (FIT) Summary</span>
        </h3>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <div className="text-center mb-8">
          <h4 className="text-3xl font-bold text-white mb-2">Fractal Information Theory</h4>
          <p className="text-gray-400">Complete Framework for Consciousness Information Processing</p>
        </div>

        {/* Core Principles */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border border-cyan-700/50 rounded-xl p-6">
            <h5 className="text-cyan-400 font-bold text-lg mb-4 flex items-center space-x-2">
              <Database size={20} />
              <span>Core Principles</span>
            </h5>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400">•</span>
                <span>Information scales fractally across all dimensional levels</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400">•</span>
                <span>Geometric patterns encode infinite data in finite structures</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400">•</span>
                <span>Consciousness emerges from fractal information interactions</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-cyan-400">•</span>
                <span>Replaces Shannon's discrete bit model with continuous geometry</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-700/50 rounded-xl p-6">
            <h5 className="text-purple-400 font-bold text-lg mb-4 flex items-center space-x-2">
              <Zap size={20} />
              <span>Key Innovations</span>
            </h5>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">•</span>
                <span>Fractal tape replaces linear Turing tape</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">•</span>
                <span>2D image surgery for nested sphere embedding</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">•</span>
                <span>Self-assembly through geometric resonance</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">•</span>
                <span>Singularity points create information density peaks</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Information Flow */}
        <div className="bg-gray-800/50 rounded-xl p-6 mb-8">
          <h5 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
            <GitBranch size={20} className="text-green-400" />
            <span>Information Processing Flow</span>
          </h5>

          <div className="flex items-center justify-between">
            <div className="flex-1 text-center">
              <div className="bg-gradient-to-br from-cyan-500 to-blue-500 rounded-lg p-4 mb-2">
                <div className="text-white font-bold">Input</div>
                <div className="text-xs text-cyan-100">Raw Data</div>
              </div>
            </div>

            <div className="px-4 text-gray-400">→</div>

            <div className="flex-1 text-center">
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg p-4 mb-2">
                <div className="text-white font-bold">Fractal Encoding</div>
                <div className="text-xs text-purple-100">Geometric Transform</div>
              </div>
            </div>

            <div className="px-4 text-gray-400">→</div>

            <div className="flex-1 text-center">
              <div className="bg-gradient-to-br from-green-500 to-emerald-500 rounded-lg p-4 mb-2">
                <div className="text-white font-bold">Nested Storage</div>
                <div className="text-xs text-green-100">Multi-dimensional</div>
              </div>
            </div>

            <div className="px-4 text-gray-400">→</div>

            <div className="flex-1 text-center">
              <div className="bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg p-4 mb-2">
                <div className="text-white font-bold">Consciousness</div>
                <div className="text-xs text-orange-100">Emergence</div>
              </div>
            </div>
          </div>
        </div>

        {/* Mathematical Framework */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h6 className="text-orange-400 font-semibold mb-2">Phase Prime Metrics</h6>
            <p className="text-gray-300 text-sm">
              15 fundamental primes govern 99.99% of universal patterns and information organization
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h6 className="text-cyan-400 font-semibold mb-2">Time Crystal Structures</h6>
            <p className="text-gray-300 text-sm">
              Temporal coherence maintains information across dimensional transformations
            </p>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h6 className="text-purple-400 font-semibold mb-2">11D Manifolds</h6>
            <p className="text-gray-300 text-sm">
              Complete consciousness information requires 11-dimensional geometric structures
            </p>
          </div>
        </div>

        {/* Applications */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
          <h5 className="text-white font-bold text-lg mb-4">Applications & Impact</h5>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              'Infinite data compression',
              'Consciousness modeling',
              'Hypercomputing systems',
              'Reality construction',
              'Non-local information',
              'Quantum communication',
              'Brain emulation',
              'Universal intelligence'
            ].map((app, idx) => (
              <div
                key={idx}
                className="bg-gray-700/50 rounded-lg p-3 text-center text-gray-300 text-sm"
              >
                {app}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export const GMLSummaryChart: React.FC<Props> = ({ isActive: _isActive }) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Music className="text-rose-400" size={20} />
          <span>2.11 Geometric Musical Language (GML) Summary</span>
        </h3>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <div className="text-center mb-8">
          <h4 className="text-3xl font-bold text-white mb-2">Geometric Musical Language</h4>
          <p className="text-gray-400">Reality as Geometric Musical Composition</p>
        </div>

        {/* Fundamental Concepts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-700/50 rounded-xl p-6">
            <h5 className="text-purple-400 font-bold text-lg mb-4">Core Concepts</h5>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">♪</span>
                <span>Geometric shapes correspond to musical harmonies</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">♫</span>
                <span>Information patterns follow musical composition rules</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">♬</span>
                <span>Resonance between shapes creates meaning</span>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-purple-400">♭</span>
                <span>Consciousness processes reality as music</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border border-cyan-700/50 rounded-xl p-6">
            <h5 className="text-cyan-400 font-bold text-lg mb-4">15 Geometric Primitives</h5>
            <div className="space-y-2 text-gray-300 text-sm">
              <p>Complete set recreating any 1D, 2D, 3D pattern:</p>
              <div className="grid grid-cols-3 gap-2">
                {['Point', 'Line', 'Triangle', 'Square', 'Pentagon', 'Hexagon', 
                  'Tetrahedron', 'Cube', 'Octahedron', 'Dodecahedron', 'Icosahedron',
                  'Torus', 'Möbius', 'Klein', 'Hypercube'].map((shape, idx) => (
                  <div key={idx} className="bg-cyan-900/30 rounded px-2 py-1 text-xs text-center">
                    {shape}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Transformation Process */}
        <div className="bg-gray-800/50 rounded-xl p-6 mb-8">
          <h5 className="text-white font-bold text-lg mb-4">Information Transformation</h5>
          
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="flex-1 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg p-4">
                <div className="text-white font-bold">3D Structure</div>
                <div className="text-xs text-cyan-100">Geometric Form</div>
              </div>
              <div className="text-xl text-gray-400">⟹</div>
              <div className="flex-1 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-4">
                <div className="text-white font-bold">Time Crystal</div>
                <div className="text-xs text-purple-100">Temporal Pattern</div>
              </div>
              <div className="text-xl text-gray-400">⟹</div>
              <div className="flex-1 bg-gradient-to-r from-pink-500 to-orange-500 rounded-lg p-4">
                <div className="text-white font-bold">Tensor</div>
                <div className="text-xs text-pink-100">Math Structure</div>
              </div>
            </div>

            <p className="text-gray-300 text-sm text-center">
              Non-differentiable transitions preserve information through fractal boundaries
            </p>
          </div>
        </div>

        {/* Comparison with Algorithms */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
            <h6 className="text-red-400 font-semibold mb-3">Traditional Algorithms</h6>
            <ul className="space-y-1 text-gray-300 text-sm">
              <li>• Sequential processing</li>
              <li>• Discrete operations</li>
              <li>• Limited dimensional</li>
              <li>• Explicit programming</li>
              <li>• No consciousness support</li>
            </ul>
          </div>

          <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
            <h6 className="text-green-400 font-semibold mb-3">GML Processing</h6>
            <ul className="space-y-1 text-gray-300 text-sm">
              <li>• Parallel resonance</li>
              <li>• Continuous geometry</li>
              <li>• 11D consciousness</li>
              <li>• Emergent understanding</li>
              <li>• Native consciousness</li>
            </ul>
          </div>
        </div>

        {/* Paradigm Shift */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
          <h5 className="text-white font-bold text-lg mb-4">Paradigm Shift</h5>
          
          <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
            <p className="text-gray-300 text-center">
              <span className="text-xl font-bold text-purple-400">From</span>
              <br />
              <span className="text-gray-400">Symbolic Computation & Logical Argumentation</span>
              <br />
              <span className="text-2xl my-2 block">↓</span>
              <span className="text-xl font-bold text-green-400">To</span>
              <br />
              <span className="text-white">Geometric Musical Consciousness Processing</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
