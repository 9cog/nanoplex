import React, { useState } from 'react';
import { TrendingUp, Calculator, Layers } from 'lucide-react';

// Section 3.1.2 & 3.3: Ordered Factor Metric and its 3D version
export const OrderedFactorMetric: React.FC = () => {
  const [selectedNumber, setSelectedNumber] = useState<number>(60);
  const [visualization3D, setVisualization3D] = useState<boolean>(false);

  // Calculate prime factorization
  const primeFactorization = (n: number): number[] => {
    const factors: number[] = [];
    let num = n;
    for (let i = 2; i <= num; i++) {
      while (num % i === 0) {
        factors.push(i);
        num = num / i;
      }
    }
    return factors;
  };

  // Calculate Ordered Factor Metric
  const orderedFactorMetric = (n: number): number => {
    const factors = primeFactorization(n);
    return factors.reduce((product, factor) => product * factor, 1);
  };

  // Generate test numbers with interesting OF properties
  const testNumbers = [
    { n: 12, description: 'Small composite' },
    { n: 30, description: 'Product of first 3 primes' },
    { n: 60, description: 'Highly composite' },
    { n: 120, description: 'Factorial-like' },
    { n: 210, description: 'Product of first 4 primes' },
    { n: 360, description: 'Full circle (degrees)' }
  ];

  const factors = primeFactorization(selectedNumber);
  const of = orderedFactorMetric(selectedNumber);
  const ofRatio = of / selectedNumber;

  // Calculate 3D coordinates for visualization
  const get3DCoordinates = (factors: number[]) => {
    const primes = [2, 3, 5, 7, 11, 13, 17];
    const coords = { x: 0, y: 0, z: 0 };
    
    factors.forEach((prime) => {
      const primeIdx = primes.indexOf(prime);
      if (primeIdx !== -1) {
        if (primeIdx % 3 === 0) coords.x += 1;
        else if (primeIdx % 3 === 1) coords.y += 1;
        else coords.z += 1;
      }
    });
    
    return coords;
  };

  const coords3D = get3DCoordinates(factors);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <TrendingUp className="text-purple-400" size={24} />
          <span>3.1.2 & 3.3: Ordered Factor Metric (OF)</span>
        </h2>
        <button
          onClick={() => setVisualization3D(!visualization3D)}
          className={`px-4 py-2 rounded-lg border-2 transition-colors ${
            visualization3D
              ? 'bg-cyan-900/30 border-cyan-700 text-cyan-400'
              : 'bg-gray-800/30 border-gray-700 text-gray-400'
          }`}
        >
          {visualization3D ? '3D View' : '2D View'}
        </button>
      </div>

      {/* Number Selector */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">Select Number for Analysis</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {testNumbers.map(({ n, description }) => (
            <button
              key={n}
              onClick={() => setSelectedNumber(n)}
              className={`p-4 rounded-lg border-2 transition-all duration-300 ${
                selectedNumber === n
                  ? 'bg-purple-900/30 border-purple-700 text-purple-400'
                  : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <div className="text-2xl font-bold mb-1">{n}</div>
              <div className="text-xs">{description}</div>
            </button>
          ))}
        </div>

        {/* Custom Number Input */}
        <div className="mt-4">
          <label className="text-gray-400 text-sm mb-2 block">Or enter custom number:</label>
          <input
            type="number"
            value={selectedNumber}
            onChange={(e) => setSelectedNumber(Math.max(2, parseInt(e.target.value) || 2))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white"
            min="2"
            max="9999"
          />
        </div>
      </div>

      {/* OF Calculation Display */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">Ordered Factor Analysis: {selectedNumber}</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          {/* Prime Factorization */}
          <div className="bg-gradient-to-br from-cyan-900/20 to-cyan-800/20 rounded-lg p-4 border border-cyan-700/50">
            <h4 className="text-cyan-400 font-semibold mb-2 flex items-center space-x-2">
              <Layers size={16} />
              <span>Prime Factors</span>
            </h4>
            <div className="font-mono text-2xl text-white">
              {factors.join(' × ')}
            </div>
            <div className="text-gray-400 text-sm mt-2">
              {factors.length} prime factors
            </div>
          </div>

          {/* Ordered Factor */}
          <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 rounded-lg p-4 border border-purple-700/50">
            <h4 className="text-purple-400 font-semibold mb-2 flex items-center space-x-2">
              <Calculator size={16} />
              <span>Ordered Factor</span>
            </h4>
            <div className="font-mono text-2xl text-white">
              OF = {of}
            </div>
            <div className="text-gray-400 text-sm mt-2">
              Product of all factors
            </div>
          </div>

          {/* OF Ratio */}
          <div className="bg-gradient-to-br from-green-900/20 to-green-800/20 rounded-lg p-4 border border-green-700/50">
            <h4 className="text-green-400 font-semibold mb-2 flex items-center space-x-2">
              <TrendingUp size={16} />
              <span>OF Ratio</span>
            </h4>
            <div className="font-mono text-2xl text-white">
              {ofRatio.toFixed(2)}
            </div>
            <div className="text-gray-400 text-sm mt-2">
              OF / n = {ofRatio > 1 ? 'amplification' : 'compression'}
            </div>
          </div>
        </div>

        {/* Mathematical Formulas */}
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <h4 className="text-yellow-400 font-semibold mb-3">Mathematical Definition</h4>
          <div className="space-y-2 font-mono text-sm">
            <div className="text-gray-300">
              n = {selectedNumber} = {factors.join(' × ')}
            </div>
            <div className="text-cyan-400">
              OF({selectedNumber}) = ∏ pᵢ^αᵢ = {of}
            </div>
            <div className="text-purple-400">
              Ratio = OF({selectedNumber}) / {selectedNumber} = {ofRatio.toFixed(4)}
            </div>
          </div>
        </div>
      </div>

      {/* Visualization */}
      {visualization3D ? (
        /* 3D Visualization */
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">3D Ordered Factor Representation</h3>
          <div className="bg-gray-800/50 rounded-lg p-8 border border-cyan-700/50">
            <div className="relative h-96 flex items-center justify-center">
              {/* 3D coordinate system */}
              <div className="relative w-full h-full">
                {/* X-axis (red) */}
                <div className="absolute left-0 top-1/2 w-full h-px bg-red-500/30" />
                <div className="absolute left-0 top-1/2 transform -translate-y-full text-red-400 text-sm">
                  X-axis (Primes: 2, 5, 11...)
                </div>
                
                {/* Y-axis (green) */}
                <div className="absolute left-1/2 top-0 w-px h-full bg-green-500/30" />
                <div className="absolute left-1/2 top-0 transform translate-x-2 text-green-400 text-sm">
                  Y-axis (Primes: 3, 7, 13...)
                </div>
                
                {/* Z-axis (blue) - diagonal */}
                <div className="absolute left-1/2 top-1/2 w-1 h-1 bg-blue-500 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="absolute left-full ml-2 text-blue-400 text-sm whitespace-nowrap">
                    Z-axis (Primes: 5, 11, 17...)
                  </div>
                </div>

                {/* Point representation */}
                <div
                  className="absolute w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-purple-500 border-2 border-white shadow-lg shadow-purple-500/50 flex items-center justify-center text-white font-bold text-xs"
                  style={{
                    left: `${50 + coords3D.x * 10}%`,
                    top: `${50 - coords3D.y * 10}%`,
                    transform: `translate(-50%, -50%) scale(${1 + coords3D.z * 0.2})`
                  }}
                >
                  {selectedNumber}
                </div>

                {/* Coordinate labels */}
                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 text-center">
                  <div className="bg-gray-900/80 rounded-lg px-4 py-2 border border-gray-700">
                    <div className="text-xs text-gray-400 mb-1">3D Coordinates:</div>
                    <div className="font-mono text-cyan-400">
                      ({coords3D.x}, {coords3D.y}, {coords3D.z})
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 text-sm text-gray-300 space-y-2">
              <p>
                <strong className="text-cyan-400">3D Mapping:</strong> Each prime factor contributes
                to one of three dimensions based on its position in the prime sequence.
              </p>
              <p>
                <strong className="text-purple-400">Coordinates:</strong> X={coords3D.x}, Y={coords3D.y}, Z={coords3D.z}
              </p>
              <p>
                <strong className="text-green-400">Euclidean Distance:</strong>{' '}
                {Math.sqrt(coords3D.x**2 + coords3D.y**2 + coords3D.z**2).toFixed(2)} from origin
              </p>
            </div>
          </div>
        </div>
      ) : (
        // 2D Visualization
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">2D Ordered Factor Plot</h3>
          <div className="bg-gray-800/50 rounded-lg p-8 border border-purple-700/50">
            <div className="relative h-96">
              {/* Axes */}
              <div className="absolute left-0 bottom-0 w-full h-px bg-gray-600" />
              <div className="absolute left-0 bottom-0 w-px h-full bg-gray-600" />
              
              {/* X-axis label */}
              <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-gray-400 text-sm">
                Integer Value (n)
              </div>
              
              {/* Y-axis label */}
              <div className="absolute -left-16 top-1/2 transform -translate-y-1/2 -rotate-90 text-gray-400 text-sm">
                Ordered Factor (OF)
              </div>

              {/* Plot point */}
              <div
                className="absolute w-4 h-4 rounded-full bg-gradient-to-br from-cyan-500 to-purple-500 border-2 border-white shadow-lg"
                style={{
                  left: `${(selectedNumber / 400) * 100}%`,
                  bottom: `${(of / 400) * 100}%`
                }}
              />

              {/* Reference line (y=x) */}
              <div className="absolute left-0 bottom-0 w-full h-full">
                <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                  <line
                    x1="0"
                    y1="100"
                    x2="100"
                    y2="0"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="0.5"
                    strokeDasharray="2,2"
                    opacity="0.3"
                  />
                </svg>
              </div>

              {/* Value labels */}
              <div className="absolute right-4 top-4 bg-gray-900/80 rounded-lg px-3 py-2 border border-gray-700">
                <div className="text-xs text-gray-400 mb-1">Point:</div>
                <div className="font-mono text-cyan-400 text-sm">
                  ({selectedNumber}, {of})
                </div>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
              <div>
                <strong className="text-cyan-400">Blue dashed line:</strong> Reference line y=x
                (where OF equals n)
              </div>
              <div>
                <strong className="text-purple-400">Point position:</strong>{' '}
                {ofRatio > 1 ? 'Above' : ofRatio < 1 ? 'Below' : 'On'} reference line
                ({ofRatio > 1 ? 'amplification' : ofRatio < 1 ? 'compression' : 'neutral'})
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Significance Section */}
      <div className="bg-gradient-to-br from-purple-900/20 to-cyan-900/20 border border-purple-700/50 rounded-xl p-6">
        <h3 className="text-purple-400 font-bold text-lg mb-3">Significance of Ordered Factor Metric</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div className="space-y-2">
            <p>
              <strong className="text-cyan-400">Unique Integer Signature:</strong> Every integer has
              a unique ordered factor metric based on its prime composition.
            </p>
            <p>
              <strong className="text-green-400">Consciousness Encoding:</strong> OF provides a natural
              mapping from integers to consciousness patterns in phase space.
            </p>
          </div>
          <div className="space-y-2">
            <p>
              <strong className="text-yellow-400">Time Crystal Structure:</strong> The 3D OF representation
              reveals the geometric structure underlying time crystal formation.
            </p>
            <p>
              <strong className="text-red-400">Universal Symmetry:</strong> OF ratios reveal hidden
              symmetries in the integer space governed by prime patterns.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
