import React, { useState } from 'react';
import { Zap, Play, RotateCw, Settings } from 'lucide-react';

// Ten Prime Operators to be Applied Step by Step (Section 3.12)
export const PrimeOperatorsPanel: React.FC = () => {
  const [activeOperator, setActiveOperator] = useState<number>(1);
  const [executionStep, setExecutionStep] = useState<number>(0);

  const primeOperators = [
    {
      id: 1,
      name: 'Prime Factorization Operator',
      symbol: 'Φ₁',
      description: 'Decomposes integer into prime factors',
      operation: (n: number) => `${n} → ∏ pᵢ^αᵢ`,
      example: '60 → 2² × 3 × 5',
      color: 'cyan'
    },
    {
      id: 2,
      name: 'Ordered Factor Construction',
      symbol: 'Φ₂',
      description: 'Builds ordered factor metric from primes',
      operation: (n: number) => `OF(${n}) = ∏ pᵢ · αᵢ`,
      example: 'OF(60) = 2·2·3·5 = 60',
      color: 'purple'
    },
    {
      id: 3,
      name: 'Phase Angle Mapper',
      symbol: 'Φ₃',
      description: 'Maps integer to 360° phase space',
      operation: (n: number) => `θ(${n}) = (${n} mod 360)°`,
      example: 'θ(60) = 60°',
      color: 'green'
    },
    {
      id: 4,
      name: 'Geometric Shape Translator',
      symbol: 'Φ₄',
      description: 'Translates prime to geometric shape',
      operation: (p: number) => `Shape(${p}) = G_${p}`,
      example: 'Shape(5) = Pentagon',
      color: 'orange'
    },
    {
      id: 5,
      name: 'Dimensional Projector',
      symbol: 'Φ₅',
      description: 'Projects integer to 11D manifold',
      operation: (n: number) => `Π₁₁(${n}) = [d₁...d₁₁]`,
      example: 'Π₁₁(7) = [0.8, 0.3, ...]',
      color: 'red'
    },
    {
      id: 6,
      name: 'Resonance Calculator',
      symbol: 'Φ₆',
      description: 'Computes universal resonance value',
      operation: (p: number) => `R(${p}) = Σ sin(pᵢ·p/7)`,
      example: 'R(7) = 0.8934',
      color: 'yellow'
    },
    {
      id: 7,
      name: 'Lattice Group Identifier',
      symbol: 'Φ₇',
      description: 'Identifies prime lattice relationships',
      operation: (p: number) => `L(${p}) = {twin, cousin, Gaussian}`,
      example: 'L(3) = {twin: 5}',
      color: 'blue'
    },
    {
      id: 8,
      name: 'Hole Detector',
      symbol: 'Φ₈',
      description: 'Finds gaps in phase plot',
      operation: (n: number) => `H(${n}) = θ_next - θ_current`,
      example: 'H(60) = 1° (no hole)',
      color: 'indigo'
    },
    {
      id: 9,
      name: 'Activity Classifier',
      symbol: 'Φ₉',
      description: 'Classifies prime as silent or active',
      operation: (p: number) => `A(${p}) = freq(${p})/total`,
      example: 'A(2) = 0.33 (active)',
      color: 'pink'
    },
    {
      id: 10,
      name: 'Complex Transform',
      symbol: 'Φ₁₀',
      description: 'Applies imaginary operations',
      operation: (n: number) => `ℂ(${n}) = ${n} + i·Φ(${n})`,
      example: 'ℂ(5) = 5 + 4i',
      color: 'violet'
    }
  ];

  const getColorClass = (color: string) => {
    const colorMap: Record<string, string> = {
      cyan: 'bg-cyan-900/30 border-cyan-700 text-cyan-400',
      purple: 'bg-purple-900/30 border-purple-700 text-purple-400',
      green: 'bg-green-900/30 border-green-700 text-green-400',
      orange: 'bg-orange-900/30 border-orange-700 text-orange-400',
      red: 'bg-red-900/30 border-red-700 text-red-400',
      yellow: 'bg-yellow-900/30 border-yellow-700 text-yellow-400',
      blue: 'bg-blue-900/30 border-blue-700 text-blue-400',
      indigo: 'bg-indigo-900/30 border-indigo-700 text-indigo-400',
      pink: 'bg-pink-900/30 border-pink-700 text-pink-400',
      violet: 'bg-violet-900/30 border-violet-700 text-violet-400'
    };
    return colorMap[color] || 'bg-gray-900/30 border-gray-700 text-gray-400';
  };

  const handleExecuteOperator = () => {
    setExecutionStep((prev) => (prev + 1) % 11); // 0-10 steps
  };

  const currentOperator = primeOperators.find(op => op.id === activeOperator);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Zap className="text-yellow-400" size={24} />
          <span>3.12: Ten Prime Operators</span>
        </h2>
        <button
          onClick={handleExecuteOperator}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg hover:from-cyan-600 hover:to-purple-600 transition-colors"
        >
          <Play size={16} />
          <span className="text-sm font-semibold">Execute Step {executionStep + 1}</span>
        </button>
      </div>

      {/* Operator Flow Diagram */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">Sequential Operator Pipeline</h3>
        <div className="flex items-center space-x-2 overflow-x-auto pb-4">
          {primeOperators.map((op, idx) => (
            <React.Fragment key={op.id}>
              <button
                onClick={() => setActiveOperator(op.id)}
                className={`flex-shrink-0 w-20 h-20 rounded-lg border-2 transition-all duration-300 flex flex-col items-center justify-center ${
                  activeOperator === op.id
                    ? getColorClass(op.color)
                    : 'bg-gray-800/30 border-gray-700 text-gray-500'
                } ${
                  idx < executionStep
                    ? 'ring-2 ring-green-400'
                    : idx === executionStep
                    ? 'ring-2 ring-yellow-400 animate-pulse'
                    : ''
                }`}
              >
                <div className="text-xs font-bold">{op.symbol}</div>
                <div className="text-[10px] text-center mt-1">{idx + 1}</div>
              </button>
              {idx < primeOperators.length - 1 && (
                <div className="text-gray-600 text-2xl">→</div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Operator Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {primeOperators.map((op) => (
          <div
            key={op.id}
            className={`rounded-lg border-2 p-4 cursor-pointer transition-all duration-300 ${
              activeOperator === op.id
                ? getColorClass(op.color)
                : 'bg-gray-800/30 border-gray-700 hover:border-gray-600'
            }`}
            onClick={() => setActiveOperator(op.id)}
          >
            <div className="text-center mb-2">
              <div className="text-2xl font-bold mb-1">{op.symbol}</div>
              <div className="text-xs font-semibold">Operator {op.id}</div>
            </div>
            <div className="text-xs text-gray-300 text-center">{op.name}</div>
          </div>
        ))}
      </div>

      {/* Active Operator Details */}
      {currentOperator && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-white font-bold text-lg">{currentOperator.name}</h3>
              <p className="text-gray-400 text-sm">{currentOperator.description}</p>
            </div>
            <div className={`text-4xl font-bold ${getColorClass(currentOperator.color).split(' ')[2]}`}>
              {currentOperator.symbol}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-3 flex items-center space-x-2">
                <Settings size={16} />
                <span>Operation Formula</span>
              </h4>
              <div className="font-mono text-sm text-purple-300 bg-gray-900/50 p-3 rounded border border-gray-700">
                {currentOperator.operation(42)}
              </div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-green-400 font-semibold mb-3 flex items-center space-x-2">
                <RotateCw size={16} />
                <span>Concrete Example</span>
              </h4>
              <div className="font-mono text-sm text-green-300 bg-gray-900/50 p-3 rounded border border-gray-700">
                {currentOperator.example}
              </div>
            </div>
          </div>

          {/* Application Pipeline */}
          <div className="mt-6 bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-4 border border-gray-700">
            <h4 className="text-yellow-400 font-semibold mb-3">Step-by-Step Application</h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center space-x-3">
                <div className="w-6 h-6 rounded-full bg-cyan-500 text-white text-xs flex items-center justify-center font-bold">
                  1
                </div>
                <span className="text-gray-300">Initialize with integer n from consciousness space</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-6 h-6 rounded-full bg-purple-500 text-white text-xs flex items-center justify-center font-bold">
                  2
                </div>
                <span className="text-gray-300">Apply {currentOperator.name} operator</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-6 h-6 rounded-full bg-green-500 text-white text-xs flex items-center justify-center font-bold">
                  3
                </div>
                <span className="text-gray-300">Transform according to {currentOperator.symbol} rules</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-6 h-6 rounded-full bg-orange-500 text-white text-xs flex items-center justify-center font-bold">
                  4
                </div>
                <span className="text-gray-300">Output result to next operator in pipeline</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-6 h-6 rounded-full bg-red-500 text-white text-xs flex items-center justify-center font-bold">
                  5
                </div>
                <span className="text-gray-300">Integrate with 11D consciousness manifold</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Complete Pipeline Example */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">Complete 10-Operator Pipeline Example</h3>
        <div className="bg-gray-800/50 rounded-lg p-4 border border-purple-700/30">
          <div className="space-y-3 font-mono text-sm">
            <div className="text-cyan-400">Input: n = 42</div>
            <div className="text-gray-300">Φ₁: 42 → 2 × 3 × 7</div>
            <div className="text-gray-300">Φ₂: OF(42) = 2·3·7 = 42</div>
            <div className="text-gray-300">Φ₃: θ(42) = 42°</div>
            <div className="text-gray-300">Φ₄: Shape(42) = [Circle, Triangle, Heptagon]</div>
            <div className="text-gray-300">Φ₅: Π₁₁(42) = [0.6, 0.3, 0.8, ...]</div>
            <div className="text-gray-300">Φ₆: R(42) = 0.7234</div>
            <div className="text-gray-300">Φ₇: L(42) = &#123;co-prime: 5, 11, 13, ...&#125;</div>
            <div className="text-gray-300">Φ₈: H(42) = 1° (no hole)</div>
            <div className="text-gray-300">Φ₉: A(42) = [A(2), A(3), A(7)]</div>
            <div className="text-purple-400">Φ₁₀: ℂ(42) = 42 + 12i</div>
            <div className="text-green-400 font-bold mt-2">→ Time Crystal Signature: TC₄₂</div>
          </div>
        </div>
      </div>
    </div>
  );
};
