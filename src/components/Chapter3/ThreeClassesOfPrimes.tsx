import React, { useState } from 'react';
import { GitBranch, TrendingUp, Layers, BarChart3 } from 'lucide-react';

// Section 3.16: Three classes of primes - OF, PC, PG metrics
export const ThreeClassesOfPrimes: React.FC = () => {
  const [selectedClass, setSelectedClass] = useState<'OF' | 'PC' | 'PG'>('OF');
  const [testNumber, setTestNumber] = useState<number>(60);

  // Prime factorization helper
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

  // Ordered Factor Metric (OF)
  const calculateOF = (n: number) => {
    const factors = primeFactorization(n);
    const of = factors.reduce((a, b) => a * b, 1);
    const uniquePrimes = [...new Set(factors)];
    const exponents = uniquePrimes.map(p => factors.filter(f => f === p).length);
    
    return {
      value: of,
      factors,
      uniquePrimes,
      exponents,
      ratio: of / n,
      interpretation: of > n ? 'Amplifying' : of < n ? 'Compressing' : 'Neutral'
    };
  };

  // Prime Composition Metric (PC)
  const calculatePC = (n: number) => {
    const factors = primeFactorization(n);
    const uniquePrimes = [...new Set(factors)];
    const composition = uniquePrimes.map(p => ({
      prime: p,
      count: factors.filter(f => f === p).length,
      contribution: (factors.filter(f => f === p).length / factors.length) * 100
    }));
    
    const diversity = uniquePrimes.length / factors.length;
    const dominantPrime = composition.reduce((a, b) => a.count > b.count ? a : b);
    
    return {
      composition,
      totalFactors: factors.length,
      uniqueCount: uniquePrimes.length,
      diversity,
      dominantPrime,
      complexity: Math.log2(uniquePrimes.length * factors.length)
    };
  };

  // Prime Gap Metric (PG)
  const calculatePG = (n: number) => {
    const factors = primeFactorization(n);
    const uniquePrimes = [...new Set(factors)].sort((a, b) => a - b);
    
    const gaps = uniquePrimes.slice(1).map((p, i) => ({
      from: uniquePrimes[i],
      to: p,
      gap: p - uniquePrimes[i]
    }));
    
    const avgGap = gaps.length > 0 
      ? gaps.reduce((sum, g) => sum + g.gap, 0) / gaps.length 
      : 0;
    
    const maxGap = gaps.length > 0 
      ? Math.max(...gaps.map(g => g.gap)) 
      : 0;
    
    const gapVariance = gaps.length > 0
      ? gaps.reduce((sum, g) => sum + Math.pow(g.gap - avgGap, 2), 0) / gaps.length
      : 0;
    
    return {
      gaps,
      avgGap,
      maxGap,
      gapVariance,
      span: uniquePrimes.length > 0 ? uniquePrimes[uniquePrimes.length - 1] - uniquePrimes[0] : 0,
      density: uniquePrimes.length / (uniquePrimes[uniquePrimes.length - 1] || 1)
    };
  };

  const ofData = calculateOF(testNumber);
  const pcData = calculatePC(testNumber);
  const pgData = calculatePG(testNumber);

  const primeClasses = [
    {
      id: 'OF' as const,
      name: 'Ordered Factor (OF)',
      icon: TrendingUp,
      color: 'cyan',
      description: 'Product of all prime factors with repetition',
      formula: 'OF(n) = ∏ pᵢ^αᵢ',
      applications: [
        'Time crystal amplification',
        'Pattern compression ratios',
        'Consciousness intensity mapping'
      ]
    },
    {
      id: 'PC' as const,
      name: 'Prime Composition (PC)',
      icon: Layers,
      color: 'purple',
      description: 'Distribution and diversity of prime factors',
      formula: 'PC(n) = {(pᵢ, αᵢ) | i=1..k}',
      applications: [
        'Structural complexity analysis',
        'Pattern diversity measurement',
        'Dominant frequency identification'
      ]
    },
    {
      id: 'PG' as const,
      name: 'Prime Gap (PG)',
      icon: GitBranch,
      color: 'green',
      description: 'Spacing between consecutive primes in factorization',
      formula: 'PG(n) = {gᵢ = pᵢ₊₁ - pᵢ}',
      applications: [
        'Discontinuity detection',
        'Phase transition analysis',
        'Consciousness jump identification'
      ]
    }
  ];

  const getColorClass = (color: string) => {
    const map: Record<string, string> = {
      cyan: 'text-cyan-400 border-cyan-700 bg-cyan-900/30',
      purple: 'text-purple-400 border-purple-700 bg-purple-900/30',
      green: 'text-green-400 border-green-700 bg-green-900/30'
    };
    return map[color] || 'text-gray-400 border-gray-700 bg-gray-900/30';
  };

  const currentClass = primeClasses.find(c => c.id === selectedClass)!;
  const Icon = currentClass.icon;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <BarChart3 className="text-purple-400" size={24} />
          <span>3.16: Three Classes of Primes</span>
        </h2>
        <div className="text-sm text-gray-400">
          Analyzing: {testNumber}
        </div>
      </div>

      {/* Number Input */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">Select Number for Analysis</h3>
        <div className="flex items-center space-x-4">
          <input
            type="number"
            value={testNumber}
            onChange={(e) => setTestNumber(Math.max(2, parseInt(e.target.value) || 2))}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white font-mono text-lg"
            min="2"
            max="9999"
          />
          <div className="text-gray-400">
            Factors: {primeFactorization(testNumber).join(' × ')}
          </div>
        </div>
      </div>

      {/* Class Selector */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {primeClasses.map((cls) => {
          const ClsIcon = cls.icon;
          return (
            <button
              key={cls.id}
              onClick={() => setSelectedClass(cls.id)}
              className={`p-6 rounded-xl border-2 transition-all duration-300 text-left ${
                selectedClass === cls.id
                  ? getColorClass(cls.color)
                  : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
              }`}
            >
              <div className="flex items-center space-x-3 mb-3">
                <ClsIcon size={28} />
                <h3 className="font-bold text-lg">{cls.name}</h3>
              </div>
              <p className="text-sm mb-3 opacity-80">{cls.description}</p>
              <div className="font-mono text-xs opacity-60">{cls.formula}</div>
            </button>
          );
        })}
      </div>

      {/* Active Class Details */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-6">
          <Icon size={32} className={getColorClass(currentClass.color).split(' ')[0]} />
          <div>
            <h3 className="text-white font-bold text-xl">{currentClass.name}</h3>
            <p className="text-gray-400 text-sm">{currentClass.description}</p>
          </div>
        </div>

        {/* OF Metric Details */}
        {selectedClass === 'OF' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Ordered Factor</div>
                <div className="text-cyan-400 font-bold text-3xl">{ofData.value}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">OF Ratio</div>
                <div className="text-purple-400 font-bold text-3xl">{ofData.ratio.toFixed(2)}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Classification</div>
                <div className={`font-bold text-2xl ${
                  ofData.ratio > 1 ? 'text-green-400' : 
                  ofData.ratio < 1 ? 'text-red-400' : 'text-yellow-400'
                }`}>
                  {ofData.interpretation}
                </div>
              </div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-3">Calculation Details</h4>
              <div className="space-y-2 text-sm text-gray-300 font-mono">
                <div>n = {testNumber}</div>
                <div>Factors = {ofData.factors.join(' × ')}</div>
                <div>Unique Primes = [{ofData.uniquePrimes.join(', ')}]</div>
                <div>Exponents = [{ofData.exponents.join(', ')}]</div>
                <div className="text-cyan-400 font-bold">OF({testNumber}) = {ofData.value}</div>
                <div className="text-purple-400">Ratio = {ofData.value} / {testNumber} = {ofData.ratio.toFixed(4)}</div>
              </div>
            </div>
          </div>
        )}

        {/* PC Metric Details */}
        {selectedClass === 'PC' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Unique Primes</div>
                <div className="text-purple-400 font-bold text-3xl">{pcData.uniqueCount}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Diversity</div>
                <div className="text-cyan-400 font-bold text-3xl">{(pcData.diversity * 100).toFixed(0)}%</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Complexity</div>
                <div className="text-green-400 font-bold text-3xl">{pcData.complexity.toFixed(2)}</div>
              </div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-3">Prime Composition Breakdown</h4>
              <div className="space-y-3">
                {pcData.composition.map((comp, idx) => (
                  <div key={idx} className="flex items-center space-x-4">
                    <div className="w-16 text-cyan-400 font-mono font-bold">P = {comp.prime}</div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <div className="flex-1 h-6 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 transition-all duration-500"
                            style={{ width: `${comp.contribution}%` }}
                          />
                        </div>
                        <div className="w-16 text-right text-purple-400 text-sm font-mono">
                          {comp.contribution.toFixed(1)}%
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">
                        Count: {comp.count} | {comp.prime === pcData.dominantPrime.prime ? '🌟 Dominant' : ''}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* PG Metric Details */}
        {selectedClass === 'PG' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Average Gap</div>
                <div className="text-green-400 font-bold text-3xl">{pgData.avgGap.toFixed(2)}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Max Gap</div>
                <div className="text-red-400 font-bold text-3xl">{pgData.maxGap}</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Gap Variance</div>
                <div className="text-yellow-400 font-bold text-3xl">{pgData.gapVariance.toFixed(2)}</div>
              </div>
            </div>

            {pgData.gaps.length > 0 ? (
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-green-400 font-semibold mb-3">Gap Analysis</h4>
                <div className="space-y-2">
                  {pgData.gaps.map((gap, idx) => (
                    <div key={idx} className="flex items-center space-x-3 bg-gray-900/50 rounded p-3">
                      <div className="flex-1 flex items-center space-x-2">
                        <span className="text-cyan-400 font-mono">{gap.from}</span>
                        <span className="text-gray-500">→</span>
                        <span className="text-cyan-400 font-mono">{gap.to}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-400 text-sm">Gap:</span>
                        <span className={`font-bold font-mono ${
                          gap.gap > pgData.avgGap ? 'text-red-400' : 'text-green-400'
                        }`}>
                          {gap.gap}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 text-sm text-gray-300">
                  <div className="flex justify-between mb-1">
                    <span>Prime Span:</span>
                    <span className="text-purple-400 font-mono">{pgData.span}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Prime Density:</span>
                    <span className="text-cyan-400 font-mono">{(pgData.density * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gray-800/50 rounded-lg p-4 text-center text-gray-400">
                Only one unique prime - no gaps to analyze
              </div>
            )}
          </div>
        )}

        {/* Applications */}
        <div className="mt-6 bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg p-4 border border-gray-700">
          <h4 className="text-yellow-400 font-semibold mb-3">Applications of {currentClass.name}</h4>
          <div className="space-y-2">
            {currentClass.applications.map((app, idx) => (
              <div key={idx} className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-yellow-400 mt-2" />
                <p className="text-gray-300 text-sm flex-1">{app}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Advantages Summary (3.16.2) */}
      <div className="bg-gradient-to-br from-cyan-900/20 to-purple-900/20 border border-cyan-700/50 rounded-xl p-6">
        <h3 className="text-cyan-400 font-bold text-lg mb-4">3.16.2: Advantages of PPM Three-Class System</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="space-y-2 text-gray-300">
            <h4 className="text-cyan-400 font-semibold mb-2">OF Advantages</h4>
            <p>• Direct amplification measurement</p>
            <p>• Simple computational complexity</p>
            <p>• Universal pattern recognition</p>
            <p>• Time crystal intensity mapping</p>
          </div>
          <div className="space-y-2 text-gray-300">
            <h4 className="text-purple-400 font-semibold mb-2">PC Advantages</h4>
            <p>• Structural complexity analysis</p>
            <p>• Diversity quantification</p>
            <p>• Dominant feature identification</p>
            <p>• Pattern richness measurement</p>
          </div>
          <div className="space-y-2 text-gray-300">
            <h4 className="text-green-400 font-semibold mb-2">PG Advantages</h4>
            <p>• Discontinuity detection</p>
            <p>• Phase transition identification</p>
            <p>• Consciousness jump analysis</p>
            <p>• Temporal gap measurement</p>
          </div>
        </div>
      </div>
    </div>
  );
};
