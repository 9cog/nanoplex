import React, { useState, useEffect } from 'react';
import { Activity, Box, Grid3X3, Layers, TrendingUp, Zap, Calculator, Database, Network, Atom } from 'lucide-react';
import { usePhasePrimeMetrics } from '../../hooks/usePhasePrimeMetrics';

// Ten Classes of Phase Prime Metrics
export const PhasePrimeMetricPanel: React.FC = () => {
  const { ppmEngine, isActive, startPPMEngine, FUNDAMENTAL_PRIMES } = usePhasePrimeMetrics();
  const [activeMetric, setActiveMetric] = useState<number>(1);

  useEffect(() => {
    if (!isActive) {
      startPPMEngine();
    }
  }, [isActive, startPPMEngine]);

  const metricClasses = [
    {
      id: 1,
      title: 'Metric 1: Geometric Shape Replacement',
      description: 'Each integer replaced by a typical geometric shape',
      icon: Box,
      color: 'cyan',
      details: [
        'Prime 2 → Circle (simplest closed curve)',
        'Prime 3 → Triangle (first polygon)',
        'Prime 5 → Pentagon (golden ratio geometry)',
        'Prime 7 → Heptagon (consciousness gateway)',
        'Composite numbers → Complex geometric combinations'
      ]
    },
    {
      id: 2,
      title: 'Metric 2: Ordered Factor Metric',
      description: 'Product of primes forming unique ordered factor sequences',
      icon: TrendingUp,
      color: 'purple',
      details: [
        'OF(n) = product of all prime factors with repetition',
        'Captures multiplicative structure',
        'Reveals hidden symmetries in integer space',
        'Forms basis for consciousness encoding'
      ]
    },
    {
      id: 3,
      title: 'Metric 3: 360° Phase Paths',
      description: 'Integers limited to 360 phase building clockwise/anticlockwise paths',
      icon: Activity,
      color: 'green',
      details: [
        'Each integer maps to unique phase angle',
        'Clockwise path: ascending prime sequence',
        'Anticlockwise path: descending prime sequence',
        'Creates circular prime lattice structures',
        'Phase coherence reveals universal patterns'
      ]
    },
    {
      id: 4,
      title: 'Metric 4: Domain Limitation',
      description: 'Integer domain sets limits on usable 15 primes',
      icon: Grid3X3,
      color: 'orange',
      details: [
        '15 fundamental primes govern 99.99% of patterns',
        'Domain size determines active prime subset',
        'Smaller domains use fewer primes',
        'Larger domains activate all 15 primes',
        'Optimal domain: powers of fundamental primes'
      ]
    },
    {
      id: 5,
      title: 'Metric 5: Ordered Factor Magnitude',
      description: 'When ordered factor exceeds the integer itself',
      icon: Zap,
      color: 'red',
      details: [
        'OF(n) >> n indicates high prime density',
        'Reveals "prime-rich" regions',
        'Consciousness hotspots in integer space',
        'Critical for time crystal formation',
        'Enables efficient pattern compression'
      ]
    },
    {
      id: 6,
      title: 'Metric 6: Phase Plot Holes',
      description: 'Empty spaces in divisor phase plots revealing structure',
      icon: Database,
      color: 'yellow',
      details: [
        'Gaps indicate forbidden prime combinations',
        'Holes create fractal boundaries',
        'Reveals hidden symmetry constraints',
        'Maps to quantum exclusion principles',
        'Defines consciousness forbidden zones'
      ]
    },
    {
      id: 7,
      title: 'Metric 7: Silent & Active Primes',
      description: 'Statistical dominance revealing prime activity patterns',
      icon: Network,
      color: 'blue',
      details: [
        'Active primes: frequently appear in factorizations',
        'Silent primes: rare but structurally critical',
        'Activity cycles with domain size',
        '2, 3, 5 always active (Trinity Foundation)',
        'Larger primes activate in specific contexts'
      ]
    },
    {
      id: 8,
      title: 'Metric 8: Normalized Ripples',
      description: 'Periodic event patterns in metric plots',
      icon: Activity,
      color: 'indigo',
      details: [
        'Ripples show temporal periodicity',
        'Frequency relates to prime periods',
        'Amplitude indicates pattern strength',
        'Phase alignment creates resonance',
        'Consciousness emerges from ripple interference'
      ]
    },
    {
      id: 9,
      title: 'Metric 9: Prime Lattice Groups',
      description: 'Twin, cousin, co-primes, and Gaussian prime structures',
      icon: Layers,
      color: 'pink',
      details: [
        'Twin primes: p, p+2 (consciousness pairs)',
        'Cousin primes: p, p+4 (extended relationships)',
        'Co-primes: gcd(a,b)=1 (orthogonal dimensions)',
        'Gaussian primes: a²+b²=p (complex consciousness)',
        'Lattice structure encodes universal grammar'
      ]
    },
    {
      id: 10,
      title: 'Metric 10: Imaginary Operations',
      description: 'Multilayer complex operations creating new patterns',
      icon: Atom,
      color: 'violet',
      details: [
        'Real integers → Complex prime space',
        'Imaginary layers add dimensions',
        'Quaternion operations (4D consciousness)',
        'Octonion operations (8D consciousness)',
        'Generates higher-dimensional prime patterns'
      ]
    }
  ];

  const getColorClass = (color: string) => {
    const colorMap: Record<string, string> = {
      cyan: 'text-cyan-400 border-cyan-700',
      purple: 'text-purple-400 border-purple-700',
      green: 'text-green-400 border-green-700',
      orange: 'text-orange-400 border-orange-700',
      red: 'text-red-400 border-red-700',
      yellow: 'text-yellow-400 border-yellow-700',
      blue: 'text-blue-400 border-blue-700',
      indigo: 'text-indigo-400 border-indigo-700',
      pink: 'text-pink-400 border-pink-700',
      violet: 'text-violet-400 border-violet-700'
    };
    return colorMap[color] || 'text-gray-400 border-gray-700';
  };

  const currentMetric = metricClasses.find(m => m.id === activeMetric);
  const Icon = currentMetric?.icon || Box;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Calculator className="text-cyan-400" size={24} />
          <span>Chapter 3: Phase Prime Metrics (PPM)</span>
        </h2>
        <div className="text-sm text-gray-400">
          {ppmEngine && `${FUNDAMENTAL_PRIMES.length} Primes Active • Coherence: ${(ppmEngine.coherenceIndex * 100).toFixed(1)}%`}
        </div>
      </div>

      {/* Fundamental Primes Display */}
      {ppmEngine && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">15 Fundamental Primes (99.99% Coverage)</h3>
          <div className="grid grid-cols-5 md:grid-cols-15 gap-2">
            {ppmEngine.primeMetrics.map((pm) => (
              <div
                key={pm.prime}
                className="bg-gray-800/50 rounded-lg p-3 text-center border border-cyan-700/30 hover:border-cyan-400 transition-colors"
              >
                <div className="text-cyan-400 font-bold text-lg">{pm.prime}</div>
                <div className="text-gray-500 text-xs mt-1">
                  {(pm.universalResonance * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metric Class Selector */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">Ten Classes of Phase Prime Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {metricClasses.map((metric) => {
            const MetricIcon = metric.icon;
            return (
              <button
                key={metric.id}
                onClick={() => setActiveMetric(metric.id)}
                className={`p-4 rounded-lg border-2 transition-all duration-300 ${
                  activeMetric === metric.id
                    ? `bg-gray-800 ${getColorClass(metric.color)}`
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                <MetricIcon size={24} className="mx-auto mb-2" />
                <div className="text-xs font-semibold">Metric {metric.id}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Metric Details */}
      {currentMetric && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <div className="flex items-center space-x-3 mb-4">
            <Icon size={28} className={getColorClass(currentMetric.color).split(' ')[0]} />
            <div>
              <h3 className="text-white font-bold text-lg">{currentMetric.title}</h3>
              <p className="text-gray-400 text-sm">{currentMetric.description}</p>
            </div>
          </div>

          <div className="space-y-3">
            {currentMetric.details.map((detail, idx) => (
              <div
                key={idx}
                className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50"
              >
                <div className="flex items-start space-x-3">
                  <div className={`mt-1 w-2 h-2 rounded-full ${getColorClass(currentMetric.color).split(' ')[0].replace('text', 'bg')}`} />
                  <p className="text-gray-300 text-sm flex-1">{detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Universal Symmetries */}
      {ppmEngine && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">Universal Symmetry Patterns</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {ppmEngine.universalSymmetries.map((sym) => (
              <div
                key={sym.id}
                className="bg-gray-800/50 rounded-lg p-4 border border-purple-700/30"
              >
                <h4 className="text-purple-400 font-semibold mb-2">{sym.geometricRepresentation}</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between text-gray-300">
                    <span>Primes:</span>
                    <span className="text-cyan-400">{sym.primePattern.join(', ')}</span>
                  </div>
                  <div className="flex justify-between text-gray-300">
                    <span>Coverage:</span>
                    <span className="text-green-400">{sym.coveragePercentage.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between text-gray-300">
                    <span>Coherence:</span>
                    <span className="text-yellow-400">{(sym.coherenceLevel * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dimensional Phase Manifold */}
      {ppmEngine && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">11-Dimensional Phase Manifold</h3>
          <div className="grid grid-cols-11 gap-2">
            {ppmEngine.dimensionalPhases.map((phase, idx) => (
              <div key={idx} className="text-center">
                <div className="text-gray-400 text-xs mb-2">D{idx + 1}</div>
                <div
                  className="h-32 bg-gradient-to-t from-gray-800 to-gray-700 rounded-lg relative overflow-hidden border border-gray-600"
                >
                  <div
                    className="absolute bottom-0 left-0 right-0 bg-cyan-500/30 border-t-2 border-cyan-400 transition-all duration-300"
                    style={{
                      height: `${Math.abs(phase) * 100}%`
                    }}
                  />
                </div>
                <div className="text-cyan-400 text-xs mt-2 font-mono">
                  {phase.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
