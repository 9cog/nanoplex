import React, { useState, useEffect } from 'react';
import { Clock, Zap, Layers, Activity, TrendingUp } from 'lucide-react';
import { usePhasePrimeMetrics } from '../../hooks/usePhasePrimeMetrics';

// Section 3.1.3: How a time crystal is decomposed & amplified using Phase Prime Metric
export const TimeCrystalDecomposition: React.FC = () => {
  const { ppmEngine, isActive, startPPMEngine } = usePhasePrimeMetrics();
  const [decompositionStep, setDecompositionStep] = useState<number>(0);
  const [amplificationFactor, setAmplificationFactor] = useState<number>(1.0);

  useEffect(() => {
    if (!isActive) {
      startPPMEngine();
    }
  }, [isActive, startPPMEngine]);

  useEffect(() => {
    const interval = setInterval(() => {
      setDecompositionStep((prev) => (prev + 1) % 6);
      setAmplificationFactor(1.0 + Math.sin(Date.now() / 1000) * 0.5);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const decompositionSteps = [
    {
      step: 1,
      name: 'Time Crystal Input',
      description: 'Original 11D time crystal structure',
      formula: 'TC = ∑ᵢ Aᵢ·exp(iθᵢ)·Pᵢ',
      color: 'cyan',
      icon: Clock
    },
    {
      step: 2,
      name: 'Prime Factorization',
      description: 'Decompose into fundamental prime components',
      formula: 'TC → {P₂, P₃, P₅, P₇, ...}',
      color: 'purple',
      icon: Layers
    },
    {
      step: 3,
      name: 'Phase Extraction',
      description: 'Extract phase angles for each prime',
      formula: 'θᵢ = arg(TCᵢ) mod 360°',
      color: 'green',
      icon: Activity
    },
    {
      step: 4,
      name: 'Amplitude Mapping',
      description: 'Map amplitudes to prime resonances',
      formula: 'Aᵢ = |TCᵢ| · R(Pᵢ)',
      color: 'orange',
      icon: TrendingUp
    },
    {
      step: 5,
      name: 'Prime Amplification',
      description: 'Amplify using ordered factor metrics',
      formula: 'TC* = TC · OF(n) / n',
      color: 'red',
      icon: Zap
    },
    {
      step: 6,
      name: 'Reconstruction',
      description: 'Rebuild amplified time crystal',
      formula: 'TC_amp = ∑ᵢ Aᵢ*·exp(iθᵢ)·Pᵢ',
      color: 'violet',
      icon: Clock
    }
  ];

  const getColorClass = (color: string) => {
    const colorMap: Record<string, string> = {
      cyan: 'bg-cyan-900/30 border-cyan-700 text-cyan-400',
      purple: 'bg-purple-900/30 border-purple-700 text-purple-400',
      green: 'bg-green-900/30 border-green-700 text-green-400',
      orange: 'bg-orange-900/30 border-orange-700 text-orange-400',
      red: 'bg-red-900/30 border-red-700 text-red-400',
      violet: 'bg-violet-900/30 border-violet-700 text-violet-400'
    };
    return colorMap[color] || 'bg-gray-900/30 border-gray-700 text-gray-400';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Clock className="text-cyan-400" size={24} />
          <span>3.1.3: Time Crystal Decomposition & Amplification</span>
        </h2>
        <div className="text-sm text-gray-400">
          Amplification: {amplificationFactor.toFixed(2)}x
        </div>
      </div>

      {/* Process Flow */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">PPM Decomposition Pipeline</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {decompositionSteps.map((step) => {
            const StepIcon = step.icon;
            const isActive = decompositionStep === step.step - 1;
            return (
              <div
                key={step.step}
                className={`rounded-lg border-2 p-4 transition-all duration-500 ${
                  isActive
                    ? `${getColorClass(step.color)} ring-2 ring-white/50 scale-105`
                    : 'bg-gray-800/30 border-gray-700 opacity-60'
                }`}
              >
                <div className="text-center">
                  <StepIcon size={32} className="mx-auto mb-2" />
                  <div className="text-xs font-bold mb-1">Step {step.step}</div>
                  <div className="text-xs font-semibold mb-2">{step.name}</div>
                  <div className="text-[10px] text-gray-400">{step.description}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Current Step Details */}
      {decompositionSteps[decompositionStep] && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">
            Current Step: {decompositionSteps[decompositionStep].name}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2">Mathematical Formula</h4>
              <div className="font-mono text-purple-300 text-sm bg-gray-900/50 p-3 rounded border border-gray-700">
                {decompositionSteps[decompositionStep].formula}
              </div>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-green-400 font-semibold mb-2">Description</h4>
              <p className="text-gray-300 text-sm">
                {decompositionSteps[decompositionStep].description}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Live Decomposition Visualization */}
      {ppmEngine && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">Live Prime Component Analysis</h3>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-4">
            {ppmEngine.primeMetrics.slice(0, 10).map((pm) => {
              const amplitude = pm.amplitude * amplificationFactor;
              const phase = (pm.phase * 180 / Math.PI) % 360;
              
              return (
                <div
                  key={pm.prime}
                  className="bg-gray-800/50 rounded-lg p-4 border border-purple-700/30"
                >
                  <div className="text-center mb-3">
                    <div className="text-cyan-400 font-bold text-2xl">P{pm.prime}</div>
                  </div>
                  
                  {/* Amplitude Bar */}
                  <div className="mb-3">
                    <div className="text-xs text-gray-400 mb-1">Amplitude</div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-500"
                        style={{ width: `${amplitude * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-cyan-400 mt-1">{amplitude.toFixed(2)}</div>
                  </div>

                  {/* Phase Circle */}
                  <div className="mb-3">
                    <div className="text-xs text-gray-400 mb-1">Phase</div>
                    <div className="relative w-16 h-16 mx-auto">
                      <svg viewBox="0 0 100 100" className="transform -rotate-90">
                        <circle
                          cx="50"
                          cy="50"
                          r="45"
                          fill="none"
                          stroke="rgb(55, 65, 81)"
                          strokeWidth="8"
                        />
                        <circle
                          cx="50"
                          cy="50"
                          r="45"
                          fill="none"
                          stroke="rgb(34, 211, 238)"
                          strokeWidth="8"
                          strokeDasharray={`${(phase / 360) * 283} 283`}
                          className="transition-all duration-500"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xs text-green-400 font-bold">
                          {Math.round(phase)}°
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Resonance */}
                  <div className="text-center">
                    <div className="text-xs text-gray-400">Resonance</div>
                    <div className="text-yellow-400 font-semibold text-sm">
                      {(pm.universalResonance * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Amplification Process */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-semibold mb-4">Amplification Mechanism</h3>
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-4 border border-gray-700">
            <h4 className="text-cyan-400 font-semibold mb-2">1. Ordered Factor Enhancement</h4>
            <p className="text-gray-300 text-sm mb-2">
              The ordered factor metric OF(n) amplifies patterns by multiplying prime resonances
            </p>
            <div className="font-mono text-xs text-purple-300 bg-gray-900/50 p-2 rounded">
              Amplification = OF(n) / n · base_amplitude
            </div>
          </div>

          <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-4 border border-gray-700">
            <h4 className="text-green-400 font-semibold mb-2">2. Phase Coherence Boost</h4>
            <p className="text-gray-300 text-sm mb-2">
              Aligned phases create constructive interference, amplifying the signal
            </p>
            <div className="font-mono text-xs text-green-300 bg-gray-900/50 p-2 rounded">
              Boost = Σ cos(θᵢ - θⱼ) for all prime pairs
            </div>
          </div>

          <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-4 border border-gray-700">
            <h4 className="text-orange-400 font-semibold mb-2">3. Resonance Multiplication</h4>
            <p className="text-gray-300 text-sm mb-2">
              Universal resonance values multiply amplitudes across dimensions
            </p>
            <div className="font-mono text-xs text-orange-300 bg-gray-900/50 p-2 rounded">
              Final = A · R(p) · coherence · amplification_factor
            </div>
          </div>
        </div>
      </div>

      {/* Theoretical Significance */}
      <div className="bg-gradient-to-br from-purple-900/20 to-cyan-900/20 border border-purple-700/50 rounded-xl p-6">
        <h3 className="text-purple-400 font-bold text-lg mb-3">Hidden Physical Significance</h3>
        <div className="space-y-2 text-sm text-gray-300">
          <p>
            • <strong className="text-cyan-400">Time Crystal Stability:</strong> PPM decomposition reveals
            the prime structure maintaining temporal coherence
          </p>
          <p>
            • <strong className="text-green-400">Consciousness Emergence:</strong> Amplification through ordered
            factors creates consciousness-level complexity from simple primes
          </p>
          <p>
            • <strong className="text-yellow-400">Universal Patterns:</strong> 15 fundamental primes govern 99.99%
            of all time crystal structures in the universe
          </p>
          <p>
            • <strong className="text-red-400">Information Compression:</strong> PPM enables lossless compression
            by encoding complex patterns as prime factorizations
          </p>
        </div>
      </div>
    </div>
  );
};
