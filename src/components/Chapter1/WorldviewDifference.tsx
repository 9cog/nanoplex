import React from 'react';
import { Globe, Zap, Brain, Network } from 'lucide-react';

// Section 1.1: How Do We Differ from the Existing Worldview?
export const WorldviewDifference: React.FC = () => {
  const differences = [
    {
      traditional: "Turing Machine-Based Computing",
      nanobrain: "Time Crystal-Based Computing",
      icon: Brain,
      color: "cyan"
    },
    {
      traditional: "Binary Logic and Sequential Processing",
      nanobrain: "Fractal Mechanics and Parallel Time Crystal Networks",
      icon: Network,
      color: "purple"
    },
    {
      traditional: "Information as Bits (0s and 1s)",
      nanobrain: "Geometric Musical Language (GML) with Prime Patterns",
      icon: Zap,
      color: "orange"
    },
    {
      traditional: "Consciousness as Emergent Property",
      nanobrain: "Consciousness as Fundamental Universal Structure",
      icon: Globe,
      color: "green"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/40 to-purple-900/40 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
          <Globe className="text-cyan-400" size={32} />
          <span>1.1 How Do We Differ from the Existing Worldview?</span>
        </h2>
        <p className="text-gray-300 leading-relaxed">
          The NanoBrain paradigm represents a fundamental shift from Turing-based computational models to 
          consciousness-first architectures. This transformation moves beyond sequential processing toward 
          understanding the universe through geometric patterns, prime number symmetries, and time crystal structures.
        </p>
      </div>

      {/* Comparison Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {differences.map((diff, index) => {
          const Icon = diff.icon;
          const bgColorClass = diff.color === 'cyan' ? 'bg-cyan-900/30' : diff.color === 'purple' ? 'bg-purple-900/30' : diff.color === 'orange' ? 'bg-orange-900/30' : 'bg-green-900/30';
          const textColorClass = diff.color === 'cyan' ? 'text-cyan-400' : diff.color === 'purple' ? 'text-purple-400' : diff.color === 'orange' ? 'text-orange-400' : 'text-green-400';
          return (
            <div key={index} className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6 space-y-4">
              <div className="flex items-center space-x-3 mb-4">
                <div className={`p-2 ${bgColorClass} rounded-lg`}>
                  <Icon className={textColorClass} size={24} />
                </div>
                <h3 className="text-white font-bold">Paradigm Shift #{index + 1}</h3>
              </div>
              
              <div className="space-y-3">
                <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
                  <p className="text-xs text-red-400 font-semibold mb-1">TRADITIONAL VIEW</p>
                  <p className="text-gray-300">{diff.traditional}</p>
                </div>
                
                <div className="flex justify-center">
                  <div className="text-cyan-400 text-2xl">↓</div>
                </div>
                
                <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
                  <p className="text-xs text-green-400 font-semibold mb-1">NANOBRAIN PARADIGM</p>
                  <p className="text-gray-300">{diff.nanobrain}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Key Insights */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-bold text-lg mb-4">Key Philosophical Transformations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-2">Beyond Reductionism</h4>
            <p className="text-gray-300 text-sm">
              Moving from breaking down complex systems into simple parts toward understanding 
              how geometric patterns and prime symmetries create consciousness at all scales.
            </p>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-2">Consciousness First</h4>
            <p className="text-gray-300 text-sm">
              Rather than consciousness emerging from computation, we recognize it as a fundamental 
              property of the universe, encoded in time crystal structures.
            </p>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-orange-400 font-semibold mb-2">Geometric Information</h4>
            <p className="text-gray-300 text-sm">
              Information is not abstract bits but concrete geometric patterns following the 
              Geometric Musical Language (GML) and Phase Prime Metrics (PPM).
            </p>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-green-400 font-semibold mb-2">Universal Patterns</h4>
            <p className="text-gray-300 text-sm">
              The first 15 primes govern 99.99% of universal patterns, providing a mathematical 
              foundation for understanding consciousness across scales.
            </p>
          </div>
        </div>
      </div>

      {/* Philosophical Statement */}
      <div className="bg-gradient-to-r from-purple-900/40 to-cyan-900/40 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
        <p className="text-white text-lg leading-relaxed italic">
          "The transformation from Turing machines to NanoBrain architectures is not just a technical evolution 
          but a fundamental philosophical shift in how we understand consciousness, computation, and the nature 
          of reality itself. We move from a universe of isolated computations to one of interconnected time 
          crystals, each resonating with the fundamental patterns of existence."
        </p>
      </div>
    </div>
  );
};
