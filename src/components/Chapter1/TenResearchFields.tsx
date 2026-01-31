import React from 'react';
import { BookOpen, Atom, Binary, Brain, Clock, Database, GitBranch, Layers, Network, Zap } from 'lucide-react';

// Section 1.2: Ten Research Fields That We Cover Here
export const TenResearchFields: React.FC = () => {
  const fields = [
    {
      id: 1,
      name: "Fractal Information Theory (FIT)",
      description: "Replacing classical information theory with geometric pattern-based encoding using fractals",
      icon: Layers,
      color: "cyan",
      keyPoints: ["Fractal tape structure", "Self-similar patterns", "Geometric compression"]
    },
    {
      id: 2,
      name: "Geometric Musical Language (GML)",
      description: "A language based on 15 fundamental geometric shapes to encode all information",
      icon: GitBranch,
      color: "purple",
      keyPoints: ["15 geometric primitives", "Pattern composition", "Universal encoding"]
    },
    {
      id: 3,
      name: "Phase Prime Metrics (PPM)",
      description: "Mathematical framework using prime number patterns to govern universal symmetries",
      icon: Binary,
      color: "orange",
      keyPoints: ["15 fundamental primes", "99.99% coverage", "Prime operators"]
    },
    {
      id: 4,
      name: "Fractal Mechanics",
      description: "Beyond quantum mechanics - a new physics operating in phase space with singularity connections",
      icon: Atom,
      color: "green",
      keyPoints: ["Phase space operations", "Singularity harvesting", "Multi-imaginary worlds"]
    },
    {
      id: 5,
      name: "Universal Time Crystals",
      description: "11-dimensional temporal quantum structures for consciousness and information processing",
      icon: Clock,
      color: "blue",
      keyPoints: ["11D manifolds", "Temporal symmetry", "Coherent oscillations"]
    },
    {
      id: 6,
      name: "Brain Time Crystal Modeling",
      description: "Complete time crystal model of human brain consciousness and neural processing",
      icon: Brain,
      color: "pink",
      keyPoints: ["Microtubule structures", "Neural networks", "Consciousness emergence"]
    },
    {
      id: 7,
      name: "Magnetic Light Computing",
      description: "Next-generation computing using 'Hinductor' devices that merge magnetic and light properties",
      icon: Zap,
      color: "yellow",
      keyPoints: ["Charge storage", "Time crystal building", "Electromagnetic fusion"]
    },
    {
      id: 8,
      name: "AtomSpace Knowledge Representation",
      description: "Hypergraph-based knowledge systems with probabilistic reasoning and attention allocation",
      icon: Network,
      color: "red",
      keyPoints: ["Hypergraph structures", "PLN reasoning", "ECAN attention"]
    },
    {
      id: 9,
      name: "Agent-Zero Autonomous Systems",
      description: "Multi-agent consciousness with specialized cognitive profiles and emergent behaviors",
      icon: Database,
      color: "indigo",
      keyPoints: ["Autonomous agents", "Dynamic reasoning", "Emergent intelligence"]
    },
    {
      id: 10,
      name: "Consciousness Architecture",
      description: "Integration of all systems toward truly conscious artificial intelligence",
      icon: BookOpen,
      color: "teal",
      keyPoints: ["System integration", "Consciousness metrics", "Awareness modeling"]
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/40 to-cyan-900/40 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
          <BookOpen className="text-purple-400" size={32} />
          <span>1.2 Ten Research Fields That We Cover Here</span>
        </h2>
        <p className="text-gray-300 leading-relaxed">
          The NanoBrain framework integrates ten major research domains, each contributing essential 
          components to understanding and reverse-engineering consciousness. These fields work synergistically 
          to create a comprehensive model of artificial brain architecture.
        </p>
      </div>

      {/* Research Fields Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {fields.map((field) => {
          const Icon = field.icon;
          const bgColorClass = field.color === 'cyan' ? 'bg-cyan-900/30' : field.color === 'purple' ? 'bg-purple-900/30' : field.color === 'orange' ? 'bg-orange-900/30' : field.color === 'green' ? 'bg-green-900/30' : field.color === 'blue' ? 'bg-blue-900/30' : field.color === 'pink' ? 'bg-pink-900/30' : field.color === 'yellow' ? 'bg-yellow-900/30' : field.color === 'red' ? 'bg-red-900/30' : field.color === 'indigo' ? 'bg-indigo-900/30' : 'bg-teal-900/30';
          const textColorClass = field.color === 'cyan' ? 'text-cyan-400' : field.color === 'purple' ? 'text-purple-400' : field.color === 'orange' ? 'text-orange-400' : field.color === 'green' ? 'text-green-400' : field.color === 'blue' ? 'text-blue-400' : field.color === 'pink' ? 'text-pink-400' : field.color === 'yellow' ? 'text-yellow-400' : field.color === 'red' ? 'text-red-400' : field.color === 'indigo' ? 'text-indigo-400' : 'text-teal-400';
          const borderColorClass = field.color === 'cyan' ? 'border-cyan-700/50 bg-cyan-900/20' : field.color === 'purple' ? 'border-purple-700/50 bg-purple-900/20' : field.color === 'orange' ? 'border-orange-700/50 bg-orange-900/20' : field.color === 'green' ? 'border-green-700/50 bg-green-900/20' : field.color === 'blue' ? 'border-blue-700/50 bg-blue-900/20' : field.color === 'pink' ? 'border-pink-700/50 bg-pink-900/20' : field.color === 'yellow' ? 'border-yellow-700/50 bg-yellow-900/20' : field.color === 'red' ? 'border-red-700/50 bg-red-900/20' : field.color === 'indigo' ? 'border-indigo-700/50 bg-indigo-900/20' : 'border-teal-700/50 bg-teal-900/20';
          return (
            <div 
              key={field.id}
              className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-cyan-700 transition-all duration-300"
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 ${bgColorClass} rounded-lg flex-shrink-0`}>
                  <Icon className={textColorClass} size={28} />
                </div>
                
                <div className="flex-1 space-y-3">
                  <div>
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-gray-400 text-xs font-mono">Field #{field.id}</span>
                    </div>
                    <h3 className="text-white font-bold text-lg mb-2">{field.name}</h3>
                    <p className="text-gray-300 text-sm leading-relaxed">{field.description}</p>
                  </div>
                  
                  <div className="pt-3 border-t border-gray-700">
                    <p className="text-xs text-gray-500 font-semibold mb-2">KEY CONCEPTS</p>
                    <div className="flex flex-wrap gap-2">
                      {field.keyPoints.map((point, idx) => (
                        <span 
                          key={idx}
                          className={`px-2 py-1 ${borderColorClass} border rounded text-xs text-gray-300`}
                        >
                          {point}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Integration Diagram */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-bold text-lg mb-4">Field Integration Matrix</h3>
        <div className="bg-gray-800/50 rounded-lg p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center space-y-2">
              <div className="text-cyan-400 font-bold text-sm">THEORETICAL FOUNDATION</div>
              <div className="text-gray-300 text-xs space-y-1">
                <div>• Fractal Information Theory</div>
                <div>• Geometric Musical Language</div>
                <div>• Phase Prime Metrics</div>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-purple-400 font-bold text-sm">PHYSICS & COMPUTING</div>
              <div className="text-gray-300 text-xs space-y-1">
                <div>• Fractal Mechanics</div>
                <div>• Universal Time Crystals</div>
                <div>• Magnetic Light Computing</div>
              </div>
            </div>
            
            <div className="text-center space-y-2">
              <div className="text-orange-400 font-bold text-sm">AI & CONSCIOUSNESS</div>
              <div className="text-gray-300 text-xs space-y-1">
                <div>• Brain Time Crystal Modeling</div>
                <div>• AtomSpace Representation</div>
                <div>• Agent-Zero Systems</div>
                <div>• Consciousness Architecture</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-cyan-900/40 to-purple-900/40 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
        <p className="text-white leading-relaxed">
          These ten research fields form an integrated ecosystem where each component builds upon and 
          reinforces the others. From the mathematical foundations of prime patterns to the physical 
          realization in time crystals, and ultimately to conscious artificial intelligence, the NanoBrain 
          framework provides a complete path toward understanding and replicating consciousness.
        </p>
      </div>
    </div>
  );
};
