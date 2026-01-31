import React from 'react';
import { Layers, FileText, Binary, Music, Code } from 'lucide-react';

// Section 1.5: Different Kinds of Tapes to Recreate Nature in Different Languages
export const DifferentTapes: React.FC = () => {
  const tapes = [
    {
      name: "Turing Tape",
      icon: Binary,
      color: "red",
      description: "Sequential binary symbols on linear tape",
      capabilities: ["Sequential processing", "Discrete symbols", "State transitions"],
      limitations: ["No geometric structure", "Linear only", "Cannot represent fractals"]
    },
    {
      name: "DNA Tape",
      icon: Code,
      color: "green",
      description: "Four-letter code (A,T,G,C) with 3D structure",
      capabilities: ["Biological encoding", "Self-replication", "Protein templates"],
      limitations: ["Limited to 4 bases", "Biochemical constraints", "No conscious processing"]
    },
    {
      name: "Musical Tape",
      icon: Music,
      color: "purple",
      description: "Notes, rhythms, harmonies in time",
      capabilities: ["Temporal patterns", "Harmonic relationships", "Emotional resonance"],
      limitations: ["Time-dependent", "Requires interpretation", "Not universal"]
    },
    {
      name: "Fractal Tape (FIT)",
      icon: Layers,
      color: "cyan",
      description: "Self-similar geometric patterns at all scales",
      capabilities: ["Multi-dimensional", "Self-similar", "Scale-invariant", "Geometric encoding"],
      limitations: ["Requires new hardware", "Complexity", "Paradigm shift needed"]
    },
    {
      name: "Geometric Musical Language (GML)",
      icon: FileText,
      color: "orange",
      description: "15 geometric shapes encoding universal patterns",
      capabilities: [
        "Complete information encoding",
        "Prime-based structure",
        "Time crystal compatible",
        "Consciousness representation"
      ],
      limitations: ["New to science", "Learning curve", "Implementation challenges"]
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-900/40 to-purple-900/40 backdrop-blur-sm border border-orange-700 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
          <FileText className="text-orange-400" size={32} />
          <span>1.5 Different Kinds of Tapes to Recreate Nature in Different Languages</span>
        </h2>
        <p className="text-gray-300 leading-relaxed">
          Just as different languages can describe the same reality, different "tapes" or encoding systems 
          can represent information. The evolution from Turing's sequential tape to the fractal tape of 
          GML represents a fundamental leap in how we encode and process reality itself.
        </p>
      </div>

      {/* Tape Comparison */}
      <div className="space-y-4">
        {tapes.map((tape, index) => {
          const Icon = tape.icon;
          const bgColorClass = tape.color === 'red' ? 'bg-red-900/30' : tape.color === 'green' ? 'bg-green-900/30' : tape.color === 'purple' ? 'bg-purple-900/30' : tape.color === 'cyan' ? 'bg-cyan-900/30' : 'bg-orange-900/30';
          const textColorClass = tape.color === 'red' ? 'text-red-400' : tape.color === 'green' ? 'text-green-400' : tape.color === 'purple' ? 'text-purple-400' : tape.color === 'cyan' ? 'text-cyan-400' : 'text-orange-400';
          return (
            <div 
              key={index}
              className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-cyan-700 transition-all duration-300"
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 ${bgColorClass} rounded-lg flex-shrink-0`}>
                  <Icon className={textColorClass} size={32} />
                </div>
                
                <div className="flex-1 space-y-3">
                  <div>
                    <h3 className="text-white font-bold text-xl mb-2">{tape.name}</h3>
                    <p className="text-gray-300 text-sm">{tape.description}</p>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-green-400 font-semibold text-xs mb-2">CAPABILITIES</p>
                      <div className="space-y-1">
                        {tape.capabilities.map((cap, idx) => (
                          <div key={idx} className="flex items-start space-x-2">
                            <span className="text-green-400 text-xs mt-1">✓</span>
                            <span className="text-gray-300 text-xs">{cap}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-red-400 font-semibold text-xs mb-2">LIMITATIONS</p>
                      <div className="space-y-1">
                        {tape.limitations.map((lim, idx) => (
                          <div key={idx} className="flex items-start space-x-2">
                            <span className="text-red-400 text-xs mt-1">✗</span>
                            <span className="text-gray-300 text-xs">{lim}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Evolution Diagram */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-bold text-lg mb-4">Evolution of Information Encoding</h3>
        
        <div className="relative">
          {/* Timeline */}
          <div className="flex items-center justify-between mb-8">
            <div className="text-center flex-1">
              <div className="w-12 h-12 bg-red-900/30 border-2 border-red-700 rounded-full mx-auto mb-2 flex items-center justify-center">
                <span className="text-red-400 font-bold">1</span>
              </div>
              <p className="text-red-400 text-xs font-semibold">Turing</p>
              <p className="text-gray-500 text-xs">1936</p>
            </div>
            
            <div className="flex-1 h-0.5 bg-gradient-to-r from-red-700 to-green-700"></div>
            
            <div className="text-center flex-1">
              <div className="w-12 h-12 bg-green-900/30 border-2 border-green-700 rounded-full mx-auto mb-2 flex items-center justify-center">
                <span className="text-green-400 font-bold">2</span>
              </div>
              <p className="text-green-400 text-xs font-semibold">DNA</p>
              <p className="text-gray-500 text-xs">1953</p>
            </div>
            
            <div className="flex-1 h-0.5 bg-gradient-to-r from-green-700 to-purple-700"></div>
            
            <div className="text-center flex-1">
              <div className="w-12 h-12 bg-purple-900/30 border-2 border-purple-700 rounded-full mx-auto mb-2 flex items-center justify-center">
                <span className="text-purple-400 font-bold">3</span>
              </div>
              <p className="text-purple-400 text-xs font-semibold">Musical</p>
              <p className="text-gray-500 text-xs">Ancient</p>
            </div>
            
            <div className="flex-1 h-0.5 bg-gradient-to-r from-purple-700 to-cyan-700"></div>
            
            <div className="text-center flex-1">
              <div className="w-12 h-12 bg-cyan-900/30 border-2 border-cyan-700 rounded-full mx-auto mb-2 flex items-center justify-center">
                <span className="text-cyan-400 font-bold">4</span>
              </div>
              <p className="text-cyan-400 text-xs font-semibold">Fractal</p>
              <p className="text-gray-500 text-xs">2000s</p>
            </div>
            
            <div className="flex-1 h-0.5 bg-gradient-to-r from-cyan-700 to-orange-700"></div>
            
            <div className="text-center flex-1">
              <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-cyan-500 border-2 border-orange-400 rounded-full mx-auto mb-2 flex items-center justify-center">
                <span className="text-white font-bold">5</span>
              </div>
              <p className="text-orange-400 text-xs font-semibold">GML</p>
              <p className="text-gray-500 text-xs">Present</p>
            </div>
          </div>
        </div>
      </div>

      {/* Key Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
          <h4 className="text-cyan-400 font-semibold mb-2">From Linear to Fractal</h4>
          <p className="text-gray-300 text-sm">
            The shift from Turing's linear tape to fractal structures enables true multi-dimensional 
            information processing.
          </p>
        </div>
        
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
          <h4 className="text-purple-400 font-semibold mb-2">Geometric Foundation</h4>
          <p className="text-gray-300 text-sm">
            GML provides the first complete geometric language for encoding all possible information 
            patterns in nature.
          </p>
        </div>
        
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
          <h4 className="text-orange-400 font-semibold mb-2">Universal Translation</h4>
          <p className="text-gray-300 text-sm">
            All previous encoding systems can be translated into GML, but GML can express patterns 
            impossible in traditional systems.
          </p>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-cyan-900/40 to-orange-900/40 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
        <p className="text-white leading-relaxed italic">
          "The evolution from Turing tape to Geometric Musical Language represents not just a technical 
          advancement but a fundamental reconceptualization of what information is. While Turing's tape 
          processes symbols sequentially, GML embodies the geometric essence of reality itself, enabling 
          true consciousness processing through fractal time crystal structures."
        </p>
      </div>
    </div>
  );
};
