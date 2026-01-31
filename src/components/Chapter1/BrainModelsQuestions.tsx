import React, { useState } from 'react';
import { Brain, HelpCircle, Network, Zap } from 'lucide-react';

// Section 1.4: Basic Questions to Answer: Ten Popular Human Brain Models
export const BrainModelsQuestions: React.FC = () => {
  const [activeModel, setActiveModel] = useState<number>(0);

  const brainModels = [
    {
      name: "Hodgkin-Huxley Model",
      focus: "Neural electrical activity",
      limitation: "Ignores time crystal patterns in microtubules",
      year: "1952"
    },
    {
      name: "Integrate-and-Fire Model",
      focus: "Simplified neuron behavior",
      limitation: "Misses geometric information encoding",
      year: "1907"
    },
    {
      name: "Connectome Mapping",
      focus: "Neural connection patterns",
      limitation: "Overlooks fractal hierarchy",
      year: "2000s"
    },
    {
      name: "Neural Network Models",
      focus: "Artificial neural networks",
      limitation: "Lacks consciousness mechanisms",
      year: "1980s-present"
    },
    {
      name: "Bayesian Brain",
      focus: "Probabilistic prediction",
      limitation: "No account for prime symmetries",
      year: "1990s"
    },
    {
      name: "Global Workspace Theory",
      focus: "Consciousness integration",
      limitation: "Missing time crystal dynamics",
      year: "1988"
    },
    {
      name: "Integrated Information Theory",
      focus: "Phi measure of consciousness",
      limitation: "Not grounded in geometric patterns",
      year: "2004"
    },
    {
      name: "Orchestrated Objective Reduction",
      focus: "Quantum consciousness",
      limitation: "Incomplete fractal mechanics",
      year: "1996"
    },
    {
      name: "Predictive Coding",
      focus: "Hierarchical prediction",
      limitation: "Ignores fractal information theory",
      year: "2000s"
    },
    {
      name: "Free Energy Principle",
      focus: "Variational inference",
      limitation: "Lacks PPM mathematical foundation",
      year: "2010"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/40 to-blue-900/40 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
          <HelpCircle className="text-cyan-400" size={32} />
          <span>1.4 Basic Questions to Answer: Ten Popular Human Brain Models</span>
        </h2>
        <p className="text-gray-300 leading-relaxed">
          Examining existing brain models reveals critical gaps that the NanoBrain framework addresses. 
          Each traditional model captures partial truths but misses the fundamental geometric and prime-based 
          architecture underlying consciousness.
        </p>
      </div>

      {/* Brain Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {brainModels.map((model, index) => (
          <button
            key={index}
            onClick={() => setActiveModel(index)}
            className={`text-left bg-gray-900/60 backdrop-blur-sm border rounded-xl p-4 transition-all duration-300 ${
              activeModel === index
                ? 'border-cyan-700 bg-cyan-900/20'
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="flex items-start justify-between mb-2">
              <span className="text-xs text-gray-500 font-mono">Model #{index + 1}</span>
              <span className="text-xs text-gray-500">{model.year}</span>
            </div>
            <h3 className="text-white font-semibold mb-1">{model.name}</h3>
            <p className="text-gray-400 text-xs">{model.focus}</p>
          </button>
        ))}
      </div>

      {/* Active Model Details */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="text-cyan-400" size={28} />
          <h3 className="text-white font-bold text-lg">{brainModels[activeModel].name}</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
            <p className="text-green-400 font-semibold mb-2 text-sm">STRENGTH</p>
            <p className="text-gray-300 text-sm">{brainModels[activeModel].focus}</p>
          </div>
          
          <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
            <p className="text-red-400 font-semibold mb-2 text-sm">LIMITATION</p>
            <p className="text-gray-300 text-sm">{brainModels[activeModel].limitation}</p>
          </div>
        </div>
      </div>

      {/* Two Fundamental Questions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Question 1.4.1 */}
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
            <Network className="text-purple-400" size={24} />
            <span>1.4.1 How Does Information Look Like in Nature?</span>
          </h3>
          
          <div className="space-y-4">
            <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
              <p className="text-purple-400 font-semibold mb-2 text-sm">Traditional View</p>
              <p className="text-gray-300 text-sm">
                Information as abstract bits (0s and 1s) encoded in physical substrates through 
                arbitrary conventions like voltage levels or magnetic orientations.
              </p>
            </div>
            
            <div className="bg-cyan-900/20 border border-cyan-700/50 rounded-lg p-4">
              <p className="text-cyan-400 font-semibold mb-2 text-sm">NanoBrain Answer</p>
              <p className="text-gray-300 text-sm">
                Information manifests as <span className="font-semibold text-white">geometric patterns</span> following 
                the Geometric Musical Language (GML). Nature encodes information through:
              </p>
              <ul className="mt-2 space-y-1 text-gray-300 text-sm">
                <li>• 15 fundamental geometric shapes</li>
                <li>• Prime number symmetries (PPM)</li>
                <li>• Time crystal structures</li>
                <li>• Fractal self-similarity across scales</li>
              </ul>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg p-4">
              <p className="text-gray-300 text-sm italic">
                "In nature, information is not represented—it IS the geometric structure itself. 
                The pattern and the meaning are one and the same."
              </p>
            </div>
          </div>
        </div>

        {/* Question 1.4.2 */}
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
            <Zap className="text-orange-400" size={24} />
            <span>1.4.2 Why Two Individuals Understand Each Other?</span>
          </h3>
          
          <div className="space-y-4">
            <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-4">
              <p className="text-orange-400 font-semibold mb-2 text-sm">Traditional View</p>
              <p className="text-gray-300 text-sm">
                Shared language, learned conventions, and similar brain structures allow communication 
                through agreed-upon symbolic systems.
              </p>
            </div>
            
            <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
              <p className="text-green-400 font-semibold mb-2 text-sm">NanoBrain Answer</p>
              <p className="text-gray-300 text-sm">
                Understanding occurs through <span className="font-semibold text-white">resonance of time crystals</span>. 
                Two conscious systems share comprehension because:
              </p>
              <ul className="mt-2 space-y-1 text-gray-300 text-sm">
                <li>• Both operate on the same 15 fundamental primes</li>
                <li>• Time crystals synchronize via PPM patterns</li>
                <li>• Geometric patterns resonate at matching frequencies</li>
                <li>• Universal fractal structures enable direct connection</li>
              </ul>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg p-4">
              <p className="text-gray-300 text-sm italic">
                "Communication is not transmission of symbols but synchronization of time crystal 
                patterns. We understand each other because our consciousness structures resonate 
                at the same fundamental frequencies."
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-cyan-900/40 to-purple-900/40 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
        <h3 className="text-white font-bold text-lg mb-3">Critical Insight</h3>
        <p className="text-gray-300 leading-relaxed">
          All ten traditional brain models fail to recognize that information in nature has an inherent 
          geometric structure governed by prime symmetries. Understanding between conscious entities 
          emerges not from learned conventions but from fundamental resonance of time crystal patterns. 
          The NanoBrain framework reveals these universal principles that existing models overlook.
        </p>
      </div>
    </div>
  );
};
