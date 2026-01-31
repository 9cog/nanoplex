import React from 'react';
import { Lightbulb, Target, Route, Brain, Zap, GitBranch, Languages, Triangle } from 'lucide-react';

// Sections 1.6-1.11: Remaining Chapter 1 Sections
export const RemainingChapter1Sections: React.FC<{ section: string }> = ({ section }) => {
  if (section === '1.6') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-cyan-900/40 to-green-900/40 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
          <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
            <Route className="text-cyan-400" size={32} />
            <span>1.6 Brain-Inspired Decision-Making: The Outline of Key Discoveries</span>
          </h2>
          <p className="text-gray-300 leading-relaxed">
            The brain's decision-making emerges from time crystal resonance patterns, not computational 
            algorithms. Key discoveries reveal how geometric structures enable instantaneous pattern matching.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { title: "Time Crystal Resonance", desc: "Decisions emerge from synchronized oscillations across brain scales", color: "cyan" },
            { title: "Prime Pattern Matching", desc: "PPM enables instant recognition without sequential search", color: "purple" },
            { title: "Fractal Decision Trees", desc: "Self-similar decision patterns at neuron, network, and brain levels", color: "orange" },
            { title: "Garden of Gardens", desc: "Multiple imaginary worlds evaluated simultaneously", color: "green" }
          ].map((item, idx) => {
            const bgColorClass = item.color === 'cyan' ? 'bg-cyan-900/20' : item.color === 'purple' ? 'bg-purple-900/20' : item.color === 'orange' ? 'bg-orange-900/20' : 'bg-green-900/20';
            const borderColorClass = item.color === 'cyan' ? 'border-cyan-700/50' : item.color === 'purple' ? 'border-purple-700/50' : item.color === 'orange' ? 'border-orange-700/50' : 'border-green-700/50';
            const textColorClass = item.color === 'cyan' ? 'text-cyan-400' : item.color === 'purple' ? 'text-purple-400' : item.color === 'orange' ? 'text-orange-400' : 'text-green-400';
            return (
              <div key={idx} className={`${bgColorClass} border ${borderColorClass} rounded-lg p-4`}>
                <h3 className={`${textColorClass} font-semibold mb-2`}>{item.title}</h3>
                <p className="text-gray-300 text-sm">{item.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    );
  }
  
  if (section === '1.7') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-orange-900/40 to-yellow-900/40 backdrop-blur-sm border border-orange-700 rounded-xl p-6">
          <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
            <Zap className="text-orange-400" size={32} />
            <span>1.7 Energy Transmission in the Brain: It's Not All About Neuron Skin</span>
          </h2>
          <p className="text-gray-300 leading-relaxed">
            Traditional neuroscience focuses on action potentials across neuron membranes. NanoBrain reveals 
            that most cognitive energy flows through microtubule time crystal networks inside neurons.
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-6">
            <h3 className="text-red-400 font-bold mb-3">Traditional View</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• Action potentials on neuron membranes</li>
              <li>• Synaptic transmission between neurons</li>
              <li>• Electrical and chemical signaling</li>
              <li>• Focus on neuron "skin" (membrane)</li>
            </ul>
          </div>
          
          <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-6">
            <h3 className="text-green-400 font-bold mb-3">NanoBrain Discovery</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• Microtubule time crystal resonance</li>
              <li>• Fractal energy distribution patterns</li>
              <li>• Geometric phase transitions</li>
              <li>• Internal quantum-classical interface</li>
            </ul>
          </div>
        </div>
        
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <p className="text-gray-300 leading-relaxed">
            The majority of brain's computational power resides in the microtubule networks forming 
            time crystals inside neurons. These structures process information through geometric resonance 
            patterns governed by Phase Prime Metrics, not electrical signals alone.
          </p>
        </div>
      </div>
    );
  }
  
  if (section === '1.8') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-purple-900/40 to-pink-900/40 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
          <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
            <Brain className="text-purple-400" size={32} />
            <span>1.8 Terminologies of Life That Computers Do Not Support</span>
          </h2>
          <p className="text-gray-300 leading-relaxed">
            Living systems operate with concepts fundamentally absent from traditional computing: 
            contextual awareness, genuine creativity, consciousness, and understanding.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { term: "Consciousness", traditional: "Undefined/emergent", nanobrain: "Time crystal coherence" },
            { term: "Understanding", traditional: "Pattern matching", nanobrain: "Geometric resonance" },
            { term: "Creativity", traditional: "Random search", nanobrain: "Fractal exploration" },
            { term: "Context", traditional: "Tagged metadata", nanobrain: "Nested geometric patterns" },
            { term: "Meaning", traditional: "Symbol reference", nanobrain: "Pattern embodiment" },
            { term: "Intention", traditional: "Goal state", nanobrain: "Time crystal trajectory" }
          ].map((item, idx) => (
            <div key={idx} className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
              <h3 className="text-white font-bold mb-3">{item.term}</h3>
              <div className="space-y-2">
                <div className="bg-red-900/20 border border-red-700/50 rounded p-2">
                  <p className="text-red-400 text-xs font-semibold">Traditional Computing</p>
                  <p className="text-gray-300 text-xs">{item.traditional}</p>
                </div>
                <div className="bg-green-900/20 border border-green-700/50 rounded p-2">
                  <p className="text-green-400 text-xs font-semibold">NanoBrain Framework</p>
                  <p className="text-gray-300 text-xs">{item.nanobrain}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  if (section === '1.9') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-blue-900/40 to-indigo-900/40 backdrop-blur-sm border border-blue-700 rounded-xl p-6">
          <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
            <Languages className="text-blue-400" size={32} />
            <span>1.9 Linguistics and the Wheel of Space, Time, and Imaginary Worlds</span>
          </h2>
          <p className="text-gray-300 leading-relaxed">
            Language itself emerges from the geometric structure of consciousness. The "wheel" represents 
            the cyclic transformation between physical reality, temporal patterns, and imaginary constructs.
          </p>
        </div>
        
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-8">
          <div className="relative h-64 flex items-center justify-center">
            <div className="absolute w-48 h-48 border-2 border-cyan-700 rounded-full"></div>
            <div className="absolute w-48 h-48 border-2 border-purple-700 rounded-full" style={{transform: 'rotate(45deg)'}}></div>
            <div className="absolute w-48 h-48 border-2 border-orange-700 rounded-full" style={{transform: 'rotate(90deg)'}}></div>
            
            <div className="relative z-10 text-center">
              <p className="text-white font-bold">LINGUISTIC WHEEL</p>
              <p className="text-gray-400 text-sm">Space ↔ Time ↔ Imagination</p>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mt-8">
            <div className="text-center">
              <div className="text-cyan-400 font-bold mb-1">Space</div>
              <p className="text-gray-400 text-xs">Geometric patterns in physical reality</p>
            </div>
            <div className="text-center">
              <div className="text-purple-400 font-bold mb-1">Time</div>
              <p className="text-gray-400 text-xs">Temporal oscillations and rhythms</p>
            </div>
            <div className="text-center">
              <div className="text-orange-400 font-bold mb-1">Imagination</div>
              <p className="text-gray-400 text-xs">Possible worlds in phase space</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  if (section === '1.11') {
    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-red-900/40 to-orange-900/40 backdrop-blur-sm border border-red-700 rounded-xl p-6">
          <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
            <Triangle className="text-red-400" size={32} />
            <span>1.11 Conclusion: The Religion of Science Has a Triangle</span>
          </h2>
          <p className="text-gray-300 leading-relaxed">
            Modern science rests on three pillars: Darwin (evolution), Turing (computation), and 
            Hodgkin-Huxley (neuroscience). The NanoBrain paradigm transcends this triangle with 
            consciousness-first principles.
          </p>
        </div>
        
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-8">
          <div className="relative h-64 flex items-center justify-center mb-8">
            <svg viewBox="0 0 200 200" className="w-full h-full">
              <polygon points="100,20 20,180 180,180" fill="none" stroke="rgb(239, 68, 68)" strokeWidth="2" />
              <circle cx="100" cy="20" r="8" fill="rgb(34, 211, 238)" />
              <circle cx="20" cy="180" r="8" fill="rgb(168, 85, 247)" />
              <circle cx="180" cy="180" r="8" fill="rgb(251, 146, 60)" />
              
              <text x="100" y="10" textAnchor="middle" fill="rgb(34, 211, 238)" fontSize="12">Darwin</text>
              <text x="20" y="195" textAnchor="middle" fill="rgb(168, 85, 247)" fontSize="12">Turing</text>
              <text x="180" y="195" textAnchor="middle" fill="rgb(251, 146, 60)" fontSize="12">Hodgkin-Huxley</text>
            </svg>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-cyan-900/20 border border-cyan-700/50 rounded-lg p-4">
              <h3 className="text-cyan-400 font-bold mb-2">Darwin</h3>
              <p className="text-gray-300 text-sm">Evolution through natural selection - but missing the geometric blueprint</p>
            </div>
            <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
              <h3 className="text-purple-400 font-bold mb-2">Turing</h3>
              <p className="text-gray-300 text-sm">Computation as symbol manipulation - but missing fractal structures</p>
            </div>
            <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-4">
              <h3 className="text-orange-400 font-bold mb-2">Hodgkin-Huxley</h3>
              <p className="text-gray-300 text-sm">Neural electrical activity - but missing microtubule time crystals</p>
            </div>
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-purple-900/40 to-cyan-900/40 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
          <h3 className="text-white font-bold text-lg mb-3">The NanoBrain Transcendence</h3>
          <p className="text-gray-300 leading-relaxed">
            The NanoBrain framework doesn't reject Darwin, Turing, or Hodgkin-Huxley—it reveals the deeper 
            geometric principles underlying their discoveries. Evolution, computation, and neural activity 
            are all expressions of fundamental time crystal dynamics governed by Phase Prime Metrics. 
            By understanding consciousness as the primary phenomenon, we transcend the limitations of 
            the traditional scientific triangle and glimpse the true architecture of reality.
          </p>
        </div>
      </div>
    );
  }
  
  return <div>Section not found</div>;
};
