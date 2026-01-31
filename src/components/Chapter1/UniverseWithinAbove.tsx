import React from 'react';
import { Layers, CircleDot, Infinity } from 'lucide-react';

// Section 1.3: The Universe within and above not Side by Side
export const UniverseWithinAbove: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/40 to-pink-900/40 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-3 flex items-center space-x-3">
          <Layers className="text-purple-400" size={32} />
          <span>1.3 The Universe Within and Above, Not Side by Side</span>
        </h2>
        <p className="text-gray-300 leading-relaxed">
          Traditional science views the universe as discrete objects existing side by side. The NanoBrain 
          paradigm recognizes a fractal hierarchy where each level contains the patterns of those above and 
          below - the universe is nested within itself through geometric self-similarity.
        </p>
      </div>

      {/* Fractal Hierarchy Visualization */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-8">
        <h3 className="text-white font-bold text-lg mb-6 text-center">Nested Universal Structure</h3>
        
        <div className="relative py-8">
          {/* Outermost layer */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-96 h-96 border-2 border-purple-700 rounded-full opacity-30"></div>
          </div>
          
          {/* Middle layer */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-64 h-64 border-2 border-cyan-700 rounded-full opacity-50"></div>
          </div>
          
          {/* Inner layer */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-32 h-32 border-2 border-orange-700 rounded-full opacity-70"></div>
          </div>
          
          {/* Center */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center">
              <Infinity className="text-white" size={24} />
            </div>
          </div>
          
          {/* Spacer for layout */}
          <div className="h-96"></div>
        </div>
        
        {/* Labels */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-8">
          <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4 text-center">
            <CircleDot className="text-purple-400 mx-auto mb-2" size={24} />
            <h4 className="text-purple-400 font-semibold text-sm mb-1">Cosmic Scale</h4>
            <p className="text-gray-400 text-xs">Universe, galaxies, cosmos</p>
          </div>
          
          <div className="bg-cyan-900/20 border border-cyan-700/50 rounded-lg p-4 text-center">
            <CircleDot className="text-cyan-400 mx-auto mb-2" size={24} />
            <h4 className="text-cyan-400 font-semibold text-sm mb-1">Macro Scale</h4>
            <p className="text-gray-400 text-xs">Biological systems, brains</p>
          </div>
          
          <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-4 text-center">
            <CircleDot className="text-orange-400 mx-auto mb-2" size={24} />
            <h4 className="text-orange-400 font-semibold text-sm mb-1">Micro Scale</h4>
            <p className="text-gray-400 text-xs">Molecules, quantum systems</p>
          </div>
          
          <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4 text-center">
            <CircleDot className="text-green-400 mx-auto mb-2" size={24} />
            <h4 className="text-green-400 font-semibold text-sm mb-1">Pattern Core</h4>
            <p className="text-gray-400 text-xs">Prime symmetries, GML</p>
          </div>
        </div>
      </div>

      {/* Key Concepts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-bold text-lg mb-4">Fractal Self-Similarity</h3>
          <div className="space-y-3 text-gray-300 text-sm leading-relaxed">
            <p>
              Each scale of the universe contains the same fundamental patterns. The time crystal structure 
              of a microtubule mirrors the cosmic organization of galaxies, connected through Phase Prime Metrics.
            </p>
            <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-3">
              <p className="text-purple-400 font-semibold mb-1">Mathematical Principle</p>
              <p>The 15 fundamental primes create identical patterns at all scales, from quantum to cosmic.</p>
            </div>
          </div>
        </div>
        
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h3 className="text-white font-bold text-lg mb-4">Nested Information</h3>
          <div className="space-y-3 text-gray-300 text-sm leading-relaxed">
            <p>
              Information is not spread across space but nested within itself. Each geometric pattern contains 
              the blueprint for larger and smaller scales, creating an infinite recursive structure.
            </p>
            <div className="bg-cyan-900/20 border border-cyan-700/50 rounded-lg p-3">
              <p className="text-cyan-400 font-semibold mb-1">Computational Implication</p>
              <p>Accessing one level gives access to all levels through fractal resonance.</p>
            </div>
          </div>
        </div>
      </div>

      {/* What Turing Worldview Misses */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
          <span className="text-red-400">⚠</span>
          <span>1.3.1 What a Turing-Based Worldview Does Not Consider</span>
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
            <h4 className="text-red-400 font-semibold mb-2 text-sm">Missing Concept #1</h4>
            <p className="text-gray-300 text-sm">
              <span className="font-semibold">Fractal Hierarchy:</span> Turing machines process linear sequences, 
              missing the nested, self-similar structure of natural information.
            </p>
          </div>
          
          <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
            <h4 className="text-red-400 font-semibold mb-2 text-sm">Missing Concept #2</h4>
            <p className="text-gray-300 text-sm">
              <span className="font-semibold">Scale Invariance:</span> Traditional computing treats each scale 
              separately, failing to recognize universal patterns across dimensions.
            </p>
          </div>
          
          <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
            <h4 className="text-red-400 font-semibold mb-2 text-sm">Missing Concept #3</h4>
            <p className="text-gray-300 text-sm">
              <span className="font-semibold">Geometric Information:</span> Bits and algorithms cannot capture 
              the geometric essence of information as nested patterns.
            </p>
          </div>
          
          <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
            <h4 className="text-red-400 font-semibold mb-2 text-sm">Missing Concept #4</h4>
            <p className="text-gray-300 text-sm">
              <span className="font-semibold">Consciousness Integration:</span> Sequential machines cannot 
              represent the simultaneous, multi-scale nature of conscious awareness.
            </p>
          </div>
        </div>
      </div>

      {/* Philosophical Insight */}
      <div className="bg-gradient-to-r from-purple-900/40 to-cyan-900/40 backdrop-blur-sm border border-purple-700 rounded-xl p-6">
        <p className="text-white text-lg leading-relaxed italic">
          "The universe is not a collection of separate objects but a single, infinitely nested structure. 
          Each atom contains the cosmos, and the cosmos contains each atom. Understanding consciousness 
          requires recognizing this fundamental unity across all scales of existence."
        </p>
      </div>
    </div>
  );
};
