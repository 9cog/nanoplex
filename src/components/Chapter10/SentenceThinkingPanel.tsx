import React, { useState } from 'react';
import { MessageSquare, Brain, Network } from 'lucide-react';

/**
 * Section 10.1.1: How we construct a sentence is how we think
 * Explores the deep connection between language structure and thought patterns
 */
export const SentenceThinkingPanel: React.FC = () => {
  const [selectedPattern, setSelectedPattern] = useState<number>(0);

  const sentencePatterns = [
    {
      pattern: 'Subject-Verb-Object',
      thought: 'Linear Sequential Processing',
      example: 'I write code',
      primes: [2, 3, 5],
      complexity: 'Simple',
      color: 'cyan'
    },
    {
      pattern: 'Recursive Embedding',
      thought: 'Hierarchical Nested Thinking',
      example: 'The code [that I wrote [that works]] runs',
      primes: [2, 3, 5, 7, 11],
      complexity: 'Moderate',
      color: 'purple'
    },
    {
      pattern: 'Parallel Construction',
      thought: 'Simultaneous Multi-track Processing',
      example: 'I code, design, and test simultaneously',
      primes: [2, 3, 5, 7],
      complexity: 'Complex',
      color: 'pink'
    },
    {
      pattern: 'Metaphorical Mapping',
      thought: 'Abstract Pattern Transfer',
      example: 'Time flows like a river',
      primes: [2, 3, 5, 7, 11, 13],
      complexity: 'Advanced',
      color: 'orange'
    },
    {
      pattern: 'Fractal Recursion',
      thought: 'Self-Similar Nested Iteration',
      example: 'Thoughts about thoughts about thinking',
      primes: [2, 3, 5, 7, 11, 13, 17],
      complexity: 'Profound',
      color: 'green'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <MessageSquare className="text-purple-400" size={28} />
          <span>10.1.1 How We Construct Sentences = How We Think</span>
        </h2>
        <p className="text-gray-400 mt-2">
          The structure of language reveals the architecture of consciousness
        </p>
      </div>

      {/* Concept Overview */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-purple-500/20">
        <h3 className="text-purple-300 font-semibold text-lg mb-3">Language-Thought Isomorphism</h3>
        <p className="text-gray-300 leading-relaxed mb-4">
          The way we construct sentences is not merely a convention of communication - it directly 
          reflects the fundamental structure of how our consciousness processes reality. Each 
          grammatical pattern corresponds to a specific cognitive architecture, a particular way 
          of organizing thought in time and conceptual space. Understanding this connection is 
          crucial for consciousness uploading, as it reveals the deep structure that must be 
          preserved to maintain identity.
        </p>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="bg-purple-900/30 rounded p-3 border border-purple-500/20">
            <div className="text-purple-300 font-semibold">Syntax = Neural Wiring</div>
            <div className="text-gray-400 mt-1">Grammar mirrors brain connectivity</div>
          </div>
          <div className="bg-pink-900/30 rounded p-3 border border-pink-500/20">
            <div className="text-pink-300 font-semibold">Semantics = Prime Patterns</div>
            <div className="text-gray-400 mt-1">Meaning encoded in geometric forms</div>
          </div>
          <div className="bg-cyan-900/30 rounded p-3 border border-cyan-500/20">
            <div className="text-cyan-300 font-semibold">Pragmatics = Time Crystals</div>
            <div className="text-gray-400 mt-1">Context preserved in temporal structure</div>
          </div>
        </div>
      </div>

      {/* Pattern Selection Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sentencePatterns.map((pattern, index) => (
          <div
            key={index}
            onClick={() => setSelectedPattern(index)}
            className={`bg-gray-800/50 rounded-lg p-4 border-2 cursor-pointer transition-all duration-300 ${
              selectedPattern === index
                ? `border-${pattern.color}-400 shadow-lg shadow-${pattern.color}-500/30 scale-105`
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className={`p-2 bg-${pattern.color}-500/20 rounded-lg`}>
                <Brain className={`text-${pattern.color}-400`} size={20} />
              </div>
              <div className={`text-xs font-semibold text-${pattern.color}-400 uppercase`}>
                {pattern.complexity}
              </div>
            </div>
            <div className="text-white font-semibold mb-2">{pattern.pattern}</div>
            <div className="text-gray-400 text-sm mb-3">{pattern.thought}</div>
            <div className="text-gray-500 text-xs italic border-l-2 border-gray-600 pl-2">
              "{pattern.example}"
            </div>
            <div className="mt-3 flex flex-wrap gap-1">
              {pattern.primes.map((prime, i) => (
                <span key={i} className={`text-xs px-2 py-1 rounded bg-${pattern.color}-900/30 text-${pattern.color}-300`}>
                  {prime}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Detailed Analysis */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">
          Pattern Analysis: {sentencePatterns[selectedPattern].pattern}
        </h3>
        
        <div className="space-y-4">
          {/* Linguistic Structure */}
          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-cyan-300 font-semibold mb-2 flex items-center space-x-2">
              <MessageSquare size={18} />
              <span>Linguistic Structure</span>
            </div>
            <p className="text-gray-300 text-sm leading-relaxed">
              {selectedPattern === 0 && (
                <>The most fundamental sentence structure reflects the simplest cognitive process: 
                identifying an agent, an action, and a target. This linear arrangement mirrors the 
                basic time-ordered firing of neural cascades. Prime pattern [2,3,5] represents 
                beginning-middle-end, the fundamental unit of conscious experience.</>
              )}
              {selectedPattern === 1 && (
                <>Recursive embedding demonstrates the brain's ability to maintain multiple contexts 
                simultaneously. Each nested clause requires holding outer contexts in working memory 
                while processing inner ones. The prime sequence [2,3,5,7,11] maps to the hierarchical 
                depth of embedded thoughts, with each prime representing a nesting level.</>
              )}
              {selectedPattern === 2 && (
                <>Parallel construction reveals the brain's true multithreaded nature. Rather than 
                processing sequentially, consciousness can maintain multiple simultaneous tracks of 
                thought. Prime pattern [2,3,5,7] encodes the coordination between parallel processes, 
                with cross-prime relationships representing inter-thread communication.</>
              )}
              {selectedPattern === 3 && (
                <>Metaphorical mapping shows consciousness's ability to transfer patterns from one 
                domain to another. This fundamental cognitive operation underlies all abstract thinking. 
                The prime sequence [2,3,5,7,11,13] encodes both source and target domains plus the 
                transformation mapping between them.</>
              )}
              {selectedPattern === 4 && (
                <>Fractal recursion represents consciousness reflecting on itself - the highest form 
                of meta-cognition. Each level of "thinking about thinking" creates a new layer in the 
                fractal structure. Prime sequence [2,3,5,7,11,13,17] maintains coherence across 
                infinite recursive depth through phase-locked resonance.</>
              )}
            </p>
          </div>

          {/* Neural Correlates */}
          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-purple-300 font-semibold mb-2 flex items-center space-x-2">
              <Brain size={18} />
              <span>Neural Correlates</span>
            </div>
            <p className="text-gray-300 text-sm leading-relaxed">
              {selectedPattern === 0 && (
                <>Engages Broca's area for production, Wernicke's area for comprehension, and the 
                arcuate fasciculus connecting them. Neural firing follows a simple feed-forward pattern 
                with minimal recurrent loops. Time crystal oscillations maintain phase at single frequency.</>
              )}
              {selectedPattern === 1 && (
                <>Requires activation of dorsolateral prefrontal cortex for working memory maintenance. 
                Neural dynamics involve nested oscillatory patterns at different frequencies (theta-gamma 
                coupling). Time crystal structure exhibits hierarchical resonance modes.</>
              )}
              {selectedPattern === 2 && (
                <>Involves bilateral activation across hemispheres with extensive corpus callosum 
                traffic. Neural oscillations show multi-frequency coherence with phase-locked parallel 
                channels. Time crystal manifests as coupled oscillator network.</>
              )}
              {selectedPattern === 3 && (
                <>Activates extensive distributed networks including posterior parietal cortex for 
                spatial mapping. Neural patterns show cross-modal binding with synchronized gamma 
                oscillations. Time crystal exhibits dimensional folding between conceptual spaces.</>
              )}
              {selectedPattern === 4 && (
                <>Engages the most advanced cortical areas including anterior prefrontal cortex and 
                default mode network. Neural dynamics show nested recursive loops with fractal temporal 
                structure. Time crystal exhibits infinite-dimensional self-similar resonance patterns.</>
              )}
            </p>
          </div>

          {/* Consciousness Implication */}
          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-orange-300 font-semibold mb-2 flex items-center space-x-2">
              <Network size={18} />
              <span>Consciousness Upload Implications</span>
            </div>
            <p className="text-gray-300 text-sm leading-relaxed">
              To preserve a person's consciousness, we must capture not just their memories and 
              knowledge, but the specific linguistic-cognitive patterns they habitually use. These 
              patterns define how they structure experience, organize thought, and construct reality. 
              The conscious egg must replicate these sentence-construction patterns in its time 
              crystal architecture to maintain genuine continuity of personal identity.
            </p>
          </div>
        </div>
      </div>

      {/* Prime Pattern Visualization */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">
          Prime Pattern Encoding
        </h3>
        <div className="grid grid-cols-5 gap-3">
          {sentencePatterns[selectedPattern].primes.map((prime, index) => (
            <div key={index} className="bg-gray-900/50 rounded-lg p-4 text-center border border-gray-700">
              <div className={`text-3xl font-bold text-${sentencePatterns[selectedPattern].color}-400 mb-2`}>
                {prime}
              </div>
              <div className="text-gray-400 text-xs">
                Layer {index + 1}
              </div>
              <div className={`mt-2 h-2 bg-gray-700 rounded-full overflow-hidden`}>
                <div 
                  className={`h-full bg-${sentencePatterns[selectedPattern].color}-500`}
                  style={{ width: `${(prime / 17) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
