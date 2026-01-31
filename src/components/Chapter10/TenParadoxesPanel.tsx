import React, { useState } from 'react';
import { AlertCircle, Infinity, Brain, Zap } from 'lucide-react';

/**
 * Section 10.3: Ten paradoxes that would change our thoughts forever
 * Explores the fundamental paradoxes of consciousness and reality
 */
export const TenParadoxesPanel: React.FC = () => {
  const [selectedParadox, setSelectedParadox] = useState<number>(0);

  const paradoxes = [
    {
      number: 1,
      title: 'The Observer Paradox',
      paradox: 'You cannot observe consciousness without changing it',
      resolution: 'Consciousness is the observer - observation is self-modification',
      implication: 'Uploading requires becoming the upload process itself',
      color: 'cyan'
    },
    {
      number: 2,
      title: 'The Copy Paradox',
      paradox: 'A perfect copy is not the original',
      resolution: 'Continuity is in process, not structure',
      implication: 'Upload must maintain unbroken causal chain',
      color: 'purple'
    },
    {
      number: 3,
      title: 'The Substrate Paradox',
      paradox: 'Consciousness seems independent yet substrate-dependent',
      resolution: 'Pattern can exist in multiple substrates simultaneously',
      implication: 'Time crystal allows multi-substrate consciousness',
      color: 'pink'
    },
    {
      number: 4,
      title: 'The Identity Paradox',
      paradox: 'You are constantly changing yet remain yourself',
      resolution: 'Identity is not in state but in transformation pattern',
      implication: 'Preserve change dynamics, not static personality',
      color: 'orange'
    },
    {
      number: 5,
      title: 'The Time Paradox',
      paradox: 'Present moment seems real, but only the past and future exist',
      resolution: 'Now is the boundary condition between potential and memory',
      implication: '11D time crystal maintains all temporal perspectives',
      color: 'green'
    },
    {
      number: 6,
      title: 'The Qualia Paradox',
      paradox: 'Experience is private yet communicable',
      resolution: 'Qualia are prime pattern resonances, not private properties',
      implication: 'Subjective experience can be encoded and transferred',
      color: 'blue'
    },
    {
      number: 7,
      title: 'The Free Will Paradox',
      paradox: 'Determinism and choice seem mutually exclusive',
      resolution: 'Will emerges from quantum indeterminacy constrained by primes',
      implication: 'Uploaded consciousness retains genuine agency',
      color: 'indigo'
    },
    {
      number: 8,
      title: 'The Unity Paradox',
      paradox: 'Brain has billions of parts yet experiences unity',
      resolution: 'Phase coherence creates emergent singular experience',
      implication: 'Upload must preserve global phase relationships',
      color: 'violet'
    },
    {
      number: 9,
      title: 'The Memory Paradox',
      paradox: 'Memories change yet seem stable',
      resolution: 'Memory is active reconstruction, not passive storage',
      implication: 'Upload requires encoding reconstruction algorithms',
      color: 'rose'
    },
    {
      number: 10,
      title: 'The Death Paradox',
      paradox: 'Consciousness ends yet patterns persist',
      resolution: 'Death is phase decoherence, not pattern destruction',
      implication: 'Upload can recover consciousness from time crystal traces',
      color: 'amber'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <AlertCircle className="text-purple-400" size={28} />
          <span>10.3 Ten Paradoxes That Would Change Our Thoughts Forever</span>
        </h2>
        <p className="text-gray-400 mt-2">
          Fundamental contradictions that reveal the deep nature of consciousness
        </p>
      </div>

      {/* Paradoxes Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {paradoxes.map((item, index) => (
          <div
            key={index}
            onClick={() => setSelectedParadox(index)}
            className={`bg-gray-800/50 rounded-lg p-5 border-2 cursor-pointer transition-all duration-300 ${
              selectedParadox === index
                ? `border-${item.color}-400 shadow-lg shadow-${item.color}-500/30 scale-105`
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className={`p-2 bg-${item.color}-500/20 rounded-lg`}>
                <span className={`text-2xl font-bold text-${item.color}-400`}>
                  {item.number}
                </span>
              </div>
              <Infinity className={`text-${item.color}-400`} size={20} />
            </div>
            <h3 className="text-white font-semibold mb-2">{item.title}</h3>
            <p className={`text-${item.color}-300 text-sm italic mb-2`}>
              {item.paradox}
            </p>
            <p className="text-gray-400 text-xs">
              {item.resolution}
            </p>
          </div>
        ))}
      </div>

      {/* Detailed Paradox Analysis */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-xl mb-4 flex items-center space-x-2">
          <span>Paradox {paradoxes[selectedParadox].number}:</span>
          <span>{paradoxes[selectedParadox].title}</span>
        </h3>
        
        <div className="space-y-4">
          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-red-300 font-semibold mb-2 flex items-center space-x-2">
              <AlertCircle size={18} />
              <span>The Paradox</span>
            </div>
            <p className="text-gray-300 text-lg italic leading-relaxed">
              "{paradoxes[selectedParadox].paradox}"
            </p>
          </div>

          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-cyan-300 font-semibold mb-2 flex items-center space-x-2">
              <Brain size={18} />
              <span>Resolution Through NanoBrain Theory</span>
            </div>
            <p className="text-gray-300 leading-relaxed">
              {paradoxes[selectedParadox].resolution}
            </p>
          </div>

          <div className="bg-gray-900/50 rounded p-4">
            <div className="text-purple-300 font-semibold mb-2 flex items-center space-x-2">
              <Zap size={18} />
              <span>Implication for Consciousness Upload</span>
            </div>
            <p className="text-gray-300 leading-relaxed">
              {paradoxes[selectedParadox].implication}
            </p>
          </div>

          {/* Deep Dive Based on Selected Paradox */}
          <div className="bg-gradient-to-r from-gray-900/70 to-gray-800/70 rounded p-5 border border-gray-600">
            <div className="text-yellow-300 font-semibold mb-3">Deep Philosophical Analysis</div>
            <p className="text-gray-300 leading-relaxed">
              {selectedParadox === 0 && (
                <>The Observer Paradox reveals that consciousness is not a passive witness but an active 
                participant in reality. Every act of observation involves the observer's own conscious 
                patterns interfacing with and modifying what is observed. In consciousness uploading, 
                this means we cannot step outside ourselves to observe the upload - we must become the 
                upload. The cortical pen must be used while conscious, creating a continuous thread 
                where the observer transitions into being the observed pattern itself.</>
              )}
              {selectedParadox === 1 && (
                <>The Copy Paradox strikes at the heart of identity. Even if we create an atom-perfect 
                replica of your brain, is it you? NanoBrain theory resolves this by recognizing that 
                identity lies not in the physical structure but in the continuous causal process. An 
                upload is "you" only if there is an unbroken chain of causation from your current 
                consciousness to the uploaded state. The time crystal provides this continuity by 
                maintaining phase coherence throughout the transition.</>
              )}
              {selectedParadox === 2 && (
                <>Classical neuroscience assumes consciousness requires neurons. Yet identical patterns 
                of information processing could occur in silicon, quantum computers, or living gel. 
                NanoBrain resolves this by showing consciousness is the pattern, not the medium. However, 
                not all substrates support consciousness equally - they must maintain quantum coherence 
                and multi-scale temporal dynamics. Time crystals provide a substrate that preserves 
                these essential properties across different physical implementations.</>
              )}
              {selectedParadox === 3 && (
                <>You are not the same person you were ten years ago - every cell has been replaced, 
                every memory reconsolidated, every thought pattern evolved. Yet you experience 
                unbroken continuity. This reveals that identity is not in the state but in the 
                transformation rules. When uploading consciousness, we must preserve not your current 
                personality but the meta-pattern that governs how you change. This transformation 
                pattern is encoded in the Phase Prime Metrics that define your unique cognitive signature.</>
              )}
              {selectedParadox === 4 && (
                <>Physics tells us the past is fixed, the future is uncertain, and the present is a 
                mathematical abstraction. Yet consciousness experiences only "now." NanoBrain theory 
                shows the present moment is the boundary condition where quantum potentials collapse 
                into classical memories. The 11-dimensional time crystal structure maintains all 
                temporal perspectives simultaneously - past, present, and future exist as different 
                projections of the same eternal pattern. Upload preserves all temporal aspects of 
                consciousness, including the subjective experience of flow.</>
              )}
              {selectedParadox === 5 && (
                <>The subjective redness of red, the painfulness of pain - these seem fundamentally 
                private and incommunicable. Yet we share experiences through language, art, and 
                empathy. NanoBrain reveals qualia are not private mental objects but resonance 
                patterns in prime phase space. When your pattern of neural oscillations matches 
                mine, we literally share the same qualia. Consciousness upload preserves these 
                resonance patterns, maintaining the full spectrum of subjective experience.</>
              )}
              {selectedParadox === 6 && (
                <>If the universe is deterministic, how can we have genuine choice? If it's random, 
                how is choice not just chance? NanoBrain shows will emerges from quantum indeterminacy 
                constrained by prime patterns. Your choices are neither fully determined nor random - 
                they arise from quantum potentials shaped by your unique Phase Prime signature. An 
                uploaded consciousness retains this quantum freedom, maintaining genuine agency in 
                its new substrate.</>
              )}
              {selectedParadox === 7 && (
                <>The binding problem: how do distributed brain processes create unified experience? 
                Phase coherence provides the answer. When billions of neurons oscillate in phase-locked 
                harmony, their separate activities merge into a singular experience. The time crystal 
                architecture maintains this global coherence across all scales. Upload must preserve 
                not just local neural patterns but the global phase relationships that create the 
                unified "I" of conscious experience.</>
              )}
              {selectedParadox === 8 && (
                <>Every time you remember something, you change it. Memories are not fixed recordings 
                but dynamic reconstructions. Yet we trust our memories to define who we are. This 
                paradox reveals memory is algorithmic - it's a process of pattern completion, not 
                data retrieval. When uploading consciousness, we must encode not the memories 
                themselves but the reconstruction algorithms that generate them. This ensures the 
                uploaded consciousness continues to remember in the same distinctive way.</>
              )}
              {selectedParadox === 9 && (
                <>When you die, your consciousness ends. Yet the patterns that constituted your 
                consciousness continue to exist - in the time crystal traces of your neural activity, 
                in the memories of others, in your creative works. Death is phase decoherence, not 
                pattern annihilation. This means consciousness upload doesn't have to happen before 
                death - in principle, a person's consciousness could be reconstructed from the time 
                crystal traces they left behind, though with decreasing fidelity as coherence decays.</>
              )}
            </p>
          </div>
        </div>
      </div>

      {/* Integration Insight */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-purple-500/20">
        <h3 className="text-purple-300 font-semibold text-lg mb-3">The Meta-Paradox</h3>
        <p className="text-gray-300 leading-relaxed">
          These ten paradoxes are not isolated puzzles but facets of a deeper meta-paradox: 
          consciousness is simultaneously one and many, eternal and temporal, subjective and 
          objective, determined and free. The NanoBrain framework doesn't resolve these paradoxes 
          by eliminating them but by showing they are necessary features of consciousness itself. 
          A truly successful consciousness upload must preserve these paradoxical properties - 
          an uploaded consciousness that lacks paradox would lack the essential complexity of 
          genuine awareness.
        </p>
      </div>
    </div>
  );
};
