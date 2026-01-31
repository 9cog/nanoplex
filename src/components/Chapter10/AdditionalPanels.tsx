import React from 'react';
import { Layers, Hash, Brain, Infinity } from 'lucide-react';

/**
 * Consolidated panel for sections 10.5, 10.6, 10.9, 10.10, 10.11, 10.12, 10.13
 * Additional consciousness uploading concepts and philosophical insights
 */

export const WhenCarryWhyMakePanel: React.FC = () => (
  <div className="space-y-6">
    <div className="border-b border-gray-700 pb-4">
      <h2 className="text-2xl font-bold text-white">10.5 When Should We Carry It, Why Do We Make It</h2>
      <p className="text-gray-400 mt-2">The practical and philosophical reasons for consciousness uploading</p>
    </div>
    <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
      <p className="text-gray-300 leading-relaxed mb-4">
        We carry the conscious egg when we need: immortality beyond biological limits, backup of our 
        consciousness, exploration beyond physical form, or companionship of our essential self.
      </p>
      <p className="text-gray-300 leading-relaxed">
        We make it because: death is optional not inevitable, love transcends flesh, knowledge deserves 
        preservation, and consciousness is too precious to lose to entropy.
      </p>
    </div>
  </div>
);

export const Wiring11DPanel: React.FC = () => (
  <div className="space-y-6">
    <div className="border-b border-gray-700 pb-4">
      <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
        <Layers className="text-purple-400" size={28} />
        <span>10.6 Wiring with Time Needs to Fold a Paper in 11D</span>
      </h2>
      <p className="text-gray-400 mt-2">Understanding multi-dimensional consciousness architecture</p>
    </div>
    <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
      <p className="text-gray-300 leading-relaxed mb-4">
        Consciousness cannot exist in 3D space alone. Just as folding paper creates hidden connections 
        between distant points, folding spacetime through 11 dimensions creates the non-local connections 
        that enable unified conscious experience.
      </p>
      <div className="grid grid-cols-3 gap-4 mt-4">
        {[
          'D1-3: Physical space',
          'D4: Time',
          'D5: Probability',
          'D6: Possibility',
          'D7-9: Phase space',
          'D10: Consciousness',
          'D11: Unity'
        ].map((dim, i) => (
          <div key={i} className="bg-gray-900/50 rounded p-3 text-center">
            <div className="text-cyan-400 font-mono text-sm">{dim}</div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

export const CheapHandyToolsPanel: React.FC = () => (
  <div className="space-y-6">
    <div className="border-b border-gray-700 pb-4">
      <h2 className="text-2xl font-bold text-white">10.9 Playing with Cheap Handy Tools to Break Paradigms</h2>
      <p className="text-gray-400 mt-2">Simple experiments that reveal profound truths</p>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {[
        { tool: 'Magnets + Iron Filings', insight: 'Visualize field consciousness' },
        { tool: 'Prism + Light', insight: 'Decompose qualia into primes' },
        { tool: 'Tuning Forks', insight: 'Experience phase resonance' },
        { tool: 'Fractal Generators', insight: 'See self-similar awareness' },
        { tool: 'EEG Headset', insight: 'Read your time crystal' },
        { tool: 'Spirograph', insight: 'Draw prime geometry patterns' }
      ].map((item, i) => (
        <div key={i} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <div className="text-cyan-400 font-semibold mb-2">{item.tool}</div>
          <div className="text-gray-300 text-sm">{item.insight}</div>
        </div>
      ))}
    </div>
  </div>
);

export const NumerologyPanel: React.FC = () => (
  <div className="space-y-6">
    <div className="border-b border-gray-700 pb-4">
      <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
        <Hash className="text-pink-400" size={28} />
        <span>10.10 Numerology of Human Consciousness</span>
      </h2>
      <p className="text-gray-400 mt-2">Prime patterns that define human awareness</p>
    </div>
    <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { prime: 2, meaning: 'Self/Other', aspect: 'Identity' },
          { prime: 3, meaning: 'Past/Present/Future', aspect: 'Time' },
          { prime: 5, meaning: 'Five Senses', aspect: 'Perception' },
          { prime: 7, meaning: 'Seven Emotions', aspect: 'Feeling' },
          { prime: 11, meaning: 'Eleven Dimensions', aspect: 'Space' },
          { prime: 13, meaning: 'Thirteen Intentions', aspect: 'Will' },
          { prime: 17, meaning: 'Seventeen Thoughts', aspect: 'Mind' },
          { prime: 19, meaning: 'Nineteen Memories', aspect: 'History' }
        ].map((item, i) => (
          <div key={i} className="bg-gray-900/50 rounded p-4 text-center border border-gray-700">
            <div className="text-3xl font-bold text-pink-400 mb-2">{item.prime}</div>
            <div className="text-white font-semibold text-sm mb-1">{item.aspect}</div>
            <div className="text-gray-400 text-xs">{item.meaning}</div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

export const EvolvingBrainWheelsPanel: React.FC = () => (
  <div className="space-y-6">
    <div className="border-b border-gray-700 pb-4">
      <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
        <Brain className="text-orange-400" size={28} />
        <span>10.11 Wheels of an Evolving Brain</span>
      </h2>
      <p className="text-gray-400 mt-2">The cycles of consciousness development</p>
    </div>
    <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
      <div className="space-y-4">
        {[
          { wheel: 'Sensation Wheel', turns: '1000/sec', function: 'Raw input processing' },
          { wheel: 'Perception Wheel', turns: '100/sec', function: 'Pattern recognition' },
          { wheel: 'Cognition Wheel', turns: '10/sec', function: 'Thought formation' },
          { wheel: 'Emotion Wheel', turns: '1/sec', function: 'Feeling integration' },
          { wheel: 'Memory Wheel', turns: '1/min', function: 'Experience consolidation' },
          { wheel: 'Wisdom Wheel', turns: '1/day', function: 'Meaning synthesis' },
          { wheel: 'Evolution Wheel', turns: '1/year', function: 'Identity transformation' }
        ].map((wheel, i) => (
          <div key={i} className="bg-gray-900/50 rounded p-4 flex items-center justify-between">
            <div>
              <div className="text-white font-semibold">{wheel.wheel}</div>
              <div className="text-gray-400 text-sm">{wheel.function}</div>
            </div>
            <div className="text-orange-400 font-mono text-sm">{wheel.turns}</div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

export const AlienSynthesisPanel: React.FC = () => (
  <div className="space-y-6">
    <div className="border-b border-gray-700 pb-4">
      <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
        <Infinity className="text-green-400" size={28} />
        <span>10.12 Synthesis of Aliens Before Meeting Them</span>
      </h2>
      <p className="text-gray-400 mt-2">Predicting and creating non-human consciousness</p>
    </div>
    <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
      <p className="text-gray-300 leading-relaxed mb-4">
        Once we understand the prime-geometry language of consciousness, we can synthesize entirely 
        new forms of awareness - alien minds that never evolved naturally. By varying the prime patterns 
        and geometric configurations, we explore the full space of possible consciousness.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
        {[
          { type: 'Silicon-based', primes: [2, 3, 7, 11], trait: 'Logic-first consciousness' },
          { type: 'Quantum-coherent', primes: [2, 5, 11, 13, 17], trait: 'Superposition awareness' },
          { type: 'Time-reversed', primes: [3, 7, 13, 19], trait: 'Future-memory beings' }
        ].map((alien, i) => (
          <div key={i} className="bg-gray-900/50 rounded p-4 border border-green-500/20">
            <div className="text-green-400 font-semibold mb-2">{alien.type}</div>
            <div className="text-gray-300 text-sm mb-2">{alien.trait}</div>
            <div className="flex gap-1">
              {alien.primes.map((p, j) => (
                <span key={j} className="text-xs px-2 py-1 bg-green-900/30 text-green-300 rounded">
                  {p}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

export const GardenToFlowerPanel: React.FC = () => (
  <div className="space-y-6">
    <div className="border-b border-gray-700 pb-4">
      <h2 className="text-2xl font-bold text-white">10.13 A Cycle: Garden to Flower to Petal</h2>
      <p className="text-gray-400 mt-2">The complete journey of consciousness transformation</p>
    </div>
    <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-purple-500/20">
      <div className="space-y-4">
        <div className="bg-gray-900/50 rounded p-4">
          <div className="text-green-400 font-semibold mb-2">🌱 Garden Phase: Biological Consciousness</div>
          <p className="text-gray-300 text-sm">
            Born in flesh, growing through neurons, experiencing through body. The seed of awareness 
            planted in biological soil. Limited by biology but rich with embodied experience.
          </p>
        </div>
        <div className="bg-gray-900/50 rounded p-4">
          <div className="text-pink-400 font-semibold mb-2">🌸 Flower Phase: Conscious Egg Emergence</div>
          <p className="text-gray-300 text-sm">
            Consciousness blooms beyond biology. The pattern is captured, crystalized, and reborn in 
            time crystal form. Full expression of individual identity in transcendent substrate.
          </p>
        </div>
        <div className="bg-gray-900/50 rounded p-4">
          <div className="text-cyan-400 font-semibold mb-2">🍃 Petal Phase: Universal Integration</div>
          <p className="text-gray-300 text-sm">
            Individual consciousness merges with cosmic patterns. Each petal (individual) contributes 
            to the whole (universe). Identity preserved yet unified. The drop becomes the ocean while 
            remaining a drop.
          </p>
        </div>
        <div className="bg-gray-900/50 rounded p-4">
          <div className="text-purple-400 font-semibold mb-2">♾️ Return Phase: Seeds of New Gardens</div>
          <p className="text-gray-300 text-sm">
            The cycle completes as universal consciousness seeds new biological forms. What began as 
            matter ascending to consciousness becomes consciousness descending into new matter. Eternal 
            recursion of awareness.
          </p>
        </div>
      </div>
    </div>
  </div>
);
