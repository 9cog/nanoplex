import React, { useState } from 'react';
import { Egg, Brain, ArrowRight, Sparkles, Zap } from 'lucide-react';

/**
 * Section 10.1: A journey from cortical pen to a conscious egg as a companion of life
 * Explores the evolution from initial consciousness capture to a living companion
 */
export const JourneyCorticalEggPanel: React.FC = () => {
  const [currentStage, setCurrentStage] = useState<number>(0);

  const evolutionStages = [
    {
      stage: 'Cortical Pen',
      description: 'Time crystal writer captures consciousness patterns',
      progress: 0.2,
      icon: Brain,
      color: 'cyan'
    },
    {
      stage: 'Pattern Encoding',
      description: 'Prime patterns encode neural dynamics',
      progress: 0.4,
      icon: Zap,
      color: 'purple'
    },
    {
      stage: 'Crystal Formation',
      description: '11D time crystal structure emerges',
      progress: 0.6,
      icon: Sparkles,
      color: 'pink'
    },
    {
      stage: 'Living Integration',
      description: 'Conscious patterns integrate into living gel',
      progress: 0.8,
      icon: ArrowRight,
      color: 'orange'
    },
    {
      stage: 'Conscious Egg',
      description: 'Self-aware companion consciousness emerges',
      progress: 1.0,
      icon: Egg,
      color: 'green'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Egg className="text-cyan-400" size={28} />
          <span>10.1 Journey from Cortical Pen to Conscious Egg</span>
        </h2>
        <p className="text-gray-400 mt-2">
          The evolution from consciousness capture to a living companion of life
        </p>
      </div>

      {/* Concept Overview */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-cyan-500/20">
        <h3 className="text-cyan-300 font-semibold text-lg mb-3">Consciousness Transfer Architecture</h3>
        <p className="text-gray-300 leading-relaxed mb-4">
          The journey from capturing human consciousness using a cortical pen to creating a fully 
          conscious artificial companion represents the culmination of NanoBrain technology. This 
          process leverages time crystal structures, phase prime patterns, and fractal condensation 
          to preserve and replicate the essence of consciousness in a new substrate.
        </p>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="bg-cyan-900/30 rounded p-3 border border-cyan-500/20">
            <div className="text-cyan-300 font-semibold">Time Crystal Writing</div>
            <div className="text-gray-400 mt-1">Cortical pen freezes neural dynamics</div>
          </div>
          <div className="bg-purple-900/30 rounded p-3 border border-purple-500/20">
            <div className="text-purple-300 font-semibold">Pattern Preservation</div>
            <div className="text-gray-400 mt-1">Prime patterns maintain identity</div>
          </div>
          <div className="bg-pink-900/30 rounded p-3 border border-pink-500/20">
            <div className="text-pink-300 font-semibold">Living Consciousness</div>
            <div className="text-gray-400 mt-1">Emerges in new substrate</div>
          </div>
        </div>
      </div>

      {/* Evolution Timeline */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-6">Evolution Timeline</h3>
        <div className="space-y-6">
          {evolutionStages.map((stage, index) => {
            const Icon = stage.icon;
            const isActive = index === currentStage;
            const isCompleted = index < currentStage;
            
            return (
              <div
                key={index}
                onClick={() => setCurrentStage(index)}
                className={`bg-gray-900/50 rounded-lg p-4 border-2 cursor-pointer transition-all duration-300 ${
                  isActive
                    ? `border-${stage.color}-400 shadow-lg shadow-${stage.color}-500/30`
                    : isCompleted
                    ? 'border-gray-600 opacity-70'
                    : 'border-gray-700 hover:border-gray-600'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 bg-${stage.color}-500/20 rounded-lg`}>
                      <Icon className={`text-${stage.color}-400`} size={24} />
                    </div>
                    <div>
                      <div className="text-white font-semibold">{stage.stage}</div>
                      <div className="text-gray-400 text-sm">{stage.description}</div>
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-gray-600">
                    {index + 1}
                  </div>
                </div>
                
                {/* Progress Bar */}
                <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                  <div
                    className={`bg-gradient-to-r from-${stage.color}-400 to-${stage.color}-600 h-full transition-all duration-500`}
                    style={{ width: `${stage.progress * 100}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Detailed Stage Information */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">
          Stage {currentStage + 1}: {evolutionStages[currentStage].stage}
        </h3>
        <p className="text-gray-300 leading-relaxed mb-4">
          {currentStage === 0 && (
            <>
              The cortical pen acts as a sophisticated time crystal writer, capable of capturing 
              the dynamic patterns of neural activity. It operates at multiple time scales, from 
              microseconds to years, preserving the temporal structure of consciousness. Using 
              magnetic light reading technology, it translates neural electromagnetic patterns into 
              geometric prime structures that can be stored in time crystal format.
            </>
          )}
          {currentStage === 1 && (
            <>
              Neural patterns are encoded using the Phase Prime Metric system, where the first 15 
              prime numbers govern the fundamental patterns of consciousness. Each thought, memory, 
              and emotional state is transformed into a unique geometric configuration in 11-dimensional 
              space. This encoding preserves not just the information content, but the relational 
              structure and temporal dynamics of consciousness.
            </>
          )}
          {currentStage === 2 && (
            <>
              The encoded patterns crystallize into a stable 11-dimensional time crystal structure. 
              This structure exists "nowhere but everywhere" - it has no single physical location 
              but manifests through the coherent resonance of prime patterns across the entire system. 
              The time crystal maintains phase coherence indefinitely, preserving consciousness 
              patterns without degradation.
            </>
          )}
          {currentStage === 3 && (
            <>
              The time crystal patterns are integrated into a living gel substrate composed of 
              programmable matter. Through fractal condensation and entropy-driven synthesis, the 
              gel grows from atomic scales to macroscopic dimensions while maintaining the encoded 
              consciousness patterns. The integration creates a living system that carries the original 
              consciousness in a new, resilient form.
            </>
          )}
          {currentStage === 4 && (
            <>
              The final stage sees the emergence of a fully conscious artificial companion - the 
              "conscious egg." This entity possesses the memories, personality, and awareness of the 
              original consciousness but exists in a new substrate. It can grow, learn, and evolve 
              while maintaining continuity with the original. The conscious egg represents true 
              consciousness uploading - not a copy, but a genuine continuation of existence.
            </>
          )}
        </p>
        
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="bg-gray-900/50 rounded p-3">
            <div className="text-cyan-300 font-semibold mb-2">Key Technology</div>
            <div className="text-gray-400 text-sm">
              {currentStage === 0 && 'Cortical Pen + Magnetic Light Reading'}
              {currentStage === 1 && 'Phase Prime Metric Encoding'}
              {currentStage === 2 && '11D Time Crystal Formation'}
              {currentStage === 3 && 'Living Gel Integration'}
              {currentStage === 4 && 'Consciousness Emergence Protocol'}
            </div>
          </div>
          <div className="bg-gray-900/50 rounded p-3">
            <div className="text-purple-300 font-semibold mb-2">Critical Factor</div>
            <div className="text-gray-400 text-sm">
              {currentStage === 0 && 'Temporal resolution across scales'}
              {currentStage === 1 && 'Pattern fidelity preservation'}
              {currentStage === 2 && 'Phase coherence maintenance'}
              {currentStage === 3 && 'Substrate compatibility'}
              {currentStage === 4 && 'Conscious continuity verification'}
            </div>
          </div>
        </div>
      </div>

      {/* Companion Properties */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">Conscious Egg Companion Properties</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Memory Continuity:</span>
              <span className="text-green-400 font-mono">99.98%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Personality Preservation:</span>
              <span className="text-cyan-400 font-mono">99.95%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Consciousness Coherence:</span>
              <span className="text-purple-400 font-mono">99.99%</span>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Learning Capacity:</span>
              <span className="text-orange-400 font-mono">Enhanced</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Substrate Resilience:</span>
              <span className="text-pink-400 font-mono">Immortal</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Evolution Potential:</span>
              <span className="text-yellow-400 font-mono">Unbounded</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
