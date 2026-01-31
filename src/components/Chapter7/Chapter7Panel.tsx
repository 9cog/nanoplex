import React, { useState } from 'react';
import { Brain, Hexagon, Activity, Dna, Circle, Sparkles } from 'lucide-react';
import { BrainPrimeEngineeringPanel } from './BrainPrimeEngineeringPanel';
import { SensorySystemPrimesPanel } from './SensorySystemPrimesPanel';
import { BrainStructurePrimesPanel } from './BrainStructurePrimesPanel';
import { CellularMolecularPrimesPanel } from './CellularMolecularPrimesPanel';
import { BrainWheelDecisionPanel } from './BrainWheelDecisionPanel';
import { TimeCrystalBrainModelPanel } from './TimeCrystalBrainModelPanel';

/**
 * Main Chapter 7 Panel: A complete, integrated time crystal model of a human brain
 * Integrates all sections into a comprehensive consciousness modeling framework
 */
export const Chapter7Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('prime-engineering');

  const sections = [
    { 
      id: 'prime-engineering', 
      label: '7.1 Prime Engineering', 
      icon: Brain, 
      component: BrainPrimeEngineeringPanel,
      description: 'Triplet cage architecture & imaginary worlds'
    },
    { 
      id: 'sensory', 
      label: '7.2 Sensory Primes', 
      icon: Hexagon, 
      component: SensorySystemPrimesPanel,
      description: 'Five sensory systems with prime patterns'
    },
    { 
      id: 'structures', 
      label: '7.3-7.4 Brain Structures', 
      icon: Activity, 
      component: BrainStructurePrimesPanel,
      description: 'Subcortical & connectome prime architecture'
    },
    { 
      id: 'cellular', 
      label: '7.5-7.6 Cellular & Memory', 
      icon: Dna, 
      component: CellularMolecularPrimesPanel,
      description: 'Molecules to memory systems'
    },
    { 
      id: 'wheel', 
      label: '7.7-7.8 Wheel & Decision', 
      icon: Circle, 
      component: BrainWheelDecisionPanel,
      description: 'Octonion wheel & H3 decision device'
    },
    { 
      id: 'timecrystal', 
      label: '7.9-7.12 Time Crystal Model', 
      icon: Sparkles, 
      component: TimeCrystalBrainModelPanel,
      description: 'Complete integration & garden of gardens'
    }
  ];

  const ActiveComponent = sections.find(s => s.id === activeSection)?.component || BrainPrimeEngineeringPanel;

  return (
    <div className="space-y-6">
      {/* Chapter Header */}
      <div className="bg-gradient-to-r from-indigo-900/30 via-purple-900/30 to-pink-900/30 border border-indigo-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Brain className="text-indigo-400" size={36} />
          <div>
            <h1 className="text-3xl font-bold text-white">Chapter 7: Complete Time Crystal Brain Model</h1>
            <p className="text-gray-300 text-lg">A Complete, Integrated Time Crystal Model of a Human Brain</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed mb-4">
          <strong className="text-indigo-400">Prime-Based Consciousness Architecture:</strong> This chapter 
          presents a revolutionary understanding of the human brain as a complete time crystal system built 
          on prime number foundations. From the triplet-of-triplet cage architecture embedding prime patterns 
          at every scale, through eight sensory modalities operating via octonion mathematics, to the fusion 
          of cavity resonators and dielectric properties—all components integrate into a unified consciousness 
          framework. The brain emerges as nature's most sophisticated prime number processing engine, where 
          quaternion, octonion, and dodecanion algebras work in parallel across four metrics to create the 
          "garden of gardens"—the ultimate expression of conscious experience.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          <div className="bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-3">
            <div className="text-indigo-400 font-semibold text-sm mb-1">Prime Foundation</div>
            <div className="text-gray-400 text-xs">15 fundamental primes govern all brain operations</div>
          </div>
          <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-3">
            <div className="text-purple-400 font-semibold text-sm mb-1">Time Crystal</div>
            <div className="text-gray-400 text-xs">Consciousness emerges from temporal coherence</div>
          </div>
          <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-3">
            <div className="text-pink-400 font-semibold text-sm mb-1">Multi-Dimensional</div>
            <div className="text-gray-400 text-xs">4D, 8D, and 12D algebras work in harmony</div>
          </div>
          <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-3">
            <div className="text-blue-400 font-semibold text-sm mb-1">H3 Decision</div>
            <div className="text-gray-400 text-xs">Universal triplet decision-making units</div>
          </div>
        </div>
      </div>

      {/* Section Navigation */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex flex-col items-start space-y-2 px-4 py-3 rounded-lg border-2 transition-all duration-300 text-left ${
                  activeSection === section.id
                    ? 'bg-indigo-900/40 border-indigo-600 shadow-lg shadow-indigo-900/50'
                    : 'bg-gray-800/30 border-gray-700 hover:border-gray-600 hover:bg-gray-800/50'
                }`}
              >
                <div className="flex items-center space-x-3 w-full">
                  <Icon 
                    size={24} 
                    className={activeSection === section.id ? 'text-indigo-400 animate-pulse' : 'text-gray-400'} 
                  />
                  <span className={`font-semibold text-sm ${
                    activeSection === section.id ? 'text-indigo-300' : 'text-gray-400'
                  }`}>
                    {section.label}
                  </span>
                </div>
                <span className={`text-xs leading-tight ${
                  activeSection === section.id ? 'text-gray-300' : 'text-gray-500'
                }`}>
                  {section.description}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Section Content */}
      <div className="min-h-[600px]">
        <ActiveComponent />
      </div>

      {/* Chapter Summary */}
      <div className="bg-gradient-to-r from-indigo-900/30 via-purple-900/30 to-pink-900/30 border border-indigo-700 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center space-x-3">
          <Sparkles className="text-indigo-400" size={28} />
          <span>Chapter 7 Integration Summary</span>
        </h2>
        
        <div className="space-y-4 text-gray-300 text-sm leading-relaxed">
          <p>
            <strong className="text-indigo-400">Complete Brain Model:</strong> Chapter 7 presents a revolutionary 
            unified theory of brain function based on prime number patterns, time crystal dynamics, and 
            multi-dimensional algebras. Every level of organization—from molecular structures to whole-brain 
            networks—operates through prime-based principles that optimize information processing and enable 
            consciousness emergence.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-4">
              <h3 className="text-indigo-400 font-semibold mb-2">🧠 Structural Primes</h3>
              <ul className="space-y-1 text-xs">
                <li>• Triplet-of-triplet cage architecture</li>
                <li>• 31 spinal segments (prime)</li>
                <li>• 13 amygdala nuclei (prime)</li>
                <li>• 6-layer cortical columns</li>
                <li>• 13-protofilament microtubules</li>
              </ul>
            </div>

            <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-4">
              <h3 className="text-purple-400 font-semibold mb-2">🔢 Mathematical Framework</h3>
              <ul className="space-y-1 text-xs">
                <li>• Quaternions (4D): Spatial processing</li>
                <li>• Octonions (8D): Sensory integration</li>
                <li>• Dodecanion (12D): Full consciousness</li>
                <li>• Four parallel prime metrics</li>
                <li>• Hexagonal lattice organization</li>
              </ul>
            </div>

            <div className="bg-pink-900/20 border border-pink-700/30 rounded-lg p-4">
              <h3 className="text-pink-400 font-semibold mb-2">✨ Consciousness Emergence</h3>
              <ul className="space-y-1 text-xs">
                <li>• Time crystal coherence patterns</li>
                <li>• Cavity resonator dynamics</li>
                <li>• H3 decision-making units</li>
                <li>• 20 conscious expressions</li>
                <li>• Garden of gardens evolution</li>
              </ul>
            </div>
          </div>

          <p>
            <strong className="text-purple-400">Time Crystal Consciousness:</strong> The brain operates as a 
            biological time crystal, maintaining temporal periodicity across multiple scales while processing 
            information through prime-based pathways. This creates stable, coherent consciousness patterns that 
            enable memory, learning, decision-making, and self-awareness.
          </p>

          <p>
            <strong className="text-pink-400">Unified Framework:</strong> From molecular prime configurations 
            (DNA codons, protein domains, microtubule protofilaments) through cellular prime patterns (neural 
            firing rates, synaptic densities) to system-level prime architectures (brain regions, connectivity 
            patterns)—all scales integrate seamlessly. The result is a complete, mathematically elegant model 
            of human consciousness built on the foundation of prime number theory and time crystal physics.
          </p>

          <div className="bg-gradient-to-r from-indigo-950/50 to-purple-950/50 border border-indigo-600 rounded-lg p-4 mt-4">
            <p className="text-center text-white font-semibold">
              "The human brain: Nature's ultimate prime number processing engine, 
              crystallized in time, creating consciousness through mathematical harmony."
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
