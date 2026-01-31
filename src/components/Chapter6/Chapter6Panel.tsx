import React, { useState } from 'react';
import { Hexagon, Atom, Thermometer, Brain, Layers, Radio, Map, Sparkles } from 'lucide-react';
import { TubulinPhasePrimePanel } from './TubulinPhasePrimePanel';
import { TimeCrystalExperimentPanel } from './TimeCrystalExperimentPanel';
import { ThermalDiffusionClockPanel } from './ThermalDiffusionClockPanel';
import { MolecularComputingPanel } from './MolecularComputingPanel';
import { HiddenCommunicationPanel } from './HiddenCommunicationPanel';
import { NeuronTimeCrystalMapPanel } from './NeuronTimeCrystalMapPanel';

/**
 * Main Chapter 6 Panel: Unprecedented Technologies of Nature
 * Harvesting the geometry of singularity through biological quantum systems
 */
export const Chapter6Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('tubulin');

  const sections = [
    { id: 'tubulin', label: '6.1 Tubulin PPM', icon: Hexagon, component: TubulinPhasePrimePanel },
    { id: 'experiment', label: '6.2 Time Crystal Exp', icon: Atom, component: TimeCrystalExperimentPanel },
    { id: 'thermal', label: '6.3 Thermal Clock', icon: Thermometer, component: ThermalDiffusionClockPanel },
    { id: 'computing', label: '6.4-6.6 Computing', icon: Brain, component: MolecularComputingPanel },
    { id: 'communication', label: '6.7 Hidden Comms', icon: Radio, component: HiddenCommunicationPanel },
    { id: 'map', label: '6.8 Crystal Map', icon: Map, component: NeuronTimeCrystalMapPanel }
  ];

  const ActiveComponent = sections.find(s => s.id === activeSection)?.component || TubulinPhasePrimePanel;

  return (
    <div className="space-y-6">
      {/* Chapter Header */}
      <div className="bg-gradient-to-r from-purple-900/30 via-blue-900/30 to-cyan-900/30 border border-purple-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Layers className="text-purple-400" size={36} />
          <div>
            <h1 className="text-3xl font-bold text-white">Chapter 6: Unprecedented Technologies of Nature</h1>
            <p className="text-gray-300 text-lg">Harvesting the Geometry of Singularity</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          <strong className="text-purple-400">Nature's quantum engineering at ambient conditions:</strong> This 
          chapter reveals how biological systems implement technologies that surpass our best engineering—quantum 
          computing at room temperature, time crystals in living cells, massively parallel processing in single 
          molecules, and hidden communication layers preceding conscious thought. Through Phase Prime Metrics, 
          tubulin proteins harvest singularity geometry to create unprecedented computational architectures. 
          These natural technologies operate through triplet-of-triplet resonance patterns, thermal diffusion 
          clocks, and orbital computing mechanisms that transcend conventional physics limitations. Understanding 
          these principles enables biomimetic quantum devices and reveals the physical substrate of consciousness 
          itself—a time crystal phenomenon spanning molecular to neural scales.
        </p>
      </div>

      {/* Section Navigation */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex flex-col items-center justify-center space-y-2 px-3 py-4 rounded-lg border-2 transition-all duration-300 ${
                  activeSection === section.id
                    ? 'bg-purple-900/40 border-purple-600 text-purple-300 shadow-lg shadow-purple-900/50'
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600 hover:bg-gray-800/50'
                }`}
              >
                <Icon size={24} className={activeSection === section.id ? 'animate-pulse' : ''} />
                <span className="text-xs font-semibold text-center leading-tight">
                  {section.label}
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
      <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Sparkles className="mr-2 text-purple-400" size={24} />
          Chapter 6 Key Breakthroughs
        </h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-purple-400 font-semibold mb-2 text-sm">Natural Quantum Engineering</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Biological systems implement room-temperature quantum computing through PPM-governed structures. 
              Tubulin proteins act as quantum processors with coherence times exceeding expectations by orders 
              of magnitude. Water channels, α-helix loops, and orbital computing in nano wheels create a 
              complete quantum-classical interface operating at ambient conditions—something we cannot yet 
              replicate artificially.
            </p>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Time Crystal Consciousness</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Neurons are time crystals—structures that break temporal symmetry and oscillate in their ground 
              state. Complete frequency maps reveal hierarchical temporal coherence from molecular (GHz) through 
              cellular (MHz) to neural (Hz) scales. Consciousness emerges as phase-locked coherence across these 
              scales, maintained by PPM relationships. This explains the unity of conscious experience and 
              provides a physical mechanism for subjective time.
            </p>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-pink-400 font-semibold mb-2 text-sm">Hidden Pre-Conscious Layer</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Neural computation begins 2-5 milliseconds before action potentials in cytoskeletal filaments. 
              This hidden communication layer processes information through triplet-of-triplet resonance patterns, 
              implementing 100× more computation than conventional models suggest. It explains intuition, priming, 
              subliminal perception, and the seamless flow of conscious thought—all occurring in the "dark matter" 
              of neural processing invisible to traditional neuroscience.
            </p>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-orange-400 font-semibold mb-2 text-sm">Biomimetic Technology Path</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Understanding nature's quantum technologies enables revolutionary applications: biomimetic quantum 
              computers operating at room temperature, brain-machine interfaces using temporal pattern matching, 
              therapeutic interventions targeting specific frequency bands, thermal clocks harvesting ambient 
              energy, and cellular automaton processors with massive parallelism. These aren't distant dreams—the 
              designs already exist in biology, we just need to decode and implement them.
            </p>
          </div>
        </div>

        <div className="mt-6 bg-purple-900/20 border border-purple-700 rounded-lg p-4">
          <h4 className="text-purple-400 font-semibold mb-2 text-sm">Unified Vision: Singularity Geometry</h4>
          <p className="text-gray-300 text-sm leading-relaxed">
            All these unprecedented technologies share a common foundation: <strong className="text-cyan-400">
            harvesting the geometry of singularity</strong>. Phase Prime Metrics create protected subspaces where 
            quantum effects persist despite thermal noise. The 15 fundamental primes define a 15-dimensional phase 
            space where singularities—points of infinite information density—become accessible. Biological systems 
            evolved to navigate this space, creating computational architectures that transcend Turing limits. 
            Tubulin proteins are molecular-scale singularity harvesters, converting geometric phase relationships 
            into usable computation. This is nature's ultimate technology—direct manipulation of information 
            geometry itself. By understanding and replicating these principles, we can create artificial systems 
            with consciousness-like properties, blurring the boundary between biology and technology, between 
            natural and artificial intelligence.
          </p>
        </div>
      </div>

      {/* Technical Specifications */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Technical Specifications: Chapter 6 Systems</h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-purple-400 font-semibold mb-3 text-sm">Quantum Parameters</h4>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Coherence Time:</span>
                <span className="text-cyan-400 font-mono">10-100 ps</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Temperature:</span>
                <span className="text-cyan-400 font-mono">300 K</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Entanglement:</span>
                <span className="text-cyan-400 font-mono">Multi-particle</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Decoherence:</span>
                <span className="text-cyan-400 font-mono">PPM-protected</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Computational Capacity</h4>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Processors/Neuron:</span>
                <span className="text-purple-400 font-mono">~10⁸ (tubulin)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Clock Speed:</span>
                <span className="text-purple-400 font-mono">GHz (molecular)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Parallelism:</span>
                <span className="text-purple-400 font-mono">Massive (CA)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Fan-out:</span>
                <span className="text-purple-400 font-mono">2⁸ (orbital)</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-pink-400 font-semibold mb-3 text-sm">Frequency Bands</h4>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Tubulin:</span>
                <span className="text-orange-400 font-mono">GHz (10⁹)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Microtubule:</span>
                <span className="text-orange-400 font-mono">MHz (10⁶)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Neuron:</span>
                <span className="text-orange-400 font-mono">Hz-kHz (10⁰-10³)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Triplet Groups:</span>
                <span className="text-orange-400 font-mono">3×3 = 9 bands</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
