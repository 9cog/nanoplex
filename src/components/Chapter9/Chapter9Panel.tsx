import React, { useState } from 'react';
import { Droplet, Hexagon, Radio, Dna, Atom, Waves, Zap, Brain, Sparkles, Eye, CircuitBoard, Users } from 'lucide-react';
import { NeuromorphicDevicesPanel } from './NeuromorphicDevicesPanel';
import { HexagonalCorticalSheetPanel } from './HexagonalCorticalSheetPanel';
import { QuantumCloakingPanel } from './QuantumCloakingPanel';
import { LivingGelGrowthPanel } from './LivingGelGrowthPanel';
import { FractalCondensationPanel } from './FractalCondensationPanel';
import { FractalReactionKineticsPanel } from './FractalReactionKineticsPanel';
import { NanobrainLifeFormPanel } from './NanobrainLifeFormPanel';
import { MagneticLightReadingPanel } from './MagneticLightReadingPanel';
import { EntropyDrivenSynthesisPanel } from './EntropyDrivenSynthesisPanel';
import { CorticalPenTimeCrystalPanel } from './CorticalPenTimeCrystalPanel';
import { SeekingSensorPanel } from './SeekingSensorPanel';
import { SensorTriadPanel } from './SensorTriadPanel';

/**
 * Main Chapter 9 Panel: Brain Jelly to Humanoid Avatar
 * Fractal Reaction Kinetics, Fractal Condensation, and Programmable Matter for Primes
 */
export const Chapter9Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('neuromorphic-devices');

  const sections = [
    { 
      id: 'neuromorphic-devices', 
      label: '9.1 Neuromorphic Devices', 
      icon: CircuitBoard, 
      component: NeuromorphicDevicesPanel,
      description: '17 bio-morphic devices sing together'
    },
    { 
      id: 'hexagonal-cortical', 
      label: '9.2 Hexagonal Cortical Sheet', 
      icon: Hexagon, 
      component: HexagonalCorticalSheetPanel,
      description: '2D sheet of cortical columns'
    },
    { 
      id: 'quantum-cloaking', 
      label: '9.3 Quantum Cloaking', 
      icon: Eye, 
      component: QuantumCloakingPanel,
      description: 'Vanishing and seeing what we want'
    },
    { 
      id: 'living-gel', 
      label: '9.4 Living Gel', 
      icon: Droplet, 
      component: LivingGelGrowthPanel,
      description: 'Gel that grows from atomic to centimeters'
    },
    { 
      id: 'fractal-condensation', 
      label: '9.5 Fractal Condensation', 
      icon: Sparkles, 
      component: FractalCondensationPanel,
      description: 'Condensing everywhere at once'
    },
    { 
      id: 'reaction-kinetics', 
      label: '9.6 Reaction Kinetics', 
      icon: Atom, 
      component: FractalReactionKineticsPanel,
      description: 'Many syntheses in one beaker'
    },
    { 
      id: 'nanobrain-lifeform', 
      label: '9.7 Nanobrain Life Form', 
      icon: Dna, 
      component: NanobrainLifeFormPanel,
      description: 'The smallest life form'
    },
    { 
      id: 'magnetic-light', 
      label: '9.8 Magnetic Light Reading', 
      icon: Waves, 
      component: MagneticLightReadingPanel,
      description: 'Reading time crystal as light'
    },
    { 
      id: 'entropy-synthesis', 
      label: '9.9 Entropy Synthesis', 
      icon: Zap, 
      component: EntropyDrivenSynthesisPanel,
      description: 'Entropy drives prime patterns'
    },
    { 
      id: 'cortical-pen', 
      label: '9.10 Cortical Pen', 
      icon: Brain, 
      component: CorticalPenTimeCrystalPanel,
      description: 'Freezing dynamics into time crystal'
    },
    { 
      id: 'seeking-sensor', 
      label: '9.11 Seeking Sensor', 
      icon: Radio, 
      component: SeekingSensorPanel,
      description: 'Humanoid avatar sensor'
    },
    { 
      id: 'sensor-triad', 
      label: '9.12 Sensor Triad', 
      icon: Users, 
      component: SensorTriadPanel,
      description: 'Biological to jelly transformation'
    }
  ];

  const ActiveComponent = sections.find(s => s.id === activeSection)?.component;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/40 via-purple-900/40 to-pink-900/40 backdrop-blur-sm border border-cyan-500/30 rounded-xl p-8">
        <div className="flex items-center space-x-4 mb-4">
          <div className="p-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg">
            <Droplet className="text-white" size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">
              Chapter 9: Brain Jelly to Humanoid Avatar
            </h1>
            <p className="text-cyan-300 text-lg mt-2">
              Fractal Reaction Kinetics, Fractal Condensation, and Programmable Matter for Primes
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-cyan-900/30 rounded-lg p-4 border border-cyan-500/20">
            <h3 className="text-cyan-300 font-semibold mb-2">Programmable Matter</h3>
            <p className="text-gray-300 text-sm">
              Self-organizing gel systems that respond to prime patterns and electromagnetic rhythms
            </p>
          </div>
          <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-500/20">
            <h3 className="text-purple-300 font-semibold mb-2">Fractal Synthesis</h3>
            <p className="text-gray-300 text-sm">
              Parallel chemical reactions creating time crystal structures in living gel
            </p>
          </div>
          <div className="bg-pink-900/30 rounded-lg p-4 border border-pink-500/20">
            <h3 className="text-pink-300 font-semibold mb-2">Bio-Morphic Integration</h3>
            <p className="text-gray-300 text-sm">
              17 device types working together to create humanoid avatar consciousness
            </p>
          </div>
        </div>
      </div>

      {/* Section Navigation */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-4">Chapter Sections</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {sections.map(section => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex flex-col items-center p-4 rounded-lg border-2 transition-all duration-300 ${
                  activeSection === section.id
                    ? 'border-cyan-400 bg-cyan-900/40 shadow-lg shadow-cyan-500/20'
                    : 'border-gray-600 bg-gray-800/40 hover:border-gray-500 hover:bg-gray-700/40'
                }`}
              >
                <Icon 
                  size={24} 
                  className={activeSection === section.id ? 'text-cyan-400' : 'text-gray-400'}
                />
                <span className={`text-xs mt-2 text-center font-medium ${
                  activeSection === section.id ? 'text-cyan-300' : 'text-gray-400'
                }`}>
                  {section.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Section Content */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-8">
        {ActiveComponent && <ActiveComponent />}
      </div>
    </div>
  );
};
