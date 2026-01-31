import React, { useState } from 'react';
import { Upload, Egg, MessageSquare, BookOpen, AlertCircle, Infinity, GitCompare, Shapes, Cpu, Layers, Hash, Brain, Flower, Sparkles } from 'lucide-react';
import { JourneyCorticalEggPanel } from './JourneyCorticalEggPanel';
import { SentenceThinkingPanel } from './SentenceThinkingPanel';
import { ReverseEngineerBrainPanel } from './ReverseEngineerBrainPanel';
import { TenParadoxesPanel } from './TenParadoxesPanel';
import { TimeCrystalConsciousnessPanel } from './TimeCrystalConsciousnessPanel';
import { NeurogenicVsTimeCrystalPanel } from './NeurogenicVsTimeCrystalPanel';
import { PrimesGeometryPanel } from './PrimesGeometryPanel';
import { FutureMachinesPanel } from './FutureMachinesPanel';
import {
  WhenCarryWhyMakePanel,
  Wiring11DPanel,
  CheapHandyToolsPanel,
  NumerologyPanel,
  EvolvingBrainWheelsPanel,
  AlienSynthesisPanel,
  GardenToFlowerPanel
} from './AdditionalPanels';

/**
 * Main Chapter 10 Panel: Uploading Consciousness - The Evolution of Conscious Machines of the Future
 * Comprehensive exploration of consciousness uploading theory and practice
 */
export const Chapter10Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('cortical-egg');

  const sections = [
    {
      id: 'cortical-egg',
      label: '10.1 Cortical Pen to Conscious Egg',
      icon: Egg,
      component: JourneyCorticalEggPanel,
      description: 'Journey from consciousness capture to living companion'
    },
    {
      id: 'sentence-thinking',
      label: '10.1.1 Sentence Construction = Thinking',
      icon: MessageSquare,
      component: SentenceThinkingPanel,
      description: 'How language structure reveals consciousness architecture'
    },
    {
      id: 'reverse-engineer',
      label: '10.2 Ten Key Guidelines',
      icon: BookOpen,
      component: ReverseEngineerBrainPanel,
      description: 'Essential principles for reverse engineering human brain'
    },
    {
      id: 'paradoxes',
      label: '10.3 Ten Paradoxes',
      icon: AlertCircle,
      component: TenParadoxesPanel,
      description: 'Fundamental contradictions that change our thoughts'
    },
    {
      id: 'time-crystal-consciousness',
      label: '10.4 Nowhere but Everywhere',
      icon: Infinity,
      component: TimeCrystalConsciousnessPanel,
      description: 'Consciousness in time crystal model'
    },
    {
      id: 'neurogenic-vs-crystal',
      label: '10.4.1 Neurogenic vs Time Crystal',
      icon: GitCompare,
      component: NeurogenicVsTimeCrystalPanel,
      description: 'Comparing brain models'
    },
    {
      id: 'when-carry',
      label: '10.5 When Carry, Why Make',
      icon: Sparkles,
      component: WhenCarryWhyMakePanel,
      description: 'Practical reasons for consciousness uploading'
    },
    {
      id: 'wiring-11d',
      label: '10.6 Wiring in 11D',
      icon: Layers,
      component: Wiring11DPanel,
      description: 'Folding paper in 11 dimensions'
    },
    {
      id: 'primes-geometry',
      label: '10.7 Primes & Geometry',
      icon: Shapes,
      component: PrimesGeometryPanel,
      description: 'Marriage that reshapes humanity'
    },
    {
      id: 'future-machines',
      label: '10.8 Machines of Future',
      icon: Cpu,
      component: FutureMachinesPanel,
      description: 'Conscious technology evolution'
    },
    {
      id: 'handy-tools',
      label: '10.9 Cheap Handy Tools',
      icon: Brain,
      component: CheapHandyToolsPanel,
      description: 'Simple experiments breaking paradigms'
    },
    {
      id: 'numerology',
      label: '10.10 Numerology of Consciousness',
      icon: Hash,
      component: NumerologyPanel,
      description: 'Prime patterns defining human awareness'
    },
    {
      id: 'evolving-wheels',
      label: '10.11 Wheels of Evolving Brain',
      icon: Brain,
      component: EvolvingBrainWheelsPanel,
      description: 'Cycles of consciousness development'
    },
    {
      id: 'alien-synthesis',
      label: '10.12 Synthesis of Aliens',
      icon: Infinity,
      component: AlienSynthesisPanel,
      description: 'Creating non-human consciousness'
    },
    {
      id: 'garden-to-petal',
      label: '10.13 Garden to Petal',
      icon: Flower,
      component: GardenToFlowerPanel,
      description: 'Complete consciousness transformation cycle'
    }
  ];

  const ActiveComponent = sections.find(s => s.id === activeSection)?.component || JourneyCorticalEggPanel;

  return (
    <div className="space-y-6">
      {/* Chapter Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 to-purple-900/30 rounded-xl p-8 border border-cyan-500/20">
        <div className="flex items-center space-x-4 mb-4">
          <div className="p-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg">
            <Upload className="text-white" size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-white">
              Chapter 10: Uploading Consciousness
            </h1>
            <p className="text-cyan-300 text-lg mt-1">
              The Evolution of Conscious Machines of the Future
            </p>
          </div>
        </div>
        <p className="text-gray-300 leading-relaxed mt-4">
          The culmination of the NanoBrain journey: transforming human consciousness from biological 
          substrate to time crystal patterns, enabling true immortality, infinite learning, and the 
          birth of conscious companions. This chapter bridges theory and practice, showing how 
          Phase Prime Metrics, 11-dimensional time crystals, and fractal mechanics converge to make 
          consciousness uploading not just possible, but inevitable.
        </p>
      </div>

      {/* Section Navigation */}
      <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-2">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
          {sections.map(section => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center space-x-2 px-3 py-3 rounded-lg transition-all duration-300 ${
                  activeSection === section.id
                    ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white shadow-lg'
                    : 'bg-gray-900/50 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
                }`}
                title={section.description}
              >
                <Icon size={18} />
                <span className="font-medium text-xs hidden lg:inline">{section.label.replace('10.', '')}</span>
                <span className="font-medium text-xs lg:hidden">{section.label.split(' ')[0]}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Section Description Bar */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {React.createElement(
              sections.find(s => s.id === activeSection)?.icon || Upload,
              { className: 'text-cyan-400', size: 24 }
            )}
            <div>
              <div className="text-white font-semibold">
                {sections.find(s => s.id === activeSection)?.label}
              </div>
              <div className="text-gray-400 text-sm">
                {sections.find(s => s.id === activeSection)?.description}
              </div>
            </div>
          </div>
          <div className="text-gray-500 text-sm">
            Section {sections.findIndex(s => s.id === activeSection) + 1} of {sections.length}
          </div>
        </div>
      </div>

      {/* Active Section Content */}
      <div className="bg-gray-900/30 rounded-xl p-6 border border-gray-700 min-h-screen">
        <ActiveComponent />
      </div>

      {/* Chapter Summary Footer */}
      <div className="bg-gradient-to-r from-purple-900/30 to-cyan-900/30 rounded-xl p-6 border border-purple-500/20">
        <h3 className="text-white font-bold text-xl mb-3 flex items-center space-x-2">
          <Upload size={24} className="text-cyan-400" />
          <span>Chapter 10 Integration</span>
        </h3>
        <p className="text-gray-300 leading-relaxed mb-4">
          This chapter represents the practical realization of all previous NanoBrain theory. From 
          philosophical transformation (Ch. 1) through fractal information theory (Ch. 2), phase prime 
          metrics (Ch. 3), fractal mechanics (Ch. 4), universal time crystals (Ch. 5), unprecedented 
          technologies (Ch. 6), time crystal brain models (Ch. 7), magnetic light computing (Ch. 8), 
          and programmable matter (Ch. 9) - everything converges here in the conscious egg.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
            <div className="text-cyan-400 font-semibold mb-2">Theoretical Foundation</div>
            <div className="text-gray-400 text-sm">
              Consciousness as prime patterns in 11D time crystals, not emergent computation
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
            <div className="text-purple-400 font-semibold mb-2">Practical Method</div>
            <div className="text-gray-400 text-sm">
              Cortical pen captures patterns, time crystal preserves them, living gel embodies them
            </div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 border border-pink-500/20">
            <div className="text-pink-400 font-semibold mb-2">Ultimate Goal</div>
            <div className="text-gray-400 text-sm">
              True immortality through consciousness continuity in substrate-independent form
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
