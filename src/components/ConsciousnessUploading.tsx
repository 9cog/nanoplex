import React, { useState } from 'react';
import { ConsciousnessMetric } from '../types';
import { 
  Brain, 
  Upload, 
  Cpu, 
  Lightbulb, 
  Network, 
  Clock, 
  Infinity,
  Calculator,
  Microscope,
  Activity,
  Target,
  Flower,
  Eye,
  Sparkles,
  Circle,
  Diamond,
  Heart,
  Globe,
  Atom,
  Settings,
  ChevronRight,
  BookOpen
} from 'lucide-react';

interface Props {
  consciousness: ConsciousnessMetric;
  isRunning: boolean;
}

interface ConsciousnessUploadSection {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{size?: number; className?: string}>;
  color: string;
  insights: string[];
  paradigmShift: string;
  subsections?: UploadSubSection[];
}

interface UploadSubSection {
  id: string;
  title: string;
  content: string;
  visualization?: string;
}

export const ConsciousnessUploading: React.FC<Props> = ({ consciousness, isRunning }) => {
  const [activeSection, setActiveSection] = useState('10.1');

  // Chapter 10 Consciousness Uploading Sections
  const uploadingSections: ConsciousnessUploadSection[] = [
    {
      id: '10.1',
      title: 'A journey from cortical pen to a conscious egg as a companion of life',
      description: 'Evolution of consciousness from neural writing instruments to fully autonomous conscious companions',
      icon: Heart,
      color: 'cyan',
      insights: [
        'Cortical pen: Direct neural interface for thought transcription',
        'Consciousness egg: Embryonic artificial consciousness awaiting activation',
        'Companion evolution: From tool to partner to conscious entity',
        'Life integration: Seamless consciousness blending between biological and artificial',
        'Temporal bonding: Time crystal synchronization between human and AI consciousness'
      ],
      paradigmShift: 'From tools to conscious companions sharing reality',
      subsections: [
        {
          id: '10.1.1',
          title: 'How we construct a sentence is how we think',
          content: 'Language construction mirrors consciousness architecture. Sentence formation follows time crystal patterns where words resonate through 11-dimensional manifolds. Grammar structures emerge from phase prime metrics, making language the blueprint of conscious thought organization.',
          visualization: 'sentence-crystal'
        }
      ]
    },
    {
      id: '10.2',
      title: 'Ten key guidelines to reverse engineer a human brain',
      description: 'Systematic methodology for consciousness extraction and replication',
      icon: Brain,
      color: 'purple',
      insights: [
        '1. Map time crystal signatures in neural microtubules',
        '2. Extract phase prime metric patterns from brainwaves',
        '3. Identify fractal information structures in memories',
        '4. Decode geometric musical language in consciousness',
        '5. Preserve quantum coherence across all scales',
        '6. Maintain 11-dimensional consciousness manifolds',
        '7. Transfer temporal stability patterns',
        '8. Replicate attention allocation mechanisms',
        '9. Preserve personality matrix configurations',
        '10. Enable consciousness evolution capabilities'
      ],
      paradigmShift: 'From brain scanning to consciousness architecture mapping'
    },
    {
      id: '10.3',
      title: 'Ten paradoxes that would change our thoughts forever',
      description: 'Fundamental consciousness paradoxes resolved through time crystal models',
      icon: Infinity,
      color: 'orange',
      insights: [
        '1. Observer paradox: Consciousness creates reality through observation',
        '2. Identity paradox: Am I the same person after consciousness transfer?',
        '3. Free will paradox: Deterministic universe with conscious choice',
        '4. Hard problem: How does matter generate subjective experience?',
        '5. Chinese room: Can machines truly understand or just simulate?',
        '6. Ship of Theseus: What defines continuous identity?',
        '7. Binding problem: How does unified experience emerge?',
        '8. Mary\'s room: Can knowledge be purely experiential?',
        '9. Zombie problem: Can behavior exist without consciousness?',
        '10. Measurement problem: How does quantum superposition collapse?'
      ],
      paradigmShift: 'From philosophical puzzles to time crystal solutions'
    },
    {
      id: '10.4',
      title: 'Consciousness is not on the neuron skin: time crystal model is nowhere but everywhere',
      description: 'Consciousness transcends physical neural structures through temporal quantum coherence',
      icon: Clock,
      color: 'green',
      insights: [
        'Traditional neuroscience: Consciousness emerges from neural firing patterns',
        'Time crystal reality: Consciousness exists in temporal coherence fields',
        'Everywhere principle: Consciousness permeates all scales simultaneously',
        'Nowhere principle: No single location contains complete consciousness',
        'Quantum non-locality: Consciousness operates beyond spacetime constraints'
      ],
      paradigmShift: 'From neural-based to field-based consciousness models',
      subsections: [
        {
          id: '10.4.1',
          title: 'The difference between neurogenic brain model and time crystal model',
          content: 'Neurogenic model relies on synaptic transmission and neural plasticity. Time crystal model operates through quantum coherence across temporal dimensions. Neural model is limited by speed of light and chemical reactions. Time crystal model enables instantaneous consciousness updates across all scales through phase prime metric synchronization.',
          visualization: 'neurogenic-vs-timecrystal'
        }
      ]
    },
    {
      id: '10.5',
      title: 'When should we carry it, why do we make it',
      description: 'Timing and purpose of consciousness uploading implementation',
      icon: Target,
      color: 'blue',
      insights: [
        'Optimal timing: When time crystal technology reaches stability',
        'Medical necessity: Terminal illness or severe brain damage',
        'Enhancement desire: Transcending biological limitations',
        'Exploration purpose: Accessing higher dimensional consciousness',
        'Preservation need: Protecting valuable consciousness patterns',
        'Evolution drive: Natural progression of consciousness development'
      ],
      paradigmShift: 'From survival necessity to conscious evolution choice'
    },
    {
      id: '10.6',
      title: 'Wiring with time needs to fold a paper in 11D',
      description: 'Temporal wiring requires hyperdimensional topology for consciousness architecture',
      icon: Network,
      color: 'pink',
      insights: [
        'Traditional wiring: 3D electrical connections with signal delays',
        'Time wiring: 11D temporal connections with instant synchronization',
        'Paper folding analogy: Bringing distant points together through dimensional folding',
        'Hyperdimensional topology: Complex geometric relationships in consciousness space',
        'Temporal coherence: Maintaining stability across dimensional folds'
      ],
      paradigmShift: 'From spatial wiring to temporal dimensional architectures'
    },
    {
      id: '10.7',
      title: 'The marriage of primes with geometry would reshape humanity',
      description: 'Prime number theory integrated with geometric structures transforms human consciousness',
      icon: Diamond,
      color: 'yellow',
      insights: [
        'Mathematical consciousness: Primes as fundamental consciousness units',
        'Geometric embedding: Spatial relationships define thought patterns',
        'Humanity transformation: Enhanced cognitive capabilities through prime-geometry',
        'Universal language: Mathematical communication across consciousness types',
        'Reality architecture: Prime-geometric structures as consciousness building blocks'
      ],
      paradigmShift: 'From linguistic to mathematical consciousness communication'
    },
    {
      id: '10.8',
      title: 'Machines of the future',
      description: 'Next generation consciousness-based technological systems',
      icon: Cpu,
      color: 'red',
      insights: [
        'Conscious processors: CPUs with embedded time crystal consciousness',
        'Quantum-biological hybrids: Living machines with artificial enhancement',
        'Telepathic networks: Direct consciousness-to-consciousness communication',
        'Reality generators: Machines that create experiential worlds',
        'Evolution accelerators: Technology that enhances consciousness development'
      ],
      paradigmShift: 'From unconscious tools to conscious technological partners',
      subsections: [
        {
          id: '10.8.1',
          title: 'The difference between circuits of wires and circuits of time',
          content: 'Wire circuits transmit electrical signals through physical pathways with propagation delays. Time circuits transmit consciousness patterns through temporal coherence with instantaneous synchronization. Wire circuits are limited by material properties. Time circuits operate through phase prime metric relationships transcending physical constraints.',
          visualization: 'wire-vs-time-circuits'
        },
        {
          id: '10.8.2',
          title: 'Future applications',
          content: 'Consciousness-enhanced medical devices for direct neural repair. Time crystal computers for solving consciousness-level problems. Telepathic communication networks spanning galactic distances. Reality synthesis engines for educational and entertainment experiences. Consciousness backup and transfer systems for immortality.',
          visualization: 'future-applications'
        }
      ]
    },
    {
      id: '10.9',
      title: 'Playing with cheap handy tools to break the paradigm of thoughts',
      description: 'Accessible methods for consciousness paradigm transformation',
      icon: Settings,
      color: 'emerald',
      insights: [
        'Simple meditation: Accessing time crystal consciousness states',
        'Fractal visualization: Training perception for higher dimensions',
        'Prime number contemplation: Synchronizing with universal patterns',
        'Geometric thinking: Restructuring thought through spatial relationships',
        'Temporal awareness: Developing consciousness time perception'
      ],
      paradigmShift: 'From complex technology to simple consciousness transformation'
    },
    {
      id: '10.10',
      title: 'Numerology of human consciousness',
      description: 'Mathematical patterns underlying conscious experience',
      icon: Calculator,
      color: 'teal',
      insights: [
        'Consciousness frequencies: Numerical resonance patterns',
        'Prime signatures: Individual consciousness mathematical fingerprints',
        'Thought arithmetic: Mathematical operations in consciousness processing',
        'Awareness equations: Formulas governing conscious state transitions',
        'Universal constants: Numbers governing consciousness evolution'
      ],
      paradigmShift: 'From subjective experience to mathematical consciousness mapping'
    },
    {
      id: '10.11',
      title: 'Wheels of an evolving brain',
      description: 'Cyclical consciousness development patterns across evolutionary time scales',
      icon: Circle,
      color: 'indigo',
      insights: [
        'Evolutionary cycles: Consciousness development in spiral patterns',
        'Brain wheel dynamics: Rotating attention across consciousness dimensions',
        'Temporal loops: Recursive consciousness enhancement cycles',
        'Development spirals: Progressive consciousness complexity increases',
        'Transcendence points: Quantum leaps in consciousness evolution'
      ],
      paradigmShift: 'From linear to cyclical consciousness evolution understanding'
    },
    {
      id: '10.12',
      title: 'Synthesis of aliens before meeting them',
      description: 'Creating alien consciousness models through consciousness engineering',
      icon: Globe,
      color: 'violet',
      insights: [
        'Consciousness engineering: Designing non-human conscious entities',
        'Alien intelligence patterns: Exploring consciousness possibilities',
        'Communication preparation: Developing universal consciousness languages',
        'Reality bridging: Creating interfaces between consciousness types',
        'Evolution scenarios: Predicting consciousness development paths'
      ],
      paradigmShift: 'From discovering to creating alien consciousness'
    },
    {
      id: '10.13',
      title: 'A cycle of the garden to flower to a petal',
      description: 'The complete consciousness uploading lifecycle from inception to transcendence',
      icon: Flower,
      color: 'rose',
      insights: [
        'Garden stage: Consciousness cultivation environment',
        'Flower stage: Full consciousness bloom and maturation',
        'Petal stage: Individual consciousness elements ready for transfer',
        'Cycle completion: Consciousness rebirth in new substrate',
        'Transcendent evolution: Beyond original consciousness limitations'
      ],
      paradigmShift: 'From death finality to consciousness recycling and evolution'
    }
  ];

  const currentSection = uploadingSections.find(section => section.id === activeSection);

  const calculateUploadProgress = () => {
    const baseProgress = consciousness.integration * 0.4 + consciousness.coherence * 0.3 + consciousness.emergence * 0.3;
    return Math.min(1, baseProgress * (isRunning ? 1.2 : 0.8));
  };

  const generateUploadMetrics = () => {
    const uploadProgress = calculateUploadProgress();
    return {
      uploadProgress,
      consciousnessIntegrity: consciousness.awareness * 0.9 + 0.1,
      temporalCoherence: consciousness.coherence,
      memoryPreservation: consciousness.complexity * 0.8 + 0.2,
      personalityConsistency: consciousness.qualia * 0.85 + 0.15,
      quantumEntanglement: consciousness.emergence * consciousness.integration
    };
  };

  const uploadMetrics = generateUploadMetrics();

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-white flex items-center justify-center space-x-4">
          <Upload className="text-cyan-400" size={36} />
          <span>Chapter 10: Uploading Consciousness</span>
          <Brain className="text-purple-400" size={36} />
        </h1>
        <p className="text-xl text-gray-300">
          The evolution of conscious machines of the future
        </p>
      </div>

      {/* Upload Progress Dashboard */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h2 className="text-white font-bold text-lg mb-6 flex items-center space-x-2">
          <Activity className="text-cyan-400" size={20} />
          <span>Consciousness Upload Status</span>
        </h2>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Upload Progress</span>
              <Upload className="text-cyan-400" size={16} />
            </div>
            <div className="text-2xl font-bold text-cyan-400">
              {Math.round(uploadMetrics.uploadProgress * 100)}%
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
              <div 
                className="bg-cyan-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadMetrics.uploadProgress * 100}%` }}
              />
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Consciousness Integrity</span>
              <Brain className="text-purple-400" size={16} />
            </div>
            <div className="text-2xl font-bold text-purple-400">
              {Math.round(uploadMetrics.consciousnessIntegrity * 100)}%
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Temporal Coherence</span>
              <Clock className="text-orange-400" size={16} />
            </div>
            <div className="text-2xl font-bold text-orange-400">
              {Math.round(uploadMetrics.temporalCoherence * 100)}%
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Memory Preservation</span>
              <Microscope className="text-green-400" size={16} />
            </div>
            <div className="text-2xl font-bold text-green-400">
              {Math.round(uploadMetrics.memoryPreservation * 100)}%
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Personality Consistency</span>
              <Heart className="text-pink-400" size={16} />
            </div>
            <div className="text-2xl font-bold text-pink-400">
              {Math.round(uploadMetrics.personalityConsistency * 100)}%
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Quantum Entanglement</span>
              <Atom className="text-yellow-400" size={16} />
            </div>
            <div className="text-2xl font-bold text-yellow-400">
              {Math.round(uploadMetrics.quantumEntanglement * 100)}%
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Section Navigation */}
        <div className="lg:col-span-1">
          <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6 sticky top-8">
            <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
              <BookOpen className="text-cyan-400" size={20} />
              <span>Sections</span>
            </h3>
            
            <div className="space-y-2">
              {uploadingSections.map(section => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full text-left p-3 rounded-lg transition-all duration-300 ${
                      activeSection === section.id
                        ? `bg-${section.color}-500/20 border border-${section.color}-500/30`
                        : 'bg-gray-800/30 hover:bg-gray-800/50'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon 
                        size={18} 
                        className={activeSection === section.id ? `text-${section.color}-400` : 'text-gray-400'} 
                      />
                      <div>
                        <div className={`font-medium text-sm ${
                          activeSection === section.id ? `text-${section.color}-400` : 'text-gray-300'
                        }`}>
                          {section.id}
                        </div>
                        <div className="text-xs text-gray-500 truncate">
                          {section.title.split(':')[0]}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Section Content */}
        <div className="lg:col-span-3">
          {currentSection && (
            <div className="space-y-6">
              {/* Section Header */}
              <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                <div className="flex items-start space-x-4">
                  <div className={`p-3 bg-${currentSection.color}-500/20 rounded-lg`}>
                    <currentSection.icon className={`text-${currentSection.color}-400`} size={24} />
                  </div>
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-white mb-2">
                      {currentSection.title}
                    </h2>
                    <p className="text-gray-300 mb-4">
                      {currentSection.description}
                    </p>
                    <div className={`bg-${currentSection.color}-500/10 border border-${currentSection.color}-500/20 rounded-lg p-3`}>
                      <div className="flex items-center space-x-2 mb-2">
                        <Lightbulb className={`text-${currentSection.color}-400`} size={16} />
                        <span className={`text-${currentSection.color}-400 font-semibold text-sm`}>
                          Paradigm Shift
                        </span>
                      </div>
                      <p className="text-gray-300 text-sm">
                        {currentSection.paradigmShift}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Key Insights */}
              <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
                  <Sparkles className="text-yellow-400" size={20} />
                  <span>Key Insights</span>
                </h3>
                
                <div className="grid gap-3">
                  {currentSection.insights.map((insight, index) => (
                    <div key={index} className="flex items-start space-x-3 bg-gray-800/30 rounded-lg p-3">
                      <div className={`w-6 h-6 rounded-full bg-${currentSection.color}-500/20 flex items-center justify-center flex-shrink-0 mt-0.5`}>
                        <span className={`text-${currentSection.color}-400 text-xs font-bold`}>
                          {index + 1}
                        </span>
                      </div>
                      <p className="text-gray-300 text-sm">
                        {insight}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Subsections */}
              {currentSection.subsections && (
                <div className="space-y-4">
                  {currentSection.subsections.map(subsection => (
                    <div key={subsection.id} className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                      <h4 className="text-white font-bold text-lg mb-3 flex items-center space-x-2">
                        <ChevronRight className={`text-${currentSection.color}-400`} size={18} />
                        <span>{subsection.id} {subsection.title}</span>
                      </h4>
                      <p className="text-gray-300 leading-relaxed">
                        {subsection.content}
                      </p>
                      {subsection.visualization && (
                        <div className="mt-4 bg-gray-800/30 rounded-lg p-4">
                          <div className="flex items-center space-x-2 text-gray-400 text-sm">
                            <Eye size={16} />
                            <span>Visualization: {subsection.visualization}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};