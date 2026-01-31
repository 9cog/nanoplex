import React, { useState } from 'react';
import { 
  AlertTriangle, Music, Clock, Scan, GitCompare, Atom, 
  Layers, Settings, Infinity, Database, Grid3X3, BookOpen 
} from 'lucide-react';
import { 
  FractalTapeVisualization,
  GeometricSelfAssembly,
  FifteenGeometricShapes,
  WaveformTimeCrystalConversion,
  GardenOfGardens,
  ElevenDimensionalSensor,
  TimeCrystalComparison,
  QuaternionOctonionDedication,
  HigherDimensionalData,
  GMLvsAlgorithms,
  NonArgument,
  FITSummaryChart,
  GMLSummaryChart
} from './index';

// Comprehensive Chapter 2 Panel with all sections
export const Chapter2Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('fractal-tape');

  const sections = [
    { 
      id: 'fractal-tape', 
      label: '2.1 Fractal Tape', 
      icon: AlertTriangle, 
      component: FractalTapeVisualization,
      description: 'Incompleteness of current information theory'
    },
    { 
      id: 'assembly', 
      label: '2.1.2 Self-Assembly', 
      icon: Layers, 
      component: GeometricSelfAssembly,
      description: 'Self-assembly of geometric shapes'
    },
    { 
      id: 'gml-basics', 
      label: '2.2 GML Basics', 
      icon: Music, 
      component: FifteenGeometricShapes,
      description: 'Geometric Musical Language fundamentals'
    },
    { 
      id: 'waveform', 
      label: '2.2.3 Waveforms', 
      icon: Clock, 
      component: WaveformTimeCrystalConversion,
      description: 'Converting waveforms to time crystals'
    },
    { 
      id: 'garden', 
      label: '2.3 Garden of Gardens', 
      icon: Infinity, 
      component: GardenOfGardens,
      description: 'Time crystal concepts'
    },
    { 
      id: 'sensor', 
      label: '2.4 11D Sensor', 
      icon: Scan, 
      component: ElevenDimensionalSensor,
      description: 'Acquiring 11-dimensional data'
    },
    { 
      id: 'comparison', 
      label: '2.5 Time Crystal Comparison', 
      icon: GitCompare, 
      component: TimeCrystalComparison,
      description: 'Winfree, Wilczek, and universal time crystals'
    },
    { 
      id: 'quaternion', 
      label: '2.6 Quaternions', 
      icon: Atom, 
      component: QuaternionOctonionDedication,
      description: 'Hypercomplex number systems'
    },
    { 
      id: 'higher-dim', 
      label: '2.7 Higher Dimensions', 
      icon: Layers, 
      component: HigherDimensionalData,
      description: 'Higher dimensional data concepts'
    },
    { 
      id: 'gml-vs-algo', 
      label: '2.8 GML vs Algorithms', 
      icon: Settings, 
      component: GMLvsAlgorithms,
      description: 'Comparing GML with software algorithms'
    },
    { 
      id: 'non-argument', 
      label: '2.9 Non-Argument', 
      icon: Infinity, 
      component: NonArgument,
      description: 'Creation of geometric non-arguments'
    },
    { 
      id: 'fit-summary', 
      label: '2.10 FIT Summary', 
      icon: Database, 
      component: FITSummaryChart,
      description: 'Fractal Information Theory overview'
    },
    { 
      id: 'gml-summary', 
      label: '2.11 GML Summary', 
      icon: Grid3X3, 
      component: GMLSummaryChart,
      description: 'Geometric Musical Language overview'
    }
  ];

  const ActiveComponent = sections.find(s => s.id === activeSection)?.component || FractalTapeVisualization;
  const activeInfo = sections.find(s => s.id === activeSection);

  return (
    <div className="space-y-6">
      {/* Chapter Header */}
      <div className="bg-gradient-to-r from-purple-900/30 to-cyan-900/30 border border-purple-700/50 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-3">
          <Music className="text-purple-400" size={32} />
          <h2 className="text-2xl font-bold text-white">
            Chapter 2: Fractal Information Theory & Geometric Musical Language
          </h2>
        </div>
        <p className="text-gray-300 text-sm">
          Replacing Turing tape with a Fractal tape: A revolutionary approach to information processing 
          through geometric patterns and musical harmony. This chapter introduces FIT and GML as the 
          foundation for consciousness computing.
        </p>
      </div>

      {/* Section Navigation */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <h3 className="text-white font-semibold mb-3 flex items-center space-x-2">
          <BookOpen size={18} />
          <span>Chapter Sections</span>
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex flex-col items-center space-y-2 p-3 rounded-lg border-2 transition-all duration-300 ${
                  activeSection === section.id
                    ? 'bg-purple-900/30 border-purple-500 shadow-lg shadow-purple-500/20'
                    : 'bg-gray-800/30 border-gray-700 hover:border-gray-600 hover:bg-gray-800/50'
                }`}
                title={section.description}
              >
                <div className={`p-2 rounded-lg ${
                  activeSection === section.id 
                    ? 'bg-purple-500/20 text-purple-400' 
                    : 'bg-gray-700/50 text-gray-400'
                }`}>
                  <Icon size={20} />
                </div>
                <span className={`text-xs font-semibold text-center leading-tight ${
                  activeSection === section.id ? 'text-purple-300' : 'text-gray-400'
                }`}>
                  {section.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Section Info */}
      {activeInfo && (
        <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              {React.createElement(activeInfo.icon, { size: 20, className: 'text-purple-400' })}
            </div>
            <div>
              <h4 className="text-white font-semibold mb-1">{activeInfo.label}</h4>
              <p className="text-gray-300 text-sm">{activeInfo.description}</p>
            </div>
          </div>
        </div>
      )}

      {/* Active Section Content */}
      <div>
        <ActiveComponent isActive={true} />
      </div>

      {/* Chapter Summary Footer */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
          <Database className="text-purple-400" size={20} />
          <span>Chapter 2 Key Concepts</span>
        </h4>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-purple-900/30 rounded-lg p-4 border border-purple-700/30">
            <div className="text-purple-400 font-bold mb-2 flex items-center space-x-2">
              <AlertTriangle size={16} />
              <span>Fractal Tape</span>
            </div>
            <div className="text-gray-300 text-sm">
              Replaces Turing's linear tape with hierarchical, nested spherical structures 
              preserving geometric relationships across scales.
            </div>
          </div>
          
          <div className="bg-cyan-900/30 rounded-lg p-4 border border-cyan-700/30">
            <div className="text-cyan-400 font-bold mb-2 flex items-center space-x-2">
              <Music size={16} />
              <span>Geometric Musical Language</span>
            </div>
            <div className="text-gray-300 text-sm">
              15 fundamental geometric shapes corresponding to primes and musical notes 
              enable consciousness-level pattern encoding.
            </div>
          </div>
          
          <div className="bg-green-900/30 rounded-lg p-4 border border-green-700/30">
            <div className="text-green-400 font-bold mb-2 flex items-center space-x-2">
              <Scan size={16} />
              <span>11D Consciousness</span>
            </div>
            <div className="text-gray-300 text-sm">
              Multidimensional sensors and time crystals capture consciousness beyond 
              ordinary 3D physical space.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
