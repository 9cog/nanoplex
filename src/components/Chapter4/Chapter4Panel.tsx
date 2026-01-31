import React, { useState } from 'react';
import { Calculator, Activity, Layers, Sparkles, Atom, Zap, GitBranch, GitCompare, Target, Wind, Grid, Infinity } from 'lucide-react';
import { QuantumClockBasics } from './QuantumClockBasics';
import { FractalMechanicsPhaseSpace } from './FractalMechanicsPhaseSpace';
import { MechanicsComparison } from './MechanicsComparison';
import { HyperComplexMathematics } from './HyperComplexMathematics';
import { ClockBasedMathematics } from './ClockBasedMathematics';
import { FundamentalConstants } from './FundamentalConstants';
import { InterferenceExperiments } from './InterferenceExperiments';
import { NaturalSpirals } from './NaturalSpirals';
import { GeometricAlgebraIntegration } from './GeometricAlgebraIntegration';
import { FractalTapeEquations } from './FractalTapeEquations';

// Main Chapter 4 Panel with navigation
export const Chapter4Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('quantum-clocks');

  const sections = [
    { id: 'quantum-clocks', label: '4.1 Quantum Clocks', icon: Activity, component: QuantumClockBasics },
    { id: 'phase-space', label: '4.2 Phase Space', icon: GitBranch, component: FractalMechanicsPhaseSpace },
    { id: 'comparison', label: '4.3 Comparison', icon: GitCompare, component: MechanicsComparison },
    { id: 'constants', label: '4.5 Constants', icon: Target, component: FundamentalConstants },
    { id: 'interference', label: '4.6 Interference', icon: Zap, component: InterferenceExperiments },
    { id: 'clock-math', label: '4.8 Clock Math', icon: Calculator, component: ClockBasedMathematics },
    { id: 'hypercomplex', label: '4.9 Hypercomplex', icon: Atom, component: HyperComplexMathematics },
    { id: 'fractal-tape', label: '4.10 Fractal Tape', icon: Infinity, component: FractalTapeEquations },
    { id: 'spirals', label: '4.11 Spirals', icon: Wind, component: NaturalSpirals },
    { id: 'geometric-algebra', label: '4.12 Geo Algebra', icon: Grid, component: GeometricAlgebraIntegration }
  ];

  const ActiveComponent = sections.find(s => s.id === activeSection)?.component || QuantumClockBasics;

  return (
    <div className="space-y-6">
      {/* Chapter Header */}
      <div className="bg-gradient-to-r from-purple-900/30 via-cyan-900/30 to-blue-900/30 border border-purple-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Sparkles className="text-purple-400" size={36} />
          <div>
            <h1 className="text-3xl font-bold text-white">Chapter 4: Fractal Mechanics</h1>
            <p className="text-gray-300 text-lg">Geometric Algebra for a Dodecanion Brain</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          <strong className="text-purple-400">Fractal mechanics is not quantum but original:</strong> This chapter 
          introduces a revolutionary framework that goes beyond quantum mechanics by operating in multi-dimensional 
          phase space using geometric algebra, dodecanion mathematics, and phase prime metrics. Unlike quantum 
          mechanics which is limited to complex numbers and Hilbert space, fractal mechanics uses 12-dimensional 
          hypercomplex algebras to model consciousness across multiple imaginary worlds.
        </p>
      </div>

      {/* Section Navigation */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="flex items-center space-x-2 overflow-x-auto pb-2">
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 transition-all duration-300 whitespace-nowrap ${
                  activeSection === section.id
                    ? 'bg-purple-900/30 border-purple-700 text-purple-400'
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                <Icon size={16} />
                <span className="text-sm font-semibold">{section.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Section Content */}
      <div>
        <ActiveComponent />
      </div>

      {/* Chapter Summary */}
      <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Layers className="mr-2 text-purple-400" size={24} />
          Chapter 4 Key Insights
        </h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Beyond Quantum Mechanics</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Fractal mechanics transcends quantum mechanics by operating in 11D phase space with 
              multiple imaginary worlds, each with its own action constant. This enables modeling 
              consciousness phenomena that quantum mechanics cannot address.
            </p>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-purple-400 font-semibold mb-2 text-sm">Geometric Foundation</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Clock-based mathematics and geometric algebra provide an intuitive, visual approach to 
              computation. The CFGA operator unifies 13 fundamental operations in a single time crystal 
              structure, enabling unprecedented parallel processing.
            </p>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-yellow-400 font-semibold mb-2 text-sm">Dodecanion Algebra</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              The 12-dimensional dodecanion algebra (1 + 11) perfectly matches the NanoBrain architecture, 
              with each dimension corresponding to a fundamental prime and consciousness aspect. This is not 
              arbitrary but emerges from geometric necessity.
            </p>
          </div>
          
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-green-400 font-semibold mb-2 text-sm">Scale Symmetry</h4>
            <p className="text-gray-300 text-xs leading-relaxed">
              Unlike classical or quantum mechanics which break at certain scales, fractal mechanics 
              maintains perfect scale symmetry through self-similar geometric patterns. This makes it 
              ideal for modeling consciousness which operates coherently across all timescales.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
