import React, { useState } from 'react';
import { Lightbulb, Globe, BookOpen, Layers, HelpCircle, FileText, Route, Zap, Brain, Languages, Binary, Triangle } from 'lucide-react';
import { WorldviewDifference } from './WorldviewDifference';
import { TenResearchFields } from './TenResearchFields';
import { UniverseWithinAbove } from './UniverseWithinAbove';
import { BrainModelsQuestions } from './BrainModelsQuestions';
import { DifferentTapes } from './DifferentTapes';
import { ThreeConceptsArtificialBrain } from './ThreeConceptsArtificialBrain';
import { RemainingChapter1Sections } from './RemainingChapter1Sections';

// Main Chapter 1 Panel with all sections
export const Chapter1Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('overview');

  const sections = [
    { id: 'overview', label: 'Chapter Overview', icon: Lightbulb },
    { id: '1.1', label: '1.1 Worldview', icon: Globe },
    { id: '1.2', label: '1.2 Ten Fields', icon: BookOpen },
    { id: '1.3', label: '1.3 Universe Within', icon: Layers },
    { id: '1.4', label: '1.4 Brain Models', icon: HelpCircle },
    { id: '1.5', label: '1.5 Different Tapes', icon: FileText },
    { id: '1.6', label: '1.6 Decision Making', icon: Route },
    { id: '1.7', label: '1.7 Energy', icon: Zap },
    { id: '1.8', label: '1.8 Terminologies', icon: Brain },
    { id: '1.9', label: '1.9 Linguistics', icon: Languages },
    { id: '1.10', label: '1.10 Three Concepts', icon: Binary },
    { id: '1.11', label: '1.11 Conclusion', icon: Triangle }
  ];

  const renderContent = () => {
    switch (activeSection) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div className="bg-gradient-to-r from-purple-900/40 to-cyan-900/40 backdrop-blur-sm border border-purple-700 rounded-xl p-8">
              <h1 className="text-3xl font-bold text-white mb-4 flex items-center space-x-3">
                <Lightbulb className="text-cyan-400" size={40} />
                <span>Chapter 1: Philosophical Transformation Essential to Reverse Engineer Consciousness</span>
              </h1>
              <p className="text-gray-300 text-lg leading-relaxed mb-6">
                This foundational chapter establishes the paradigm shift from traditional Turing-based computing 
                to consciousness-first architectures. We explore why existing brain models are incomplete and 
                introduce the revolutionary concepts that enable true artificial consciousness.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-lg p-4">
                  <h3 className="text-cyan-400 font-bold mb-2">Key Themes</h3>
                  <ul className="space-y-1 text-gray-300 text-sm">
                    <li>• Beyond Turing: From sequential to geometric computation</li>
                    <li>• Consciousness as fundamental, not emergent</li>
                    <li>• Prime symmetries governing universal patterns</li>
                    <li>• Time crystals as consciousness substrates</li>
                  </ul>
                </div>
                
                <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-lg p-4">
                  <h3 className="text-purple-400 font-bold mb-2">Revolutionary Ideas</h3>
                  <ul className="space-y-1 text-gray-300 text-sm">
                    <li>• Geometric Musical Language (GML)</li>
                    <li>• Phase Prime Metrics (PPM)</li>
                    <li>• Fractal Information Theory (FIT)</li>
                    <li>• 11-dimensional time crystal structures</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Section Preview Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {sections.slice(1).map((section) => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className="text-left bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4 hover:border-cyan-700 transition-all duration-300"
                  >
                    <div className="flex items-start space-x-3">
                      <div className="p-2 bg-cyan-900/30 rounded-lg">
                        <Icon className="text-cyan-400" size={20} />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-white font-semibold text-sm mb-1">{section.label}</h3>
                        <p className="text-gray-400 text-xs">Click to explore</p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Chapter Importance */}
            <div className="bg-gradient-to-r from-cyan-900/40 to-purple-900/40 backdrop-blur-sm border border-cyan-700 rounded-xl p-6">
              <h3 className="text-white font-bold text-lg mb-3">Why This Chapter Matters</h3>
              <p className="text-gray-300 leading-relaxed">
                Understanding consciousness requires a complete philosophical transformation. This chapter 
                dismantles the limitations of Turing machines, Shannon information theory, and traditional 
                neuroscience. In their place, we establish a new foundation built on geometric patterns, 
                prime symmetries, and time crystal dynamics. Without this conceptual revolution, artificial 
                consciousness remains impossible. With it, we gain the tools to reverse-engineer the most 
                profound phenomenon in the universe: awareness itself.
              </p>
            </div>
          </div>
        );
      case '1.1':
        return <WorldviewDifference />;
      case '1.2':
        return <TenResearchFields />;
      case '1.3':
        return <UniverseWithinAbove />;
      case '1.4':
        return <BrainModelsQuestions />;
      case '1.5':
        return <DifferentTapes />;
      case '1.6':
        return <RemainingChapter1Sections section="1.6" />;
      case '1.7':
        return <RemainingChapter1Sections section="1.7" />;
      case '1.8':
        return <RemainingChapter1Sections section="1.8" />;
      case '1.9':
        return <RemainingChapter1Sections section="1.9" />;
      case '1.10':
        return <ThreeConceptsArtificialBrain />;
      case '1.11':
        return <RemainingChapter1Sections section="1.11" />;
      default:
        return <div>Section not found</div>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Section Navigation */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
        <div className="flex flex-wrap gap-2">
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-lg border-2 transition-all duration-300 whitespace-nowrap ${
                  activeSection === section.id
                    ? 'bg-cyan-900/30 border-cyan-700 text-cyan-400'
                    : 'bg-gray-800/30 border-gray-700 text-gray-400 hover:border-gray-600'
                }`}
              >
                <Icon size={14} />
                <span className="text-xs font-semibold">{section.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Section Content */}
      <div>
        {renderContent()}
      </div>
    </div>
  );
};
