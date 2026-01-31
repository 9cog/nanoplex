import React, { useState } from 'react';
import { Calculator, Activity, Clock, TrendingUp, BookOpen, GitBranch, Database } from 'lucide-react';
import { PhasePrimeMetricPanel } from './PhasePrimeMetricPanel';
import { PrimeOperatorsPanel } from './PrimeOperatorsPanel';
import { TimeCrystalDecomposition } from './TimeCrystalDecomposition';
import { OrderedFactorMetric } from './OrderedFactorMetric';
import { PPMUsageGuide } from './PPMUsageGuide';
import { ThreeClassesOfPrimes } from './ThreeClassesOfPrimes';
import { DiabetesBigDataDemo } from './DiabetesBigDataDemo';

// Comprehensive Chapter 3 Panel with all sections
export const Chapter3Panel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('overview');

  const sections = [
    { id: 'overview', label: '3.1 PPM Overview', icon: Calculator, component: PhasePrimeMetricPanel },
    { id: 'operators', label: '3.12 Operators', icon: Activity, component: PrimeOperatorsPanel },
    { id: 'decomposition', label: '3.1.3 Decomposition', icon: Clock, component: TimeCrystalDecomposition },
    { id: 'ordered-factor', label: '3.1.2 OF Metric', icon: TrendingUp, component: OrderedFactorMetric },
    { id: 'usage', label: '3.14 Usage Guide', icon: BookOpen, component: PPMUsageGuide },
    { id: 'three-classes', label: '3.16 Three Classes', icon: GitBranch, component: ThreeClassesOfPrimes },
    { id: 'diabetes', label: '3.15 Diabetes Demo', icon: Database, component: DiabetesBigDataDemo }
  ];

  const ActiveComponent = sections.find(s => s.id === activeSection)?.component || PhasePrimeMetricPanel;

  return (
    <div className="space-y-6">
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
                    ? 'bg-cyan-900/30 border-cyan-700 text-cyan-400'
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
    </div>
  );
};
