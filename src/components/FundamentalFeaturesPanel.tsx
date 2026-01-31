import React, { useEffect, useState } from 'react';
import { Brain, Network, Calculator, GitBranch, Eye, Code2 } from 'lucide-react';
import { 
  AtomeseNode, 
  PhilosophicalTransformation, 
  FractalTape, 
  PhasePrimeMetric, 
  CognitiveFlowchart 
} from '../types';
import { AtomeseSchemePanel } from './AtomeseSchemePanel';

interface FundamentalFeaturesProps {
  fundamentalFeatures: {
    atomeseNodes: AtomeseNode[];
    philosophicalTransformation: PhilosophicalTransformation | null;
    fractalTape: FractalTape | null;
    phasePrimeMetric: PhasePrimeMetric | null;
    cognitiveFlowcharts: CognitiveFlowchart[];
    initializeAllFeatures: () => {
      philosophicalTransformation: PhilosophicalTransformation;
      fractalTape: FractalTape;
      phasePrimeMetric: PhasePrimeMetric;
      cognitiveFlowcharts: CognitiveFlowchart[];
      totalAtomeseNodes: number;
    };
    totalNodes: number;
    averageAttention: number;
    fundamentalPrimes: number[];
  };
}

export const FundamentalFeaturesPanel: React.FC<FundamentalFeaturesProps> = ({ 
  fundamentalFeatures 
}) => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [activeFeature, setActiveFeature] = useState<'philosophical' | 'fractal' | 'prime-metric' | 'atomese-scheme'>('philosophical');

  useEffect(() => {
    if (!isInitialized && fundamentalFeatures.initializeAllFeatures) {
      fundamentalFeatures.initializeAllFeatures();
      setIsInitialized(true);
    }
  }, [fundamentalFeatures, isInitialized]);

  const getFeatureStatus = (feature: PhilosophicalTransformation | FractalTape | PhasePrimeMetric | null) => {
    return feature ? 'Active' : 'Inactive';
  };

  const getAttentionColor = (attention: number) => {
    if (attention > 0.8) return 'text-green-400';
    if (attention > 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-gray-900/90 backdrop-blur-sm rounded-lg p-6 border border-cyan-500/30">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-6 h-6 text-cyan-400" />
        <h2 className="text-xl font-bold text-white">Fundamental Features System</h2>
        <div className={`px-2 py-1 rounded text-xs ${isInitialized ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'}`}>
          {isInitialized ? 'Initialized' : 'Initializing...'}
        </div>
      </div>

      {/* Feature Selection Tabs */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'philosophical', label: 'Philosophical', icon: Brain },
          { id: 'fractal', label: 'Fractal IT', icon: Network },
          { id: 'prime-metric', label: 'Phase Prime', icon: Calculator },
          { id: 'atomese-scheme', label: 'Atomese/Scheme', icon: Code2 }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveFeature(id as 'philosophical' | 'fractal' | 'prime-metric' | 'atomese-scheme')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeFeature === id
                ? 'bg-cyan-600/30 text-cyan-400 border border-cyan-500/50'
                : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">AtomSpace Nodes</div>
          <div className="text-2xl font-bold text-white">{fundamentalFeatures.totalNodes}</div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Avg Attention</div>
          <div className={`text-2xl font-bold ${getAttentionColor(fundamentalFeatures.averageAttention)}`}>
            {fundamentalFeatures.averageAttention.toFixed(3)}
          </div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Prime Count</div>
          <div className="text-2xl font-bold text-purple-400">{fundamentalFeatures.fundamentalPrimes.length}</div>
        </div>
      </div>

      {/* Feature Details */}
      {activeFeature === 'philosophical' && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-5 h-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-white">Philosophical Transformation</h3>
            <span className={`text-sm px-2 py-1 rounded ${
              fundamentalFeatures.philosophicalTransformation ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
            }`}>
              {getFeatureStatus(fundamentalFeatures.philosophicalTransformation)}
            </span>
          </div>
          
          {fundamentalFeatures.philosophicalTransformation && (
            <div className="space-y-3">
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Research Fields</div>
                <div className="flex flex-wrap gap-1">
                  {fundamentalFeatures.philosophicalTransformation.researchFields.slice(0, 5).map((field: string, idx: number) => (
                    <span key={idx} className="bg-blue-600/20 text-blue-400 px-2 py-1 rounded text-xs">
                      Field {idx + 1}
                    </span>
                  ))}
                  {fundamentalFeatures.philosophicalTransformation.researchFields.length > 5 && (
                    <span className="text-gray-400 text-xs">+{fundamentalFeatures.philosophicalTransformation.researchFields.length - 5} more</span>
                  )}
                </div>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Universe Model</div>
                <div className="text-cyan-400 capitalize">{fundamentalFeatures.philosophicalTransformation.universeModel}</div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeFeature === 'fractal' && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 mb-3">
            <Network className="w-5 h-5 text-green-400" />
            <h3 className="text-lg font-semibold text-white">Fractal Information Theory</h3>
            <span className={`text-sm px-2 py-1 rounded ${
              fundamentalFeatures.fractalTape ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
            }`}>
              {getFeatureStatus(fundamentalFeatures.fractalTape)}
            </span>
          </div>

          {fundamentalFeatures.fractalTape && (
            <div className="space-y-3">
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Geometric Musical Language</div>
                <div className="text-green-400">
                  {fundamentalFeatures.fractalTape.geometricMusicalLanguage.basePatternsCount} Base Patterns
                </div>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Nested Spheres</div>
                <div className="text-green-400">{fundamentalFeatures.fractalTape.nestedSpheres} Configuration</div>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Waveform Transformation</div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${fundamentalFeatures.fractalTape.geometricMusicalLanguage.waveformTransformation ? 'bg-green-400' : 'bg-red-400'}`}></div>
                  <span className="text-white text-sm">
                    {fundamentalFeatures.fractalTape.geometricMusicalLanguage.waveformTransformation ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeFeature === 'prime-metric' && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 mb-3">
            <Calculator className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Phase Prime Metric</h3>
            <span className={`text-sm px-2 py-1 rounded ${
              fundamentalFeatures.phasePrimeMetric ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
            }`}>
              {getFeatureStatus(fundamentalFeatures.phasePrimeMetric)}
            </span>
          </div>

          {fundamentalFeatures.phasePrimeMetric && (
            <div className="space-y-3">
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Metric Classes</div>
                <div className="text-purple-400">
                  {fundamentalFeatures.phasePrimeMetric.metricClasses.length} Active Classes
                </div>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Prime Operators</div>
                <div className="text-purple-400">
                  {fundamentalFeatures.phasePrimeMetric.primeOperators.length} Operators
                </div>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Fundamental Primes</div>
                <div className="flex flex-wrap gap-1">
                  {fundamentalFeatures.fundamentalPrimes.slice(0, 10).map((prime: number, idx: number) => (
                    <span key={idx} className="bg-purple-600/20 text-purple-400 px-2 py-1 rounded text-xs">
                      {prime}
                    </span>
                  ))}
                  {fundamentalFeatures.fundamentalPrimes.length > 10 && (
                    <span className="text-gray-400 text-xs">+{fundamentalFeatures.fundamentalPrimes.length - 10} more</span>
                  )}
                </div>
              </div>
              
              <div className="bg-gray-800/30 rounded-lg p-3">
                <div className="text-sm text-gray-400 mb-2">Symmetry Encoding</div>
                <div className="text-purple-400 capitalize">
                  {fundamentalFeatures.phasePrimeMetric.symmetryEncoding}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeFeature === 'atomese-scheme' && (
        <div className="space-y-4">
          <AtomeseSchemePanel />
        </div>
      )}

      {/* Cognitive Flowcharts */}
      <div className="mt-6 pt-6 border-t border-gray-700/50">
        <div className="flex items-center gap-2 mb-3">
          <GitBranch className="w-5 h-5 text-orange-400" />
          <h3 className="text-lg font-semibold text-white">Cognitive Flowcharts</h3>
          <span className="text-sm bg-orange-600/20 text-orange-400 px-2 py-1 rounded">
            {fundamentalFeatures.cognitiveFlowcharts.length} Active
          </span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {fundamentalFeatures.cognitiveFlowcharts.map((flowchart: CognitiveFlowchart, idx: number) => (
            <div key={idx} className="bg-gray-800/30 rounded-lg p-3">
              <div className="text-sm font-medium text-white mb-1">
                {flowchart.featureType.charAt(0).toUpperCase() + flowchart.featureType.slice(1)}
              </div>
              <div className="text-xs text-gray-400">
                {flowchart.nodes.length} Nodes, {flowchart.connections.length} Connections
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Attention Allocation */}
      <div className="mt-6 pt-6 border-t border-gray-700/50">
        <div className="flex items-center gap-2 mb-3">
          <Eye className="w-5 h-5 text-yellow-400" />
          <h3 className="text-lg font-semibold text-white">Adaptive Attention Allocation</h3>
        </div>
        
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="text-sm text-gray-400 mb-2">System Status</div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
              <span className="text-green-400 text-sm">Auto-allocation Active</span>
            </div>
            <div className="text-gray-400 text-sm">
              Update Interval: 2s
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};