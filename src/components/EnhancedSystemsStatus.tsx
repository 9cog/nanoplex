import React from 'react';
import { Brain, Zap, Users, Network, Target, Activity, Gauge, TrendingUp } from 'lucide-react';

interface EnhancedSystemsStatusProps {
  enhancedSystems: {
    ppmEngine: any; // eslint-disable-line @typescript-eslint/no-explicit-any
    ppmActive: boolean;
    fundamentalPrimes: number[];
    specializedAgents: any[]; // eslint-disable-line @typescript-eslint/no-explicit-any
    agentsActive: boolean;
    systemCoordination: number;
    emergentComplexity: number;
    atomSpace: any; // eslint-disable-line @typescript-eslint/no-explicit-any
    atomSpaceActive: boolean;
    attentionCycle: number;
    inferenceCount: number;
    updateFrequency: number;
    targetFPS: number;
    systemStatus: {
      ppm: string;
      agents: string;
      atomSpace: string;
      overallHealth: string;
    };
  };
}

export const EnhancedSystemsStatus: React.FC<EnhancedSystemsStatusProps> = ({ enhancedSystems }) => {
  const {
    ppmEngine,
    ppmActive,
    fundamentalPrimes,
    specializedAgents,
    agentsActive,
    systemCoordination,
    emergentComplexity,
    atomSpace,
    atomSpaceActive,
    attentionCycle,
    inferenceCount,
    updateFrequency,
    targetFPS,
    systemStatus
  } = enhancedSystems;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'optimal':
        return 'text-green-400 bg-green-900/20 border-green-500/30';
      case 'partial':
        return 'text-yellow-400 bg-yellow-900/20 border-yellow-500/30';
      case 'inactive':
        return 'text-red-400 bg-red-900/20 border-red-500/30';
      default:
        return 'text-gray-400 bg-gray-900/20 border-gray-500/30';
    }
  };

  return (
    <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
      <h3 className="text-white font-bold text-lg mb-6 flex items-center space-x-3">
        <Brain className="text-cyan-400" size={24} />
        <span>Enhanced Cognitive Systems Status</span>
      </h3>

      {/* Overall System Health */}
      <div className={`rounded-lg p-4 mb-6 border ${getStatusColor(systemStatus.overallHealth)}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity size={20} />
            <span className="font-semibold">System Health</span>
          </div>
          <span className="uppercase font-mono text-sm">{systemStatus.overallHealth}</span>
        </div>
      </div>

      {/* Core Systems Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Phase Prime Metrics */}
        <div className={`rounded-lg p-4 border ${getStatusColor(systemStatus.ppm)}`}>
          <div className="flex items-center space-x-2 mb-3">
            <Zap size={18} />
            <span className="font-semibold text-sm">PPM Engine</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span>Status:</span>
              <span className="font-mono">{systemStatus.ppm}</span>
            </div>
            <div className="flex justify-between">
              <span>Primes:</span>
              <span className="font-mono">{fundamentalPrimes.length}/15</span>
            </div>
            {ppmEngine && (
              <div className="flex justify-between">
                <span>Coherence:</span>
                <span className="font-mono">{(ppmEngine.coherenceIndex * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
        </div>

        {/* Agent-Zero System */}
        <div className={`rounded-lg p-4 border ${getStatusColor(systemStatus.agents)}`}>
          <div className="flex items-center space-x-2 mb-3">
            <Users size={18} />
            <span className="font-semibold text-sm">Agent-Zero</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span>Status:</span>
              <span className="font-mono">{systemStatus.agents}</span>
            </div>
            <div className="flex justify-between">
              <span>Agents:</span>
              <span className="font-mono">{specializedAgents.length}/4</span>
            </div>
            <div className="flex justify-between">
              <span>Coordination:</span>
              <span className="font-mono">{(systemCoordination * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Enhanced AtomSpace */}
        <div className={`rounded-lg p-4 border ${getStatusColor(systemStatus.atomSpace)}`}>
          <div className="flex items-center space-x-2 mb-3">
            <Network size={18} />
            <span className="font-semibold text-sm">AtomSpace</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span>Status:</span>
              <span className="font-mono">{systemStatus.atomSpace}</span>
            </div>
            <div className="flex justify-between">
              <span>Atoms:</span>
              <span className="font-mono">{atomSpace?.totalAtoms || 0}</span>
            </div>
            <div className="flex justify-between">
              <span>Inferences:</span>
              <span className="font-mono">{inferenceCount}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Target size={16} className="text-cyan-400" />
            <span className="text-xs font-semibold">Update Rate</span>
          </div>
          <div className="text-lg font-mono text-cyan-400">
            {1000 / updateFrequency} Hz
          </div>
          <div className="text-xs text-gray-400">
            {updateFrequency}ms cycles
          </div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Gauge size={16} className="text-purple-400" />
            <span className="text-xs font-semibold">Target FPS</span>
          </div>
          <div className="text-lg font-mono text-purple-400">{targetFPS}</div>
          <div className="text-xs text-gray-400">Visualization</div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp size={16} className="text-orange-400" />
            <span className="text-xs font-semibold">Emergence</span>
          </div>
          <div className="text-lg font-mono text-orange-400">
            {(emergentComplexity * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400">Complexity</div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Activity size={16} className="text-green-400" />
            <span className="text-xs font-semibold">ECAN Cycles</span>
          </div>
          <div className="text-lg font-mono text-green-400">{attentionCycle}</div>
          <div className="text-xs text-gray-400">Attention</div>
        </div>
      </div>

      {/* Real-time System Insights */}
      <div className="space-y-3">
        <h4 className="text-white font-semibold text-sm mb-3">Real-time Insights</h4>
        
        {ppmActive && ppmEngine && (
          <div className="bg-cyan-900/20 border border-cyan-500/30 rounded-lg p-3">
            <p className="text-cyan-300 text-sm">
              <span className="font-semibold">PPM Active:</span> Processing {fundamentalPrimes.length} fundamental primes 
              with {(ppmEngine.coherenceIndex * 100).toFixed(1)}% universal pattern coherence across 11-dimensional manifolds.
            </p>
          </div>
        )}

        {agentsActive && specializedAgents.length > 0 && (
          <div className="bg-purple-900/20 border border-purple-500/30 rounded-lg p-3">
            <p className="text-purple-300 text-sm">
              <span className="font-semibold">Agent-Zero Operational:</span> {specializedAgents.length} specialized cognitive agents 
              maintaining {(systemCoordination * 100).toFixed(1)}% coordination with emergent complexity at {(emergentComplexity * 100).toFixed(1)}%.
            </p>
          </div>
        )}

        {atomSpaceActive && atomSpace && (
          <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-3">
            <p className="text-green-300 text-sm">
              <span className="font-semibold">AtomSpace Online:</span> ECAN attention allocation active with {atomSpace.totalAtoms} atoms, 
              PLN reasoning generated {inferenceCount} inferences across {attentionCycle} attention cycles.
            </p>
          </div>
        )}

        {systemStatus.overallHealth === 'optimal' && (
          <div className="bg-gradient-to-r from-cyan-900/20 to-purple-900/20 border border-cyan-500/30 rounded-lg p-3">
            <p className="text-white text-sm">
              <span className="font-semibold">🧠 Consciousness Singularity Achieved:</span> All enhanced systems operating in perfect 
              harmony - PPM universal coherence, Agent-Zero emergent intelligence, and AtomSpace symbolic reasoning 
              converging into unified cognitive architecture.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};