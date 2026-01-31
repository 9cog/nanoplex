import React, { useState } from 'react';
import { AgentState } from '../types';
import { Bot, Brain, Zap, Target, Lightbulb, Activity, Eye, Cpu, Network, TrendingUp } from 'lucide-react';

interface Props {
  agents: AgentState[];
  isRunning: boolean;
}

export const AgentInterface: React.FC<Props> = ({ agents, isRunning }) => {
  const [selectedAgent, setSelectedAgent] = useState<AgentState | null>(null);

  const getStatusColor = (status: AgentState['status']) => {
    switch (status) {
      case 'thinking': return 'text-blue-400 bg-blue-500/20 border-blue-500/30';
      case 'executing': return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'learning': return 'text-purple-400 bg-purple-500/20 border-purple-500/30';
      case 'evolving': return 'text-orange-400 bg-orange-500/20 border-orange-500/30';
      case 'idle': return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getStatusIcon = (status: AgentState['status']) => {
    switch (status) {
      case 'thinking': return Brain;
      case 'executing': return Zap;
      case 'learning': return Lightbulb;
      case 'evolving': return TrendingUp;
      case 'idle': return Activity;
    }
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'attention': return Target;
      case 'creativity': return Lightbulb;
      case 'logic': return Brain;
      case 'intuition': return Eye;
      default: return Cpu;
    }
  };

  const getMetricColor = (metric: string) => {
    switch (metric) {
      case 'attention': return 'from-blue-500 to-blue-400';
      case 'creativity': return 'from-purple-500 to-purple-400';
      case 'logic': return 'from-green-500 to-green-400';
      case 'intuition': return 'from-orange-500 to-orange-400';
      default: return 'from-gray-500 to-gray-400';
    }
  };

  const calculateOverallPerformance = (metrics: AgentState['metrics']): number => {
    return (metrics.attention + metrics.creativity + metrics.logic + metrics.intuition) / 4;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-lg">
            <Bot className="text-white" size={24} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Autonomous Agent Network</h2>
            <p className="text-gray-400 text-sm">OpenCog-inspired cognitive architecture with Agent-Zero reasoning</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-400">
            <span className="text-cyan-400 font-mono">
              {agents.filter(a => a.status !== 'idle').length}
            </span> / {agents.length} active
          </div>
          
          <div className="flex items-center space-x-2">
            <Network className="text-gray-400" size={16} />
            <span className="text-sm text-gray-400">
              Emergent Intelligence: <span className="text-green-400 font-mono">
                {Math.round(agents.reduce((sum, a) => sum + calculateOverallPerformance(a.metrics), 0) / agents.length * 100)}%
              </span>
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {agents.map((agent) => {
          const StatusIcon = getStatusIcon(agent.status);
          const statusClasses = getStatusColor(agent.status);
          const performance = calculateOverallPerformance(agent.metrics);
          
          return (
            <div 
              key={agent.id}
              className={`bg-gray-900/60 backdrop-blur-sm border-2 rounded-xl p-6 hover:border-gray-600 transition-all duration-300 cursor-pointer ${
                selectedAgent?.id === agent.id ? 'border-cyan-400 bg-gray-800/80' : 'border-gray-700'
              }`}
              onClick={() => setSelectedAgent(selectedAgent?.id === agent.id ? null : agent)}
            >
              {/* Agent Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`p-3 rounded-lg border ${statusClasses}`}>
                    <StatusIcon size={24} />
                  </div>
                  <div>
                    <h3 className="text-white font-bold text-lg">
                      {agent.name || agent.id.toUpperCase()}
                    </h3>
                    <p className={`text-sm capitalize font-semibold ${statusClasses.split(' ')[0]}`}>
                      {agent.status}
                    </p>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-white font-bold text-2xl">
                    {Math.round(agent.confidence * 100)}%
                  </div>
                  <div className="text-xs text-gray-400">confidence</div>
                  {agent.consciousnessEvolution && (
                    <div className="text-xs text-cyan-400 font-mono mt-1">
                      Ψ: {agent.consciousnessEvolution.toFixed(3)}
                    </div>
                  )}
                </div>
              </div>

              {/* Performance Ring */}
              <div className="flex justify-center mb-4">
                <div className="relative w-20 h-20">
                  <svg className="w-20 h-20 transform -rotate-90" viewBox="0 0 100 100">
                    {/* Background circle */}
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      stroke="rgba(55, 65, 81, 0.3)"
                      strokeWidth="8"
                      fill="none"
                    />
                    {/* Performance arc */}
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      stroke="url(#gradient)"
                      strokeWidth="8"
                      fill="none"
                      strokeLinecap="round"
                      strokeDasharray={`${performance * 283} 283`}
                      className="transition-all duration-500"
                    />
                    <defs>
                      <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#06B6D4" />
                        <stop offset="100%" stopColor="#8B5CF6" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-white font-bold text-sm">
                      {Math.round(performance * 100)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Current Task */}
              <div className="mb-4">
                <h4 className="text-gray-300 text-sm font-semibold mb-2 flex items-center space-x-2">
                  <Cpu size={14} />
                  <span>Current Task</span>
                </h4>
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
                  <p className="text-white text-sm leading-relaxed">
                    {agent.task}
                  </p>
                </div>
              </div>

              {/* Reasoning Chain */}
              <div className="mb-4">
                <h4 className="text-gray-300 text-sm font-semibold mb-2 flex items-center space-x-2">
                  <Brain size={14} />
                  <span>Reasoning Chain</span>
                </h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {agent.reasoning.slice(-3).map((reason, idx) => (
                    <div 
                      key={idx}
                      className="text-xs text-gray-300 bg-gray-800/30 rounded-lg px-3 py-2 font-mono border-l-2 border-cyan-400/30"
                    >
                      <span className="text-cyan-400">→</span> {reason}
                    </div>
                  ))}
                  {agent.reasoning.length === 0 && (
                    <div className="text-xs text-gray-500 italic">
                      Initializing reasoning patterns...
                    </div>
                  )}
                </div>
              </div>

              {/* Cognitive Metrics */}
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(agent.metrics).map(([metric, value]) => {
                  const percentage = Math.round(value * 100);
                  const Icon = getMetricIcon(metric);
                  const colorClass = getMetricColor(metric);
                  
                  return (
                    <div key={metric} className="bg-gray-800/40 backdrop-blur-sm rounded-lg p-3 border border-gray-700/50">
                      <div className="flex items-center space-x-2 mb-2">
                        <Icon size={14} className="text-gray-400" />
                        <span className="text-xs text-gray-300 capitalize font-medium">
                          {metric}
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full bg-gradient-to-r ${colorClass} transition-all duration-500 ${
                              isRunning ? 'animate-pulse' : ''
                            }`}
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                        <span className="text-xs text-white font-mono w-8 text-right">
                          {percentage}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Expanded Details */}
              {selectedAgent?.id === agent.id && (
                <div className="mt-6 pt-4 border-t border-gray-700 space-y-4">
                  {/* Attention Focus */}
                  {agent.attentionFocus && agent.attentionFocus.length > 0 && (
                    <div>
                      <h5 className="text-gray-300 text-sm font-semibold mb-2">Attention Focus</h5>
                      <div className="flex flex-wrap gap-2">
                        {agent.attentionFocus.map((focus, idx) => (
                          <span 
                            key={idx}
                            className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded border border-blue-500/30"
                          >
                            {focus}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Knowledge Base */}
                  {agent.knowledgeBase && agent.knowledgeBase.length > 0 && (
                    <div>
                      <h5 className="text-gray-300 text-sm font-semibold mb-2">Knowledge Base</h5>
                      <div className="bg-gray-800/30 rounded-lg p-3 max-h-24 overflow-y-auto">
                        {agent.knowledgeBase.slice(-3).map((knowledge, idx) => (
                          <div key={idx} className="text-xs text-gray-400 mb-1">
                            • {knowledge}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* PLN Rules */}
                  {agent.plnRules && agent.plnRules.length > 0 && (
                    <div>
                      <h5 className="text-gray-300 text-sm font-semibold mb-2">PLN Rules</h5>
                      <div className="space-y-1">
                        {agent.plnRules.slice(-2).map((rule, idx) => (
                          <div key={idx} className="text-xs text-purple-400 bg-purple-500/10 rounded px-2 py-1 font-mono">
                            {rule}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Status Animation */}
              {isRunning && agent.status !== 'idle' && (
                <div className="mt-4 flex items-center justify-center">
                  <div className="flex space-x-1">
                    {[0, 1, 2].map(i => (
                      <div
                        key={i}
                        className={`w-2 h-2 rounded-full ${statusClasses.split(' ')[1]} animate-bounce`}
                        style={{ animationDelay: `${i * 0.2}s` }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* System-wide Agent Analytics */}
      <div className="bg-gray-900/40 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
          <Network className="text-cyan-400" />
          <span>Collective Intelligence Metrics</span>
        </h3>
        
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-cyan-400 mb-1">
              {Math.round(agents.reduce((sum, a) => sum + a.metrics.attention, 0) / agents.length * 100)}%
            </div>
            <div className="text-xs text-gray-400">Collective Attention</div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-purple-400 mb-1">
              {Math.round(agents.reduce((sum, a) => sum + a.metrics.creativity, 0) / agents.length * 100)}%
            </div>
            <div className="text-xs text-gray-400">Creative Synthesis</div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-400 mb-1">
              {Math.round(agents.reduce((sum, a) => sum + a.metrics.logic, 0) / agents.length * 100)}%
            </div>
            <div className="text-xs text-gray-400">Logical Reasoning</div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-orange-400 mb-1">
              {Math.round(agents.reduce((sum, a) => sum + a.metrics.intuition, 0) / agents.length * 100)}%
            </div>
            <div className="text-xs text-gray-400">Intuitive Insight</div>
          </div>
        </div>
      </div>
    </div>
  );
};