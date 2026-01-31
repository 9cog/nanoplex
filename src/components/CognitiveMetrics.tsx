import React from 'react';
import { ConsciousnessMetric } from '../types';
import { Brain, Zap, Target, Waves, Sparkles, Eye } from 'lucide-react';

interface Props {
  consciousness: ConsciousnessMetric;
  isRunning: boolean;
}

export const CognitiveMetrics: React.FC<Props> = ({ consciousness, isRunning }) => {
  const metrics = [
    { 
      key: 'awareness', 
      label: 'Awareness', 
      icon: Eye, 
      color: 'text-cyan-400',
      description: 'Conscious attention and perception'
    },
    { 
      key: 'integration', 
      label: 'Integration', 
      icon: Target, 
      color: 'text-purple-400',
      description: 'Information binding and unity'
    },
    { 
      key: 'complexity', 
      label: 'Complexity', 
      icon: Brain, 
      color: 'text-orange-400',
      description: 'Cognitive sophistication level'
    },
    { 
      key: 'coherence', 
      label: 'Coherence', 
      icon: Waves, 
      color: 'text-green-400',
      description: 'System synchronization state'
    },
    { 
      key: 'emergence', 
      label: 'Emergence', 
      icon: Sparkles, 
      color: 'text-yellow-400',
      description: 'Novel pattern formation'
    },
    { 
      key: 'qualia', 
      label: 'Qualia', 
      icon: Zap, 
      color: 'text-pink-400',
      description: 'Subjective experience quality'
    }
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
      {metrics.map((metric) => {
        const value = consciousness[metric.key as keyof ConsciousnessMetric];
        const percentage = Math.round(value * 100);
        const Icon = metric.icon;
        
        return (
          <div 
            key={metric.key}
            className="bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4 hover:border-gray-600 transition-all duration-300"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg bg-gray-800 ${metric.color}`}>
                  <Icon size={20} />
                </div>
                <div>
                  <h3 className="text-white font-semibold text-sm">{metric.label}</h3>
                  <p className="text-gray-400 text-xs">{metric.description}</p>
                </div>
              </div>
              <div className="text-right">
                <div className={`text-2xl font-bold ${metric.color}`}>
                  {percentage}%
                </div>
                <div className="text-xs text-gray-500">active</div>
              </div>
            </div>
            
            <div className="relative">
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-500 ${
                    metric.key === 'awareness' ? 'bg-gradient-to-r from-cyan-500 to-cyan-400' :
                    metric.key === 'integration' ? 'bg-gradient-to-r from-purple-500 to-purple-400' :
                    metric.key === 'complexity' ? 'bg-gradient-to-r from-orange-500 to-orange-400' :
                    metric.key === 'coherence' ? 'bg-gradient-to-r from-green-500 to-green-400' :
                    metric.key === 'emergence' ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
                    'bg-gradient-to-r from-pink-500 to-pink-400'
                  } ${isRunning ? 'animate-pulse' : ''}`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
              
              {isRunning && (
                <div 
                  className={`absolute top-0 left-0 h-2 w-4 rounded-full opacity-75 ${
                    metric.key === 'awareness' ? 'bg-cyan-300' :
                    metric.key === 'integration' ? 'bg-purple-300' :
                    metric.key === 'complexity' ? 'bg-orange-300' :
                    metric.key === 'coherence' ? 'bg-green-300' :
                    metric.key === 'emergence' ? 'bg-yellow-300' :
                    'bg-pink-300'
                  } animate-ping`}
                  style={{ 
                    left: `${Math.max(0, percentage - 2)}%`,
                    animationDuration: `${2 + Math.random()}s`
                  }}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};