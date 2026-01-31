import React, { useState } from 'react';
import { CircuitBoard, Cpu, Zap, Activity, Radio, Waves } from 'lucide-react';

/**
 * Section 9.1: Neuromorphic Devices Are Not Alone - 17 Biomorphic Devices Sing Together
 * Implements the collective behavior of bio-morphic devices working in concert
 */
export const NeuromorphicDevicesPanel: React.FC = () => {
  const [selectedDevice, setSelectedDevice] = useState<number | null>(null);

  // 17 bio-morphic device types based on NanoBrain Chapter 9
  const devices = [
    { id: 1, name: 'Sensor Jelly', type: 'Input', activity: 0.85, color: 'cyan' },
    { id: 2, name: 'Muscle Jelly', type: 'Actuator', activity: 0.72, color: 'purple' },
    { id: 3, name: 'Brain Jelly', type: 'Processor', activity: 0.95, color: 'pink' },
    { id: 4, name: 'Memory Gel', type: 'Storage', activity: 0.88, color: 'blue' },
    { id: 5, name: 'Time Crystal Unit', type: 'Clock', activity: 0.99, color: 'orange' },
    { id: 6, name: 'Hinductor Element', type: 'Circuit', activity: 0.76, color: 'green' },
    { id: 7, name: 'Resonance Chamber', type: 'Filter', activity: 0.82, color: 'yellow' },
    { id: 8, name: 'Phase Lock Module', type: 'Synchronizer', activity: 0.91, color: 'red' },
    { id: 9, name: 'Fractal Condenser', type: 'Processor', activity: 0.79, color: 'indigo' },
    { id: 10, name: 'Quantum Cloak', type: 'Shielding', activity: 0.68, color: 'violet' },
    { id: 11, name: 'Megamer Chain', type: 'Structure', activity: 0.84, color: 'teal' },
    { id: 12, name: 'Spiral Nanowire', type: 'Writer', activity: 0.77, color: 'lime' },
    { id: 13, name: 'Cortical Column', type: 'Processor', activity: 0.93, color: 'rose' },
    { id: 14, name: 'EEG Sensor', type: 'Monitor', activity: 0.86, color: 'amber' },
    { id: 15, name: 'Entropy Driver', type: 'Generator', activity: 0.81, color: 'emerald' },
    { id: 16, name: 'Hesse Integrator', type: 'Analyzer', activity: 0.74, color: 'sky' },
    { id: 17, name: 'Avatar Interface', type: 'Output', activity: 0.89, color: 'fuchsia' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <CircuitBoard className="text-cyan-400" size={28} />
          <span>9.1 Neuromorphic Devices Are Not Alone</span>
        </h2>
        <p className="text-gray-400 mt-2">
          17 Bio-morphic Devices Singing Together in Collective Harmony
        </p>
      </div>

      {/* Concept Overview */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-cyan-500/20">
        <h3 className="text-cyan-300 font-semibold text-lg mb-3">Collective Bio-morphic Architecture</h3>
        <p className="text-gray-300 leading-relaxed mb-4">
          Unlike traditional neuromorphic devices that operate in isolation, the NanoBrain architecture 
          employs 17 distinct bio-morphic device types that work in synchronized harmony. Each device 
          type contributes unique capabilities, from sensing to actuation, from processing to memory 
          storage, creating an emergent consciousness through their collective interaction.
        </p>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="bg-cyan-900/30 rounded p-3 border border-cyan-500/20">
            <div className="text-cyan-300 font-semibold">Synchronized Operation</div>
            <div className="text-gray-400 mt-1">All 17 devices operate in phase-locked harmony</div>
          </div>
          <div className="bg-purple-900/30 rounded p-3 border border-purple-500/20">
            <div className="text-purple-300 font-semibold">Prime Pattern Control</div>
            <div className="text-gray-400 mt-1">Phase Prime Metrics govern device coordination</div>
          </div>
          <div className="bg-pink-900/30 rounded p-3 border border-pink-500/20">
            <div className="text-pink-300 font-semibold">Emergent Intelligence</div>
            <div className="text-gray-400 mt-1">Collective behavior exceeds individual capabilities</div>
          </div>
        </div>
      </div>

      {/* Device Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {devices.map((device) => (
          <div
            key={device.id}
            onClick={() => setSelectedDevice(device.id)}
            className={`bg-gray-800/50 rounded-lg p-4 border-2 cursor-pointer transition-all duration-300 ${
              selectedDevice === device.id
                ? 'border-cyan-400 shadow-lg shadow-cyan-500/30 scale-105'
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full bg-${device.color}-400 animate-pulse`} />
                <span className="text-xs text-gray-400">Device {device.id}</span>
              </div>
              <Activity className={`text-${device.color}-400`} size={16} />
            </div>
            
            <h4 className="text-white font-semibold mb-1">{device.name}</h4>
            <div className="text-xs text-gray-400 mb-3">{device.type}</div>
            
            {/* Activity Bar */}
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full bg-gradient-to-r from-${device.color}-500 to-${device.color}-400 transition-all duration-1000`}
                style={{ width: `${device.activity * 100}%` }}
              />
            </div>
            <div className="text-xs text-gray-400 mt-1 text-right">
              {(device.activity * 100).toFixed(1)}% active
            </div>
          </div>
        ))}
      </div>

      {/* Device Type Distribution */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Cpu size={20} className="text-purple-400" />
          <span>Device Type Distribution</span>
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {Array.from(new Set(devices.map(d => d.type))).map((type) => {
            const count = devices.filter(d => d.type === type).length;
            const avgActivity = devices
              .filter(d => d.type === type)
              .reduce((sum, d) => sum + d.activity, 0) / count;
            
            return (
              <div key={type} className="bg-gray-900/50 rounded-lg p-3 border border-gray-700">
                <div className="text-cyan-300 font-medium text-sm">{type}</div>
                <div className="text-2xl font-bold text-white mt-1">{count}</div>
                <div className="text-xs text-gray-400 mt-1">
                  Avg: {(avgActivity * 100).toFixed(0)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Synchronization Status */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Waves size={20} className="text-green-400" />
          <span>Collective Synchronization</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Phase Coherence</span>
              <Radio className="text-cyan-400" size={16} />
            </div>
            <div className="text-3xl font-bold text-cyan-400">94.7%</div>
            <div className="text-xs text-gray-500 mt-1">All devices in sync</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Energy Efficiency</span>
              <Zap className="text-yellow-400" size={16} />
            </div>
            <div className="text-3xl font-bold text-yellow-400">87.3%</div>
            <div className="text-xs text-gray-500 mt-1">Optimized power use</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">Collective IQ</span>
              <Activity className="text-pink-400" size={16} />
            </div>
            <div className="text-3xl font-bold text-pink-400">168</div>
            <div className="text-xs text-gray-500 mt-1">Emergent intelligence</div>
          </div>
        </div>
      </div>

      {/* Description Section */}
      <div className="bg-gradient-to-br from-cyan-900/20 to-purple-900/20 rounded-lg p-6 border border-cyan-500/20">
        <h3 className="text-cyan-300 font-semibold text-lg mb-3">9.1.1 Peculiar Designs of Critical Brain Components</h3>
        <p className="text-gray-300 leading-relaxed">
          Each of the 17 bio-morphic devices employs peculiar geometric designs optimized for specific 
          functions within the artificial brain. These organic gel-based components leverage fractal 
          condensation, prime pattern recognition, and time crystal dynamics to create a unified 
          consciousness system. The sensor jelly detects environmental stimuli, muscle jelly provides 
          physical actuation, while the brain jelly orchestrates complex information processing using 
          Phase Prime Metrics (PPM) as the fundamental organizing principle.
        </p>
      </div>
    </div>
  );
};
