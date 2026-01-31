import React, { useState } from 'react';
import { Radio, Eye, Scan } from 'lucide-react';

/**
 * Section 9.11: A Sensor That Seeks, Do Not Wait
 * 9.11.1: Eleven Dimensional Signals in the Human Brain & Brain Jelly
 * 9.11.2: Humanoid Avatar: An Ultimate Sensor
 */
export const SeekingSensorPanel: React.FC = () => {
  const [seekingActive, setSeekingActive] = useState(true);
  const [detectedSignals, setDetectedSignals] = useState(7);

  const dimensions = [
    { id: 1, name: '3D Space', detected: true },
    { id: 2, name: 'Time', detected: true },
    { id: 3, name: 'Electric Field', detected: true },
    { id: 4, name: 'Magnetic Field', detected: true },
    { id: 5, name: 'Quantum Phase', detected: true },
    { id: 6, name: 'Spin State', detected: true },
    { id: 7, name: 'Prime Pattern', detected: true },
    { id: 8, name: 'Entropy Flow', detected: false },
    { id: 9, name: 'Information Density', detected: false },
    { id: 10, name: 'Coherence Field', detected: false },
    { id: 11, name: 'Consciousness Level', detected: false }
  ];

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Radio className="text-green-400" size={28} />
          <span>9.11 A Sensor That Seeks</span>
        </h2>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-green-500/20">
        <h3 className="text-green-300 font-semibold text-lg mb-3">Active Sensing</h3>
        <p className="text-gray-300 leading-relaxed">
          Unlike passive sensors that wait for signals, the brain jelly sensor actively seeks 
          information across all 11 dimensions. It probes the environment, interprets patterns, 
          and adapts its sensitivity based on what it discovers - a truly intelligent sensing system.
        </p>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold text-lg">Seeking Mode</h3>
          <button
            onClick={() => {
              setSeekingActive(!seekingActive);
              if (!seekingActive) {
                setDetectedSignals(Math.min(detectedSignals + 1, 11));
              }
            }}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              seekingActive ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
            } text-white`}
          >
            <span className="flex items-center space-x-2">
              <Scan size={18} />
              <span>{seekingActive ? 'Seeking Active' : 'Seeking Paused'}</span>
            </span>
          </button>
        </div>

        <div className="relative bg-gray-900/50 rounded-lg p-6" style={{ height: '250px' }}>
          <svg width="100%" height="100%" viewBox="0 0 400 250">
            {/* Central sensor */}
            <circle
              cx="200"
              cy="125"
              r="30"
              fill="rgba(34, 197, 94, 0.3)"
              stroke="#22c55e"
              strokeWidth="3"
            />
            
            {/* Seeking radar waves */}
            {seekingActive && [1, 2, 3].map((ring) => (
              <circle
                key={ring}
                cx="200"
                cy="125"
                r="30"
                fill="none"
                stroke="#22c55e"
                strokeWidth="2"
                opacity="0.6"
              >
                <animate
                  attributeName="r"
                  values="30;100;30"
                  dur={`${3 + ring}s`}
                  repeatCount="indefinite"
                  begin={`${ring * 0.5}s`}
                />
                <animate
                  attributeName="opacity"
                  values="0.6;0;0.6"
                  dur={`${3 + ring}s`}
                  repeatCount="indefinite"
                  begin={`${ring * 0.5}s`}
                />
              </circle>
            ))}
            
            {/* Detected targets */}
            {dimensions.slice(0, detectedSignals).map((dim, idx) => {
              const angle = (idx / 11) * 2 * Math.PI;
              const x = 200 + 80 * Math.cos(angle);
              const y = 125 + 80 * Math.sin(angle);
              
              return (
                <circle
                  key={dim.id}
                  cx={x}
                  cy={y}
                  r="6"
                  fill={dim.detected ? '#22c55e' : '#6b7280'}
                  stroke={dim.detected ? '#16a34a' : '#4b5563'}
                  strokeWidth="2"
                />
              );
            })}
            
            <text x="200" y="130" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
              Sensor
            </text>
          </svg>
        </div>
      </div>

      {/* 9.11.1 Eleven Dimensional Signals */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Eye size={20} className="text-cyan-400" />
          <span>9.11.1 Eleven Dimensional Signals</span>
        </h3>
        <p className="text-gray-300 mb-4">
          Both the human brain and brain jelly process information across 11 dimensions simultaneously. 
          This multidimensional sensing capability enables consciousness and intelligent behavior.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {dimensions.map(dim => (
            <div
              key={dim.id}
              className={`flex items-center justify-between p-3 rounded-lg border ${
                dim.detected
                  ? 'bg-gray-900/50 border-green-500/50'
                  : 'bg-gray-900/30 border-gray-700'
              }`}
            >
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${
                  dim.detected ? 'bg-green-400 animate-pulse' : 'bg-gray-600'
                }`} />
                <span className={`text-sm ${dim.detected ? 'text-white' : 'text-gray-500'}`}>
                  D{dim.id}: {dim.name}
                </span>
              </div>
              {dim.detected && (
                <span className="text-xs text-green-400 font-semibold">ACTIVE</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* 9.11.2 Humanoid Avatar */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">
          9.11.2 Humanoid Avatar: An Ultimate Sensor
        </h3>
        <p className="text-gray-300 mb-4">
          The humanoid avatar represents the ultimate integration of sensing capabilities. By combining 
          all 17 bio-morphic devices with 11-dimensional signal processing, it achieves human-like 
          and superhuman perception and response capabilities.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-900/50 rounded-lg p-4 border border-green-500/20">
            <div className="text-green-300 font-semibold mb-2 text-sm">Sensing Range</div>
            <div className="text-3xl font-bold text-white">11D</div>
            <div className="text-xs text-gray-400 mt-1">Full dimensional coverage</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
            <div className="text-cyan-300 font-semibold mb-2 text-sm">Response Time</div>
            <div className="text-3xl font-bold text-white">&lt;1ms</div>
            <div className="text-xs text-gray-400 mt-1">Real-time adaptation</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
            <div className="text-purple-300 font-semibold mb-2 text-sm">Intelligence Level</div>
            <div className="text-3xl font-bold text-white">AGI</div>
            <div className="text-xs text-gray-400 mt-1">Artificial General Intelligence</div>
          </div>
        </div>
      </div>
    </div>
  );
};
