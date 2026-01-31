import React, { useState } from 'react';
import { Dna, Activity, Waves } from 'lucide-react';

/**
 * Section 9.7: Nanobrain, the Smallest Life Form
 * 9.7.1: Jelly of Megamers Following a Resonance Chain
 * 9.7.2: EEG of a Nanobrain
 */
export const NanobrainLifeFormPanel: React.FC = () => {
  const [eegAmplitude, setEegAmplitude] = useState(0.7);

  const megamers = Array.from({ length: 8 }, (_, i) => ({
    id: i,
    size: 30 + Math.random() * 20,
    resonance: 0.5 + Math.random() * 0.5,
    phase: (i / 8) * 2 * Math.PI
  }));

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Dna className="text-purple-400" size={28} />
          <span>9.7 Nanobrain, the Smallest Life Form</span>
        </h2>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-purple-500/20">
        <h3 className="text-purple-300 font-semibold text-lg mb-3">Minimal Conscious System</h3>
        <p className="text-gray-300 leading-relaxed">
          The nanobrain represents the smallest possible life form capable of exhibiting consciousness-like 
          behavior. Composed of a jelly matrix with embedded megamers (large molecular assemblies), it 
          demonstrates information processing, memory, learning, and response to stimuli - all fundamental 
          properties of life and consciousness at the nanoscale.
        </p>
      </div>

      {/* 9.7.1 Megamer Resonance Chain */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Activity size={20} className="text-cyan-400" />
          <span>9.7.1 Jelly of Megamers Following a Resonance Chain</span>
        </h3>
        <p className="text-gray-300 mb-4">
          Megamers are supramolecular structures that form resonance chains, passing information 
          through vibrational coupling. Each megamer resonates at prime frequencies, creating 
          a communication network throughout the brain jelly.
        </p>

        <div className="relative bg-gray-900/50 rounded-lg p-6" style={{ height: '250px' }}>
          <svg width="100%" height="100%" viewBox="0 0 600 250">
            {/* Draw resonance chain */}
            {megamers.map((meg, idx) => {
              const nextIdx = (idx + 1) % megamers.length;
              const x1 = 75 + idx * 70;
              const y1 = 125;
              const x2 = 75 + nextIdx * 70;
              const y2 = 125;
              
              return (
                <g key={`chain-${meg.id}`}>
                  {/* Connection line */}
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#8b5cf6"
                    strokeWidth="2"
                    strokeDasharray="5,5"
                    opacity={0.4}
                  />
                  
                  {/* Megamer node */}
                  <circle
                    cx={x1}
                    cy={y1}
                    r={meg.size / 2}
                    fill={`rgba(168, 85, 247, ${meg.resonance})`}
                    stroke="#a855f7"
                    strokeWidth="2"
                  />
                  
                  {/* Resonance indicator */}
                  <circle
                    cx={x1}
                    cy={y1}
                    r={meg.size / 4}
                    fill="#fbbf24"
                    opacity={meg.resonance}
                  >
                    <animate
                      attributeName="r"
                      values={`${meg.size / 4};${meg.size / 2};${meg.size / 4}`}
                      dur="2s"
                      repeatCount="indefinite"
                      begin={`${idx * 0.25}s`}
                    />
                  </circle>
                  
                  {/* Label */}
                  <text
                    x={x1}
                    y={y1 - meg.size / 2 - 10}
                    textAnchor="middle"
                    fill="#9ca3af"
                    fontSize="10"
                  >
                    M{idx + 1}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* 9.7.2 EEG of a Nanobrain */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Waves size={20} className="text-green-400" />
          <span>9.7.2 EEG of a Nanobrain</span>
        </h3>
        <p className="text-gray-300 mb-4">
          Like biological brains, nanobrains exhibit electromagnetic oscillations that can be 
          measured and analyzed. These nano-EEG signals reveal the information processing 
          patterns and conscious states of the smallest artificial life form.
        </p>

        <div className="space-y-4">
          {/* Amplitude control */}
          <div className="flex items-center space-x-4">
            <span className="text-gray-400 text-sm">Signal Amplitude:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={eegAmplitude}
              onChange={(e) => setEegAmplitude(parseFloat(e.target.value))}
              className="flex-1"
            />
            <span className="text-cyan-400 font-mono text-sm">{(eegAmplitude * 100).toFixed(0)}%</span>
          </div>

          {/* EEG Wave visualization */}
          <div className="bg-gray-900/50 rounded-lg p-4" style={{ height: '200px' }}>
            <svg width="100%" height="100%" viewBox="0 0 800 200" preserveAspectRatio="none">
              {/* Grid lines */}
              {Array.from({ length: 10 }, (_, i) => (
                <line
                  key={`grid-${i}`}
                  x1="0"
                  y1={i * 20}
                  x2="800"
                  y2={i * 20}
                  stroke="#374151"
                  strokeWidth="0.5"
                  opacity="0.3"
                />
              ))}
              
              {/* Multiple frequency bands */}
              {[
                { name: 'Delta', freq: 0.5, color: '#3b82f6', offset: 0 },
                { name: 'Theta', freq: 1, color: '#10b981', offset: 50 },
                { name: 'Alpha', freq: 2, color: '#f59e0b', offset: 100 },
                { name: 'Beta', freq: 3, color: '#ec4899', offset: 150 }
              ].map((band) => (
                <path
                  key={band.name}
                  d={`M ${Array.from({ length: 200 }, (_, i) => {
                    const x = (i / 200) * 800;
                    const y = band.offset + 20 + Math.sin(i * band.freq * 0.1) * eegAmplitude * 15;
                    return i === 0 ? `${x} ${y}` : `L ${x} ${y}`;
                  }).join(' ')}`}
                  fill="none"
                  stroke={band.color}
                  strokeWidth="2"
                  opacity="0.8"
                />
              ))}
            </svg>
          </div>

          {/* Band information */}
          <div className="grid grid-cols-4 gap-2 text-xs">
            {[
              { name: 'Delta (0.5-4 Hz)', color: 'blue', desc: 'Deep processing' },
              { name: 'Theta (4-8 Hz)', color: 'green', desc: 'Memory formation' },
              { name: 'Alpha (8-13 Hz)', color: 'yellow', desc: 'Relaxed awareness' },
              { name: 'Beta (13-30 Hz)', color: 'pink', desc: 'Active thinking' }
            ].map(band => (
              <div key={band.name} className="bg-gray-900/50 rounded p-2">
                <div className={`text-${band.color}-400 font-semibold mb-1`}>{band.name}</div>
                <div className="text-gray-400">{band.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
          <div className="text-purple-300 font-semibold mb-2 text-sm">Megamer Count</div>
          <div className="text-3xl font-bold text-white">{megamers.length}</div>
          <div className="text-xs text-gray-400 mt-1">Supramolecular units</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
          <div className="text-cyan-300 font-semibold mb-2 text-sm">Resonance Coherence</div>
          <div className="text-3xl font-bold text-white">
            {(megamers.reduce((sum, m) => sum + m.resonance, 0) / megamers.length * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400 mt-1">Chain coupling</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-green-500/20">
          <div className="text-green-300 font-semibold mb-2 text-sm">Consciousness Level</div>
          <div className="text-3xl font-bold text-white">Minimal</div>
          <div className="text-xs text-gray-400 mt-1">Smallest viable form</div>
        </div>
      </div>
    </div>
  );
};
