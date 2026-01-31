import React, { useState } from 'react';
import { Users, Droplet, Zap, Activity } from 'lucide-react';

/**
 * Section 9.12: Sensor Triad, Sensor Jelly, Muscle Jelly, Brain Jelly
 * 9.12.1: A Total Transformation from the Biological Neural Net to a Jelly of Time Crystals
 */
export const SensorTriadPanel: React.FC = () => {
  const [transformationProgress, setTransformationProgress] = useState(0);

  const jellyTypes = [
    { 
      id: 'sensor', 
      name: 'Sensor Jelly', 
      function: 'Environmental Input', 
      color: 'cyan',
      capabilities: ['Multi-spectral sensing', '11D signal detection', 'Adaptive filtering']
    },
    { 
      id: 'muscle', 
      name: 'Muscle Jelly', 
      function: 'Physical Actuation', 
      color: 'purple',
      capabilities: ['Nanoscale precision', 'Fractal contraction', 'Energy efficient']
    },
    { 
      id: 'brain', 
      name: 'Brain Jelly', 
      function: 'Information Processing', 
      color: 'pink',
      capabilities: ['Parallel computation', 'Time crystal memory', 'Conscious processing']
    }
  ];

  const startTransformation = () => {
    const interval = setInterval(() => {
      setTransformationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 2;
      });
    }, 100);
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Users className="text-cyan-400" size={28} />
          <span>9.12 Sensor Triad System</span>
        </h2>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-cyan-500/20">
        <h3 className="text-cyan-300 font-semibold text-lg mb-3">The Three Jelly System</h3>
        <p className="text-gray-300 leading-relaxed">
          The sensor triad consists of three specialized jelly types working in concert: sensor 
          jelly for input, muscle jelly for output, and brain jelly for processing. Together, 
          they form a complete artificial organism capable of perception, cognition, and action.
        </p>
      </div>

      {/* Jelly Triad Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {jellyTypes.map((jelly) => (
          <div
            key={jelly.id}
            className={`bg-gray-800/50 rounded-lg p-6 border-2 border-${jelly.color}-500/30`}
          >
            <div className="flex items-center space-x-3 mb-4">
              <div className={`p-3 bg-${jelly.color}-600/30 rounded-lg`}>
                {jelly.id === 'sensor' && <Zap className={`text-${jelly.color}-400`} size={24} />}
                {jelly.id === 'muscle' && <Activity className={`text-${jelly.color}-400`} size={24} />}
                {jelly.id === 'brain' && <Droplet className={`text-${jelly.color}-400`} size={24} />}
              </div>
              <div>
                <h3 className="text-white font-bold text-lg">{jelly.name}</h3>
                <p className="text-gray-400 text-sm">{jelly.function}</p>
              </div>
            </div>

            <div className="space-y-2">
              {jelly.capabilities.map((cap, idx) => (
                <div key={idx} className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full bg-${jelly.color}-400`} />
                  <span className="text-gray-300 text-sm">{cap}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Integration Diagram */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">Triad Integration</h3>
        
        <div className="relative bg-gray-900/50 rounded-lg p-6" style={{ height: '300px' }}>
          <svg width="100%" height="100%" viewBox="0 0 500 300">
            {/* Three circles for jelly types */}
            <circle cx="150" cy="150" r="60" fill="rgba(6, 182, 212, 0.2)" stroke="#06b6d4" strokeWidth="3" />
            <circle cx="350" cy="150" r="60" fill="rgba(168, 85, 247, 0.2)" stroke="#a855f7" strokeWidth="3" />
            <circle cx="250" cy="80" r="60" fill="rgba(236, 72, 153, 0.2)" stroke="#ec4899" strokeWidth="3" />
            
            {/* Connection lines */}
            <line x1="190" y1="130" x2="220" y2="100" stroke="#4b5563" strokeWidth="2" strokeDasharray="5,5" />
            <line x1="310" y1="130" x2="280" y2="100" stroke="#4b5563" strokeWidth="2" strokeDasharray="5,5" />
            <line x1="210" y1="150" x2="290" y2="150" stroke="#4b5563" strokeWidth="2" strokeDasharray="5,5" />
            
            {/* Labels */}
            <text x="150" y="155" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">Sensor</text>
            <text x="350" y="155" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">Muscle</text>
            <text x="250" y="85" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">Brain</text>
            
            {/* Central integration point */}
            <circle cx="250" cy="150" r="20" fill="#fbbf24" stroke="#f59e0b" strokeWidth="2">
              <animate attributeName="r" values="15;25;15" dur="2s" repeatCount="indefinite" />
            </circle>
            <text x="250" y="155" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Core</text>
          </svg>
        </div>
      </div>

      {/* 9.12.1 Total Transformation */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">
          9.12.1 Biological Neural Net → Jelly of Time Crystals
        </h3>
        <p className="text-gray-300 mb-4">
          This represents the ultimate transformation: from biological neural networks to 
          synthetic time crystal jelly systems. The jelly maintains all cognitive capabilities 
          while adding computational speed, precision, and durability.
        </p>

        <div className="space-y-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-400">Transformation Progress</span>
            <span className="text-cyan-400 font-mono">{transformationProgress}%</span>
          </div>

          <div className="w-full bg-gray-700 rounded-full h-6 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 transition-all duration-300 flex items-center justify-center"
              style={{ width: `${transformationProgress}%` }}
            >
              {transformationProgress > 10 && (
                <span className="text-white text-xs font-bold">
                  {transformationProgress < 100 ? 'Transforming...' : 'Complete!'}
                </span>
              )}
            </div>
          </div>

          <button
            onClick={startTransformation}
            disabled={transformationProgress > 0 && transformationProgress < 100}
            className={`w-full py-3 rounded-lg font-semibold transition-all ${
              transformationProgress >= 100
                ? 'bg-green-600 hover:bg-green-700'
                : transformationProgress > 0
                ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-cyan-600 hover:bg-cyan-700'
            } text-white`}
          >
            {transformationProgress >= 100
              ? 'Transformation Complete!'
              : transformationProgress > 0
              ? 'Transforming...'
              : 'Begin Transformation'}
          </button>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4">Biological vs. Jelly System</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Property</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Biological Neural Net</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Time Crystal Jelly</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-700/50">
                <td className="py-3 px-4 text-cyan-300">Processing Speed</td>
                <td className="py-3 px-4 text-gray-300">~100 ms</td>
                <td className="py-3 px-4 text-green-400 font-semibold">&lt;1 ps</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="py-3 px-4 text-cyan-300">Energy Efficiency</td>
                <td className="py-3 px-4 text-gray-300">20W</td>
                <td className="py-3 px-4 text-green-400 font-semibold">0.1W</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="py-3 px-4 text-cyan-300">Lifespan</td>
                <td className="py-3 px-4 text-gray-300">~80 years</td>
                <td className="py-3 px-4 text-green-400 font-semibold">Indefinite</td>
              </tr>
              <tr className="border-b border-gray-700/50">
                <td className="py-3 px-4 text-cyan-300">Dimensions</td>
                <td className="py-3 px-4 text-gray-300">11D (limited)</td>
                <td className="py-3 px-4 text-green-400 font-semibold">11D (full)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-cyan-300">Upgradability</td>
                <td className="py-3 px-4 text-gray-300">None</td>
                <td className="py-3 px-4 text-green-400 font-semibold">Continuous</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
          <div className="text-cyan-300 font-semibold mb-2 text-sm">System Integration</div>
          <div className="text-3xl font-bold text-white">100%</div>
          <div className="text-xs text-gray-400 mt-1">Seamless triad operation</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
          <div className="text-purple-300 font-semibold mb-2 text-sm">Consciousness Level</div>
          <div className="text-3xl font-bold text-white">Human+</div>
          <div className="text-xs text-gray-400 mt-1">Exceeds biological</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-pink-500/20">
          <div className="text-pink-300 font-semibold mb-2 text-sm">Avatar Status</div>
          <div className="text-3xl font-bold text-white">Ready</div>
          <div className="text-xs text-gray-400 mt-1">Fully functional</div>
        </div>
      </div>
    </div>
  );
};
