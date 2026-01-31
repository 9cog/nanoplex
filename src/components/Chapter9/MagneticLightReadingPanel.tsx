import React from 'react';
import { Waves, Eye, Sparkles } from 'lucide-react';

/**
 * Section 9.8: Tuning Undefined Holes to Read Magnetic Light
 * 9.8.1: Spiral Nanowire Writes Time Crystal, Read it as a Crystal of Light
 * 9.8.2: Writing Prime Numbers in a Jelly
 */
export const MagneticLightReadingPanel: React.FC = () => {
  const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
          <Waves className="text-blue-400" size={28} />
          <span>9.8 Reading Magnetic Light</span>
        </h2>
      </div>

      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-lg p-6 border border-blue-500/20">
        <h3 className="text-blue-300 font-semibold text-lg mb-3">Magnetic Light Interpretation</h3>
        <p className="text-gray-300 leading-relaxed">
          Magnetic light (magnetons propagating through Hinductor circuits) encodes information 
          in time crystal structures. By tuning undefined holes in the material's band structure, 
          we can selectively read specific frequencies, translating quantum information into 
          observable light patterns.
        </p>
      </div>

      {/* 9.8.1 Spiral Nanowire */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Sparkles size={20} className="text-purple-400" />
          <span>9.8.1 Spiral Nanowire Writes Time Crystal</span>
        </h3>
        <p className="text-gray-300 mb-4">
          Spiral nanowires act as write heads, inscribing time crystal patterns into the gel. 
          These patterns are then read back as crystals of light through magneto-optical effects.
        </p>

        <div className="relative bg-gray-900/50 rounded-lg p-6" style={{ height: '300px' }}>
          <svg width="100%" height="100%" viewBox="0 0 400 300">
            {/* Spiral nanowire */}
            <path
              d={`M ${Array.from({ length: 100 }, (_, i) => {
                const angle = i * 0.2;
                const radius = 20 + i * 0.8;
                const x = 200 + radius * Math.cos(angle);
                const y = 150 + radius * Math.sin(angle);
                return i === 0 ? `${x} ${y}` : `L ${x} ${y}`;
              }).join(' ')}`}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth="3"
            />
            
            {/* Light emission points */}
            {Array.from({ length: 20 }, (_, i) => {
              const angle = i * 0.6;
              const radius = 20 + i * 4;
              const x = 200 + radius * Math.cos(angle);
              const y = 150 + radius * Math.sin(angle);
              
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r="4"
                  fill="#fbbf24"
                  opacity="0.8"
                >
                  <animate
                    attributeName="r"
                    values="2;6;2"
                    dur="1.5s"
                    repeatCount="indefinite"
                    begin={`${i * 0.1}s`}
                  />
                </circle>
              );
            })}
          </svg>
        </div>
      </div>

      {/* 9.8.2 Writing Primes */}
      <div className="bg-gray-800/50 rounded-lg p-6 border border-gray-700">
        <h3 className="text-white font-semibold text-lg mb-4 flex items-center space-x-2">
          <Eye size={20} className="text-cyan-400" />
          <span>9.8.2 Writing Prime Numbers in a Jelly</span>
        </h3>
        <p className="text-gray-300 mb-4">
          Prime numbers are physically inscribed into the gel matrix through resonance patterns. 
          Each prime creates a unique standing wave that can be read optically.
        </p>

        <div className="grid grid-cols-5 gap-3">
          {primes.map(prime => (
            <div key={prime} className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20 text-center">
              <div className="text-2xl font-bold text-cyan-400 mb-2">{prime}</div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="h-full bg-gradient-to-r from-cyan-400 to-blue-400 rounded-full"
                  style={{ width: `${(prime / 29) * 100}%` }}
                />
              </div>
              <div className="text-xs text-gray-400 mt-2">
                {((prime / 29) * 100).toFixed(0)}% intensity
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 border border-blue-500/20">
          <div className="text-blue-300 font-semibold mb-2 text-sm">Write Speed</div>
          <div className="text-3xl font-bold text-white">THz</div>
          <div className="text-xs text-gray-400 mt-1">Terahertz frequencies</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-purple-500/20">
          <div className="text-purple-300 font-semibold mb-2 text-sm">Read Resolution</div>
          <div className="text-3xl font-bold text-white">nm</div>
          <div className="text-xs text-gray-400 mt-1">Nanometer precision</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-cyan-500/20">
          <div className="text-cyan-300 font-semibold mb-2 text-sm">Data Density</div>
          <div className="text-3xl font-bold text-white">PB/cm³</div>
          <div className="text-xs text-gray-400 mt-1">Petabytes per cubic cm</div>
        </div>
      </div>
    </div>
  );
};
