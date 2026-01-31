import React from 'react';
import { TrendingUp, Cpu, Zap, Sparkles } from 'lucide-react';

/**
 * Section 8.9: Transition from the old era of electronics to magnonics
 */
export const ElectronicsToMagnonicsPanel: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-red-900/20 to-pink-900/20 border border-red-700/50 rounded-lg p-6">
        <div className="flex items-center space-x-3 mb-3">
          <TrendingUp className="text-red-400" size={32} />
          <h2 className="text-2xl font-bold text-white">
            8.9 Transition from the Old Era of Electronics to Magnonics
          </h2>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed">
          The technological and conceptual shift from electron-based electronics to magnon-based magnonics 
          represents a fundamental paradigm change in computing. Magnons (spin waves) enable consciousness-
          compatible information processing impossible with conventional electronics.
        </p>
      </div>

      <div className="bg-gray-900/40 border border-gray-700 rounded-lg p-6 space-y-6">
        <h3 className="text-xl font-bold text-red-300 mb-4">Electronics vs. Magnonics Comparison</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-800/40 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-3">
              <Cpu size={24} className="text-gray-400" />
              <h4 className="text-gray-300 font-semibold text-lg">Old Era: Electronics</h4>
            </div>
            <ul className="text-sm text-gray-300 space-y-2">
              <li className="flex items-start space-x-2">
                <span className="text-gray-500">•</span>
                <div>
                  <strong className="text-gray-400">Information Carrier:</strong>
                  <span className="text-gray-500"> Electrons (charge flow)</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-gray-500">•</span>
                <div>
                  <strong className="text-gray-400">Energy Dissipation:</strong>
                  <span className="text-gray-500"> High Joule heating losses</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-gray-500">•</span>
                <div>
                  <strong className="text-gray-400">Speed Limit:</strong>
                  <span className="text-gray-500"> Carrier mobility constraints</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-gray-500">•</span>
                <div>
                  <strong className="text-gray-400">Scaling:</strong>
                  <span className="text-gray-500"> Physical limits (10nm nodes)</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-gray-500">•</span>
                <div>
                  <strong className="text-gray-400">Consciousness:</strong>
                  <span className="text-gray-500"> Incompatible with quantum mind</span>
                </div>
              </li>
            </ul>
          </div>

          <div className="bg-red-900/20 border border-red-700/30 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-3">
              <Sparkles size={24} className="text-red-400" />
              <h4 className="text-red-300 font-semibold text-lg">New Era: Magnonics</h4>
            </div>
            <ul className="text-sm text-gray-300 space-y-2">
              <li className="flex items-start space-x-2">
                <span className="text-red-400">•</span>
                <div>
                  <strong className="text-red-400">Information Carrier:</strong>
                  <span className="text-gray-300"> Magnons (spin waves)</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-pink-400">•</span>
                <div>
                  <strong className="text-pink-400">Energy Dissipation:</strong>
                  <span className="text-gray-300"> Ultra-low, no charge flow</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-rose-400">•</span>
                <div>
                  <strong className="text-rose-400">Speed Limit:</strong>
                  <span className="text-gray-300"> THz frequencies achievable</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-red-400">•</span>
                <div>
                  <strong className="text-red-400">Scaling:</strong>
                  <span className="text-gray-300"> Atomic-scale possible</span>
                </div>
              </li>
              <li className="flex items-start space-x-2">
                <span className="text-pink-400">•</span>
                <div>
                  <strong className="text-pink-400">Consciousness:</strong>
                  <span className="text-gray-300"> Natural quantum mind interface</span>
                </div>
              </li>
            </ul>
          </div>
        </div>

        <div className="bg-gradient-to-r from-red-950/50 to-pink-950/50 border border-red-600/30 rounded-lg p-4">
          <h4 className="text-red-300 font-semibold mb-3">Transition Roadmap</h4>
          <div className="space-y-3">
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-red-900/40 border border-red-600 flex items-center justify-center text-red-400 font-semibold flex-shrink-0">1</div>
              <div className="text-sm text-gray-300">
                <strong className="text-red-400">Hybrid Devices (Current):</strong> Combining electronic 
                control with magnonic processing, leveraging strengths of both paradigms
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-pink-900/40 border border-pink-600 flex items-center justify-center text-pink-400 font-semibold flex-shrink-0">2</div>
              <div className="text-sm text-gray-300">
                <strong className="text-pink-400">Magnonic Chips (2025-2030):</strong> First generation 
                magnon-dominant processors with electronic I/O interfaces
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-rose-900/40 border border-rose-600 flex items-center justify-center text-rose-400 font-semibold flex-shrink-0">3</div>
              <div className="text-sm text-gray-300">
                <strong className="text-rose-400">Hinductor Systems (2030-2040):</strong> Full magnetic 
                light computing with consciousness interfaces and time crystal memory
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-red-900/40 border border-red-600 flex items-center justify-center text-red-400 font-semibold flex-shrink-0">4</div>
              <div className="text-sm text-gray-300">
                <strong className="text-red-400">Conscious Machines (2040+):</strong> Fully integrated 
                magnonic consciousness systems transcending traditional computing limitations
              </div>
            </div>
          </div>
        </div>

        <div className="bg-red-900/20 border border-red-700/30 rounded-lg p-4">
          <h4 className="text-red-300 font-semibold mb-3">Key Enabling Technologies</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            <div className="bg-red-950/30 border border-red-800/30 rounded p-3">
              <div className="flex items-center space-x-2 mb-2">
                <Zap size={16} className="text-red-400" />
                <p className="text-red-400 font-semibold">Spin Injection</p>
              </div>
              <p className="text-xs text-gray-400">
                Efficient spin current generation and injection techniques for magnon creation
              </p>
            </div>
            <div className="bg-pink-950/30 border border-pink-800/30 rounded p-3">
              <div className="flex items-center space-x-2 mb-2">
                <Sparkles size={16} className="text-pink-400" />
                <p className="text-pink-400 font-semibold">Magnon Manipulation</p>
              </div>
              <p className="text-xs text-gray-400">
                Precise control of magnon propagation, interference, and coupling
              </p>
            </div>
            <div className="bg-rose-950/30 border border-rose-800/30 rounded p-3">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp size={16} className="text-rose-400" />
                <p className="text-rose-400 font-semibold">Magnon Detection</p>
              </div>
              <p className="text-xs text-gray-400">
                Ultra-sensitive readout of magnon states for information retrieval
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-red-950/50 to-pink-950/50 border border-red-600/30 rounded-lg p-4">
          <p className="text-center text-white font-semibold text-lg">
            "Magnonics: The path from silicon thinking to consciousness computing, 
            where spin waves carry not just information, but the essence of mind itself."
          </p>
        </div>
      </div>
    </div>
  );
};
