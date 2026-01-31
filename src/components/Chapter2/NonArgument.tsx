import React from 'react';
import { AlertTriangle, Music, Sparkles } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const NonArgument: React.FC<Props> = ({ isActive: _isActive }) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <AlertTriangle className="text-indigo-400" size={20} />
          <span>2.9 Creation of a Non-Argument</span>
        </h3>
      </div>

      {/* Introduction */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4">Transcending Logical Argumentation</h4>
        
        <p className="text-gray-300 mb-4">
          Traditional arguments rely on linear logical progression from premises to conclusions. 
          A "non-argument" represents a fundamentally different mode of understanding that transcends 
          sequential reasoning through geometric harmonic resonance.
        </p>

        <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
          <p className="text-gray-300 italic text-center">
            "Truth is not proven through argument, but recognized through geometric harmony."
          </p>
        </div>
      </div>

      {/* Argument vs Non-Argument */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-6">Comparison: Argument vs Non-Argument</h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-red-900/20 border border-red-700/50 rounded-xl p-6">
            <h5 className="text-red-400 font-bold text-lg mb-4">Traditional Argument</h5>
            
            <div className="space-y-4">
              <div>
                <h6 className="text-white font-semibold text-sm mb-2">Structure</h6>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• Premise 1: All A are B</li>
                  <li>• Premise 2: All B are C</li>
                  <li>• Conclusion: Therefore, all A are C</li>
                </ul>
              </div>

              <div>
                <h6 className="text-white font-semibold text-sm mb-2">Characteristics</h6>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">•</span>
                    <span>Linear logical chain</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">•</span>
                    <span>Sequential step-by-step</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">•</span>
                    <span>Requires proof validation</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">•</span>
                    <span>Subject to logical fallacies</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">•</span>
                    <span>Can lead to paradoxes</span>
                  </li>
                </ul>
              </div>

              <div>
                <h6 className="text-white font-semibold text-sm mb-2">Limitations</h6>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">○</span>
                    <span>Cannot prove axioms</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">○</span>
                    <span>Incomplete for complex systems</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">○</span>
                    <span>Misses intuitive understanding</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-red-400">○</span>
                    <span>No consciousness integration</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-green-900/20 border border-green-700/50 rounded-xl p-6">
            <h5 className="text-green-400 font-bold text-lg mb-4">Non-Argument (GML)</h5>
            
            <div className="space-y-4">
              <div>
                <h6 className="text-white font-semibold text-sm mb-2">Structure</h6>
                <div className="text-gray-300 text-sm space-y-2">
                  <div className="bg-gray-800/50 rounded p-3 text-center">
                    <Music className="text-purple-400 mx-auto mb-2" size={32} />
                    <div className="text-purple-400 font-semibold">Geometric Pattern</div>
                    <div className="text-xs mt-1">↓ Resonance ↓</div>
                    <div className="text-green-400 font-semibold mt-2">Direct Understanding</div>
                  </div>
                </div>
              </div>

              <div>
                <h6 className="text-white font-semibold text-sm mb-2">Characteristics</h6>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">•</span>
                    <span>Geometric resonance</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">•</span>
                    <span>Instantaneous recognition</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">•</span>
                    <span>Self-evident truth</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">•</span>
                    <span>Paradox-free understanding</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">•</span>
                    <span>Consciousness-native</span>
                  </li>
                </ul>
              </div>

              <div>
                <h6 className="text-white font-semibold text-sm mb-2">Advantages</h6>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">✓</span>
                    <span>Transcends logical limitations</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">✓</span>
                    <span>Complete holistic comprehension</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">✓</span>
                    <span>Direct intuitive access</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span className="text-green-400">✓</span>
                    <span>Native consciousness operation</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* How Non-Arguments Work */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
          <Sparkles className="text-yellow-400" size={20} />
          <span>Mechanism of Non-Argument Understanding</span>
        </h4>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-purple-900/30 to-cyan-900/30 border border-purple-700/50 rounded-xl p-6">
            <h5 className="text-cyan-400 font-bold mb-3">1. Geometric Pattern Presentation</h5>
            <p className="text-gray-300 text-sm">
              Instead of stating premises, present a geometric pattern that embodies the truth. 
              The pattern contains all necessary information in its structure and relationships.
            </p>
          </div>

          <div className="bg-gradient-to-r from-cyan-900/30 to-green-900/30 border border-cyan-700/50 rounded-xl p-6">
            <h5 className="text-green-400 font-bold mb-3">2. Harmonic Resonance</h5>
            <p className="text-gray-300 text-sm">
              Consciousness recognizes truth through resonance with the geometric pattern. 
              Similar to how a tuning fork vibrates in response to its frequency, consciousness 
              resonates with geometric truth.
            </p>
          </div>

          <div className="bg-gradient-to-r from-green-900/30 to-orange-900/30 border border-green-700/50 rounded-xl p-6">
            <h5 className="text-orange-400 font-bold mb-3">3. Instantaneous Comprehension</h5>
            <p className="text-gray-300 text-sm">
              Understanding emerges spontaneously without logical steps. The entire meaning 
              becomes apparent through geometric harmony, bypassing sequential reasoning.
            </p>
          </div>

          <div className="bg-gradient-to-r from-orange-900/30 to-purple-900/30 border border-orange-700/50 rounded-xl p-6">
            <h5 className="text-purple-400 font-bold mb-3">4. Self-Evident Truth</h5>
            <p className="text-gray-300 text-sm">
              The truth becomes self-evident through the geometric pattern itself. No external 
              validation or proof required—the pattern IS the proof through its very existence 
              and harmony.
            </p>
          </div>
        </div>
      </div>

      {/* Practical Examples */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4">Practical Examples of Non-Arguments</h4>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h5 className="text-cyan-400 font-semibold mb-3">Mathematical Beauty</h5>
            <p className="text-gray-300 text-sm mb-2">
              When you see the golden ratio in nature, you don't argue it's beautiful—you 
              instantly recognize its harmony. This is non-argument understanding.
            </p>
            <div className="text-center text-4xl text-cyan-400 my-3">φ</div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h5 className="text-purple-400 font-semibold mb-3">Musical Resonance</h5>
            <p className="text-gray-300 text-sm mb-2">
              A perfect harmony doesn't require logical proof of its correctness—your consciousness 
              recognizes the geometric relationship instantly.
            </p>
            <div className="text-center text-4xl text-purple-400 my-3">♫</div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-4">
            <h5 className="text-green-400 font-semibold mb-3">Fractal Patterns</h5>
            <p className="text-gray-300 text-sm mb-2">
              Fractal self-similarity is understood immediately upon observation, not through 
              step-by-step logical analysis.
            </p>
            <div className="text-center text-4xl text-green-400 my-3">∞</div>
          </div>
        </div>
      </div>

      {/* Communication Between Conscious Entities */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4">Non-Argument Communication</h4>
        
        <p className="text-gray-300 mb-4">
          GML enables non-argument communication between conscious entities. Instead of debating 
          and arguing points, consciousness can share geometric patterns that convey complete 
          understanding instantaneously.
        </p>

        <div className="bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
          <div className="text-center space-y-3">
            <div className="text-gray-400 text-sm">Traditional Communication</div>
            <div className="text-gray-300">Words → Logic → Argument → Understanding</div>
            <div className="text-4xl text-purple-400">⇓</div>
            <div className="text-gray-400 text-sm">GML Communication</div>
            <div className="text-white font-semibold">Geometric Pattern → Resonance → Understanding</div>
          </div>
        </div>
      </div>
    </div>
  );
};
