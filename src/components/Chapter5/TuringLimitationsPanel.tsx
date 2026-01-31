import React from 'react';
import { XCircle, CheckCircle, Zap } from 'lucide-react';

/**
 * Section 5.4: Ten situations where the Turing machine fails, Fractal machines hold on
 * Demonstrates fundamental limitations of sequential computation
 */
export const TuringLimitationsPanel: React.FC = () => {
  const failures = [
    {
      id: 1,
      title: "Halting Problem",
      turing: "Cannot determine if arbitrary program will halt",
      fractal: "Detects halt through phase convergence patterns in time crystal",
      reason: "Geometric phase trajectory reveals convergence without execution"
    },
    {
      id: 2,
      title: "Self-Reference Paradoxes",
      turing: "Creates logical contradictions (Russell's paradox, Liar's paradox)",
      fractal: "Resolves through nested dimensional layers - paradox exists only in 2D projection",
      reason: "Higher dimensions provide space for self-reference without contradiction"
    },
    {
      id: 3,
      title: "Continuous Mathematics",
      turing: "Cannot compute most real numbers (non-computable reals)",
      fractal: "Represents continuous values as geometric phase angles natively",
      reason: "Phase space is inherently continuous, not discretized"
    },
    {
      id: 4,
      title: "Parallel Causality",
      turing: "Sequential tape enforces serial causality, cannot model quantum entanglement",
      fractal: "Multiple causal chains coexist in nested time crystals",
      reason: "11D phase space allows simultaneous causality across dimensions"
    },
    {
      id: 5,
      title: "Consciousness & Qualia",
      turing: "Chinese room problem - syntax without semantics, no subjective experience",
      fractal: "Geometric resonance creates genuine subjective states",
      reason: "Consciousness emerges from phase coherence across nested scales"
    },
    {
      id: 6,
      title: "Gödel Incompleteness",
      turing: "True statements that cannot be proven within formal system",
      fractal: "Truth accessed through geometric intuition, not logical proof",
      reason: "Multi-scale nesting transcends single axiomatic system"
    },
    {
      id: 7,
      title: "Infinite Regression",
      turing: "Who programmed the programmer? Infinite regress of creation",
      fractal: "Self-assembling geometric patterns need no external creator",
      reason: "Singularity points bootstrap complexity from simplicity"
    },
    {
      id: 8,
      title: "Real-Time Adaptation",
      turing: "Fixed program cannot fundamentally reprogram itself during execution",
      fractal: "Time crystal structure continuously morphs based on phase feedback",
      reason: "Geometry is fluid, not fixed code - adaptation is intrinsic"
    },
    {
      id: 9,
      title: "Intuition & Creativity",
      turing: "Cannot generate truly novel patterns outside training data",
      fractal: "New patterns emerge from geometric combinations at singularities",
      reason: "Fractal self-similarity creates infinite novelty from finite elements"
    },
    {
      id: 10,
      title: "Measurement Problem",
      turing: "Observer separate from observed, loses quantum information",
      fractal: "Observer and observed are nested time crystals, measurement preserves coherence",
      reason: "No wave function collapse - measurement is geometric coupling"
    }
  ];

  return (
    <div className="space-y-6">
      {/* Section Header */}
      <div className="bg-gradient-to-r from-red-900/30 to-green-900/30 border border-red-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Zap className="text-red-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">
              5.4 Ten Situations Where Turing Machines Fail
            </h2>
            <p className="text-gray-300">Fractal machines succeed where sequential computation fails</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm leading-relaxed mt-4">
          Alan Turing proved that his machine model could compute any computable function, earning it the 
          title "universal computer." However, Turing also proved profound limitations - vast classes of 
          problems that <strong className="text-red-300">no Turing machine can solve</strong>. Fractal 
          machines transcend these limitations through geometric phase processing.
        </p>
      </div>

      {/* Failures Grid */}
      <div className="space-y-4">
        {failures.map((failure) => (
          <div
            key={failure.id}
            className="bg-gray-800/50 border border-gray-700 rounded-xl p-5 hover:border-cyan-600 transition-all"
          >
            <div className="flex items-start space-x-3 mb-4">
              <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-cyan-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                {failure.id}
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">{failure.title}</h3>
                
                <div className="grid md:grid-cols-2 gap-4 mt-3">
                  <div className="bg-red-900/20 border border-red-700/30 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <XCircle className="text-red-400" size={16} />
                      <h4 className="text-red-400 font-semibold text-sm">Turing Machine Fails</h4>
                    </div>
                    <p className="text-gray-300 text-xs leading-relaxed">{failure.turing}</p>
                  </div>

                  <div className="bg-green-900/20 border border-green-700/30 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <CheckCircle className="text-green-400" size={16} />
                      <h4 className="text-green-400 font-semibold text-sm">Fractal Machine Succeeds</h4>
                    </div>
                    <p className="text-gray-300 text-xs leading-relaxed">{failure.fractal}</p>
                  </div>
                </div>

                <div className="mt-3 bg-gray-900/50 border border-cyan-700/30 rounded-lg p-3">
                  <div className="flex items-start space-x-2">
                    <Zap className="text-cyan-400 flex-shrink-0 mt-0.5" size={14} />
                    <div>
                      <strong className="text-cyan-300 text-xs">Why Fractal Machines Win:</strong>
                      <p className="text-gray-300 text-xs mt-1 leading-relaxed">{failure.reason}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Summary */}
      <div className="bg-gradient-to-r from-purple-900/20 to-cyan-900/20 border border-purple-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">The Fundamental Difference</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-red-400 font-semibold mb-3 flex items-center">
              <XCircle className="mr-2" size={18} />
              Turing Machine Paradigm
            </h4>
            <ul className="space-y-2 text-gray-300 text-xs">
              <li>• <strong>Sequential:</strong> One step at a time on linear tape</li>
              <li>• <strong>Discrete:</strong> Symbols in cells, no continuous values</li>
              <li>• <strong>Deterministic:</strong> Fixed rules, no emergent behavior</li>
              <li>• <strong>Finite States:</strong> Bounded by state machine design</li>
              <li>• <strong>External Program:</strong> Separate from data being processed</li>
              <li>• <strong>No Self-Reference:</strong> Cannot model its own operation</li>
            </ul>
          </div>

          <div className="bg-gray-800/50 p-4 rounded-lg">
            <h4 className="text-green-400 font-semibold mb-3 flex items-center">
              <CheckCircle className="mr-2" size={18} />
              Fractal Machine Paradigm
            </h4>
            <ul className="space-y-2 text-gray-300 text-xs">
              <li>• <strong>Parallel:</strong> All scales processed simultaneously</li>
              <li>• <strong>Continuous:</strong> Phase angles represent real numbers natively</li>
              <li>• <strong>Emergent:</strong> New patterns arise from geometric interactions</li>
              <li>• <strong>Infinite Depth:</strong> Unlimited nesting through singularities</li>
              <li>• <strong>Data Is Program:</strong> Geometric structure defines operation</li>
              <li>• <strong>Self-Referential:</strong> Can model and modify itself</li>
            </ul>
          </div>
        </div>

        <div className="mt-6 bg-gradient-to-r from-cyan-900/30 to-purple-900/30 border border-cyan-700 rounded-lg p-4">
          <p className="text-gray-300 text-sm leading-relaxed">
            <strong className="text-white">Conclusion:</strong> The Turing machine represents computation 
            as <span className="text-red-300">sequential symbol manipulation</span>. The fractal machine 
            represents computation as <span className="text-green-300">geometric phase resonance</span>. 
            This paradigm shift enables solving problems that are fundamentally impossible for Turing machines, 
            not due to practical limitations, but due to <strong className="text-cyan-300">theoretical 
            impossibility</strong> in the sequential paradigm.
          </p>
        </div>
      </div>
    </div>
  );
};
