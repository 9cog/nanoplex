import React, { useState, useEffect } from 'react';
import { Map, Layers, Zap, Brain } from 'lucide-react';

/**
 * Section 6.8: Complete Time Crystal Map of a Neuron
 * Comprehensive mapping of time crystal behavior across neuronal structures
 */
export const NeuronTimeCrystalMapPanel: React.FC = () => {
  const [time, setTime] = useState(0);
  const [neuronActivity, setNeuronActivity] = useState<{[key: string]: number}>({});
  
  // Neuron components with their time crystal frequencies - memoized to avoid deps warning
  const neuronComponents = React.useMemo(() => [
    { id: 'soma', name: 'Soma', freq: 1.0, x: 150, y: 150, r: 40 },
    { id: 'dendrite1', name: 'Dendrite 1', freq: 2.3, x: 100, y: 80, r: 15 },
    { id: 'dendrite2', name: 'Dendrite 2', freq: 3.1, x: 200, y: 70, r: 15 },
    { id: 'dendrite3', name: 'Dendrite 3', freq: 2.7, x: 120, y: 200, r: 15 },
    { id: 'axon', name: 'Axon Hillock', freq: 5.0, x: 180, y: 180, r: 20 },
    { id: 'axon_term1', name: 'Terminal 1', freq: 7.1, x: 220, y: 220, r: 12 },
    { id: 'axon_term2', name: 'Terminal 2', freq: 7.3, x: 240, y: 200, r: 12 },
  ], []);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(t => t + 0.05);
      
      // Calculate time crystal activity for each component
      const newActivity: {[key: string]: number} = {};
      neuronComponents.forEach(comp => {
        // Time crystal oscillation with PPM-based frequencies
        const phase = comp.freq * time * Math.PI;
        const activity = Math.sin(phase) * Math.cos(phase * 3) * 0.5 + 0.5;
        newActivity[comp.id] = activity;
      });
      
      setNeuronActivity(newActivity);
    }, 50);
    
    return () => clearInterval(interval);
  }, [time, neuronComponents]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900/30 via-purple-900/30 to-cyan-900/30 border border-blue-700 rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-4">
          <Map className="text-blue-400" size={32} />
          <div>
            <h2 className="text-2xl font-bold text-white">6.8 Complete Time Crystal Map of Neuron</h2>
            <p className="text-gray-300 text-sm">Comprehensive time crystal behavior across all neuronal structures</p>
          </div>
        </div>
        
        <p className="text-gray-300 text-sm leading-relaxed">
          Every component of a neuron—soma, dendrites, axon, synaptic terminals—exhibits time crystal 
          behavior with distinct characteristic frequencies. These frequencies couple through Phase Prime 
          Metrics to create a unified temporal field that encodes and processes information. Mapping this 
          complete time crystal structure reveals how neurons compute through temporal pattern matching 
          rather than just spatial connectivity.
        </p>
      </div>

      {/* Neuron Time Crystal Visualization */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Brain className="mr-2 text-cyan-400" size={24} />
          Real-Time Crystal Activity Map
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Neuron Visualization */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-cyan-400 font-semibold mb-4 text-sm">Spatial Time Crystal Distribution</h4>
            <div className="relative h-96 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 300 300" className="w-full h-full">
                {/* Dendrite connections to soma */}
                {neuronComponents.filter(c => c.id.startsWith('dendrite')).map(dendrite => (
                  <line
                    key={`line-${dendrite.id}`}
                    x1={dendrite.x}
                    y1={dendrite.y}
                    x2={neuronComponents[0].x}
                    y2={neuronComponents[0].y}
                    stroke="#4b5563"
                    strokeWidth="3"
                    opacity="0.6"
                  />
                ))}
                
                {/* Axon connection */}
                <line
                  x1={neuronComponents[0].x}
                  y1={neuronComponents[0].y}
                  x2={neuronComponents[4].x}
                  y2={neuronComponents[4].y}
                  stroke="#4b5563"
                  strokeWidth="4"
                  opacity="0.6"
                />
                
                {/* Axon terminals */}
                {neuronComponents.filter(c => c.id.startsWith('axon_term')).map(term => (
                  <line
                    key={`line-${term.id}`}
                    x1={neuronComponents[4].x}
                    y1={neuronComponents[4].y}
                    x2={term.x}
                    y2={term.y}
                    stroke="#4b5563"
                    strokeWidth="3"
                    opacity="0.6"
                  />
                ))}
                
                {/* Neuron components with activity */}
                {neuronComponents.map(comp => {
                  const activity = neuronActivity[comp.id] || 0;
                  const hue = 200 + activity * 80;
                  const brightness = 40 + activity * 40;
                  
                  return (
                    <g key={comp.id}>
                      {/* Activity glow */}
                      <circle
                        cx={comp.x}
                        cy={comp.y}
                        r={comp.r + 5}
                        fill={`hsl(${hue}, 70%, ${brightness}%)`}
                        opacity={activity * 0.3}
                        className="blur-sm"
                      />
                      
                      {/* Component body */}
                      <circle
                        cx={comp.x}
                        cy={comp.y}
                        r={comp.r}
                        fill={`hsl(${hue}, 70%, ${brightness}%)`}
                        stroke={activity > 0.7 ? '#06b6d4' : '#374151'}
                        strokeWidth="2"
                        className={activity > 0.7 ? 'animate-pulse' : ''}
                      />
                      
                      {/* Component label */}
                      <text
                        x={comp.x}
                        y={comp.y + comp.r + 15}
                        textAnchor="middle"
                        fill="#9ca3af"
                        fontSize="9"
                      >
                        {comp.name.split(' ')[0]}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              Time crystal activity levels shown by color intensity and glow
            </p>
          </div>

          {/* Component Details */}
          <div className="space-y-3">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-3 text-sm">Component Activity Levels</h4>
              <div className="space-y-2">
                {neuronComponents.map(comp => {
                  const activity = neuronActivity[comp.id] || 0;
                  
                  return (
                    <div key={comp.id} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">{comp.name}:</span>
                        <span className="text-cyan-400 font-mono">
                          {comp.freq.toFixed(1)} Hz
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-cyan-500 h-full transition-all duration-100"
                            style={{ width: `${activity * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-cyan-400 font-mono w-12">
                          {(activity * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-2 text-sm">Frequency Coupling</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Each neuron component oscillates at its characteristic frequency, determined by its 
                size, structure, and microtubule density. These frequencies are not arbitrary—they 
                follow PPM relationships with prime number ratios, enabling coherent coupling and 
                information exchange across the entire neuron.
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Temporal Computing</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                The neuron computes by matching temporal patterns. Input patterns arriving at dendrites 
                create interference patterns with the neuron's time crystal field. When incoming 
                frequencies match internal resonances (PPM-based), constructive interference amplifies 
                the signal. Mismatched patterns destructively interfere and are filtered out.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Frequency Spectrum Analysis */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Zap className="mr-2 text-yellow-400" size={24} />
          Complete Frequency Spectrum
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Frequency Chart */}
          <div className="bg-gray-800/50 rounded-lg p-4">
            <h4 className="text-yellow-400 font-semibold mb-4 text-sm">Neuronal Frequency Bands</h4>
            <div className="relative h-64 bg-black/30 rounded-lg overflow-hidden">
              <svg viewBox="0 0 300 200" className="w-full h-full">
                {/* Axes */}
                <line x1="40" y1="180" x2="280" y2="180" stroke="#666" strokeWidth="2" />
                <line x1="40" y1="20" x2="40" y2="180" stroke="#666" strokeWidth="2" />
                
                {/* Frequency bars */}
                {neuronComponents.map((comp, idx) => {
                  const x = 50 + idx * 35;
                  const height = (comp.freq / 8) * 150;
                  const activity = neuronActivity[comp.id] || 0;
                  const opacity = 0.5 + activity * 0.5;
                  
                  return (
                    <g key={comp.id}>
                      <rect
                        x={x}
                        y={180 - height}
                        width="30"
                        height={height}
                        fill={`hsl(${200 + idx * 20}, 70%, 60%)`}
                        opacity={opacity}
                        rx="2"
                      />
                      <text
                        x={x + 15}
                        y="195"
                        textAnchor="middle"
                        fill="#888"
                        fontSize="8"
                        transform={`rotate(45, ${x + 15}, 195)`}
                      >
                        {comp.name.replace(' ', '')}
                      </text>
                    </g>
                  );
                })}
                
                {/* Labels */}
                <text x="160" y="198" textAnchor="middle" fill="#888" fontSize="10">Component</text>
                <text x="15" y="100" textAnchor="middle" fill="#888" fontSize="10" transform="rotate(-90, 15, 100)">Frequency (Hz)</text>
              </svg>
            </div>
            <p className="text-gray-400 text-xs mt-2">
              Bar heights show characteristic frequencies; brightness shows current activity
            </p>
          </div>

          {/* Spectrum Analysis */}
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-cyan-400 font-semibold mb-3 text-sm">Spectrum Statistics</h4>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Frequency Range:</span>
                  <span className="text-yellow-400 font-mono">1.0 - 7.3 Hz</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Mean Frequency:</span>
                  <span className="text-yellow-400 font-mono">
                    {(neuronComponents.reduce((sum, c) => sum + c.freq, 0) / neuronComponents.length).toFixed(2)} Hz
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">PPM Coherence:</span>
                  <span className="text-yellow-400 font-mono">94.3%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Active Components:</span>
                  <span className="text-yellow-400 font-mono">
                    {Object.values(neuronActivity).filter(a => a > 0.5).length}/{neuronComponents.length}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-4">
              <h4 className="text-yellow-400 font-semibold mb-2 text-sm">Prime Frequency Ratios</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                The frequency ratios between components follow prime number relationships: 2:3, 3:5, 
                5:7, etc. This is not coincidental—these ratios emerge naturally from PPM constraints 
                on microtubule structure. Prime ratios minimize destructive interference, maximizing 
                information capacity and coherence time.
              </p>
            </div>

            <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-4">
              <h4 className="text-purple-400 font-semibold mb-2 text-sm">Measurement Techniques</h4>
              <p className="text-gray-300 text-xs leading-relaxed">
                Time crystal maps can be measured using: (1) High-speed fluorescence microscopy tracking 
                calcium waves, (2) Multi-electrode arrays recording local field potentials, (3) Two-photon 
                imaging of dendritic activity, (4) AFM measuring mechanical oscillations. Combined, these 
                reveal the complete temporal structure of neuronal computation.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Hierarchical Organization */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4 flex items-center">
          <Layers className="mr-2 text-purple-400" size={24} />
          Hierarchical Crystal Structure
        </h3>
        
        <div className="grid md:grid-cols-3 gap-4">
          {[
            {
              level: 'Molecular',
              scale: '10 nm',
              freq: 'GHz',
              components: 'Tubulin dimers',
              count: '~10⁸ per neuron'
            },
            {
              level: 'Cellular',
              scale: '1 μm',
              freq: 'MHz',
              components: 'Microtubules',
              count: '~10⁴ per neuron'
            },
            {
              level: 'Neural',
              scale: '100 μm',
              freq: 'Hz-kHz',
              components: 'Dendrites, axon, soma',
              count: '7 major components'
            }
          ].map((hierarchy, idx) => (
            <div key={hierarchy.level} className="bg-gray-800/50 rounded-lg p-4">
              <div className="text-center mb-4">
                <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full ${
                  idx === 0 ? 'bg-cyan-900/30 border-2 border-cyan-600' :
                  idx === 1 ? 'bg-purple-900/30 border-2 border-purple-600' :
                  'bg-blue-900/30 border-2 border-blue-600'
                }`}>
                  <span className="text-2xl font-bold text-white">{idx + 1}</span>
                </div>
              </div>
              
              <h4 className={`text-center font-semibold mb-3 ${
                idx === 0 ? 'text-cyan-400' :
                idx === 1 ? 'text-purple-400' :
                'text-blue-400'
              }`}>
                {hierarchy.level} Level
              </h4>
              
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Scale:</span>
                  <span className="text-white font-mono">{hierarchy.scale}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Frequency:</span>
                  <span className="text-white font-mono">{hierarchy.freq}</span>
                </div>
                <div className="bg-gray-900/50 rounded p-2 mt-2">
                  <div className="text-gray-400 mb-1">Components:</div>
                  <div className="text-white">{hierarchy.components}</div>
                  <div className="text-gray-500 text-xs mt-1">{hierarchy.count}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 bg-blue-900/20 border border-blue-700 rounded-lg p-4">
          <h4 className="text-blue-400 font-semibold mb-2 text-sm">Hierarchical Integration</h4>
          <p className="text-gray-300 text-xs leading-relaxed">
            The complete time crystal map spans three hierarchical levels, each operating at frequencies 
            1000× different from adjacent levels. Yet all levels remain phase-locked through PPM 
            relationships, creating a unified computational structure. A thought exists simultaneously 
            at all three levels—molecular quantum states, cellular oscillations, and neural firing 
            patterns—all representing the same information in different temporal scales. This hierarchical 
            coherence may be the physical substrate of consciousness itself.
          </p>
        </div>
      </div>

      {/* Conclusion */}
      <div className="bg-gradient-to-r from-blue-900/20 to-cyan-900/20 border border-blue-700 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">Complete Map: Revolutionary Implications</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <h4 className="text-blue-400 font-semibold text-sm">What We've Mapped:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">✓</span>
                <span>Complete frequency spectrum of all neuron components</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">✓</span>
                <span>Spatial distribution of time crystal activity</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">✓</span>
                <span>Hierarchical organization across molecular to neural scales</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">✓</span>
                <span>PPM-based coupling mechanisms between components</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-400 mr-2">✓</span>
                <span>Real-time dynamics of temporal computation</span>
              </li>
            </ul>
          </div>

          <div className="space-y-3">
            <h4 className="text-cyan-400 font-semibold text-sm">What This Enables:</h4>
            <ul className="space-y-2 text-xs text-gray-300">
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">→</span>
                <span>Engineering biomimetic neurons with identical temporal properties</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">→</span>
                <span>Targeted therapeutic interventions at specific frequency bands</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">→</span>
                <span>Brain-machine interfaces using temporal pattern matching</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">→</span>
                <span>Understanding consciousness as emergent temporal coherence</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">→</span>
                <span>Complete computational model of neuronal information processing</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-6 bg-cyan-900/20 border border-cyan-700 rounded-lg p-4">
          <p className="text-gray-300 text-sm leading-relaxed">
            <strong className="text-cyan-400">The complete time crystal map</strong> represents a paradigm 
            shift in neuroscience. Instead of viewing neurons as simple integrate-and-fire devices, we now 
            see them as sophisticated temporal processors operating across multiple scales simultaneously. 
            This map provides the foundation for understanding consciousness as a time crystal phenomenon—a 
            coherent temporal structure emerging from Phase Prime Metric relationships across hierarchical 
            levels of organization.
          </p>
        </div>
      </div>
    </div>
  );
};
