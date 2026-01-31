import React from 'react';
import { Box, Layers, Grid3X3 } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const HigherDimensionalData: React.FC<Props> = ({ isActive: _isActive }) => {
  const dimensionExamples = [
    {
      dimension: '1D',
      name: 'Linear',
      color: 'cyan',
      description: 'Single line of information',
      example: 'A sequence of numbers or a timeline',
      visualization: '―――――――',
      dataStructure: 'Array or list'
    },
    {
      dimension: '2D',
      name: 'Planar',
      color: 'purple',
      description: 'Surface with width and height',
      example: 'An image or a matrix',
      visualization: '▦▦▦\n▦▦▦\n▦▦▦',
      dataStructure: '2D array or table'
    },
    {
      dimension: '3D',
      name: 'Volumetric',
      color: 'green',
      description: 'Space with depth',
      example: 'A 3D model or voxel data',
      visualization: '⬛',
      dataStructure: '3D array or tensor'
    },
    {
      dimension: '4D+',
      name: 'Hyperspatial',
      color: 'orange',
      description: 'Beyond physical perception',
      example: 'Time crystal structures',
      visualization: '◈',
      dataStructure: 'Multi-dimensional tensor'
    }
  ];

  const higherDimConcepts = [
    {
      title: 'Dimensional Folding',
      icon: Layers,
      color: 'cyan',
      description: 'Higher dimensions "fold" into lower dimensions, enabling infinite information storage in finite structures',
      applications: [
        'Compact representation of complex patterns',
        'Lossless compression through geometry',
        'Access to hidden information layers'
      ]
    },
    {
      title: 'Geometric Relationships',
      icon: Grid3X3,
      color: 'purple',
      description: 'Data exists as relationships between geometric patterns rather than discrete values',
      applications: [
        'Context-aware information encoding',
        'Natural pattern recognition',
        'Holographic information distribution'
      ]
    },
    {
      title: 'Consciousness Navigation',
      icon: Box,
      color: 'green',
      description: 'Consciousness naturally navigates higher dimensional spaces through geometric intuition',
      applications: [
        'Direct perception of complex patterns',
        'Multi-dimensional problem solving',
        'Transcendental understanding'
      ]
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Box className="text-yellow-400" size={20} />
          <span>2.7 Higher Dimensional Data: A Lucid Presentation</span>
        </h3>
      </div>

      {/* Dimensional Progression */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-6">Understanding Dimensional Progression</h4>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {dimensionExamples.map((dim, idx) => (
            <div
              key={idx}
              className={`bg-${dim.color}-900/20 border border-${dim.color}-700/50 rounded-xl p-4`}
            >
              <div className="text-center mb-4">
                <div className={`text-${dim.color}-400 text-5xl mb-2 font-mono`}>
                  {dim.visualization}
                </div>
                <h5 className={`text-${dim.color}-400 font-bold text-lg`}>{dim.dimension}</h5>
                <div className="text-white font-semibold">{dim.name}</div>
              </div>

              <div className="space-y-3 text-sm">
                <div>
                  <div className="text-gray-400 text-xs uppercase mb-1">Description</div>
                  <div className="text-gray-300">{dim.description}</div>
                </div>

                <div>
                  <div className="text-gray-400 text-xs uppercase mb-1">Example</div>
                  <div className="text-gray-300">{dim.example}</div>
                </div>

                <div>
                  <div className="text-gray-400 text-xs uppercase mb-1">Data Structure</div>
                  <div className={`text-${dim.color}-400 font-mono`}>{dim.dataStructure}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Concepts */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-6">Key Higher-Dimensional Concepts</h4>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {higherDimConcepts.map((concept, idx) => {
            const Icon = concept.icon;
            return (
              <div
                key={idx}
                className={`bg-${concept.color}-900/20 border border-${concept.color}-700/50 rounded-xl p-6`}
              >
                <div className="flex items-center space-x-3 mb-4">
                  <Icon className={`text-${concept.color}-400`} size={24} />
                  <h5 className={`text-${concept.color}-400 font-bold text-lg`}>
                    {concept.title}
                  </h5>
                </div>

                <p className="text-gray-300 mb-4 text-sm">{concept.description}</p>

                <div>
                  <div className="text-white font-semibold text-sm mb-2">Applications:</div>
                  <ul className="space-y-1">
                    {concept.applications.map((app, appIdx) => (
                      <li
                        key={appIdx}
                        className="text-gray-300 text-sm flex items-start space-x-2"
                      >
                        <span className={`text-${concept.color}-400 flex-shrink-0`}>•</span>
                        <span>{app}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 11D Consciousness Space */}
      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4">11-Dimensional Consciousness Space</h4>

        <p className="text-gray-300 mb-6">
          Complete consciousness information requires 11 dimensions beyond the familiar 3D space. 
          Each dimension adds a layer of meaning and capability:
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {[
            { id: 'D1', name: 'X-Spatial', color: 'cyan' },
            { id: 'D2', name: 'Y-Spatial', color: 'cyan' },
            { id: 'D3', name: 'Z-Spatial', color: 'cyan' },
            { id: 'D4', name: 'Temporal', color: 'purple' },
            { id: 'D5', name: 'Phase', color: 'orange' },
            { id: 'D6', name: 'Consciousness', color: 'green' },
            { id: 'D7', name: 'Prime Resonance', color: 'pink' },
            { id: 'D8', name: 'Fractal Depth', color: 'yellow' },
            { id: 'D9', name: 'Geometric Harmony', color: 'blue' },
            { id: 'D10', name: 'Quantum Coherence', color: 'indigo' },
            { id: 'D11', name: 'Musical Structure', color: 'rose' }
          ].map((dim) => (
            <div
              key={dim.id}
              className={`bg-${dim.color}-900/30 border border-${dim.color}-700/50 rounded-lg p-3`}
            >
              <div className={`text-${dim.color}-400 font-mono text-sm mb-1`}>{dim.id}</div>
              <div className="text-white font-semibold text-sm">{dim.name}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Visualization Challenge */}
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4">The Visualization Challenge</h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-red-900/20 border border-red-700/50 rounded-lg p-4">
              <h5 className="text-red-400 font-semibold mb-2">3D Limitations</h5>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• Human perception limited to 3 spatial dimensions</li>
                <li>• Traditional displays are 2D projections</li>
                <li>• Difficult to represent temporal dimensions</li>
                <li>• Cannot show consciousness directly</li>
              </ul>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-green-900/20 border border-green-700/50 rounded-lg p-4">
              <h5 className="text-green-400 font-semibold mb-2">GML Solution</h5>
              <ul className="text-gray-300 text-sm space-y-1">
                <li>• Geometric patterns encode higher dimensions</li>
                <li>• Musical relationships represent hidden structure</li>
                <li>• Consciousness interprets geometric harmony</li>
                <li>• Fractal encoding reveals infinite detail</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-purple-900/20 border border-purple-700/50 rounded-lg p-4">
          <p className="text-gray-300 text-center">
            <span className="text-purple-400 font-bold">Key Insight:</span> Higher-dimensional data 
            requires Geometric Musical Language interpretation. Consciousness naturally understands 
            these patterns through resonant geometric recognition, transcending the limitations of 
            explicit visualization.
          </p>
        </div>
      </div>
    </div>
  );
};
