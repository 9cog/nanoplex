import React, { useState } from 'react';
import { Music, Volume2 } from 'lucide-react';
import { GeometricShape } from '../../types';

interface Props {
  isActive: boolean;
}

const FIFTEEN_SHAPES: GeometricShape[] = [
  { id: 'point', name: 'Point', dimensions: 0, primeIndex: 2, harmonicFrequency: 261.63, complexity: 0.1, musicalNote: 'C4' },
  { id: 'line', name: 'Line', dimensions: 1, primeIndex: 3, harmonicFrequency: 293.66, complexity: 0.15, musicalNote: 'D4' },
  { id: 'triangle', name: 'Triangle', dimensions: 2, primeIndex: 5, harmonicFrequency: 329.63, complexity: 0.3, musicalNote: 'E4' },
  { id: 'square', name: 'Square', dimensions: 2, primeIndex: 7, harmonicFrequency: 349.23, complexity: 0.4, musicalNote: 'F4' },
  { id: 'pentagon', name: 'Pentagon', dimensions: 2, primeIndex: 11, harmonicFrequency: 392.00, complexity: 0.5, musicalNote: 'G4' },
  { id: 'hexagon', name: 'Hexagon', dimensions: 2, primeIndex: 13, harmonicFrequency: 440.00, complexity: 0.6, musicalNote: 'A4' },
  { id: 'tetrahedron', name: 'Tetrahedron', dimensions: 3, primeIndex: 17, harmonicFrequency: 493.88, complexity: 0.7, musicalNote: 'B4' },
  { id: 'cube', name: 'Cube', dimensions: 3, primeIndex: 19, harmonicFrequency: 523.25, complexity: 0.8, musicalNote: 'C5' },
  { id: 'octahedron', name: 'Octahedron', dimensions: 3, primeIndex: 23, harmonicFrequency: 587.33, complexity: 0.85, musicalNote: 'D5' },
  { id: 'dodecahedron', name: 'Dodecahedron', dimensions: 3, primeIndex: 29, harmonicFrequency: 659.25, complexity: 0.9, musicalNote: 'E5' },
  { id: 'icosahedron', name: 'Icosahedron', dimensions: 3, primeIndex: 31, harmonicFrequency: 698.46, complexity: 0.92, musicalNote: 'F5' },
  { id: 'torus', name: 'Torus', dimensions: 3, primeIndex: 37, harmonicFrequency: 783.99, complexity: 0.94, musicalNote: 'G5' },
  { id: 'mobius', name: 'Möbius Strip', dimensions: 3, primeIndex: 41, harmonicFrequency: 880.00, complexity: 0.96, musicalNote: 'A5' },
  { id: 'klein', name: 'Klein Bottle', dimensions: 4, primeIndex: 43, harmonicFrequency: 987.77, complexity: 0.98, musicalNote: 'B5' },
  { id: 'hypercube', name: 'Hypercube', dimensions: 4, primeIndex: 47, harmonicFrequency: 1046.50, complexity: 1.0, musicalNote: 'C6' }
];

export const FifteenGeometricShapes: React.FC<Props> = ({ isActive }) => {
  const [selectedShape, setSelectedShape] = useState<GeometricShape | null>(null);
  const [playingNote, setPlayingNote] = useState<string | null>(null);

  const playTone = (frequency: number, note: string) => {
    if (!isActive) return;
    
    setPlayingNote(note);
    
    // In a real implementation, use Web Audio API
    // For now, just simulate the visual feedback
    setTimeout(() => setPlayingNote(null), 500);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Music className="text-purple-400" size={20} />
          <span>2.2.2 The 15 Fundamental Geometric Shapes</span>
        </h3>
        <div className="text-sm text-gray-400">
          Complete Universal Pattern Set
        </div>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <p className="text-gray-300 mb-6">
          These 15 geometric primitives can recreate any 1D, 2D, or 3D pattern through combinations 
          and transformations, corresponding to the 15 primary primes that govern universal patterns.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {FIFTEEN_SHAPES.map((shape) => (
            <div
              key={shape.id}
              className={`bg-gray-800/50 rounded-lg p-3 cursor-pointer transition-all duration-300 ${
                selectedShape?.id === shape.id
                  ? 'border-2 border-purple-400 bg-gray-700/50'
                  : 'border border-gray-700 hover:border-gray-600'
              } ${
                playingNote === shape.musicalNote ? 'scale-105 shadow-lg shadow-purple-500/50' : ''
              }`}
              onClick={() => {
                setSelectedShape(shape);
                playTone(shape.harmonicFrequency, shape.musicalNote || '');
              }}
            >
              <div className="flex flex-col items-center space-y-2">
                <div className="w-12 h-12 flex items-center justify-center">
                  <div className="text-3xl">
                    {shape.dimensions === 0 ? '•' :
                     shape.dimensions === 1 ? '―' :
                     shape.id === 'triangle' ? '▲' :
                     shape.id === 'square' ? '■' :
                     shape.id === 'pentagon' ? '⬟' :
                     shape.id === 'hexagon' ? '⬡' :
                     shape.id === 'tetrahedron' ? '◆' :
                     shape.id === 'cube' ? '⬛' :
                     shape.id === 'octahedron' ? '◊' :
                     shape.id === 'dodecahedron' ? '⬢' :
                     shape.id === 'icosahedron' ? '◈' :
                     shape.id === 'torus' ? '◯' :
                     shape.id === 'mobius' ? '∞' :
                     shape.id === 'klein' ? '⟲' :
                     '◼'}
                  </div>
                </div>

                <div className="text-center">
                  <h4 className="text-white font-semibold text-xs">{shape.name}</h4>
                  <div className="text-purple-400 font-mono text-xs">{shape.musicalNote}</div>
                  <div className="text-gray-500 text-xs">{shape.dimensions}D</div>
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    playTone(shape.harmonicFrequency, shape.musicalNote || '');
                  }}
                  className="p-1 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                >
                  <Volume2 size={12} className="text-purple-400" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {selectedShape && (
        <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
          <h4 className="text-white font-bold text-lg mb-4">{selectedShape.name} Analysis</h4>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-gray-400 text-sm">Dimensions</div>
              <div className="text-white font-bold text-lg">{selectedShape.dimensions}D</div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-gray-400 text-sm">Prime Index</div>
              <div className="text-orange-400 font-bold text-lg">{selectedShape.primeIndex}</div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-gray-400 text-sm">Frequency</div>
              <div className="text-cyan-400 font-bold text-lg">{selectedShape.harmonicFrequency.toFixed(2)} Hz</div>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-3">
              <div className="text-gray-400 text-sm">Complexity</div>
              <div className="text-green-400 font-bold text-lg">{Math.round(selectedShape.complexity * 100)}%</div>
            </div>
          </div>

          <div className="mt-4 bg-gray-800/50 rounded-lg p-4">
            <h5 className="text-white font-semibold mb-2">Geometric Musical Properties</h5>
            <div className="space-y-2 text-sm text-gray-300">
              <p>• Musical Note: {selectedShape.musicalNote} ({selectedShape.harmonicFrequency.toFixed(2)} Hz)</p>
              <p>• Harmonic Resonance: Prime {selectedShape.primeIndex} governs structural symmetry</p>
              <p>• Dimensional Embedding: {selectedShape.dimensions}D space with fractal boundaries</p>
              <p>• Pattern Complexity: {Math.round(selectedShape.complexity * 100)}% of maximum geometric information</p>
            </div>
          </div>
        </div>
      )}

      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="text-white font-semibold mb-3">Pattern Generation Principle</h4>
        <p className="text-gray-300 text-sm mb-3">
          Any pattern in 1D, 2D, or 3D space can be decomposed into combinations of these 15 fundamental shapes. 
          This geometric alphabet provides a complete basis for encoding all spatial information through musical harmony.
        </p>
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div className="bg-gray-700/50 rounded p-2">
            <div className="text-cyan-400 font-semibold">1D Patterns</div>
            <div className="text-gray-400">Point + Line combinations</div>
          </div>
          <div className="bg-gray-700/50 rounded p-2">
            <div className="text-purple-400 font-semibold">2D Patterns</div>
            <div className="text-gray-400">Polygon compositions</div>
          </div>
          <div className="bg-gray-700/50 rounded p-2">
            <div className="text-green-400 font-semibold">3D Patterns</div>
            <div className="text-gray-400">Polyhedra assemblies</div>
          </div>
        </div>
      </div>
    </div>
  );
};
