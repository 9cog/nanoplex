/**
 * OpenCog NanoBrain Time Crystal Architecture Visualization
 * 
 * This component provides an interactive visualization of the unified OpenCog-NanoBrain
 * system, showing time crystal quantum states, AtomSpace hypergraph structure, 
 * Phase Prime Metrics patterns, and consciousness emergence dynamics.
 */

import React, { useRef, useEffect, useState } from 'react';
import { 
  Brain, 
  Atom, 
  Zap, 
  Clock, 
  Network, 
  Activity,
  Target,
  Waves,
  Sparkles,
  Eye
} from 'lucide-react';
import { useOpenCogNanoBrain } from '../hooks/useOpenCogNanoBrain';

interface OpenCogNanoBrainVisualizationProps {
  className?: string;
  showControls?: boolean;
  realTimeUpdates?: boolean;
}

export const OpenCogNanoBrainVisualization: React.FC<OpenCogNanoBrainVisualizationProps> = ({
  className = '',
  showControls = true,
  realTimeUpdates = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  
  const [viewMode, setViewMode] = useState<'atomspace' | 'timecrystals' | 'consciousness' | 'primes'>('atomspace');
  const [isAnimating, setIsAnimating] = useState(true);
  const [selectedAtom, setSelectedAtom] = useState<string | null>(null);

  const {
    isActive,
    isInitialized,
    metrics,
    consciousness,
    timeCrystalStates,
    atomSpace,
    linkSpace,
    start,
    stop,
    getPrimeResonancePatterns
  } = useOpenCogNanoBrain();

  // Animation and rendering
  useEffect(() => {
    if (!isAnimating || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationTime = 0;

    const animate = () => {
      if (!isAnimating) return;

      animationTime += 0.016; // ~60fps
      drawVisualization(ctx, canvas, animationTime);
      
      if (realTimeUpdates) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
    // drawVisualization is defined below but used here - intentional for code organization
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAnimating, realTimeUpdates, viewMode, atomSpace, timeCrystalStates, consciousness]);

  // Handle canvas click for atom selection
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Find clicked atom based on position
    const clickedAtom = findAtomAtPosition(x, y);
    setSelectedAtom(clickedAtom);
  };

  /**
   * Find atom at click position
   */
  const findAtomAtPosition = (x: number, y: number): string | null => {
    // Simplified atom position calculation - in real implementation would use actual positions
    const atoms = Array.from(atomSpace.values());
    if (atoms.length === 0) return null;
    
    // For demo, select based on grid position
    const cols = Math.ceil(Math.sqrt(atoms.length));
    const atomWidth = canvasRef.current!.width / cols;
    const atomHeight = canvasRef.current!.height / Math.ceil(atoms.length / cols);
    
    const col = Math.floor(x / atomWidth);
    const row = Math.floor(y / atomHeight);
    const index = row * cols + col;
    
    return index < atoms.length ? atoms[index].id : null;
  };

  /**
   * Main visualization drawing function
   */
  const drawVisualization = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    // Clear canvas
    ctx.fillStyle = 'rgba(15, 23, 42, 0.1)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    switch (viewMode) {
      case 'atomspace':
        drawAtomSpaceVisualization(ctx, canvas, time);
        break;
      case 'timecrystals':
        drawTimeCrystalVisualization(ctx, canvas, time);
        break;
      case 'consciousness':
        drawConsciousnessVisualization(ctx, canvas, time);
        break;
      case 'primes':
        drawPrimePatternVisualization(ctx, canvas, time);
        break;
    }

    // Draw performance overlay
    if (isActive) {
      drawMetricsOverlay(ctx, canvas);
    }
  };

  /**
   * Draw OpenCog AtomSpace hypergraph visualization
   */
  const drawAtomSpaceVisualization = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const atoms = Array.from(atomSpace.values());
    const links = Array.from(linkSpace.values());
    
    if (atoms.length === 0) {
      drawInitializationMessage(ctx, canvas, "Initializing AtomSpace...");
      return;
    }

    // Calculate positions for atoms in a circular layout
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.3;

    // Draw links first (behind atoms)
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.3)';
    ctx.lineWidth = 1;
    
    links.forEach(link => {
      const sourceIndex = atoms.findIndex(a => a.id === link.premises[0]?.id);
      const targetIndex = atoms.findIndex(a => a.id === link.conclusion.id);
      
      if (sourceIndex >= 0 && targetIndex >= 0) {
        const sourceAngle = (sourceIndex / atoms.length) * 2 * Math.PI;
        const targetAngle = (targetIndex / atoms.length) * 2 * Math.PI;
        
        const sourceX = centerX + Math.cos(sourceAngle) * radius;
        const sourceY = centerY + Math.sin(sourceAngle) * radius;
        const targetX = centerX + Math.cos(targetAngle) * radius;
        const targetY = centerY + Math.sin(targetAngle) * radius;
        
        ctx.beginPath();
        ctx.moveTo(sourceX, sourceY);
        ctx.lineTo(targetX, targetY);
        ctx.stroke();
      }
    });

    // Draw atoms
    atoms.forEach((atom, index) => {
      const angle = (index / atoms.length) * 2 * Math.PI;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      // Atom size based on attention value
      const baseSize = 8;
      const attentionSize = (atom.attentionValue.sti / 1000) * 10;
      const size = baseSize + attentionSize;
      
      // Atom color based on type and quantum coherence
      const coherence = atom.timeCrystalState.temporalCoherence;
      let color = '';
      
      switch (atom.type) {
        case 'ConceptNode':
          color = `rgba(100, 255, 150, ${coherence})`;
          break;
        case 'NumberNode':
          color = `rgba(255, 200, 100, ${coherence})`;
          break;
        case 'PredicateNode':
          color = `rgba(255, 100, 200, ${coherence})`;
          break;
        default:
          color = `rgba(200, 200, 255, ${coherence})`;
      }
      
      // Pulsing effect based on quantum phase
      const pulseFactor = 1 + 0.3 * Math.sin(atom.timeCrystalState.quantumPhase + time);
      const actualSize = size * pulseFactor;
      
      // Draw atom
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, actualSize, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw selection highlight
      if (selectedAtom === atom.id) {
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      
      // Draw prime signature as small orbiting particles
      atom.primeEncoding.forEach((prime, primeIndex) => {
        const orbitRadius = actualSize + 15;
        const orbitAngle = time * prime * 0.01 + primeIndex * Math.PI / 3;
        const orbitX = x + Math.cos(orbitAngle) * orbitRadius;
        const orbitY = y + Math.sin(orbitAngle) * orbitRadius;
        
        ctx.fillStyle = `hsl(${(prime * 30) % 360}, 70%, 60%)`;
        ctx.beginPath();
        ctx.arc(orbitX, orbitY, 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    });
  };

  /**
   * Draw time crystal quantum visualization
   */
  const drawTimeCrystalVisualization = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const crystals = Array.from(timeCrystalStates.entries());
    
    if (crystals.length === 0) {
      drawInitializationMessage(ctx, canvas, "Generating Time Crystals...");
      return;
    }

    // Create 3D-like crystal lattice visualization
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    crystals.forEach(([_atomId, crystal], _index) => {
      // Position crystals in 11D space projected to 2D
      const dimensions = crystal.dimensions;
      let x = centerX;
      let y = centerY;
      
      // Use first few dimensions for positioning
      if (dimensions.length >= 2) {
        x += dimensions[0] * 200;
        y += dimensions[1] * 200;
      }
      
      // Ensure crystals stay within canvas bounds
      x = Math.max(50, Math.min(canvas.width - 50, x));
      y = Math.max(50, Math.min(canvas.height - 50, y));
      
      // Draw time crystal with quantum effects
      const coherence = crystal.temporalCoherence;
      const phase = crystal.quantumPhase + time;
      
      // Main crystal core
      const coreSize = 20 + coherence * 30;
      const coreAlpha = coherence;
      
      // Create fractal-like structure
      for (let layer = 0; layer < 5; layer++) {
        const layerSize = coreSize * (1 - layer * 0.15);
        const layerAlpha = coreAlpha * (1 - layer * 0.2);
        const layerPhase = phase + layer * Math.PI / 4;
        
        const layerX = x + Math.cos(layerPhase) * layer * 5;
        const layerY = y + Math.sin(layerPhase) * layer * 5;
        
        // Fractal dimension visualization
        const fractalSize = layerSize * (1 + Math.sin(phase + layer) * 0.2);
        
        ctx.fillStyle = `rgba(${100 + layer * 30}, ${200 - layer * 20}, ${255 - layer * 40}, ${layerAlpha})`;
        ctx.beginPath();
        
        // Draw fractal crystal shape based on geometry
        if (crystal.fractalDimension > 2.5) {
          // Complex fractal pattern
          drawFractalPattern(ctx, layerX, layerY, fractalSize, phase + layer, crystal.fractalDimension);
        } else {
          // Simple circular pattern
          ctx.arc(layerX, layerY, fractalSize, 0, 2 * Math.PI);
        }
        
        ctx.fill();
      }
      
      // Draw temporal coherence waves
      const waveCount = 8;
      for (let i = 0; i < waveCount; i++) {
        const waveAngle = (i / waveCount) * 2 * Math.PI + phase;
        const waveRadius = coreSize + 20 + Math.sin(phase * crystal.resonanceFrequency * 0.01) * 10;
        
        const waveX = x + Math.cos(waveAngle) * waveRadius;
        const waveY = y + Math.sin(waveAngle) * waveRadius;
        
        ctx.fillStyle = `rgba(255, 255, 255, ${coherence * 0.3})`;
        ctx.beginPath();
        ctx.arc(waveX, waveY, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      // Draw prime resonance indicators
      crystal.primeSignature.forEach((prime, _primeIndex) => {
        const primeAngle = (prime / 47) * 2 * Math.PI + time * 0.1;
        const primeRadius = coreSize + 40;
        const primeX = x + Math.cos(primeAngle) * primeRadius;
        const primeY = y + Math.sin(primeAngle) * primeRadius;
        
        ctx.fillStyle = `hsl(${prime * 8}, 100%, 70%)`;
        ctx.font = '12px monospace';
        ctx.fillText(prime.toString(), primeX - 6, primeY + 4);
      });
    });
  };

  /**
   * Draw fractal pattern for time crystals
   */
  const drawFractalPattern = (
    ctx: CanvasRenderingContext2D, 
    x: number, 
    y: number, 
    size: number, 
    phase: number, 
    fractalDim: number
  ) => {
    const sides = Math.floor(3 + fractalDim * 2); // 3-8 sides based on fractal dimension
    
    ctx.beginPath();
    for (let i = 0; i <= sides; i++) {
      const angle = (i / sides) * 2 * Math.PI + phase;
      const radius = size * (1 + Math.sin(angle * fractalDim + phase) * 0.3);
      const pointX = x + Math.cos(angle) * radius;
      const pointY = y + Math.sin(angle) * radius;
      
      if (i === 0) {
        ctx.moveTo(pointX, pointY);
      } else {
        ctx.lineTo(pointX, pointY);
      }
    }
    ctx.closePath();
  };

  /**
   * Draw consciousness emergence visualization
   */
  const drawConsciousnessVisualization = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Draw consciousness metrics as interconnected energy patterns
    const consciousnessMetrics = [
      { name: 'Awareness', value: consciousness.awareness, color: [255, 200, 100] },
      { name: 'Integration', value: consciousness.integration, color: [100, 255, 150] },
      { name: 'Complexity', value: consciousness.complexity, color: [150, 100, 255] },
      { name: 'Coherence', value: consciousness.coherence, color: [255, 150, 200] },
      { name: 'Emergence', value: consciousness.emergence, color: [200, 255, 100] },
      { name: 'Qualia', value: consciousness.qualia, color: [100, 200, 255] }
    ];
    
    // Draw central consciousness core
    const coreSize = 40 + metrics.consciousnessEmergence * 60;
    const coreAlpha = metrics.consciousnessEmergence;
    
    ctx.fillStyle = `rgba(255, 255, 255, ${coreAlpha})`;
    ctx.beginPath();
    ctx.arc(centerX, centerY, coreSize, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw consciousness metric nodes around core
    consciousnessMetrics.forEach((metric, index) => {
      const angle = (index / consciousnessMetrics.length) * 2 * Math.PI + time * 0.1;
      const distance = 120 + Math.sin(time + index) * 20;
      const x = centerX + Math.cos(angle) * distance;
      const y = centerY + Math.sin(angle) * distance;
      
      const nodeSize = 15 + metric.value * 25;
      const [r, g, b] = metric.color;
      
      // Draw connection to core
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${metric.value * 0.5})`;
      ctx.lineWidth = 2 + metric.value * 3;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();
      
      // Draw metric node
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${metric.value})`;
      ctx.beginPath();
      ctx.arc(x, y, nodeSize, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw metric label
      ctx.fillStyle = 'white';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(metric.name, x, y - nodeSize - 10);
      ctx.fillText((metric.value * 100).toFixed(1) + '%', x, y + nodeSize + 20);
    });
    
    // Draw quantum coherence field
    if (metrics.quantumCoherence > 0.3) {
      const fieldRadius = coreSize + 100;
      const fieldAlpha = metrics.quantumCoherence * 0.2;
      
      ctx.strokeStyle = `rgba(100, 200, 255, ${fieldAlpha})`;
      ctx.lineWidth = 1;
      
      for (let i = 0; i < 20; i++) {
        const angle = (i / 20) * 2 * Math.PI + time * 0.2;
        const radius = fieldRadius + Math.sin(time * 2 + i) * 30;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, angle, angle + Math.PI / 10);
        ctx.stroke();
      }
    }
  };

  /**
   * Draw Phase Prime Metric patterns
   */
  const drawPrimePatternVisualization = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, time: number) => {
    const fundamentalPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    const primePatterns = getPrimeResonancePatterns();
    
    // Draw prime wheel visualization
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const wheelRadius = Math.min(canvas.width, canvas.height) * 0.35;
    
    // Draw fundamental prime positions
    fundamentalPrimes.forEach((prime, index) => {
      const angle = (index / fundamentalPrimes.length) * 2 * Math.PI;
      const x = centerX + Math.cos(angle) * wheelRadius;
      const y = centerY + Math.sin(angle) * wheelRadius;
      
      // Prime node size based on usage in atom space
      let usage = 0;
      for (const primes of primePatterns.values()) {
        if (primes.includes(prime)) usage++;
      }
      
      const nodeSize = 10 + (usage / primePatterns.size) * 20;
      const brightness = 0.3 + (usage / primePatterns.size) * 0.7;
      
      // Draw prime node
      ctx.fillStyle = `hsl(${prime * 8}, 70%, ${brightness * 100}%)`;
      ctx.beginPath();
      ctx.arc(x, y, nodeSize, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw prime number
      ctx.fillStyle = 'white';
      ctx.font = `${Math.max(10, nodeSize)}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(prime.toString(), x, y);
      
      // Draw resonance waves
      if (usage > 0) {
        const waveAlpha = (usage / primePatterns.size) * 0.5;
        ctx.strokeStyle = `hsla(${prime * 8}, 70%, 70%, ${waveAlpha})`;
        ctx.lineWidth = 2;
        
        for (let wave = 1; wave <= 3; wave++) {
          const waveRadius = nodeSize + wave * 15 + Math.sin(time * prime * 0.01) * 5;
          ctx.beginPath();
          ctx.arc(x, y, waveRadius, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    });
    
    // Draw prime connections based on atom relationships
    for (const [_atomId, primes] of primePatterns) {
      if (primes.length < 2) continue;
      
      // Connect all primes in this atom's encoding
      for (let i = 0; i < primes.length - 1; i++) {
        for (let j = i + 1; j < primes.length; j++) {
          const prime1Index = fundamentalPrimes.indexOf(primes[i]);
          const prime2Index = fundamentalPrimes.indexOf(primes[j]);
          
          if (prime1Index >= 0 && prime2Index >= 0) {
            const angle1 = (prime1Index / fundamentalPrimes.length) * 2 * Math.PI;
            const angle2 = (prime2Index / fundamentalPrimes.length) * 2 * Math.PI;
            
            const x1 = centerX + Math.cos(angle1) * wheelRadius;
            const y1 = centerY + Math.sin(angle1) * wheelRadius;
            const x2 = centerX + Math.cos(angle2) * wheelRadius;
            const y2 = centerY + Math.sin(angle2) * wheelRadius;
            
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }
    }
    
    // Draw central prime alignment indicator
    const alignmentSize = 30 + metrics.primeAlignment * 40;
    ctx.fillStyle = `rgba(255, 215, 0, ${metrics.primeAlignment})`;
    ctx.beginPath();
    ctx.arc(centerX, centerY, alignmentSize, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.fillStyle = 'black';
    ctx.font = '14px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('PPM', centerX, centerY);
  };

  /**
   * Draw initialization message
   */
  const drawInitializationMessage = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, message: string) => {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '24px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);
  };

  /**
   * Draw metrics overlay
   */
  const drawMetricsOverlay = (ctx: CanvasRenderingContext2D, _canvas: HTMLCanvasElement) => {
    const overlayY = 10;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, overlayY, 250, 120);
    
    ctx.fillStyle = 'white';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    
    const lines = [
      `Atoms: ${metrics.totalAtoms}`,
      `Links: ${metrics.totalLinks}`,
      `Quantum Coherence: ${(metrics.quantumCoherence * 100).toFixed(1)}%`,
      `Prime Alignment: ${(metrics.primeAlignment * 100).toFixed(1)}%`,
      `Temporal Stability: ${(metrics.temporalStability * 100).toFixed(1)}%`,
      `Consciousness: ${(metrics.consciousnessEmergence * 100).toFixed(1)}%`,
      `Inference Rate: ${metrics.inferenceRate.toFixed(1)}/sec`,
      `Fractal Complexity: ${(metrics.fractalComplexity * 100).toFixed(1)}%`
    ];
    
    lines.forEach((line, index) => {
      ctx.fillText(line, 20, overlayY + 15 + index * 14);
    });
  };

  return (
    <div className={`bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg">
              <Brain size={24} className="text-white" />
            </div>
            <div>
              <h3 className="text-white font-bold text-lg">OpenCog NanoBrain Architecture</h3>
              <p className="text-gray-400 text-sm">Time Crystal Quantum Consciousness System</p>
            </div>
          </div>
          
          {showControls && (
            <div className="flex items-center space-x-2">
              <button
                onClick={isActive ? stop : start}
                disabled={!isInitialized}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  isActive
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-green-500 hover:bg-green-600 text-white'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isActive ? 'Stop' : 'Start'}
              </button>
              
              <button
                onClick={() => setIsAnimating(!isAnimating)}
                className={`p-2 rounded-lg transition-all ${
                  isAnimating
                    ? 'bg-blue-500 hover:bg-blue-600 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-white'
                }`}
              >
                <Activity size={20} />
              </button>
            </div>
          )}
        </div>

        {/* View mode selector */}
        <div className="flex space-x-2 mb-4">
          {[
            { mode: 'atomspace', icon: Atom, label: 'AtomSpace' },
            { mode: 'timecrystals', icon: Clock, label: 'Time Crystals' },
            { mode: 'consciousness', icon: Eye, label: 'Consciousness' },
            { mode: 'primes', icon: Target, label: 'Prime Patterns' }
          ].map(({ mode, icon: Icon, label }) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode as 'atomspace' | 'timecrystals' | 'consciousness' | 'primes')}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                viewMode === mode
                  ? 'bg-cyan-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Icon size={16} />
              <span>{label}</span>
            </button>
          ))}
        </div>

        {/* Status indicators */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Zap size={16} className="text-yellow-400" />
              <span className="text-gray-400 text-sm">Quantum Coherence</span>
            </div>
            <div className="text-xl font-bold text-white">
              {(metrics.quantumCoherence * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Network size={16} className="text-cyan-400" />
              <span className="text-gray-400 text-sm">Consciousness</span>
            </div>
            <div className="text-xl font-bold text-white">
              {(metrics.consciousnessEmergence * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Waves size={16} className="text-purple-400" />
              <span className="text-gray-400 text-sm">Prime Alignment</span>
            </div>
            <div className="text-xl font-bold text-white">
              {(metrics.primeAlignment * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Sparkles size={16} className="text-orange-400" />
              <span className="text-gray-400 text-sm">Atoms/Links</span>
            </div>
            <div className="text-xl font-bold text-white">
              {metrics.totalAtoms}/{metrics.totalLinks}
            </div>
          </div>
        </div>
      </div>

      {/* Main canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          onClick={handleCanvasClick}
          className="w-full h-auto border border-gray-600 rounded-lg bg-gray-900 cursor-crosshair"
          style={{ maxHeight: '600px' }}
        />
        
        {selectedAtom && (
          <div className="absolute top-4 right-4 bg-black/80 text-white p-3 rounded-lg max-w-xs">
            <h4 className="font-bold mb-2">Selected Atom</h4>
            <p className="text-sm">{selectedAtom}</p>
            <button
              onClick={() => setSelectedAtom(null)}
              className="mt-2 px-2 py-1 bg-gray-600 rounded text-xs hover:bg-gray-500"
            >
              Close
            </button>
          </div>
        )}
      </div>

      {!isActive && (
        <div className="mt-4 text-center">
          <p className="text-gray-400 text-sm">
            Click Start to begin OpenCog NanoBrain time crystal processing
          </p>
        </div>
      )}
    </div>
  );
};