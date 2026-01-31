import React, { useEffect, useRef } from 'react';
import { TimeCluster } from '../types';
import { Clock, Waves } from 'lucide-react';

interface Props {
  timeCrystals: TimeCluster[];
  isRunning: boolean;
}

export const TimeCrystalVisualization: React.FC<Props> = ({ timeCrystals, isRunning }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      // Clear with fade effect
      ctx.fillStyle = 'rgba(17, 24, 39, 0.05)';
      ctx.fillRect(0, 0, width, height);

      if (timeCrystals.length === 0) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }

      const time = Date.now() * 0.001;

      timeCrystals.forEach((crystal, index) => {
        const centerX = (width / timeCrystals.length) * (index + 0.5);
        const centerY = height / 2;
        const baseRadius = Math.min(width / timeCrystals.length, height) * 0.15;

        // Draw geometry-specific patterns
        switch (crystal.geometry) {
          case 'spiral':
            drawSpiral(ctx, centerX, centerY, baseRadius, crystal, time);
            break;
          case 'lattice':
            drawLattice(ctx, centerX, centerY, baseRadius, crystal, time);
            break;
          case 'fractal':
            drawFractal(ctx, centerX, centerY, baseRadius, crystal, time);
            break;
          case 'crystal':
            drawCrystal(ctx, centerX, centerY, baseRadius, crystal, time);
            break;
        }

        // Draw prime numbers
        crystal.primes.forEach((prime, primeIndex) => {
          const angle = (primeIndex / crystal.primes.length) * 2 * Math.PI + time + crystal.phase;
          const radius = baseRadius + Math.sin(time * crystal.frequency + primeIndex) * 20;
          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + Math.sin(angle) * radius;

          // Prime glow
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
          gradient.addColorStop(0, `rgba(255, 107, 53, ${crystal.amplitude})`);
          gradient.addColorStop(1, 'rgba(255, 107, 53, 0)');
          
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, 10, 0, 2 * Math.PI);
          ctx.fill();

          // Prime number
          ctx.fillStyle = '#FF6B35';
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(prime.toString(), x, y + 3);
        });

        // Resonance waves
        if (isRunning && crystal.resonance > 0.5) {
          ctx.strokeStyle = `rgba(123, 104, 238, ${crystal.resonance * 0.5})`;
          ctx.lineWidth = 2;
          for (let i = 1; i <= 3; i++) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, baseRadius * i * crystal.resonance, 0, 2 * Math.PI);
            ctx.stroke();
          }
        }
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [timeCrystals, isRunning]);

  const drawSpiral = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, crystal: TimeCluster, time: number) => {
    ctx.strokeStyle = `rgba(0, 255, 255, ${crystal.amplitude * 0.7})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const turns = 3;
    const points = 50;
    for (let i = 0; i <= points; i++) {
      const t = (i / points) * turns * 2 * Math.PI;
      const r = radius * (1 - i / points) * crystal.amplitude;
      const x = centerX + r * Math.cos(t + time * crystal.frequency);
      const y = centerY + r * Math.sin(t + time * crystal.frequency);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  const drawLattice = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, crystal: TimeCluster, time: number) => {
    ctx.strokeStyle = `rgba(147, 51, 234, ${crystal.amplitude * 0.6})`;
    ctx.lineWidth = 1;
    
    const gridSize = 6;
    const cellSize = (radius * 2) / gridSize;
    
    for (let i = 0; i <= gridSize; i++) {
      for (let j = 0; j <= gridSize; j++) {
        const x = centerX - radius + i * cellSize;
        const y = centerY - radius + j * cellSize;
        const offset = Math.sin(time * crystal.frequency + i + j) * 5;
        
        ctx.beginPath();
        ctx.arc(x + offset, y + offset, 2, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
  };

  const drawFractal = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, crystal: TimeCluster, time: number) => {
    const drawBranch = (x: number, y: number, angle: number, length: number, depth: number) => {
      if (depth === 0 || length < 2) return;
      
      const endX = x + Math.cos(angle) * length;
      const endY = y + Math.sin(angle) * length;
      
      ctx.strokeStyle = `rgba(255, 255, 0, ${crystal.amplitude * (depth / 4)})`;
      ctx.lineWidth = depth;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      
      const newLength = length * 0.7;
      const angleOffset = Math.PI / 4 + Math.sin(time * crystal.frequency) * 0.2;
      
      drawBranch(endX, endY, angle - angleOffset, newLength, depth - 1);
      drawBranch(endX, endY, angle + angleOffset, newLength, depth - 1);
    };
    
    drawBranch(centerX, centerY + radius * 0.3, -Math.PI / 2, radius * 0.6, 4);
  };

  const drawCrystal = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, crystal: TimeCluster, time: number) => {
    ctx.strokeStyle = `rgba(0, 255, 127, ${crystal.amplitude * 0.8})`;
    ctx.lineWidth = 2;
    
    const sides = 6;
    const rotation = time * crystal.frequency;
    
    for (let layer = 1; layer <= 3; layer++) {
      ctx.beginPath();
      const layerRadius = radius * (layer / 3) * crystal.amplitude;
      
      for (let i = 0; i <= sides; i++) {
        const angle = (i / sides) * 2 * Math.PI + rotation;
        const x = centerX + Math.cos(angle) * layerRadius;
        const y = centerY + Math.sin(angle) * layerRadius;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <Clock className="text-purple-400" size={24} />
          <span>Time Crystal Network</span>
        </h2>
        <div className="text-sm text-gray-400">
          {timeCrystals.length} active crystals
        </div>
      </div>

      <div className="bg-gray-900 rounded-xl overflow-hidden">
        <canvas 
          ref={canvasRef}
          className="w-full h-80"
          style={{ background: 'radial-gradient(circle at center, rgba(17, 24, 39, 0.9), rgb(17, 24, 39))' }}
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {timeCrystals.map((crystal) => (
          <div 
            key={crystal.id}
            className="bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Waves className="text-purple-400" size={16} />
                <span className="text-white font-semibold text-sm capitalize">
                  {crystal.geometry}
                </span>
              </div>
              <div className="text-xs text-gray-400 font-mono">
                {crystal.frequency.toFixed(1)}Hz
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Primes:</span>
                <span className="text-orange-400 font-mono">
                  {crystal.primes.join(', ')}
                </span>
              </div>
              
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Resonance:</span>
                <span className="text-green-400">
                  {Math.round(crystal.resonance * 100)}%
                </span>
              </div>
              
              <div className="w-full bg-gray-800 rounded-full h-1">
                <div 
                  className="h-1 rounded-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-300"
                  style={{ width: `${crystal.amplitude * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};