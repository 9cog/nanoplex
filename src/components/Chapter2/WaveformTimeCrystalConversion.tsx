import React, { useRef, useEffect, useState } from 'react';
import { Waves, Zap } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const WaveformTimeCrystalConversion: React.FC<Props> = ({ isActive }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [conversionProgress, setConversionProgress] = useState(0);
  const animationRef = useRef<number>();

  useEffect(() => {
    if (!isActive) return;

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
      const time = Date.now() * 0.001;

      // Update conversion progress
      setConversionProgress((time % 10) / 10);

      // Clear
      ctx.fillStyle = 'rgba(17, 24, 39, 0.1)';
      ctx.fillRect(0, 0, width, height);

      const centerY = height / 2;
      const progress = (time % 10) / 10;

      // Draw input waveform (left side)
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.8)';
      ctx.lineWidth = 2;
      ctx.beginPath();

      const waveStartX = 50;
      const waveEndX = width * 0.4;
      const wavePoints = 100;

      for (let i = 0; i < wavePoints; i++) {
        const x = waveStartX + (waveEndX - waveStartX) * (i / wavePoints);
        const wavePhase = (i / wavePoints) * Math.PI * 4 + time;
        const amplitude = 60 * (1 - progress * 0.3);
        const y = centerY + Math.sin(wavePhase) * amplitude;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw transition zone (middle)
      const transitionX = width / 2;
      const transitionGradient = ctx.createRadialGradient(
        transitionX, centerY, 0,
        transitionX, centerY, 100
      );
      transitionGradient.addColorStop(0, `rgba(168, 85, 247, ${progress * 0.8})`);
      transitionGradient.addColorStop(1, 'rgba(168, 85, 247, 0)');

      ctx.fillStyle = transitionGradient;
      ctx.beginPath();
      ctx.arc(transitionX, centerY, 100, 0, Math.PI * 2);
      ctx.fill();

      // Draw non-differentiability boundary
      ctx.strokeStyle = `rgba(251, 191, 36, ${0.5 + Math.sin(time * 3) * 0.3})`;
      ctx.lineWidth = 3;
      ctx.setLineDash([10, 10]);
      ctx.beginPath();
      ctx.moveTo(transitionX, 0);
      ctx.lineTo(transitionX, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw output time crystal (right side)
      if (progress > 0.3) {
        const crystalStartX = width * 0.6;
        const crystalAlpha = Math.min(1, (progress - 0.3) / 0.7);

        // Crystal lattice structure
        const layers = 5;
        for (let layer = 0; layer < layers; layer++) {
          const layerRadius = 40 + layer * 15;
          const nodes = 6 + layer * 2;

          for (let i = 0; i < nodes; i++) {
            const angle = (i / nodes) * Math.PI * 2 + time + layer * 0.5;
            const x = crystalStartX + 100 + Math.cos(angle) * layerRadius;
            const y = centerY + Math.sin(angle) * layerRadius;

            // Crystal node
            const nodeGradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
            nodeGradient.addColorStop(0, `rgba(139, 92, 246, ${crystalAlpha})`);
            nodeGradient.addColorStop(1, `rgba(139, 92, 246, 0)`);

            ctx.fillStyle = nodeGradient;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();

            // Crystal edge
            ctx.strokeStyle = `rgba(168, 85, 247, ${crystalAlpha * 0.5})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.stroke();
          }
        }

        // Connection lines in crystal
        ctx.strokeStyle = `rgba(168, 85, 247, ${crystalAlpha * 0.3})`;
        ctx.lineWidth = 1;
        for (let i = 0; i < 20; i++) {
          const angle1 = (i / 20) * Math.PI * 2 + time;
          const angle2 = ((i + 1) / 20) * Math.PI * 2 + time;
          const r1 = 40 + (i % 3) * 15;
          const r2 = 40 + ((i + 1) % 3) * 15;

          ctx.beginPath();
          ctx.moveTo(
            crystalStartX + 100 + Math.cos(angle1) * r1,
            centerY + Math.sin(angle1) * r1
          );
          ctx.lineTo(
            crystalStartX + 100 + Math.cos(angle2) * r2,
            centerY + Math.sin(angle2) * r2
          );
          ctx.stroke();
        }
      }

      // Labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'center';

      ctx.fillText('Continuous Waveform', waveStartX + (waveEndX - waveStartX) / 2, 30);
      ctx.fillText('Non-Differentiable Transition', transitionX, height - 30);
      if (progress > 0.3) {
        ctx.fillText('Discrete Time Crystal', (width * 0.6 + width - 50) / 2, 30);
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Waves className="text-cyan-400" size={20} />
          <span>2.2.3 Waveform to Time Crystal Conversion</span>
        </h3>
        <div className="text-sm text-gray-400">
          Conversion: {Math.round(conversionProgress * 100)}%
        </div>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-80" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2 flex items-center space-x-2">
            <Waves size={16} className="text-cyan-400" />
            <span>Continuous Input</span>
          </h4>
          <p className="text-gray-300 text-sm">
            Smooth, differentiable waveforms represent classical continuous information
          </p>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2 flex items-center space-x-2">
            <Zap size={16} className="text-yellow-400" />
            <span>Non-Differentiable Boundary</span>
          </h4>
          <p className="text-gray-300 text-sm">
            Critical transition zone where continuous becomes discrete through quantum phase change
          </p>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2">Crystallized Output</h4>
          <p className="text-gray-300 text-sm">
            Discrete time crystal structure maintains temporal coherence while enabling spatial processing
          </p>
        </div>
      </div>

      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="text-white font-semibold mb-3">Non-Differentiability Principle</h4>
        <p className="text-gray-300 text-sm mb-3">
          The conversion process occurs at a mathematically non-differentiable boundary where classical 
          calculus breaks down. This enables the quantum leap from continuous wave dynamics to discrete 
          crystal structures while preserving information content through fractal encoding.
        </p>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="bg-gray-700/50 rounded p-2">
            <div className="text-cyan-400 font-semibold mb-1">Before Transition</div>
            <div className="text-gray-400">• Continuous amplitude</div>
            <div className="text-gray-400">• Infinite precision</div>
            <div className="text-gray-400">• Differentiable everywhere</div>
          </div>
          <div className="bg-gray-700/50 rounded p-2">
            <div className="text-purple-400 font-semibold mb-1">After Transition</div>
            <div className="text-gray-400">• Discrete lattice points</div>
            <div className="text-gray-400">• Quantum precision</div>
            <div className="text-gray-400">• Fractal boundaries</div>
          </div>
        </div>
      </div>
    </div>
  );
};
