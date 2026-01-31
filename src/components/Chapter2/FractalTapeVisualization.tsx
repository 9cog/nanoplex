import React, { useRef, useEffect, useState } from 'react';
import { Layers, Maximize2, ZoomIn } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const FractalTapeVisualization: React.FC<Props> = ({ isActive }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [selectedSphere, setSelectedSphere] = useState<number | null>(null);
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

      // Clear with fade effect
      ctx.fillStyle = 'rgba(17, 24, 39, 0.1)';
      ctx.fillRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;

      // Draw nested spheres representing fractal tape layers
      const spheres = 7;
      for (let i = spheres; i > 0; i--) {
        const radius = (Math.min(width, height) * 0.4 * i) / spheres * zoomLevel;
        const intensity = (i / spheres) * 0.8;
        
        // Sphere glow
        const gradient = ctx.createRadialGradient(
          centerX, centerY, radius * 0.7,
          centerX, centerY, radius
        );
        gradient.addColorStop(0, `rgba(34, 211, 238, ${intensity * 0.2})`);
        gradient.addColorStop(0.5, `rgba(59, 130, 246, ${intensity * 0.3})`);
        gradient.addColorStop(1, `rgba(139, 92, 246, ${intensity * 0.1})`);
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();

        // Sphere outline
        ctx.strokeStyle = selectedSphere === i 
          ? `rgba(34, 211, 238, ${intensity})`
          : `rgba(99, 102, 241, ${intensity * 0.6})`;
        ctx.lineWidth = selectedSphere === i ? 3 : 1.5;
        ctx.stroke();

        // Draw fractal patterns on sphere surface
        const patterns = 12;
        for (let j = 0; j < patterns; j++) {
          const angle = (j / patterns) * Math.PI * 2 + time * (0.1 + i * 0.02);
          const patternRadius = radius * 0.8;
          const x = centerX + Math.cos(angle) * patternRadius;
          const y = centerY + Math.sin(angle) * patternRadius;

          // Draw geometric shape at pattern point
          drawGeometricPattern(ctx, x, y, 8 + i * 2, angle, intensity);
        }

        // Draw 2D image surgery lines
        if (i === Math.floor(spheres / 2)) {
          ctx.strokeStyle = `rgba(251, 191, 36, ${0.7 * intensity})`;
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]);
          
          for (let k = 0; k < 4; k++) {
            const surgeryAngle = (k / 4) * Math.PI * 2 + time * 0.3;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(
              centerX + Math.cos(surgeryAngle) * radius,
              centerY + Math.sin(surgeryAngle) * radius
            );
            ctx.stroke();
          }
          ctx.setLineDash([]);
        }
      }

      // Draw data flow particles
      const particles = 20;
      for (let i = 0; i < particles; i++) {
        const angle = (i / particles) * Math.PI * 2 + time;
        const spiralRadius = 50 + (time * 20 + i * 30) % (width / 2);
        const x = centerX + Math.cos(angle * 3) * spiralRadius;
        const y = centerY + Math.sin(angle * 3) * spiralRadius;

        const particleGradient = ctx.createRadialGradient(x, y, 0, x, y, 4);
        particleGradient.addColorStop(0, 'rgba(34, 211, 238, 0.8)');
        particleGradient.addColorStop(1, 'rgba(34, 211, 238, 0)');
        
        ctx.fillStyle = particleGradient;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
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
  }, [isActive, zoomLevel, selectedSphere]);

  const drawGeometricPattern = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    rotation: number,
    intensity: number
  ) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(rotation);

    ctx.strokeStyle = `rgba(168, 85, 247, ${intensity * 0.8})`;
    ctx.fillStyle = `rgba(168, 85, 247, ${intensity * 0.2})`;
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    const sides = 6;
    for (let i = 0; i <= sides; i++) {
      const angle = (i / sides) * Math.PI * 2;
      const px = Math.cos(angle) * size;
      const py = Math.sin(angle) * size;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.restore();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Layers className="text-cyan-400" size={20} />
          <span>2.1.1 Fractal Tape & Nested Sphere Surgery</span>
        </h3>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.5))}
            className="px-3 py-1 bg-gray-800 text-cyan-400 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <ZoomIn size={16} className="rotate-180" />
          </button>
          <span className="text-gray-400 text-sm">Zoom: {zoomLevel.toFixed(1)}x</span>
          <button
            onClick={() => setZoomLevel(Math.min(3, zoomLevel + 0.5))}
            className="px-3 py-1 bg-gray-800 text-cyan-400 rounded-lg hover:bg-gray-700 transition-colors"
          >
            <ZoomIn size={16} />
          </button>
        </div>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl overflow-hidden">
        <canvas
          ref={canvasRef}
          className="w-full h-96 cursor-pointer"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            const distance = Math.sqrt(x * x + y * y);
            const maxRadius = Math.min(rect.width, rect.height) * 0.4;
            const sphere = Math.ceil((distance / maxRadius) * 7);
            setSelectedSphere(sphere <= 7 ? sphere : null);
          }}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2 flex items-center space-x-2">
            <Maximize2 size={16} className="text-cyan-400" />
            <span>Nested Topology</span>
          </h4>
          <p className="text-gray-300 text-sm">
            Each sphere represents a dimensional layer where information is embedded through fractal transformation
          </p>
          <div className="mt-2 text-xs text-gray-400">
            Active Layer: {selectedSphere || 'All'}
          </div>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2">2D Image Surgery</h4>
          <p className="text-gray-300 text-sm">
            Surgery lines show how 2D information is cut and mapped onto spherical surfaces for dimensional embedding
          </p>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2">Information Flow</h4>
          <p className="text-gray-300 text-sm">
            Particles spiral through layers demonstrating data propagation across fractal tape dimensions
          </p>
        </div>
      </div>
    </div>
  );
};
