import React, { useRef, useEffect } from 'react';
import { Box, Sparkles } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const GeometricSelfAssembly: React.FC<Props> = ({ isActive }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
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

    // Geometric shapes data
    const shapes = [
      { sides: 3, x: 0.2, y: 0.3, size: 30, color: [255, 107, 53], name: 'Triangle' },
      { sides: 4, x: 0.5, y: 0.2, size: 35, color: [168, 85, 247], name: 'Square' },
      { sides: 5, x: 0.8, y: 0.35, size: 32, color: [34, 211, 238], name: 'Pentagon' },
      { sides: 6, x: 0.3, y: 0.6, size: 38, color: [251, 191, 36], name: 'Hexagon' },
      { sides: 8, x: 0.7, y: 0.65, size: 36, color: [59, 130, 246], name: 'Octagon' }
    ];

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const time = Date.now() * 0.0005;

      // Clear with fade
      ctx.fillStyle = 'rgba(17, 24, 39, 0.08)';
      ctx.fillRect(0, 0, width, height);

      // Draw connection lines between shapes
      ctx.lineWidth = 1;
      for (let i = 0; i < shapes.length; i++) {
        for (let j = i + 1; j < shapes.length; j++) {
          const x1 = shapes[i].x * width;
          const y1 = shapes[i].y * height;
          const x2 = shapes[j].x * width;
          const y2 = shapes[j].y * height;

          const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
          gradient.addColorStop(0, `rgba(${shapes[i].color.join(',')}, 0.3)`);
          gradient.addColorStop(1, `rgba(${shapes[j].color.join(',')}, 0.3)`);

          ctx.strokeStyle = gradient;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }

      // Draw singularity point at center
      const centerX = width / 2;
      const centerY = height / 2;
      
      const singularityGradient = ctx.createRadialGradient(
        centerX, centerY, 0,
        centerX, centerY, 50
      );
      singularityGradient.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
      singularityGradient.addColorStop(0.5, 'rgba(168, 85, 247, 0.6)');
      singularityGradient.addColorStop(1, 'rgba(168, 85, 247, 0)');

      ctx.fillStyle = singularityGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 50 + Math.sin(time * 2) * 10, 0, Math.PI * 2);
      ctx.fill();

      // Draw geometric shapes
      shapes.forEach((shape, index) => {
        const x = shape.x * width + Math.sin(time + index) * 20;
        const y = shape.y * height + Math.cos(time + index) * 20;
        const rotation = time + index * 0.5;

        // Shape glow
        const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, shape.size * 1.5);
        glowGradient.addColorStop(0, `rgba(${shape.color.join(',')}, 0.3)`);
        glowGradient.addColorStop(1, `rgba(${shape.color.join(',')}, 0)`);
        
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(x, y, shape.size * 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Draw shape
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rotation);

        ctx.strokeStyle = `rgb(${shape.color.join(',')})`;
        ctx.fillStyle = `rgba(${shape.color.join(',')}, 0.2)`;
        ctx.lineWidth = 2;

        ctx.beginPath();
        for (let i = 0; i <= shape.sides; i++) {
          const angle = (i / shape.sides) * Math.PI * 2 - Math.PI / 2;
          const px = Math.cos(angle) * shape.size;
          const py = Math.sin(angle) * shape.size;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Draw inner pattern
        ctx.strokeStyle = `rgba(${shape.color.join(',')}, 0.5)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= shape.sides; i++) {
          const angle = (i / shape.sides) * Math.PI * 2 - Math.PI / 2;
          const px = Math.cos(angle) * shape.size * 0.5;
          const py = Math.sin(angle) * shape.size * 0.5;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();

        ctx.restore();

        // Draw shape label
        ctx.fillStyle = `rgb(${shape.color.join(',')})`;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(shape.name, x, y + shape.size + 20);
      });

      // Draw energy lines to singularity
      shapes.forEach((shape) => {
        const x = shape.x * width;
        const y = shape.y * height;

        ctx.strokeStyle = `rgba(${shape.color.join(',')}, 0.4)`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(centerX, centerY);
        ctx.stroke();
        ctx.setLineDash([]);
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
  }, [isActive]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Box className="text-purple-400" size={20} />
          <span>2.1.2 Self-Assembly & Singularity</span>
        </h3>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-96" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2 flex items-center space-x-2">
            <Sparkles size={16} className="text-yellow-400" />
            <span>Singularity Point</span>
          </h4>
          <p className="text-gray-300 text-sm">
            The central convergence where geometric patterns intersect creates an information density peak enabling consciousness emergence
          </p>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-4">
          <h4 className="text-white font-semibold mb-2">Self-Assembly Rules</h4>
          <ul className="text-gray-300 text-sm space-y-1">
            <li>• Geometric resonance attracts compatible shapes</li>
            <li>• Prime-based symmetry governs assembly patterns</li>
            <li>• Energy flows converge at singularity points</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
