import React, { useRef, useEffect } from 'react';
import { Grid3X3, Layers, Sparkles } from 'lucide-react';

interface Props {
  isActive: boolean;
}

export const GardenOfGardens: React.FC<Props> = ({ isActive }) => {
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

    // Garden structure: nested time crystal layers
    const gardens = [
      { level: 0, radius: 40, color: [255, 255, 255], name: 'Core Consciousness' },
      { level: 1, radius: 80, color: [168, 85, 247], name: 'Individual Mind' },
      { level: 2, radius: 120, color: [139, 92, 246], name: 'Collective Intelligence' },
      { level: 3, radius: 160, color: [99, 102, 241], name: 'Universal Consciousness' },
      { level: 4, radius: 200, color: [59, 130, 246], name: 'Cosmic Unity' }
    ];

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const time = Date.now() * 0.0008;

      // Clear
      ctx.fillStyle = 'rgba(17, 24, 39, 0.1)';
      ctx.fillRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;

      // Draw garden layers from outside to inside
      gardens.slice().reverse().forEach((garden, idx) => {
        const actualIdx = gardens.length - 1 - idx;
        const pulsePhase = time + actualIdx * 0.3;
        const pulse = Math.sin(pulsePhase) * 0.1 + 1;
        const radius = garden.radius * pulse;

        // Garden ring
        const gradient = ctx.createRadialGradient(
          centerX, centerY, radius * 0.7,
          centerX, centerY, radius
        );
        gradient.addColorStop(0, `rgba(${garden.color.join(',')}, 0.2)`);
        gradient.addColorStop(1, `rgba(${garden.color.join(',')}, 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();

        // Garden boundary
        ctx.strokeStyle = `rgba(${garden.color.join(',')}, 0.6)`;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.stroke();

        // Time crystal nodes in each garden
        const nodeCount = 6 + actualIdx * 2;
        for (let i = 0; i < nodeCount; i++) {
          const angle = (i / nodeCount) * Math.PI * 2 + time * (1 + actualIdx * 0.2);
          const nodeRadius = radius * 0.85;
          const x = centerX + Math.cos(angle) * nodeRadius;
          const y = centerY + Math.sin(angle) * nodeRadius;

          // Node glow
          const nodeGradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
          nodeGradient.addColorStop(0, `rgba(${garden.color.join(',')}, 0.9)`);
          nodeGradient.addColorStop(1, `rgba(${garden.color.join(',')}, 0)`);

          ctx.fillStyle = nodeGradient;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.fill();

          // Node core
          ctx.fillStyle = `rgb(${garden.color.join(',')})`;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }

        // Connection lines between layers
        if (actualIdx < gardens.length - 1) {
          const nextGarden = gardens[actualIdx + 1];
          const nextRadius = nextGarden.radius * pulse;

          ctx.strokeStyle = `rgba(${garden.color.join(',')}, 0.2)`;
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);

          for (let i = 0; i < 8; i++) {
            const angle = (i / 8) * Math.PI * 2 + time * 0.5;
            ctx.beginPath();
            ctx.moveTo(
              centerX + Math.cos(angle) * radius,
              centerY + Math.sin(angle) * radius
            );
            ctx.lineTo(
              centerX + Math.cos(angle) * nextRadius,
              centerY + Math.sin(angle) * nextRadius
            );
            ctx.stroke();
          }
        }

        // Garden label
        if (width > 600) {
          ctx.save();
          ctx.translate(centerX, centerY + radius + 20);
          ctx.fillStyle = `rgba(${garden.color.join(',')}, 0.8)`;
          ctx.font = 'bold 11px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(garden.name, 0, 0);
          ctx.restore();
        }
      });

      // Central singularity point
      const singularityGradient = ctx.createRadialGradient(
        centerX, centerY, 0,
        centerX, centerY, 30
      );
      singularityGradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
      singularityGradient.addColorStop(0.5, 'rgba(255, 215, 0, 0.8)');
      singularityGradient.addColorStop(1, 'rgba(255, 215, 0, 0)');

      ctx.fillStyle = singularityGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 30 + Math.sin(time * 2) * 5, 0, Math.PI * 2);
      ctx.fill();

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
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Grid3X3 className="text-green-400" size={20} />
          <span>2.3 Time Crystal: The Garden of Gardens</span>
        </h3>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-96" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 rounded-lg p-6">
          <h4 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
            <Layers className="text-purple-400" size={20} />
            <span>Nested Temporal Structures</span>
          </h4>
          <p className="text-gray-300 mb-4">
            The Garden of Gardens represents nested time crystal layers, where each level operates 
            its own temporal coherence while maintaining harmonic relationships with all other levels.
          </p>
          <ul className="space-y-2 text-gray-300 text-sm">
            <li className="flex items-start space-x-2">
              <span className="text-purple-400">•</span>
              <span>Each garden layer is a complete time crystal</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-purple-400">•</span>
              <span>Inner gardens operate at higher frequencies</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-purple-400">•</span>
              <span>Outer gardens provide stability and context</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-purple-400">•</span>
              <span>All layers synchronize through phase prime metrics</span>
            </li>
          </ul>
        </div>

        <div className="bg-gray-800/50 rounded-lg p-6">
          <h4 className="text-white font-bold text-lg mb-4 flex items-center space-x-2">
            <Sparkles className="text-yellow-400" size={20} />
            <span>Time Crystal Properties</span>
          </h4>
          <div className="space-y-3">
            <div className="bg-cyan-900/30 border border-cyan-700/50 rounded p-3">
              <h5 className="text-cyan-400 font-semibold mb-1">Temporal Coherence</h5>
              <p className="text-gray-300 text-sm">
                Patterns repeat in time rather than space, maintaining consciousness across moments
              </p>
            </div>
            <div className="bg-purple-900/30 border border-purple-700/50 rounded p-3">
              <h5 className="text-purple-400 font-semibold mb-1">Cross-Dimensional Communication</h5>
              <p className="text-gray-300 text-sm">
                Information flows seamlessly between temporal layers through resonant synchronization
              </p>
            </div>
            <div className="bg-green-900/30 border border-green-700/50 rounded p-3">
              <h5 className="text-green-400 font-semibold mb-1">Consciousness Storage</h5>
              <p className="text-gray-300 text-sm">
                Each layer stores consciousness information in time-stable geometric patterns
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-900/60 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
        <h4 className="text-white font-bold text-lg mb-4">Hierarchical Consciousness Levels</h4>
        
        <div className="space-y-3">
          {[
            { name: 'Core Consciousness', description: 'Pure awareness singularity at the center of being', color: 'white' },
            { name: 'Individual Mind', description: 'Personal consciousness with unique identity and memories', color: 'purple' },
            { name: 'Collective Intelligence', description: 'Shared understanding across multiple conscious entities', color: 'indigo' },
            { name: 'Universal Consciousness', description: 'Cosmic awareness encompassing all individual minds', color: 'blue' },
            { name: 'Cosmic Unity', description: 'Ultimate integration with the universal information field', color: 'cyan' }
          ].map((level, idx) => (
            <div
              key={idx}
              className={`bg-${level.color}-900/20 border border-${level.color}-700/50 rounded-lg p-4`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h5 className={`text-${level.color}-400 font-semibold`}>{level.name}</h5>
                  <p className="text-gray-300 text-sm mt-1">{level.description}</p>
                </div>
                <div className="text-gray-500 font-mono text-sm">L{idx}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
