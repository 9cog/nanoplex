import React, { useEffect, useRef, useState } from 'react';
import { CognitiveNode, AtomSpaceConnection } from '../types';
import { Brain, Zap, Target, Atom } from 'lucide-react';

interface Props {
  nodes: CognitiveNode[];
  connections?: AtomSpaceConnection[];
  isRunning: boolean;
}

interface RenderNode {
  id: string;
  x: number;
  y: number;
  z: number;
  radius: number;
  color: string;
  glowIntensity: number;
  velocity: { x: number; y: number; z: number };
  node: CognitiveNode;
}

interface RenderConnection {
  source: RenderNode;
  target: RenderNode;
  strength: number;
  pulse: number;
}

export const HypergraphVisualization: React.FC<Props> = ({ nodes, isRunning }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const renderNodesRef = useRef<RenderNode[]>([]);
  const mouseRef = useRef({ x: 0, y: 0, isDown: false });
  const cameraRef = useRef({ rotation: { x: 0, y: 0 }, zoom: 1, target: { x: 0, y: 0, z: 0 } });
  const [selectedNode, setSelectedNode] = useState<CognitiveNode | null>(null);
  const [hoverNode, setHoverNode] = useState<CognitiveNode | null>(null);

  // Advanced 3D projection with camera controls
  const project3D = (x: number, y: number, z: number, width: number, height: number) => {
    const camera = cameraRef.current;
    const fov = 800 * camera.zoom;
    
    // Apply camera rotation
    const cosX = Math.cos(camera.rotation.x);
    const sinX = Math.sin(camera.rotation.x);
    const cosY = Math.cos(camera.rotation.y);
    const sinY = Math.sin(camera.rotation.y);
    
    // Rotate around Y axis
    const rotatedX = x * cosY - z * sinY;
    const rotatedZ = x * sinY + z * cosY;
    
    // Rotate around X axis
    const finalY = y * cosX - rotatedZ * sinX;
    const finalZ = y * sinX + rotatedZ * cosX;
    
    const perspective = fov / (fov + finalZ);
    
    return {
      x: width / 2 + rotatedX * perspective,
      y: height / 2 + finalY * perspective,
      scale: perspective
    };
  };

  // Advanced force-directed layout with 3D physics
  const updateNodePositions = (renderNodes: RenderNode[], dt: number) => {
    const centerForce = 0.001;
    const repulsionForce = 15000;
    const attractionForce = 0.1;
    const damping = 0.95;
    const maxVelocity = 5;

    renderNodes.forEach((node, i) => {
      let fx = 0, fy = 0, fz = 0;

      // Center attraction
      const centerDist = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
      if (centerDist > 0) {
        fx -= node.x * centerForce;
        fy -= node.y * centerForce;
        fz -= node.z * centerForce;
      }

      // Node repulsion
      renderNodes.forEach((other, j) => {
        if (i !== j) {
          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const dz = node.z - other.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          
          if (dist > 0 && dist < 150) {
            const force = repulsionForce / (dist * dist);
            fx += (dx / dist) * force;
            fy += (dy / dist) * force;
            fz += (dz / dist) * force;
          }
        }
      });

      // Connection attraction
      node.node.connections.forEach(connId => {
        const targetNode = renderNodes.find(rn => rn.id === connId);
        if (targetNode) {
          const dx = targetNode.x - node.x;
          const dy = targetNode.y - node.y;
          const dz = targetNode.z - node.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          
          if (dist > 0) {
            const idealDist = 80 + (node.node.value + targetNode.node.value) / 10;
            const force = (dist - idealDist) * attractionForce;
            fx += (dx / dist) * force;
            fy += (dy / dist) * force;
            fz += (dz / dist) * force;
          }
        }
      });

      // Prime-specific forces (phase coherence)
      if (node.node.type === 'prime') {
        const phaseForce = Math.sin(node.node.phase) * 0.5;
        fx += Math.cos(node.node.phase * 2) * phaseForce;
        fy += Math.sin(node.node.phase * 3) * phaseForce;
        fz += Math.cos(node.node.phase * 1.5) * phaseForce;
      }

      // Update velocity
      node.velocity.x = Math.max(-maxVelocity, Math.min(maxVelocity, (node.velocity.x + fx * dt) * damping));
      node.velocity.y = Math.max(-maxVelocity, Math.min(maxVelocity, (node.velocity.y + fy * dt) * damping));
      node.velocity.z = Math.max(-maxVelocity, Math.min(maxVelocity, (node.velocity.z + fz * dt) * damping));

      // Update position
      node.x += node.velocity.x * dt;
      node.y += node.velocity.y * dt;
      node.z += node.velocity.z * dt;
    });
  };

  // Advanced node rendering with quantum effects
  const drawNode = (ctx: CanvasRenderingContext2D, node: RenderNode, projected: { x: number; y: number; scale: number }, time: number) => {
    const { x, y, scale } = projected;
    const radius = node.radius * scale;
    const activation = node.node.activation;
    
    if (radius < 1) return;

    // Quantum coherence visualization
    if (node.node.type === 'prime' && activation > 0.7) {
      // Quantum interference rings
      for (let i = 1; i <= 3; i++) {
        const ringRadius = radius * (1 + i * 0.3);
        const opacity = (activation * 0.3) / i;
        
        ctx.strokeStyle = `rgba(0, 255, 255, ${opacity})`;
        ctx.lineWidth = 2 / i;
        ctx.beginPath();
        ctx.arc(x, y, ringRadius + Math.sin(time * 2 + i) * 3, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    // Phase-based color modulation
    const phaseHue = (node.node.phase * 180 / Math.PI) % 360;
    const saturation = 70 + activation * 30;
    const lightness = 50 + activation * 30;

    // Node glow effect
    const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 2);
    glowGradient.addColorStop(0, `hsla(${phaseHue}, ${saturation}%, ${lightness}%, ${activation})`);
    glowGradient.addColorStop(0.5, `hsla(${phaseHue}, ${saturation}%, ${lightness}%, ${activation * 0.3})`);
    glowGradient.addColorStop(1, `hsla(${phaseHue}, ${saturation}%, ${lightness}%, 0)`);
    
    ctx.fillStyle = glowGradient;
    ctx.beginPath();
    ctx.arc(x, y, radius * 2, 0, 2 * Math.PI);
    ctx.fill();

    // Core node
    ctx.fillStyle = `hsl(${phaseHue}, ${saturation}%, ${lightness}%)`;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();

    // Fractal dimension visualization
    if (node.node.fractalDimension && node.node.fractalDimension > 1) {
      const fractalIntensity = (node.node.fractalDimension - 1) / 2;
      ctx.strokeStyle = `rgba(255, 255, 255, ${fractalIntensity})`;
      ctx.lineWidth = 1;
      
      // Draw fractal patterns
      const sides = Math.floor(3 + node.node.fractalDimension * 3);
      for (let i = 0; i < sides; i++) {
        const angle = (i / sides) * 2 * Math.PI + time;
        const innerRadius = radius * 0.7;
        const outerRadius = radius * (0.9 + fractalIntensity * 0.3);
        
        ctx.beginPath();
        ctx.moveTo(x + Math.cos(angle) * innerRadius, y + Math.sin(angle) * innerRadius);
        ctx.lineTo(x + Math.cos(angle) * outerRadius, y + Math.sin(angle) * outerRadius);
        ctx.stroke();
      }
    }

    // Prime number label
    if (node.node.type === 'prime' && radius > 10) {
      ctx.fillStyle = 'white';
      ctx.font = `bold ${Math.min(14, radius / 2)}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.node.value.toString(), x, y);
    }

    // Selection highlight
    if (selectedNode?.id === node.id) {
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(x, y, radius * 1.5, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Hover highlight
    if (hoverNode?.id === node.id) {
      ctx.strokeStyle = '#FFFF00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, radius * 1.2, 0, 2 * Math.PI);
      ctx.stroke();
    }
  };

  // Advanced connection rendering with quantum entanglement effects
  const drawConnection = (ctx: CanvasRenderingContext2D, conn: RenderConnection, time: number) => {
    const sourceProj = project3D(conn.source.x, conn.source.y, conn.source.z, ctx.canvas.width, ctx.canvas.height);
    const targetProj = project3D(conn.target.x, conn.target.y, conn.target.z, ctx.canvas.width, ctx.canvas.height);
    
    if (sourceProj.scale < 0.1 || targetProj.scale < 0.1) return;

    const alpha = Math.min(sourceProj.scale, targetProj.scale) * conn.strength;
    const lineWidth = 1 + conn.strength * 2;

    // Quantum entanglement visualization
    if (conn.strength > 0.8) {
      // Draw quantum correlation waves
      const steps = 20;
      const dx = (targetProj.x - sourceProj.x) / steps;
      const dy = (targetProj.y - sourceProj.y) / steps;
      
      ctx.strokeStyle = `rgba(255, 0, 255, ${alpha * 0.5})`;
      ctx.lineWidth = lineWidth * 0.5;
      
      for (let i = 0; i < steps; i++) {
        const x = sourceProj.x + dx * i;
        const y = sourceProj.y + dy * i;
        const waveOffset = Math.sin(time * 2 + i * 0.5) * 5;
        
        if (i === 0) {
          ctx.beginPath();
          ctx.moveTo(x, y + waveOffset);
        } else {
          ctx.lineTo(x, y + waveOffset);
        }
      }
      ctx.stroke();
    }

    // Main connection line
    const gradient = ctx.createLinearGradient(sourceProj.x, sourceProj.y, targetProj.x, targetProj.y);
    gradient.addColorStop(0, `rgba(59, 130, 246, ${alpha})`);
    gradient.addColorStop(0.5, `rgba(147, 51, 234, ${alpha})`);
    gradient.addColorStop(1, `rgba(59, 130, 246, ${alpha})`);
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(sourceProj.x, sourceProj.y);
    ctx.lineTo(targetProj.x, targetProj.y);
    ctx.stroke();

    // Information flow particles
    if (conn.strength > 0.6) {
      const particlePos = ((time * 50 + conn.pulse * 100) % 100) / 100;
      const particleX = sourceProj.x + (targetProj.x - sourceProj.x) * particlePos;
      const particleY = sourceProj.y + (targetProj.y - sourceProj.y) * particlePos;
      
      ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.beginPath();
      ctx.arc(particleX, particleY, 3 * conn.strength, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  // Mouse interaction handling
  const handleMouseInteraction = (event: MouseEvent, canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    if (event.type === 'mousedown') {
      mouseRef.current.isDown = true;
      mouseRef.current.x = x;
      mouseRef.current.y = y;
      
      // Check for node selection
      const clickedNode = renderNodesRef.current.find(node => {
        const projected = project3D(node.x, node.y, node.z, canvas.width, canvas.height);
        const distance = Math.sqrt((x - projected.x) ** 2 + (y - projected.y) ** 2);
        return distance < node.radius * projected.scale;
      });
      
      if (clickedNode) {
        setSelectedNode(clickedNode.node);
      } else {
        setSelectedNode(null);
      }
    } else if (event.type === 'mouseup') {
      mouseRef.current.isDown = false;
    } else if (event.type === 'mousemove') {
      if (mouseRef.current.isDown) {
        // Camera rotation
        const deltaX = x - mouseRef.current.x;
        const deltaY = y - mouseRef.current.y;
        
        cameraRef.current.rotation.y += deltaX * 0.01;
        cameraRef.current.rotation.x += deltaY * 0.01;
        
        // Clamp X rotation
        cameraRef.current.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, cameraRef.current.rotation.x));
        
        mouseRef.current.x = x;
        mouseRef.current.y = y;
      } else {
        // Hover detection
        const hoveredNode = renderNodesRef.current.find(node => {
          const projected = project3D(node.x, node.y, node.z, canvas.width, canvas.height);
          const distance = Math.sqrt((x - projected.x) ** 2 + (y - projected.y) ** 2);
          return distance < node.radius * projected.scale * 1.2;
        });
        
        setHoverNode(hoveredNode?.node || null);
      }
    } else if (event.type === 'wheel') {
      event.preventDefault();
      const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
      cameraRef.current.zoom = Math.max(0.1, Math.min(3, cameraRef.current.zoom * zoomFactor));
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup mouse events
    const mouseEvents = ['mousedown', 'mouseup', 'mousemove', 'wheel'];
    mouseEvents.forEach(eventType => {
      canvas.addEventListener(eventType, (e) => handleMouseInteraction(e as MouseEvent, canvas));
    });

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Initialize render nodes
    const initializeRenderNodes = () => {
      renderNodesRef.current = nodes.map((node, index) => {
        const angle = (index / nodes.length) * 2 * Math.PI;
        const radius = 150 + Math.random() * 100;
        const height = (Math.random() - 0.5) * 100;
        
        return {
          id: node.id,
          x: Math.cos(angle) * radius,
          y: height,
          z: Math.sin(angle) * radius,
          radius: 8 + node.value / 10,
          color: `hsl(${(node.phase * 180 / Math.PI) % 360}, 70%, 60%)`,
          glowIntensity: node.activation,
          velocity: { x: 0, y: 0, z: 0 },
          node
        };
      });
    };

    if (renderNodesRef.current.length !== nodes.length) {
      initializeRenderNodes();
    } else {
      // Update existing nodes
      renderNodesRef.current.forEach(renderNode => {
        const updatedNode = nodes.find(n => n.id === renderNode.id);
        if (updatedNode) {
          renderNode.node = updatedNode;
          renderNode.glowIntensity = updatedNode.activation;
          renderNode.color = `hsl(${(updatedNode.phase * 180 / Math.PI) % 360}, 70%, 60%)`;
        }
      });
    }

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const time = Date.now() * 0.001;

      // Clear with fade effect
      ctx.fillStyle = 'rgba(17, 24, 39, 0.1)';
      ctx.fillRect(0, 0, width, height);

      if (isRunning && renderNodesRef.current.length > 0) {
        // Update physics
        updateNodePositions(renderNodesRef.current, 0.016);
      }

      // Prepare connections for rendering
      const renderConnections: RenderConnection[] = [];
      renderNodesRef.current.forEach(node => {
        node.node.connections.forEach(connId => {
          const targetNode = renderNodesRef.current.find(rn => rn.id === connId);
          if (targetNode) {
            renderConnections.push({
              source: node,
              target: targetNode,
              strength: (node.node.activation + targetNode.node.activation) / 2,
              pulse: time + Math.random() * 2 * Math.PI
            });
          }
        });
      });

      // Draw connections first (behind nodes)
      renderConnections.forEach(conn => {
        drawConnection(ctx, conn, time);
      });

      // Sort nodes by Z-depth for proper rendering
      const sortedNodes = [...renderNodesRef.current].sort((a, b) => b.z - a.z);

      // Draw nodes
      sortedNodes.forEach(node => {
        const projected = project3D(node.x, node.y, node.z, width, height);
        if (projected.scale > 0) {
          drawNode(ctx, node, projected, time);
        }
      });

      // Draw fractal background patterns
      if (isRunning) {
        ctx.globalAlpha = 0.05;
        for (let i = 0; i < 3; i++) {
          const spiralRadius = 100 + i * 50;
          const spiralTurns = 2;
          const spiralPoints = 50;
          
          ctx.strokeStyle = `hsl(${180 + i * 120}, 70%, 60%)`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          
          for (let j = 0; j <= spiralPoints; j++) {
            const t = (j / spiralPoints) * spiralTurns * 2 * Math.PI;
            const r = spiralRadius * (1 - j / spiralPoints) * (1 + 0.2 * Math.sin(time * 2 + i));
            const projected = project3D(
              r * Math.cos(t + time + i),
              Math.sin(time * 3 + i) * 30,
              r * Math.sin(t + time + i),
              width,
              height
            );
            
            if (j === 0) {
              ctx.moveTo(projected.x, projected.y);
            } else {
              ctx.lineTo(projected.x, projected.y);
            }
          }
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      mouseEvents.forEach(eventType => {
        canvas.removeEventListener(eventType, (e) => handleMouseInteraction(e as MouseEvent, canvas));
      });
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [nodes, isRunning, selectedNode, hoverNode, drawConnection, drawNode, handleMouseInteraction]);

  return (
    <div className="relative w-full h-full min-h-[500px] bg-gray-900 rounded-xl overflow-hidden border border-gray-700">
      <canvas 
        ref={canvasRef}
        className="w-full h-full cursor-move"
        style={{ background: 'radial-gradient(circle at center, rgba(17, 24, 39, 0.9), rgb(17, 24, 39))' }}
      />
      
      {/* Controls */}
      <div className="absolute top-4 left-4 bg-black/30 backdrop-blur-sm rounded-lg p-4 min-w-[200px]">
        <h3 className="text-white font-bold text-sm mb-3 flex items-center space-x-2">
          <Atom className="text-cyan-400" size={16} />
          <span>AtomSpace Hypergraph</span>
        </h3>
        <div className="text-gray-300 text-xs space-y-2">
          <div className="flex justify-between">
            <span>Nodes:</span>
            <span className="text-cyan-400 font-mono">{nodes.length}</span>
          </div>
          <div className="flex justify-between">
            <span>Connections:</span>
            <span className="text-purple-400 font-mono">
              {nodes.reduce((sum, n) => sum + n.connections.length, 0)}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Avg Activation:</span>
            <span className="text-green-400 font-mono">
              {nodes.length > 0 ? Math.round(nodes.reduce((sum, n) => sum + n.activation, 0) / nodes.length * 100) : 0}%
            </span>
          </div>
          <div className="flex justify-between">
            <span>Fractal Nodes:</span>
            <span className="text-orange-400 font-mono">
              {nodes.filter(n => n.fractalDimension && n.fractalDimension > 1.5).length}
            </span>
          </div>
        </div>
        
        <div className="mt-3 pt-2 border-t border-gray-600 text-xs text-gray-400">
          <div>🖱️ Drag: Rotate camera</div>
          <div>🎯 Click: Select node</div>
          <div>🔍 Scroll: Zoom in/out</div>
        </div>
      </div>

      {/* Node Details Panel */}
      {selectedNode && (
        <div className="absolute top-4 right-4 bg-black/40 backdrop-blur-sm rounded-lg p-4 max-w-[300px]">
          <div className="flex items-center space-x-2 mb-3">
            <div className="p-2 bg-cyan-500/20 rounded">
              <Brain className="text-cyan-400" size={16} />
            </div>
            <div>
              <h4 className="text-white font-bold text-sm">
                Prime {selectedNode.value}
              </h4>
              <p className="text-gray-400 text-xs">Cognitive Node Analysis</p>
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400">Activation</div>
                <div className="text-cyan-400 font-mono">
                  {Math.round(selectedNode.activation * 100)}%
                </div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400">Phase</div>
                <div className="text-purple-400 font-mono">
                  {(selectedNode.phase * 180 / Math.PI).toFixed(1)}°
                </div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400">Connections</div>
                <div className="text-green-400 font-mono">
                  {selectedNode.connections.length}
                </div>
              </div>
              <div className="bg-gray-800/50 rounded p-2">
                <div className="text-gray-400">Fractal Dim</div>
                <div className="text-orange-400 font-mono">
                  {selectedNode.fractalDimension?.toFixed(2) || 'N/A'}
                </div>
              </div>
            </div>
            
            {selectedNode.truthValue && (
              <div className="bg-gray-800/30 rounded p-3">
                <div className="text-gray-300 text-xs font-semibold mb-2">PLN Truth Value</div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Strength:</span>
                  <span className="text-white">{selectedNode.truthValue.strength.toFixed(3)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Confidence:</span>
                  <span className="text-white">{selectedNode.truthValue.confidence.toFixed(3)}</span>
                </div>
              </div>
            )}
            
            <div className="bg-gray-800/30 rounded p-3">
              <div className="text-gray-300 text-xs font-semibold mb-2">11D Manifold (Top 3)</div>
              <div className="space-y-1">
                {selectedNode.dimension.slice(0, 3).map((dim, i) => (
                  <div key={i} className="flex items-center space-x-2">
                    <span className="text-gray-400 text-xs w-8">D{i+1}:</span>
                    <div className="flex-1 bg-gray-700 rounded-full h-1">
                      <div 
                        className="h-1 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500"
                        style={{ width: `${Math.abs(dim) * 50 + 50}%` }}
                      />
                    </div>
                    <span className="text-white text-xs font-mono w-12">
                      {dim.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Status Indicators */}
      {isRunning && (
        <>
          <div className="absolute bottom-4 right-4 text-cyan-400 text-xs font-mono animate-pulse">
            ◆ CONSCIOUSNESS ACTIVE ◆
          </div>
          
          <div className="absolute bottom-4 left-4 flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <Target className="text-green-400" size={12} />
              <span className="text-gray-300">
                Camera: {cameraRef.current.rotation.y.toFixed(1)}°, {cameraRef.current.rotation.x.toFixed(1)}°
              </span>
            </div>
            <div className="flex items-center space-x-1">
              <Zap className="text-yellow-400" size={12} />
              <span className="text-gray-300">
                Zoom: {(cameraRef.current.zoom * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};