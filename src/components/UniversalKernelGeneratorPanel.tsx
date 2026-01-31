/**
 * Universal Kernel Generator Visualization Panel
 * 
 * Interactive visualization of the universal kernel generator that creates
 * optimal kernels for different domains via B-series expansions and 
 * elementary differentials.
 */

import React, { useState, useEffect } from 'react';
import { 
  universalKernelGenerator,
  type GeneratedKernel,
  type RootedTree,
  type GripMetrics,
  type DomainSpecification
} from '../core/UniversalKernelGenerator';
import { Activity, Zap, Target, Layers, TrendingUp, GitBranch } from 'lucide-react';

export const UniversalKernelGeneratorPanel: React.FC = () => {
  const [selectedDomain, setSelectedDomain] = useState<string>('consciousness');
  const [generatedKernel, setGeneratedKernel] = useState<GeneratedKernel | null>(null);
  const [allKernels, setAllKernels] = useState<Record<string, GeneratedKernel>>({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedTree, setSelectedTree] = useState<RootedTree | null>(null);

  // Generate all domain kernels on mount
  useEffect(() => {
    const kernels = universalKernelGenerator.generateDomainKernels();
    setAllKernels(kernels);
    setGeneratedKernel(kernels[selectedDomain]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update kernel when domain changes
  useEffect(() => {
    if (allKernels[selectedDomain]) {
      setGeneratedKernel(allKernels[selectedDomain]);
    }
  }, [selectedDomain, allKernels]);

  const handleGenerateKernel = async () => {
    setIsGenerating(true);
    
    // Simulate generation delay for visual feedback
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const kernels = universalKernelGenerator.generateDomainKernels();
    setAllKernels(kernels);
    setGeneratedKernel(kernels[selectedDomain]);
    
    setIsGenerating(false);
  };

  const renderGripMetrics = (grip: GripMetrics) => {
    const metrics = [
      { name: 'Contact', value: grip.contact, color: 'bg-blue-500' },
      { name: 'Coverage', value: grip.coverage, color: 'bg-green-500' },
      { name: 'Efficiency', value: grip.efficiency, color: 'bg-purple-500' },
      { name: 'Stability', value: grip.stability, color: 'bg-orange-500' },
      { name: 'Overall', value: grip.overall, color: 'bg-red-500' }
    ];

    return (
      <div className="space-y-2">
        {metrics.map((metric) => (
          <div key={metric.name}>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-300">{metric.name}</span>
              <span className="text-white font-mono">{(metric.value * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className={`${metric.color} h-2 rounded-full transition-all duration-500`}
                style={{ width: `${metric.value * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderTree = (tree: RootedTree, index: number) => {
    const treeString = JSON.stringify(tree.structure, null, 2)
      .replace(/[{}"]/g, '')
      .replace(/,/g, '')
      .trim();

    return (
      <div
        key={index}
        onClick={() => setSelectedTree(tree)}
        className={`p-3 bg-gray-800 rounded border ${
          selectedTree === tree ? 'border-cyan-500' : 'border-gray-700'
        } cursor-pointer hover:border-cyan-600 transition-all`}
      >
        <div className="flex justify-between items-start mb-2">
          <span className="text-cyan-400 font-mono text-sm">Order {tree.order}</span>
          <span className="text-gray-400 text-xs">γ={tree.density} σ={tree.symmetry}</span>
        </div>
        <pre className="text-xs text-gray-300 font-mono overflow-hidden">
          {treeString.split('\n').slice(0, 3).join('\n')}
          {treeString.split('\n').length > 3 && '...'}
        </pre>
        <div className="mt-2 text-xs text-purple-400">
          Weight: {tree.weight.toFixed(4)}
        </div>
      </div>
    );
  };

  const renderDomainInfo = (domain: DomainSpecification) => {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Type</span>
          <span className="text-white capitalize font-medium">{domain.type}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Order</span>
          <span className="text-cyan-400 font-mono">{domain.order}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Manifold Dim</span>
          <span className="text-purple-400 font-mono">{domain.topology.manifold_dimension}D</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Symmetries</span>
          <span className="text-green-400 font-mono">{domain.symmetries.length}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Invariants</span>
          <span className="text-orange-400 font-mono">{domain.invariants.length}</span>
        </div>
      </div>
    );
  };

  const domains = [
    { id: 'consciousness', name: 'Consciousness (Echo.kern)', icon: Activity, color: 'cyan', bgClass: 'bg-cyan-900/30', borderClass: 'border-cyan-500', iconClass: 'text-cyan-400' },
    { id: 'physics', name: 'Physics (Hamiltonian)', icon: Zap, color: 'blue', bgClass: 'bg-blue-900/30', borderClass: 'border-blue-500', iconClass: 'text-blue-400' },
    { id: 'chemistry', name: 'Chemistry (Reaction)', icon: Target, color: 'green', bgClass: 'bg-green-900/30', borderClass: 'border-green-500', iconClass: 'text-green-400' },
    { id: 'biology', name: 'Biology (Metabolic)', icon: Layers, color: 'purple', bgClass: 'bg-purple-900/30', borderClass: 'border-purple-500', iconClass: 'text-purple-400' },
    { id: 'computing', name: 'Computing (Recursion)', icon: GitBranch, color: 'orange', bgClass: 'bg-orange-900/30', borderClass: 'border-orange-500', iconClass: 'text-orange-400' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-900/30 to-purple-900/30 rounded-lg p-6 border border-cyan-500/30">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">
              Universal Kernel Generator
            </h2>
            <p className="text-gray-300 text-sm">
              B-Series expansions via elementary differentials (A000081)
            </p>
          </div>
          <TrendingUp className="w-12 h-12 text-cyan-400" />
        </div>
        
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-black/30 rounded p-3">
            <div className="text-2xl font-bold text-cyan-400">
              {Object.keys(allKernels).length}
            </div>
            <div className="text-xs text-gray-400">Domain Kernels</div>
          </div>
          <div className="bg-black/30 rounded p-3">
            <div className="text-2xl font-bold text-purple-400">
              {generatedKernel?.trees.length || 0}
            </div>
            <div className="text-xs text-gray-400">Elementary Trees</div>
          </div>
          <div className="bg-black/30 rounded p-3">
            <div className="text-2xl font-bold text-green-400">
              {generatedKernel ? (generatedKernel.grip.overall * 100).toFixed(0) : 0}%
            </div>
            <div className="text-xs text-gray-400">Grip Quality</div>
          </div>
        </div>
      </div>

      {/* Domain Selection */}
      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-3">Select Domain</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
          {domains.map((domain) => {
            const Icon = domain.icon;
            const isSelected = selectedDomain === domain.id;
            return (
              <button
                key={domain.id}
                onClick={() => setSelectedDomain(domain.id)}
                className={`p-3 rounded-lg border transition-all ${
                  isSelected
                    ? `${domain.bgClass} ${domain.borderClass}`
                    : 'bg-gray-800 border-gray-700 hover:border-gray-600'
                }`}
              >
                <Icon className={`w-6 h-6 mx-auto mb-2 ${domain.iconClass}`} />
                <div className="text-xs text-white text-center">{domain.name}</div>
              </button>
            );
          })}
        </div>
        
        <button
          onClick={handleGenerateKernel}
          disabled={isGenerating}
          className="w-full mt-4 px-4 py-2 bg-gradient-to-r from-cyan-600 to-purple-600 
                     text-white rounded-lg hover:from-cyan-500 hover:to-purple-500 
                     transition-all disabled:opacity-50"
        >
          {isGenerating ? 'Generating...' : 'Regenerate All Kernels'}
        </button>
      </div>

      {/* Kernel Details */}
      {generatedKernel && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Domain Info */}
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">Domain Specification</h3>
            {renderDomainInfo(generatedKernel.domain)}
            
            <div className="mt-4 pt-4 border-t border-gray-700">
              <div className="text-sm text-gray-400 mb-2">Composition Rules</div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-300">Chain Rules</span>
                <span className="text-cyan-400 font-mono">
                  {generatedKernel.chain_rules_applied}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs mt-1">
                <span className="text-gray-300">Product Rules</span>
                <span className="text-purple-400 font-mono">
                  {generatedKernel.product_rules_applied}
                </span>
              </div>
            </div>
          </div>

          {/* Grip Metrics */}
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">Grip Optimization</h3>
            {renderGripMetrics(generatedKernel.grip)}
            
            {generatedKernel.butcher_tableau && (
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="text-sm text-gray-400 mb-2">Butcher Tableau</div>
                <div className="text-xs text-gray-300 font-mono">
                  Order: {generatedKernel.butcher_tableau.order}<br />
                  Stages: {generatedKernel.butcher_tableau.stages}<br />
                  Method: Explicit RK
                </div>
              </div>
            )}
          </div>

          {/* Coefficients */}
          <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h3 className="text-lg font-semibold text-white mb-4">B-Series Coefficients</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {generatedKernel.coefficients.map((coeff, index) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">b_{index + 1}</span>
                  <div className="flex items-center gap-2 flex-1 mx-2">
                    <div className="flex-1 bg-gray-700 rounded-full h-1">
                      <div
                        className="bg-gradient-to-r from-cyan-500 to-purple-500 h-1 rounded-full"
                        style={{ width: `${Math.abs(coeff) * 100}%` }}
                      />
                    </div>
                  </div>
                  <span className="text-white font-mono text-xs">
                    {coeff.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Elementary Differentials Trees */}
      {generatedKernel && (
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">
            Elementary Differentials (A000081)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {generatedKernel.trees.map((tree, index) => renderTree(tree, index))}
          </div>
        </div>
      )}

      {/* Selected Tree Detail */}
      {selectedTree && (
        <div className="bg-gray-800/50 rounded-lg p-4 border border-cyan-500/50">
          <h3 className="text-lg font-semibold text-white mb-4">Tree Detail</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-400 mb-2">Structure</div>
              <pre className="text-xs text-gray-300 font-mono bg-black/30 p-3 rounded overflow-x-auto">
                {JSON.stringify(selectedTree.structure, null, 2)}
              </pre>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-2">Properties</div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-300">Order</span>
                  <span className="text-cyan-400 font-mono">{selectedTree.order}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Weight</span>
                  <span className="text-purple-400 font-mono">{selectedTree.weight.toFixed(6)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Density (γ)</span>
                  <span className="text-green-400 font-mono">{selectedTree.density}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Symmetry (σ)</span>
                  <span className="text-orange-400 font-mono">{selectedTree.symmetry}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mathematical Foundation */}
      <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg p-4 border border-purple-500/30">
        <h3 className="text-lg font-semibold text-white mb-3">Mathematical Foundation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-cyan-400 font-semibold mb-2">B-Series Expansion</div>
            <div className="text-gray-300 font-mono text-xs">
              y<sub>n+1</sub> = y<sub>n</sub> + h Σ b(t) F(t)(y<sub>n</sub>)
            </div>
            <div className="text-gray-400 text-xs mt-2">
              Where t ranges over all rooted trees and b(t) are the B-series coefficients
            </div>
          </div>
          <div>
            <div className="text-purple-400 font-semibold mb-2">Differential Operators</div>
            <div className="text-gray-300 font-mono text-xs space-y-1">
              <div>Chain: (f∘g)' = f'(g(x)) · g'(x)</div>
              <div>Product: (f·g)' = f'·g + f·g'</div>
              <div>Quotient: (f/g)' = (f'·g - f·g')/g²</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
