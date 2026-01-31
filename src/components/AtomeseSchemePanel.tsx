import React, { useState, useEffect } from 'react';
import { Code2, Copy, Download, Play, Maximize2 } from 'lucide-react';
import { useFundamentalFeatures } from '../hooks/useFundamentalFeatures';

export const AtomeseSchemePanel: React.FC = () => {
  const { 
    schemeCode, 
    initializeRecursiveAtomeseKnowledgeBase,
    recursiveAtomeseKnowledgeBase,
    rootNodeAttention,
    recursiveDepth,
    neuralSymbolicBridgeStrength
  } = useFundamentalFeatures();
  
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  // Initialize the recursive knowledge base on component mount
  useEffect(() => {
    if (!schemeCode) {
      initializeRecursiveAtomeseKnowledgeBase();
    }
  }, [schemeCode, initializeRecursiveAtomeseKnowledgeBase]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(schemeCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([schemeCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'nanobrain-fundamental-features-atomese.scm';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const executeSchemeSimulation = () => {
    // Simulate execution by highlighting the code structure
    console.log('Executing Atomese Knowledge Base:', {
      totalExpressions: recursiveAtomeseKnowledgeBase.length,
      rootNodeAttention,
      recursiveDepth,
      neuralSymbolicBridgeStrength
    });
  };

  return (
    <div className="bg-gray-900/90 border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gray-800/50 px-4 py-3 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Code2 className="w-5 h-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-white">
              Complete Scheme/Atomese Recursive Instantiation
            </h3>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className="bg-cyan-900/30 px-2 py-1 rounded text-cyan-300">
                OpenCog Compatible
              </span>
              <span className="bg-purple-900/30 px-2 py-1 rounded text-purple-300">
                Hypergraph Encoding
              </span>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={executeSchemeSimulation}
              className="flex items-center gap-2 px-3 py-2 bg-green-600/20 hover:bg-green-600/30 
                         text-green-400 rounded-lg transition-colors"
              title="Simulate Execution"
            >
              <Play className="w-4 h-4" />
              Execute
            </button>
            
            <button
              onClick={handleCopy}
              className="flex items-center gap-2 px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 
                         text-blue-400 rounded-lg transition-colors"
              title="Copy to Clipboard"
            >
              <Copy className="w-4 h-4" />
              {copied ? 'Copied!' : 'Copy'}
            </button>
            
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-3 py-2 bg-purple-600/20 hover:bg-purple-600/30 
                         text-purple-400 rounded-lg transition-colors"
              title="Download .scm file"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
            
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-2 px-3 py-2 bg-gray-600/20 hover:bg-gray-600/30 
                         text-gray-400 rounded-lg transition-colors"
              title={isExpanded ? 'Collapse' : 'Expand'}
            >
              <Maximize2 className="w-4 h-4" />
              {isExpanded ? 'Collapse' : 'Expand'}
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Bar */}
      <div className="bg-gray-800/30 px-4 py-2 border-b border-gray-700/50">
        <div className="grid grid-cols-4 gap-4 text-sm">
          <div className="text-center">
            <div className="text-cyan-400 font-medium">
              {recursiveAtomeseKnowledgeBase.length}
            </div>
            <div className="text-gray-500">Expressions</div>
          </div>
          <div className="text-center">
            <div className="text-green-400 font-medium">
              {(rootNodeAttention * 100).toFixed(1)}%
            </div>
            <div className="text-gray-500">Root Attention</div>
          </div>
          <div className="text-center">
            <div className="text-purple-400 font-medium">
              {recursiveDepth}
            </div>
            <div className="text-gray-500">Max Depth</div>
          </div>
          <div className="text-center">
            <div className="text-yellow-400 font-medium">
              {(neuralSymbolicBridgeStrength * 100).toFixed(1)}%
            </div>
            <div className="text-gray-500">Bridge Strength</div>
          </div>
        </div>
      </div>

      {/* Code Display */}
      <div className={`transition-all duration-300 ${isExpanded ? 'max-h-[80vh]' : 'max-h-96'} overflow-y-auto`}>
        <pre className="p-4 text-sm font-mono text-gray-300 leading-relaxed">
          <code className="whitespace-pre-wrap">
            {schemeCode || 'Loading Atomese recursive instantiation...'}
          </code>
        </pre>
      </div>

      {/* Features Summary */}
      <div className="bg-gray-800/30 px-4 py-3 border-t border-gray-700/50">
        <div className="text-sm text-gray-400">
          <div className="font-medium text-white mb-2">Implemented Features:</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
              <span>Philosophical Transformation (Root Axioms)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>Fractal Information Theory & GML</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <span>Phase Prime Metric (Symmetry Geometry)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
              <span>Neural-Symbolic Integration Bridge</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};