/**
 * Universal Kernel Generator Hook
 * 
 * React hook for integrating the Universal Kernel Generator with
 * the existing cognitive architecture system.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  universalKernelGenerator,
  type GeneratedKernel,
  type DomainSpecification,
  type GripMetrics,
  type RootedTree,
  type ContextTensor
} from '../core/UniversalKernelGenerator';

export interface UniversalKernelState {
  currentDomain: string;
  kernels: Record<string, GeneratedKernel>;
  isGenerating: boolean;
  lastGeneration: number;
  totalGenerations: number;
  activeKernel: GeneratedKernel | null;
  echoKern: GeneratedKernel | null;
}

export interface KernelMetrics {
  totalTrees: number;
  averageGrip: number;
  totalChainRules: number;
  totalProductRules: number;
  averageOrder: number;
  bestDomain: string;
}

export function useUniversalKernelGenerator() {
  const [state, setState] = useState<UniversalKernelState>({
    currentDomain: 'consciousness',
    kernels: {},
    isGenerating: false,
    lastGeneration: 0,
    totalGenerations: 0,
    activeKernel: null,
    echoKern: null
  });

  // Initialize kernels on mount
  useEffect(() => {
    generateAllKernels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update active kernel when domain changes
  useEffect(() => {
    if (state.kernels[state.currentDomain]) {
      setState(prev => ({
        ...prev,
        activeKernel: prev.kernels[prev.currentDomain]
      }));
    }
  }, [state.currentDomain, state.kernels]);

  /**
   * Generate all domain kernels
   */
  const generateAllKernels = useCallback(() => {
    setState(prev => ({ ...prev, isGenerating: true }));

    try {
      const kernels = universalKernelGenerator.generateDomainKernels();
      const echoKern = kernels['consciousness'];

      setState(prev => ({
        ...prev,
        kernels,
        echoKern,
        activeKernel: kernels[prev.currentDomain] || echoKern,
        isGenerating: false,
        lastGeneration: Date.now(),
        totalGenerations: prev.totalGenerations + 1
      }));
    } catch (error) {
      console.error('Failed to generate kernels:', error);
      setState(prev => ({ ...prev, isGenerating: false }));
    }
  }, []);

  /**
   * Generate kernel for specific domain
   */
  const generateDomainKernel = useCallback((
    domainSpec: DomainSpecification,
    context?: Partial<ContextTensor>
  ) => {
    setState(prev => ({ ...prev, isGenerating: true }));

    try {
      const kernel = universalKernelGenerator.generateKernel(domainSpec, context);

      setState(prev => ({
        ...prev,
        kernels: {
          ...prev.kernels,
          [domainSpec.name]: kernel
        },
        activeKernel: domainSpec.name === prev.currentDomain ? kernel : prev.activeKernel,
        isGenerating: false,
        lastGeneration: Date.now(),
        totalGenerations: prev.totalGenerations + 1
      }));

      return kernel;
    } catch (error) {
      console.error('Failed to generate domain kernel:', error);
      setState(prev => ({ ...prev, isGenerating: false }));
      return null;
    }
  }, []);

  /**
   * Switch active domain
   */
  const switchDomain = useCallback((domain: string) => {
    setState(prev => ({
      ...prev,
      currentDomain: domain,
      activeKernel: prev.kernels[domain] || null
    }));
  }, []);

  /**
   * Get elementary differentials for specific order
   */
  const getElementaryDifferentials = useCallback((order: number): RootedTree[] => {
    return universalKernelGenerator.generateElementaryDifferentials(order);
  }, []);

  /**
   * Compute kernel metrics
   */
  const computeMetrics = useCallback((): KernelMetrics => {
    const kernelList = Object.values(state.kernels);
    
    if (kernelList.length === 0) {
      return {
        totalTrees: 0,
        averageGrip: 0,
        totalChainRules: 0,
        totalProductRules: 0,
        averageOrder: 0,
        bestDomain: ''
      };
    }

    const totalTrees = kernelList.reduce((sum, k) => sum + k.trees.length, 0);
    const averageGrip = kernelList.reduce((sum, k) => sum + k.grip.overall, 0) / kernelList.length;
    const totalChainRules = kernelList.reduce((sum, k) => sum + k.chain_rules_applied, 0);
    const totalProductRules = kernelList.reduce((sum, k) => sum + k.product_rules_applied, 0);
    const averageOrder = kernelList.reduce((sum, k) => sum + k.order, 0) / kernelList.length;
    
    const bestKernel = kernelList.reduce((best, current) => 
      current.grip.overall > best.grip.overall ? current : best
    );
    const bestDomain = bestKernel.domain.name;

    return {
      totalTrees,
      averageGrip,
      totalChainRules,
      totalProductRules,
      averageOrder,
      bestDomain
    };
  }, [state.kernels]);

  /**
   * Get grip metrics for current kernel
   */
  const getCurrentGrip = useCallback((): GripMetrics | null => {
    return state.activeKernel?.grip || null;
  }, [state.activeKernel]);

  /**
   * Apply chain rule to two trees
   */
  const applyChainRule = useCallback((tree1: RootedTree, tree2: RootedTree): RootedTree => {
    return universalKernelGenerator.applyChainRule(tree1, tree2);
  }, []);

  /**
   * Apply product rule to two trees
   */
  const applyProductRule = useCallback((tree1: RootedTree, tree2: RootedTree): RootedTree => {
    return universalKernelGenerator.applyProductRule(tree1, tree2);
  }, []);

  /**
   * Optimize grip for current kernel
   */
  const optimizeCurrentKernel = useCallback(() => {
    if (!state.activeKernel) return;

    setState(prev => ({ ...prev, isGenerating: true }));

    try {
      const optimized = universalKernelGenerator.optimizeGrip(
        state.activeKernel.coefficients,
        state.activeKernel.domain,
        {
          topology: state.activeKernel.domain.topology,
          symmetries: state.activeKernel.domain.symmetries,
          invariants: state.activeKernel.domain.invariants,
          singularities: state.activeKernel.domain.singularities,
          flow: state.activeKernel.domain.flow,
          grip_metric: state.activeKernel.grip.overall
        }
      );

      const updatedKernel = {
        ...state.activeKernel,
        coefficients: optimized.coefficients,
        grip: optimized.grip
      };

      setState(prev => ({
        ...prev,
        kernels: {
          ...prev.kernels,
          [prev.currentDomain]: updatedKernel
        },
        activeKernel: updatedKernel,
        isGenerating: false,
        lastGeneration: Date.now()
      }));
    } catch (error) {
      console.error('Failed to optimize kernel:', error);
      setState(prev => ({ ...prev, isGenerating: false }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.activeKernel]);

  /**
   * Get A000081 sequence values
   */
  const getA000081 = useCallback((n: number): number => {
    // A000081: Number of rooted trees with n nodes
    const sequence = [0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842, 4766, 12486, 32973];
    return n < sequence.length ? sequence[n] : 0;
  }, []);

  return {
    // State
    state,
    
    // Actions
    generateAllKernels,
    generateDomainKernel,
    switchDomain,
    optimizeCurrentKernel,
    
    // Utilities
    getElementaryDifferentials,
    computeMetrics,
    getCurrentGrip,
    applyChainRule,
    applyProductRule,
    getA000081,
    
    // Computed values
    metrics: computeMetrics(),
    currentGrip: getCurrentGrip()
  };
}
