import { useState, useEffect, useCallback, useRef } from 'react';
import { AtomSpaceConnection, PLNRule, CognitiveNode } from '../types';

// Enhanced OpenCog-inspired AtomSpace with PLN and ECAN
export interface EnhancedAtomSpace {
  atoms: AtomSpaceAtom[];
  links: AtomSpaceLink[];
  attentionBank: AttentionBank;
  plnEngine: PLNEngine;
  patternMatcher: PatternMatcher;
  totalAtoms: number;
  attentionSpread: AttentionSpread[];
}

export interface AtomSpaceAtom {
  id: string;
  type: 'ConceptNode' | 'PredicateNode' | 'NumberNode' | 'VariableNode';
  name: string;
  truthValue: TruthValue;
  attentionValue: AttentionValue;
  importance: number;
  timestamp: number;
}

export interface AtomSpaceLink {
  id: string;
  type: 'InheritanceLink' | 'SimilarityLink' | 'ImplicationLink' | 'EvaluationLink' | 'ListLink';
  outgoing: string[]; // IDs of connected atoms
  truthValue: TruthValue;
  attentionValue: AttentionValue;
  strength: number;
}

export interface TruthValue {
  strength: number; // [0, 1]
  confidence: number; // [0, 1]
  count: number; // Evidence count
}

export interface AttentionValue {
  sti: number; // Short-term importance
  lti: number; // Long-term importance
  vlti: boolean; // Very long-term importance flag
}

// Economic Attention Allocation (ECAN) system
export interface AttentionBank {
  totalSTI: number;
  totalLTI: number;
  funds: number;
  rentCollected: number;
  attentionFocus: string[]; // IDs of atoms with highest attention
  forgettingThreshold: number;
}

export interface AttentionSpread {
  sourceAtom: string;
  targetAtom: string;
  spreadAmount: number;
  reason: string;
  timestamp: number;
}

// Probabilistic Logic Networks (PLN) engine
export interface PLNEngine {
  rules: PLNInferenceRule[];
  inferences: PLNInference[];
  truthValueSystem: TruthValueSystem;
  uncertaintyHandling: UncertaintyHandler;
}

export interface PLNInferenceRule {
  id: string;
  name: string;
  type: 'deduction' | 'induction' | 'abduction' | 'modus_ponens' | 'modus_tollens' | 'inheritance';
  formula: string;
  applicability: (atoms: AtomSpaceAtom[], links: AtomSpaceLink[]) => boolean;
  apply: (premises: string[]) => PLNInference;
}

export interface PLNInference {
  id: string;
  rule: string;
  premises: string[];
  conclusion: string;
  truthValue: TruthValue;
  confidence: number;
  timestamp: number;
}

export interface TruthValueSystem {
  revision: (tv1: TruthValue, tv2: TruthValue) => TruthValue;
  deduction: (tvA: TruthValue, tvB: TruthValue) => TruthValue;
  induction: (tvA: TruthValue, tvB: TruthValue) => TruthValue;
  abduction: (tvA: TruthValue, tvB: TruthValue) => TruthValue;
}

export interface UncertaintyHandler {
  handleInconsistency: (conflictingAtoms: string[]) => void;
  computeEntropy: (atoms: AtomSpaceAtom[]) => number;
  bayesianUpdate: (prior: TruthValue, evidence: TruthValue) => TruthValue;
}

export interface PatternMatcher {
  patterns: AtomSpacePattern[];
  matches: PatternMatch[];
  bindLink: (pattern: AtomSpacePattern, atomSpace: EnhancedAtomSpace) => PatternMatch[];
}

export interface AtomSpacePattern {
  id: string;
  structure: string;
  variables: string[];
  constraints: string[];
}

export interface PatternMatch {
  id: string;
  pattern: string;
  bindings: Record<string, string>;
  confidence: number;
  atoms: string[];
}

export const useEnhancedAtomSpace = () => {
  const [atomSpace, setAtomSpace] = useState<EnhancedAtomSpace | null>(null);
  const [isActive, setIsActive] = useState(false);
  const attentionCycleRef = useRef<number>(0);
  const inferenceCountRef = useRef<number>(0);

  // Initialize truth value system
  const createTruthValueSystem = useCallback((): TruthValueSystem => {
    return {
      revision: (tv1: TruthValue, tv2: TruthValue): TruthValue => {
        // PLN revision formula
        const n1 = tv1.count;
        const n2 = tv2.count;
        const s1 = tv1.strength;
        const s2 = tv2.strength;
        
        const newCount = n1 + n2;
        const newStrength = (s1 * n1 + s2 * n2) / newCount;
        const newConfidence = newCount / (newCount + 1);
        
        return { strength: newStrength, confidence: newConfidence, count: newCount };
      },

      deduction: (tvA: TruthValue, tvB: TruthValue): TruthValue => {
        // (A → B) ∧ A ⊢ B
        const strength = tvA.strength * tvB.strength;
        const confidence = tvA.confidence * tvB.confidence;
        const count = Math.min(tvA.count, tvB.count);
        
        return { strength, confidence, count };
      },

      induction: (tvA: TruthValue, tvB: TruthValue): TruthValue => {
        // A ∧ B ⊢ (A → B)
        const strength = tvB.strength;
        const confidence = tvA.confidence * tvB.confidence * tvA.strength;
        const count = Math.min(tvA.count, tvB.count);
        
        return { strength, confidence, count };
      },

      abduction: (tvA: TruthValue, tvB: TruthValue): TruthValue => {
        // (A → B) ∧ B ⊢ A
        const strength = tvB.strength * tvA.strength;
        const confidence = tvA.confidence * tvB.confidence * (1 - tvA.strength);
        const count = Math.min(tvA.count, tvB.count);
        
        return { strength, confidence, count };
      }
    };
  }, []);

  // Create PLN inference rules
  const createPLNRules = useCallback((_: TruthValueSystem): PLNInferenceRule[] => {
    return [
      {
        id: 'inheritance-deduction',
        name: 'Inheritance Deduction',
        type: 'deduction',
        formula: 'Inheritance(A,B) ∧ Inheritance(B,C) ⊢ Inheritance(A,C)',
        applicability: (atoms, links) => {
          return links.some(l1 => 
            l1.type === 'InheritanceLink' && 
            links.some(l2 => 
              l2.type === 'InheritanceLink' && 
              l1.outgoing[1] === l2.outgoing[0]
            )
          );
        },
        apply: (premises) => {
          inferenceCountRef.current++;
          return {
            id: `inference-${inferenceCountRef.current}`,
            rule: 'inheritance-deduction',
            premises,
            conclusion: `Inheritance(${premises[0]}, ${premises[2]})`,
            truthValue: { strength: 0.8, confidence: 0.7, count: 1 },
            confidence: 0.75,
            timestamp: Date.now()
          };
        }
      },
      {
        id: 'similarity-inheritance',
        name: 'Similarity to Inheritance',
        type: 'induction',
        formula: 'Similarity(A,B) ⊢ Inheritance(A,B)',
        applicability: (atoms, links) => {
          return links.some(l => l.type === 'SimilarityLink');
        },
        apply: (premises) => {
          inferenceCountRef.current++;
          return {
            id: `inference-${inferenceCountRef.current}`,
            rule: 'similarity-inheritance',
            premises,
            conclusion: `Inheritance(${premises[0]}, ${premises[1]})`,
            truthValue: { strength: 0.6, confidence: 0.5, count: 1 },
            confidence: 0.6,
            timestamp: Date.now()
          };
        }
      },
      {
        id: 'modus-ponens',
        name: 'Modus Ponens',
        type: 'modus_ponens',
        formula: 'Implication(A,B) ∧ A ⊢ B',
        applicability: (atoms, links) => {
          return links.some(l => l.type === 'ImplicationLink');
        },
        apply: (premises) => {
          inferenceCountRef.current++;
          return {
            id: `inference-${inferenceCountRef.current}`,
            rule: 'modus-ponens',
            premises,
            conclusion: premises[1],
            truthValue: { strength: 0.85, confidence: 0.8, count: 1 },
            confidence: 0.8,
            timestamp: Date.now()
          };
        }
      }
    ];
  }, []);

  // Economic Attention Allocation (ECAN) system
  const performAttentionAllocation = useCallback((atomSpace: EnhancedAtomSpace): AttentionBank => {
    const { atoms, attentionBank } = atomSpace;
    
    // Collect attention rent from all atoms
    const rentRate = 0.01;
    let totalRentCollected = 0;
    
    const updatedAtoms = atoms.map(atom => {
      const rent = atom.attentionValue.sti * rentRate;
      totalRentCollected += rent;
      
      return {
        ...atom,
        attentionValue: {
          ...atom.attentionValue,
          sti: Math.max(0, atom.attentionValue.sti - rent)
        }
      };
    });

    // Distribute attention based on importance and activity
    const distributionFunds = attentionBank.funds + totalRentCollected;
    const highImportanceAtoms = updatedAtoms
      .filter(atom => atom.importance > 0.7)
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10);

    // Focus attention on most important atoms
    const attentionFocus = highImportanceAtoms.map(atom => atom.id);

    return {
      totalSTI: updatedAtoms.reduce((sum, atom) => sum + atom.attentionValue.sti, 0),
      totalLTI: updatedAtoms.reduce((sum, atom) => sum + atom.attentionValue.lti, 0),
      funds: distributionFunds * 0.1, // Keep 10% for next cycle
      rentCollected: totalRentCollected,
      attentionFocus,
      forgettingThreshold: 0.1
    };
  }, []);

  // Perform attention spreading
  const performAttentionSpreading = useCallback((atomSpace: EnhancedAtomSpace): AttentionSpread[] => {
    const spreads: AttentionSpread[] = [];
    const { atoms, links } = atomSpace;
    
    // Spread attention along high-strength links
    links.forEach(link => {
      if (link.strength > 0.7 && link.outgoing.length >= 2) {
        const sourceAtom = atoms.find(a => a.id === link.outgoing[0]);
        const targetAtom = atoms.find(a => a.id === link.outgoing[1]);
        
        if (sourceAtom && targetAtom && sourceAtom.attentionValue.sti > 10) {
          const spreadAmount = sourceAtom.attentionValue.sti * 0.1 * link.strength;
          
          spreads.push({
            sourceAtom: sourceAtom.id,
            targetAtom: targetAtom.id,
            spreadAmount,
            reason: `High-strength ${link.type} connection`,
            timestamp: Date.now()
          });
        }
      }
    });

    return spreads;
  }, []);

  // Pattern matching system
  const performPatternMatching = useCallback((atomSpace: EnhancedAtomSpace): PatternMatch[] => {
    const matches: PatternMatch[] = [];
    const { links } = atomSpace;

    // Define basic patterns to match
    const basicPatterns: AtomSpacePattern[] = [
      {
        id: 'inheritance-chain',
        structure: 'Inheritance($A, $B) ∧ Inheritance($B, $C)',
        variables: ['$A', '$B', '$C'],
        constraints: ['$A ≠ $B', '$B ≠ $C', '$A ≠ $C']
      }
    ];

    basicPatterns.forEach(pattern => {
      // Simplified pattern matching - in real implementation would be more sophisticated
      const inheritanceLinks = links.filter(l => l.type === 'InheritanceLink');

      if (pattern.id === 'inheritance-chain' && inheritanceLinks.length >= 2) {
        inheritanceLinks.forEach(link1 => {
          inheritanceLinks.forEach(link2 => {
            if (link1.outgoing[1] === link2.outgoing[0] && link1.id !== link2.id) {
              matches.push({
                id: `match-${pattern.id}-${Date.now()}-${Math.random()}`,
                pattern: pattern.id,
                bindings: {
                  '$A': link1.outgoing[0],
                  '$B': link1.outgoing[1],
                  '$C': link2.outgoing[1]
                },
                confidence: (link1.truthValue.confidence + link2.truthValue.confidence) / 2,
                atoms: [link1.outgoing[0], link1.outgoing[1], link2.outgoing[1]]
              });
            }
          });
        });
      }
    });

    return matches;
  }, []);

  // Initialize AtomSpace with cognitive nodes
  const initializeAtomSpace = useCallback((nodes: CognitiveNode[]) => {
    const truthValueSystem = createTruthValueSystem();
    const plnRules = createPLNRules(truthValueSystem);

    // Convert cognitive nodes to AtomSpace atoms
    const atoms: AtomSpaceAtom[] = nodes.map(node => ({
      id: node.id,
      type: node.type === 'prime' ? 'NumberNode' : 'ConceptNode',
      name: node.type === 'prime' ? node.value.toString() : node.id,
      truthValue: node.truthValue || { strength: node.activation, confidence: 0.8, count: 1 },
      attentionValue: {
        sti: (node.attentionValue || 0) * 100,
        lti: node.activation * 50,
        vlti: node.activation > 0.9
      },
      importance: node.activation * (node.confidence || 0.5),
      timestamp: node.timestamp
    }));

    // Create links from node connections
    const links: AtomSpaceLink[] = [];
    nodes.forEach(node => {
      node.connections.forEach(connId => {
        const targetNode = nodes.find(n => n.id === connId);
        if (targetNode) {
          links.push({
            id: `link-${node.id}-${connId}`,
            type: 'InheritanceLink',
            outgoing: [node.id, connId],
            truthValue: {
              strength: (node.activation + targetNode.activation) / 2,
              confidence: 0.7,
              count: 1
            },
            attentionValue: {
              sti: ((node.attentionValue || 0) + (targetNode.attentionValue || 0)) * 50,
              lti: 25,
              vlti: false
            },
            strength: (node.activation + targetNode.activation) / 2
          });
        }
      });
    });

    const initialAtomSpace: EnhancedAtomSpace = {
      atoms,
      links,
      attentionBank: {
        totalSTI: atoms.reduce((sum, atom) => sum + atom.attentionValue.sti, 0),
        totalLTI: atoms.reduce((sum, atom) => sum + atom.attentionValue.lti, 0),
        funds: 1000,
        rentCollected: 0,
        attentionFocus: [],
        forgettingThreshold: 0.1
      },
      plnEngine: {
        rules: plnRules,
        inferences: [],
        truthValueSystem,
        uncertaintyHandling: {
          handleInconsistency: (conflictingAtoms) => {
            console.log('Handling inconsistency:', conflictingAtoms);
          },
          computeEntropy: (atoms) => {
            return atoms.reduce((sum, atom) => 
              sum - atom.truthValue.confidence * Math.log2(atom.truthValue.confidence + 1e-10), 0
            ) / atoms.length;
          },
          bayesianUpdate: (prior, evidence) => {
            const newStrength = (prior.strength * prior.confidence + evidence.strength * evidence.confidence) / 
                              (prior.confidence + evidence.confidence);
            const newConfidence = Math.min(1, prior.confidence + evidence.confidence);
            const newCount = prior.count + evidence.count;
            return { strength: newStrength, confidence: newConfidence, count: newCount };
          }
        }
      },
      patternMatcher: {
        patterns: [],
        matches: [],
        bindLink: (pattern, atomSpace) => performPatternMatching(atomSpace)
      },
      totalAtoms: atoms.length,
      attentionSpread: []
    };

    setAtomSpace(initialAtomSpace);
    setIsActive(true);
  }, [createTruthValueSystem, createPLNRules, performPatternMatching]);

  // Update AtomSpace with ECAN and PLN cycles
  const updateAtomSpace = useCallback(() => {
    if (!atomSpace || !isActive) return;

    attentionCycleRef.current += 1;

    setAtomSpace(prevAtomSpace => {
      if (!prevAtomSpace) return null;

      // Perform attention allocation cycle
      const newAttentionBank = performAttentionAllocation(prevAtomSpace);
      
      // Perform attention spreading
      const newAttentionSpread = performAttentionSpreading(prevAtomSpace);
      
      // Perform pattern matching
      const newMatches = performPatternMatching(prevAtomSpace);
      
      // Apply PLN inferences
      const newInferences: PLNInference[] = [];
      prevAtomSpace.plnEngine.rules.forEach(rule => {
        if (rule.applicability(prevAtomSpace.atoms, prevAtomSpace.links)) {
          const applicablePremises = prevAtomSpace.atoms.slice(0, 3).map(a => a.name); // Simplified
          const inference = rule.apply(applicablePremises);
          newInferences.push(inference);
        }
      });

      return {
        ...prevAtomSpace,
        attentionBank: newAttentionBank,
        attentionSpread: [...prevAtomSpace.attentionSpread, ...newAttentionSpread].slice(-100),
        patternMatcher: {
          ...prevAtomSpace.patternMatcher,
          matches: [...prevAtomSpace.patternMatcher.matches, ...newMatches].slice(-50)
        },
        plnEngine: {
          ...prevAtomSpace.plnEngine,
          inferences: [...prevAtomSpace.plnEngine.inferences, ...newInferences].slice(-100)
        }
      };
    });
  }, [atomSpace, isActive, performAttentionAllocation, performAttentionSpreading, performPatternMatching]);

  // Auto-update cycle
  useEffect(() => {
    if (isActive) {
      const interval = setInterval(updateAtomSpace, 200); // 200ms cycle for ECAN/PLN
      return () => clearInterval(interval);
    }
  }, [isActive, updateAtomSpace]);

  const stopAtomSpace = useCallback(() => {
    setIsActive(false);
    setAtomSpace(null);
  }, []);

  // Convert to legacy format for compatibility
  const getLegacyConnections = useCallback((): AtomSpaceConnection[] => {
    if (!atomSpace) return [];
    
    return atomSpace.links.map(link => ({
      id: link.id,
      source: link.outgoing[0],
      target: link.outgoing[1],
      type: link.type === 'InheritanceLink' ? 'inheritance' : 
            link.type === 'SimilarityLink' ? 'similarity' : 
            link.type === 'ImplicationLink' ? 'implication' : 'equivalence',
      strength: link.strength,
      confidence: link.truthValue.confidence,
      attentionValue: link.attentionValue.sti
    }));
  }, [atomSpace]);

  const getLegacyPLNRules = useCallback((): PLNRule[] => {
    if (!atomSpace) return [];
    
    return atomSpace.plnEngine.inferences.map(inference => ({
      id: inference.id,
      type: inference.rule.includes('inheritance') ? 'inheritance' : 
            inference.rule.includes('similarity') ? 'similarity' : 
            inference.rule.includes('implication') ? 'implication' : 'deduction',
      premise: inference.premises,
      conclusion: inference.conclusion,
      strength: inference.truthValue.strength,
      confidence: inference.truthValue.confidence,
      truthValue: inference.confidence
    }));
  }, [atomSpace]);

  return {
    atomSpace,
    isActive,
    initializeAtomSpace,
    updateAtomSpace,
    stopAtomSpace,
    getLegacyConnections,
    getLegacyPLNRules,
    attentionCycle: attentionCycleRef.current,
    inferenceCount: inferenceCountRef.current
  };
};