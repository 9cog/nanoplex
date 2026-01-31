/**
 * AtomSpace Tensor Encoding Subsystem
 * 
 * This module implements routines to encode hypergraph nodes and links as ggml-compatible tensors,
 * enabling neural-symbolic integration within the cognitive kernel.
 */

import { GgmlTensorKernel, GgmlTensor, NodeTensor, LinkTensor, AtomSpaceTensorConfig } from './GgmlTensorKernel';
import { AtomSpaceAtom, AtomSpaceLink, TruthValue, AttentionValue } from '../hooks/useEnhancedAtomSpace';
import { AtomeseNode } from '../types';

/**
 * AtomSpace tensor encoding/decoding engine
 */
export class AtomSpaceTensorEncoder {
  private kernel: GgmlTensorKernel;
  private config: AtomSpaceTensorConfig;
  private nodeEmbeddings: Map<string, NodeTensor>;
  private linkEmbeddings: Map<string, LinkTensor>;
  private vocabularyMap: Map<string, number>;
  private reverseVocabularyMap: Map<number, string>;
  private nextVocabId: number;

  constructor(config: AtomSpaceTensorConfig) {
    this.config = config;
    this.kernel = new GgmlTensorKernel(config);
    this.nodeEmbeddings = new Map();
    this.linkEmbeddings = new Map();
    this.vocabularyMap = new Map();
    this.reverseVocabularyMap = new Map();
    this.nextVocabId = 0;
    
    this.initializeVocabulary();
  }

  /**
   * Initialize vocabulary with common atom types
   */
  private initializeVocabulary(): void {
    const atomTypes = [
      'ConceptNode', 'PredicateNode', 'NumberNode', 'VariableNode',
      'InheritanceLink', 'SimilarityLink', 'ImplicationLink', 'EvaluationLink',
      'ListLink', 'SetLink', 'AndLink', 'OrLink', 'NotLink'
    ];

    atomTypes.forEach(type => {
      this.addToVocabulary(type);
    });
  }

  /**
   * Add symbol to vocabulary
   */
  private addToVocabulary(symbol: string): number {
    if (!this.vocabularyMap.has(symbol)) {
      const id = this.nextVocabId++;
      this.vocabularyMap.set(symbol, id);
      this.reverseVocabularyMap.set(id, symbol);
    }
    return this.vocabularyMap.get(symbol)!;
  }

  /**
   * Encode AtomSpace atom as tensor
   */
  encodeAtom(atom: AtomSpaceAtom): NodeTensor {
    const existingTensor = this.nodeEmbeddings.get(atom.id);
    if (existingTensor) {
      return existingTensor;
    }

    // Create base embedding tensor
    const embedding = this.kernel.createTensor(
      [this.config.node_embedding_dim], 
      'f32', 
      `node_${atom.id}`,
      true
    );

    // Encode atom type
    const typeId = this.addToVocabulary(atom.type);
    embedding.data[0] = typeId / 100.0; // Normalize type ID

    // Encode atom name if available
    if (atom.name) {
      const nameHash = this.hashString(atom.name);
      embedding.data[1] = (nameHash % 1000) / 1000.0; // Normalize hash
    }

    // Encode truth value
    const truthTensor = this.encodeTruthValue(atom.truthValue);
    
    // Encode attention value
    const attentionTensor = this.encodeAttentionValue(atom.attentionValue);

    // Create symbolic features tensor
    const symbolicFeatures = this.createSymbolicFeatures(atom);

    // Calculate symbolic depth based on atom complexity
    const symbolicDepth = this.calculateSymbolicDepth(atom);

    const nodeTensor: NodeTensor = {
      id: `node_tensor_${atom.id}`,
      atom_id: atom.id,
      embedding,
      attention_weights: attentionTensor,
      truth_value_tensor: truthTensor,
      symbolic_features: symbolicFeatures,
      metadata: {
        atom_type: atom.type,
        symbolic_depth: symbolicDepth,
        degree_of_freedom: this.calculateDegreesOfFreedom(atom)
      }
    };

    this.nodeEmbeddings.set(atom.id, nodeTensor);
    return nodeTensor;
  }

  /**
   * Encode AtomSpace link as tensor
   */
  encodeLink(link: AtomSpaceLink): LinkTensor {
    const existingTensor = this.linkEmbeddings.get(link.id);
    if (existingTensor) {
      return existingTensor;
    }

    // Create relation tensor based on link type and outgoing atoms
    const relationTensor = this.kernel.createTensor(
      [this.config.link_embedding_dim],
      'f32',
      `link_${link.id}`,
      true
    );

    // Encode link type
    const typeId = this.addToVocabulary(link.type);
    relationTensor.data[0] = typeId / 100.0;

    // Encode arity (number of outgoing atoms)
    relationTensor.data[1] = link.outgoing.length / 10.0;

    // Encode outgoing atom relationships
    for (let i = 0; i < Math.min(link.outgoing.length, 5); i++) {
      const atomId = link.outgoing[i];
      const atomHash = this.hashString(atomId);
      relationTensor.data[2 + i] = (atomHash % 1000) / 1000.0;
    }

    // Encode truth value and attention
    const truthTensor = this.encodeTruthValue(link.truthValue);
    const attentionTensor = this.encodeAttentionValue(link.attentionValue);

    const linkTensor: LinkTensor = {
      id: `link_tensor_${link.id}`,
      atom_id: link.id,
      source_nodes: link.outgoing.slice(0, -1),
      target_nodes: [link.outgoing[link.outgoing.length - 1]],
      relation_tensor: relationTensor,
      attention_weights: attentionTensor,
      truth_value_tensor: truthTensor,
      metadata: {
        link_type: link.type,
        arity: link.outgoing.length,
        symbolic_depth: this.calculateLinkSymbolicDepth(link)
      }
    };

    this.linkEmbeddings.set(link.id, linkTensor);
    return linkTensor;
  }

  /**
   * Decode tensor back to AtomSpace atom
   */
  decodeAtom(nodeTensor: NodeTensor): AtomSpaceAtom {
    const typeId = Math.round(nodeTensor.embedding.data[0] * 100);
    const atomType = this.reverseVocabularyMap.get(typeId) || 'ConceptNode';

    const truthValue = this.decodeTruthValue(nodeTensor.truth_value_tensor);
    const attentionValue = this.decodeAttentionValue(nodeTensor.attention_weights);

    return {
      id: nodeTensor.atom_id,
      type: atomType as 'ConceptNode' | 'PredicateNode' | 'NumberNode' | 'VariableNode',
      name: `decoded_${nodeTensor.atom_id}`,
      truthValue,
      attentionValue,
      importance: truthValue.strength * truthValue.confidence,
      timestamp: Date.now()
    };
  }

  /**
   * Decode tensor back to AtomSpace link
   */
  decodeLink(linkTensor: LinkTensor): AtomSpaceLink {
    const typeId = Math.round(linkTensor.relation_tensor.data[0] * 100);
    const linkType = this.reverseVocabularyMap.get(typeId) || 'InheritanceLink';

    const truthValue = this.decodeTruthValue(linkTensor.truth_value_tensor);
    const attentionValue = this.decodeAttentionValue(linkTensor.attention_weights);

    return {
      id: linkTensor.atom_id,
      type: linkType as 'InheritanceLink' | 'SimilarityLink' | 'ImplicationLink' | 'EvaluationLink' | 'ListLink',
      outgoing: [...linkTensor.source_nodes, ...linkTensor.target_nodes],
      truthValue,
      attentionValue,
      strength: truthValue.strength
    };
  }

  /**
   * Encode Atomese node to tensor
   */
  encodeAtomeseNode(node: AtomeseNode): NodeTensor {
    const embedding = this.kernel.createTensor(
      [this.config.node_embedding_dim],
      'f32',
      `atomese_${node.id}`,
      true
    );

    // Encode node type
    const typeId = this.addToVocabulary(node.type);
    embedding.data[0] = typeId / 100.0;

    // Encode node name
    if (node.name) {
      const nameHash = this.hashString(node.name);
      embedding.data[1] = (nameHash % 1000) / 1000.0;
    }

    // Encode truth value if available
    const truthTensor = node.truthValue ? 
      this.encodeTruthValue(node.truthValue) : 
      this.kernel.createTensor([3], 'f32');

    // Encode attention value if available
    const attentionData = new Float32Array(3);
    if (node.attentionValue) {
      attentionData[0] = node.attentionValue;
      attentionData[1] = node.attentionValue * 0.8; // LTI approximation
      attentionData[2] = node.attentionValue > 0.9 ? 1.0 : 0.0; // VLTI
    }
    const attentionTensor = this.kernel.createTensor([3], 'f32');
    attentionTensor.data.set(attentionData);

    // Create symbolic features
    const symbolicFeatures = this.createAtomeseSymbolicFeatures(node);

    const nodeTensor: NodeTensor = {
      id: `atomese_tensor_${node.id}`,
      atom_id: node.id,
      embedding,
      attention_weights: attentionTensor,
      truth_value_tensor: truthTensor,
      symbolic_features: symbolicFeatures,
      metadata: {
        atom_type: node.type,
        symbolic_depth: this.calculateAtomeseSymbolicDepth(node),
        degree_of_freedom: node.children ? node.children.length : 1
      }
    };

    this.nodeEmbeddings.set(node.id, nodeTensor);
    return nodeTensor;
  }

  /**
   * Encode truth value as tensor
   */
  private encodeTruthValue(truthValue: TruthValue): GgmlTensor {
    const tensor = this.kernel.createTensor([3], 'f32');
    tensor.data[0] = truthValue.strength;
    tensor.data[1] = truthValue.confidence;
    tensor.data[2] = Math.log(truthValue.count + 1) / 10.0; // Log-normalized count
    return tensor;
  }

  /**
   * Encode attention value as tensor
   */
  private encodeAttentionValue(attentionValue: AttentionValue): GgmlTensor {
    const tensor = this.kernel.createTensor([3], 'f32');
    tensor.data[0] = attentionValue.sti / 100.0; // Normalize STI
    tensor.data[1] = attentionValue.lti / 100.0; // Normalize LTI
    tensor.data[2] = attentionValue.vlti ? 1.0 : 0.0; // Boolean VLTI
    return tensor;
  }

  /**
   * Decode truth value from tensor
   */
  private decodeTruthValue(tensor: GgmlTensor): TruthValue {
    return {
      strength: tensor.data[0],
      confidence: tensor.data[1],
      count: Math.exp(tensor.data[2] * 10.0) - 1
    };
  }

  /**
   * Decode attention value from tensor
   */
  private decodeAttentionValue(tensor: GgmlTensor): AttentionValue {
    return {
      sti: tensor.data[0] * 100.0,
      lti: tensor.data[1] * 100.0,
      vlti: tensor.data[2] > 0.5
    };
  }

  /**
   * Create symbolic features tensor for atom
   */
  private createSymbolicFeatures(atom: AtomSpaceAtom): GgmlTensor {
    const features = this.kernel.createTensor([this.config.node_embedding_dim], 'f32');
    
    // Feature 0: Type complexity
    features.data[0] = atom.type.length / 20.0;
    
    // Feature 1: Name complexity
    if (atom.name) {
      features.data[1] = atom.name.length / 50.0;
    }
    
    // Feature 2: Importance
    features.data[2] = atom.importance;
    
    // Feature 3: Timestamp recency
    features.data[3] = Math.min(1.0, (Date.now() - atom.timestamp) / (1000 * 60 * 60 * 24));

    return features;
  }

  /**
   * Create symbolic features for Atomese node
   */
  private createAtomeseSymbolicFeatures(node: AtomeseNode): GgmlTensor {
    const features = this.kernel.createTensor([this.config.node_embedding_dim], 'f32');
    
    // Feature 0: Type complexity
    features.data[0] = node.type.length / 20.0;
    
    // Feature 1: Name complexity
    if (node.name) {
      features.data[1] = node.name.length / 50.0;
    }
    
    // Feature 2: Attention value
    features.data[2] = node.attentionValue || 0.0;
    
    // Feature 3: Truth value strength
    features.data[3] = node.truthValue?.strength || 0.0;
    
    // Feature 4: Children count
    features.data[4] = node.children ? node.children.length / 10.0 : 0.0;

    return features;
  }

  /**
   * Calculate symbolic depth for atom
   */
  private calculateSymbolicDepth(atom: AtomSpaceAtom): number {
    let depth = 1;
    
    if (atom.name) {
      depth += Math.min(3, atom.name.split('-').length);
    }
    
    if (atom.type.includes('Link')) {
      depth += 1;
    }
    
    return depth;
  }

  /**
   * Calculate symbolic depth for link
   */
  private calculateLinkSymbolicDepth(link: AtomSpaceLink): number {
    return 1 + Math.min(3, link.outgoing.length);
  }

  /**
   * Calculate symbolic depth for Atomese node
   */
  private calculateAtomeseSymbolicDepth(node: AtomeseNode): number {
    let depth = 1;
    
    if (node.children) {
      depth += Math.min(3, node.children.length);
    }
    
    if (node.type.includes('Link')) {
      depth += 1;
    }
    
    return depth;
  }

  /**
   * Calculate degrees of freedom for atom
   */
  private calculateDegreesOfFreedom(atom: AtomSpaceAtom): number {
    let dof = 1;
    
    // Truth value adds 2 DOF (strength, confidence)
    dof += 2;
    
    // Attention value adds 3 DOF (STI, LTI, VLTI)
    dof += 3;
    
    // Type complexity
    if (atom.type.includes('Link')) {
      dof += 2;
    }
    
    return dof;
  }

  /**
   * Simple string hash function
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Get all encoded node tensors
   */
  getNodeTensors(): NodeTensor[] {
    return Array.from(this.nodeEmbeddings.values());
  }

  /**
   * Get all encoded link tensors
   */
  getLinkTensors(): LinkTensor[] {
    return Array.from(this.linkEmbeddings.values());
  }

  /**
   * Get vocabulary statistics
   */
  getVocabularyStats(): { size: number; symbols: string[] } {
    return {
      size: this.vocabularyMap.size,
      symbols: Array.from(this.vocabularyMap.keys())
    };
  }

  /**
   * Clear all cached tensors
   */
  clearCache(): void {
    this.nodeEmbeddings.clear();
    this.linkEmbeddings.clear();
  }

  /**
   * Get tensor kernel for direct operations
   */
  getKernel(): GgmlTensorKernel {
    return this.kernel;
  }
}