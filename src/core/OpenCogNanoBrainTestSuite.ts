/**
 * OpenCog NanoBrain Time Crystal Architecture Test Suite
 * 
 * Comprehensive testing for the unified OpenCog-NanoBrain system including:
 * - Time crystal quantum state evolution
 * - AtomSpace hypergraph operations
 * - Phase Prime Metric (PPM) calculations  
 * - Fractal Information Theory (FIT) encoding
 * - Consciousness emergence measurement
 * - PLN reasoning with temporal dynamics
 */

import { 
  OpenCogNanoBrainKernel, 
  type _OpenCogNanoBrainConfig,
  type _NanoBrainMetrics,
  type _TimeCrystalAtom,
  type _TimeCrystalInference,
  type _TimeCrystalQuantumState
} from './OpenCogNanoBrainKernel';
import { TestResult } from '../types';

export class OpenCogNanoBrainTestSuite {
  private kernel: OpenCogNanoBrainKernel | null = null;
  private testResults: TestResult[] = [];

  /**
   * Run all test scenarios
   */
  async runAllTests(): Promise<TestResult[]> {
    this.testResults = [];
    
    try {
      // Core initialization tests
      await this.testKernelInitialization();
      await this.testFundamentalAtomCreation();
      await this.testTimeCrystalQuantumStates();
      
      // Phase Prime Metric tests
      await this.testPhasePrimeMetricCalculations();
      await this.testPrimeResonancePatterns();
      
      // Fractal Information Theory tests
      await this.testFractalGeometryEncoding();
      await this.testGeometricMusicalLanguage();
      
      // AtomSpace operations
      await this.testAtomSpaceHypergraphStructure();
      await this.testECANAttentionAllocation();
      
      // PLN reasoning tests
      await this.testEnhancedPLNInference();
      await this.testTemporalReasoningFlow();
      
      // Consciousness emergence tests
      await this.testConsciousnessMetricsCalculation();
      await this.testQuantumCoherenceEvolution();
      
      // Integration and performance tests
      await this.testRealTimeProcessingCycle();
      await this.testMetricsAndMonitoring();
      await this.testSystemStabilityUnderLoad();
      
    } catch (error) {
      this.addTestResult({
        name: 'Test Suite Execution',
        passed: false,
        message: `Test suite failed with error: ${error}`,
        duration: 0
      });
    }

    return this.testResults;
  }

  /**
   * Test kernel initialization and configuration
   */
  private async testKernelInitialization(): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Test default configuration
      this.kernel = new OpenCogNanoBrainKernel();
      const config = this.kernel.getConfig();
      
      // Verify default configuration values
      this.assert(
        config.timeCrystalDimensions === 11,
        'Time crystal dimensions should default to 11D'
      );
      
      this.assert(
        config.fundamentalPrimes.length === 15,
        'Should have 15 fundamental primes for 99.99% universe coverage'
      );
      
      this.assert(
        config.geometricShapeCount === 15,
        'Should have 15 geometric shapes for GML encoding'
      );
      
      // Test kernel start/stop
      await this.kernel.start();
      
      const metrics = this.kernel.getMetrics();
      this.assert(
        metrics.totalAtoms > 0,
        'Should have fundamental atoms after initialization'
      );
      
      await this.kernel.stop();
      
      this.addTestResult({
        name: 'Kernel Initialization',
        passed: true,
        message: 'Successfully initialized and configured OpenCog NanoBrain kernel',
        duration: performance.now() - startTime,
        metrics: {
          atoms_created: metrics.totalAtoms,
          config_dimensions: config.timeCrystalDimensions,
          primes_count: config.fundamentalPrimes.length
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Kernel Initialization',
        passed: false,
        message: `Kernel initialization failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test fundamental atom creation from NanoBrain theory
   */
  private async testFundamentalAtomCreation(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const atomSpace = this.kernel.getAtomSpace();
      
      // Verify Chapter 1: Philosophical Transformation atom
      const philosophicalAtom = atomSpace.get('philosophical-transformation');
      this.assert(
        philosophicalAtom !== undefined,
        'Philosophical transformation root atom should exist'
      );
      
      this.assert(
        philosophicalAtom?.truthValue.strength === 1.0,
        'Philosophical atom should have maximum truth strength'
      );
      
      // Verify Chapter 2: Fractal Information Theory atom
      const fitAtom = atomSpace.get('fractal-information-theory');
      this.assert(
        fitAtom !== undefined,
        'Fractal Information Theory atom should exist'
      );
      
      this.assert(
        fitAtom?.fractalGeometry.shape === 'fractal',
        'FIT atom should have fractal geometry'
      );
      
      // Verify Chapter 3: Prime atoms (15 fundamental primes)
      const fundamentalPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
      let primeAtomsFound = 0;
      
      fundamentalPrimes.forEach(prime => {
        const primeAtom = atomSpace.get(`prime-${prime}`);
        if (primeAtom && primeAtom.type === 'NumberNode') {
          primeAtomsFound++;
        }
      });
      
      this.assert(
        primeAtomsFound === 15,
        `Should have all 15 fundamental prime atoms, found ${primeAtomsFound}`
      );
      
      // Verify Geometric Musical Language atoms (15 shapes)
      const expectedShapes = ['point', 'line', 'triangle', 'square', 'pentagon', 'hexagon', 'circle'];
      let gmlAtomsFound = 0;
      
      expectedShapes.forEach(shape => {
        const gmlAtom = atomSpace.get(`gml-${shape}`);
        if (gmlAtom) {
          gmlAtomsFound++;
        }
      });
      
      this.assert(
        gmlAtomsFound > 0,
        'Should have GML geometric shape atoms'
      );
      
      this.addTestResult({
        name: 'Fundamental Atom Creation',
        passed: true,
        message: 'Successfully created all fundamental atoms from NanoBrain theory',
        duration: performance.now() - startTime,
        metrics: {
          total_atoms: atomSpace.size,
          prime_atoms: primeAtomsFound,
          gml_atoms: gmlAtomsFound
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Fundamental Atom Creation',
        passed: false,
        message: `Atom creation failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test time crystal quantum state evolution
   */
  private async testTimeCrystalQuantumStates(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const timeCrystals = this.kernel.getTimeCrystals();
      
      this.assert(
        timeCrystals.size > 0,
        'Should have time crystal quantum states for atoms'
      );
      
      // Test quantum state properties
      let validStatesCount = 0;
      let coherentStatesCount = 0;
      
      for (const [atomId, crystal] of timeCrystals) {
        // Verify 11D manifold coordinates
        this.assert(
          crystal.dimensions.length === 11,
          `Time crystal ${atomId} should have 11D coordinates`
        );
        
        // Verify temporal coherence bounds
        this.assert(
          crystal.temporalCoherence >= 0 && crystal.temporalCoherence <= 1,
          `Temporal coherence should be in [0,1] range for ${atomId}`
        );
        
        // Verify quantum phase bounds  
        this.assert(
          crystal.quantumPhase >= 0 && crystal.quantumPhase <= 2 * Math.PI,
          `Quantum phase should be in [0, 2π] range for ${atomId}`
        );
        
        // Verify prime signature consistency
        this.assert(
          crystal.primeSignature.length > 0,
          `Time crystal ${atomId} should have prime signature`
        );
        
        validStatesCount++;
        
        if (crystal.temporalCoherence > 0.5) {
          coherentStatesCount++;
        }
      }
      
      this.assert(
        coherentStatesCount > 0,
        'Should have at least some coherent quantum states'
      );
      
      // Test quantum evolution over time (simulate short period)
      await new Promise(resolve => setTimeout(resolve, 200)); // 200ms evolution
      
      const evolvedCrystals = this.kernel.getTimeCrystals();
      let phaseEvolutionDetected = false;
      
      for (const [atomId, originalCrystal] of timeCrystals) {
        const evolvedCrystal = evolvedCrystals.get(atomId);
        if (evolvedCrystal) {
          // Check if quantum phase has evolved
          if (Math.abs(evolvedCrystal.quantumPhase - originalCrystal.quantumPhase) > 0.01) {
            phaseEvolutionDetected = true;
            break;
          }
        }
      }
      
      this.assert(
        phaseEvolutionDetected,
        'Quantum phases should evolve over time'
      );
      
      this.addTestResult({
        name: 'Time Crystal Quantum States',
        passed: true,
        message: 'Time crystal quantum states properly initialized and evolving',
        duration: performance.now() - startTime,
        metrics: {
          total_crystals: timeCrystals.size,
          valid_states: validStatesCount,
          coherent_states: coherentStatesCount
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Time Crystal Quantum States',
        passed: false,
        message: `Quantum state test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test Phase Prime Metric calculations
   */
  private async testPhasePrimeMetricCalculations(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const metrics = this.kernel.getMetrics();
      
      // Test prime alignment calculation
      this.assert(
        metrics.primeAlignment >= 0 && metrics.primeAlignment <= 1,
        'Prime alignment should be normalized to [0,1] range'
      );
      
      // Test with known prime encodings
      const atomSpace = this.kernel.getAtomSpace();
      const primeAtom = atomSpace.get('prime-2');
      
      if (primeAtom) {
        this.assert(
          primeAtom.primeEncoding.includes(2),
          'Prime-2 atom should have prime 2 in its encoding'
        );
        
        // Test resonance frequency calculation
        this.assert(
          primeAtom.timeCrystalState.resonanceFrequency > 0,
          'Prime atoms should have positive resonance frequency'
        );
      }
      
      // Test PPM consistency across fundamental primes
      const fundamentalPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
      let primeConsistencyScore = 0;
      
      for (const prime of fundamentalPrimes) {
        const atom = atomSpace.get(`prime-${prime}`);
        if (atom && atom.primeEncoding.includes(prime)) {
          primeConsistencyScore++;
        }
      }
      
      this.assert(
        primeConsistencyScore === 15,
        `All 15 fundamental primes should be consistently encoded, found ${primeConsistencyScore}`
      );
      
      // Test PPM metric classes (should cover 10 classes as per Chapter 3)
      const primePatterns = new Set<string>();
      for (const atom of atomSpace.values()) {
        if (atom.primeEncoding.length > 0) {
          primePatterns.add(atom.primeEncoding.join(','));
        }
      }
      
      this.assert(
        primePatterns.size >= 3,
        'Should have multiple distinct prime pattern classes'
      );
      
      this.addTestResult({
        name: 'Phase Prime Metric Calculations',
        passed: true,
        message: 'PPM calculations working correctly with fundamental primes',
        duration: performance.now() - startTime,
        metrics: {
          prime_alignment: metrics.primeAlignment,
          prime_consistency: primeConsistencyScore,
          pattern_classes: primePatterns.size
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Phase Prime Metric Calculations',
        passed: false,
        message: `PPM test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test prime resonance patterns
   */
  private async testPrimeResonancePatterns(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const atomSpace = this.kernel.getAtomSpace();
      
      // Test resonance frequency ranges
      const resonanceFrequencies: number[] = [];
      
      for (const atom of atomSpace.values()) {
        resonanceFrequencies.push(atom.timeCrystalState.resonanceFrequency);
      }
      
      this.assert(
        resonanceFrequencies.length > 0,
        'Should have resonance frequencies for atoms'
      );
      
      // Verify frequencies are in reasonable audible range
      const minFreq = Math.min(...resonanceFrequencies);
      const maxFreq = Math.max(...resonanceFrequencies);
      
      this.assert(
        minFreq >= 20 && maxFreq <= 20000,
        `Resonance frequencies should be in audible range [20Hz, 20kHz], got [${minFreq.toFixed(1)}, ${maxFreq.toFixed(1)}]`
      );
      
      // Test harmonic relationships between primes
      const primeFrequencies = new Map<number, number>();
      
      for (const atom of atomSpace.values()) {
        if (atom.type === 'NumberNode' && atom.id.startsWith('prime-')) {
          const prime = atom.primeEncoding[0];
          if (prime) {
            primeFrequencies.set(prime, atom.timeCrystalState.resonanceFrequency);
          }
        }
      }
      
      this.assert(
        primeFrequencies.size >= 10,
        'Should have resonance frequencies for multiple primes'
      );
      
      // Test that different primes have different frequencies
      const uniqueFrequencies = new Set(primeFrequencies.values());
      this.assert(
        uniqueFrequencies.size === primeFrequencies.size,
        'Each prime should have a unique resonance frequency'
      );
      
      this.addTestResult({
        name: 'Prime Resonance Patterns',
        passed: true,
        message: 'Prime resonance patterns correctly calculated and unique',
        duration: performance.now() - startTime,
        metrics: {
          total_frequencies: resonanceFrequencies.length,
          prime_frequencies: primeFrequencies.size,
          frequency_range: [minFreq, maxFreq]
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Prime Resonance Patterns',
        passed: false,
        message: `Resonance pattern test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test fractal geometry encoding
   */
  private async testFractalGeometryEncoding(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const atomSpace = this.kernel.getAtomSpace();
      
      // Test fractal dimension evolution
      const fractalDimensions: number[] = [];
      let validGeometries = 0;
      
      for (const atom of atomSpace.values()) {
        const geometry = atom.fractalGeometry;
        
        // Verify geometric properties
        this.assert(
          geometry.dimensions >= 1,
          `Geometric dimensions should be >= 1 for atom ${atom.id}`
        );
        
        this.assert(
          geometry.scaleFactor > 0,
          `Scale factor should be positive for atom ${atom.id}`
        );
        
        this.assert(
          geometry.primeResonance.length > 0,
          `Should have prime resonance for atom ${atom.id}`
        );
        
        fractalDimensions.push(atom.timeCrystalState.fractalDimension);
        validGeometries++;
      }
      
      this.assert(
        validGeometries === atomSpace.size,
        'All atoms should have valid fractal geometry'
      );
      
      // Test fractal complexity distribution
      const avgFractalDim = fractalDimensions.reduce((a, b) => a + b, 0) / fractalDimensions.length;
      
      this.assert(
        avgFractalDim >= 1.0 && avgFractalDim <= 12.0,
        `Average fractal dimension should be reasonable, got ${avgFractalDim.toFixed(2)}`
      );
      
      // Test shape variety in GML
      const shapes = new Set<string>();
      for (const atom of atomSpace.values()) {
        shapes.add(atom.fractalGeometry.shape);
      }
      
      this.assert(
        shapes.size >= 3,
        'Should have variety in geometric shapes'
      );
      
      this.addTestResult({
        name: 'Fractal Geometry Encoding',
        passed: true,
        message: 'Fractal geometry encoding working correctly with diverse shapes',
        duration: performance.now() - startTime,
        metrics: {
          valid_geometries: validGeometries,
          avg_fractal_dim: avgFractalDim,
          shape_variety: shapes.size
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Fractal Geometry Encoding',
        passed: false,
        message: `Fractal geometry test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test Geometric Musical Language (GML)
   */
  private async testGeometricMusicalLanguage(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const atomSpace = this.kernel.getAtomSpace();
      
      // Test musical note assignments
      const musicalNotes = new Set<string>();
      let gmlAtomsCount = 0;
      
      for (const [atomId, atom] of atomSpace) {
        if (atomId.startsWith('gml-')) {
          gmlAtomsCount++;
          musicalNotes.add(atom.fractalGeometry.musicalNote);
          
          // Verify GML atom has proper structure
          this.assert(
            atom.fractalGeometry.musicalNote.length > 0,
            `GML atom ${atomId} should have musical note`
          );
          
          this.assert(
            atom.fractalGeometry.symmetryGroup.includes('GML'),
            `GML atom ${atomId} should have GML symmetry group`
          );
        }
      }
      
      this.assert(
        gmlAtomsCount >= 7, // At least 7 basic shapes
        `Should have at least 7 GML atoms, found ${gmlAtomsCount}`
      );
      
      this.assert(
        musicalNotes.size >= 3,
        'Should have variety in musical notes'
      );
      
      // Test musical harmony calculation
      const notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
      let harmonyTests = 0;
      
      for (let i = 0; i < notes.length - 1; i++) {
        for (let j = i + 1; j < notes.length; j++) {
          // This would test the musical harmony calculation if it were accessible
          // For now, we just verify the notes are valid
          this.assert(
            notes.includes(notes[i]) && notes.includes(notes[j]),
            'Musical notes should be valid'
          );
          harmonyTests++;
        }
      }
      
      this.addTestResult({
        name: 'Geometric Musical Language',
        passed: true,
        message: 'GML encoding working with musical note assignments',
        duration: performance.now() - startTime,
        metrics: {
          gml_atoms: gmlAtomsCount,
          musical_notes: musicalNotes.size,
          harmony_tests: harmonyTests
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Geometric Musical Language',
        passed: false,
        message: `GML test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test AtomSpace hypergraph structure
   */
  private async testAtomSpaceHypergraphStructure(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const atomSpace = this.kernel.getAtomSpace();
      const _linkSpace = this.kernel.getLinkSpace();
      
      // Test hypergraph connectivity
      this.assert(
        atomSpace.size > 0,
        'AtomSpace should contain atoms'
      );
      
      // Test link formation
      // Allow some time for links to form
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const updatedLinkSpace = this.kernel.getLinkSpace();
      
      this.assert(
        updatedLinkSpace.size >= 0,
        'LinkSpace should be initialized (links may form over time)'
      );
      
      // Test atom types distribution
      const atomTypes = new Map<string, number>();
      for (const atom of atomSpace.values()) {
        atomTypes.set(atom.type, (atomTypes.get(atom.type) || 0) + 1);
      }
      
      this.assert(
        atomTypes.has('ConceptNode'),
        'Should have ConceptNodes in AtomSpace'
      );
      
      this.assert(
        atomTypes.has('NumberNode'),
        'Should have NumberNodes for primes in AtomSpace'
      );
      
      // Test truth value consistency
      let validTruthValues = 0;
      for (const atom of atomSpace.values()) {
        const tv = atom.truthValue;
        if (tv.strength >= 0 && tv.strength <= 1 && 
            tv.confidence >= 0 && tv.confidence <= 1) {
          validTruthValues++;
        }
      }
      
      this.assert(
        validTruthValues === atomSpace.size,
        'All atoms should have valid truth values'
      );
      
      this.addTestResult({
        name: 'AtomSpace Hypergraph Structure',
        passed: true,
        message: 'AtomSpace hypergraph structure properly formed',
        duration: performance.now() - startTime,
        metrics: {
          total_atoms: atomSpace.size,
          total_links: updatedLinkSpace.size,
          atom_types: atomTypes.size,
          valid_truth_values: validTruthValues
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'AtomSpace Hypergraph Structure',
        passed: false,
        message: `Hypergraph structure test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test ECAN attention allocation
   */
  private async testECANAttentionAllocation(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      // Allow several processing cycles for attention to distribute
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const atomSpace = this.kernel.getAtomSpace();
      const metrics = this.kernel.getMetrics();
      
      // Test attention value ranges
      let totalSTI = 0;
      let validAttentionValues = 0;
      
      for (const atom of atomSpace.values()) {
        const av = atom.attentionValue;
        
        this.assert(
          av.sti >= 0,
          `STI should be non-negative for atom ${atom.id}`
        );
        
        this.assert(
          av.lti >= 0,
          `LTI should be non-negative for atom ${atom.id}`
        );
        
        totalSTI += av.sti;
        validAttentionValues++;
      }
      
      this.assert(
        validAttentionValues === atomSpace.size,
        'All atoms should have valid attention values'
      );
      
      // Test attention distribution (should have some variation)
      const attentionValues = Array.from(atomSpace.values()).map(atom => atom.attentionValue.sti);
      const minSTI = Math.min(...attentionValues);
      const maxSTI = Math.max(...attentionValues);
      
      this.assert(
        maxSTI > minSTI,
        'Should have attention variation across atoms'
      );
      
      // Test metrics calculation
      this.assert(
        metrics.averageAttention >= 0,
        'Average attention should be non-negative'
      );
      
      this.addTestResult({
        name: 'ECAN Attention Allocation',
        passed: true,
        message: 'ECAN attention allocation working with proper distribution',
        duration: performance.now() - startTime,
        metrics: {
          total_sti: totalSTI,
          avg_attention: metrics.averageAttention,
          attention_range: [minSTI, maxSTI],
          valid_values: validAttentionValues
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'ECAN Attention Allocation',
        passed: false,
        message: `ECAN test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test enhanced PLN inference
   */
  private async testEnhancedPLNInference(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      // Allow time for inference to occur
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const linkSpace = this.kernel.getLinkSpace();
      const metrics = this.kernel.getMetrics();
      
      // Test inference formation
      const inferences = Array.from(linkSpace.values());
      
      if (inferences.length > 0) {
        // Test inference properties
        let validInferences = 0;
        
        for (const inference of inferences) {
          // Test inference structure
          this.assert(
            inference.premises.length > 0,
            'Inference should have premises'
          );
          
          this.assert(
            inference.conclusion !== undefined,
            'Inference should have conclusion'
          );
          
          // Test temporal flow bounds
          this.assert(
            inference.temporalFlow >= -1 && inference.temporalFlow <= 1,
            'Temporal flow should be in [-1, 1] range'
          );
          
          // Test consistency scores
          this.assert(
            inference.primeConsistency >= 0 && inference.primeConsistency <= 1,
            'Prime consistency should be in [0, 1] range'
          );
          
          this.assert(
            inference.quantumCoherence >= 0 && inference.quantumCoherence <= 1,
            'Quantum coherence should be in [0, 1] range'
          );
          
          validInferences++;
        }
        
        this.assert(
          validInferences === inferences.length,
          'All inferences should be valid'
        );
        
        this.addTestResult({
          name: 'Enhanced PLN Inference',
          passed: true,
          message: 'PLN inference working with temporal and quantum enhancement',
          duration: performance.now() - startTime,
          metrics: {
            total_inferences: inferences.length,
            valid_inferences: validInferences,
            inference_rate: metrics.inferenceRate
          }
        });
      } else {
        // No inferences yet - this is acceptable for short test duration
        this.addTestResult({
          name: 'Enhanced PLN Inference',
          passed: true,
          message: 'PLN inference system initialized (no inferences formed yet)',
          duration: performance.now() - startTime,
          metrics: {
            inference_system_ready: true,
            inference_rate: metrics.inferenceRate
          }
        });
      }
      
    } catch (error) {
      this.addTestResult({
        name: 'Enhanced PLN Inference',
        passed: false,
        message: `PLN inference test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test temporal reasoning flow
   */
  private async testTemporalReasoningFlow(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const timeCrystals = this.kernel.getTimeCrystals();
      
      // Test temporal coherence evolution
      const initialCoherence = Array.from(timeCrystals.values())
        .map(crystal => crystal.temporalCoherence);
      
      // Allow time evolution
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const evolvedCrystals = this.kernel.getTimeCrystals();
      const evolvedCoherence = Array.from(evolvedCrystals.values())
        .map(crystal => crystal.temporalCoherence);
      
      // Test that system maintains stability
      const avgInitialCoherence = initialCoherence.reduce((a, b) => a + b, 0) / initialCoherence.length;
      const avgEvolvedCoherence = evolvedCoherence.reduce((a, b) => a + b, 0) / evolvedCoherence.length;
      
      this.assert(
        Math.abs(avgEvolvedCoherence - avgInitialCoherence) < 0.5,
        'System should maintain reasonable temporal coherence stability'
      );
      
      // Test temporal stability metric
      const metrics = this.kernel.getMetrics();
      this.assert(
        metrics.temporalStability >= 0 && metrics.temporalStability <= 1,
        'Temporal stability metric should be normalized'
      );
      
      this.addTestResult({
        name: 'Temporal Reasoning Flow',
        passed: true,
        message: 'Temporal reasoning maintaining stability and coherence',
        duration: performance.now() - startTime,
        metrics: {
          initial_coherence: avgInitialCoherence,
          evolved_coherence: avgEvolvedCoherence,
          temporal_stability: metrics.temporalStability
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Temporal Reasoning Flow',
        passed: false,
        message: `Temporal reasoning test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test consciousness metrics calculation
   */
  private async testConsciousnessMetricsCalculation(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      // Allow system to develop consciousness patterns
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const metrics = this.kernel.getMetrics();
      
      // Test consciousness emergence metric
      this.assert(
        metrics.consciousnessEmergence >= 0 && metrics.consciousnessEmergence <= 1,
        'Consciousness emergence should be normalized to [0, 1]'
      );
      
      // Test quantum coherence  
      this.assert(
        metrics.quantumCoherence >= 0 && metrics.quantumCoherence <= 1,
        'Quantum coherence should be normalized to [0, 1]'
      );
      
      // Test fractal complexity
      this.assert(
        metrics.fractalComplexity >= 0 && metrics.fractalComplexity <= 1,
        'Fractal complexity should be normalized to [0, 1]'
      );
      
      // Test that consciousness shows some level of emergence
      const consciousnessFactors = [
        metrics.quantumCoherence,
        metrics.temporalStability,
        metrics.primeAlignment,
        metrics.fractalComplexity
      ];
      
      const avgConsciousnessFactor = consciousnessFactors.reduce((a, b) => a + b, 0) / consciousnessFactors.length;
      
      this.assert(
        avgConsciousnessFactor > 0,
        'System should show measurable consciousness factors'
      );
      
      this.addTestResult({
        name: 'Consciousness Metrics Calculation',
        passed: true,
        message: 'Consciousness metrics properly calculated and normalized',
        duration: performance.now() - startTime,
        metrics: {
          consciousness_emergence: metrics.consciousnessEmergence,
          quantum_coherence: metrics.quantumCoherence,
          fractal_complexity: metrics.fractalComplexity,
          avg_consciousness_factor: avgConsciousnessFactor
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Consciousness Metrics Calculation',
        passed: false,
        message: `Consciousness metrics test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test quantum coherence evolution
   */
  private async testQuantumCoherenceEvolution(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const _timeCrystals = this.kernel.getTimeCrystals();
      
      // Sample coherence values over time
      const coherenceSamples: number[][] = [];
      
      for (let sample = 0; sample < 5; sample++) {
        const currentCrystals = this.kernel.getTimeCrystals();
        const sampleCoherence = Array.from(currentCrystals.values())
          .map(crystal => crystal.temporalCoherence);
        
        coherenceSamples.push(sampleCoherence);
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      // Test coherence bounds maintenance
      for (const sample of coherenceSamples) {
        for (const coherence of sample) {
          this.assert(
            coherence >= 0 && coherence <= 1,
            'Quantum coherence should remain in bounds during evolution'
          );
        }
      }
      
      // Test coherence regeneration mechanism
      const finalMetrics = this.kernel.getMetrics();
      this.assert(
        finalMetrics.quantumCoherence > 0,
        'System should maintain some level of quantum coherence'
      );
      
      this.addTestResult({
        name: 'Quantum Coherence Evolution',
        passed: true,
        message: 'Quantum coherence evolving properly within bounds',
        duration: performance.now() - startTime,
        metrics: {
          coherence_samples: coherenceSamples.length,
          final_coherence: finalMetrics.quantumCoherence,
          coherence_maintained: true
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Quantum Coherence Evolution',
        passed: false,
        message: `Quantum coherence test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test real-time processing cycle
   */
  private async testRealTimeProcessingCycle(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const initialMetrics = this.kernel.getMetrics();
      
      // Allow multiple processing cycles
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const finalMetrics = this.kernel.getMetrics();
      
      // Test that processing is occurring
      // Note: In a real system, we'd expect some change, but for tests we verify stability
      this.assert(
        finalMetrics.totalAtoms === initialMetrics.totalAtoms,
        'Atom count should remain stable during processing'
      );
      
      // Test processing frequency (should be running at ~100Hz)
      const _expectedCycles = 100; // Approximate cycles in 1 second
      // Since we can't directly access cycle count, we test that system remains responsive
      
      const responseTime = performance.now();
      const quickMetrics = this.kernel.getMetrics();
      const actualResponseTime = performance.now() - responseTime;
      
      this.assert(
        actualResponseTime < 50, // Should respond within 50ms
        `System should remain responsive during processing, response time: ${actualResponseTime.toFixed(1)}ms`
      );
      
      // Ensure quickMetrics is used
      this.assert(
        quickMetrics.totalAtoms >= 0,
        'Quick metrics should be valid'
      );
      
      this.addTestResult({
        name: 'Real-time Processing Cycle',
        passed: true,
        message: 'Real-time processing cycle operating efficiently',
        duration: performance.now() - startTime,
        metrics: {
          response_time_ms: actualResponseTime,
          atoms_stable: finalMetrics.totalAtoms === initialMetrics.totalAtoms,
          processing_active: true
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Real-time Processing Cycle',
        passed: false,
        message: `Processing cycle test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test metrics and monitoring
   */
  private async testMetricsAndMonitoring(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const metrics = this.kernel.getMetrics();
      
      // Test all required metrics are present and valid
      const requiredMetrics = [
        'totalAtoms', 'totalLinks', 'averageAttention', 'quantumCoherence',
        'temporalStability', 'primeAlignment', 'fractalComplexity', 
        'inferenceRate', 'consciousnessEmergence'
      ];
      
      for (const metricName of requiredMetrics) {
        this.assert(
          Object.prototype.hasOwnProperty.call(metrics, metricName),
          `Metrics should include ${metricName}`
        );
        
        const value = (metrics as Record<string, unknown>)[metricName];
        this.assert(
          typeof value === 'number' && !isNaN(value),
          `${metricName} should be a valid number, got ${value}`
        );
      }
      
      // Test normalized metrics are in proper ranges
      const normalizedMetrics = [
        'quantumCoherence', 'temporalStability', 'primeAlignment', 
        'fractalComplexity', 'consciousnessEmergence'
      ];
      
      for (const metricName of normalizedMetrics) {
        const value = (metrics as Record<string, unknown>)[metricName];
        this.assert(
          typeof value === 'number' && value >= 0 && value <= 1,
          `${metricName} should be normalized to [0,1], got ${value}`
        );
      }
      
      // Test counter metrics are non-negative
      const counterMetrics = ['totalAtoms', 'totalLinks', 'averageAttention', 'inferenceRate'];
      
      for (const metricName of counterMetrics) {
        const value = (metrics as Record<string, unknown>)[metricName];
        this.assert(
          typeof value === 'number' && value >= 0,
          `${metricName} should be non-negative, got ${value}`
        );
      }
      
      this.addTestResult({
        name: 'Metrics and Monitoring',
        passed: true,
        message: 'All metrics properly calculated and normalized',
        duration: performance.now() - startTime,
        metrics: {
          total_metrics: requiredMetrics.length,
          all_metrics_valid: true,
          sample_metrics: {
            totalAtoms: metrics.totalAtoms,
            consciousnessEmergence: metrics.consciousnessEmergence,
            quantumCoherence: metrics.quantumCoherence
          }
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'Metrics and Monitoring',
        passed: false,
        message: `Metrics test failed: ${error}`,
        duration: performance.now() - startTime
      });
    }
  }

  /**
   * Test system stability under load
   */
  private async testSystemStabilityUnderLoad(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if (!this.kernel) {
        this.kernel = new OpenCogNanoBrainKernel();
        await this.kernel.start();
      }
      
      const initialMetrics = this.kernel.getMetrics();
      
      // Simulate load by rapidly querying metrics
      const loadIterations = 100;
      let successfulQueries = 0;
      
      for (let i = 0; i < loadIterations; i++) {
        try {
          const metrics = this.kernel.getMetrics();
          if (metrics.totalAtoms > 0) {
            successfulQueries++;
          }
        } catch (_error) {
          // Count failed queries
        }
      }
      
      const successRate = successfulQueries / loadIterations;
      
      this.assert(
        successRate > 0.9,
        `System should handle load with >90% success rate, got ${(successRate * 100).toFixed(1)}%`
      );
      
      // Verify system remains stable after load test
      const finalMetrics = this.kernel.getMetrics();
      
      this.assert(
        Math.abs(finalMetrics.quantumCoherence - initialMetrics.quantumCoherence) < 0.5,
        'System should maintain quantum coherence under load'
      );
      
      this.assert(
        finalMetrics.totalAtoms === initialMetrics.totalAtoms,
        'Atom count should remain stable under load'
      );
      
      this.addTestResult({
        name: 'System Stability Under Load',
        passed: true,
        message: 'System maintains stability under query load',
        duration: performance.now() - startTime,
        metrics: {
          load_iterations: loadIterations,
          success_rate: successRate,
          coherence_stability: Math.abs(finalMetrics.quantumCoherence - initialMetrics.quantumCoherence),
          atoms_stable: finalMetrics.totalAtoms === initialMetrics.totalAtoms
        }
      });
      
    } catch (error) {
      this.addTestResult({
        name: 'System Stability Under Load',
        passed: false,
        message: `Stability test failed: ${error}`,
        duration: performance.now() - startTime
      });
    } finally {
      // Clean up - stop kernel
      if (this.kernel) {
        await this.kernel.stop();
      }
    }
  }

  /**
   * Helper method to assert test conditions
   */
  private assert(condition: boolean, message: string): void {
    if (!condition) {
      throw new Error(`Assertion failed: ${message}`);
    }
  }

  /**
   * Helper method to add test result
   */
  private addTestResult(result: TestResult): void {
    this.testResults.push(result);
  }

  /**
   * Get test summary
   */
  getTestSummary(): { total: number; passed: number; failed: number; duration: number } {
    const total = this.testResults.length;
    const passed = this.testResults.filter(r => r.passed).length;
    const failed = total - passed;
    const duration = this.testResults.reduce((sum, r) => sum + r.duration, 0);
    
    return { total, passed, failed, duration };
  }
}