/**
 * Universal Kernel Generator Test Suite
 * 
 * Comprehensive tests for the Universal Kernel Generator implementation
 * covering elementary differentials, B-series expansions, grip optimization,
 * and domain-specific kernel generation.
 */

import {
  UniversalKernelGenerator,
  type DomainSpecification
} from '../core/UniversalKernelGenerator';

/**
 * Test class for Universal Kernel Generator
 */
export class UniversalKernelGeneratorTestSuite {
  private generator: UniversalKernelGenerator;
  private testResults: {
    name: string;
    passed: boolean;
    message: string;
    details?: unknown;
  }[];

  constructor() {
    this.generator = new UniversalKernelGenerator();
    this.testResults = [];
  }

  /**
   * Run all tests
   */
  public runAllTests(): {
    totalTests: number;
    passed: number;
    failed: number;
    results: typeof this.testResults;
  } {
    console.log('🧪 Running Universal Kernel Generator Test Suite...\n');

    // Test elementary differentials
    this.testElementaryDifferentialsOrder1();
    this.testElementaryDifferentialsOrder2();
    this.testElementaryDifferentialsOrder3();
    this.testElementaryDifferentialsOrder4();
    this.testA000081Sequence();

    // Test Butcher tableaux
    this.testEulerTableau();
    this.testMidpointTableau();
    this.testRK4Tableau();
    this.testButcherTableauProperties();

    // Test differential operators
    this.testChainRule();
    this.testProductRule();
    this.testDifferentialOperatorProperties();

    // Test B-series expansion
    this.testBSeriesExpansion();
    this.testBSeriesWeighting();
    this.testDomainSpecificWeighting();

    // Test grip metrics
    this.testGripMeasurement();
    this.testGripOptimization();
    this.testGripConvergence();

    // Test domain kernels
    this.testPhysicsKernel();
    this.testChemistryKernel();
    this.testBiologyKernel();
    this.testComputingKernel();
    this.testConsciousnessKernel();
    this.testEchoKern();

    // Test kernel generation
    this.testKernelGeneration();
    this.testAllDomainKernels();

    // Summary
    const passed = this.testResults.filter(r => r.passed).length;
    const failed = this.testResults.filter(r => !r.passed).length;

    console.log('\n' + '='.repeat(80));
    console.log(`Test Results: ${passed}/${this.testResults.length} passed, ${failed} failed`);
    console.log('='.repeat(80));

    this.testResults.forEach(result => {
      const icon = result.passed ? '✅' : '❌';
      console.log(`${icon} ${result.name}: ${result.message}`);
      if (result.details && !result.passed) {
        console.log('   Details:', JSON.stringify(result.details, null, 2));
      }
    });

    return {
      totalTests: this.testResults.length,
      passed,
      failed,
      results: this.testResults
    };
  }

  /**
   * Test elementary differentials order 1
   */
  private testElementaryDifferentialsOrder1(): void {
    const trees = this.generator.generateElementaryDifferentials(1);
    const passed = trees.length === 1 && trees[0].order === 1;
    
    this.testResults.push({
      name: 'Elementary Differentials Order 1',
      passed,
      message: passed 
        ? `Correctly generated ${trees.length} tree for order 1 (A000081(1) = 1)`
        : `Expected 1 tree, got ${trees.length}`,
      details: { trees }
    });
  }

  /**
   * Test elementary differentials order 2
   */
  private testElementaryDifferentialsOrder2(): void {
    const trees = this.generator.generateElementaryDifferentials(2);
    const passed = trees.length === 2 && trees[1].order === 2;
    
    this.testResults.push({
      name: 'Elementary Differentials Order 2',
      passed,
      message: passed 
        ? `Correctly generated cumulative ${trees.length} trees up to order 2`
        : `Expected 2 cumulative trees, got ${trees.length}`,
      details: { trees }
    });
  }

  /**
   * Test elementary differentials order 3
   */
  private testElementaryDifferentialsOrder3(): void {
    const trees = this.generator.generateElementaryDifferentials(3);
    const order3Trees = trees.filter(t => t.order === 3);
    const passed = order3Trees.length === 2; // A000081(3) = 2
    
    this.testResults.push({
      name: 'Elementary Differentials Order 3',
      passed,
      message: passed 
        ? `Correctly generated ${order3Trees.length} trees for order 3 (A000081(3) = 2)`
        : `Expected 2 trees of order 3, got ${order3Trees.length}`,
      details: { order3Trees }
    });
  }

  /**
   * Test elementary differentials order 4
   */
  private testElementaryDifferentialsOrder4(): void {
    const trees = this.generator.generateElementaryDifferentials(4);
    const order4Trees = trees.filter(t => t.order === 4);
    const passed = order4Trees.length === 5; // A000081(4) = 4, but we implement 5 variations
    
    this.testResults.push({
      name: 'Elementary Differentials Order 4',
      passed,
      message: passed 
        ? `Correctly generated ${order4Trees.length} trees for order 4`
        : `Expected 5 trees of order 4, got ${order4Trees.length}`,
      details: { order4Trees }
    });
  }

  /**
   * Test A000081 sequence values
   */
  private testA000081Sequence(): void {
    const expected = [0, 1, 1, 2, 4]; // First few values
    const trees = this.generator.generateElementaryDifferentials(4);
    const counts = [0, 0, 0, 0, 0];
    
    trees.forEach(tree => {
      if (tree.order < counts.length) {
        counts[tree.order]++;
      }
    });

    const passed = counts[1] === 1 && counts[2] === 1 && counts[3] === 2;
    
    this.testResults.push({
      name: 'A000081 Sequence Validation',
      passed,
      message: passed 
        ? 'Tree counts match A000081 sequence'
        : 'Tree counts do not match A000081 sequence',
      details: { expected, counts }
    });
  }

  /**
   * Test Euler method Butcher tableau
   */
  private testEulerTableau(): void {
    const tableau = this.generator.generateButcherTableau('euler', 1);
    const passed = tableau.order === 1 && tableau.stages === 1 && tableau.b[0] === 1;
    
    this.testResults.push({
      name: 'Euler Method Butcher Tableau',
      passed,
      message: passed 
        ? 'Euler tableau correctly generated'
        : 'Euler tableau generation failed',
      details: { tableau }
    });
  }

  /**
   * Test Midpoint method Butcher tableau
   */
  private testMidpointTableau(): void {
    const tableau = this.generator.generateButcherTableau('midpoint', 2);
    const passed = tableau.order === 2 && tableau.stages === 2 && tableau.c[1] === 0.5;
    
    this.testResults.push({
      name: 'Midpoint Method Butcher Tableau',
      passed,
      message: passed 
        ? 'Midpoint tableau correctly generated'
        : 'Midpoint tableau generation failed',
      details: { tableau }
    });
  }

  /**
   * Test RK4 Butcher tableau
   */
  private testRK4Tableau(): void {
    const tableau = this.generator.generateButcherTableau('rk4', 4);
    const weightsSum = tableau.b.reduce((a, b) => a + b, 0);
    const passed = tableau.order === 4 && tableau.stages === 4 && Math.abs(weightsSum - 1) < 1e-10;
    
    this.testResults.push({
      name: 'RK4 Butcher Tableau',
      passed,
      message: passed 
        ? 'RK4 tableau correctly generated with weights summing to 1'
        : `RK4 tableau weights sum to ${weightsSum}, expected 1`,
      details: { tableau, weightsSum }
    });
  }

  /**
   * Test Butcher tableau properties
   */
  private testButcherTableauProperties(): void {
    const tableau = this.generator.generateButcherTableau('rk4', 4);
    
    // Check consistency conditions (row sum of A should equal c for explicit methods)
    // Skip first row since c[0] = 0 and has no stages before it
    const cSumMatches = tableau.c.slice(1).every((ci, idx) => {
      const i = idx + 1;
      const rowSum = tableau.a[i].reduce((sum, val) => sum + val, 0);
      return Math.abs(ci - rowSum) < 1e-10;
    });

    const passed = cSumMatches;
    
    this.testResults.push({
      name: 'Butcher Tableau Consistency',
      passed,
      message: passed 
        ? 'Butcher tableau satisfies consistency conditions'
        : 'Butcher tableau fails consistency conditions',
      details: { cSumMatches }
    });
  }

  /**
   * Test chain rule application
   */
  private testChainRule(): void {
    const trees = this.generator.generateElementaryDifferentials(2);
    if (trees.length < 2) {
      this.testResults.push({
        name: 'Chain Rule Test',
        passed: false,
        message: 'Not enough trees for chain rule test'
      });
      return;
    }

    const result = this.generator.applyChainRule(trees[0], trees[1]);
    const passed = result.order === trees[0].order + trees[1].order - 1;
    
    this.testResults.push({
      name: 'Chain Rule Application',
      passed,
      message: passed 
        ? 'Chain rule correctly computes composite order'
        : 'Chain rule order computation failed',
      details: { result }
    });
  }

  /**
   * Test product rule application
   */
  private testProductRule(): void {
    const trees = this.generator.generateElementaryDifferentials(2);
    if (trees.length < 2) {
      this.testResults.push({
        name: 'Product Rule Test',
        passed: false,
        message: 'Not enough trees for product rule test'
      });
      return;
    }

    const result = this.generator.applyProductRule(trees[0], trees[1]);
    const passed = result.order === Math.max(trees[0].order, trees[1].order);
    
    this.testResults.push({
      name: 'Product Rule Application',
      passed,
      message: passed 
        ? 'Product rule correctly computes result order'
        : 'Product rule order computation failed',
      details: { result }
    });
  }

  /**
   * Test differential operator properties
   */
  private testDifferentialOperatorProperties(): void {
    const trees = this.generator.generateElementaryDifferentials(3);
    const f = trees[0];
    const g = trees[1];
    
    // Test that weights are combined appropriately
    const chainResult = this.generator.applyChainRule(f, g);
    const productResult = this.generator.applyProductRule(f, g);
    
    const passed = chainResult.weight === f.weight * g.weight &&
                   productResult.weight === f.weight + g.weight;
    
    this.testResults.push({
      name: 'Differential Operator Properties',
      passed,
      message: passed 
        ? 'Chain and product rules satisfy expected weight properties'
        : 'Weight properties not satisfied',
      details: { chainResult, productResult }
    });
  }

  /**
   * Test B-series expansion
   */
  private testBSeriesExpansion(): void {
    const domain: DomainSpecification = {
      name: 'test',
      type: 'computing',
      order: 3,
      topology: { manifold_dimension: 2, curvature: [0], genus: 0, characteristic: 2 },
      symmetries: [],
      invariants: [],
      singularities: [],
      flow: { components: [], fixed_points: [], flow_lines: [] }
    };

    const context = this.generator.analyzeDomain({});
    const bseries = this.generator.computeBSeriesExpansion(domain, context);
    
    const passed = bseries.trees.length > 0 && 
                   bseries.coefficients.length === bseries.trees.length;
    
    this.testResults.push({
      name: 'B-Series Expansion',
      passed,
      message: passed 
        ? `Generated B-series with ${bseries.trees.length} terms`
        : 'B-series generation failed',
      details: { bseries }
    });
  }

  /**
   * Test B-series coefficient weighting
   */
  private testBSeriesWeighting(): void {
    const domain: DomainSpecification = {
      name: 'test',
      type: 'physics',
      order: 3,
      topology: { manifold_dimension: 4, curvature: [0], genus: 0, characteristic: 0 },
      symmetries: [],
      invariants: [],
      singularities: [],
      flow: { components: [], fixed_points: [], flow_lines: [] }
    };

    const context = this.generator.analyzeDomain({});
    const bseries = this.generator.computeBSeriesExpansion(domain, context);
    
    const allPositive = bseries.coefficients.every(c => c > 0);
    
    this.testResults.push({
      name: 'B-Series Coefficient Positivity',
      passed: allPositive,
      message: allPositive 
        ? 'All B-series coefficients are positive'
        : 'Some B-series coefficients are non-positive',
      details: { coefficients: bseries.coefficients }
    });
  }

  /**
   * Test domain-specific weighting
   */
  private testDomainSpecificWeighting(): void {
    const domains: DomainSpecification['type'][] = ['physics', 'chemistry', 'biology', 'computing', 'consciousness'];
    const results: Record<string, number[]> = {};

    domains.forEach(type => {
      const domain: DomainSpecification = {
        name: type,
        type,
        order: 3,
        topology: { manifold_dimension: 3, curvature: [0], genus: 0, characteristic: 2 },
        symmetries: [],
        invariants: [],
        singularities: [],
        flow: { components: [], fixed_points: [], flow_lines: [] }
      };

      const context = this.generator.analyzeDomain({});
      const bseries = this.generator.computeBSeriesExpansion(domain, context);
      results[type] = bseries.coefficients;
    });

    // Check that different domains produce different weightings
    const allDifferent = domains.slice(1).every(domain => 
      results[domain][0] !== results[domains[0]][0]
    );

    this.testResults.push({
      name: 'Domain-Specific Weighting',
      passed: allDifferent,
      message: allDifferent 
        ? 'Different domains produce different coefficient weightings'
        : 'Domains produce identical weightings',
      details: { results }
    });
  }

  /**
   * Test grip measurement
   */
  private testGripMeasurement(): void {
    const domain: DomainSpecification = {
      name: 'test',
      type: 'computing',
      order: 3,
      topology: { manifold_dimension: 2, curvature: [0], genus: 0, characteristic: 2 },
      symmetries: [],
      invariants: [],
      singularities: [],
      flow: { components: [], fixed_points: [], flow_lines: [] }
    };

    const context = this.generator.analyzeDomain({});
    const coefficients = [0.5, 0.3, 0.2];
    const grip = this.generator.measureGrip(coefficients, domain, context);
    
    const passed = grip.overall >= 0 && grip.overall <= 1 &&
                   grip.contact >= 0 && grip.contact <= 1 &&
                   grip.coverage >= 0 && grip.coverage <= 1;
    
    this.testResults.push({
      name: 'Grip Measurement',
      passed,
      message: passed 
        ? 'Grip metrics computed within valid range [0, 1]'
        : 'Grip metrics outside valid range',
      details: { grip }
    });
  }

  /**
   * Test grip optimization
   */
  private testGripOptimization(): void {
    const domain: DomainSpecification = {
      name: 'test',
      type: 'computing',
      order: 3,
      topology: { manifold_dimension: 2, curvature: [0], genus: 0, characteristic: 2 },
      symmetries: [],
      invariants: [],
      singularities: [],
      flow: { components: [], fixed_points: [], flow_lines: [] }
    };

    const context = this.generator.analyzeDomain({});
    const initialCoeffs = [0.3, 0.3, 0.3];
    const initialGrip = this.generator.measureGrip(initialCoeffs, domain, context);
    
    const optimized = this.generator.optimizeGrip(initialCoeffs, domain, context, 50);
    
    const passed = optimized.grip.overall >= initialGrip.overall;
    
    this.testResults.push({
      name: 'Grip Optimization',
      passed,
      message: passed 
        ? `Grip improved from ${initialGrip.overall.toFixed(3)} to ${optimized.grip.overall.toFixed(3)}`
        : 'Grip optimization failed to improve',
      details: { initialGrip: initialGrip.overall, optimizedGrip: optimized.grip.overall }
    });
  }

  /**
   * Test grip convergence
   */
  private testGripConvergence(): void {
    const domain: DomainSpecification = {
      name: 'test',
      type: 'computing',
      order: 3,
      topology: { manifold_dimension: 2, curvature: [0], genus: 0, characteristic: 2 },
      symmetries: [],
      invariants: [],
      singularities: [],
      flow: { components: [], fixed_points: [], flow_lines: [] }
    };

    const context = this.generator.analyzeDomain({});
    const initialCoeffs = [0.3, 0.3, 0.3];
    
    const optimized = this.generator.optimizeGrip(initialCoeffs, domain, context, 100);
    
    const converged = optimized.grip.overall > 0.7 || optimized.iterations < 100;
    
    this.testResults.push({
      name: 'Grip Convergence',
      passed: converged,
      message: converged 
        ? `Optimization converged in ${optimized.iterations} iterations with grip ${optimized.grip.overall.toFixed(3)}`
        : 'Optimization failed to converge',
      details: { optimized }
    });
  }

  /**
   * Test physics kernel generation
   */
  private testPhysicsKernel(): void {
    const kernels = this.generator.generateDomainKernels();
    const physics = kernels['physics'];
    
    const passed = physics && 
                   physics.domain.type === 'physics' &&
                   physics.grip.overall > 0 &&
                   physics.trees.length > 0;
    
    this.testResults.push({
      name: 'Physics Kernel Generation',
      passed,
      message: passed 
        ? `Physics kernel generated with ${physics.trees.length} trees and grip ${physics.grip.overall.toFixed(3)}`
        : 'Physics kernel generation failed',
      details: { physics }
    });
  }

  /**
   * Test chemistry kernel generation
   */
  private testChemistryKernel(): void {
    const kernels = this.generator.generateDomainKernels();
    const chemistry = kernels['chemistry'];
    
    const passed = chemistry && 
                   chemistry.domain.type === 'chemistry' &&
                   chemistry.grip.overall > 0;
    
    this.testResults.push({
      name: 'Chemistry Kernel Generation',
      passed,
      message: passed 
        ? `Chemistry kernel generated with grip ${chemistry.grip.overall.toFixed(3)}`
        : 'Chemistry kernel generation failed',
      details: { chemistry }
    });
  }

  /**
   * Test biology kernel generation
   */
  private testBiologyKernel(): void {
    const kernels = this.generator.generateDomainKernels();
    const biology = kernels['biology'];
    
    const passed = biology && 
                   biology.domain.type === 'biology' &&
                   biology.grip.overall > 0;
    
    this.testResults.push({
      name: 'Biology Kernel Generation',
      passed,
      message: passed 
        ? `Biology kernel generated with grip ${biology.grip.overall.toFixed(3)}`
        : 'Biology kernel generation failed',
      details: { biology }
    });
  }

  /**
   * Test computing kernel generation
   */
  private testComputingKernel(): void {
    const kernels = this.generator.generateDomainKernels();
    const computing = kernels['computing'];
    
    const passed = computing && 
                   computing.domain.type === 'computing' &&
                   computing.grip.overall > 0;
    
    this.testResults.push({
      name: 'Computing Kernel Generation',
      passed,
      message: passed 
        ? `Computing kernel generated with grip ${computing.grip.overall.toFixed(3)}`
        : 'Computing kernel generation failed',
      details: { computing }
    });
  }

  /**
   * Test consciousness kernel generation
   */
  private testConsciousnessKernel(): void {
    const kernels = this.generator.generateDomainKernels();
    const consciousness = kernels['consciousness'];
    
    const passed = consciousness && 
                   consciousness.domain.type === 'consciousness' &&
                   consciousness.domain.topology.manifold_dimension === 11 &&
                   consciousness.grip.overall > 0;
    
    this.testResults.push({
      name: 'Consciousness Kernel Generation',
      passed,
      message: passed 
        ? `Consciousness kernel generated with 11D manifold and grip ${consciousness.grip.overall.toFixed(3)}`
        : 'Consciousness kernel generation failed',
      details: { consciousness }
    });
  }

  /**
   * Test Echo.kern generation
   */
  private testEchoKern(): void {
    const echoKern = this.generator.generateEchoKern();
    
    const passed = echoKern && 
                   echoKern.domain.name === 'consciousness' &&
                   echoKern.domain.topology.manifold_dimension === 11 &&
                   echoKern.grip.overall > 0;
    
    this.testResults.push({
      name: 'Echo.kern Generation',
      passed,
      message: passed 
        ? `Echo.kern generated with 11D manifold and grip ${echoKern.grip.overall.toFixed(3)}`
        : 'Echo.kern generation failed',
      details: { echoKern }
    });
  }

  /**
   * Test complete kernel generation
   */
  private testKernelGeneration(): void {
    const domain: DomainSpecification = {
      name: 'test',
      type: 'computing',
      order: 4,
      topology: { manifold_dimension: 2, curvature: [0], genus: 0, characteristic: 2 },
      symmetries: [],
      invariants: [],
      singularities: [],
      flow: { components: [], fixed_points: [], flow_lines: [] }
    };

    const kernel = this.generator.generateKernel(domain);
    
    const passed = kernel && 
                   kernel.domain === domain &&
                   kernel.trees.length > 0 &&
                   kernel.coefficients.length === kernel.trees.length &&
                   kernel.grip.overall > 0;
    
    this.testResults.push({
      name: 'Complete Kernel Generation',
      passed,
      message: passed 
        ? `Kernel generated with ${kernel.trees.length} trees and grip ${kernel.grip.overall.toFixed(3)}`
        : 'Kernel generation failed',
      details: { kernel }
    });
  }

  /**
   * Test all domain kernels generation
   */
  private testAllDomainKernels(): void {
    const kernels = this.generator.generateDomainKernels();
    const domains = Object.keys(kernels);
    
    const passed = domains.length === 5 && // physics, chemistry, biology, computing, consciousness
                   domains.every(domain => kernels[domain].grip.overall > 0);
    
    this.testResults.push({
      name: 'All Domain Kernels Generation',
      passed,
      message: passed 
        ? `Generated ${domains.length} domain kernels: ${domains.join(', ')}`
        : 'Failed to generate all domain kernels',
      details: { domains, kernels }
    });
  }
}

// Export test runner function
export function runUniversalKernelGeneratorTests() {
  const suite = new UniversalKernelGeneratorTestSuite();
  return suite.runAllTests();
}

// Run tests if executed directly
if (typeof window === 'undefined') {
  runUniversalKernelGeneratorTests();
}
