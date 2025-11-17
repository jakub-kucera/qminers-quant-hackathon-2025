#!/usr/bin/env python3
"""
Example demonstrating parallel hyperparameter optimization.
This script shows different usage patterns for the optimizer.
"""

from src.optimize_hyperparameters import ModelOptimizer
import multiprocessing

def example_sequential():
    """Sequential execution - good for testing"""
    print("="*60)
    print("EXAMPLE 1: Sequential Execution")
    print("="*60)
    
    optimizer = ModelOptimizer(validation_period=365)
    best_params = optimizer.optimize(
        n_trials=10,
        study_name='test_sequential',
        n_jobs=1
    )
    print(f"\nBest parameters: {best_params}")


def example_parallel():
    """Parallel execution - good for production"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Parallel Execution (4 workers)")
    print("="*60)
    
    optimizer = ModelOptimizer(validation_period=365)
    best_params = optimizer.optimize(
        n_trials=20,
        study_name='test_parallel',
        n_jobs=4,
        storage='sqlite:///optuna_parallel.db'
    )
    print(f"\nBest parameters: {best_params}")


def example_all_cpus():
    """Use all available CPUs"""
    print("\n" + "="*60)
    print("EXAMPLE 3: All Available CPUs")
    print("="*60)
    
    n_cpus = multiprocessing.cpu_count()
    print(f"Detected {n_cpus} CPUs")
    
    optimizer = ModelOptimizer(validation_period=365)
    best_params = optimizer.optimize(
        n_trials=30,
        study_name='test_all_cpus',
        n_jobs=-1,  # Use all CPUs
        storage='sqlite:///optuna_all_cpus.db'
    )
    print(f"\nBest parameters: {best_params}")


def main():
    """Run examples"""
    print("""
Parallel Hyperparameter Optimization Examples
=============================================

This script demonstrates different execution modes:

1. Sequential: Good for quick testing (n_jobs=1)
2. Parallel: Use specific number of workers (n_jobs=4)
3. All CPUs: Maximize throughput (n_jobs=-1)

For parallel execution:
- Trials are distributed across workers
- Results are shared via SQLite database
- Speed-up is approximately linear with workers
- Each worker evaluates models independently

Note: These examples use small n_trials for demonstration.
For real optimization, use 100-200 trials.
""")
    
    # Run examples (comment out slow ones for testing)
    example_sequential()
    # example_parallel()
    # example_all_cpus()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == '__main__':
    main()
