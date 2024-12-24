"""
Performance comparison between pure Python and Cythonized implementations.
"""

import dual_autodiff as df
import dual_autodiff_x as dfx
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import perf_counter
import sys
import psutil
import gc

def measure_time(func, *args, n_runs=1000):
    """Measure execution time of a function."""
    gc.collect()  # Force garbage collection
    times = []
    
    for _ in range(n_runs):
        start = perf_counter()
        func(*args)
        end = perf_counter()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def measure_memory(func, *args):
    """Measure memory usage of a function."""
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    result = func(*args)
    
    mem_after = process.memory_info().rss
    mem_used = mem_after - mem_before
    
    return mem_used, result

def plot_performance_comparison(df_results, title, ylabel='Time (seconds)', logscale=False):
    """Create a bar plot comparing performance metrics."""
    plt.figure()
    
    x = np.arange(len(df_results['Operation']))
    width = 0.35
    
    plt.bar(x - width/2, df_results['Python'], width, label='Pure Python',
            yerr=df_results['Python_std'], capsize=5)
    plt.bar(x + width/2, df_results['Cython'], width, label='Cythonized',
            yerr=df_results['Cython_std'], capsize=5)
    
    plt.xlabel('Operation')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, df_results['Operation'], rotation=30)
    plt.legend()
    
    if logscale:
        plt.yscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def calculate_speedup(python_time, cython_time):
    """Calculate speedup factor and percentage improvement."""
    speedup = python_time / cython_time
    improvement = (python_time - cython_time) / python_time * 100
    return speedup, improvement

# ============================
# Specific benchmarks
# ============================
def run_basic_operations_benchmark():
    """Run benchmarks for basic operations."""
    x_val, y_val = 2.0, 3.0
    
    operations = {
        'Creation': (
            lambda: df.Dual(x_val, 1.0),
            lambda: dfx.Dual(x_val, 1.0)
        ),
        'Addition': (
            lambda: df.Dual(x_val, 1.0) + df.Dual(y_val, 1.0),
            lambda: dfx.Dual(x_val, 1.0) + dfx.Dual(y_val, 1.0)
        ),
        'Multiplication': (
            lambda: df.Dual(x_val, 1.0) * df.Dual(y_val, 1.0),
            lambda: dfx.Dual(x_val, 1.0) * dfx.Dual(y_val, 1.0)
        ),
        'Division': (
            lambda: df.Dual(x_val, 1.0) / df.Dual(y_val, 1.0),
            lambda: dfx.Dual(x_val, 1.0) / dfx.Dual(y_val, 1.0)
        ),
        'Power': (
            lambda: df.Dual(x_val, 1.0) ** 2,
            lambda: dfx.Dual(x_val, 1.0) ** 2
        )
    }
    
    results = []
    for op_name, (py_func, cy_func) in operations.items():
        py_results = measure_time(py_func, n_runs=1000000)
        cy_results = measure_time(cy_func, n_runs=1000000)
        
        speedup, improvement = calculate_speedup(py_results['mean'], cy_results['mean'])
        
        results.append({
            'Operation': op_name,
            'Python': py_results['mean'],
            'Python_std': py_results['std'],
            'Cython': cy_results['mean'],
            'Cython_std': cy_results['std'],
            'Speedup': speedup,
            'Improvement': improvement
        })
    
    return pd.DataFrame(results)

def run_math_functions_benchmark():
    """Run benchmarks for mathematical functions."""
    x_val = 0.5  # Safe value for all functions
    
    functions = {
        'Exponential': (
            lambda: df.Dual(x_val, 1.0).exp(),
            lambda: dfx.Dual(x_val, 1.0).exp()
        ),
        'Sine': (
            lambda: df.Dual(x_val, 1.0).sin(),
            lambda: dfx.Dual(x_val, 1.0).sin()
        ),
        'Cosine': (
            lambda: df.Dual(x_val, 1.0).cos(),
            lambda: dfx.Dual(x_val, 1.0).cos()
        ),
        'Logarithm': (
            lambda: df.Dual(x_val, 1.0).log(),
            lambda: dfx.Dual(x_val, 1.0).log()
        )
    }
    
    results = []
    for func_name, (py_func, cy_func) in functions.items():
        py_results = measure_time(py_func, n_runs=1000000)
        cy_results = measure_time(cy_func, n_runs=1000000)
        
        speedup, improvement = calculate_speedup(py_results['mean'], cy_results['mean'])
        
        results.append({
            'Function': func_name,
            'Python': py_results['mean'],
            'Python_std': py_results['std'],
            'Cython': cy_results['mean'],
            'Cython_std': cy_results['std'],
            'Speedup': speedup,
            'Improvement': improvement
        })
    
    return pd.DataFrame(results)


def run_application_benchmarks():
    """Run benchmarks for real-world applications."""
    results = []
    
    # Neural Network Forward Pass
    def neural_network(x, impl):
        """Simple neural network forward pass with automatic differentiation."""
        w1 = impl.Dual(0.5, 1.0)
        w2 = impl.Dual(0.3, 1.0)
        b1 = impl.Dual(0.1, 1.0)
        b2 = impl.Dual(0.2, 1.0)
        
        # Forward pass with activation functions
        h = (w1 * x + b1).sin()
        y = (w2 * h + b2).exp()
        return y
    
    # Compute gradient of a multiple variable function  
    def gradient_computation(x_dual, y_dual, impl):
        # Compute partial derivatives using dual numbers
        grad_x = (x_dual**2 + y_dual**2).dual  # ∂f/∂x
        grad_y = (x_dual**2 + y_dual**2).dual  # ∂f/∂y

        return grad_x, grad_y
    
    # Physics Simulation
    def physics_sim(t, impl):
        """Simple harmonic motion with damping."""
        omega = impl.Dual(2.0, 0.0)
        gamma = impl.Dual(0.1, 0.0)
        return (omega * t).sin() * (-gamma * t).exp()
    
    applications = {
        'Neural Network': (
            lambda: neural_network(df.Dual(1.0, 1.0), df),
            lambda: neural_network(dfx.Dual(1.0, 1.0), dfx)
        ),
        'Gradient Computation': (
            lambda: gradient_computation(df.Dual(2.0, 1.0), df.Dual(3.0, 1.0), df),
            lambda: gradient_computation(dfx.Dual(2.0, 1.0), dfx.Dual(3.0, 1.0), dfx)
        ),
        'Physics Simulation': (
            lambda: physics_sim(df.Dual(1.0, 1.0), df),
            lambda: physics_sim(dfx.Dual(1.0, 1.0), dfx)
        )
    }
    
    for app_name, (py_func, cy_func) in applications.items():
        py_results = measure_time(py_func, n_runs=1000000)
        cy_results = measure_time(cy_func, n_runs=1000000)
        
        speedup, improvement = calculate_speedup(py_results['mean'], cy_results['mean'])
        
        results.append({
            'Application': app_name,
            'Python': py_results['mean'],
            'Python_std': py_results['std'],
            'Cython': cy_results['mean'],
            'Cython_std': cy_results['std'],
            'Speedup': speedup,
            'Improvement': improvement
        })
    
    return pd.DataFrame(results)
