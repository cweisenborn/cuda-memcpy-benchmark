"""
CUDA Memory Copy Benchmark Plotting

This module provides plotting functions for visualizing Host-to-Device (H2D) and 
Host-to-Host (H2H) memory transfer benchmark results.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PNG generation
from typing import List
import numpy as np


def plot_transfer_timings(
    timings: List[float],
    avg_time: float,
    num_elements: int,
    transfer_type: str,
    mode: str,
    output_file: str
):
    """
    Generate a time series plot of transfer timings
    
    Args:
        timings: List of transfer times in milliseconds
        avg_time: Average transfer time in milliseconds
        num_elements: Number of elements transferred
        transfer_type: Type of transfer ('H2D' or 'H2H')
        mode: Transfer mode description (e.g., 'Async', 'Host memcpy')
        output_file: Path to save the PNG file
    """
    # Convert timings to microseconds
    timings_us = [t * 1000 for t in timings]
    avg_time_us = avg_time * 1000
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot transfer times
    ax.plot(range(len(timings_us)), timings_us, 
            linewidth=0.8, alpha=0.7, color='steelblue' if transfer_type == 'H2D' else 'purple',
            label=f'{transfer_type} Transfer Time')
    
    # Plot average line
    ax.axhline(y=avg_time_us, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_time_us:.2f} µs')
    
    # Plot 50µs threshold line
    ax.axhline(y=50, color='orange', linestyle=':', linewidth=2,
               label='50 µs threshold')
    
    # Labels and title
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Transfer Time (µs)', fontsize=12)
    ax.set_title(f'{transfer_type} Transfer Time\n{num_elements:,} elements, {mode}', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved to: {output_file}")


def plot_h2d_timings(timings: List[float], avg_time: float, num_elements: int, 
                     use_async: bool, output_file: str):
    """
    Generate time series plot for H2D benchmark
    
    Args:
        timings: List of transfer times in milliseconds
        avg_time: Average transfer time in milliseconds
        num_elements: Number of elements transferred
        use_async: Whether async transfer mode was used
        output_file: Path to save the PNG file
    """
    mode = 'Async (with stream)' if use_async else 'Sync'
    plot_transfer_timings(timings, avg_time, num_elements, 'H2D', mode, output_file)


def plot_h2h_timings(timings: List[float], avg_time: float, num_elements: int, 
                     output_file: str):
    """
    Generate time series plot for H2H benchmark
    
    Args:
        timings: List of transfer times in milliseconds
        avg_time: Average transfer time in milliseconds
        num_elements: Number of elements transferred
        output_file: Path to save the PNG file
    """
    plot_transfer_timings(timings, avg_time, num_elements, 'H2H', 'Host memcpy', output_file)
