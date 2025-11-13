#!/usr/bin/env python3
"""
Memory Copy Benchmark Plotting Tool

Reads JSON benchmark results from H2D and H2H benchmarks and generates
time series plots with statistics.
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PNG generation
import numpy as np
from pathlib import Path


def load_benchmark_json(json_path):
    """Load benchmark results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def plot_benchmark_timings(data, output_file):
    """
    Generate a time series plot of benchmark timings
    
    Args:
        data: Dictionary containing benchmark data from JSON
        output_file: Path to save the PNG file
    """
    timings_ms = data['timings_ms']
    avg_time = data['avg_time_ms']
    min_time = data['min_time_ms']
    max_time = data['max_time_ms']
    num_elements = data['num_elements']
    num_iterations = data['num_iterations']
    benchmark_type = data['benchmark_type']
    total_bytes = data['total_bytes']
    memory_type = data.get('memory_type', 'N/A')  # Get memory type if available
    
    # Convert timings to microseconds
    timings_us = [t * 1000 for t in timings_ms]
    avg_time_us = avg_time * 1000
    min_time_us = min_time * 1000
    max_time_us = max_time * 1000
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Choose color based on benchmark type
    color = 'steelblue' if benchmark_type == 'H2D' else 'purple'
    if benchmark_type == 'H2D':
        benchmark_name = f'Host-to-Device (CUDA, {memory_type} memory)'
    else:
        benchmark_name = 'Host-to-Host (memcpy)'
    
    # Plot transfer times
    ax.plot(range(len(timings_us)), timings_us, 
            linewidth=0.8, alpha=0.7, color=color,
            label=f'{benchmark_type} Transfer Time')
    
    # Plot average line
    ax.axhline(y=avg_time_us, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_time_us:.2f} µs')
    
    # Plot min/max lines
    ax.axhline(y=min_time_us, color='green', linestyle=':', linewidth=1.5,
               label=f'Min: {min_time_us:.2f} µs', alpha=0.7)
    ax.axhline(y=max_time_us, color='orange', linestyle=':', linewidth=1.5,
               label=f'Max: {max_time_us:.2f} µs', alpha=0.7)
    
    # Plot 50µs threshold line
    ax.axhline(y=50, color='darkred', linestyle='-.', linewidth=2,
               label='50 µs threshold', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transfer Time (µs)', fontsize=12, fontweight='bold')
    
    title_text = f'{benchmark_name} Memory Copy Benchmark\n'
    title_text += f'{num_elements:,} elements ({total_bytes:,} bytes = {total_bytes/1024:.2f} KB), '
    title_text += f'{num_iterations:,} iterations'
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    # Disable scientific notation on x-axis and use plain integer format
    ax.ticklabel_format(style='plain', axis='x')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Min: {min_time_us:.2f} µs\n'
    stats_text += f'Avg: {avg_time_us:.2f} µs\n'
    stats_text += f'Max: {max_time_us:.2f} µs\n'
    stats_text += f'Range: {max_time_us - min_time_us:.2f} µs'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved to: {output_file}")


def plot_comparison(h2d_data, h2h_data, output_file):
    """
    Generate a comparison plot of H2D vs H2H benchmarks
    
    Args:
        h2d_data: Dictionary containing H2D benchmark data
        h2h_data: Dictionary containing H2H benchmark data
        output_file: Path to save the PNG file
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # H2D plot
    timings_h2d_us = [t * 1000 for t in h2d_data['timings_ms']]
    avg_h2d_us = h2d_data['avg_time_ms'] * 1000
    memory_type_h2d = h2d_data.get('memory_type', 'unknown')
    
    ax1.plot(range(len(timings_h2d_us)), timings_h2d_us, 
            linewidth=0.8, alpha=0.7, color='steelblue',
            label='H2D Transfer Time')
    ax1.axhline(y=avg_h2d_us, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_h2d_us:.2f} µs')
    ax1.axhline(y=50, color='darkred', linestyle='-.', linewidth=2,
               label='50 µs threshold', alpha=0.5)
    
    ax1.set_ylabel('Transfer Time (µs)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Host-to-Device (CUDA, {memory_type_h2d} memory) - {h2d_data["num_elements"]:,} elements', 
                  fontsize=13, fontweight='bold')
    ax1.ticklabel_format(style='plain', axis='x')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=10)
    
    # H2H plot
    timings_h2h_us = [t * 1000 for t in h2h_data['timings_ms']]
    avg_h2h_us = h2h_data['avg_time_ms'] * 1000
    
    ax2.plot(range(len(timings_h2h_us)), timings_h2h_us, 
            linewidth=0.8, alpha=0.7, color='purple',
            label='H2H Transfer Time')
    ax2.axhline(y=avg_h2h_us, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_h2h_us:.2f} µs')
    ax2.axhline(y=50, color='darkred', linestyle='-.', linewidth=2,
               label='50 µs threshold', alpha=0.5)
    
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Transfer Time (µs)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Host-to-Host (memcpy) - {h2h_data["num_elements"]:,} elements', 
                  fontsize=13, fontweight='bold')
    ax2.ticklabel_format(style='plain', axis='x')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Main title
    fig.suptitle('Memory Copy Benchmark Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison plot saved to: {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_benchmarks.py <output_directory> <json_file1> [json_file2 ...]")
        print("\nExamples:")
        print("  Single benchmark:")
        print("    python plot_benchmarks.py ./output h2d_results.json")
        print("\n  Multiple benchmarks (will create comparison plot):")
        print("    python plot_benchmarks.py ./output h2d_results.json h2h_results.json")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    json_files = sys.argv[2:]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Processing {len(json_files)} JSON file(s)...\n")
    
    # Load all benchmark data
    benchmarks = []
    for json_file in json_files:
        print(f"Loading {json_file}...")
        data = load_benchmark_json(json_file)
        benchmarks.append(data)
        
        # Generate individual plot
        basename = Path(json_file).stem
        output_file = os.path.join(output_dir, f"{basename}_plot.png")
        print(f"Generating plot for {data['benchmark_type']} benchmark...")
        plot_benchmark_timings(data, output_file)
        print()
    
    # Generate comparison plot if we have both H2D and H2H
    h2d_data = None
    h2h_data = None
    
    for data in benchmarks:
        if data['benchmark_type'] == 'H2D':
            h2d_data = data
        elif data['benchmark_type'] == 'H2H':
            h2h_data = data
    
    if h2d_data and h2h_data:
        comparison_file = os.path.join(output_dir, "comparison_plot.png")
        print("Generating comparison plot...")
        plot_comparison(h2d_data, h2h_data, comparison_file)
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()
