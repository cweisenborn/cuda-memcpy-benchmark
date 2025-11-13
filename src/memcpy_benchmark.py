"""
CUDA Memory Copy Benchmark

This module provides benchmarking functions for Host-to-Device (H2D) and 
Host-to-Host (H2H) memory transfers using CuPy.
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List
import json
import sys
import time
from memcpy_plots import plot_h2d_timings, plot_h2h_timings


@dataclass
class MemcpyStats:
    """Statistics for memory copy benchmark"""
    min_time: float
    max_time: float
    avg_time: float
    total_time: float
    count_above_50us: int
    num_elements: int
    bytes_transferred: int
    bandwidth_gbps: float
    timings: List[float]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'min_time': self.min_time,
            'max_time': self.max_time,
            'avg_time': self.avg_time,
            'total_time': self.total_time,
            'count_above_50us': self.count_above_50us,
            'num_elements': self.num_elements,
            'bytes_transferred': self.bytes_transferred,
            'bandwidth_gbps': self.bandwidth_gbps,
            'timings': self.timings
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)


def simple_h2d_benchmark(num_elements: int, num_iterations: int, use_async: bool = True) -> MemcpyStats:
    """
    Benchmark Host-to-Device memory transfers
    
    Args:
        num_elements: Number of int16 elements to transfer
        num_iterations: Number of benchmark iterations
        use_async: Use async transfers (with stream synchronization)
    
    Returns:
        MemcpyStats object containing benchmark results
    """
    # Allocate host memory (pinned)
    h_data = cp.cuda.alloc_pinned_memory(num_elements * 2)  # 2 bytes per int16
    h_array = np.frombuffer(h_data, dtype=np.int16, count=num_elements)
    
    # Initialize host data
    h_array[:] = np.arange(num_elements, dtype=np.int16) % 32767
    
    # Allocate device memory
    d_data = cp.empty(num_elements, dtype=cp.int16)
    
    # Create CUDA stream if using async
    stream = cp.cuda.Stream() if use_async else cp.cuda.Stream.null
    
    # Warm-up runs
    for _ in range(10):
        if use_async:
            d_data.set(h_array, stream=stream)
            stream.synchronize()
        else:
            d_data.set(h_array)
            cp.cuda.Device().synchronize()
    
    # Timing runs
    timings = []
    
    with stream:
        for _ in range(num_iterations):
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            
            start_event.record(stream=stream)
            
            if use_async:
                d_data.set(h_array, stream=stream)
            else:
                d_data.set(h_array)
            
            end_event.record(stream=stream)
            end_event.synchronize()
            
            # Get elapsed time in milliseconds
            elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            timings.append(elapsed_ms)
    
    # Calculate statistics
    min_time = min(timings)
    max_time = max(timings)
    total_time = sum(timings)
    avg_time = total_time / num_iterations
    
    # Count transfers above 50 microseconds
    count_above_50us = sum(1 for t in timings if t * 1000 > 50.0)
    
    # Calculate bandwidth (GB/s)
    bytes_transferred = num_elements * 2  # 2 bytes per int16
    bytes_per_ms = bytes_transferred / avg_time
    bandwidth_gbps = bytes_per_ms / 1e6
    
    return MemcpyStats(
        min_time=min_time,
        max_time=max_time,
        avg_time=avg_time,
        total_time=total_time,
        count_above_50us=count_above_50us,
        num_elements=num_elements,
        bytes_transferred=bytes_transferred,
        bandwidth_gbps=bandwidth_gbps,
        timings=timings
    )


def simple_h2h_benchmark(num_elements: int, num_iterations: int) -> MemcpyStats:
    """
    Benchmark Host-to-Host memory transfers (memcpy)
    
    Args:
        num_elements: Number of int16 elements to transfer
        num_iterations: Number of benchmark iterations
    
    Returns:
        MemcpyStats object containing benchmark results
    """
    # Allocate two host arrays (pinned memory)
    h_src_data = cp.cuda.alloc_pinned_memory(num_elements * 2)  # 2 bytes per int16
    h_src_array = np.frombuffer(h_src_data, dtype=np.int16, count=num_elements)
    
    h_dst_data = cp.cuda.alloc_pinned_memory(num_elements * 2)
    h_dst_array = np.frombuffer(h_dst_data, dtype=np.int16, count=num_elements)
    
    # Initialize source data
    h_src_array[:] = np.arange(num_elements, dtype=np.int16) % 32767
    
    # Warm-up runs
    for _ in range(10):
        np.copyto(h_dst_array, h_src_array)
    
    # Timing runs using time.perf_counter() for high-resolution CPU timing
    timings = []
    
    for _ in range(num_iterations):
        # Use perf_counter for high-resolution monotonic timing
        start_time = time.perf_counter()
        
        # Perform host-to-host memcpy
        np.copyto(h_dst_array, h_src_array)
        
        end_time = time.perf_counter()
        
        # Calculate elapsed time in milliseconds
        elapsed_ms = (end_time - start_time) * 1000.0
        timings.append(elapsed_ms)
    
    # Calculate statistics
    min_time = min(timings)
    max_time = max(timings)
    total_time = sum(timings)
    avg_time = total_time / num_iterations
    
    # Count transfers above 50 microseconds
    count_above_50us = sum(1 for t in timings if t * 1000 > 50.0)
    
    # Calculate bandwidth (GB/s)
    bytes_transferred = num_elements * 2  # 2 bytes per int16
    bytes_per_ms = bytes_transferred / avg_time
    bandwidth_gbps = bytes_per_ms / 1e6
    
    return MemcpyStats(
        min_time=min_time,
        max_time=max_time,
        avg_time=avg_time,
        total_time=total_time,
        count_above_50us=count_above_50us,
        num_elements=num_elements,
        bytes_transferred=bytes_transferred,
        bandwidth_gbps=bandwidth_gbps,
        timings=timings
    )


def print_stats(stats: MemcpyStats, label: str, num_iterations: int):
    """Print statistics for a benchmark"""
    print("="*60)
    print(f"{label} Statistics")
    print("="*60)
    print(f"Number of elements:     {stats.num_elements:,}")
    print(f"Bytes transferred:      {stats.bytes_transferred:,} ({stats.bytes_transferred / 1024:.1f} KB)")
    print(f"Number of iterations:   {num_iterations:,}")
    print()
    print(f"Min time:               {stats.min_time:.6f} ms  ({stats.min_time * 1000:.3f} µs)")
    print(f"Avg time:               {stats.avg_time:.6f} ms  ({stats.avg_time * 1000:.3f} µs)")
    print(f"Max time:               {stats.max_time:.6f} ms  ({stats.max_time * 1000:.3f} µs)")
    print(f"Range:                  {stats.max_time - stats.min_time:.6f} ms")
    print(f"Total time:             {stats.total_time:.6f} ms")
    print()
    print(f"Bandwidth (avg):        {stats.bandwidth_gbps:.6f} GB/s")
    print()
    print(f"Transfers > 50 µs:      {stats.count_above_50us:,} ({100.0 * stats.count_above_50us / num_iterations:.1f}%)")
    print("="*60)


def main():
    """Main function for command-line execution"""
    # Default configuration
    num_elements = 32768
    num_iterations = 10000000
    use_async = True
    output_basename = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        num_elements = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_iterations = int(sys.argv[2])
    if len(sys.argv) > 3:
        use_async = sys.argv[3].lower() in ('true', '1', 'yes')
    if len(sys.argv) > 4:
        output_basename = sys.argv[4]
    
    print("CUDA Memory Copy Benchmark")
    print("="*60)
    print(f"Number of elements: {num_elements:,}")
    print(f"Buffer size: {num_elements * 2 / 1024:.1f} KB ({num_elements * 2 / (1024*1024):.6f} MB)")
    print(f"Iterations: {num_iterations:,}")
    print(f"H2D Transfer mode: {'Async (with stream)' if use_async else 'Sync'}")
    print(f"H2H Transfer mode: Host memcpy")
    print("="*60)
    print()
    
    # Run H2D benchmark
    print("Running H2D benchmark...")
    h2d_stats = simple_h2d_benchmark(num_elements, num_iterations, use_async)
    print("H2D benchmark complete!\n")
    
    # Run H2H benchmark
    print("Running H2H benchmark...")
    h2h_stats = simple_h2h_benchmark(num_elements, num_iterations)
    print("H2H benchmark complete!\n")
    
    # Print statistics
    print_stats(h2d_stats, "Host-to-Device (H2D)", num_iterations)
    print()
    print_stats(h2h_stats, "Host-to-Host (H2H)", num_iterations)
    
    # Save results to files if output basename specified
    if output_basename:
        # Save JSON results
        json_file = f"{output_basename}.json"
        results = {
            'configuration': {
                'num_elements': num_elements,
                'num_iterations': num_iterations,
                'use_async': use_async
            },
            'h2d': h2d_stats.to_dict(),
            'h2h': h2h_stats.to_dict()
        }
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {json_file}")
        
        # Generate H2D plot
        h2d_plot_file = f"{output_basename}_h2d.png"
        plot_h2d_timings(h2d_stats.timings, h2d_stats.avg_time, num_elements, 
                        use_async, h2d_plot_file)
        
        # Generate H2H plot
        h2h_plot_file = f"{output_basename}_h2h.png"
        plot_h2h_timings(h2h_stats.timings, h2h_stats.avg_time, num_elements, 
                        h2h_plot_file)


if __name__ == "__main__":
    main()
