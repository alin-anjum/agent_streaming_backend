# inference_system/utils/profiler.py
import time
import psutil
import GPUtil
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Callable
import json
from datetime import datetime
import torch
import threading
import csv
import re

class PerformanceProfiler:
    """
    Enhanced performance profiler with GPU/memory tracking and detailed timeline analysis.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.gpu_usage = defaultdict(list)
        self.current_timers = {}
        self.enabled = True
        
        # System info
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available and GPUtil.getGPUs():
            self.gpu = GPUtil.getGPUs()[0]
        else:
            self.gpu = None
            self.gpu_available = False
            
        self.process = psutil.Process()

        # Timeline Analysis
        self._log = []
        self._start_time = time.perf_counter()
        self._lock = threading.Lock()
        
        # NEW: Pipeline-specific tracking
        self.queue_sizes = defaultdict(list)
        self.stage_counts = defaultdict(int)
        self.gpu_sync_operations = set()  # Operations that need GPU sync
        
    @contextmanager
    def timer(self, operation: str, track_memory: bool = True, gpu_sync: bool = None):
        """
        Context manager for timing operations.
        
        Args:
            operation: Name of the operation
            track_memory: Whether to track memory usage
            gpu_sync: Force GPU synchronization. If None, auto-detect based on operation name
        """
        if not self.enabled:
            yield
            return
        
        # Auto-detect GPU operations that need synchronization
        if gpu_sync is None:
            gpu_sync = self.gpu_available and any(keyword in operation.lower() 
                                                  for keyword in ['gpu', 'cuda', 'model', 'infer'])
        
        # Start tracking
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        if self.gpu_available and track_memory:
            if gpu_sync:
                torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            # End tracking with proper GPU synchronization
            if self.gpu_available and gpu_sync:
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            with self._lock:
                self.timings[operation].append(elapsed_time)
                self.stage_counts[operation] += 1
                
                # Mark as GPU operation if synchronized
                if gpu_sync:
                    self.gpu_sync_operations.add(operation)
                
                # Log the detailed event for timeline visualization
                thread_name = threading.current_thread().name
                self._log.append((operation, thread_name, start_time, end_time, elapsed_time))
                
                if track_memory:
                    end_memory = self.process.memory_info().rss / 1024 / 1024
                    self.memory_usage[operation].append({
                        'start': start_memory,
                        'end': end_memory,
                        'delta': end_memory - start_memory
                    })
                    
                    if self.gpu_available:
                        end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        self.gpu_usage[operation].append({
                            'start': start_gpu_memory,
                            'end': end_gpu_memory,
                            'delta': end_gpu_memory - start_gpu_memory
                        })

    def measure_block(self, operation: str):
        """
        A semantic alias for timing blocking operations like queue.get().
        Memory tracking is disabled by default as it's typically irrelevant for waits.
        """
        return self.timer(operation, track_memory=False, gpu_sync=False)
    
    def record_queue_size(self, queue_name: str, size: int):
        """Record queue size for pipeline monitoring"""
        with self._lock:
            timestamp = time.perf_counter() - self._start_time
            self.queue_sizes[queue_name].append((timestamp, size))
    
    def get_pipeline_bottleneck(self, batch_size: int = 1) -> Dict:
        """Analyze pipeline and identify bottlenecks"""
        with self._lock:
            bottleneck_analysis = {
                'stages': {},
                'bottleneck': None,
                'theoretical_fps': 0,
                'efficiency': 0
            }
            
            # Analyze batch processing stages
            batch_stages = {}
            for op, times in self.timings.items():
                if 'batch' in op.lower() and times:
                    avg_time = np.mean(times)
                    
                    # Extract frame count from operation name if present
                    match = re.search(r'\((\d+) frames?\)', op)
                    actual_batch_size = int(match.group(1)) if match else batch_size
                    
                    per_frame_ms = (avg_time * 1000) / actual_batch_size
                    
                    # Only calculate meaningful FPS for operations > 0.1ms per frame
                    if per_frame_ms > 0.1:
                        fps_limit = 1000.0 / per_frame_ms
                    else:
                        fps_limit = float('inf')  # Too fast to be a bottleneck
                    
                    batch_stages[op] = {
                        'avg_time_ms': avg_time * 1000,
                        'per_frame_ms': per_frame_ms,
                        'fps_limit': fps_limit,
                        'batch_size': actual_batch_size
                    }
            
            # Find bottleneck (ignoring operations with infinite FPS)
            finite_fps_stages = {k: v for k, v in batch_stages.items() 
                               if v['fps_limit'] != float('inf')}
            
            if finite_fps_stages:
                slowest = min(finite_fps_stages.items(), key=lambda x: x[1]['fps_limit'])
                bottleneck_analysis['bottleneck'] = slowest[0]
                bottleneck_analysis['theoretical_fps'] = slowest[1]['fps_limit']
                bottleneck_analysis['stages'] = batch_stages
            
            # Calculate pipeline efficiency
            total_compute_time = sum(np.mean(times) for op, times in self.timings.items() 
                                   if 'wait' not in op.lower() and times)
            total_wait_time = sum(np.mean(times) for op, times in self.timings.items() 
                                if 'wait' in op.lower() and times)
            
            if total_compute_time > 0:
                bottleneck_analysis['efficiency'] = total_compute_time / (total_compute_time + total_wait_time)
            
            # Analyze queue buildups
            queue_analysis = {}
            for queue_name, sizes in self.queue_sizes.items():
                if sizes:
                    queue_sizes = [s[1] for s in sizes]
                    queue_analysis[queue_name] = {
                        'avg': np.mean(queue_sizes),
                        'max': max(queue_sizes),
                        'min': min(queue_sizes),
                        'std': np.std(queue_sizes)
                    }
            bottleneck_analysis['queues'] = queue_analysis
            
            return bottleneck_analysis

    def get_summary(self, operation: Optional[str] = None) -> Dict:
        """Get performance summary for specific operation or all operations"""
        if operation:
            return self._get_operation_summary(operation)
        
        summary = {
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'operations': {}
        }
        
        for op in self.timings.keys():
            summary['operations'][op] = self._get_operation_summary(op)
            
        return summary
    
    def _get_operation_summary(self, operation: str) -> Dict:
        """Get summary for a specific operation"""
        times = self.timings.get(operation, [])
        if not times:
            return {}
            
        summary = {
            'time': {
                'total': sum(times),
                'mean': np.mean(times) if times else 0,
                'std': np.std(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0,
                'count': len(times)
            },
            'gpu_sync': operation in self.gpu_sync_operations
        }
        
        if operation in self.memory_usage:
            mem_deltas = [m['delta'] for m in self.memory_usage[operation]]
            summary['memory_mb'] = {
                'mean_delta': np.mean(mem_deltas) if mem_deltas else 0,
                'max_delta': max(mem_deltas) if mem_deltas else 0,
                'total_delta': sum(mem_deltas)
            }
        
        if operation in self.gpu_usage:
            gpu_deltas = [g['delta'] for g in self.gpu_usage[operation]]
            summary['gpu_memory_mb'] = {
                'mean_delta': np.mean(gpu_deltas) if gpu_deltas else 0,
                'max_delta': max(gpu_deltas) if gpu_deltas else 0,
                'total_delta': sum(gpu_deltas)
            }
            
        return summary
    
    def print_summary(self, detailed: bool = True, show_waits: bool = False):
        """Print formatted summary"""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print(f"Performance Summary: {self.name}")
        print(f"{'='*60}")
        
        # Separate operations by type
        compute_ops = {}
        wait_ops = {}
        
        for op, stats in summary['operations'].items():
            if "wait" in op.lower():
                wait_ops[op] = stats
            else:
                compute_ops[op] = stats
        
        # Print compute operations
        # print("\nCompute Operations:")
        # for op, stats in sorted(compute_ops.items(), key=lambda x: x[1]['time']['total'], reverse=True):
        #     self._print_operation_stats(op, stats, detailed)
        
        # Print wait operations if requested
        if show_waits and wait_ops:
            print("\nWait Operations:")
            for op, stats in sorted(wait_ops.items(), key=lambda x: x[1]['time']['total'], reverse=True):
                self._print_operation_stats(op, stats, detailed)
        
        # Print pipeline analysis if available
        self._print_pipeline_analysis()
    
    def _print_operation_stats(self, op: str, stats: Dict, detailed: bool):
        """Helper to print operation statistics"""
        time_stats = stats['time']
        gpu_marker = " [GPU]" if stats.get('gpu_sync', False) else ""
        print(f"\n{op}{gpu_marker}:")
        print(f"  Time: {time_stats['total']:.3f}s total, "
              f"{time_stats['mean']*1000:.2f}ms avg "
              f"({time_stats['count']} calls)")
        
        if detailed:
            print(f"    Range: [{time_stats['min']*1000:.2f}ms - "
                  f"{time_stats['max']*1000:.2f}ms], "
                  f"Std: {time_stats['std']*1000:.2f}ms")
        
        if 'memory_mb' in stats:
            mem = stats['memory_mb']
            print(f"  Memory: {mem['mean_delta']:.1f}MB avg delta")
        
        if 'gpu_memory_mb' in stats:
            gpu = stats['gpu_memory_mb']
            print(f"  GPU Memory: {gpu['mean_delta']:.1f}MB avg delta")
    
    def _print_pipeline_analysis(self):
        """Print pipeline-specific analysis"""
        # Check if this is a pipeline profiler
        if not any('batch' in op.lower() for op in self.timings.keys()):
            return
        
        print(f"\n{'-'*60}")
        print("PIPELINE ANALYSIS")
        print(f"{'-'*60}")
        
        # Find batch size from the data
        batch_size = 32  # Default, should be passed in
        for op in self.timings.keys():
            if 'batch' in op.lower() and 'gpu' in op.lower():
                # Try to infer batch size from GPU operations
                # This is a heuristic - better to pass it explicitly
                break
        
        bottleneck = self.get_pipeline_bottleneck(batch_size)
        
        if bottleneck['bottleneck']:
            print(f"\nBottleneck: {bottleneck['bottleneck']}")
            print(f"Theoretical Max FPS: {bottleneck['theoretical_fps']:.1f}")
            print(f"Pipeline Efficiency: {bottleneck['efficiency']*100:.1f}%")
        
        if bottleneck['stages']:
            print("\nStage Performance:")
            for stage, perf in sorted(bottleneck['stages'].items(), 
                                    key=lambda x: x[1]['fps_limit']):
                # Only show FPS for meaningful operations
                if perf['fps_limit'] == float('inf'):
                    # Very fast operation - just show time
                    print(f"  {stage}: {perf['avg_time_ms']:.2f}ms/batch "
                          f"({perf['batch_size']} frames), <0.1ms/frame")
                elif perf['fps_limit'] > 10000:
                    # Still too fast to be meaningful as FPS
                    print(f"  {stage}: {perf['avg_time_ms']:.2f}ms/batch "
                          f"({perf['batch_size']} frames), {perf['per_frame_ms']:.2f}ms/frame")
                else:
                    # Meaningful FPS
                    print(f"  {stage}: {perf['avg_time_ms']:.2f}ms/batch "
                          f"({perf['batch_size']} frames), {perf['per_frame_ms']:.2f}ms/frame, "
                          f"~{perf['fps_limit']:.1f} FPS")
        
        if bottleneck['queues']:
            print("\nQueue Statistics:")
            for queue, stats in bottleneck['queues'].items():
                print(f"  {queue}: avg={stats['avg']:.1f}, max={stats['max']}, "
                      f"std={stats['std']:.1f}")
    
    def print_timeline(self):
        """Prints a timeline visualization of all recorded events."""
        print(f"\n{'='*100}")
        print(f"Event Timeline (started at t=0.00s)")
        print(f"{'='*100}")

        with self._lock:
            if not self._log:
                print("  No events logged.")
                return

            sorted_log = sorted(self._log, key=lambda x: x[2])
            abs_start_time = self._start_time
            
            print(f"{'Event Name':<30} | {'Thread':<16} | {'Start (s)':>10} | {'Duration (ms)':>12} | Visual")
            print("-" * 100)

            for name, thread_name, start, end, duration in sorted_log:
                start_relative = start - abs_start_time
                prefix_width = int(start_relative * 20)
                bar_width = max(1, int(duration * 200))
                visual = " " * prefix_width + "#" * bar_width

                if len(visual) > 120:
                    visual = visual[:117] + "..."

                print(f"{name:<30} | {thread_name:<16} | {start_relative:>10.4f} | {duration*1000:>12.2f} | {visual}")
        print("-" * 100)
    
    def save_timeline_csv(self, filepath: str):
        """Saves the detailed event timeline to a CSV file for analysis."""
        print(f"Saving detailed timeline report to {filepath}...")
        with self._lock:
            if not self._log:
                print("  No timeline data to save.")
                return

            sorted_log = sorted(self._log, key=lambda x: x[2])
            abs_start_time = self._start_time

            header = ["event_name", "thread_name", "start_time_s", "end_time_s", "duration_ms", "gpu_sync"]
            
            try:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for name, thread_name, start, end, duration in sorted_log:
                        writer.writerow([
                            name,
                            thread_name,
                            f"{start - abs_start_time:.6f}",
                            f"{end - abs_start_time:.6f}",
                            f"{duration * 1000:.6f}",
                            name in self.gpu_sync_operations
                        ])
                print(f"Successfully saved timeline to {filepath}")
            except IOError as e:
                print(f"Error: Could not write to file {filepath}. Reason: {e}")

    def save_report(self, filepath: str):
        """Save detailed report to file"""
        summary = self.get_summary()
        # Add pipeline analysis to the report
        summary['pipeline_analysis'] = self.get_pipeline_bottleneck()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def reset(self):
        """Reset all measurements"""
        with self._lock:
            self.timings.clear()
            self.memory_usage.clear()
            self.gpu_usage.clear()
            self.current_timers.clear()
            self._log.clear()
            self.queue_sizes.clear()
            self.stage_counts.clear()
            self.gpu_sync_operations.clear()
            self._start_time = time.perf_counter()