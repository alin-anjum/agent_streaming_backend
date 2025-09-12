"""
Metrics collection and performance monitoring for Rapido system
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from contextlib import contextmanager
import statistics


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, **metadata):
        """Mark operation as finished"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.metadata.update(metadata)


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, metrics_collector: 'MetricsCollector' = None):
        self.operation_name = operation_name
        self.metrics_collector = metrics_collector
        self.metrics = None
    
    def __enter__(self):
        self.metrics = PerformanceMetrics(
            operation_name=self.operation_name,
            start_time=time.time()
        )
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        self.metrics.finish(success=success)
        
        if self.metrics_collector:
            self.metrics_collector.record_performance(self.metrics)


class FPSCounter:
    """FPS counter for video/audio streams"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self.last_frame_time = None
    
    def record_frame(self):
        """Record a new frame"""
        current_time = time.time()
        
        with self._lock:
            if self.last_frame_time is not None:
                frame_interval = current_time - self.last_frame_time
                self.frame_times.append(frame_interval)
            self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS"""
        with self._lock:
            if len(self.frame_times) < 2:
                return 0.0
            
            avg_interval = statistics.mean(self.frame_times)
            return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get detailed FPS statistics"""
        with self._lock:
            if len(self.frame_times) < 2:
                return {
                    "fps": 0.0,
                    "min_fps": 0.0,
                    "max_fps": 0.0,
                    "std_fps": 0.0,
                    "frame_count": 0
                }
            
            intervals = list(self.frame_times)
            fps_values = [1.0 / interval if interval > 0 else 0 for interval in intervals]
            
            return {
                "fps": statistics.mean(fps_values),
                "min_fps": min(fps_values),
                "max_fps": max(fps_values),
                "std_fps": statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0,
                "frame_count": len(intervals)
            }


class MetricsCollector:
    """Central metrics collector for the Rapido system"""
    
    def __init__(self):
        self._performance_metrics: List[PerformanceMetrics] = []
        self._fps_counters: Dict[str, FPSCounter] = {}
        self._custom_metrics: Dict[str, Any] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Initialize FPS counters for key components
        self._fps_counters = {
            "slide_frames": FPSCounter(),
            "synctalk_input": FPSCounter(),
            "synctalk_output": FPSCounter(),
            "composer": FPSCounter(),
            "livekit_output": FPSCounter()
        }
    
    def get_fps_counter(self, name: str) -> FPSCounter:
        """Get or create an FPS counter"""
        if name not in self._fps_counters:
            self._fps_counters[name] = FPSCounter()
        return self._fps_counters[name]
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        with self._lock:
            self._performance_metrics.append(metrics)
    
    def record_custom_metric(self, name: str, value: Any, metadata: Dict[str, Any] = None):
        """Record a custom metric"""
        with self._lock:
            self._custom_metrics[name].append({
                "value": value,
                "timestamp": time.time(),
                "metadata": metadata or {}
            })
    
    def get_fps_metrics(self) -> Dict[str, float]:
        """Get current FPS metrics for all counters"""
        return {name: counter.get_fps() for name, counter in self._fps_counters.items()}
    
    def get_detailed_fps_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed FPS statistics for all counters"""
        return {name: counter.get_stats() for name, counter in self._fps_counters.items()}
    
    def get_performance_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get performance summary for operations"""
        with self._lock:
            metrics = self._performance_metrics
            if operation_name:
                metrics = [m for m in metrics if m.operation_name == operation_name]
            
            if not metrics:
                return {}
            
            durations = [m.duration for m in metrics if m.duration is not None]
            success_count = sum(1 for m in metrics if m.success)
            
            return {
                "operation_name": operation_name or "all",
                "total_operations": len(metrics),
                "successful_operations": success_count,
                "success_rate": success_count / len(metrics),
                "avg_duration": statistics.mean(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return {
            "fps_metrics": self.get_detailed_fps_metrics(),
            "performance_metrics": {
                op: self.get_performance_summary(op)
                for op in set(m.operation_name for m in self._performance_metrics)
            },
            "custom_metrics": dict(self._custom_metrics),
            "timestamp": time.time()
        }
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self._lock:
            self._performance_metrics.clear()
            self._custom_metrics.clear()
            for counter in self._fps_counters.values():
                counter.frame_times.clear()
                counter.last_frame_time = None
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        with PerformanceTimer(operation_name, self) as timer:
            yield timer


# Global metrics collector instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
