from abc import ABC, abstractmethod
from typing import Generator, Optional, List, Tuple
import queue
import threading
from dataclasses import dataclass

class PipelineStage(ABC):
    """Base class for pipeline stages"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process a single item"""
        pass
    
    @abstractmethod
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Process a batch of items"""
        pass

class OrderedOutputBuffer:
    """Thread-safe ordered output buffer - reusable for both modes"""
    
    def __init__(self, total_frames: Optional[int] = None):
        self.total_frames = total_frames
        self.buffer = {}
        self.next_idx = 0
        self.lock = threading.Lock()
        self._complete = threading.Event()
        
    def add(self, idx: int, data: Any):
        """Add data with its sequence index"""
        with self.lock:
            self.buffer[idx] = data
            
    def get_ready_items(self) -> Generator[Any, None, None]:
        """Yield items that are ready in order"""
        with self.lock:
            while self.next_idx in self.buffer:
                item = self.buffer.pop(self.next_idx)
                self.next_idx += 1
                yield item
                
                # Check if we're done (for batch mode)
                if self.total_frames and self.next_idx >= self.total_frames:
                    self._complete.set()
    
    def wait_completion(self, timeout: Optional[float] = None):
        """Wait for all frames to be processed (batch mode)"""
        if self.total_frames:
            return self._complete.wait(timeout)
        return True

class BasePipeline(ABC):
    """Base pipeline class for both streaming and batch processing"""
    
    def __init__(self, model_manager: ModelManager, batch_size: int = 8, 
                 num_workers: int = 4, profiler: Optional[PerformanceProfiler] = None):
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.profiler = profiler or PerformanceProfiler("Pipeline")
        
        # Queues
        self.preprocess_queue = queue.Queue(maxsize=batch_size * 10)
        self.gpu_queue = queue.Queue(maxsize=batch_size * 10)
        self.postprocess_queue = queue.Queue(maxsize=batch_size * 10)
        
        # Workers
        self.workers = []
        self._running = False
        
    @abstractmethod
    def start(self):
        """Start the pipeline workers"""
        pass
        
    @abstractmethod
    def stop(self):
        """Stop the pipeline workers"""
        pass