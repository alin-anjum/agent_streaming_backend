import cv2
from tqdm import tqdm
from typing import Optional

from ..core.pipeline import BasePipeline, OrderedOutputBuffer
from ..core.data_structures import FrameData, ProcessingMode
from ..core.stages import PreProcessingStage, GPUInferenceStage, PostProcessingStage

class BatchProcessor(BasePipeline):
    """Batch/file processing implementation"""
    
    def __init__(self, model_manager, model_name: str, output_path: str,
                 video_writer: cv2.VideoWriter, total_frames: int, **kwargs):
        super().__init__(model_manager, batch_size=8, **kwargs)
        self.model_name = model_name
        self.output_path = output_path
        self.video_writer = video_writer
        self.total_frames = total_frames
        self.output_buffer = OrderedOutputBuffer(total_frames)
        
        # Progress tracking
        self.pbar = tqdm(total=total_frames, desc="Processing frames")
        
    def start(self):
        """Start pipeline workers"""
        self._running = True
        
        # Similar to streaming but with video writer
        self.workers = [
            threading.Thread(target=self._preprocessing_worker, daemon=True),
            threading.Thread(target=self._gpu_worker, daemon=True),
            *[threading.Thread(target=self._postprocessing_worker, daemon=True) 
              for _ in range(self.num_workers)]
        ]
        
        for worker in self.workers:
            worker.start()
    
    def process_frames(self, frame_generator):
        """Process all frames from generator"""
        self.start()
        
        try:
            # Submit all frames
            for frame_data in frame_generator:
                frame_data.metadata = {'model_name': self.model_name}
                self.preprocess_queue.put(frame_data)
            
            # Signal completion
            self.preprocess_queue.put(None)
            
            # Wait for completion
            self.output_buffer.wait_completion(timeout=300)
            
        finally:
            self.stop()
            self.pbar.close()
    
    def _postprocessing_worker(self):
        """Post-processing worker with video writing"""
        while self._running:
            try:
                item = self.postprocess_queue.get(timeout=0.1)
                if item is None:
                    break
                    
                frame_idx, processed_frame = self.postprocess_stage.process(item)
                
                # Add to ordered buffer
                self.output_buffer.add(frame_idx, processed_frame)
                
                # Write any ready frames
                for frame in self.output_buffer.get_ready_items():
                    self.video_writer.write(frame)
                    self.pbar.update(1)
                    
            except queue.Empty:
                continue