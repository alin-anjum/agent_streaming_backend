class ModelAwareGPUScheduler:
    """
    Schedules GPU work across multiple models to minimize latency
    while maximizing throughput
    """
    
    def __init__(self, target_batch_latency_ms: float = 20.0):
        self.target_batch_latency_ms = target_batch_latency_ms
        self.model_queues = {}  # model_name -> deque of work items
        self.model_timings = {}  # model_name -> recent inference times
        self.lock = threading.Lock()
        
    def add_work(self, model_name: str, work_item):
        """Add work item for a specific model"""
        with self.lock:
            if model_name not in self.model_queues:
                self.model_queues[model_name] = deque()
                self.model_timings[model_name] = deque(maxlen=100)
            
            self.model_queues[model_name].append(work_item)
    
    def get_next_batch(self) -> Optional[Tuple[str, List]]:
        """
        Get the next batch to process, prioritizing:
        1. Models with the most waiting frames
        2. Models that haven't been processed recently
        3. Optimal batch sizes based on recent timings
        """
        with self.lock:
            if not any(self.model_queues.values()):
                return None
            
            # Find model with most urgent work
            best_model = None
            max_urgency = -1
            
            for model_name, queue in self.model_queues.items():
                if not queue:
                    continue
                    
                # Urgency based on queue length and age of oldest item
                urgency = len(queue)
                if urgency > max_urgency:
                    max_urgency = urgency
                    best_model = model_name
            
            if best_model is None:
                return None
            
            # Determine optimal batch size based on recent timings
            if best_model in self.model_timings and self.model_timings[best_model]:
                avg_time_per_item = np.mean(self.model_timings[best_model])
                optimal_batch_size = int(self.target_batch_latency_ms / avg_time_per_item)
                optimal_batch_size = max(1, min(8, optimal_batch_size))
            else:
                optimal_batch_size = 4  # Default
            
            # Get batch
            batch = []
            queue = self.model_queues[best_model]
            for _ in range(min(optimal_batch_size, len(queue))):
                batch.append(queue.popleft())
            
            return best_model, batch
    
    def record_timing(self, model_name: str, batch_size: int, time_ms: float):
        """Record timing for a batch inference"""
        with self.lock:
            if model_name in self.model_timings:
                time_per_item = time_ms / batch_size
                self.model_timings[model_name].append(time_per_item)

class GPUInferenceStage(PipelineStage):
    """Enhanced GPU inference stage with model-aware scheduling"""
    
    def __init__(self, model_manager: ModelManager, device: torch.device):
        self.model_manager = model_manager
        self.device = device
        self.scheduler = ModelAwareGPUScheduler()
        
        # Pre-allocate GPU memory for common tensor sizes
        self._preallocate_gpu_memory()
        
    def _preallocate_gpu_memory(self):
        """Pre-allocate GPU memory to avoid allocation overhead during inference"""
        # Pre-allocate tensors for common batch sizes
        self.preallocated_tensors = {}
        for batch_size in [1, 2, 4, 8]:
            self.preallocated_tensors[batch_size] = {
                'real': torch.zeros((batch_size, 3, 320, 320), 
                                  device=self.device, dtype=torch.half),
                'masked': torch.zeros((batch_size, 3, 320, 320), 
                                    device=self.device, dtype=torch.half),
                'audio': torch.zeros((batch_size, 32, 16, 16), 
                                   device=self.device, dtype=torch.half)
            }
    
    def process_batch(self, batch_tuple: Tuple) -> List[Tuple]:
        """Process batch with optimal GPU utilization"""
        batch_data, batch_real, batch_masked, batch_canvases, batch_crops = batch_tuple
        batch_size = len(batch_data)
        
        # Get model name (assuming all items in batch are for same model)
        model_name = batch_data[0].metadata.get('model_name', 'default')
        
        start_time = time.time()
        
        # Get model
        net, audio_encoder, config = self.model_manager.get_model(model_name)
        mode = config['mode']
        
        # Use pre-allocated tensors if possible
        if batch_size in self.preallocated_tensors:
            real_batch = self.preallocated_tensors[batch_size]['real']
            masked_batch = self.preallocated_tensors[batch_size]['masked']
            audio_batch = self.preallocated_tensors[batch_size]['audio']
            
            # Copy data to pre-allocated tensors
            for i in range(batch_size):
                real_batch[i] = torch.from_numpy(batch_real[i].transpose(2,0,1)) / 255.0
                masked_batch[i] = torch.from_numpy(batch_masked[i].transpose(2,0,1)) / 255.0
                audio_batch[i] = torch.from_numpy(batch_data[i].audio_feat)
        else:
            # Fall back to dynamic allocation
            real_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in batch_real]
            masked_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in batch_masked]
            audio_tensors = [torch.from_numpy(d.audio_feat) for d in batch_data]
            
            real_batch = torch.stack(real_tensors).to(self.device, non_blocking=True).half() / 255.0
            masked_batch = torch.stack(masked_tensors).to(self.device, non_blocking=True).half() / 255.0
            audio_batch = torch.stack(audio_tensors).to(self.device, non_blocking=True).half()
        
        if mode == "ave":
            audio_batch = audio_batch.view(batch_size, 32, 16, 16)
        
        # Prepare input
        img_for_net = torch.cat([real_batch[:batch_size], masked_batch[:batch_size]], dim=1)
        
        # Inference with minimal overhead
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                pred_batch = net(img_for_net, audio_batch[:batch_size])
        
        # Ensure GPU operations complete
        torch.cuda.synchronize()
        
        # Convert results
        pred_batch_np = pred_batch.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        
        # Record timing
        inference_time_ms = (time.time() - start_time) * 1000
        self.scheduler.record_timing(model_name, batch_size, inference_time_ms)
        
        # Return results
        results = []
        for i in range(batch_size):
            results.append((batch_data[i], pred_batch_np[i], 
                          batch_canvases[i], batch_crops[i]))
        
        return results