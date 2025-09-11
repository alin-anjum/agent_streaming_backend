import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
import logging

# Try to import CUDA/GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Try to import numba for JIT compilation
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


def download_background(url: str, width: int, height: int) -> np.ndarray:
    """
    Download and prepare background image
    Returns RGB array of specified dimensions (height, width, 3)
    """
    try:
        # Download image
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        
        # Force RGB mode (in case it's RGBA or another format)
        if img.mode != 'RGB':
            logger.info('Converting background to RGB')
            img = img.convert('RGB')
        
        # Resize to match dimensions
        img = img.resize((width, height))
        
        # Convert to numpy array (will be RGB)
        bg_array = np.array(img)
        
        # Log confirmation
        logger.info(f"Background loaded successfully: {url}, shape: {bg_array.shape}")
        
        return bg_array
    except Exception as e:
        logger.error(f"Error loading background image: {e}")
        # Return black background as fallback
        return np.zeros((height, width, 3), dtype=np.uint8)


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# Numba JIT-compiled functions for CPU optimization
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def create_mask_numba(r, g, b, color_threshold, lower_g, upper_g):
        """Fast mask creation using Numba JIT compilation"""
        height, width = r.shape
        mask = np.zeros((height, width), dtype=np.float32)
        
        for i in range(height):
            for j in range(width):
                g_val = g[i, j]
                r_val = r[i, j]
                b_val = b[i, j]
                
                # Green dominance check
                green_dominance = (g_val > r_val + color_threshold) and (g_val > b_val + color_threshold)
                # Color range check
                color_range = (g_val > lower_g) and (g_val < upper_g)
                
                if green_dominance and color_range:
                    mask[i, j] = 1.0
        
        return mask

    @jit(nopython=True, cache=True)
    def apply_despill_numba(frame, mask, despill_factor):
        """Fast despill correction using Numba"""
        height, width = frame.shape[:2]
        
        for i in range(height):
            for j in range(width):
                if mask[i, j] < 0.9:  # Only process non-green areas
                    r_val = frame[i, j, 0]
                    g_val = frame[i, j, 1]
                    b_val = frame[i, j, 2]
                    
                    avg_rb = (r_val + b_val) * 0.5
                    if g_val > avg_rb * 1.05:  # Subtle green spill
                        green_excess = g_val - avg_rb
                        reduction = green_excess * despill_factor
                        new_g = max(0, min(255, g_val - reduction))
                        frame[i, j, 1] = new_g
        
        return frame


def create_cuda_kernels():
    """Create CUDA kernels for GPU processing"""
    if not NUMBA_AVAILABLE:
        return None, None
    
    @cuda.jit
    def cuda_create_mask(r, g, b, mask, color_threshold, lower_g, upper_g):
        """CUDA kernel for mask creation"""
        i, j = cuda.grid(2)
        if i < r.shape[0] and j < r.shape[1]:
            g_val = g[i, j]
            r_val = r[i, j]
            b_val = b[i, j]
            
            green_dominance = (g_val > r_val + color_threshold) and (g_val > b_val + color_threshold)
            color_range = (g_val > lower_g) and (g_val < upper_g)
            
            mask[i, j] = 1.0 if (green_dominance and color_range) else 0.0
    
    @cuda.jit
    def cuda_blend_frames(frame, background, mask, result):
        """CUDA kernel for frame blending"""
        i, j, c = cuda.grid(3)
        if i < frame.shape[0] and j < frame.shape[1] and c < frame.shape[2]:
            mask_val = mask[i, j]
            result[i, j, c] = frame[i, j, c] * (1.0 - mask_val) + background[i, j, c] * mask_val
    
    return cuda_create_mask, cuda_blend_frames


class FastChromaKey:
    def __init__(self, 
                 width: int,
                 height: int,
                 background: np.ndarray,  # RGB array
                 target_color: str = '#bcfeb6',
                 color_threshold: int = 35,
                 edge_blur: float = 0.4,
                 despill_factor: float = 0.9,
                 use_gpu: bool = True):
        # Store dimensions
        self.width = width
        self.height = height
        
        # Determine processing mode
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.use_cuda_kernels = use_gpu and NUMBA_AVAILABLE
        self.use_numba = not self.use_gpu and NUMBA_AVAILABLE
        
        logger.info(f"ChromaKey acceleration: GPU={self.use_gpu}, CUDA_kernels={self.use_cuda_kernels}, Numba={self.use_numba}")
        
        # Background safety check - ensure RGB format and match dimensions
        expected_shape = (self.height, self.width, 3)
        if background.shape != expected_shape:
            logger.warning(f"Background shape {background.shape} doesn't match expected {expected_shape}")
            try:
                # Fix background if needed
                if len(background.shape) == 3 and background.shape[2] == 4:
                    # RGBA to RGB
                    logger.info("Converting background from RGBA to RGB")
                    background = background[:, :, :3]
                elif len(background.shape) == 2:
                    # Grayscale to RGB
                    logger.info("Converting background from grayscale to RGB")
                    background = np.stack([background] * 3, axis=2)
                
                # Ensure correct dimensions
                if background.shape[0] != self.height or background.shape[1] != self.width:
                    logger.info(f"Resizing background from {background.shape[:2]} to {(self.height, self.width)}")
                    background = cv2.resize(background, (self.width, self.height))
            except Exception as e:
                logger.error(f"Failed to convert background: {e}, using black background")
                background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Store background - convert to GPU if using GPU acceleration
        if self.use_gpu:
            self.background = cp.asarray(background, dtype=cp.uint8)
        else:
            self.background = background.astype(np.uint8)
        
        # Precompute color parameters for RGB format
        self.target_color = np.array(hex_to_rgb(target_color))  # RGB format
        
        # Adjust thresholds for better edge detection
        self.color_threshold = color_threshold
        self.lower_g = self.target_color[1] - color_threshold
        self.upper_g = self.target_color[1] + color_threshold
        
        self.edge_blur = edge_blur
        self.despill_factor = despill_factor
        
        # Pre-allocate buffers based on processing mode
        if self.use_gpu:
            # GPU buffers
            self.mask_gpu = cp.zeros((self.height, self.width), dtype=cp.float32)
            self.mask_3d_gpu = cp.zeros((self.height, self.width, 3), dtype=cp.float32)
            self.result_gpu = cp.zeros((self.height, self.width, 3), dtype=cp.uint8)
            
            # Precompute blur kernel for GPU
            if self.edge_blur > 0:
                kernel_size = int(max(3, self.edge_blur * 20)) | 1
                self.kernel_size = kernel_size
                # Create Gaussian kernel on GPU
                self._create_gaussian_kernel_gpu()
        else:
            # CPU buffers
            self.mask = np.zeros((self.height, self.width), dtype=np.float32)
            self.mask_3d = np.zeros((self.height, self.width, 3), dtype=np.float32)
            self.result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Precompute blur kernel for CPU
            if self.edge_blur > 0:
                kernel_size = int(max(3, self.edge_blur * 20)) | 1
                self.kernel_size = (kernel_size, kernel_size)
        
        # Initialize CUDA kernels if available
        if self.use_cuda_kernels:
            self.cuda_mask_kernel, self.cuda_blend_kernel = create_cuda_kernels()
            # Pre-calculate grid dimensions
            self.grid_2d = ((self.height + 15) // 16, (self.width + 15) // 16)
            self.block_2d = (16, 16)
            self.grid_3d = ((self.height + 7) // 8, (self.width + 7) // 8, (3 + 1) // 2)
            self.block_3d = (8, 8, 2)
        
        logger.info(f"ChromaKey initialized: {self.width}x{self.height}")
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = None
        self.processing_times = []

    def process(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Process a video frame to replace chroma key background with GPU acceleration
        
        Args:
            frame_data: Input frame as numpy array (height, width, 3) in RGB format
            
        Returns:
            Processed frame with background replaced
        """
        process_start = time.time()
        
        if self.last_log_time is None:
            self.last_log_time = time.time()
        self.frame_count += 1
        
        # Prepare frame data
        frame = self._prepare_frame(frame_data)
        
        try:
            if self.use_gpu:
                result = self._process_gpu(frame)
            elif self.use_numba:
                result = self._process_numba(frame)
            else:
                result = self._process_cpu(frame)
        except Exception as e:
            # Fallback to CPU processing on any GPU error
            logger.warning(f"GPU processing failed, falling back to CPU: {e}")
            result = self._process_cpu(frame)
        
        # Track processing time
        process_time = time.time() - process_start
        self.processing_times.append(process_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        # Log performance
        if self.frame_count % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.last_log_time
            fps = 100 / elapsed_time if elapsed_time > 0 else 0
            average_fps = self.frame_count / (current_time - self.start_time)
            avg_process_time = np.mean(self.processing_times) * 1000  # ms
            
            logger.info(f"ChromaKey processed {self.frame_count} frames | "
                    f"Current FPS: {fps:.2f} | "
                    f"Average FPS: {average_fps:.2f} | "
                    f"Avg process time: {avg_process_time:.2f}ms | "
                    f"Mode: {'GPU' if self.use_gpu else 'Numba' if self.use_numba else 'CPU'}")
            
            self.last_log_time = current_time
        
        return result

    def _create_gaussian_kernel_gpu(self):
        """Create Gaussian kernel on GPU for blur operations"""
        if not self.use_gpu:
            return
            
        # Create 1D Gaussian kernel
        sigma = self.edge_blur * 10
        kernel_1d = cp.exp(-0.5 * (cp.arange(self.kernel_size) - self.kernel_size // 2) ** 2 / sigma ** 2)
        kernel_1d = kernel_1d / cp.sum(kernel_1d)
        
        # Store for separable convolution
        self.blur_kernel_gpu = kernel_1d

    def _gaussian_blur_gpu(self, image_gpu):
        """Fast separable Gaussian blur on GPU using CuPy with optimized convolution"""
        if self.edge_blur <= 0 or not self.use_gpu:
            return image_gpu
        
        try:
            # Use CuPy's built-in convolution if available
            import cupyx.scipy.ndimage as ndi_gpu
            
            # Convert 1D kernel to 2D for Gaussian filter
            sigma = self.edge_blur * 10
            return ndi_gpu.gaussian_filter(image_gpu, sigma=sigma, mode='nearest')
            
        except ImportError:
            # Fallback to manual separable convolution
            padding = self.kernel_size // 2
            
            # Horizontal pass
            padded = cp.pad(image_gpu, ((0, 0), (padding, padding)), mode='edge')
            h_blurred = cp.zeros_like(image_gpu)
            
            for i, weight in enumerate(self.blur_kernel_gpu):
                h_blurred += padded[:, i:i + image_gpu.shape[1]] * weight
            
            # Vertical pass
            padded = cp.pad(h_blurred, ((padding, padding), (0, 0)), mode='edge')
            result = cp.zeros_like(image_gpu)
            
            for i, weight in enumerate(self.blur_kernel_gpu):
                result += padded[i:i + image_gpu.shape[0], :] * weight
                
            return result

    def _prepare_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """Prepare frame data for processing"""
        # Ensure frame is in correct format
        if len(frame_data.shape) == 1:
            # Flat array - reshape it using dimensions
            total_pixels = self.height * self.width
            if len(frame_data) == total_pixels * 4:
                # RGBA
                frame = frame_data.reshape(self.height, self.width, 4)[:, :, :3].copy()
            else:
                # RGB
                frame = frame_data.reshape(self.height, self.width, 3)
        else:
            # Already shaped array
            if len(frame_data.shape) == 3 and frame_data.shape[2] == 4:
                # RGBA to RGB
                frame = frame_data[:, :, :3].copy()
            else:
                frame = frame_data.copy()
        
        # Ensure frame matches expected dimensions
        if frame.shape[:2] != (self.height, self.width):
            logger.warning(f"Frame dimensions {frame.shape[:2]} don't match expected {(self.height, self.width)}, resizing")
            frame = cv2.resize(frame, (self.width, self.height))
        
        return frame.astype(np.uint8)



    def _process_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-accelerated processing using CuPy"""
        # Transfer frame to GPU
        frame_gpu = cp.asarray(frame, dtype=cp.uint8)
        
        # Extract channels
        r, g, b = frame_gpu[:, :, 0], frame_gpu[:, :, 1], frame_gpu[:, :, 2]
        
        # Create mask using vectorized operations on GPU
        green_dominance = (g > (r + self.color_threshold)) & (g > (b + self.color_threshold))
        color_range = (g > self.lower_g) & (g < self.upper_g)
        self.mask_gpu = (green_dominance & color_range).astype(cp.float32)
        
        # Apply edge blur using pure CuPy
        if self.edge_blur > 0:
            self.mask_gpu = self._gaussian_blur_gpu(self.mask_gpu)
            self.mask_gpu = cp.clip(self.mask_gpu, 0, 1)
        
        # Simplified despill on GPU
        if self.despill_factor > 0:
            # Fast despill using vectorized operations
            avg_rb = (r.astype(cp.float32) + b.astype(cp.float32)) * 0.5
            green_excess = g.astype(cp.float32) - avg_rb
            spill_mask = (g > avg_rb * 1.05).astype(cp.float32)
            despill_amount = cp.clip(green_excess * self.despill_factor * spill_mask, 0, None)
            
            # Apply correction
            g_corrected = cp.clip(g.astype(cp.float32) - despill_amount, 0, 255)
            frame_gpu = frame_gpu.astype(cp.float32)
            frame_gpu[:, :, 1] = g_corrected
            frame_gpu = frame_gpu.astype(cp.uint8)
        
        # Create 3D mask and blend
        self.mask_3d_gpu = cp.repeat(self.mask_gpu[:, :, cp.newaxis], 3, axis=2)
        
        # Vectorized blending on GPU
        frame_float = frame_gpu.astype(cp.float32)
        bg_float = self.background.astype(cp.float32)
        
        self.result_gpu = (frame_float * (1 - self.mask_3d_gpu) + 
                          bg_float * self.mask_3d_gpu).astype(cp.uint8)
        
        # Transfer result back to CPU
        return cp.asnumpy(self.result_gpu)

    def _process_numba(self, frame: np.ndarray) -> np.ndarray:
        """Numba JIT-accelerated processing"""
        # Extract channels
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        
        # Create mask using Numba
        self.mask = create_mask_numba(r, g, b, self.color_threshold, self.lower_g, self.upper_g)
        
        # Apply edge blur
        if self.edge_blur > 0:
            # Ensure kernel_size is a tuple for OpenCV
            ksize = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
            cv2.GaussianBlur(self.mask, ksize, 0, dst=self.mask)
            self.mask = np.clip(self.mask, 0, 1)
        
        # Apply despill using Numba
        if self.despill_factor > 0:
            frame = apply_despill_numba(frame, self.mask, self.despill_factor)
        
        # Create 3D mask and blend
        self.mask_3d = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2)
        
        # Vectorized blending
        frame_float = frame.astype(np.float32)
        bg_float = self.background.astype(np.float32)
        
        result = (frame_float * (1 - self.mask_3d) + bg_float * self.mask_3d)
        
        return result.astype(np.uint8)

    def _process_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Optimized CPU processing"""
        # Extract channels
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        
        # Create mask with vectorized operations
        green_dominance = (g > (r + self.color_threshold)) & (g > (b + self.color_threshold))
        color_range = (g > self.lower_g) & (g < self.upper_g)
        self.mask = (green_dominance & color_range).astype(np.float32)
        
        # Apply edge blur
        if self.edge_blur > 0:
            # Ensure kernel_size is a tuple for OpenCV
            ksize = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
            cv2.GaussianBlur(self.mask, ksize, 0, dst=self.mask)
            self.mask = np.clip(self.mask, 0, 1)
        
        # Simplified despill
        if self.despill_factor > 0:
            avg_rb = (r.astype(np.float32) + b.astype(np.float32)) * 0.5
            green_excess = g.astype(np.float32) - avg_rb
            spill_mask = (g > avg_rb * 1.05).astype(np.float32)
            despill_amount = np.clip(green_excess * self.despill_factor * spill_mask, 0, None)
            
            # Apply correction
            g_corrected = np.clip(g.astype(np.float32) - despill_amount, 0, 255)
            frame = frame.astype(np.float32)
            frame[:, :, 1] = g_corrected
            frame = frame.astype(np.uint8)
        
        # Create 3D mask and blend
        self.mask_3d = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2)
        
        # Vectorized blending
        frame_float = frame.astype(np.float32)
        bg_float = self.background.astype(np.float32)
        
        result = (frame_float * (1 - self.mask_3d) + bg_float * self.mask_3d)
        
        return result.astype(np.uint8)


def create_chroma_key_processor(
    width: int, 
    height: int,
    background_image_url: Optional[str] = None, 
    target_color: str = '#00FF00',  # Hex color of your green screen
    color_threshold: int = 35,
    edge_blur: float = 0.4,
    despill_factor: float = 0.9,
    use_gpu: bool = True
) -> Optional[FastChromaKey]:
    """
    Creates a chroma key processor with configurable parameters for green screen removal.
    
    Args:
        width: Input frame width
        height: Input frame height  
        background_image_url: URL to background image (optional)
        target_color: Hex color of chroma key (default green screen)
        color_threshold: Color matching threshold
        edge_blur: Edge blur amount
        despill_factor: Green spill removal factor
        use_gpu: Enable GPU acceleration if available
        
    Returns:
        FastChromaKey processor or None if disabled
    """
    if background_image_url is None:
        return None
        
    try: 
        # Download background and resize to match output size
        background = download_background(background_image_url, width, height)

        # Log acceleration capabilities
        if use_gpu and CUPY_AVAILABLE:
            logger.info("Creating GPU-accelerated chroma key processor (CuPy)")
        elif NUMBA_AVAILABLE:
            logger.info("Creating JIT-accelerated chroma key processor (Numba)")
        else:
            logger.info("Creating CPU-optimized chroma key processor")

        return FastChromaKey(
            width=width,
            height=height,
            background=background,
            target_color=target_color,
            color_threshold=color_threshold,
            edge_blur=edge_blur,
            despill_factor=despill_factor,
            use_gpu=use_gpu
        )
    except Exception as e:
        logger.error(f"Exception in create_chroma_key_processor: {e}")
        # Return processor with black background as fallback
        return FastChromaKey(
            width=width,
            height=height,
            background=np.zeros((height, width, 3), dtype=np.uint8),
            target_color=target_color,
            color_threshold=color_threshold,
            edge_blur=edge_blur,
            despill_factor=despill_factor,
            use_gpu=use_gpu
        )
