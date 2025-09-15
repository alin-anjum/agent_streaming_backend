import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class FrameOverlayEngine:
    """Engine for compositing avatar frames onto slide frames."""
    
    def __init__(self, slide_frames_path: str, output_size: Tuple[int, int] = (1280, 720), dynamic_mode: bool = False):
        self.slide_frames_path = Path(slide_frames_path)
        self.output_size = output_size
        self.slide_frames = {}  # Cache for slide frames
        self.dynamic_mode = dynamic_mode  # Flag for dynamic capture mode
        
        # Check for GPU acceleration
        try:
            import torch
            self.use_gpu = torch.cuda.is_available()
            if self.use_gpu:
                self.device = torch.device('cuda')
                logger.info("🚀 Frame processor GPU acceleration enabled")
            else:
                self.device = torch.device('cpu')
                logger.info("💻 Frame processor using CPU")
        except ImportError:
            self.use_gpu = False
            self.device = None
            logger.info("💻 Frame processor using CPU (PyTorch not available)")
        
        # For dynamic capture, always load fresh (no cache) to ensure real-time updates
        if self.dynamic_mode:
            logger.info("🎬 Dynamic mode - skipping cache, minimal frame loading")
            # For dynamic mode, we don't load any frames here - they come from cache system
        else:
            # Try to load from cache first, then load fresh if needed
            if not self.load_from_cache():
                logger.info(f"🔍 Looking for frames in: {self.slide_frames_path.resolve()}")
                self.load_slide_frames()
                if len(self.slide_frames) > 0:
                    # Save to cache for static frames, but dynamic frames will reload as needed
                    self.save_to_cache()
                else:
                    logger.error(f"❌ No frames loaded! Check path: {self.slide_frames_path}")
        
    def load_slide_frames(self):
        """Load all slide frames into memory with optimized parallel processing."""
        try:
            frame_files = sorted(list(self.slide_frames_path.glob("frame_*.png")))
            total_frames = len(frame_files)
            logger.info(f"Loading {total_frames} slide frames into memory...")
            
            import concurrent.futures
            import threading
            
            def load_and_resize_frame(args):
                i, frame_file = args
                try:
                    frame = Image.open(frame_file)
                    # Resize to output size if needed
                    if frame.size != self.output_size:
                        frame = frame.resize(self.output_size, Image.Resampling.LANCZOS)
                    return i, frame
                except Exception as e:
                    logger.error(f"Error loading frame {frame_file}: {e}")
                    return i, None
            
            # Use ThreadPoolExecutor for parallel loading
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Progress tracking
                loaded_count = 0
                
                # Submit all tasks
                future_to_index = {
                    executor.submit(load_and_resize_frame, (i, frame_file)): i 
                    for i, frame_file in enumerate(frame_files)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_index):
                    i, frame = future.result()
                    if frame is not None:
                        self.slide_frames[i] = frame
                    
                    loaded_count += 1
                    
                    # Progress logging every 500 frames
                    if loaded_count % 500 == 0 or loaded_count == total_frames:
                        progress = (loaded_count / total_frames) * 100
                        logger.info(f"📊 Loading progress: {loaded_count}/{total_frames} ({progress:.1f}%)")
                
            logger.info(f"✅ Loaded {len(self.slide_frames)} slide frames into memory!")
            
        except Exception as e:
            logger.error(f"Error loading slide frames: {e}")
            raise
    
    def get_cache_path(self) -> Path:
        """Get cache file path based on frames path and output size"""
        cache_name = f"frames_cache_{self.output_size[0]}x{self.output_size[1]}.pkl"
        return self.slide_frames_path.parent / cache_name
    
    def load_from_cache(self) -> bool:
        """Load frames from cache if available and valid"""
        try:
            import pickle
            cache_path = self.get_cache_path()
            
            if not cache_path.exists():
                logger.info("No frame cache found, will load fresh")
                return False
            
            # Check if cache is newer than source frames
            cache_time = cache_path.stat().st_mtime
            
            # Check a few sample frame files for modification time
            sample_files = list(self.slide_frames_path.glob("frame_*.png"))[:10]
            if sample_files:
                newest_frame_time = max(f.stat().st_mtime for f in sample_files)
                if newest_frame_time > cache_time:
                    logger.info("Frame cache is outdated, will reload")
                    return False
            
            # Load from cache
            logger.info(f"📦 Loading frames from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            if cached_data.get('output_size') == self.output_size:
                self.slide_frames = cached_data['frames']
                logger.info(f"✅ Loaded {len(self.slide_frames)} frames from cache instantly!")
                return True
            else:
                logger.info("Cache output size mismatch, will reload")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will load fresh")
            return False
    
    def save_to_cache(self):
        """Save loaded frames to cache for next time"""
        try:
            import pickle
            cache_path = self.get_cache_path()
            
            logger.info(f"💾 Saving frames to cache: {cache_path}")
            
            cache_data = {
                'frames': self.slide_frames,
                'output_size': self.output_size,
                'frame_count': len(self.slide_frames)
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"✅ Frames cached successfully!")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_slide_frame(self, frame_index: int) -> Optional[Image.Image]:
        """Get slide frame by index - STATIC FRAMES ONLY (dynamic uses cache)."""
        # For static frames, just return from loaded frames
        if frame_index >= len(self.slide_frames) and len(self.slide_frames) > 0:
            frame_index = frame_index % len(self.slide_frames)
        
        return self.slide_frames.get(frame_index)
    
    def overlay_avatar_on_slide(
        self,
        slide_frame: Image.Image,
        avatar_frame: Image.Image,
        position: str = "bottom-right",
        scale: float = 0.3,
        offset: Tuple[int, int] = (50, 50),
        blend_mode: str = "normal",
        fixed_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Overlay avatar frame onto slide frame.
        
        Args:
            slide_frame: Background slide frame
            avatar_frame: Avatar frame to overlay
            position: Position for avatar ("bottom-right", "bottom-left", "top-right", "top-left", "center", "center-bottom")
            scale: Scale factor for avatar frame (ignored if fixed_size is provided)
            offset: Offset from edge in pixels (x, y)
            blend_mode: Blending mode ("normal", "multiply", "screen", "overlay")
            fixed_size: Fixed dimensions (width, height) for avatar. If provided, overrides scale.
            
        Returns:
            PIL.Image: Composited frame
        """
        try:
            # Create a copy of the slide frame
            result_frame = slide_frame.copy()
            
            # Determine avatar dimensions
            if fixed_size is not None:
                # Use fixed dimensions
                avatar_width, avatar_height = fixed_size
            else:
                # Use scale factor
                avatar_width = int(avatar_frame.width * scale)
                avatar_height = int(avatar_frame.height * scale)
            
            # Resize avatar frame
            scaled_avatar = avatar_frame.resize((avatar_width, avatar_height), Image.Resampling.LANCZOS)
            
            # Calculate position
            x, y = self._calculate_position(
                result_frame.size,
                (avatar_width, avatar_height),
                position,
                offset
            )
            
            # Apply blending
            if blend_mode == "normal":
                # Standard alpha compositing
                if scaled_avatar.mode == "RGBA":
                    result_frame.paste(scaled_avatar, (x, y), scaled_avatar)
                else:
                    result_frame.paste(scaled_avatar, (x, y))
            else:
                # Custom blend modes
                result_frame = self._apply_blend_mode(result_frame, scaled_avatar, (x, y), blend_mode)
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Error overlaying avatar on slide: {e}")
            return slide_frame  # Return original slide frame on error
    
    def _calculate_position(
        self,
        canvas_size: Tuple[int, int],
        overlay_size: Tuple[int, int],
        position: str,
        offset: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate overlay position based on position string."""
        canvas_w, canvas_h = canvas_size
        overlay_w, overlay_h = overlay_size
        offset_x, offset_y = offset
        
        if position == "bottom-right":
            x = canvas_w - overlay_w - offset_x
            y = canvas_h - overlay_h - offset_y
        elif position == "bottom-left":
            x = offset_x
            y = canvas_h - overlay_h - offset_y
        elif position == "top-right":
            x = canvas_w - overlay_w - offset_x
            y = offset_y
        elif position == "top-left":
            x = offset_x
            y = offset_y
        elif position == "center":
            x = (canvas_w - overlay_w) // 2
            y = (canvas_h - overlay_h) // 2
        elif position == "center-bottom":
            x = (canvas_w - overlay_w) // 2  # Center horizontally
            y = canvas_h - overlay_h - offset_y  # Bottom edge matches screen bottom
        else:
            # Default to bottom-right
            x = canvas_w - overlay_w - offset_x
            y = canvas_h - overlay_h - offset_y
        
        # Ensure position is within bounds
        x = max(0, min(x, canvas_w - overlay_w))
        y = max(0, min(y, canvas_h - overlay_h))
        
        return (x, y)
    
    def _apply_blend_mode(
        self,
        base: Image.Image,
        overlay: Image.Image,
        position: Tuple[int, int],
        blend_mode: str
    ) -> Image.Image:
        """Apply custom blend modes."""
        # Convert to numpy arrays for blend operations
        base_array = np.array(base.convert("RGB"))
        overlay_array = np.array(overlay.convert("RGB"))
        
        x, y = position
        overlay_h, overlay_w = overlay_array.shape[:2]
        
        # Extract region of interest from base
        roi = base_array[y:y+overlay_h, x:x+overlay_w]
        
        if blend_mode == "multiply":
            blended = (roi.astype(float) * overlay_array.astype(float) / 255.0).astype(np.uint8)
        elif blend_mode == "screen":
            blended = (255 - ((255 - roi.astype(float)) * (255 - overlay_array.astype(float)) / 255.0)).astype(np.uint8)
        elif blend_mode == "overlay":
            # Simplified overlay blend
            mask = roi < 128
            blended = np.where(
                mask,
                (2 * roi.astype(float) * overlay_array.astype(float) / 255.0).astype(np.uint8),
                (255 - 2 * (255 - roi.astype(float)) * (255 - overlay_array.astype(float)) / 255.0).astype(np.uint8)
            )
        else:
            blended = overlay_array  # Fallback to normal
        
        # Apply blended region back to base
        result_array = base_array.copy()
        result_array[y:y+overlay_h, x:x+overlay_w] = blended
        
        return Image.fromarray(result_array)
    
    def create_avatar_mask(
        self,
        avatar_frame: Image.Image,
        mask_type: str = "circular",
        feather: int = 5
    ) -> Image.Image:
        """
        Create a mask for the avatar frame.
        
        Args:
            avatar_frame: Avatar frame to create mask for
            mask_type: Type of mask ("circular", "rounded_rect", "none")
            feather: Feather amount for soft edges
            
        Returns:
            PIL.Image: Mask image
        """
        width, height = avatar_frame.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        if mask_type == "circular":
            # Create circular mask
            radius = min(width, height) // 2
            center_x, center_y = width // 2, height // 2
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill=255
            )
        elif mask_type == "rounded_rect":
            # Create rounded rectangle mask
            corner_radius = min(width, height) // 10
            draw.rounded_rectangle([0, 0, width, height], radius=corner_radius, fill=255)
        else:
            # No mask - full rectangle
            draw.rectangle([0, 0, width, height], fill=255)
        
        # Apply feathering
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        
        return mask
    
    def apply_avatar_effects(
        self,
        avatar_frame: Image.Image,
        effects: Dict[str, Any]
    ) -> Image.Image:
        """
        Apply visual effects to avatar frame.
        
        Args:
            avatar_frame: Avatar frame to process
            effects: Dictionary of effects to apply
            
        Returns:
            PIL.Image: Processed avatar frame
        """
        result = avatar_frame.copy()
        
        # Apply brightness adjustment
        if "brightness" in effects:
            brightness = effects["brightness"]  # -100 to 100
            if brightness != 0:
                result = self._adjust_brightness(result, brightness)
        
        # Apply contrast adjustment
        if "contrast" in effects:
            contrast = effects["contrast"]  # -100 to 100
            if contrast != 0:
                result = self._adjust_contrast(result, contrast)
        
        # Apply saturation adjustment
        if "saturation" in effects:
            saturation = effects["saturation"]  # -100 to 100
            if saturation != 0:
                result = self._adjust_saturation(result, saturation)
        
        # Apply blur
        if "blur" in effects:
            blur_radius = effects["blur"]  # 0 to 10
            if blur_radius > 0:
                result = result.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Apply shadow
        if "shadow" in effects:
            shadow_config = effects["shadow"]
            result = self._add_shadow(result, shadow_config)
        
        return result
    
    def _adjust_brightness(self, image: Image.Image, brightness: float) -> Image.Image:
        """Adjust image brightness."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        factor = 1 + (brightness / 100.0)  # -1 to 2
        return enhancer.enhance(max(0, factor))
    
    def _adjust_contrast(self, image: Image.Image, contrast: float) -> Image.Image:
        """Adjust image contrast."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        factor = 1 + (contrast / 100.0)  # -1 to 2
        return enhancer.enhance(max(0, factor))
    
    def _adjust_saturation(self, image: Image.Image, saturation: float) -> Image.Image:
        """Adjust image saturation."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        factor = 1 + (saturation / 100.0)  # -1 to 2
        return enhancer.enhance(max(0, factor))
    
    def _add_shadow(self, image: Image.Image, shadow_config: Dict[str, Any]) -> Image.Image:
        """Add drop shadow to image."""
        offset_x = shadow_config.get("offset_x", 5)
        offset_y = shadow_config.get("offset_y", 5)
        blur_radius = shadow_config.get("blur", 3)
        opacity = shadow_config.get("opacity", 128)  # 0-255
        
        # Create shadow
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        
        # Create shadow shape (simplified - just a rectangle)
        shadow_draw.rectangle([0, 0, image.width, image.height], fill=(0, 0, 0, opacity))
        
        # Blur shadow
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Create result canvas
        canvas_width = image.width + abs(offset_x) + blur_radius * 2
        canvas_height = image.height + abs(offset_y) + blur_radius * 2
        result = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        
        # Paste shadow
        shadow_x = blur_radius + max(0, offset_x)
        shadow_y = blur_radius + max(0, offset_y)
        result.paste(shadow, (shadow_x, shadow_y), shadow)
        
        # Paste original image
        image_x = blur_radius + max(0, -offset_x)
        image_y = blur_radius + max(0, -offset_y)
        result.paste(image, (image_x, image_y), image if image.mode == "RGBA" else None)
        
        return result
    
    def get_frame_count(self) -> int:
        """Get total number of slide frames."""
        return len(self.slide_frames)
    
    def batch_process_frames(
        self,
        avatar_frames: Dict[int, Image.Image],
        config: Dict[str, Any]
    ) -> Dict[int, Image.Image]:
        """
        Process multiple frames in batch.
        
        Args:
            avatar_frames: Dictionary of frame_index -> avatar_frame
            config: Processing configuration
            
        Returns:
            Dict[int, Image.Image]: Processed frames
        """
        processed_frames = {}
        
        for frame_index, avatar_frame in avatar_frames.items():
            slide_frame = self.get_slide_frame(frame_index)
            if slide_frame:
                # Apply effects to avatar if specified
                if "avatar_effects" in config:
                    avatar_frame = self.apply_avatar_effects(avatar_frame, config["avatar_effects"])
                
                # Create mask if specified
                if config.get("use_mask", False):
                    mask = self.create_avatar_mask(
                        avatar_frame,
                        config.get("mask_type", "circular"),
                        config.get("mask_feather", 5)
                    )
                    # Apply mask to avatar
                    avatar_frame.putalpha(mask)
                
                # Overlay avatar on slide
                result_frame = self.overlay_avatar_on_slide(
                    slide_frame,
                    avatar_frame,
                    config.get("position", "bottom-right"),
                    config.get("scale", 0.3),
                    config.get("offset", (50, 50)),
                    config.get("blend_mode", "normal")
                )
                
                processed_frames[frame_index] = result_frame
        
        return processed_frames
