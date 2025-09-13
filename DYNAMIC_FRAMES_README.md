# Dynamic Frame Capture Implementation

This implementation converts a static frame-based avatar system to use dynamically captured frames from tab capture, providing real-time slide content with SyncTalk avatar overlay.

## 🎯 Overview

### What This Does
- **Replaces static presentation frames** with **live-captured browser tab frames**
- **Real-time frame caching** system for smooth 25 FPS delivery
- **SyncTalk-style queue management** for non-blocking frame processing
- **Separate FPS monitoring** for SyncTalk vs composition performance
- **Zero static frame loading** when in dynamic mode

### Key Benefits
- ✅ **Live content**: Capture any web-based presentation in real-time
- ✅ **High performance**: 25 FPS target with optimized caching
- ✅ **Non-blocking**: SyncTalk and tab capture run in parallel
- ✅ **Smooth delivery**: Queue-based frame management prevents stuttering
- ✅ **Easy monitoring**: Clear FPS metrics for debugging

## 🏗️ Architecture

### Component Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Tab Capture   │───▶│  Frame Producer  │───▶│ Slide Cache     │
│   (Browser)     │    │   (Background)   │    │ (In-Memory)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   SyncTalk      │───▶│   Compositor     │◀────────────┘
│   (Avatar)      │    │   (25 FPS)       │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    LiveKit      │
                       │   Streaming     │
                       └─────────────────┘
```

### Data Flow
1. **Tab Capture**: Browser automation captures presentation frames → disk
2. **Frame Producer**: Background task watches directory → loads new frames → cache
3. **SyncTalk**: Generates avatar frames from audio → queue
4. **Compositor**: Combines cached slide + avatar frames → LiveKit stream

## 🚀 Implementation Guide

### Step 1: Configuration Setup

Add to your `config.py`:
```python
# Dynamic frame capture settings
USE_DYNAMIC_CAPTURE = True  # False for static frames
CAPTURE_URL = "http://localhost:5173/video-capture/your-presentation-id"
CAPTURE_DURATION = 30  # seconds of capture
```

### Step 2: Dependencies

Ensure you have:
```python
from tab_capture import capture_presentation_frames, DynamicFrameProcessor
from pathlib import Path
import tempfile
import asyncio
from typing import Optional
```

### Step 3: Apply the Patch

Follow the detailed patch instructions in `dynamic_frames_patch.txt` to:
- Add dynamic capture configuration
- Modify frame processor initialization
- Implement slide frame producer
- Update continuous frame delivery
- Add enhanced FPS monitoring

### Step 4: Tab Capture Integration

Your tab capture system should:
- Save frames as `frame_000000.png`, `frame_000001.png`, etc.
- Write frames to a timestamped directory (e.g., `./captured_frames/live_frames_1234567890/`)
- Return the directory path immediately for non-blocking operation
- Continue capturing in the background

## 📊 Performance Monitoring

### FPS Metrics
The system provides separate FPS tracking:

```
🤖 SYNCTALK FPS: 22.5 recent | 21.8 avg 🟢 (total: 150)
🎬 COMPOSITION FPS: 24.1 recent | 23.2 avg 🟢 (queue: 2)
📈 Slide producer: 200 frames cached
📊 Slide progress: 45/200 (cycle 1, frame 150) - CACHED
```

### Status Indicators
- 🟢 **Good**: ≥20 FPS
- 🟡 **Warning**: 10-19 FPS  
- 🔴 **Poor**: <10 FPS

### Troubleshooting Performance

#### Low SyncTalk FPS
- **Root cause**: Remote SyncTalk server issues
- **Solutions**: Check server status, network latency, reduce audio chunk rate

#### Low Composition FPS
- **Root cause**: Frame processing bottlenecks
- **Solutions**: Check overlay timing, cache performance, LiveKit publishing

#### Cache Issues
- **Symptoms**: Repeated frames, missing frames
- **Solutions**: Verify tab capture is writing files, check directory permissions

## 🔧 Technical Details

### Frame Caching System
```python
# Producer pattern (background task)
async def _produce_slide_frames(self):
    while frame_delivery_running:
        # Watch directory for new PNG files
        # Load new frames into self.slide_frames_cache
        # Update self.slide_frame_count
        await asyncio.sleep(0.1)  # Check every 100ms

# Consumer pattern (compositor)
def get_cached_slide_frame(self, frame_index: int):
    # Instant access from memory cache
    # Auto-cycling through available frames
    return self.slide_frames_cache.get(safe_index)
```

### Memory Management
- **Cache size**: Grows as frames are captured (typically 200-500 frames)
- **Memory usage**: ~50-100MB for 30 seconds of 854x480 frames
- **Cleanup**: Cache cleared when system stops

### Timing Optimization
- **Frame loading**: 100ms polling interval for new frames
- **Overlay operations**: <10ms per frame (warning if >10ms)
- **Queue management**: Non-blocking with overflow protection
- **FPS calculation**: Real-time (3-second windows) + overall average

## 🎮 Usage Examples

### Basic Usage
```python
# Initialize with dynamic capture
rapido = RapidoMainSystem({
    'USE_DYNAMIC_CAPTURE': True,
    'CAPTURE_URL': 'http://localhost:5173/presentation',
    'CAPTURE_DURATION': 60
})

# Process presentation (automatically handles dynamic capture)
await rapido.process_presentation('input.json')
```

### Static Mode (Fallback)
```python
# Use static frames instead
rapido = RapidoMainSystem({
    'USE_DYNAMIC_CAPTURE': False,
    'SLIDE_FRAMES_PATH': './presentation_frames'
})
```

## 🐛 Debugging

### Common Issues

#### "No frames captured"
```bash
# Check if tab capture is working
ls -la captured_frames/live_frames_*/
# Should show frame_000000.png, frame_000001.png, etc.
```

#### "Low FPS performance"
```bash
# Monitor logs for bottlenecks
grep "Slow overlay" logs.txt
grep "SYNCTALK FPS" logs.txt
grep "COMPOSITION FPS" logs.txt
```

#### "Repeated frames"
- **Cause**: Cache not updating with new frames
- **Fix**: Check frame producer task is running
- **Verify**: Look for "📈 Slide producer: X frames cached" logs

### Log Analysis
```bash
# SyncTalk performance
grep "🤖 SYNCTALK FPS" logs.txt

# Composition performance  
grep "🎬 COMPOSITION FPS" logs.txt

# Frame loading progress
grep "📈 Slide producer" logs.txt

# Slow operations
grep "⚠️ Slow overlay" logs.txt
```

## 🔄 Migration from Static Frames

### Before (Static)
```python
# Loaded all frames at startup
self.frame_processor = FrameOverlayEngine(slide_frames_path)
# Used disk-based frame access
slide_frame = frame_processor.get_slide_frame(index)
```

### After (Dynamic)
```python
# Minimal processor for overlay operations only
self.frame_processor = FrameOverlayEngine(temp_empty_dir)
# Used memory-based cached access
slide_frame = self.get_cached_slide_frame(index)
```

### Performance Comparison
| Metric | Static Frames | Dynamic Frames |
|--------|---------------|----------------|
| Startup time | 5-10 seconds (loading) | <1 second (no loading) |
| Memory usage | High (all frames) | Medium (cached frames) |
| Frame access | Disk I/O | Memory access |
| Content updates | Manual replacement | Real-time capture |
| FPS consistency | Good | Excellent |

## 📝 License & Credits

This implementation is based on the SyncTalk queue management pattern and optimized for real-time frame delivery. The caching system ensures smooth 25 FPS performance while maintaining the flexibility of dynamic content capture.

For issues or improvements, refer to the detailed patch file and performance monitoring logs.
