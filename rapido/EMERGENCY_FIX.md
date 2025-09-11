# EMERGENCY FIX - Audio Still Bad

## Changes Made

### 1. Reduced Buffer Thresholds
- **Min playback buffer**: 160ms → 40ms (only generate silence when critically low)
- **Initial requirement**: 320ms → 256ms (start playback sooner)

### 2. Disabled Audio Smoother
- **TEMPORARILY DISABLED** to test if it's causing issues
- Can be re-enabled by uncommenting in rapido_main.py line 198

### 3. Size Check Added
- Audio smoother now verifies it returns same size audio

## Current Status
- Buffer settings optimized for less aggressive silence generation
- Audio smoother disabled to eliminate it as a variable
- Should reduce underruns significantly

## To Test
Run the system again and check:
1. Are underruns reduced? (was 36)
2. Is buffer maintaining better? (should stay 200-320ms)
3. Is audio quality better without smoother?

## If Still Bad
The issue might be:
1. **TTS latency** - ElevenLabs might be slow
2. **Network issues** - WebSocket delays
3. **CPU overload** - System can't keep up
4. **SyncTalk delays** - Processing bottleneck

## Quick Rollback
To re-enable smoother:
```python
# Line 198 in rapido_main.py
self.audio_smoother = AudioSmoother(sample_rate=16000, crossfade_ms=2.0)
```
