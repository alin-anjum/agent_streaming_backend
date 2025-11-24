#!/usr/bin/env python3
"""
Cloudflare Live Streaming Client
Streams video frames (PNG) and audio (PCM 16kHz mono) to Cloudflare via RTMPS
"""

import subprocess
import time
import signal
import sys
import os
import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Generator, Callable, Any, Dict
from queue import Queue, Empty, Full
from dataclasses import dataclass
import numpy as np
import json
from urllib import request as urlrequest
from urllib import error as urlerror

# Optional: set Cloudflare credentials as environment variables here
# Replace the placeholders below, or set them externally in your shell.
os.environ.setdefault('CLOUDFLARE_ACCOUNT_ID', '734078922139028728d981b1e8c49b3d')
os.environ.setdefault('CLOUDFLARE_API_TOKEN', 'EV1O-l0jLw_CH6P1s7MetJjryTQH0m6DQnM6HBVU')
os.environ.setdefault('CLOUDFLARE_CUSTOMER_CODE', 'z9q1p2plut4sfb4q')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for the stream"""
    rtmps_url: str
    stream_key: str
    width: int = 1920
    height: int = 1080
    fps: int = 25
    video_bitrate: str = "2500k"
    audio_sample_rate: int = 16000
    audio_channels: int = 1  # Mono
    audio_bitrate: str = "64k"
    # Auto-create Cloudflare Live Input before streaming
    auto_create_live_input: bool = False
    cloudflare_account_id: Optional[str] = None
    cloudflare_api_token: Optional[str] = None
    stream_name: Optional[str] = None
    playback_url: Optional[str] = None
    live_input_uid: Optional[str] = None
    # Output options for passing data downstream
    write_playback_json: Optional[str] = None
    emit_playback_stdout: bool = False
    await_playback_url_seconds: int = 0
    # Buffer sizes
    frame_buffer_size: int = 150
    audio_buffer_size: int = 500
    # Cloudflare customer code to build HLS URL if API playback is absent
    cloudflare_customer_code: Optional[str] = None
    # Optional Python callback invoked with playback info dict on initial and updates
    on_playback_update: Optional[Callable[[Dict[str, Any]], None]] = None
    # Optional Python callback invoked when all enqueued pairs are fully egressed
    on_egress_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    # Auto-delete recordings after N days (Cloudflare Stream setting)
    delete_recording_after_days: Optional[int] = 30
    # Repeat last frame when no new frame is available to maintain CFR
    enable_frame_hold: bool = True
    
    @property
    def rtmps_full_url(self) -> str:
        """Get the complete RTMPS URL with stream key"""
        return f"{self.rtmps_url}/{self.stream_key}"
    
    @property
    def frame_duration_ms(self) -> float:
        """Duration of each frame in milliseconds"""
        return 1000.0 / self.fps
    
    @property
    def audio_chunk_duration_ms(self) -> float:
        """Duration of audio chunk in milliseconds"""
        return 40.0  # 40ms chunks as specified
    
    @property
    def audio_chunk_size_bytes(self) -> int:
        """Size of audio chunk in bytes"""
        # 16kHz * 0.04s * 1 channel * 2 bytes (16-bit)
        return int(self.audio_sample_rate * 0.04 * self.audio_channels * 2)


class FrameAudioSource(ABC):
    """Abstract base class for frame and audio sources"""
    
    @abstractmethod
    def get_next_frame(self) -> Optional[bytes]:
        """
        Get the next video frame as PNG bytes
        Returns None when no more frames available
        """
        pass
    
    @abstractmethod
    def get_next_audio_chunk(self) -> Optional[bytes]:
        """
        Get the next audio chunk as PCM bytes (16kHz mono, 16-bit signed)
        Returns None when no more audio available
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the source to start from beginning"""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources"""
        pass


class SimulationSource(FrameAudioSource):
    """
    Simulation source that extracts frames and audio from an MP4 file
    Plays the video once and ends (no looping)
    """
    
    def __init__(self, mp4_path: str, config: StreamConfig):
        """
        Initialize simulation source
        
        Args:
            mp4_path: Path to the MP4 file
            config: Stream configuration
        """
        self.mp4_path = mp4_path
        self.config = config
        self.frame_queue = Queue(maxsize=self.config.frame_buffer_size)
        self.audio_queue = Queue(maxsize=self.config.audio_buffer_size)
        self.stop_event = threading.Event()
        self.frame_thread = None
        self.audio_thread = None
        self.frames_done = False
        self.audio_done = False
        
        # Verify file exists
        if not os.path.exists(mp4_path):
            raise FileNotFoundError(f"MP4 file not found: {mp4_path}")
        
        # Start extraction threads
        self._start_extraction()
        logger.info(f"Initialized simulation source from: {mp4_path}")
    
    def _start_extraction(self):
        """Start the frame and audio extraction threads"""
        self.stop_event.clear()
        
        # Start frame extraction thread
        self.frame_thread = threading.Thread(
            target=self._extract_frames,
            daemon=True,
            name="FrameExtractor"
        )
        self.frame_thread.start()
        
        # Start audio extraction thread
        self.audio_thread = threading.Thread(
            target=self._extract_audio,
            daemon=True,
            name="AudioExtractor"
        )
        self.audio_thread.start()
        
        # Wait for initial buffering
        time.sleep(0.5)
    
    def _extract_frames(self):
        """Extract frames from MP4 and convert to PNG"""
        while not self.stop_event.is_set():
            try:
                # FFmpeg command to extract frames as PNG (no looping)
                command = [
                    'ffmpeg',
                    '-i', self.mp4_path,
                    '-vf', f'scale={self.config.width}:{self.config.height}',
                    '-r', str(self.config.fps),
                    '-f', 'image2pipe',
                    '-vcodec', 'png',
                    '-'
                ]
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8  # Large buffer
                )
                
                # Read PNG frames from FFmpeg using a persistent buffer
                png_signature = b'\x89PNG\r\n\x1a\n'
                iend_marker = b'IEND\xae\x42\x60\x82'
                buffer = b''

                while not self.stop_event.is_set():
                    chunk = process.stdout.read(8192)
                    if not chunk:
                        break
                    buffer += chunk

                    # Extract as many PNGs as are fully present in buffer
                    while True:
                        start = buffer.find(png_signature)
                        if start == -1:
                            # Keep only last 7 bytes in case signature spans boundary
                            if len(buffer) > 7:
                                buffer = buffer[-7:]
                            break

                        if start > 0:
                            # Discard any leading noise before PNG signature
                            buffer = buffer[start:]

                        end_index = buffer.find(iend_marker)
                        if end_index == -1:
                            # Need more data for a complete PNG
                            break

                        end = end_index + len(iend_marker)
                        png_data = buffer[:end]
                        buffer = buffer[end:]

                        # Add to queue; block to avoid dropping frames
                        if not self.stop_event.is_set():
                            self.frame_queue.put(png_data)  # block until space
                
                process.terminate()
                process.wait(timeout=5)
                
                # End extraction after source ends
                self.frames_done = True
                break
                    
            except Exception as e:
                logger.error(f"Frame extraction error: {e}")
                time.sleep(1)
    
    def _extract_audio(self):
        """Extract audio from MP4 and convert to 16kHz mono PCM"""
        while not self.stop_event.is_set():
            try:
                # FFmpeg command to extract audio as PCM (no looping)
                command = [
                    'ffmpeg',
                    '-i', self.mp4_path,
                    '-f', 's16le',  # 16-bit signed little-endian PCM
                    '-ar', str(self.config.audio_sample_rate),  # 16kHz
                    '-ac', str(self.config.audio_channels),  # Mono
                    '-'
                ]
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8
                )
                
                chunk_size = self.config.audio_chunk_size_bytes
                last_log_time = time.monotonic()
                chunks_since_log = 0
                
                while not self.stop_event.is_set():
                    audio_chunk = process.stdout.read(chunk_size)
                    
                    if len(audio_chunk) < chunk_size:
                        # End of stream or incomplete chunk
                        if len(audio_chunk) > 0:
                            # Pad with silence
                            audio_chunk += b'\x00' * (chunk_size - len(audio_chunk))
                            self.audio_queue.put(audio_chunk, timeout=1)
                        break
                    
                    # Add to queue; block to avoid dropping audio
                    if not self.stop_event.is_set():
                        self.audio_queue.put(audio_chunk)  # block until space
                        chunks_since_log += 1
                        now = time.monotonic()
                        if now - last_log_time >= 1.0:
                            logger.debug(
                                f"Audio debug - chunk_size: {len(audio_chunk)}, chunks/sec: {chunks_since_log}"
                            )
                            last_log_time = now
                            chunks_since_log = 0
                
                process.terminate()
                process.wait(timeout=5)
                
                # End extraction after source ends
                self.audio_done = True
                break
                    
            except Exception as e:
                logger.error(f"Audio extraction error: {e}")
                time.sleep(1)
    
    def get_next_frame(self) -> Optional[bytes]:
        """Get the next PNG frame"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except Empty:
            if not self.frames_done:
                logger.warning("Frame queue empty")
            return None
    
    def get_next_audio_chunk(self) -> Optional[bytes]:
        """Get the next audio chunk"""
        try:
            return self.audio_queue.get(timeout=0.1)
        except Empty:
            if not self.audio_done:
                logger.warning("Audio queue empty")
            # Return silence only if audio is ongoing; if done, return empty to signal exhaustion
            return b'' if self.audio_done else (b'\x00' * self.config.audio_chunk_size_bytes)

    def is_exhausted(self) -> bool:
        """Return True when both extractors finished and queues are empty."""
        return self.frames_done and self.audio_done and self.frame_queue.empty() and self.audio_queue.empty()
    
    def reset(self):
        """Reset is handled automatically by looping"""
        pass
    
    def close(self):
        """Stop extraction threads and clean up"""
        logger.info("Closing simulation source")
        self.stop_event.set()
        if self.frame_thread:
            self.frame_thread.join(timeout=5)
        if self.audio_thread:
            self.audio_thread.join(timeout=5)


class ProductionSource(FrameAudioSource):
    """
    Production source for actual frame and audio data
    Override these methods with your actual data source
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        logger.info("Initialized production source")
    
    def get_next_frame(self) -> Optional[bytes]:
        """
        Get the next PNG frame from your production system
        
        Returns:
            PNG bytes for a 1920x1080 frame, or None if no frame available
        """
        # TODO: Replace with your actual frame source
        # Example:
        # return get_frame_from_your_system()
        raise NotImplementedError("Implement your production frame source here")
    
    def get_next_audio_chunk(self) -> Optional[bytes]:
        """
        Get the next 40ms audio chunk from your production system
        
        Returns:
            PCM bytes (16kHz, mono, 16-bit signed), or None if no audio available
            Expected size: 1280 bytes (16000 * 0.04 * 1 * 2)
        """
        # TODO: Replace with your actual audio source
        # Example:
        # return get_audio_from_your_system()
        raise NotImplementedError("Implement your production audio source here")
    
    def reset(self):
        """Reset the production source if needed"""
        pass
    
    def close(self):
        """Clean up production resources"""
        pass


class LivePipelineSource(FrameAudioSource):
    """
    Utility source that accepts synchronized pairs: (PNG frame bytes, 40ms PCM audio bytes).

    - Upstream MUST call `submit_pair(png_bytes, pcm_bytes)` to maintain perfect A/V pairing.
    - Pairs are queued atomically; the streamer will consume video and audio from the same pair.
    - Blocking semantics: if the internal queue is full, `submit_pair` blocks (no drops).

    Audio requirements:
    - PCM must be 16-bit little-endian, mono, 16kHz, exactly 40ms (1280 bytes). We pad/trim if needed.
    """

    def __init__(self, config: StreamConfig, max_pairs: Optional[int] = None):
        self.config = config
        self.pair_queue: Queue[Tuple[bytes, bytes]] = Queue(maxsize=max_pairs or self.config.frame_buffer_size)
        self.stop_event = threading.Event()
        self._finished = False
        # Current in-flight pair and consumption state
        self._pair_lock = threading.Lock()
        self._current_pair: Optional[Tuple[bytes, bytes]] = None
        self._video_taken: bool = False
        self._audio_taken: bool = False

    # Upstream hook: submit synchronized pair
    def submit_pair(self, png_bytes: bytes, pcm_bytes: bytes) -> None:
        if self.stop_event.is_set() or self._finished:
            return
        expected = self.config.audio_chunk_size_bytes
        if len(pcm_bytes) != expected:
            if len(pcm_bytes) < expected:
                pcm_bytes = pcm_bytes + (b'\x00' * (expected - len(pcm_bytes)))
            else:
                pcm_bytes = pcm_bytes[:expected]
        self.pair_queue.put((png_bytes, pcm_bytes))  # block if full

    def finish(self) -> None:
        """Signal that no more pairs will be submitted."""
        self._finished = True

    def is_exhausted(self) -> bool:
        """True when finish() called and no pairs are left (including in-flight)."""
        with self._pair_lock:
            no_current = self._current_pair is None
        return self._finished and no_current and self.pair_queue.empty()

    # Streamer pulls
    def get_next_frame(self) -> Optional[bytes]:
        # Try to use current pair first
        with self._pair_lock:
            if self._current_pair is not None and not self._video_taken:
                png, _ = self._current_pair
                self._video_taken = True
                if self._audio_taken:
                    self._current_pair = None
                    self._audio_taken = False
                    self._video_taken = False
                return png
        # Need a new pair
        try:
            pair = self.pair_queue.get(timeout=0.1)
        except Empty:
            logger.warning("Pair queue empty (LivePipelineSource)")
            return None
        with self._pair_lock:
            self._current_pair = pair
            self._video_taken = True
            self._audio_taken = False
            png, _ = pair
            return png

    def get_next_audio_chunk(self) -> Optional[bytes]:
        # Try to use current pair first
        with self._pair_lock:
            if self._current_pair is not None and not self._audio_taken:
                _, pcm = self._current_pair
                self._audio_taken = True
                if self._video_taken:
                    self._current_pair = None
                    self._audio_taken = False
                    self._video_taken = False
                return pcm
        # Need a new pair
        try:
            pair = self.pair_queue.get(timeout=0.1)
        except Empty:
            logger.warning("Pair queue empty (LivePipelineSource)")
            return b'\x00' * self.config.audio_chunk_size_bytes
        with self._pair_lock:
            self._current_pair = pair
            self._video_taken = False
            self._audio_taken = True
            _, pcm = pair
            return pcm

    def reset(self):
        pass

    def close(self):
        self.stop_event.set()

class CloudflareStreamer:
    """
    Main streamer class that handles encoding and streaming to Cloudflare
    """
    
    def __init__(self, config: StreamConfig, source: FrameAudioSource):
        """
        Initialize the Cloudflare streamer
        
        Args:
            config: Stream configuration
            source: Frame and audio source (simulation or production)
        """
        self.config = config
        self.source = source
        self.ffmpeg_process = None
        self.audio_pipe_r = None
        self.audio_pipe_w = None
        self.is_streaming = False
        self.stats = {
            'frames_sent': 0,
            'audio_chunks_sent': 0,
            'start_time': None,
            'errors': 0
        }
        # FFmpeg restart/backoff control
        self.ffmpeg_lock = threading.Lock()
        self.restart_attempts = 0
        self.max_restart_attempts = 5
        self.last_restart_time = 0.0
        
        # Optionally create a new Cloudflare live input and update credentials
        if self.config.auto_create_live_input:
            self._ensure_live_input()
        
        # Setup FFmpeg process with possibly updated credentials
        self._setup_ffmpeg()
        logger.info("CloudflareStreamer initialized")
    
    def _setup_ffmpeg(self):
        """Initialize FFmpeg process for streaming"""
        try:
            # Create pipe for audio
            if self.audio_pipe_r is None or self.audio_pipe_w is None:
                self.audio_pipe_r, self.audio_pipe_w = os.pipe()
            
            # FFmpeg command for streaming
            command = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-fflags', '+genpts',

                # Video input (PNG over image2pipe via stdin)
                '-thread_queue_size', '1024',
                '-f', 'image2pipe',
                '-vcodec', 'png',
                '-framerate', str(self.config.fps),
                '-i', '-',

                # Audio input (PCM via pipe)
                '-thread_queue_size', '1024',
                '-f', 's16le',
                '-ar', str(self.config.audio_sample_rate),
                '-ac', str(self.config.audio_channels),
                '-i', f'pipe:{self.audio_pipe_r}',

                # Video encoding settings
                '-c:v', 'libx264',
                '-preset', 'veryfast',  # Fast encoding for low latency
                '-tune', 'zerolatency',  # Optimize for latency
                '-pix_fmt', 'yuv420p',
                '-vf', 'setpts=PTS-STARTPTS',
                '-vsync', 'cfr',
                '-g', str(self.config.fps * 2),  # GOP size (2 seconds)
                '-b:v', self.config.video_bitrate,
                '-maxrate', self.config.video_bitrate,
                '-bufsize', str(int(self.config.video_bitrate[:-1]) * 2) + 'k',

                # Audio encoding settings
                '-c:a', 'aac',
                '-af', 'aresample=async=1:first_pts=0',
                '-b:a', self.config.audio_bitrate,

                # Output settings
                '-f', 'flv',
                '-flvflags', 'no_duration_filesize',
                '-muxpreload', '0',
                '-muxdelay', '0',

                # RTMPS URL
                self.config.rtmps_full_url
            ]
            
            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(self.audio_pipe_r,),
                bufsize=10**8  # Large buffer
            )
            
            # Start stderr reader thread
            threading.Thread(
                target=self._read_ffmpeg_stderr,
                daemon=True,
                name="FFmpegStderr"
            ).start()
            
            logger.info(f"FFmpeg process started, streaming to: {self.config.rtmps_url}")
            
        except Exception as e:
            logger.error(f"Failed to setup FFmpeg: {e}")
            raise

    def _is_ffmpeg_running(self) -> bool:
        return self.ffmpeg_process is not None and self.ffmpeg_process.poll() is None

    def _restart_ffmpeg_with_backoff(self):
        with self.ffmpeg_lock:
            if self.max_restart_attempts is not None and self.restart_attempts >= self.max_restart_attempts:
                logger.error("Max FFmpeg restart attempts reached; stopping stream")
                self.is_streaming = False
                return
            # Exponential backoff up to 8s
            backoff = min(8.0, 2.0 ** self.restart_attempts)
            now = time.monotonic()
            sleep_needed = (self.last_restart_time + backoff) - now
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            try:
                # Close previous ffmpeg resources only
                if self.ffmpeg_process:
                    try:
                        if self.ffmpeg_process.stdin:
                            self.ffmpeg_process.stdin.close()
                    except:
                        pass
                    try:
                        self.ffmpeg_process.terminate()
                        self.ffmpeg_process.wait(timeout=3)
                    except Exception:
                        try:
                            self.ffmpeg_process.kill()
                        except Exception:
                            pass
                # Recreate audio pipe for safety
                try:
                    if self.audio_pipe_w:
                        os.close(self.audio_pipe_w)
                except:
                    pass
                try:
                    if self.audio_pipe_r:
                        os.close(self.audio_pipe_r)
                except:
                    pass
                self.audio_pipe_r = None
                self.audio_pipe_w = None
                # Setup new ffmpeg
                self._setup_ffmpeg()
                self.restart_attempts += 1
                self.last_restart_time = time.monotonic()
                logger.info("FFmpeg restarted (with backoff)")
            except Exception as e:
                logger.error(f"FFmpeg restart failed: {e}")
                self.restart_attempts += 1
                self.last_restart_time = time.monotonic()

    def _ensure_live_input(self):
        """Create a new Cloudflare Stream Live Input and set credentials."""
        try:
            account_id = self.config.cloudflare_account_id or os.getenv('CLOUDFLARE_ACCOUNT_ID')
            api_token = self.config.cloudflare_api_token or os.getenv('CLOUDFLARE_API_TOKEN')
            # Treat placeholders as missing
            if account_id == 'YOUR_CLOUDFLARE_ACCOUNT_ID':
                account_id = None
            if api_token == 'YOUR_CLOUDFLARE_API_TOKEN':
                api_token = None
            stream_name = self.config.stream_name or f"Live Stream {int(time.time())}"
            if not account_id or not api_token:
                raise ValueError("Cloudflare account id or API token missing. Set in config or env.")

            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/stream/live_inputs"
            payload = {
                "meta": {"name": stream_name},
                "recording": {"mode": "automatic"}
            }
            if self.config.delete_recording_after_days is not None:
                payload["deleteRecordingAfterDays"] = int(self.config.delete_recording_after_days)
            data = json.dumps(payload).encode('utf-8')
            req = urlrequest.Request(url, data=data, method='POST')
            req.add_header('Authorization', f'Bearer {api_token}')
            req.add_header('Content-Type', 'application/json')
            with urlrequest.urlopen(req, timeout=20) as resp:
                resp_data = resp.read()
            resp_json = json.loads(resp_data.decode('utf-8'))
            if not resp_json.get('success'):
                raise RuntimeError(f"Cloudflare API error: {resp_json.get('errors')}")
            result = resp_json.get('result', {})
            rtmps = result.get('rtmps', {})
            rtmps_url = rtmps.get('url')
            stream_key = rtmps.get('streamKey')
            uid = result.get('uid')

            if not rtmps_url or not stream_key:
                raise RuntimeError("Missing RTMPS credentials in response")

            # Update config for this run
            self.config.rtmps_url = rtmps_url.rstrip('/')
            self.config.stream_key = stream_key
            self.config.live_input_uid = uid

            # Attempt to get playback URL (HLS)
            # Try one immediate non-blocking fetch of playback URL
            try:
                get_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/stream/live_inputs/{uid}"
                get_req = urlrequest.Request(get_url, method='GET')
                get_req.add_header('Authorization', f'Bearer {api_token}')
                with urlrequest.urlopen(get_req, timeout=10) as get_resp:
                    get_data = get_resp.read()
                get_json = json.loads(get_data.decode('utf-8'))
                if get_json.get('success'):
                    playback = get_json.get('result', {}).get('playback', {})
                    hls_url = playback.get('hls')
                    if hls_url:
                        self.config.playback_url = hls_url
            except Exception as e:
                logger.debug(f"Immediate playback fetch not available: {e}")

            if not self.config.playback_url:
                # Construct fallback HLS URL if customer code is provided
                if self.config.cloudflare_customer_code and self.config.live_input_uid:
                    self.config.playback_url = (
                        f"https://customer-{self.config.cloudflare_customer_code}.cloudflarestream.com/"
                        f"{self.config.live_input_uid}/manifest/video.m3u8"
                    )
                    logger.info(f"Constructed playback URL: {self.config.playback_url}")
                else:
                    logger.info("Playback URL not available yet; it becomes active when live starts.")

            # Emit initial credentials/playback for downstream if requested
            self._emit_output_credentials(event="initial")

            # Start a background watcher to emit update when playback.hls becomes available
            if self.config.await_playback_url_seconds and uid:
                threading.Thread(
                    target=self._playback_watch_loop,
                    args=(account_id, api_token, uid, int(self.config.await_playback_url_seconds)),
                    daemon=True,
                    name="PlaybackWatcher"
                ).start()

        except urlerror.HTTPError as e:
            body = e.read().decode('utf-8', errors='ignore') if hasattr(e, 'read') else ''
            logger.error(f"Cloudflare API HTTPError {e.code}: {body}")
            raise

    def _playback_watch_loop(self, account_id: str, api_token: str, uid: str, max_seconds: int):
        """Poll in background for playback.hls and emit an update when available."""
        deadline = time.time() + max(0, max_seconds)
        while time.time() < deadline and self.is_streaming:
            try:
                get_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/stream/live_inputs/{uid}"
                get_req = urlrequest.Request(get_url, method='GET')
                get_req.add_header('Authorization', f'Bearer {api_token}')
                with urlrequest.urlopen(get_req, timeout=10) as get_resp:
                    get_data = get_resp.read()
                get_json = json.loads(get_data.decode('utf-8'))
                if get_json.get('success'):
                    playback = get_json.get('result', {}).get('playback', {})
                    hls_url = playback.get('hls')
                    if hls_url and hls_url != self.config.playback_url:
                        self.config.playback_url = hls_url
                        logger.info(f"Playback URL ready: {self.config.playback_url}")
                        self._emit_output_credentials(event="playback_update")
                        return
            except Exception:
                pass
            time.sleep(1)

    def _emit_output_credentials(self, event: str = "initial"):
        """Write credentials/playback info to file/stdout and notify callback."""
        output = {
            "rtmps_url": self.config.rtmps_url,
            "stream_key": self.config.stream_key,
            "playback_url": self.config.playback_url,
            "live_input_uid": self.config.live_input_uid,
            "event": event
        }
        if self.config.write_playback_json:
            try:
                with open(self.config.write_playback_json, 'w') as f:
                    json.dump(output, f)
                logger.info(f"Wrote playback info to {self.config.write_playback_json}")
            except Exception as e:
                logger.warning(f"Failed to write playback info: {e}")
        if self.config.emit_playback_stdout:
            print(json.dumps(output), flush=True)
        # Notify Python callback asynchronously so we don't block streaming
        if self.config.on_playback_update:
            try:
                threading.Thread(target=self.config.on_playback_update, args=(output,), daemon=True).start()
            except Exception as e:
                logger.warning(f"Playback callback error: {e}")
    
    def _read_ffmpeg_stderr(self):
        """Read and log FFmpeg stderr output"""
        try:
            for line in iter(self.ffmpeg_process.stderr.readline, b''):
                if line:
                    # Log FFmpeg output for debugging
                    logger.debug(f"FFmpeg: {line.decode('utf-8', errors='ignore').strip()}")
        except Exception as e:
            logger.error(f"Error reading FFmpeg stderr: {e}")
    
    def _send_frame(self, png_data: bytes) -> bool:
        """
        Send a PNG frame to FFmpeg
        
        Args:
            png_data: PNG image bytes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                self.ffmpeg_process.stdin.write(png_data)
                self.ffmpeg_process.stdin.flush()
                self.stats['frames_sent'] += 1
                return True
            else:
                logger.error("FFmpeg process is not running")
                return False
        except BrokenPipeError:
            logger.error("FFmpeg pipe broken")
            self.stats['errors'] += 1
            return False
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            self.stats['errors'] += 1
            # Attempt restart on video pipe errors
            self._restart_ffmpeg_with_backoff()
            return False
    
    def _send_audio(self, pcm_data: bytes) -> bool:
        """
        Send PCM audio data to FFmpeg
        
        Args:
            pcm_data: Raw PCM audio bytes (16kHz mono 16-bit)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.audio_pipe_w:
                os.write(self.audio_pipe_w, pcm_data)
                self.stats['audio_chunks_sent'] += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            self.stats['errors'] += 1
            # Attempt restart on audio pipe errors
            self._restart_ffmpeg_with_backoff()
            return False
    
    def stream(self):
        """
        Main streaming loop
        Handles timing, frame/audio synchronization, and error recovery
        """
        logger.info("Starting streaming...")
        self.is_streaming = True
        self.stats['start_time'] = time.monotonic()
        
        # Timing variables
        frame_interval = self.config.frame_duration_ms / 1000.0  # Convert to seconds
        audio_interval = self.config.audio_chunk_duration_ms / 1000.0
        
        # Initialize indices and prebuffer some data
        next_frame_index = 0
        next_audio_index = 0
        prebuffer_deadline = time.monotonic() + 2.0
        first_frame = None
        prebuffer_audio = []
        # Try to grab one frame
        while time.monotonic() < prebuffer_deadline and first_frame is None:
            candidate = self.source.get_next_frame()
            if candidate:
                first_frame = candidate
                break
            time.sleep(0.005)
        # Try to grab a few audio chunks
        while time.monotonic() < prebuffer_deadline and len(prebuffer_audio) < 3:
            chunk = self.source.get_next_audio_chunk()
            if chunk:
                prebuffer_audio.append(chunk)
            else:
                time.sleep(0.005)

        # Lock start clock and emit prebuffer immediately so timelines start at t0
        start_clock = time.monotonic()
        if first_frame:
            self._send_frame(first_frame)
            next_frame_index = 1
        for chunk in prebuffer_audio:
            self._send_audio(chunk)
        next_audio_index = len(prebuffer_audio)
        
        # Audio thread for better synchronization
        def audio_sender():
            """Separate thread for audio streaming with monotonic scheduling"""
            nonlocal next_audio_index
            while self.is_streaming:
                try:
                    current_time = time.monotonic()
                    target_time = start_clock + (next_audio_index + 1) * audio_interval
                    if current_time + 0.0005 < target_time:
                        # Sleep until next chunk boundary (with a small guard)
                        time.sleep(min(0.005, max(0.0, target_time - current_time)))
                        continue
                    audio_chunk = self.source.get_next_audio_chunk()
                    if audio_chunk:
                        self._send_audio(audio_chunk)
                    next_audio_index += 1
                
                except Exception as e:
                    logger.error(f"Audio sender error: {e}")
                    time.sleep(0.01)
        
        # Start audio thread
        audio_thread = threading.Thread(target=audio_sender, daemon=True, name="AudioSender")
        audio_thread.start()
        
        try:
            while self.is_streaming:
                current_time = time.monotonic()
                target_time = start_clock + (next_frame_index) * frame_interval
                # Wait until the scheduled frame time
                if current_time + 0.0005 < target_time:
                    time.sleep(min(0.005, max(0.0, target_time - current_time)))
                    continue
                
                # If source is exhausted (simulation or live pipeline), stop gracefully
                if (isinstance(self.source, SimulationSource) and self.source.is_exhausted()) or \
                   (isinstance(self.source, LivePipelineSource) and self.source.is_exhausted()):
                    logger.info("Simulation input exhausted; stopping stream")
                    # Notify egress completion if configured (for LivePipelineSource)
                    if isinstance(self.source, LivePipelineSource) and self.config.on_egress_complete:
                        summary = {
                            "frames_sent": self.stats['frames_sent'],
                            "audio_chunks_sent": self.stats['audio_chunks_sent'],
                            "errors": self.stats['errors'],
                            "playback_url": self.config.playback_url,
                            "event": "egress_complete"
                        }
                        try:
                            threading.Thread(target=self.config.on_egress_complete, args=(summary,), daemon=True).start()
                        except Exception as _:
                            pass
                    break

                # Time to send a frame
                frame = self.source.get_next_frame()
                if frame is None and self.config.enable_frame_hold:
                    # Hold last frame (no-op send if none yet)
                    if self.stats['frames_sent'] > 0:
                        # We can't resend without storing. Cache last_frame on successful send.
                        frame = getattr(self, '_last_frame_bytes', None)
                if frame:
                    if not self._send_frame(frame):
                        logger.warning("Frame send failed; attempting FFmpeg restart...")
                        self._restart_ffmpeg_with_backoff()
                    else:
                        # Cache last successfully sent frame for hold
                        try:
                            self._last_frame_bytes = frame
                        except Exception:
                            pass
                next_frame_index += 1
                
                # Log statistics every second
                if self.stats['frames_sent'] % self.config.fps == 0 and self.stats['frames_sent'] > 0:
                    elapsed = time.monotonic() - self.stats['start_time']
                    if elapsed > 0:
                        logger.info(
                            f"Stats - Frames: {self.stats['frames_sent']}, "
                            f"Audio chunks: {self.stats['audio_chunks_sent']}, "
                            f"Elapsed: {elapsed:.1f}s, "
                            f"FPS: {self.stats['frames_sent']/elapsed:.1f}, "
                            f"Errors: {self.stats['errors']}"
                        )
                    
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Keep running if recoverable; attempt to restart ffmpeg
            self._restart_ffmpeg_with_backoff()
        finally:
            self.is_streaming = False
            logger.info("Streaming stopped")
    
    def _restart_ffmpeg(self):
        """Attempt to restart FFmpeg process"""
        try:
            self.stop()
            time.sleep(1)
            self._setup_ffmpeg()
            logger.info("FFmpeg restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart FFmpeg: {e}")
            raise
    
    def stop(self):
        """Stop streaming and clean up resources"""
        logger.info("Stopping streamer...")
        self.is_streaming = False
        
        # Close FFmpeg stdin
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            try:
                self.ffmpeg_process.stdin.close()
            except:
                pass
        
        # Close audio pipe
        if self.audio_pipe_w:
            try:
                os.close(self.audio_pipe_w)
            except:
                pass
        
        if self.audio_pipe_r:
            try:
                os.close(self.audio_pipe_r)
            except:
                pass
        
        # Terminate FFmpeg process
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
            except:
                pass
        
        # Close source
        if self.source:
            self.source.close()
        
        # Print final statistics
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            logger.info(
                f"Final stats - Total frames: {self.stats['frames_sent']}, "
                f"Total audio chunks: {self.stats['audio_chunks_sent']}, "
                f"Total time: {elapsed:.1f}s, "
                f"Average FPS: {self.stats['frames_sent']/elapsed:.1f}, "
                f"Total errors: {self.stats['errors']}"
            )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream to Cloudflare')
    parser.add_argument('--mode', choices=['simulation', 'production'], 
                       default='simulation', help='Streaming mode')
    parser.add_argument('--mp4', type=str, default='input.mp4',
                       help='MP4 file path for simulation mode')
    parser.add_argument('--rtmps-url', type=str, 
                       default='rtmps://live.cloudflare.com/live',
                       help='RTMPS URL')
    parser.add_argument('--stream-key', type=str, required=False,
                       help='Stream key from Cloudflare')
    parser.add_argument('--auto-create', action='store_true',
                       help='Automatically create a new Cloudflare live input')
    parser.add_argument('--cf-account-id', type=str, default=None,
                       help='Cloudflare account ID (or env CLOUDFLARE_ACCOUNT_ID)')
    parser.add_argument('--cf-api-token', type=str, default=None,
                       help='Cloudflare API token (or env CLOUDFLARE_API_TOKEN)')
    parser.add_argument('--stream-name', type=str, default=None,
                       help='Optional name for the live input')
    parser.add_argument('--write-playback-json', type=str, default=None,
                       help='File path to write RTMPS credentials and playback URL as JSON')
    parser.add_argument('--emit-playback-stdout', action='store_true',
                       help='Emit a single JSON line to stdout with credentials/playback URL')
    parser.add_argument('--await-playback-url-seconds', type=int, default=0,
                       help='Wait up to N seconds for playback URL to become available')
    parser.add_argument('--frame-buffer-size', type=int, default=150,
                       help='Max queued frames before extractor blocks')
    parser.add_argument('--audio-buffer-size', type=int, default=500,
                       help='Max queued audio chunks before extractor blocks')
    parser.add_argument('--cf-customer-code', type=str, default=None,
                       help='Cloudflare customer code to construct HLS playback URL')
    parser.add_argument('--fps', type=int, default=25,
                       help='Frames per second')
    parser.add_argument('--video-bitrate', type=str, default='2500k',
                       help='Video bitrate')
    parser.add_argument('--audio-bitrate', type=str, default='64k',
                       help='Audio bitrate')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create configuration
    config = StreamConfig(
        rtmps_url=args.rtmps_url,
        stream_key=args.stream_key or '',
        fps=args.fps,
        video_bitrate=args.video_bitrate,
        audio_bitrate=args.audio_bitrate,
        auto_create_live_input=args.auto_create,
        cloudflare_account_id=args.cf_account_id or os.getenv('CLOUDFLARE_ACCOUNT_ID'),
        cloudflare_api_token=args.cf_api_token or os.getenv('CLOUDFLARE_API_TOKEN'),
        stream_name=args.stream_name,
        write_playback_json=args.write_playback_json,
        emit_playback_stdout=args.emit_playback_stdout,
        await_playback_url_seconds=args.await_playback_url_seconds
        ,frame_buffer_size=args.frame_buffer_size
        ,audio_buffer_size=args.audio_buffer_size
        ,cloudflare_customer_code=args.cf_customer_code or os.getenv('CLOUDFLARE_CUSTOMER_CODE')
    )
    
    # Create appropriate source
    if args.mode == 'simulation':
        logger.info(f"Starting in SIMULATION mode with file: {args.mp4}")
        source = SimulationSource(args.mp4, config)
    else:
        logger.info("Starting in PRODUCTION mode")
        source = ProductionSource(config)
    
    # Create and start streamer
    streamer = CloudflareStreamer(config, source)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        streamer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start streaming
    try:
        streamer.stream()
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        streamer.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Library documentation
=====================

Overview
--------
This module publishes video frames (PNG over stdin) and audio (PCM over pipe) to Cloudflare Live via FFmpeg (H.264/AAC → RTMPS).
It supports automatic Cloudflare Live Input creation, resilient FFmpeg restarts, and notifying a client when the HLS playback URL becomes available.

Key classes
-----------
- `StreamConfig`: Configuration for streaming. Important fields:
  - `rtmps_url` / `stream_key`: RTMPS credentials. If `auto_create_live_input=True`, these are fetched automatically.
  - `fps`, `video_bitrate`, `audio_sample_rate`, `audio_channels`.
  - `auto_create_live_input`, `cloudflare_account_id`, `cloudflare_api_token`, `stream_name`.
  - `cloudflare_customer_code`: used to construct a fallback HLS `.m3u8` until Cloudflare returns the official `playback.hls`.
  - `await_playback_url_seconds`: how long to background-poll for the official `playback.hls` (non-blocking to egress).
  - `on_playback_update`: optional Python callback invoked with a dict containing `rtmps_url`, `stream_key`, `playback_url`, `live_input_uid`, `event`.
  - `on_egress_complete`: optional Python callback invoked when all enqueued pairs have been egressed.
  - `frame_buffer_size`, `audio_buffer_size`: queue sizes; producers block when full (no drops).

- `FrameAudioSource`: Abstract interface providing `get_next_frame()` (PNG bytes) and `get_next_audio_chunk()` (40 ms PCM bytes).

- `SimulationSource`: Implements `FrameAudioSource` by extracting PNG frames and PCM audio from an MP4.

- `CloudflareStreamer`: Orchestrates FFmpeg, timing, error handling, auto-creation of live inputs, and notifications.

Callback contract
-----------------
If `StreamConfig.on_playback_update` is provided, the function will be called asynchronously with a dict:
```
{
  "rtmps_url": str,
  "stream_key": str,
  "playback_url": Optional[str],  # may be a constructed fallback first
  "live_input_uid": Optional[str],
  "event": "initial" | "playback_update"
}
```
Events:
- `initial`: Emitted right after creating/fetching credentials (or on startup if provided manually). May include a constructed `.m3u8` using your customer code.
- `playback_update`: Emitted when Cloudflare returns the official `playback.hls` URL after we begin streaming.

Egress completion callback
-------------------------
If `StreamConfig.on_egress_complete` is provided and you use `LivePipelineSource`, the function will be called asynchronously once the client has called `source.finish()` and the streamer has fully drained all queued pairs:
```
{
  "frames_sent": int,
  "audio_chunks_sent": int,
  "errors": int,
  "playback_url": Optional[str],
  "event": "egress_complete"
}
```

Environment variables
---------------------
- `CLOUDFLARE_ACCOUNT_ID`: Account ID for Cloudflare Stream API.
- `CLOUDFLARE_API_TOKEN`: API token with Stream permissions.
- `CLOUDFLARE_CUSTOMER_CODE`: Customer code for fallback `.m3u8` construction.

Command-line usage
------------------
Example:
```bash
python rapido/demo/cloudflare_streamer.py \
  --mode simulation \
  --mp4 rapido/demo/Jason-cc_25fps.mp4 \
  --auto-create \
  --fps 25 \
  --video-bitrate 2500k \
  --audio-bitrate 64k \
  --await-playback-url-seconds 60 \
  --emit-playback-stdout \
  --log-level INFO
```

Programmatic usage (external Python module)
------------------------------------------
```python
from rapido.demo.cloudflare_streamer import StreamConfig, SimulationSource, CloudflareStreamer

def on_playback_update(info: dict) -> None:
    # info contains: rtmps_url, stream_key, playback_url, live_input_uid, event
    # store or forward the playback_url as soon as it becomes available
    print("Playback update:", info)

config = StreamConfig(
    rtmps_url="rtmps://live.cloudflare.com/live",
    stream_key="",  # leave empty when auto-creating
    auto_create_live_input=True,
    fps=25,
    video_bitrate="2500k",
    audio_bitrate="64k",
    on_playback_update=on_playback_update,
)

source = SimulationSource("rapido/demo/Jason-cc_25fps.mp4", config)
streamer = CloudflareStreamer(config, source)

try:
    streamer.stream()
finally:
    streamer.stop()
```

Error handling and resiliency
-----------------------------
- FFmpeg restarts with exponential backoff on transient errors for both audio/video pipes.
- Streaming continues while retrying within `max_restart_attempts` (currently internal; can be exposed if needed).
- Blocking queues prevent frame/audio drops by back-pressuring extractors.

Sync considerations
-------------------
- Monotonic, index-based scheduling aligns frame and audio chunk timing.
- FFmpeg flags normalize timestamps: `setpts=PTS-STARTPTS` for video and `aresample=async=1:first_pts=0` for audio.

Notes
-----
- When using auto-creation, ensure the Cloudflare credentials are configured via env or `StreamConfig`.
- The HLS playback URL may appear only after the stream becomes active; the background watcher emits an update when ready.

Feeding synchronized PNG+audio pairs (production pattern)
--------------------------------------------------------
Use the built-in `LivePipelineSource` to submit a single synchronized pair every 40 ms. When finished producing, call `source.finish()` to signal no more input so the streamer can drain and trigger `on_egress_complete`:

```python
import threading
from rapido.demo.cloudflare_streamer import StreamConfig, LivePipelineSource, CloudflareStreamer

def on_playback_update(info: dict):
    print("Playback:", info)

def on_egress_complete(info: dict):
    print("Egress complete:", info)

config = StreamConfig(
    rtmps_url="rtmps://live.cloudflare.com/live",
    stream_key="",  # auto-create
    auto_create_live_input=True,
    on_playback_update=on_playback_update,
    on_egress_complete=on_egress_complete,
)

source = LivePipelineSource(config, max_pairs=20000)  # large buffer (blocks when full, no drops)
streamer = CloudflareStreamer(config, source)

def producer_loop():
    for png_bytes, pcm_bytes in generate_pairs():  # implement: yields (png, pcm) 40ms pairs
        source.submit_pair(png_bytes, pcm_bytes)   # atomic pair submission (blocks if buffer is full)
    # Signal no more input so the streamer can drain and notify via on_egress_complete
    source.finish()

threading.Thread(target=producer_loop, daemon=True).start()

try:
    streamer.stream()
finally:
    streamer.stop()
```
"""