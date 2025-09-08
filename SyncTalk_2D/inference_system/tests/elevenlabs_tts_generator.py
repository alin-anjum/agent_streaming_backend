# inference_system/tests/elevenlabs_tts_generator.py

import os
import asyncio
import websockets
import json
import base64
import time
import numpy as np
from typing import Generator, Optional
import io
import wave
from dotenv import load_dotenv
from collections import deque
import threading
import queue

# Load environment variables
load_dotenv()

class ElevenLabsTTSGenerator:
    """Generator that uses real ElevenLabs TTS with intermittent pauses"""
    
    def __init__(self, 
                 voice_id: str = 'Xb7hH8MSUJpSbSDYk0k2',
                 model_id: str = 'eleven_flash_v2_5',
                 pause_duration_seconds: float = 10.0):
        self.api_key = "5247420f33ae186ddea9a70843d469ff"
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
            
        self.voice_id = voice_id
        self.model_id = model_id
        self.pause_duration_seconds = pause_duration_seconds

        # Queue for audio data
        self.audio_queue = queue.Queue()
        self.is_running = True
        
        # Test sentences
        self.test_sentences = [
            "Hello, how smart is it to invest in technology stocks right now?",
            "The market has been quite volatile lately, with significant fluctuations.",
            "I recommend diversifying your portfolio across different sectors.",
            "Have you considered looking at emerging markets as well?",
            "The artificial intelligence sector shows promising growth potential.",            
            "However, always remember that past performance doesn't guarantee future results.",
            "It's important to do your own research before making any investment.",
            "Would you like me to explain more about risk management strategies?",
            "Dollar cost averaging can be an effective approach for long-term investors.",
            "Thank you for listening to my investment advice today."
        ]

    async def _process_audio_stream(self, websocket):
        """Process incoming audio stream from ElevenLabs, preserving natural chunks"""
        chunks_received = 0
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("audio"):
                    # Decode MP3 to PCM
                    mp3_data = base64.b64decode(data["audio"])
                    pcm_data = self._mp3_to_pcm(mp3_data)
                    
                    # Immediately put the chunk in queue as-is
                    self.audio_queue.put(pcm_data)
                    chunks_received += 1
                    
                    print(f"[ElevenLabs] Received chunk {chunks_received}, size: {len(pcm_data)} bytes "
                        f"({len(pcm_data)/32000:.3f}s)")  # 16kHz, 16-bit = 32000 bytes/sec
                        
                elif data.get('isFinal'):
                    print(f"[ElevenLabs] Stream complete, received {chunks_received} chunks")
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                print("[ElevenLabs] WebSocket connection closed")
                break

    def _mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """Convert MP3 to PCM using pydub or ffmpeg"""
        try:
            from pydub import AudioSegment
            import io
            
            # Load MP3 from bytes
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to 16kHz mono 16-bit PCM
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            audio = audio.set_sample_width(2)
            
            return audio.raw_data
            
        except ImportError:
            print("[WARNING] pydub not installed, using subprocess fallback")
            # Fallback using ffmpeg subprocess
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tmp_mp3.write(mp3_data)
                tmp_mp3_path = tmp_mp3.name
                
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as tmp_pcm:
                tmp_pcm_path = tmp_pcm.name
                
            try:
                # Convert MP3 to raw PCM using ffmpeg
                cmd = [
                    'ffmpeg', '-y', '-i', tmp_mp3_path,
                    '-f', 's16le',  # 16-bit little-endian
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',  # 16kHz
                    '-ac', '1',  # Mono
                    tmp_pcm_path
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                
                # Read the PCM data
                with open(tmp_pcm_path, 'rb') as f:
                    pcm_data = f.read()
                    
                return pcm_data
                
            finally:
                # Clean up temp files
                os.unlink(tmp_mp3_path)
                os.unlink(tmp_pcm_path)

    async def _process_audio_chunk(self, base64_audio: str) -> bytes:
        """Convert base64 audio to PCM format"""
        # Decode base64 to MP3
        mp3_data = base64.b64decode(base64_audio)
        
        # For now, we'll need to convert MP3 to PCM
        # In production, you'd use a library like pydub or ffmpeg
        # For testing, let's create synthetic PCM data of appropriate length
        # This simulates the conversion process
        
        # Estimate PCM size based on MP3 (rough approximation)
        pcm_samples = len(mp3_data) * 8  # Very rough estimate
        pcm_data = np.random.randint(-1000, 1000, pcm_samples, dtype=np.int16)
        
        return pcm_data.tobytes()
    
    async def _generate_tts_for_sentence(self, text: str, sentence_num: int):
        """Generate TTS for a single sentence using WebSocket"""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}"
        
        print(f"[ElevenLabs] Generating audio for sentence {sentence_num}: '{text[:50]}...'")
        
        try:
            async with websockets.connect(uri) as websocket:
                # Initialize connection with voice settings
                await websocket.send(json.dumps({
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                        "use_speaker_boost": False
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290]
                    },
                    "xi_api_key": self.api_key,
                }))
                
                # Send the actual text
                await websocket.send(json.dumps({"text": text, "flush": True}))
                
                # Close the stream
                await websocket.send(json.dumps({"text": ""}))
                
                # Process the audio stream using the new method
                await self._process_audio_stream(websocket)
                
                print(f"[ElevenLabs] Completed sentence {sentence_num}")
                
        except Exception as e:
            print(f"[ElevenLabs] Error generating TTS: {e}")
    
    async def _run_tts_loop(self):
        """Main loop that generates TTS with pauses"""
        for i, sentence in enumerate(self.test_sentences):
            if not self.is_running:
                break
                
            # Generate TTS for sentence
            await self._generate_tts_for_sentence(sentence, i + 1)
            
            # Pause for N seconds
            print(f"[ElevenLabs] Pausing for {self.pause_duration_seconds} seconds...")
            
            # Put None in queue at natural intervals (about every 100ms to signal no audio)
            pause_start = time.time()
            none_interval = 0.1  # 100ms
            last_none_time = pause_start
            
            while time.time() - pause_start < self.pause_duration_seconds:
                current_time = time.time()
                if current_time - last_none_time >= none_interval:
                    self.audio_queue.put(None)  # Signal "no audio available"
                    last_none_time = current_time
                await asyncio.sleep(0.01)  # Small sleep
            
            print(f"[ElevenLabs] Resuming...")
        
        # Signal completion
        self.audio_queue.put(StopIteration)
        print("[ElevenLabs] All sentences completed")
    
    def start_async_loop(self):
        """Start the async loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_tts_loop())
    
    def generate_chunks(self) -> Generator[Optional[bytes], None, None]:
        """Generator that yields audio chunks or None during pauses"""
        # Start TTS generation in background thread
        tts_thread = threading.Thread(target=self.start_async_loop, daemon=True)
        tts_thread.start()
        
        # Yield chunks from queue
        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Check for end marker
                if chunk is StopIteration:
                    break
                
                # Yield the chunk - this will be bytes for audio, None for pauses
                yield chunk
                
            except queue.Empty:
                # No data available yet, just continue
                continue
        
        self.is_running = False

def create_elevenlabs_tts_generator(pause_duration_seconds: float = 10.0) -> Generator[Optional[bytes], None, None]:
    """
    Create a generator that uses ElevenLabs TTS with realistic pauses.
    
    This simulates a real conversational AI system where:
    - TTS generates audio for a sentence
    - System pauses (yielding None) while waiting for next input
    - TTS generates next sentence
    - Repeats for 10 test sentences
    
    Args:
        chunk_duration_ms: Duration of each audio chunk in milliseconds
        
    Yields:
        bytes: PCM audio chunks (16kHz, mono, 16-bit) or None during pauses
    """
    generator = ElevenLabsTTSGenerator(pause_duration_seconds=pause_duration_seconds)
    return generator.generate_chunks()

# For testing with synthetic data (no actual ElevenLabs API calls)
def create_synthetic_tts_generator(chunk_duration_ms: int = 500) -> Generator[Optional[bytes], None, None]:
    """
    Create a synthetic generator that mimics ElevenLabs behavior without API calls.
    Useful for testing without consuming API credits.
    """
    test_sentences_durations = [3.2, 4.1, 3.8, 3.5, 4.3, 5.2, 4.0, 4.5, 4.8, 3.9]  # seconds
    chunk_size = int(16000 * chunk_duration_ms / 1000)  # samples per chunk
    
    for i, duration in enumerate(test_sentences_durations):
        print(f"[Synthetic TTS] Generating sentence {i+1} ({duration}s)...")
        
        # Generate synthetic audio for the sentence
        total_samples = int(duration * 16000)
        chunks_to_generate = total_samples // chunk_size
        
        for j in range(chunks_to_generate):
            # Create synthetic audio chunk (sine wave with slight variation)
            t = np.linspace(j * chunk_duration_ms / 1000, 
                           (j + 1) * chunk_duration_ms / 1000, 
                           chunk_size)
            # Mix of frequencies to simulate speech
            audio = (np.sin(2 * np.pi * 200 * t) * 0.3 + 
                    np.sin(2 * np.pi * 400 * t) * 0.2 +
                    np.sin(2 * np.pi * 800 * t) * 0.1 +
                    np.random.randn(chunk_size) * 0.1)
            audio = (audio * 5000).astype(np.int16)
            
            yield audio.tobytes()
        
        print(f"[Synthetic TTS] Sentence {i+1} complete. Pausing for 5s...")
        
        # Pause for 5 seconds, yielding None
        pause_chunks = int(5000 / chunk_duration_ms)
        for _ in range(pause_chunks):
            yield None
            time.sleep(chunk_duration_ms / 1000.0)
        
        print(f"[Synthetic TTS] Resuming...")
    
    print("[Synthetic TTS] All sentences completed")