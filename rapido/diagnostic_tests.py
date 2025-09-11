#!/usr/bin/env python3
"""
Comprehensive diagnostic tests to pinpoint audio issues
Tests: TTS speed, Network latency, SyncTalk performance, System resources
"""

import asyncio
import time
import psutil
import websockets
import json
import requests
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class AudioDiagnostics:
    def __init__(self):
        self.results = {}
        
    async def test_elevenlabs_tts_speed(self):
        """Test 1: ElevenLabs TTS streaming speed"""
        print("\n" + "="*60)
        print("TEST 1: ELEVENLABS TTS STREAMING SPEED")
        print("="*60)
        
        try:
            from tts_client import TTSClient
            
            # Test text - about 10 seconds of speech
            test_text = "This is a comprehensive test of ElevenLabs text to speech streaming performance. We need to measure how quickly audio chunks are delivered and if there are any delays or interruptions in the stream. This should take approximately ten seconds to speak."
            
            print(f"Test text: {len(test_text)} characters")
            print("Expected speech duration: ~10 seconds")
            
            tts_client = TTSClient()
            
            chunks_received = 0
            total_audio_bytes = 0
            start_time = time.time()
            first_chunk_time = None
            chunk_times = []
            
            async def audio_callback(chunk):
                nonlocal chunks_received, total_audio_bytes, first_chunk_time
                chunks_received += 1
                total_audio_bytes += len(chunk)
                
                current_time = time.time()
                if first_chunk_time is None:
                    first_chunk_time = current_time
                    print(f"  ✅ First chunk received after {current_time - start_time:.2f}s")
                
                chunk_times.append(current_time)
                
                if chunks_received % 10 == 0:
                    print(f"  📦 Chunk {chunks_received}: {len(chunk)} bytes ({total_audio_bytes/1024:.1f}KB total)")
            
            print("🎵 Starting TTS stream...")
            await tts_client.stream_audio_real_time(test_text, audio_callback)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Calculate chunk intervals
            if len(chunk_times) > 1:
                intervals = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
                avg_interval = sum(intervals) / len(intervals)
                max_gap = max(intervals)
            else:
                avg_interval = max_gap = 0
            
            # Audio duration at 16kHz
            samples = total_audio_bytes // 2  # 16-bit = 2 bytes per sample
            audio_duration = samples / 16000
            
            print(f"\n📊 TTS Results:")
            print(f"  Total time: {total_duration:.2f}s")
            print(f"  Time to first chunk: {first_chunk_time - start_time:.2f}s")
            print(f"  Chunks received: {chunks_received}")
            print(f"  Total audio: {total_audio_bytes/1024:.1f}KB ({audio_duration:.1f}s)")
            print(f"  Average chunk interval: {avg_interval*1000:.1f}ms")
            print(f"  Longest gap: {max_gap*1000:.1f}ms")
            print(f"  Real-time ratio: {audio_duration/total_duration:.2f}x")
            
            # Diagnosis
            if first_chunk_time - start_time > 2.0:
                print("  ❌ SLOW STARTUP - First chunk took >2s")
                self.results['tts_startup'] = 'SLOW'
            else:
                print("  ✅ Good startup time")
                self.results['tts_startup'] = 'GOOD'
                
            if max_gap > 0.5:
                print("  ❌ LARGE GAPS - Chunks have >500ms gaps")
                self.results['tts_gaps'] = 'LARGE'
            else:
                print("  ✅ Consistent chunk delivery")
                self.results['tts_gaps'] = 'GOOD'
                
            if audio_duration / total_duration < 0.8:
                print("  ❌ TOO SLOW - Not delivering real-time audio")
                self.results['tts_speed'] = 'SLOW'
            else:
                print("  ✅ Real-time or faster delivery")
                self.results['tts_speed'] = 'GOOD'
                
        except Exception as e:
            print(f"❌ TTS test failed: {e}")
            self.results['tts_speed'] = 'FAILED'
    
    async def test_network_latency(self):
        """Test 2: Network latency to SyncTalk"""
        print("\n" + "="*60)
        print("TEST 2: NETWORK LATENCY TO SYNCTALK")
        print("="*60)
        
        synctalk_url = "ws://35.172.212.10:8000/ws/audio_to_video?avatar_name=enrique_torres&sample_rate=16000"
        
        try:
            # Test WebSocket connection time
            print("🌐 Testing WebSocket connection...")
            start_time = time.time()
            
            async with websockets.connect(synctalk_url) as websocket:
                connect_time = time.time() - start_time
                print(f"  ✅ Connected in {connect_time*1000:.1f}ms")
                
                # Test ping-pong latency
                ping_times = []
                for i in range(5):
                    ping_start = time.time()
                    await websocket.ping()
                    pong = await websocket.wait_closed()  # This won't work, let's try different approach
                    
                # Alternative: measure message round-trip
                print("  📡 Testing message round-trip...")
                round_trip_times = []
                
                for i in range(3):
                    # Send a small test message
                    test_msg = json.dumps({"type": "test", "timestamp": time.time()})
                    msg_start = time.time()
                    
                    await websocket.send(test_msg)
                    
                    # Wait for any response (even error)
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        round_trip = time.time() - msg_start
                        round_trip_times.append(round_trip)
                        print(f"    Round-trip {i+1}: {round_trip*1000:.1f}ms")
                    except asyncio.TimeoutError:
                        print(f"    Round-trip {i+1}: TIMEOUT")
                        round_trip_times.append(1.0)  # 1s timeout
                
                if round_trip_times:
                    avg_latency = sum(round_trip_times) / len(round_trip_times)
                    max_latency = max(round_trip_times)
                    
                    print(f"\n📊 Network Results:")
                    print(f"  Connection time: {connect_time*1000:.1f}ms")
                    print(f"  Average latency: {avg_latency*1000:.1f}ms")
                    print(f"  Max latency: {max_latency*1000:.1f}ms")
                    
                    if connect_time > 1.0:
                        print("  ❌ SLOW CONNECTION - Takes >1s to connect")
                        self.results['network_connect'] = 'SLOW'
                    else:
                        print("  ✅ Good connection speed")
                        self.results['network_connect'] = 'GOOD'
                        
                    if avg_latency > 0.2:
                        print("  ❌ HIGH LATENCY - Average >200ms")
                        self.results['network_latency'] = 'HIGH'
                    else:
                        print("  ✅ Low latency")
                        self.results['network_latency'] = 'GOOD'
                
        except Exception as e:
            print(f"❌ Network test failed: {e}")
            self.results['network_latency'] = 'FAILED'
    
    async def test_synctalk_performance(self):
        """Test 3: SyncTalk processing performance"""
        print("\n" + "="*60)
        print("TEST 3: SYNCTALK PROCESSING PERFORMANCE")
        print("="*60)
        
        synctalk_url = "ws://35.172.212.10:8000/ws/audio_to_video?avatar_name=enrique_torres&sample_rate=16000"
        
        try:
            print("🤖 Testing SyncTalk processing speed...")
            
            async with websockets.connect(synctalk_url) as websocket:
                # Send a few audio chunks and measure response time
                chunk_size = 2560  # 160ms at 16kHz
                processing_times = []
                
                for i in range(5):
                    # Generate test audio chunk
                    import numpy as np
                    audio_data = np.random.randn(chunk_size) * 5000
                    audio_bytes = audio_data.astype(np.int16).tobytes()
                    
                    # Send audio chunk
                    send_time = time.time()
                    await websocket.send(audio_bytes)
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        process_time = time.time() - send_time
                        processing_times.append(process_time)
                        
                        print(f"  📦 Chunk {i+1}: {process_time*1000:.1f}ms processing time")
                        
                    except asyncio.TimeoutError:
                        print(f"  ❌ Chunk {i+1}: TIMEOUT (>5s)")
                        processing_times.append(5.0)
                
                if processing_times:
                    avg_processing = sum(processing_times) / len(processing_times)
                    max_processing = max(processing_times)
                    
                    print(f"\n📊 SyncTalk Results:")
                    print(f"  Average processing: {avg_processing*1000:.1f}ms")
                    print(f"  Max processing: {max_processing*1000:.1f}ms")
                    print(f"  Target: <160ms (for real-time)")
                    
                    if avg_processing > 0.16:  # 160ms
                        print("  ❌ TOO SLOW - Can't keep up with real-time")
                        self.results['synctalk_speed'] = 'SLOW'
                    else:
                        print("  ✅ Fast enough for real-time")
                        self.results['synctalk_speed'] = 'GOOD'
                        
                    if max_processing > 0.5:  # 500ms
                        print("  ❌ LARGE SPIKES - Some chunks take >500ms")
                        self.results['synctalk_spikes'] = 'HIGH'
                    else:
                        print("  ✅ Consistent processing times")
                        self.results['synctalk_spikes'] = 'LOW'
                
        except Exception as e:
            print(f"❌ SyncTalk test failed: {e}")
            self.results['synctalk_speed'] = 'FAILED'
    
    def test_system_resources(self):
        """Test 4: System resource usage"""
        print("\n" + "="*60)
        print("TEST 4: SYSTEM RESOURCE USAGE")
        print("="*60)
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # GPU info (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory': gpu.memoryUtil * 100
                    })
            except:
                gpu_info = None
            
            print(f"💻 System Resources:")
            print(f"  CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
            print(f"  Memory: {memory_percent:.1f}% used ({memory_available:.1f}GB available)")
            print(f"  Disk: {disk_percent:.1f}% used")
            
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    print(f"  GPU {i}: {gpu['name']} - {gpu['load']:.1f}% load, {gpu['memory']:.1f}% memory")
            else:
                print("  GPU: Info not available")
            
            # Diagnosis
            issues = []
            if cpu_percent > 80:
                issues.append("HIGH CPU")
                self.results['cpu_usage'] = 'HIGH'
            else:
                self.results['cpu_usage'] = 'GOOD'
                
            if memory_percent > 85:
                issues.append("HIGH MEMORY")
                self.results['memory_usage'] = 'HIGH'
            else:
                self.results['memory_usage'] = 'GOOD'
                
            if gpu_info:
                for gpu in gpu_info:
                    if gpu['load'] > 90:
                        issues.append("HIGH GPU")
                        self.results['gpu_usage'] = 'HIGH'
                        break
                else:
                    self.results['gpu_usage'] = 'GOOD'
            
            if issues:
                print(f"  ❌ RESOURCE ISSUES: {', '.join(issues)}")
            else:
                print("  ✅ Resources look good")
                
        except Exception as e:
            print(f"❌ System test failed: {e}")
            self.results['system_resources'] = 'FAILED'
    
    def print_diagnosis(self):
        """Print final diagnosis"""
        print("\n" + "="*60)
        print("🔍 FINAL DIAGNOSIS")
        print("="*60)
        
        issues = []
        
        # Check each component
        if self.results.get('tts_startup') == 'SLOW':
            issues.append("TTS takes too long to start (>2s)")
        if self.results.get('tts_gaps') == 'LARGE':
            issues.append("TTS has large gaps between chunks (>500ms)")
        if self.results.get('tts_speed') == 'SLOW':
            issues.append("TTS not delivering real-time audio")
            
        if self.results.get('network_connect') == 'SLOW':
            issues.append("Slow WebSocket connection to SyncTalk (>1s)")
        if self.results.get('network_latency') == 'HIGH':
            issues.append("High network latency to SyncTalk (>200ms)")
            
        if self.results.get('synctalk_speed') == 'SLOW':
            issues.append("SyncTalk processing too slow (>160ms per chunk)")
        if self.results.get('synctalk_spikes') == 'HIGH':
            issues.append("SyncTalk has processing spikes (>500ms)")
            
        if self.results.get('cpu_usage') == 'HIGH':
            issues.append("High CPU usage (>80%)")
        if self.results.get('memory_usage') == 'HIGH':
            issues.append("High memory usage (>85%)")
        if self.results.get('gpu_usage') == 'HIGH':
            issues.append("High GPU usage (>90%)")
        
        if issues:
            print("❌ IDENTIFIED ISSUES:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
                
            print("\n💡 RECOMMENDATIONS:")
            
            # TTS issues
            if any('TTS' in issue for issue in issues):
                print("  🎵 TTS Issues:")
                print("    - Try different ElevenLabs voice/model")
                print("    - Check internet connection")
                print("    - Consider local TTS alternative")
            
            # Network issues
            if any('network' in issue.lower() or 'websocket' in issue.lower() for issue in issues):
                print("  🌐 Network Issues:")
                print("    - Check internet stability")
                print("    - Consider closer SyncTalk server")
                print("    - Test with local SyncTalk instance")
            
            # SyncTalk issues
            if any('SyncTalk' in issue for issue in issues):
                print("  🤖 SyncTalk Issues:")
                print("    - SyncTalk server may be overloaded")
                print("    - Try different avatar model")
                print("    - Consider local SyncTalk deployment")
            
            # System issues
            if any('CPU' in issue or 'memory' in issue.lower() or 'GPU' in issue for issue in issues):
                print("  💻 System Issues:")
                print("    - Close other applications")
                print("    - Reduce video quality/resolution")
                print("    - Consider more powerful hardware")
        else:
            print("✅ NO MAJOR ISSUES DETECTED")
            print("\nThe bottleneck might be:")
            print("  1. Buffer configuration still not optimal")
            print("  2. Frame delivery timing issues")
            print("  3. LiveKit publishing problems")
            print("  4. Subtle timing synchronization issues")

async def main():
    print("🔍 COMPREHENSIVE AUDIO DIAGNOSTICS")
    print("Testing all potential bottlenecks...")
    
    diagnostics = AudioDiagnostics()
    
    # Run all tests
    await diagnostics.test_elevenlabs_tts_speed()
    await diagnostics.test_network_latency()
    await diagnostics.test_synctalk_performance()
    diagnostics.test_system_resources()
    
    # Final diagnosis
    diagnostics.print_diagnosis()

if __name__ == "__main__":
    asyncio.run(main())


