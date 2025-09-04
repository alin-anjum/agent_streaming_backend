#!/usr/bin/env python3
"""
System starter script for Rapido + SyncTalk
This script helps coordinate starting both the SyncTalk server and Rapido client
"""

import os
import sys
import time
import subprocess
import asyncio
import aiohttp
from pathlib import Path
import argparse

def check_synctalk_server(url="http://localhost:8000"):
    """Check if SyncTalk server is running."""
    try:
        import requests
        response = requests.get(f"{url}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

async def wait_for_server(url="http://localhost:8000", max_wait=60):
    """Wait for SyncTalk server to be ready."""
    print(f"Waiting for SyncTalk server at {url}...")
    
    for i in range(max_wait):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/status", timeout=2) as response:
                    if response.status == 200:
                        print("✅ SyncTalk server is ready!")
                        return True
        except:
            pass
        
        print(f"⏳ Waiting... ({i+1}/{max_wait})")
        await asyncio.sleep(1)
    
    print("❌ SyncTalk server did not start in time")
    return False

def start_synctalk_server(synctalk_path):
    """Start SyncTalk FastAPI server."""
    if not synctalk_path.exists():
        print(f"❌ SyncTalk directory not found: {synctalk_path}")
        return None
    
    server_script = synctalk_path / "synctalk_fastapi.py"
    if not server_script.exists():
        print(f"❌ SyncTalk FastAPI script not found: {server_script}")
        return None
    
    print("🚀 Starting SyncTalk FastAPI server...")
    
    # Start server in background
    try:
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=str(synctalk_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        
        print(f"✅ SyncTalk server started with PID: {process.pid}")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start SyncTalk server: {e}")
        return None

async def start_rapido_client():
    """Start Rapido client."""
    print("\n🎬 Starting Rapido client...")
    
    try:
        # Import and run Rapido
        from rapido_main import RapidoOrchestrator
        
        orchestrator = RapidoOrchestrator()
        
        if not await orchestrator.initialize():
            print("❌ Failed to initialize Rapido")
            return False
        
        print("✅ Rapido initialized successfully!")
        
        # Process presentation
        output_path = await orchestrator.process_presentation()
        
        if output_path:
            print(f"\n🎉 Success! Generated video: {output_path}")
            return True
        else:
            print("❌ Failed to generate video")
            return False
            
    except Exception as e:
        print(f"❌ Error running Rapido: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Start Rapido + SyncTalk system")
    parser.add_argument("--synctalk-path", help="Path to SyncTalk-FastAPI directory", 
                       default="../SyncTalk-FastAPI")
    parser.add_argument("--server-only", action="store_true", 
                       help="Only start SyncTalk server")
    parser.add_argument("--client-only", action="store_true", 
                       help="Only start Rapido client (assumes server is running)")
    parser.add_argument("--server-url", default="http://localhost:8000",
                       help="SyncTalk server URL")
    
    args = parser.parse_args()
    
    synctalk_path = Path(args.synctalk_path).resolve()
    
    print("🚀 Rapido + SyncTalk System Starter")
    print("=" * 50)
    
    server_process = None
    
    try:
        if not args.client_only:
            # Check if server is already running
            if check_synctalk_server(args.server_url):
                print("✅ SyncTalk server is already running")
            else:
                # Start SyncTalk server
                server_process = start_synctalk_server(synctalk_path)
                if not server_process:
                    return 1
                
                # Wait for server to be ready
                if not asyncio.run(wait_for_server(args.server_url)):
                    return 1
        
        if args.server_only:
            print("🖥️  SyncTalk server is running. Press Ctrl+C to stop.")
            try:
                if server_process:
                    server_process.wait()
            except KeyboardInterrupt:
                print("\n⏹️  Stopping server...")
            return 0
        
        # Start Rapido client
        success = asyncio.run(start_rapido_client())
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  System startup interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        # Cleanup server process if we started it
        if server_process and not args.server_only:
            print("🧹 Cleaning up SyncTalk server...")
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
            except:
                server_process.kill()

if __name__ == "__main__":
    # Add src to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    sys.exit(main())
