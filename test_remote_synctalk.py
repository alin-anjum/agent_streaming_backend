#!/usr/bin/env python3
"""
Test script to check connectivity to the remote SyncTalk server.
"""

import asyncio
import aiohttp
import json

async def test_synctalk_server():
    """Test connection to the remote SyncTalk server."""
    server_url = "http://13.221.206.227:8000"
    
    print(f"Testing connection to SyncTalk server: {server_url}")
    
    try:
        # Test basic HTTP connection
        async with aiohttp.ClientSession() as session:
            print("1. Testing basic HTTP connection...")
            async with session.get(f"{server_url}/", timeout=10) as response:
                print(f"   Status: {response.status}")
                if response.status == 200:
                    text = await response.text()
                    print(f"   Response: {text[:200]}...")
                else:
                    print(f"   Error response: {response.status}")
            
            # Test model loading endpoint
            print("\n2. Testing model loading endpoint...")
            model_request = {"model_name": "enrique_torres"}
            async with session.post(
                f"{server_url}/load_model", 
                json=model_request,
                timeout=30
            ) as response:
                print(f"   Status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    print(f"   Model loaded: {result}")
                else:
                    text = await response.text()
                    print(f"   Error: {text}")
            
            # Test WebSocket endpoint availability
            print("\n3. Testing WebSocket endpoint availability...")
            try:
                import websockets
                ws_url = "ws://13.221.206.227:8000/ws/audio_to_video"
                async with websockets.connect(ws_url, open_timeout=10) as websocket:
                    print(f"   WebSocket connection successful!")
                    await websocket.close()
            except Exception as e:
                print(f"   WebSocket connection failed: {e}")
                
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_synctalk_server())
