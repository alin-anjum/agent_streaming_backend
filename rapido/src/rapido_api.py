#!/usr/bin/env python3
"""
Rapido API Service - Real-time Avatar Presentation API

FastAPI service that accepts presentation requests from frontend:
- Token-based authentication
- Presentation ID becomes LiveKit room name
- Real-time progress updates
- Optimized SyncTalk integration
"""

import asyncio
import os
import sys
import json
import logging
import time
import jwt
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
rapido_root = os.path.dirname(current_dir)
project_root = os.path.dirname(rapido_root)
sys.path.extend([rapido_root, current_dir, os.path.join(project_root, 'SyncTalk_2D')])

# Import Rapido system
from rapido_main import RapidoMainSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Rapido API",
    description="Real-time Avatar Presentation API",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to log all requests and handle URL issues
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Clean up URL path (remove trailing spaces/encoded characters)
    cleaned_path = request.url.path.rstrip().rstrip('/')
    
    # Log request details with cleaned path
    logger.info(f"📥 {request.method} {cleaned_path} from {request.client.host if request.client else 'unknown'}")
    logger.info(f"🔍 Original path: '{request.url.path}' → Cleaned: '{cleaned_path}'")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(f"📤 Response: {response.status_code} in {process_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"🚨 Request processing error: {e}")
        logger.error(f"🔍 Request details: method={request.method}, path='{request.url.path}', query={request.url.query}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Note: Catch-all route moved to end of file to avoid intercepting valid routes

# Active sessions storage
active_sessions: Dict[str, Dict[str, Any]] = {}

# Request models
class LiveKitTokenRequest(BaseModel):
    lessonId: str
    videoJobId: str
    documentId: str
    userId: str
    organizationId: str
    authToken: str

class AuthenticateDocumentPayload(BaseModel):
    id: str         # lessonId
    userId: str
    organizationId: str
    documentId: str

class LiveKitTokenResponse(BaseModel):
    room: str           # lessonId  
    serverUrl: str      # LiveKit server URL
    token: str          # JWT token

# Removed old complex models - keeping it simple for token service

def generate_livekit_token(room_name: str, participant_name: str) -> str:
    """Generate LiveKit JWT token for frontend room access"""
    current_time = int(time.time())
    
    # LiveKit credentials (should be moved to environment variables)
    LIVEKIT_API_KEY = "APImuXsSp8NH5jY"
    LIVEKIT_API_SECRET = "6k9Swe5O6NxeI0WvVTCTrs2k1Ec25byeM4NlnTCKn5GB"
    
    token_payload = {
        "iss": LIVEKIT_API_KEY,
        "sub": participant_name,
        "aud": "livekit",
        "exp": current_time + 3600,  # 1 hour expiry
        "nbf": current_time - 10,
        "iat": current_time,
        "jti": f"{participant_name}_{current_time}",
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,     # Allow frontend to publish (they might send audio/video too)
            "canSubscribe": True    # Allow subscribing to avatar stream
        }
    }
    
    return jwt.encode(token_payload, LIVEKIT_API_SECRET, algorithm="HS256")

@app.post("/api/v1/presentations/start")
async def start_presentation_endpoint(
    room: str,
    identity: str, 
    narration_text: str,
    avatar_name: str = "enrique_torres"
):
    """Start avatar presentation in specified LiveKit room"""
    try:
        session_id = f"{room}_{identity}_{int(time.time())}"
        
        logger.info(f"🚀 Starting presentation in room: {room} for identity: {identity}")
        
        # Store session
        session_info = {
            "session_id": session_id,
            "room": room,
            "identity": identity,
            "avatar_name": avatar_name,
            "status": "starting",
            "created_at": datetime.now()
        }
        active_sessions[session_id] = session_info
        
        # Start Rapido system with room override
        config_override = {'LIVEKIT_ROOM': room}
        
        # Start processing in background
        task = asyncio.create_task(
            start_rapido_session(session_id, narration_text, avatar_name, config_override)
        )
        session_info["task"] = task
        
        return {
            "session_id": session_id,
            "status": "starting",
            "message": f"Presentation started in room {room}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start presentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions)
    }

@app.get("/")
async def root():
    """Root endpoint - often hit by load balancers/health checks"""
    logger.info("📍 Root endpoint accessed")
    return {
        "service": "Rapido API",
        "status": "running",
        "endpoints": {
            "token": "/api/v1/livekit/token",
            "health": "/health"
        }
    }

@app.head("/")
async def root_head():
    """HEAD request to root - common for health checks"""
    logger.info("📍 HEAD request to root")
    return {}

@app.options("/api/v1/livekit/token")
async def token_options():
    """OPTIONS request for CORS preflight"""
    logger.info("📍 OPTIONS request for token endpoint")
    return {}

async def authenticate_with_creatium(auth_payload: AuthenticateDocumentPayload, auth_token: str) -> bool:
    """Authenticate with Creatium API"""
    import aiohttp
    
    # Configuration - should be moved to environment variables
    CREATIUM_API_BASE_URL = "https://api.creatium.com"  # Update with actual URL
    
    auth_url = f"{CREATIUM_API_BASE_URL}/v1/Content/AuthenticateDocument"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            auth_url,
            json={
                "id": auth_payload.id,
                "userId": auth_payload.userId,
                "organizationId": auth_payload.organizationId,
                "documentId": auth_payload.documentId
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {auth_token}"
            }
        ) as response:
            if response.status != 200:
                logger.warning(f"Authentication failed: {response.status} {response.reason}")
                return False
            
            data = await response.json()
            is_authenticated = data.get("data", {}).get("isAuthenticated", False)
            
            logger.info(f"Authentication result: {is_authenticated} for user {auth_payload.userId}")
            return is_authenticated

@app.post("/api/v1/livekit/token", response_model=LiveKitTokenResponse)
@app.post("/api/v1/livekit/token/", response_model=LiveKitTokenResponse)  # Handle trailing slash
async def generate_livekit_token_endpoint(request: LiveKitTokenRequest):
    """Generate LiveKit JWT token after authentication validation"""
    try:
        logger.info(f"🔑 Token request for lesson: {request.lessonId}, user: {request.userId}, jobId: {request.videoJobId}")
        
        # Step 1: Authenticate with Creatium API
        # TEMPORARILY BYPASSED FOR TESTING - uncomment for production
        # auth_payload = AuthenticateDocumentPayload(
        #     id=request.lessonId,
        #     userId=request.userId,
        #     organizationId=request.organizationId,
        #     documentId=request.documentId
        # )
        # 
        # is_authenticated = await authenticate_with_creatium(auth_payload, request.authToken)
        # 
        # if not is_authenticated:
        #     logger.warning(f"❌ Authentication failed for user {request.userId}, lesson {request.lessonId}")
        #     raise HTTPException(status_code=401, detail="Authentication failed")
        
        # TESTING: Skip auth for now
        logger.info("🧪 TESTING: Authentication bypassed - using real auth for production")
        
        # Step 2: Generate LiveKit token (lessonId becomes room name)
        room_name = request.lessonId  # Use lessonId as room name
        participant_name = f"user_{request.userId[:8]}"  # Use shortened userId as participant name
        livekit_token = generate_livekit_token(room_name, participant_name)
        livekit_url = "wss://rapido-pme0lo9d.livekit.cloud"
        
        logger.info(f"✅ Token generated for {participant_name} in lesson room {room_name}")
        
        # Step 3: Auto-trigger avatar presentation in this room
        asyncio.create_task(auto_start_presentation(room_name, request.lessonId, request.videoJobId))
        
        return LiveKitTokenResponse(
            room=room_name,
            serverUrl=livekit_url,
            token=livekit_token
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Failed to generate LiveKit token: {e}")
        raise HTTPException(status_code=500, detail="Token generation failed")

async def auto_start_presentation(room_name: str, lesson_id: str, video_job_id: str):
    """Auto-start avatar presentation when someone gets a token"""
    try:
        logger.info(f"🚀 Auto-starting presentation for lesson: {lesson_id}, jobId: {video_job_id} in room: {room_name}")
        
        # Create Rapido system with room override
        config_override = {'LIVEKIT_ROOM': room_name}
        rapido = RapidoMainSystem(config_override)
        
        # Connect to LiveKit
        await rapido.connect_livekit()
        logger.info(f"✅ LiveKit connected for lesson room: {room_name}")
        
        # Connect to SyncTalk  
        if not await rapido.connect_to_synctalk():
            raise Exception("SyncTalk connection failed")
        logger.info(f"✅ SyncTalk connected for lesson: {lesson_id}")
        
        # Load actual lesson content - use existing test1.json for now
        # TODO: Replace with actual lesson content loading based on lesson_id
        from data_parser import SlideDataParser
        
        # Use existing test content (replace with real lesson loading)
        test_json_path = "/home/ubuntu/agent_streaming_backend/test1.json"
        parser = SlideDataParser(test_json_path)
        
        if parser.load_data():
            actual_narration = parser.get_narration_text()
            logger.info(f"📝 Loaded actual narration: {len(actual_narration)} characters")
        else:
            logger.error("Failed to load lesson content - using fallback")
            actual_narration = "Failed to load lesson content"
        
        success = await rapido.stream_real_time_tts(actual_narration)
        
        if success:
            logger.info(f"✅ Presentation completed for lesson: {lesson_id}")
        else:
            logger.error(f"❌ Presentation failed for lesson: {lesson_id}")
            
    except Exception as e:
        logger.error(f"❌ Auto-start failed for lesson {lesson_id}: {e}")

async def start_rapido_session(session_id: str, narration_text: str, avatar_name: str, config_override: Dict[str, Any]):
    """Background task to run Rapido presentation"""
    session = active_sessions[session_id]
    
    try:
        session["status"] = "initializing"
        
        # Create and configure Rapido system
        rapido = RapidoMainSystem(config_override)
        session["rapido_system"] = rapido
        
        # Connect to LiveKit
        session["status"] = "connecting_livekit" 
        await rapido.connect_livekit()
        logger.info(f"✅ LiveKit connected for session: {session_id}")
        
        # Connect to SyncTalk
        session["status"] = "connecting_synctalk"
        if not await rapido.connect_to_synctalk(avatar_name):
            raise Exception("SyncTalk connection failed")
        logger.info(f"✅ SyncTalk connected for session: {session_id}")
        
        # Start streaming
        session["status"] = "streaming"
        logger.info(f"🎭 Starting streaming for session: {session_id}")
        
        success = await rapido.stream_real_time_tts(narration_text)
        
        if success:
            session["status"] = "complete"
            logger.info(f"✅ Presentation completed: {session_id}")
        else:
            raise Exception("Streaming failed")
            
    except Exception as e:
        logger.error(f"❌ Session {session_id} failed: {e}")
        session["status"] = "error"
        session["error"] = str(e)
        
    finally:
        # Cleanup connections
        if session.get("rapido_system"):
            try:
                rapido_system = session["rapido_system"]
                if hasattr(rapido_system, 'websocket') and rapido_system.websocket:
                    await rapido_system.websocket.close()
                if hasattr(rapido_system, 'aiohttp_session') and rapido_system.aiohttp_session:
                    await rapido_system.aiohttp_session.close()
            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")

@app.get("/api/v1/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get status of a presentation session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    return {
        "session_id": session["session_id"],
        "room": session["room"], 
        "identity": session["identity"],
        "status": session["status"],
        "created_at": session["created_at"].isoformat()
    }


# Note: Removed catch-all route to avoid interference with main endpoints

if __name__ == "__main__":
    print("🚀 Starting Rapido API Service")
    print("=" * 50)
    
    # Configuration
    HOST = "0.0.0.0"
    PORT = 8080
    
    logger.info(f"Starting Rapido API on {HOST}:{PORT}")
    logger.info(f"🔍 Enhanced logging enabled - will capture all requests including invalid ones")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="debug",  # More verbose logging
        access_log=True,
        server_header=False,  # Reduce server info
        date_header=False     # Reduce headers
    )
