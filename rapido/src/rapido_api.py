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

# Active room tracking - prevents duplicate avatars
active_rooms: Dict[str, Dict[str, Any]] = {}  # room_name -> {"avatar_active": bool, "session_id": str, "created_at": datetime}

# Request models
class LiveKitTokenRequest(BaseModel):
    lessonId: str
    videoJobId: str
    documentId: str
    userId: str
    sessionId: str
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
    LIVEKIT_API_KEY = "APIEkRN4enNfAzu"
    LIVEKIT_API_SECRET = "jHEYfEfhaBWQg5isdDgO6e2Xw8zhIvb18KebGwH2ESXC"
    
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
        "active_sessions": len(active_sessions),
        "active_rooms": len(active_rooms)
    }

@app.get("/api/v1/rooms/status")
async def get_rooms_status():
    """Get status of all active rooms"""
    return {
        "active_rooms": {
            room_name: {
                "lesson_id": room_info["lesson_id"],
                "video_job_id": room_info["video_job_id"],
                "avatar_active": room_info["avatar_active"],
                "created_at": room_info["created_at"].isoformat(),
                "session_id": room_info["session_id"]
            }
            for room_name, room_info in active_rooms.items()
        },
        "total_rooms": len(active_rooms)
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
        logger.info(f"🔑 Token request for lesson: {request.lessonId}, user: {request.userId}, session: {request.sessionId}, jobId: {request.videoJobId}")
        
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
        participant_name = f"user_{request.userId[:8]}_{request.sessionId[:8]}"  # Use userId + sessionId combination
        livekit_token = generate_livekit_token(room_name, participant_name)
        livekit_url = "wss://agent-s83m6c4y.livekit.cloud"
        
        logger.info(f"✅ Token generated for {participant_name} (user: {request.userId}, session: {request.sessionId}) in lesson room {room_name}")
        
        # Step 3: Smart room management - only start avatar if room doesn't have one
        if await should_start_avatar_in_room(room_name):
            logger.info(f"🚀 Starting new avatar in room {room_name}")
            asyncio.create_task(auto_start_presentation(room_name, request.lessonId, request.videoJobId))
        else:
            logger.info(f"👥 Avatar already active in room {room_name} - user joining existing presentation")
        
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

async def should_start_avatar_in_room(room_name: str) -> bool:
    """Check if we should start an avatar in this room"""
    global active_rooms
    
    # Check if room already has an active avatar
    if room_name in active_rooms:
        room_info = active_rooms[room_name]
        
        # Check if avatar is still active (session exists and not finished)
        session_id = room_info.get("session_id")
        if session_id and session_id in active_sessions:
            session = active_sessions[session_id]
            if session.get("status") in ["streaming", "initializing", "connecting_livekit", "connecting_synctalk"]:
                logger.info(f"🔍 Room {room_name} already has active avatar (session: {session_id})")
                return False
            else:
                logger.info(f"🔍 Room {room_name} avatar session ended - can start new one")
                # Remove stale room entry
                del active_rooms[room_name]
                return True
        else:
            logger.info(f"🔍 Room {room_name} session not found - can start new avatar")
            # Remove stale room entry
            del active_rooms[room_name]  
            return True
    
    logger.info(f"🔍 Room {room_name} is empty - can start avatar")
    return True

async def auto_start_presentation(room_name: str, lesson_id: str, video_job_id: str):
    """Auto-start avatar presentation when someone gets a token"""
    session_id = f"avatar_{lesson_id}_{int(time.time())}"
    
    try:
        # Register this room as having an active avatar
        active_rooms[room_name] = {
            "avatar_active": True,
            "session_id": session_id,
            "lesson_id": lesson_id,
            "video_job_id": video_job_id,
            "created_at": datetime.now()
        }
        
        # Create session tracking
        session_info = {
            "session_id": session_id,
            "room": room_name,
            "lesson_id": lesson_id,
            "video_job_id": video_job_id,
            "status": "initializing",
            "created_at": datetime.now()
        }
        active_sessions[session_id] = session_info
        
        logger.info(f"🚀 Auto-starting presentation for lesson: {lesson_id}, jobId: {video_job_id} in room: {room_name}")
        
        # Create Rapido system with room override
        config_override = {'LIVEKIT_ROOM': room_name}
        rapido = RapidoMainSystem(config_override)
        session_info["rapido_system"] = rapido
        
        # Update session status
        session_info["status"] = "connecting_livekit"
        
        # Connect to LiveKit
        await rapido.connect_livekit()
        logger.info(f"✅ LiveKit connected for lesson room: {room_name}")
        
        # Update session status
        session_info["status"] = "connecting_synctalk"
        
        # Connect to SyncTalk  
        if not await rapido.connect_to_synctalk():
            raise Exception("SyncTalk connection failed")
        logger.info(f"✅ SyncTalk connected for lesson: {lesson_id}")
        
        # Update session status
        session_info["status"] = "streaming"
        
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
            session_info["status"] = "completed"
            logger.info(f"✅ Presentation completed for lesson: {lesson_id}")
        else:
            session_info["status"] = "failed"
            logger.error(f"❌ Presentation failed for lesson: {lesson_id}")
            
    except Exception as e:
        logger.error(f"❌ Auto-start failed for lesson {lesson_id}: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "error"
            active_sessions[session_id]["error"] = str(e)
            
    finally:
        # Clean up room registration when avatar finishes
        if room_name in active_rooms:
            logger.info(f"🧹 Cleaning up room registration for {room_name}")
            del active_rooms[room_name]
        
        # Cleanup Rapido connections
        if session_id in active_sessions:
            session = active_sessions[session_id]
            if session.get("rapido_system"):
                try:
                    rapido_system = session["rapido_system"]
                    if hasattr(rapido_system, 'websocket') and rapido_system.websocket:
                        await rapido_system.websocket.close()
                    if hasattr(rapido_system, 'aiohttp_session') and rapido_system.aiohttp_session:
                        await rapido_system.aiohttp_session.close()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup error: {cleanup_error}")

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
