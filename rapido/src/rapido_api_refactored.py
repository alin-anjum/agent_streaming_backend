"""
Refactored Rapido API using new modular architecture
"""

import asyncio
import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the refactored Rapido application
from rapido_refactored_main import RapidoAPI
from core.logging_manager import get_logging_manager
from core.security import SecurityManager, SecurityConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Rapido API - Refactored",
    description="Real-time Avatar Presentation API with modular architecture",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Rapido API instance
rapido_api: Optional[RapidoAPI] = None
security_manager: Optional[SecurityManager] = None

# Active sessions storage
active_sessions: Dict[str, Dict[str, Any]] = {}

# Request/Response models
class ProcessLessonRequest(BaseModel):
    lesson_id: str
    slide_data_path: Optional[str] = None
    enable_tts: bool = True
    enable_synctalk: bool = True
    enable_livekit: bool = True

class ProcessLessonResponse(BaseModel):
    success: bool
    lesson_id: str
    processing_time: Optional[float] = None
    slides_processed: Optional[int] = None
    audio_segments: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    session_id: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    application_initialized: bool
    is_processing: bool
    current_lesson_id: Optional[str] = None
    services_status: Dict[str, Any]
    processing_stats: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


@app.on_event("startup")
async def startup_event():
    """Initialize Rapido API on startup"""
    global rapido_api, security_manager
    
    try:
        logger.info("Initializing Rapido API...")
        
        # Initialize security manager
        security_config = SecurityConfig(
            jwt_secret=os.getenv("JWT_SECRET", "rapido_api_secret_change_in_production")
        )
        security_manager = SecurityManager(security_config)
        
        # Create API configuration
        api_config = {
            "logging": {
                "log_dir": os.getenv("RAPIDO_LOG_DIR", "./logs")
            },
            "security": {
                "jwt_secret": security_config.jwt_secret
            },
            "tts": {
                "api_key": os.getenv("ELEVENLABS_API_KEY"),
                "voice_id": os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
            },
            "synctalk": {
                "server_url": os.getenv("SYNCTALK_SERVER_URL", "http://35.172.212.10:8000"),
                "model_name": os.getenv("SYNCTALK_MODEL_NAME", "enrique_torres")
            },
            "livekit": {
                "url": os.getenv("LIVEKIT_URL"),
                "api_key": os.getenv("LIVEKIT_API_KEY"),
                "api_secret": os.getenv("LIVEKIT_API_SECRET")
            },
            "paths": {
                "slide_frames": os.getenv("SLIDE_FRAMES_PATH", "./presentation_frames"),
                "input_data": os.getenv("INPUT_DATA_PATH", "./test1.json")
            }
        }
        
        # Initialize Rapido API
        rapido_api = RapidoAPI(config=api_config)
        
        if await rapido_api.initialize():
            logger.info("✅ Rapido API initialized successfully")
        else:
            logger.error("❌ Failed to initialize Rapido API")
            raise Exception("Rapido API initialization failed")
            
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rapido_api
    
    if rapido_api:
        await rapido_api.shutdown_async()
        logger.info("Rapido API shutdown completed")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {"api": "healthy"}
    
    if rapido_api:
        try:
            status = await rapido_api.get_status_async()
            components["orchestrator"] = "healthy" if status["application_initialized"] else "unhealthy"
        except:
            components["orchestrator"] = "unhealthy"
    else:
        components["orchestrator"] = "not_initialized"
    
    return HealthResponse(
        status="healthy" if all(c == "healthy" for c in components.values()) else "degraded",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
        components=components
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed system status"""
    if not rapido_api:
        raise HTTPException(status_code=503, detail="Rapido API not initialized")
    
    try:
        status = await rapido_api.get_status_async()
        
        return StatusResponse(
            status="operational" if status["application_initialized"] else "initializing",
            application_initialized=status["application_initialized"],
            is_processing=status.get("is_processing", False),
            current_lesson_id=status.get("current_lesson_id"),
            services_status=status.get("services_initialized", {}),
            processing_stats=status.get("processing_stats", {}),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.post("/process_lesson", response_model=ProcessLessonResponse)
async def process_lesson(request: ProcessLessonRequest, background_tasks: BackgroundTasks):
    """Process a lesson"""
    if not rapido_api:
        raise HTTPException(status_code=503, detail="Rapido API not initialized")
    
    if not security_manager:
        raise HTTPException(status_code=503, detail="Security manager not initialized")
    
    try:
        # Validate lesson ID
        security_manager.validate_request(
            lesson_id=request.lesson_id,
            file_path=request.slide_data_path,
            client_ip="api_client"  # In production, get from request
        )
        
        # Check if already processing
        current_status = await rapido_api.get_status_async()
        if current_status.get("is_processing", False):
            raise HTTPException(
                status_code=409, 
                detail=f"Another lesson is being processed: {current_status.get('current_lesson_id')}"
            )
        
        # Generate session ID
        session_id = security_manager.generate_session_token(request.lesson_id, "api_user")
        
        # Store session
        active_sessions[request.lesson_id] = {
            "session_id": session_id,
            "started_at": datetime.utcnow().isoformat(),
            "status": "processing"
        }
        
        # Process lesson
        logger.info(f"Starting lesson processing via API: {request.lesson_id}")
        
        result = await rapido_api.process_lesson_async(
            lesson_id=request.lesson_id,
            slide_data_path=request.slide_data_path
        )
        
        # Update session status
        if request.lesson_id in active_sessions:
            active_sessions[request.lesson_id]["status"] = "completed" if result["success"] else "failed"
            active_sessions[request.lesson_id]["completed_at"] = datetime.utcnow().isoformat()
        
        return ProcessLessonResponse(
            success=result["success"],
            lesson_id=result["lesson_id"],
            processing_time=result.get("processing_time"),
            slides_processed=result.get("slides_processed"),
            audio_segments=result.get("audio_segments"),
            error=result.get("error"),
            error_type=result.get("error_type"),
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lesson processing failed: {e}")
        
        # Update session status
        if request.lesson_id in active_sessions:
            active_sessions[request.lesson_id]["status"] = "failed"
            active_sessions[request.lesson_id]["error"] = str(e)
        
        return ProcessLessonResponse(
            success=False,
            lesson_id=request.lesson_id,
            error=str(e),
            error_type=type(e).__name__
        )


@app.get("/sessions")
async def get_active_sessions():
    """Get active processing sessions"""
    return {
        "active_sessions": active_sessions,
        "total_sessions": len(active_sessions),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/sessions/{lesson_id}")
async def get_session_status(lesson_id: str):
    """Get status of specific session"""
    if lesson_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "lesson_id": lesson_id,
        "session_info": active_sessions[lesson_id],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.delete("/sessions/{lesson_id}")
async def stop_session(lesson_id: str):
    """Stop a processing session"""
    if not rapido_api:
        raise HTTPException(status_code=503, detail="Rapido API not initialized")
    
    try:
        # Stop processing
        stopped = await rapido_api.app.orchestrator.stop_processing() if rapido_api.app.orchestrator else False
        
        # Remove from active sessions
        if lesson_id in active_sessions:
            active_sessions[lesson_id]["status"] = "stopped"
            active_sessions[lesson_id]["stopped_at"] = datetime.utcnow().isoformat()
        
        return {
            "message": f"Session stopped: {lesson_id}",
            "stopped": stopped,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to stop session {lesson_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")


@app.websocket("/ws/lesson/{lesson_id}")
async def websocket_lesson_updates(websocket: WebSocket, lesson_id: str):
    """WebSocket endpoint for real-time lesson processing updates"""
    await websocket.accept()
    
    try:
        # Send initial status
        if rapido_api:
            status = await rapido_api.get_status_async()
            await websocket.send_json({
                "type": "status",
                "lesson_id": lesson_id,
                "data": status,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            if rapido_api:
                status = await rapido_api.get_status_async()
                await websocket.send_json({
                    "type": "update",
                    "lesson_id": lesson_id,
                    "data": {
                        "is_processing": status.get("is_processing", False),
                        "current_lesson_id": status.get("current_lesson_id"),
                        "processing_stats": status.get("processing_stats", {})
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for lesson: {lesson_id}")
    except Exception as e:
        logger.error(f"WebSocket error for lesson {lesson_id}: {e}")
        await websocket.close(code=1011, reason=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An internal error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rapido API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Rapido API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "rapido_api_refactored:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
