#!/usr/bin/env python3
"""
Azure Proxy Wrapper for Rapido API

Lightweight FastAPI proxy that forwards all requests to the AWS rapido API.
Deploy this on Azure with a CNAME for stable domain access.
"""

import asyncio
import httpx
import os
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Rapido API Proxy",
    description="Azure proxy wrapper for AWS Rapido API",
    version="1.0.0"
)

# CORS middleware - production ready
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domains
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configuration - Direct connection to AWS server IP
AWS_RAPIDO_BASE_URL = os.getenv("AWS_RAPIDO_BASE_URL", "http://54.82.5.74:8080")

# HTTP client with timeout
client = httpx.AsyncClient(timeout=300.0)

@app.on_event("startup")
async def startup():
    """Log startup information"""
    logger.info(f"🚀 Azure Proxy Wrapper starting")
    logger.info(f"🎯 Proxying to: {AWS_RAPIDO_BASE_URL}")
    logger.info(f"🔗 Health check: {AWS_RAPIDO_BASE_URL}/health")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await client.aclose()

@app.get("/health")
async def health_check():
    """Proxy health check"""
    try:
        # Check both proxy and target health
        response = await client.get(f"{AWS_RAPIDO_BASE_URL}/health", timeout=10.0)
        target_healthy = response.status_code == 200
        
        return {
            "proxy_status": "healthy",
            "target_status": "healthy" if target_healthy else "unhealthy",
            "target_url": AWS_RAPIDO_BASE_URL,
            "proxy_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "proxy_status": "healthy", 
            "target_status": "unhealthy",
            "target_url": AWS_RAPIDO_BASE_URL,
            "error": str(e)
        }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
async def proxy_all_requests(request: Request, path: str):
    """Proxy all requests to AWS rapido API"""
    try:
        # Construct target URL
        target_url = f"{AWS_RAPIDO_BASE_URL}/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
        
        # Get request body
        body = await request.body()
        
        # Forward headers (exclude host-specific ones)
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        
        logger.info(f"🔄 {request.method} {path} -> {target_url}")
        
        # Make proxied request
        response = await client.request(
            method=request.method,
            url=target_url,
            content=body,
            headers=headers,
            timeout=300.0
        )
        
        # Forward response headers (exclude problematic ones)
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("transfer-encoding", None)
        response_headers.pop("connection", None)
        
        # Return proxied response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response_headers.get("content-type")
        )
        
    except httpx.TimeoutException:
        logger.error(f"Timeout proxying {request.method} {path}")
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except httpx.ConnectError:
        logger.error(f"Connection error proxying {request.method} {path}")
        raise HTTPException(status_code=502, detail="Bad gateway - target service unreachable")
    except Exception as e:
        logger.error(f"Error proxying {request.method} {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    
    logger.info(f"🌐 Starting Azure Proxy Wrapper on {HOST}:{PORT}")
    logger.info(f"🎯 Target: {AWS_RAPIDO_BASE_URL}")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )
