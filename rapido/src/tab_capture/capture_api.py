#!/usr/bin/env python3
"""
Tab Capture API
Main integration function for capturing presentation frames dynamically
"""

import asyncio
import logging
from typing import Optional
from PIL import Image
import base64
import io

from .browser_service import BrowserAutomationService, BrowserConfig
from .frame_processor import DynamicFrameProcessor

logger = logging.getLogger(__name__)

async def capture_presentation_frames_to_queue(capture_url: str, frame_queue: asyncio.Queue, duration_seconds: Optional[int] = None) -> bool:
    """
    Start Playwright capture and feed frames directly to queue (no file system)
    """
    try:
        logger.info(f"🚀 Starting dynamic presentation frame capture (direct to queue)")
        logger.info(f"🌐 URL: {capture_url}")
        if duration_seconds:
            logger.info(f"⏱️ Duration: {duration_seconds} seconds")
        else:
            logger.info(f"⏱️ Duration: Auto-detect (ends when slide completes)")
        
        # Configure browser automation
        config = BrowserConfig()
        config.capture_url = capture_url
        config.tab_capture_wait_seconds = duration_seconds or 300  # 5 min max if auto-detect
        
        # Create and start browser service
        browser_service = BrowserAutomationService(config)
        
        if not await browser_service.start():
            logger.error("❌ Failed to start browser service")
            return False
        
        # Navigate and setup capture
        if not await browser_service.navigate_and_setup_capture():
            logger.error("❌ Failed to navigate and setup capture")
            await browser_service.cleanup()
            return False
        
        # Start the long-running capture task in background (non-blocking)
        async def background_capture():
            try:
                await browser_service.capture_frames_live_to_queue(duration_seconds or 300, frame_queue)
            finally:
                await browser_service.cleanup()
        
        # Fire and forget - capture runs in background
        asyncio.create_task(background_capture())
        
        logger.info(f"✅ Dynamic frame capture started in background!")
        logger.info(f"📥 Frames will be fed directly to queue")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dynamic frame capture failed: {e}")
        return False

# Keep the old function for backward compatibility but mark it as deprecated
async def capture_presentation_frames(capture_url: str, duration_seconds: Optional[int] = None) -> Optional[str]:
    """
    DEPRECATED: Use capture_presentation_frames_to_queue instead
    Start Playwright capture **non-blocking** and return the directory
    immediately so Rapido can consume frames while they are still being written.
    """
    try:
        logger.info(f"🚀 Starting dynamic presentation frame capture (non-blocking)")
        logger.info(f"🌐 URL: {capture_url}")
        if duration_seconds:
            logger.info(f"⏱️ Duration: {duration_seconds} seconds")
        else:
            logger.info(f"⏱️ Duration: Auto-detect (ends when slide completes)")
        
        # Configure browser automation
        config = BrowserConfig()
        config.capture_url = capture_url
        config.tab_capture_wait_seconds = duration_seconds or 300  # 5 min max if auto-detect
        
        # Create and start browser service
        browser_service = BrowserAutomationService(config)
        
        if not await browser_service.start():
            logger.error("❌ Failed to start browser service")
            return None
        
        # Navigate and setup capture
        if not await browser_service.navigate_and_setup_capture():
            logger.error("❌ Failed to navigate and setup capture")
            await browser_service.cleanup()
            return None
        
        # Prepare frames directory immediately
        frames_directory = await browser_service.prepare_frames_directory()
        
        # Start the long-running capture task in background (non-blocking)
        async def background_capture():
            try:
                await browser_service.capture_frames_live_to_directory(duration_seconds or 300, frames_directory)
            finally:
                await browser_service.cleanup()
        
        # Fire and forget - capture runs in background
        asyncio.create_task(background_capture())
        
        logger.info(f"✅ Dynamic frame capture started in background!")
        logger.info(f"📁 Frames will be saved to: {frames_directory}")
        
        return frames_directory
        
    except Exception as e:
        logger.error(f"❌ Dynamic frame capture failed: {e}")
        return None

