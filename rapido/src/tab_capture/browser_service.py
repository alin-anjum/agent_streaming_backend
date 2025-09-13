#!/usr/bin/env python3
"""
Browser Automation Service for Dynamic Frame Capture
Integrates with Rapido system to capture presentation frames in real-time using Playwright and Chrome extension
"""

import asyncio
import logging
import time
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import os

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
except ImportError:
    raise ImportError("Playwright not installed. Run: pip install playwright && playwright install chromium")

logger = logging.getLogger(__name__)

class BrowserConfig:
    """Configuration for browser automation"""
    def __init__(self):
        self.headless = False  # Use headed browser for tab capture
        # Primary Resolution (Used Throughout)
        self.viewport_width = 1920
        self.viewport_height = 1080
        self.tab_capture_wait_seconds = 30  # How long to capture
        # CDP Screencast FPS - optimized for 25fps
        self.capture_fps = 25
        self.download_folder = "./captured_frames"
        self.extension_path = "./rapido/chrome_extension"
        self.browser_data_dir = "./browser_data"  # Chrome user profile directory
        self.capture_url = "https://test.creatium.com/presentation"  # Default capture URL
        self.capture_method = "cdp"  # Options: "cdp", "extension", "flag_based"
        
        # Cleanup options
        self.auto_cleanup_browser_data = False  # Auto-remove browser data after each run
        self.max_browser_data_size_mb = 500     # Max size before cleanup (MB)
        self.cleanup_every_n_runs = 0          # Clean every N runs (0 = disabled)

class BrowserAutomationService:
    """Browser automation service for dynamic frame capture"""
    
    def __init__(self, config: BrowserConfig = None):
        self.config = config or BrowserConfig()
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.frames_captured = []
        
        # Ensure directories exist
        os.makedirs(self.config.download_folder, exist_ok=True)
        
        # Track run count for periodic cleanup
        self.run_count = 0
        self._load_run_count()
        
        logger.info("🚀 Browser Automation Service initialized")
        logger.info(f"📁 Frames will be saved to: {self.config.download_folder}")
        
        # Check if cleanup is needed
        if self._should_cleanup_before_start():
            logger.info("🧹 Browser data cleanup needed before start")
            self._cleanup_browser_data()
    
    async def start(self):
        """Start browser automation service"""
        try:
            logger.info("🌐 Starting browser automation service...")
            
            self.playwright = await async_playwright().start()
            
            # Launch browser with extension
            extension_path = Path(self.config.extension_path).resolve()
            if not extension_path.exists():
                raise FileNotFoundError(f"Chrome extension not found at: {extension_path}")
            
            logger.info(f"🔌 Loading Chrome extension from: {extension_path}")
            
            # Viewport Settings
            viewport = {"width": self.config.viewport_width, "height": self.config.viewport_height}
            
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.config.browser_data_dir,
                headless=self.config.headless,
                viewport=viewport,
                args=[
                    f"--load-extension={extension_path}",
                    "--disable-extensions-except=" + str(extension_path),
                    "--disable-web-security",
                    "--allow-running-insecure-content",
                    "--autoplay-policy=no-user-gesture-required",
                    "--auto-select-tab-capture-source-by-title=Creatium",  # Auto-select Creatium tab
                    "--enable-usermedia-tab-capturing",  # Enable tab capture
                    "--disable-features=VizDisplayCompositor",  # Better capture
                    "--disable-user-media-security",  # Bypass media security
                    "--allow-http-background-page",  # Allow background operations
                    # Window Size Arguments - exact 1920x1080
                    f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
                    "--force-device-scale-factor=1.0",  # 1:1 scale for exact 1920x1080
                    "--window-position=0,0"
                ],
                ignore_default_args=["--disable-extensions"]
            )
            
            # Get the first page or create one
            pages = self.context.pages
            if pages:
                self.page = pages[0]
            else:
                self.page = await self.context.new_page()
            
            logger.info("✅ Browser started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start browser: {e}")
            return False

    async def navigate_and_setup_capture(self):
        """Navigate to capture URL and setup for frame capture"""
        try:
            logger.info(f"🌐 Navigating to: {self.config.capture_url}")
            
            # Navigate to the capture URL
            await self.page.goto(self.config.capture_url, wait_until="networkidle", timeout=30000)
            
            logger.info("✅ Page loaded successfully")
            
            # Wait a moment for any dynamic content to load
            await asyncio.sleep(2)
            
            # Handle tunnel protection if present
            logger.info("🔍 Checking for tunnel protection after navigation...")
            await self._handle_tunnel_protection_button()
            
            logger.info("🎯 Ready for frame capture")
            return True
            
        except Exception as e:
            logger.error(f"❌ Navigation failed: {e}")
            return False

    async def prepare_frames_directory(self) -> str:
        """Create and return the frames directory immediately"""
        import time
        timestamp = int(time.time())
        frames_dir = f"{self.config.download_folder}/live_frames_{timestamp}"
        os.makedirs(frames_dir, exist_ok=True)
        logger.info(f"📁 Created frames directory: {frames_dir}")
        return frames_dir

    async def capture_frames_live_to_queue(self, duration_seconds: int, frame_queue: "asyncio.Queue"):
        """Capture frames live and feed directly to queue (no file system)"""
        try:
            logger.info(f"🎬 Starting live frame capture for {duration_seconds} seconds (direct to queue)")
            
            # Use slide capture method but feed to queue instead of directory
            await self._automated_play_and_slide_capture_to_queue(duration_seconds, frame_queue)
            
            logger.info(f"✅ Live capture completed - frames fed to queue")
                
        except Exception as e:
            logger.error(f"❌ Live frame capture to queue failed: {e}")
            raise

    async def capture_frames_live_to_directory(self, duration_seconds: int, target_directory: str):
        """Capture frames live and save to specific directory"""
        try:
            logger.info(f"🎬 Starting live frame capture for {duration_seconds} seconds to {target_directory}")
            
            # Use slide capture method but save to specified directory
            await self._automated_play_and_slide_capture_to_dir(duration_seconds, target_directory)
            
            logger.info(f"✅ Live capture completed in: {target_directory}")
                
        except Exception as e:
            logger.error(f"❌ Live frame capture failed: {e}")
            raise

    async def capture_frames_live(self, duration_seconds: int) -> Optional[str]:
        """Capture frames live from the browser (legacy method)"""
        try:
            logger.info(f"🎬 Starting live frame capture for {duration_seconds} seconds")
            
            # Use slide capture method (your existing working approach)
            frames_directory = await self._automated_play_and_slide_capture(duration_seconds)
            
            if frames_directory:
                logger.info(f"✅ Live capture completed: {frames_directory}")
                return frames_directory
            else:
                logger.error("❌ Live capture failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Live frame capture failed: {e}")
            return None

    async def _setup_slide_capture_hooks(self):
        """Install JavaScript hooks for slide capture monitoring"""
        try:
            logger.info("🔗 Installing slide capture hooks...")
            
            # Install comprehensive slide capture monitoring
            await self.page.evaluate("""
                () => {
                    // Initialize global variables for slide capture
                    window.frameBuffer = window.frameBuffer || [];
                    window.bufferSize = 0;
                    window.slideCaptureComplete = false;
                    
                    // Hook into console.log to monitor slide's useDomCapture
                    const originalLog = console.log;
                    console.log = function(...args) {
                        originalLog.apply(console, args);
                        
                        // Monitor for slide capture events
                        const logStr = args.join(' ');
                        if (logStr.includes('useDomCapture') || logStr.includes('captureFrame')) {
                            window.slideSystemActive = true;
                        }
                        
                        // Monitor for completion signals
                        if (logStr.includes('complete') || logStr.includes('finished') || logStr.includes('done')) {
                            window.slideCaptureComplete = true;
                        }
                    };
                    
                    // Hook into canvas operations to intercept frames
                    const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
                    window.lastCaptureTime = 0;
                    const targetFPS = 25;
                    const frameInterval = 1000 / targetFPS; // 40ms between frames
                    
                    HTMLCanvasElement.prototype.toDataURL = function(type, quality) {
                        const dataURL = originalToDataURL.call(this, type, quality);
                        
                        // Store ALL frames (no throttling for debugging)
                        if (dataURL && dataURL.startsWith('data:image/')) {
                            window.frameBuffer[window.bufferSize] = dataURL;
                            window.bufferSize++;
                            
                            // Log progress every 10 frames for better monitoring
                            if (window.bufferSize % 10 === 0) {
                                console.log(`📸 Slide capture buffer: ${window.bufferSize} frames (NO throttling)`);
                            }
                        }
                        
                        return dataURL;
                    };
                    
                    // Hook into toBlob as alternative
                    const originalToBlob = HTMLCanvasElement.prototype.toBlob;
                    HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
                        const canvas = this;
                        const wrappedCallback = function(blob) {
                            if (blob) {
                                // Convert blob to data URL for consistent storage
                                const reader = new FileReader();
                                reader.onload = function() {
                                    window.frameBuffer[window.bufferSize] = reader.result;
                                    window.bufferSize++;
                                    
                                    if (window.bufferSize % 10 === 0) {
                                        console.log(`📸 Slide capture buffer: ${window.bufferSize} frames (toBlob)`);
                                    }
                                };
                                reader.readAsDataURL(blob);
                            }
                            callback(blob);
                        };
                        originalToBlob.call(canvas, wrappedCallback, type, quality);
                    };
                    
                    console.log('🔧 Slide capture hooks installed successfully');
                    
                    // Debug: Log when any canvas method is called
                    console.log('🎯 Canvas hooks ready - waiting for toDataURL/toBlob calls...');
                }
            """)
            
            logger.info("✅ Slide capture hooks installed")
            
        except Exception as e:
            logger.error(f"❌ Failed to install slide capture hooks: {e}")
            raise

    async def _handle_tunnel_protection_button(self):
        """Handle tunnel/phishing protection button if present"""
        try:
            # Look for the specific tunnel protection button
            tunnel_button_selectors = [
                'button#continue.tunnel--dwithport-button',
                'button[id="continue"]',
                'button.tunnel--dwithport-button',
                'button:has-text("Continue")',
                'button[onclick*="setCookie"][onclick*="tunnel_phishing_protection"]'
            ]
            
            tunnel_button_found = False
            for selector in tunnel_button_selectors:
                try:
                    logger.info(f"🔍 Checking for tunnel protection button: {selector}")
                    # Use a shorter timeout since this is optional
                    await self.page.wait_for_selector(selector, timeout=3000)
                    logger.info(f"✅ Found tunnel protection button: {selector}")
                    tunnel_button_found = True
                    
                    # Click the tunnel protection button
                    logger.info("🛡️ Clicking tunnel protection 'Continue' button...")
                    await self.page.click(selector)
                    logger.info("✅ Tunnel protection button clicked successfully")
                    
                    # Wait for the page to process the tunnel protection
                    logger.info("⏱️ Waiting for tunnel protection to process...")
                    await asyncio.sleep(2)
                    
                    break
                    
                except Exception as e:
                    logger.debug(f"Tunnel button selector {selector} failed: {e}")
                    continue
            
            if tunnel_button_found:
                logger.info("🛡️ Tunnel protection handled successfully")
            else:
                logger.info("🔍 No tunnel protection button found - proceeding normally")
                
        except Exception as e:
            logger.warning(f"⚠️ Error handling tunnel protection button: {e}")
            # Don't raise - this is optional functionality

    async def _automated_play_and_slide_capture_to_queue(self, duration: int, frame_queue: "asyncio.Queue"):
        """Automated play button click and slide capture monitoring directly to queue"""
        try:
            logger.info("🎬 Starting automated play and slide capture (direct to queue)...")
            
            # First, check for tunnel/phishing protection button
            logger.info("🔍 Checking for tunnel protection button...")
            await self._handle_tunnel_protection_button()
            
            # Then, look for play button
            logger.info("🔍 Looking for play button on the webpage...")
            
            play_button_selectors = [
                'button[class*="play"]',
                'button[aria-label*="play" i]',
                'button[title*="play" i]', 
                'button:has-text("Play")',
                'button:has-text("▶")',
                '.play-button',
                '[data-testid*="play"]',
                'input[type="button"][value*="play" i]'
            ]
            
            play_button_found = False
            for selector in play_button_selectors:
                try:
                    logger.info(f"🔍 Trying selector: {selector}")
                    await self.page.wait_for_selector(selector, timeout=2000)
                    logger.info(f"✅ Found play button with selector: {selector}")
                    play_button_found = True
                    
                    # Install capture hooks right before clicking
                    logger.info("🔗 Installing capture hooks right before play button click...")
                    await self._setup_slide_capture_hooks()
                    
                    # Enable CDP screencast for non-canvas presentations
                    logger.info("📹 Setting up CDP screencast for non-canvas presentation...")
                    session = await self.page.context.new_cdp_session(self.page)
                    await session.send("Page.startScreencast", {
                        "format": "png",
                        "quality": 90,
                        "everyNthFrame": 1  # Capture every frame, no throttling
                    })
                    logger.info("✅ CDP screencast enabled - will capture presentation frames")
                    
                    # Click the play button using JavaScript for reliability
                    logger.info("🎯 Clicking play button with JavaScript...")
                    await self.page.evaluate(f"""
                        () => {{
                            const button = document.querySelector('{selector}');
                            if (button) {{
                                button.click();
                                console.log('✅ Play button clicked via JavaScript');
                                return true;
                            }}
                            return false;
                        }}
                    """)
                    
                    logger.info("✅ Play button clicked with JavaScript!")
                    break
                    
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not play_button_found:
                logger.error("❌ No play button found")
                raise Exception("No play button found")
            
            # Wait for slide capture to activate
            logger.info("⏱️ Waiting for slide's useDomCapture to activate...")
            await asyncio.sleep(0.5)
            
            # Monitor slide capture directly to queue (with CDP support)
            await self._handle_slide_capture_monitoring_to_queue_with_cdp(duration, frame_queue)
            
        except Exception as e:
            logger.error(f"❌ Automated play and slide capture to queue failed: {e}")
            raise

    async def _automated_play_and_slide_capture_to_dir(self, duration: int, target_directory: str):
        """Automated play button click and slide capture monitoring to specific directory"""
        try:
            logger.info("🎬 Starting automated play and slide capture...")
            
            # First, check for tunnel/phishing protection button
            logger.info("🔍 Checking for tunnel protection button...")
            await self._handle_tunnel_protection_button()
            
            # Then, look for play button
            logger.info("🔍 Looking for play button on the webpage...")
            
            play_button_selectors = [
                'button[class*="play"]',
                'button[aria-label*="play" i]',
                'button[title*="play" i]', 
                'button:has-text("Play")',
                'button:has-text("▶")',
                '.play-button',
                '[data-testid*="play"]',
                'input[type="button"][value*="play" i]'
            ]
            
            play_button_found = False
            for selector in play_button_selectors:
                try:
                    logger.info(f"🔍 Trying selector: {selector}")
                    await self.page.wait_for_selector(selector, timeout=2000)
                    logger.info(f"✅ Found play button with selector: {selector}")
                    play_button_found = True
                    
                    # Install capture hooks right before clicking
                    logger.info("🔗 Installing capture hooks right before play button click...")
                    await self._setup_slide_capture_hooks()
                    
                    # Enable CDP screencast for non-canvas presentations
                    logger.info("📹 Setting up CDP screencast for non-canvas presentation...")
                    session = await self.page.context.new_cdp_session(self.page)
                    await session.send("Page.startScreencast", {
                        "format": "png",
                        "quality": 90,
                        "everyNthFrame": 1  # Capture every frame, no throttling
                    })
                    logger.info("✅ CDP screencast enabled - will capture presentation frames")
                    
                    # Click the play button using JavaScript for reliability
                    logger.info("🎯 Clicking play button with JavaScript...")
                    await self.page.evaluate(f"""
                        () => {{
                            const button = document.querySelector('{selector}');
                            if (button) {{
                                button.click();
                                console.log('✅ Play button clicked via JavaScript');
                                return true;
                            }}
                            return false;
                        }}
                    """)
                    
                    logger.info("✅ Play button clicked with JavaScript!")
                    break
                    
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not play_button_found:
                logger.error("❌ No play button found")
                raise Exception("No play button found")
            
            # Wait for slide capture to activate
            logger.info("⏱️ Waiting for slide's useDomCapture to activate...")
            await asyncio.sleep(0.5)
            
            # Monitor slide capture to specific directory (with CDP support)
            await self._handle_slide_capture_monitoring_to_dir_with_cdp(duration, target_directory)
            
        except Exception as e:
            logger.error(f"❌ Automated play and slide capture failed: {e}")
            raise

    async def _automated_play_and_slide_capture(self, duration: int) -> Optional[str]:
        """Automated play button click and slide capture monitoring"""
        try:
            logger.info("🎬 Starting automated play and slide capture...")
            
            # First, check for tunnel/phishing protection button
            logger.info("🔍 Checking for tunnel protection button...")
            await self._handle_tunnel_protection_button()
            
            # Then, look for play button
            logger.info("🔍 Looking for play button on the webpage...")
            
            play_button_selectors = [
                'button[class*="play"]',
                'button[aria-label*="play" i]',
                'button[title*="play" i]', 
                'button:has-text("Play")',
                'button:has-text("▶")',
                '.play-button',
                '[data-testid*="play"]',
                'input[type="button"][value*="play" i]'
            ]
            
            play_button_found = False
            for selector in play_button_selectors:
                try:
                    logger.info(f"🔍 Trying selector: {selector}")
                    await self.page.wait_for_selector(selector, timeout=2000)
                    logger.info(f"✅ Found play button with selector: {selector}")
                    play_button_found = True
                    
                    # Install capture hooks right before clicking
                    logger.info("🔗 Installing capture hooks right before play button click...")
                    await self._setup_slide_capture_hooks()
                    
                    # Click the play button using JavaScript for reliability
                    logger.info("🎯 Clicking play button with JavaScript...")
                    await self.page.evaluate(f"""
                        () => {{
                            const button = document.querySelector('{selector}');
                            if (button) {{
                                button.click();
                                console.log('✅ Play button clicked via JavaScript');
                                return true;
                            }}
                            return false;
                        }}
                    """)
                    
                    logger.info("✅ Play button clicked with JavaScript!")
                    break
                    
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not play_button_found:
                logger.error("❌ No play button found")
                return None
            
            # Wait for slide capture to activate
            logger.info("⏱️ Waiting for slide's useDomCapture to activate...")
            await asyncio.sleep(0.5)
            
            # Monitor slide capture
            frames_directory = await self._handle_slide_capture_monitoring(duration)
            
            return frames_directory
            
        except Exception as e:
            logger.error(f"❌ Automated play and slide capture failed: {e}")
            return None

    async def _handle_slide_capture_monitoring_to_dir_with_cdp(self, duration: int, target_directory: str):
        """Monitor slide capture with CDP screencast fallback"""
        try:
            logger.info("🎬 Starting slide capture monitoring with CDP support...")
            frames_dir = Path(target_directory)
            
            total_saved = 0
            start_time = time.time()
            cdp_session = None
            
            # Get CDP session for screencast
            try:
                cdp_session = await self.page.context.new_cdp_session(self.page)
                logger.info("📹 CDP session established for frame capture")
            except Exception as e:
                logger.warning(f"⚠️ Could not establish CDP session: {e}")
            
            # Monitor for frames
            while time.time() - start_time < duration:
                try:
                    frames_captured_this_cycle = 0
                    
                    # First try canvas buffer (if presentation uses canvas)
                    buffer_status = await self.page.evaluate("""
                        () => {
                            return {
                                bufferSize: window.bufferSize || 0,
                                isComplete: window.slideCaptureComplete || false
                            };
                        }
                    """)
                    
                    buffer_size = buffer_status.get('bufferSize', 0)
                    
                    if buffer_size > total_saved:
                        # Process canvas frames
                        frames_to_process = min(buffer_size - total_saved, 8)
                        
                        for frame_idx in range(total_saved, min(total_saved + frames_to_process, buffer_size)):
                            try:
                                frame_data = await self.page.evaluate(f"""
                                    () => {{
                                        if (window.frameBuffer && window.frameBuffer[{frame_idx}]) {{
                                            return window.frameBuffer[{frame_idx}];
                                        }}
                                        return null;
                                    }}
                                """)
                                
                                if frame_data and isinstance(frame_data, str) and frame_data.startswith('data:image/png;base64,'):
                                    import base64
                                    image_data = base64.b64decode(frame_data.split(',')[1])
                                    
                                    frame_filename = f"frame_{total_saved:06d}.png"
                                    frame_path = frames_dir / frame_filename
                                    with open(frame_path, 'wb') as f:
                                        f.write(image_data)
                                    
                                    total_saved += 1
                                    frames_captured_this_cycle += 1
                                    
                            except Exception as e:
                                logger.debug(f"Canvas frame {frame_idx} processing failed: {e}")
                                continue
                    
                    # If no canvas frames, try CDP screencast
                    if frames_captured_this_cycle == 0 and cdp_session:
                        try:
                            # Request screencast frame
                            result = await cdp_session.send("Page.captureScreenshot", {
                                "format": "png",
                                "quality": 90
                            })
                            
                            if result and 'data' in result:
                                import base64
                                image_data = base64.b64decode(result['data'])
                                
                                frame_filename = f"frame_{total_saved:06d}.png"
                                frame_path = frames_dir / frame_filename
                                with open(frame_path, 'wb') as f:
                                    f.write(image_data)
                                
                                total_saved += 1
                                frames_captured_this_cycle += 1
                                
                        except Exception as e:
                            logger.debug(f"CDP screenshot failed: {e}")
                    
                    # Log progress
                    if total_saved % 25 == 0 and frames_captured_this_cycle > 0:
                        elapsed = time.time() - start_time
                        fps = total_saved / elapsed if elapsed > 0 else 0
                        logger.info(f"📸 Saved {total_saved} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
                    
                    # Sleep for next cycle
                    await asyncio.sleep(0.04)  # 25 FPS timing
                    
                except Exception as e:
                    logger.debug(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(0.1)
                    continue
            
            logger.info(f"✅ Slide capture monitoring completed: {total_saved} frames saved to {frames_dir}")
            
        except Exception as e:
            logger.error(f"❌ Slide capture monitoring failed: {e}")
            raise

    async def _handle_slide_capture_monitoring_to_queue_with_cdp(self, duration: int, frame_queue: "asyncio.Queue"):
        """Monitor slide capture and feed frames directly to queue with CDP screencast fallback"""
        try:
            logger.info("🎬 Starting slide capture monitoring (direct to queue)...")
            
            total_saved = 0
            start_time = time.time()
            cdp_session = None
            
            # Get CDP session for screencast
            try:
                cdp_session = await self.page.context.new_cdp_session(self.page)
                logger.info("📹 CDP session established for frame capture")
            except Exception as e:
                logger.warning(f"⚠️ Could not establish CDP session: {e}")
            
            # Monitor for frames
            while time.time() - start_time < duration:
                try:
                    frames_captured_this_cycle = 0
                    
                    # First try canvas buffer (if presentation uses canvas)
                    buffer_status = await self.page.evaluate("""
                        () => {
                            return {
                                bufferSize: window.bufferSize || 0,
                                isComplete: window.slideCaptureComplete || false
                            };
                        }
                    """)
                    
                    buffer_size = buffer_status.get('bufferSize', 0)
                    
                    if buffer_size > total_saved:
                        # Process canvas frames
                        frames_to_process = min(buffer_size - total_saved, 8)
                        
                        for frame_idx in range(total_saved, min(total_saved + frames_to_process, buffer_size)):
                            try:
                                frame_data = await self.page.evaluate(f"""
                                    () => {{
                                        if (window.frameBuffer && window.frameBuffer[{frame_idx}]) {{
                                            return window.frameBuffer[{frame_idx}];
                                        }}
                                        return null;
                                    }}
                                """)
                                
                                if frame_data and isinstance(frame_data, str) and frame_data.startswith('data:image/png;base64,'):
                                    # Convert base64 to PIL Image and feed to queue
                                    import base64
                                    from PIL import Image
                                    import io
                                    
                                    image_data = base64.b64decode(frame_data.split(',')[1])
                                    frame_image = Image.open(io.BytesIO(image_data)).resize((854, 480))
                                    
                                    # Put frame in queue (non-blocking)
                                    try:
                                        frame_queue.put_nowait((total_saved, frame_image))
                                        total_saved += 1
                                        frames_captured_this_cycle += 1
                                    except Exception as queue_error:
                                        # If queue is full, remove oldest and add new
                                        try:
                                            frame_queue.get_nowait()  # Remove oldest
                                            frame_queue.put_nowait((total_saved, frame_image))
                                            total_saved += 1
                                            frames_captured_this_cycle += 1
                                        except Exception:
                                            logger.debug(f"Queue operation failed: {queue_error}")
                                    
                            except Exception as e:
                                logger.debug(f"Canvas frame {frame_idx} processing failed: {e}")
                                continue
                    
                    # If no canvas frames, try CDP screencast
                    if frames_captured_this_cycle == 0 and cdp_session:
                        try:
                            # Request screencast frame
                            result = await cdp_session.send("Page.captureScreenshot", {
                                "format": "png",
                                "quality": 90
                            })
                            
                            if result and 'data' in result:
                                # Convert base64 to PIL Image and feed to queue
                                import base64
                                from PIL import Image
                                import io
                                
                                image_data = base64.b64decode(result['data'])
                                frame_image = Image.open(io.BytesIO(image_data)).resize((854, 480))
                                
                                # Put frame in queue (non-blocking)
                                try:
                                    frame_queue.put_nowait((total_saved, frame_image))
                                    total_saved += 1
                                    frames_captured_this_cycle += 1
                                except Exception as queue_error:
                                    # If queue is full, remove oldest and add new
                                    try:
                                        frame_queue.get_nowait()  # Remove oldest
                                        frame_queue.put_nowait((total_saved, frame_image))
                                        total_saved += 1
                                        frames_captured_this_cycle += 1
                                    except Exception:
                                        logger.debug(f"Queue operation failed: {queue_error}")
                                
                        except Exception as e:
                            logger.debug(f"CDP screenshot failed: {e}")
                    
                    # Log progress
                    if total_saved % 25 == 0 and frames_captured_this_cycle > 0:
                        elapsed = time.time() - start_time
                        fps = total_saved / elapsed if elapsed > 0 else 0
                        queue_size = frame_queue.qsize()
                        logger.info(f"📸 Fed {total_saved} frames to queue in {elapsed:.1f}s ({fps:.1f} FPS, queue size: {queue_size})")
                    
                    # Sleep for next cycle
                    await asyncio.sleep(0.04)  # 25 FPS timing
                    
                except Exception as e:
                    logger.debug(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(0.1)
                    continue
            
            queue_size = frame_queue.qsize()
            logger.info(f"✅ Slide capture monitoring completed: {total_saved} frames fed to queue (final queue size: {queue_size})")
            
        except Exception as e:
            logger.error(f"❌ Slide capture monitoring to queue failed: {e}")
            raise

    async def _handle_slide_capture_monitoring_to_dir_old(self, duration: int, target_directory: str):
        """Monitor slide capture and save frames to specific directory"""
        try:
            logger.info("🎬 Starting slide capture monitoring...")
            frames_dir = Path(target_directory)
            
            total_saved = 0
            start_time = time.time()
            
            # Monitor for frames
            while time.time() - start_time < duration:
                try:
                    # Check buffer status
                    buffer_status = await self.page.evaluate("""
                        () => {
                            return {
                                bufferSize: window.bufferSize || 0,
                                isComplete: window.slideCaptureComplete || false
                            };
                        }
                    """)
                    
                    buffer_size = buffer_status.get('bufferSize', 0)
                    is_complete = buffer_status.get('isComplete', False)
                    
                    if buffer_size > total_saved:
                        # Process new frames (optimized batch size for performance)
                        frames_to_process = min(buffer_size - total_saved, 8)
                        
                        for frame_idx in range(total_saved, min(total_saved + frames_to_process, buffer_size)):
                            try:
                                # Get frame data
                                frame_data = await self.page.evaluate(f"""
                                    () => {{
                                        if (window.frameBuffer && window.frameBuffer[{frame_idx}]) {{
                                            return window.frameBuffer[{frame_idx}];
                                        }}
                                        return null;
                                    }}
                                """)
                                
                                if frame_data:
                                    # Handle different frame data formats
                                    if isinstance(frame_data, str) and frame_data.startswith('data:image/png;base64,'):
                                        # Base64 string format
                                        import base64
                                        image_data = base64.b64decode(frame_data.split(',')[1])
                                    elif isinstance(frame_data, dict) and 'data' in frame_data:
                                        # Dictionary format with data field
                                        data_url = frame_data['data']
                                        if data_url and data_url.startswith('data:image/png;base64,'):
                                            import base64
                                            image_data = base64.b64decode(data_url.split(',')[1])
                                        else:
                                            continue
                                    else:
                                        logger.debug(f"Unknown frame data format: {type(frame_data)}")
                                        continue
                                    
                                    # Save frame
                                    frame_filename = f"frame_{frame_idx:06d}.png"
                                    frame_path = frames_dir / frame_filename
                                    with open(frame_path, 'wb') as f:
                                        f.write(image_data)
                                    
                            except Exception as e:
                                logger.debug(f"Error processing frame {frame_idx}: {e}")
                                continue
                        
                        total_saved = min(total_saved + frames_to_process, buffer_size)
                        
                        # Log progress
                        if total_saved % 50 == 0 or total_saved == buffer_size:
                            elapsed = time.time() - start_time
                            fps = total_saved / elapsed if elapsed > 0 else 0
                            if total_saved % 50 == 0:  # Log every 50 frames
                                logger.info(f"📸 Saved {total_saved} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
                    
                    # Check for completion
                    if is_complete:
                        logger.info("🏁 Slide capture completed by animation")
                        break
                    
                    # Short sleep to avoid busy waiting
                    await asyncio.sleep(0.02)  # 50Hz check rate
                    
                except Exception as e:
                    logger.debug(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(0.1)
                    continue
            
            logger.info(f"✅ Slide capture monitoring completed: {total_saved} frames saved to {frames_dir}")
            
        except Exception as e:
            logger.error(f"❌ Slide capture monitoring failed: {e}")
            raise

    async def _handle_slide_capture_monitoring(self, duration: int) -> Optional[str]:
        """Monitor slide capture and save frames"""
        try:
            logger.info("🎬 Starting slide capture monitoring...")
            
            # Create timestamp for this capture session
            timestamp = int(time.time())
            frames_dir = Path(self.config.download_folder) / f"user_initiated_frames_{timestamp}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created frames directory: {frames_dir}")
            
            total_saved = 0
            start_time = time.time()
            
            # Monitor for frames
            while time.time() - start_time < duration:
                try:
                    # Check buffer status
                    buffer_status = await self.page.evaluate("""
                        () => {
                            return {
                                bufferSize: window.bufferSize || 0,
                                isComplete: window.slideCaptureComplete || false
                            };
                        }
                    """)
                    
                    buffer_size = buffer_status.get('bufferSize', 0)
                    is_complete = buffer_status.get('isComplete', False)
                    
                    if buffer_size > total_saved:
                        # Process new frames (optimized batch size for performance)
                        frames_to_process = min(buffer_size - total_saved, 8)
                        
                        for frame_idx in range(total_saved, min(total_saved + frames_to_process, buffer_size)):
                            try:
                                # Get frame data
                                frame_data = await self.page.evaluate(f"""
                                    () => {{
                                        if (window.frameBuffer && window.frameBuffer[{frame_idx}]) {{
                                            return window.frameBuffer[{frame_idx}];
                                        }}
                                        return null;
                                    }}
                                """)
                                
                                if frame_data:
                                    # Handle different frame data formats
                                    if isinstance(frame_data, str) and frame_data.startswith('data:image/png;base64,'):
                                        # Base64 string format
                                        import base64
                                        image_data = base64.b64decode(frame_data.split(',')[1])
                                    elif isinstance(frame_data, dict) and 'data' in frame_data:
                                        # Dictionary format with data field
                                        data_url = frame_data['data']
                                        if data_url and data_url.startswith('data:image/png;base64,'):
                                            import base64
                                            image_data = base64.b64decode(data_url.split(',')[1])
                                        else:
                                            continue
                                    else:
                                        logger.debug(f"Unknown frame data format: {type(frame_data)}")
                                        continue
                                    
                                    # Save frame
                                    frame_filename = f"frame_{frame_idx:06d}.png"
                                    frame_path = frames_dir / frame_filename
                                    
                                    with open(frame_path, 'wb') as f:
                                        f.write(image_data)
                                    
                                    total_saved += 1
                                    
                            except Exception as e:
                                logger.warning(f"⚠️ Frame {frame_idx} processing failed: {e}")
                                continue
                        
                        # Log progress every 25 frames
                        if total_saved % 25 == 0:
                            elapsed = time.time() - start_time
                            fps = total_saved / elapsed if elapsed > 0 else 0
                            if total_saved % 50 == 0:  # Log every 50 frames
                                logger.info(f"📸 Saved {total_saved} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
                    
                    # Check completion but continue for full duration
                    if is_complete:
                        logger.info("🎯 Slide capture marked complete, continuing capture for full duration...")
                    
                    # Wait before next check (optimized for 25 FPS)
                    await asyncio.sleep(0.04)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Monitoring iteration failed: {e}")
                    await asyncio.sleep(0.1)
                    continue
            
            logger.info(f"✅ Slide capture monitoring completed: {total_saved} frames saved to {frames_dir}")
            
            if total_saved > 0:
                return str(frames_dir)
            else:
                logger.error("❌ No frames were captured")
                return None
            
        except Exception as e:
            logger.error(f"❌ Slide capture monitoring failed: {e}")
            return None

    def _load_run_count(self):
        """Load run count from file for periodic cleanup"""
        try:
            run_count_file = Path(self.config.browser_data_dir).parent / ".browser_run_count"
            if run_count_file.exists():
                self.run_count = int(run_count_file.read_text().strip())
            else:
                self.run_count = 0
        except:
            self.run_count = 0
    
    def _save_run_count(self):
        """Save run count to file"""
        try:
            run_count_file = Path(self.config.browser_data_dir).parent / ".browser_run_count"
            run_count_file.write_text(str(self.run_count))
        except:
            pass
    
    def _get_browser_data_size_mb(self) -> int:
        """Get browser data directory size in MB"""
        try:
            total_size = 0
            browser_data_path = Path(self.config.browser_data_dir)
            if not browser_data_path.exists():
                return 0
                
            for dirpath, dirnames, filenames in os.walk(browser_data_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except (OSError, FileNotFoundError):
                        continue
            return total_size // (1024 * 1024)  # Convert to MB
        except:
            return 0
    
    def _should_cleanup_before_start(self) -> bool:
        """Check if cleanup is needed before starting"""
        # Check size-based cleanup
        if self.config.max_browser_data_size_mb > 0:
            current_size = self._get_browser_data_size_mb()
            if current_size > self.config.max_browser_data_size_mb:
                logger.info(f"🧹 Browser data size ({current_size}MB) exceeds limit ({self.config.max_browser_data_size_mb}MB)")
                return True
        
        # Check periodic cleanup
        if self.config.cleanup_every_n_runs > 0:
            if self.run_count > 0 and self.run_count % self.config.cleanup_every_n_runs == 0:
                logger.info(f"🧹 Periodic cleanup triggered (run {self.run_count})")
                return True
        
        return False
    
    def _cleanup_browser_data(self):
        """Clean browser data directory (synchronous)"""
        try:
            import shutil
            browser_data_path = Path(self.config.browser_data_dir)
            if browser_data_path.exists():
                shutil.rmtree(browser_data_path)
                logger.info(f"🧹 Browser data directory removed: {browser_data_path}")
                
                # Reset run count after cleanup
                self.run_count = 0
                self._save_run_count()
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup browser data: {e}")

    # REMOVED: Extension and flag-based tab capture methods - unused
 
    async def cleanup(self, remove_browser_data: bool = None):
        """Clean up browser resources"""
        try:
            # Increment run count
            self.run_count += 1
            self._save_run_count()
            
            if self.context:
                await self.context.close()
                logger.info("🧹 Browser context closed")
            
            if self.playwright:
                await self.playwright.stop()
                logger.info("🧹 Playwright stopped")
            
            # Determine if should remove browser data
            should_remove = remove_browser_data
            if should_remove is None:  # Auto-decide
                should_remove = self.config.auto_cleanup_browser_data
            
            # Remove browser data directory
            if should_remove:
                import shutil
                browser_data_path = Path(self.config.browser_data_dir)
                if browser_data_path.exists():
                    shutil.rmtree(browser_data_path)
                    logger.info(f"🧹 Browser data directory removed: {browser_data_path}")
                    # Reset run count after manual cleanup
                    self.run_count = 0
                    self._save_run_count()
            else:
                # Log current browser data size
                size_mb = self._get_browser_data_size_mb()
                logger.info(f"📊 Browser data size: {size_mb}MB (run #{self.run_count})")
                
        except Exception as e:
            logger.error(f"⚠️ Error during cleanup: {e}")
