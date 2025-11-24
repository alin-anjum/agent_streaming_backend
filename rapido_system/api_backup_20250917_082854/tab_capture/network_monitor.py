"""
Network monitoring module for capturing document JSON from XHR requests
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, Optional
from playwright.async_api import Page, Response

logger = logging.getLogger(__name__)

def extract_video_job_id_from_url(url: str) -> Optional[str]:
    """
    Extract video job ID from capture URL
    
    Supports various URL patterns:
    - https://domain.com/video-capture/uuid
    - https://domain.com/capture/uuid  
    - https://domain.com/job/uuid
    - Any UUID pattern in URL
    
    Args:
        url: The URL to extract video job ID from
        
    Returns:
        Video job ID if found, None otherwise
    """
    try:
        # Look for UUID pattern (most common)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        uuid_match = re.search(uuid_pattern, url, re.IGNORECASE)
        if uuid_match:
            return uuid_match.group(0)
        
        # Look for video-capture/ or capture/ followed by any identifier
        capture_patterns = [
            r'/video-capture/([^/?&#]+)',
            r'/capture/([^/?&#]+)',
            r'/job/([^/?&#]+)',
            r'/presentation/([^/?&#]+)'
        ]
        
        for pattern in capture_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        
        logger.debug(f"No video job ID pattern found in URL: {url}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting video job ID from URL {url}: {e}")
        return None

class NetworkDocumentCapture:
    """Monitors network requests to capture document JSON"""
    
    def __init__(self, video_job_id: str, output_dir: str = "./captured_documents"):
        self.video_job_id = video_job_id
        self.output_dir = output_dir
        self.monitoring_active = False
        self.capture_complete = False
        self.captured_document = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🔍 Network monitor initialized for video job ID: {video_job_id}")
        logger.info(f"📁 Documents will be saved to: {output_dir}")
    
    async def start_monitoring(self, page: Page):
        """Start monitoring network requests on the given page"""
        self.monitoring_active = True
        self.capture_complete = False
        
        # Set up request and response handlers
        page.on("request", self._handle_request)
        page.on("response", self._handle_response)
        
        logger.info(f"🌐 Network monitoring started - looking for requests containing: {self.video_job_id}")
        logger.info(f"🔍 Debug mode enabled - will log all network requests and responses")
    
    async def _handle_request(self, request):
        """Handle outgoing requests"""
        if not self.monitoring_active or self.capture_complete:
            return
            
        try:
            url = request.url
            if self.video_job_id in url:
                logger.info(f"🚀 Target request detected: {url}")
        except Exception as e:
            logger.debug(f"Error processing request: {e}")
    
    async def _handle_response(self, response: Response):
        """Handle network responses and check for matching document"""
        if not self.monitoring_active or self.capture_complete:
            return
            
        try:
            # Get request details for better matching
            resource_type = str(response.request.resource_type).lower() if response.request else 'unknown'
            is_xhr = resource_type == 'xhr'
            is_fetch = resource_type == 'fetch'  
            is_api_call = is_xhr or is_fetch
            is_successful = 200 <= response.status < 300
            
            # Check multiple conditions for matching:
            url_matches = self.video_job_id in response.url
            url_path_matches = self.video_job_id in response.url.split('/')[-1] if '/' in response.url else False
            
            # More aggressive matching - check any successful API call or URL match
            should_check = (
                url_matches or 
                url_path_matches or 
                (is_api_call and is_successful)
            )
            
            if should_check:
                logger.info(f"🎯 Checking response: {response.url}")
                
                # Try to get response body as JSON
                try:
                    response_data = await response.json()
                    
                    # Check if this response contains our video job ID or stringifiedDocJson
                    if self._contains_target_data(response_data):
                        logger.info(f"✅ Found target data in response from: {response.url}")
                        await self._process_document_response(response_data, response.url)
                        
                except Exception as json_error:
                    logger.debug(f"Response not JSON from {response.url}: {json_error}")
                    
        except Exception as e:
            logger.debug(f"Error processing response {response.url}: {e}")
    
    def _contains_target_data(self, data: Any, depth: int = 0) -> bool:
        """Recursively check if data contains target video job ID or stringifiedDocJson"""
        if depth > 10:  # Prevent infinite recursion
            return False
            
        try:
            if isinstance(data, dict):
                # Check if this dict has stringifiedDocJson field
                if 'stringifiedDocJson' in data:
                    return True
                    
                # Check if any value contains our video job ID
                for key, value in data.items():
                    if isinstance(value, str) and self.video_job_id in value:
                        return True
                    elif self._contains_target_data(value, depth + 1):
                        return True
                        
            elif isinstance(data, list):
                for item in data:
                    if self._contains_target_data(item, depth + 1):
                        return True
                        
            elif isinstance(data, str):
                if self.video_job_id in data or 'stringifiedDocJson' in data:
                    return True
                    
        except Exception as e:
            logger.debug(f"Error checking data at depth {depth}: {e}")
            
        return False
    
    async def _process_document_response(self, response_data: Dict[str, Any], source_url: str):
        """Process response that contains target data"""
        try:
            logger.info(f"📄 Processing document response from: {source_url}")
            
            # Look for stringifiedDocJson in the response
            stringified_doc_json = self._extract_stringified_doc_json(response_data)
            
            if stringified_doc_json:
                logger.info(f"✅ Found stringifiedDocJson at root level")
                
                # Parse the stringified JSON
                try:
                    document_json = json.loads(stringified_doc_json)
                    
                    # Save the document
                    import time
                    timestamp = int(time.time())
                    filename = f"document_{self.video_job_id}_{timestamp}.json"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(document_json, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"✅ Document JSON saved to: {filepath}")
                    logger.info(f"📊 Document contains {len(document_json)} top-level fields")
                    
                    # Mark capture as complete
                    self.capture_complete = True
                    self.captured_document = {
                        "file_path": filepath,
                        "data": document_json,
                        "source_url": source_url
                    }
                    
                    logger.info(f"🎉 Document capture completed for video job ID: {self.video_job_id}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse stringifiedDocJson: {e}")
            else:
                logger.warning(f"⚠️ No stringifiedDocJson found in response from {source_url}")
                
        except Exception as e:
            logger.error(f"❌ Error processing document response: {e}")
    
    def _extract_stringified_doc_json(self, data: Any, depth: int = 0) -> Optional[str]:
        """Recursively extract stringifiedDocJson from response data"""
        if depth > 10:  # Prevent infinite recursion
            return None
            
        try:
            if isinstance(data, dict):
                # Direct check for stringifiedDocJson field
                if 'stringifiedDocJson' in data:
                    return data['stringifiedDocJson']
                    
                # Recursive search in nested objects
                for value in data.values():
                    result = self._extract_stringified_doc_json(value, depth + 1)
                    if result:
                        return result
                        
            elif isinstance(data, list):
                for item in data:
                    result = self._extract_stringified_doc_json(item, depth + 1)
                    if result:
                        return result
                        
        except Exception as e:
            logger.debug(f"Error extracting stringifiedDocJson at depth {depth}: {e}")
            
        return None
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.monitoring_active = False
        logger.info(f"🛑 Network monitoring stopped")
    
    def is_capture_complete(self) -> bool:
        """Check if document capture is complete"""
        return self.capture_complete
    
    def get_captured_document(self) -> Optional[Dict[str, Any]]:
        """Get the captured document if available"""
        return self.captured_document

async def capture_document_from_network(page: Page, video_job_id: str, timeout_seconds: int = 60, output_dir: str = "./captured_documents") -> Optional[Dict[str, Any]]:
    """
    Capture document JSON from network requests on a page
    
    Args:
        page: Playwright page to monitor
        video_job_id: Video job ID to look for in requests
        timeout_seconds: How long to wait for document capture
        output_dir: Directory to save captured documents
        
    Returns:
        Dictionary with captured document info or None if not found
    """
    monitor = NetworkDocumentCapture(video_job_id, output_dir)
    
    try:
        await monitor.start_monitoring(page)
        
        # Wait for capture to complete or timeout
        start_time = asyncio.get_event_loop().time()
        while not monitor.is_capture_complete():
            if (asyncio.get_event_loop().time() - start_time) > timeout_seconds:
                logger.warning(f"⏰ Timeout reached ({timeout_seconds}s) - document capture incomplete")
                break
            await asyncio.sleep(0.1)
        
        return monitor.get_captured_document()
        
    finally:
        monitor.stop_monitoring()