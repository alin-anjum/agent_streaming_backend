#!/usr/bin/env python3
"""
Document Parser for Captured Documents
Parses captured document JSON to extract slide information
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_canvas_slide_ids(document_data: Dict[str, Any]) -> List[str]:
    """
    Extract slide IDs for slides with contentType 'canvas' from document data
    
    Args:
        document_data: Parsed JSON document data
        
    Returns:
        List of slide IDs that have contentType 'canvas'
    """
    canvas_slide_ids = []
    
    try:
        # Check if document has slides array
        if 'slides' not in document_data:
            logger.warning("No 'slides' field found in document data")
            return canvas_slide_ids
        
        slides = document_data['slides']
        if not isinstance(slides, list):
            logger.warning("'slides' field is not a list")
            return canvas_slide_ids
        
        logger.info(f"Found {len(slides)} slides in document")
        
        # Iterate through slides and extract canvas slide IDs
        for slide in slides:
            if not isinstance(slide, dict):
                logger.warning("Slide is not a dictionary, skipping")
                continue
            
            slide_id = slide.get('id')
            content_type = slide.get('contentType')
            
            if not slide_id:
                logger.warning("Slide missing 'id' field, skipping")
                continue
            
            if content_type == 'canvas':
                canvas_slide_ids.append(slide_id)
                logger.info(f"Added canvas slide ID: {slide_id}")
            else:
                logger.debug(f"Skipping slide {slide_id} with contentType: {content_type}")
        
        logger.info(f"Extracted {len(canvas_slide_ids)} canvas slide IDs")
        return canvas_slide_ids
        
    except Exception as e:
        logger.error(f"Error extracting canvas slide IDs: {e}")
        return []

def extract_canvas_slides_with_narration(document_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Extract canvas slides with their narration text
    
    Args:
        document_data: Parsed JSON document data
        
    Returns:
        Dictionary mapping slide IDs to their details:
        {
            "slide-id": {
                "narration": "narration text",
                "slide_id": "slide-id"
            }
        }
    """
    canvas_slides = {}
    
    try:
        # Check if document has slides array
        if 'slides' not in document_data:
            logger.warning("No 'slides' field found in document data")
            return canvas_slides
        
        slides = document_data['slides']
        if not isinstance(slides, list):
            logger.warning("'slides' field is not a list")
            return canvas_slides
        
        logger.info(f"Found {len(slides)} slides in document")
        
        # Iterate through slides and extract canvas slides with narration
        for slide in slides:
            if not isinstance(slide, dict):
                logger.warning("Slide is not a dictionary, skipping")
                continue
            
            slide_id = slide.get('id')
            content_type = slide.get('contentType')
            
            if not slide_id:
                logger.warning("Slide missing 'id' field, skipping")
                continue
            
            if content_type == 'canvas':
                # Extract narration text
                narration_text = ""
                narration_data = slide.get('narrationData', {})
                
                if isinstance(narration_data, dict):
                    narration_text = narration_data.get('text', '')
                
                if not narration_text:
                    logger.warning(f"No narration text found for slide {slide_id}")
                    narration_text = f"No narration available for slide {slide_id}"
                
                canvas_slides[slide_id] = {
                    "narration": narration_text,
                    "slide_id": slide_id
                }
                
                logger.info(f"Added canvas slide: {slide_id} with narration ({len(narration_text)} chars)")
            else:
                logger.debug(f"Skipping slide {slide_id} with contentType: {content_type}")
        
        logger.info(f"Extracted {len(canvas_slides)} canvas slides with narration")
        return canvas_slides
        
    except Exception as e:
        logger.error(f"Error extracting canvas slides with narration: {e}")
        return {}

def generate_slide_url_dictionary(document_data: Dict[str, Any], video_job_id: str, base_url: str = "https://xgzhc339-5173.inc1.devtunnels.ms/video-capture") -> Dict[str, Dict[str, str]]:
    """
    Generate URL dictionary for canvas slides with slideId parameters
    
    Args:
        document_data: Parsed JSON document data
        video_job_id: Video job ID for URL generation
        base_url: Base URL for capture endpoints
        
    Returns:
        Dictionary mapping slide IDs to their capture URLs and narration:
        {
            "slide-id": {
                "url": "https://example.com/video-capture/jobId?slideId=slide-id",
                "narration": "narration text",
                "slide_id": "slide-id"
            }
        }
    """
    try:
        # First extract canvas slides with narration
        canvas_slides = extract_canvas_slides_with_narration(document_data)
        
        # Generate URLs for each slide
        slide_url_dict = {}
        for slide_id, slide_data in canvas_slides.items():
            capture_url = f"{base_url}/{video_job_id}?slideId={slide_id}"
            
            slide_url_dict[slide_id] = {
                "url": capture_url,
                "narration": slide_data["narration"],
                "slide_id": slide_id
            }
            
            logger.info(f"Generated URL for slide {slide_id}: {capture_url}")
        
        logger.info(f"Generated {len(slide_url_dict)} slide URLs")
        return slide_url_dict
        
    except Exception as e:
        logger.error(f"Error generating slide URL dictionary: {e}")
        return {}

def parse_document_file(file_path: str) -> List[str]:
    """
    Parse a document JSON file and extract canvas slide IDs
    
    Args:
        file_path: Path to the JSON document file
        
    Returns:
        List of canvas slide IDs
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Document file not found: {file_path}")
            return []
        
        logger.info(f"Parsing document file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            document_data = json.load(f)
        
        return extract_canvas_slide_ids(document_data)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in document file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing document file {file_path}: {e}")
        return []

def get_slide_details(document_data: Dict[str, Any], slide_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific slide
    
    Args:
        document_data: Parsed JSON document data
        slide_id: ID of the slide to get details for
        
    Returns:
        Slide details dictionary or None if not found
    """
    try:
        if 'slides' not in document_data:
            return None
        
        for slide in document_data['slides']:
            if isinstance(slide, dict) and slide.get('id') == slide_id:
                return slide
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting slide details for {slide_id}: {e}")
        return None

def test_document_parser():
    """
    Test function to parse an existing document and print canvas slide IDs
    """
    # Test with the captured document
    test_file = "/home/ubuntu/agent_streaming_backend/captured_documents/document_81eceadf-2503-4915-a2bf-12eb252329e4_1757838848.json"
    video_job_id = "81eceadf-2503-4915-a2bf-12eb252329e4"
    
    print("🧪 Testing Document Parser V2 (with narration & URLs)")
    print("=" * 60)
    
    try:
        # Test original function
        canvas_slide_ids = parse_document_file(test_file)
        
        if canvas_slide_ids:
            print(f"✅ Found {len(canvas_slide_ids)} canvas slides:")
            for i, slide_id in enumerate(canvas_slide_ids, 1):
                print(f"  {i}. {slide_id}")
        else:
            print("❌ No canvas slides found")
        
        # Test new functionality with document data
        print("\n🔍 Testing narration extraction...")
        with open(test_file, 'r', encoding='utf-8') as f:
            document_data = json.load(f)
        
        # Test canvas slides with narration
        canvas_slides = extract_canvas_slides_with_narration(document_data)
        print(f"📊 Extracted {len(canvas_slides)} canvas slides with narration:")
        
        for slide_id, slide_data in canvas_slides.items():
            narration_preview = slide_data['narration'][:100] + "..." if len(slide_data['narration']) > 100 else slide_data['narration']
            print(f"  - {slide_id[:8]}...: {narration_preview}")
        
        # Test URL dictionary generation
        print(f"\n🌐 Testing URL dictionary generation...")
        url_dict = generate_slide_url_dictionary(document_data, video_job_id)
        print(f"📊 Generated {len(url_dict)} slide URLs:")
        
        for slide_id, slide_info in url_dict.items():
            print(f"  - {slide_id[:8]}...: {slide_info['url']}")
            print(f"    Narration: {len(slide_info['narration'])} characters")
        
        return url_dict
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {}

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Run test
    test_document_parser()
