import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SlideDataParser:
    """Parser for extracting narration data from slide JSON files."""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = Path(json_file_path)
        self.slide_data = None
        self.narration_data = None
        
    def load_data(self) -> bool:
        """Load and parse the JSON data file."""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                self.slide_data = json.load(file)
            
            # Extract narration data
            if 'slide_data' in self.slide_data and 'narrationData' in self.slide_data['slide_data']:
                self.narration_data = self.slide_data['slide_data']['narrationData']
                logger.info(f"Successfully loaded slide data from {self.json_file_path}")
                return True
            else:
                logger.error("No narrationData found in slide_data")
                return False
                
        except FileNotFoundError:
            logger.error(f"File not found: {self.json_file_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_narration_text(self) -> Optional[str]:
        """Extract the narration text."""
        if not self.narration_data:
            return None
        return self.narration_data.get('text', '')
    
    def get_total_duration(self) -> Optional[int]:
        """Get total duration in milliseconds."""
        if not self.narration_data:
            return None
        return self.narration_data.get('totalDuration', 0)
    
    def get_tokens(self) -> Optional[List[Dict[str, Any]]]:
        """Get timing tokens for word-level synchronization."""
        if not self.narration_data:
            return None
        return self.narration_data.get('tokens', [])
    
    def get_presenter_config(self) -> Optional[Dict[str, Any]]:
        """Get presenter configuration for avatar positioning."""
        if not self.slide_data or 'slide_data' not in self.slide_data:
            return None
        return self.slide_data['slide_data'].get('presenterConfig', {})
    
    def get_slide_dimensions(self) -> tuple:
        """Extract slide dimensions from objects (assuming first object defines canvas size)."""
        if not self.slide_data or 'slide_data' not in self.slide_data:
            return (1920, 1080)  # Default HD dimensions
        
        objects = self.slide_data['slide_data'].get('objects', [])
        for obj in objects:
            if obj.get('type') == 'shape' and 'width' in obj and 'height' in obj:
                return (int(obj['width']), int(obj['height']))
        
        return (1920, 1080)  # Default if no dimensions found
    
    def get_timing_info(self) -> List[Dict[str, Any]]:
        """
        Extract timing information for synchronization.
        Returns list of timing events with start/end times and associated elements.
        """
        timing_info = []
        
        if not self.narration_data or 'tokens' not in self.narration_data:
            return timing_info
        
        tokens = self.narration_data['tokens']
        
        for token in tokens:
            timing_info.append({
                'id': token.get('id'),
                'text': token.get('text', ''),
                'start_time': token.get('startTime', 0),
                'end_time': token.get('endTime', 0),
                'duration': token.get('duration', 0),
                'type': token.get('type', 'word')
            })
        
        return timing_info
    
    def extract_animation_triggers(self) -> List[Dict[str, Any]]:
        """Extract animation triggers from slide objects."""
        triggers = []
        
        if not self.slide_data or 'slide_data' not in self.slide_data:
            return triggers
        
        objects = self.slide_data['slide_data'].get('objects', [])
        
        for obj in objects:
            # Check for entry animations
            if 'entryAnimation' in obj:
                entry_anim = obj['entryAnimation']
                triggers.append({
                    'object_id': obj.get('id'),
                    'trigger_type': 'entry',
                    'trigger_id': entry_anim.get('triggerId'),
                    'animation_type': entry_anim.get('animationType'),
                    'duration': entry_anim.get('duration', 1000),
                    'delay': entry_anim.get('delay', 0)
                })
            
            # Check for exit animations
            if 'exitAnimation' in obj:
                exit_anim = obj['exitAnimation']
                triggers.append({
                    'object_id': obj.get('id'),
                    'trigger_type': 'exit',
                    'trigger_id': exit_anim.get('triggerId'),
                    'animation_type': exit_anim.get('animationType'),
                    'duration': exit_anim.get('duration', 1000),
                    'delay': exit_anim.get('delay', 0)
                })
        
        return triggers
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the parsed data."""
        if not self.slide_data:
            return {}
        
        return {
            'slide_id': self.slide_data.get('slide_data', {}).get('id', 'unknown'),
            'narration_text_length': len(self.get_narration_text() or ''),
            'total_duration_ms': self.get_total_duration(),
            'total_duration_seconds': (self.get_total_duration() or 0) / 1000,
            'token_count': len(self.get_tokens() or []),
            'slide_dimensions': self.get_slide_dimensions(),
            'presenter_config': self.get_presenter_config(),
            'animation_triggers_count': len(self.extract_animation_triggers()),
            'has_audio_url': bool(self.narration_data and self.narration_data.get('audioUrl'))
        }
