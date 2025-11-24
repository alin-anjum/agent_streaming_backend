"""
Tab Capture Module for Rapido
Handles browser automation and dynamic frame capture functionality
"""

from .browser_service import BrowserAutomationService, BrowserConfig
from .frame_processor import DynamicFrameProcessor
from .capture_api import capture_presentation_frames

__all__ = [
    'BrowserAutomationService',
    'BrowserConfig', 
    'DynamicFrameProcessor',
    'capture_presentation_frames'
]

