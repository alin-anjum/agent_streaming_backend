// Content script for tab capture automation
console.log('🎯 Tab Capture Extension Content Script Loaded');

// Listen for messages from the page (automation)
window.addEventListener('message', (event) => {
  // Only accept messages from same origin for security
  if (event.origin !== window.location.origin) {
    return;
  }
  
  const { type, data } = event.data;
  
  console.log('📨 Content script received message:', { type, data });
  
  switch (type) {
    case 'TAB_CAPTURE_START':
      handleCaptureStart(data);
      break;
      
    case 'TAB_CAPTURE_STOP':
      handleCaptureStop(data);
      break;
      
    case 'TAB_CAPTURE_STATUS':
      handleCaptureStatus(data);
      break;
      
    default:
      // console.log('Unknown message type:', type);
  }
});

async function handleCaptureStart(data = {}) {
  try {
    console.log('🎬 Starting tab capture with data:', data);
    
    // Send message to background script
    const response = await chrome.runtime.sendMessage({
      action: 'START_TAB_CAPTURE',
      includeAudio: data.includeAudio !== false, // Default true
      duration: data.duration || 30000, // Default 30 seconds
      format: data.format || 'webm'
    });
    
    console.log('📡 Background response:', response);
    
    // Send response back to page
    window.postMessage({
      type: 'TAB_CAPTURE_START_RESPONSE',
      data: response
    }, window.location.origin);
    
  } catch (error) {
    console.error('❌ Capture start failed:', error);
    
    window.postMessage({
      type: 'TAB_CAPTURE_START_RESPONSE',
      data: { 
        success: false, 
        error: error.message 
      }
    }, window.location.origin);
  }
}

// Expose helper functions to the page for easy automation
window.TabCaptureAutomation = {
  start: (options = {}) => {
    window.postMessage({
      type: 'TAB_CAPTURE_START',
      data: options
    }, window.location.origin);
  },
  
  stop: () => {
    window.postMessage({
      type: 'TAB_CAPTURE_STOP',
      data: {}
    }, window.location.origin);
  },
  
  status: () => {
    window.postMessage({
      type: 'TAB_CAPTURE_STATUS',
      data: {}
    }, window.location.origin);
  }
};

console.log('✅ TabCaptureAutomation helper exposed to window');
